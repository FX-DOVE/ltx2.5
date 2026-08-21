"""
src/handler.py
─────────────────────────────────────────────────────────────────────────────
Runpod Serverless entrypoint for the LTX-2.5 video generation model.
Target GPU: NVIDIA L40S (48 GB VRAM).

Cold-start sequence (executed once at module import time, before any request):
  1. Assert network volume is mounted at /runpod-volume.
  2. ensure_weights_present() — downloads from Hugging Face if not cached.
  3. load_pipeline()          — builds the DistilledPipeline on the GPU.

Per-request sequence:
  1. Validate job["input"] with Pydantic (schema.py).
  2. Run inference (inference.py) → lazy frame iterator + audio track.
  3. Stream-encode to H.264 MP4 with the audio muxed in, using the upstream
     `ltx_pipelines.utils.media_io.encode_video` (PyAV/libx264). This is the
     same encoder the reference CLI uses, so the chunked VAE decoder is
     consumed lazily and the generated audio is not lost.
  4. Upload to S3/R2 (or return base64 for small outputs).
  5. Return structured JSON.

Error handling:
  • OOM errors → {"error": "out_of_memory", "retryable": false}
  • Validation errors → {"error": "validation_error", "message": ...}
  • Everything else → {"error": "internal_error", "message": ...}
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import base64
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import runpod
import torch
from loguru import logger
from pydantic import ValidationError

import inference as inference_module
import model_loader
from inference import InferenceResult
from schema import InferenceInput

# ─────────────────────────────────────────────────────────────────────────────
# Cold-start initialisation
# ─────────────────────────────────────────────────────────────────────────────

logger.info("[handler] Cold start: beginning initialisation...")
_COLD_START_T0 = time.monotonic()

# Step 1 & 2: volume check + weight download (idempotent)
try:
    model_loader.ensure_weights_present()
except (EnvironmentError, RuntimeError) as _exc:
    logger.critical(f"[handler] Fatal cold-start error: {_exc}")
    # Re-raise so Runpod marks the worker as unhealthy and rotates it.
    raise

# Step 3: build the pipeline on the GPU
_PIPELINE = model_loader.load_pipeline()

_COLD_START_ELAPSED = time.monotonic() - _COLD_START_T0
logger.info(
    f"[handler] Cold start complete in {_COLD_START_ELAPSED:.1f}s. "
    "Worker is ready to serve requests."
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Videos larger than this threshold are uploaded to storage instead of being
# returned as base64.  Base64 adds ~33% overhead, so keep this conservative.
_MAX_BASE64_BYTES = 5 * 1024 * 1024  # 5 MB

# libx264 constant rate factor for the output MP4 (upstream default is 19).
_CRF = int(os.environ.get("LTX_CRF", "19"))


# ─────────────────────────────────────────────────────────────────────────────
# Handler
# ─────────────────────────────────────────────────────────────────────────────


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """
    Runpod serverless handler function.

    Args:
        job: Runpod job dict.  The payload is under job["input"].

    Returns:
        dict: Structured success or error response.
    """
    job_id = job.get("id", "unknown")
    logger.info(f"[handler] Job {job_id} received.")

    # Autograd is dead weight here and actively harmful: the graph keeps the
    # text encoder's ~24 GB alive past dispose(). inference.py guards its own
    # scopes; this makes the whole request grad-free even on paths added later.
    # Set inside handler() rather than at import because the flag is
    # thread-local and Runpod may dispatch the handler off the import thread.
    torch.set_grad_enabled(False)

    # ── 1. Input validation ───────────────────────────────────────────────────
    try:
        params = InferenceInput.model_validate(job.get("input", {}))
    except ValidationError as exc:
        logger.warning(f"[handler] Job {job_id} validation failed: {exc}")
        return _error_response("validation_error", str(exc), retryable=False)

    # ── 2. Inference ──────────────────────────────────────────────────────────
    t_inference_start = time.monotonic()
    result: InferenceResult | None = None
    try:
        result = inference_module.run_inference(_PIPELINE, params)
    except RuntimeError as exc:
        if "out_of_memory" in str(exc):
            logger.error(f"[handler] Job {job_id} -> OOM.")
            return _error_response(
                "out_of_memory",
                (
                    "CUDA out of memory. Try a lower 'resolution' or fewer "
                    "'num_frames'. On a 48 GB L40S with fp8-cast quantization, "
                    "450p/241f and 720p/121f fit; 1080p needs short clips or "
                    "LTX_OFFLOAD_MODE=cpu on the endpoint (LTX_OFFLOAD_MODE=auto "
                    "picks that automatically below 44 GiB of VRAM)."
                ),
                retryable=False,
            )
        logger.exception(f"[handler] Job {job_id} -> inference error: {exc}")
        return _error_response("inference_error", str(exc), retryable=False)
    except ValueError as exc:
        # Image decode errors raised by inference._decode_image
        logger.warning(f"[handler] Job {job_id} -> image decode error: {exc}")
        return _error_response("image_decode_error", str(exc), retryable=False)
    except Exception as exc:
        logger.exception(f"[handler] Job {job_id} -> unexpected error: {exc}")
        return _error_response(
            "internal_error",
            f"An unexpected error occurred: {type(exc).__name__}: {exc}",
            retryable=False,
        )

    # ── 3. Encode to MP4 (VAE decode streams during this step) ────────────────
    try:
        try:
            video_bytes = _encode_video(result, fps=params.fps, job_id=job_id)
        finally:
            # The conditioning temp files are only needed until the pipeline
            # has consumed them, which is guaranteed once encoding returns.
            result.cleanup()
    except RuntimeError as exc:
        if "out_of_memory" in str(exc):
            logger.error(f"[handler] Job {job_id} -> OOM during VAE decode.")
            return _error_response(
                "out_of_memory",
                "CUDA out of memory during VAE decode. Lower the resolution or "
                "frame count.",
                retryable=False,
            )
        logger.exception(f"[handler] Job {job_id} -> video encoding failed: {exc}")
        return _error_response("encoding_error", f"Video encoding failed: {exc}", False)
    except Exception as exc:
        logger.exception(f"[handler] Job {job_id} -> video encoding failed: {exc}")
        return _error_response("encoding_error", f"Video encoding failed: {exc}", False)

    generation_time = time.monotonic() - t_inference_start

    # ── 4. Upload or base64 ───────────────────────────────────────────────────
    duration_s = result.num_frames / params.fps
    video_url: str | None = None
    video_b64: str | None = None

    if len(video_bytes) > _MAX_BASE64_BYTES:
        try:
            video_url = _upload_video(video_bytes, job_id)
        except Exception as exc:
            logger.warning(
                f"[handler] Job {job_id} -> upload failed ({exc}); "
                f"returning {len(video_bytes) / 1e6:.1f} MB as base64."
            )
            video_b64 = base64.b64encode(video_bytes).decode("utf-8")
    else:
        video_b64 = base64.b64encode(video_bytes).decode("utf-8")

    # ── 5. Build response ─────────────────────────────────────────────────────
    response: dict[str, Any] = {
        "status": "success",
        "mode": params.mode.value,  # resolved mode — shows auto-detected mode
        "duration_seconds": round(duration_s, 2),
        "generation_time_seconds": round(generation_time, 2),
        "seed_used": result.seed,
        "resolution": f"{result.width}x{result.height}",
        "num_frames": result.num_frames,
        "fps": params.fps,
        "has_audio": _has_audio(result.audio),
    }
    if video_url:
        response["video_url"] = video_url
    if video_b64:
        response["video_base64"] = video_b64

    logger.info(
        f"[handler] Job {job_id} complete. "
        f"gen={generation_time:.1f}s | {result.width}x{result.height} | "
        f"{result.num_frames}f | audio={response['has_audio']} | seed={result.seed}"
    )
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _has_audio(audio: Any) -> bool:
    """True when the pipeline produced a non-empty audio waveform."""
    waveform = getattr(audio, "waveform", None)
    if waveform is None:
        return False
    try:
        return int(waveform.numel()) > 0
    except Exception:
        return False


def _encode_video(result: InferenceResult, fps: int, job_id: str) -> bytes:
    """
    Stream-encode the pipeline's frame chunks to an H.264 MP4 and return bytes.

    Uses the upstream encoder (`ltx_pipelines.utils.media_io.encode_video`,
    PyAV + libx264 + yuv420p) rather than an ffmpeg/imageio round-trip, for two
    reasons:
      • it consumes the lazy chunk iterator returned by the VAE decoder, so
        peak host memory stays at one chunk instead of the whole clip; and
      • it muxes the generated audio track, which LTX-2.5 produces alongside
        the video and which an image-only encoder silently discards.
    """
    from ltx_core.model.video_vae import get_video_chunks_number
    from ltx_pipelines.utils.media_io import encode_video

    chunks = get_video_chunks_number(result.num_frames, result.tiling_config)

    with tempfile.TemporaryDirectory(prefix="ltx-out-") as tmpdir:
        out_path = Path(tmpdir) / f"{job_id}.mp4"
        # The VAE decode runs *inside* this call as the encoder pulls chunks, so
        # the grad guard has to cover it too (inference._grad_free_chunks also
        # guards each pull; this covers the audio path and any future callers).
        with torch.no_grad():
            encode_video(
                video=result.video,
                fps=fps,
                audio=result.audio,
                output_path=str(out_path),
                video_chunks_number=chunks,
                crf=_CRF,
            )
        if not out_path.exists() or out_path.stat().st_size == 0:
            raise RuntimeError(
                f"encode_video produced no output at {out_path}. "
                "The VAE decoder may have yielded zero chunks."
            )
        return out_path.read_bytes()


def _upload_video(video_bytes: bytes, job_id: str) -> str:
    """
    Upload the video and return a URL.

    Priority:
      1. S3_BUCKET configured → upload with boto3 to S3-compatible storage
         (this endpoint uses Cloudflare R2).
      2. BUCKET_ENDPOINT_URL configured → Runpod's own bucket helper
         (runpod.serverless.utils.rp_upload), which returns a presigned URL.

    Raises RuntimeError when neither is configured, so the caller can fall
    back to base64.
    """
    s3_bucket = os.environ.get("S3_BUCKET")
    if s3_bucket:
        return _upload_to_s3(video_bytes, job_id, s3_bucket)

    if os.environ.get("BUCKET_ENDPOINT_URL"):
        return _upload_to_runpod_storage(video_bytes, job_id)

    raise RuntimeError(
        "No object storage configured. Set S3_BUCKET (+ S3_ENDPOINT_URL, "
        "S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY) or Runpod's BUCKET_ENDPOINT_URL "
        "(+ BUCKET_ACCESS_KEY_ID, BUCKET_SECRET_ACCESS_KEY)."
    )


def _upload_to_runpod_storage(video_bytes: bytes, job_id: str) -> str:
    """
    Upload via Runpod's bucket helper and return a presigned URL.

    Note: the correct API is runpod.serverless.utils.rp_upload — there is no
    top-level `runpod.upload_file`.
    """
    from runpod.serverless.utils import rp_upload

    return rp_upload.upload_in_memory_object(
        f"{job_id}.mp4",
        video_bytes,
        prefix=os.environ.get("S3_KEY_PREFIX", "ltx-2.5"),
    )


def _upload_to_s3(video_bytes: bytes, job_id: str, bucket: str) -> str:
    """
    Upload to an S3-compatible bucket and return a presigned URL.

    Required env vars (set as Runpod secrets):
      S3_BUCKET              — bucket name
      S3_ACCESS_KEY_ID       — access key id
      S3_SECRET_ACCESS_KEY   — secret access key
      S3_ENDPOINT_URL        — endpoint (omit for AWS S3; required for R2/B2)
      S3_REGION              — region (default: us-east-1; R2 uses 'auto')
      S3_KEY_PREFIX          — object key prefix (default: ltx-2.5)
      PRESIGNED_URL_TTL_SECONDS — URL expiry in seconds (default: 86400)
    """
    import boto3  # deferred; boto3 is only needed on this path

    s3_client = boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
        aws_access_key_id=os.environ.get("S3_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("S3_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("S3_REGION", "us-east-1"),
    )

    key_prefix = os.environ.get("S3_KEY_PREFIX", "ltx-2.5")
    ttl = int(os.environ.get("PRESIGNED_URL_TTL_SECONDS", "86400"))
    object_key = f"{key_prefix}/{job_id}.mp4"
    s3_client.put_object(
        Bucket=bucket,
        Key=object_key,
        Body=video_bytes,
        ContentType="video/mp4",
    )

    return s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": object_key},
        ExpiresIn=ttl,
    )


def _error_response(error_code: str, message: str, retryable: bool) -> dict[str, Any]:
    return {
        "status": "error",
        "error": error_code,
        "message": message,
        "retryable": retryable,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Runpod entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
