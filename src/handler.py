"""
src/handler.py
─────────────────────────────────────────────────────────────────────────────
Runpod Serverless entrypoint for the LTX-2.5 video generation model.
Target GPU: NVIDIA L40S (48 GB VRAM).

Cold-start sequence (executed once at module import time, before any request):
  1. Assert network volume is mounted at /runpod-volume.
  2. ensure_weights_present() — downloads from Hugging Face if not cached.
  3. load_pipeline()          — loads all model components into GPU VRAM.

Per-request sequence:
  1. Validate job["input"] with Pydantic (schema.py).
  2. Run inference (inference.py).
  3. Encode frames to MP4.
  4. Upload to Runpod temp storage (or return base64 for small outputs).
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

import numpy as np
import runpod
from loguru import logger
from pydantic import ValidationError

import inference as inference_module
import model_loader
from schema import InferenceInput, RESOLUTION_MAP

# ─────────────────────────────────────────────────────────────────────────────
# Cold-start initialisation
# ─────────────────────────────────────────────────────────────────────────────

logger.info("[handler] Cold start: beginning initialisation …")
_COLD_START_T0 = time.monotonic()

# Step 1 & 2: volume check + weight download (idempotent)
try:
    model_loader.ensure_weights_present()
except (EnvironmentError, RuntimeError) as _exc:
    logger.critical(f"[handler] Fatal cold-start error: {_exc}")
    # Re-raise so Runpod marks the worker as unhealthy and rotates it.
    raise

# Step 3: load model into VRAM
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

    # ── 1. Input validation ───────────────────────────────────────────────────
    try:
        params = InferenceInput.model_validate(job.get("input", {}))
    except ValidationError as exc:
        logger.warning(f"[handler] Job {job_id} validation failed: {exc}")
        return _error_response(
            error_code="validation_error",
            message=str(exc),
            retryable=False,
        )

    # ── 2. Inference ──────────────────────────────────────────────────────────
    t_inference_start = time.monotonic()
    try:
        frames_uint8, seed_used = inference_module.run_inference(_PIPELINE, params)
    except RuntimeError as exc:
        err_str = str(exc)
        if "out_of_memory" in err_str:
            logger.error(f"[handler] Job {job_id} -> OOM.")
            return _error_response(
                error_code="out_of_memory",
                message=(
                    "CUDA out of memory. Try reducing resolution, num_frames, or "
                    "num_inference_steps. The L40S has 48 GB VRAM; "
                    "450p/241f uses ~30 GB, 720p/97f uses ~42 GB, 1080p will OOM."
                ),
                retryable=False,
            )
        logger.exception(f"[handler] Job {job_id} -> inference error: {exc}")
        return _error_response(
            error_code="inference_error",
            message=err_str,
            retryable=False,
        )
    except ValueError as exc:
        # Image decode errors raised by inference._decode_image
        logger.warning(f"[handler] Job {job_id} -> image decode error: {exc}")
        return _error_response(
            error_code="image_decode_error",
            message=str(exc),
            retryable=False,
        )
    except Exception as exc:
        logger.exception(f"[handler] Job {job_id} -> unexpected error: {exc}")
        return _error_response(
            error_code="internal_error",
            message=f"An unexpected error occurred: {type(exc).__name__}: {exc}",
            retryable=False,
        )

    generation_time = time.monotonic() - t_inference_start

    # ── 3. Encode to MP4 ──────────────────────────────────────────────────────
    try:
        video_bytes = _encode_video(frames_uint8, fps=params.fps)
    except Exception as exc:
        logger.exception(f"[handler] Job {job_id} -> video encoding failed: {exc}")
        return _error_response(
            error_code="encoding_error",
            message=f"Video encoding failed: {exc}",
            retryable=False,
        )

    # ── 4. Upload or base64 ───────────────────────────────────────────────────
    width, height = RESOLUTION_MAP[params.resolution]
    duration_s = params.num_frames / params.fps

    video_url: str | None = None
    video_b64: str | None = None

    if len(video_bytes) > _MAX_BASE64_BYTES:
        try:
            video_url = _upload_video(video_bytes, job_id)
        except Exception as exc:
            logger.warning(
                f"[handler] Job {job_id} -> upload failed ({exc}), falling back to base64."
            )
            video_b64 = base64.b64encode(video_bytes).decode("utf-8")
    else:
        video_b64 = base64.b64encode(video_bytes).decode("utf-8")

    # ── 5. Build response ─────────────────────────────────────────────────────
    response: dict[str, Any] = {
        "status": "success",
        "mode": params.mode.value,     # resolved mode — shows auto-detected mode
        "duration_seconds": round(duration_s, 2),
        "generation_time_seconds": round(generation_time, 2),
        "seed_used": seed_used,
        "resolution": f"{width}x{height}",
        "num_frames": params.num_frames,
        "fps": params.fps,
    }
    if video_url:
        response["video_url"] = video_url
    if video_b64:
        response["video_base64"] = video_b64

    logger.info(
        f"[handler] Job {job_id} complete. "
        f"gen={generation_time:.1f}s | {width}x{height} | "
        f"{params.num_frames}f | seed={seed_used}"
    )
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _encode_video(frames: np.ndarray, fps: int) -> bytes:
    """
    Encode a (T, H, W, 3) uint8 numpy array to an H.264 MP4 bytestring.

    Uses imageio v3 with the ffmpeg backend (imageio-ffmpeg).  Output is
    libx264 / yuv420p — the most widely compatible format for web/mobile.
    """
    import imageio.v3 as iio  # v3 API — avoids deprecation warnings from mimwrite

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            f"Expected frames shape (T, H, W, 3), got {frames.shape}."
        )
    if frames.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 frames, got dtype={frames.dtype}."
        )

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        iio.imwrite(
            tmp_path,
            frames,                # (T, H, W, 3) uint8
            plugin="FFMPEG",
            fps=fps,
            codec="libx264",       # explicit codec — avoids mpeg4 fallback
            pixelformat="yuv420p", # broadest browser/mobile compatibility
            output_params=["-crf", "18", "-movflags", "+faststart"],
        )
        video_bytes = Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return video_bytes


def _upload_video(video_bytes: bytes, job_id: str) -> str:
    """
    Upload the video and return a URL.

    Priority:
      1. If S3_BUCKET is configured, upload to S3-compatible storage.
      2. Otherwise, use Runpod's built-in temp file upload (runpod.upload_file).

    The Runpod temp URL is valid for 1 hour by default — enough for the caller
    to download and store it themselves.
    """
    # Option 1: S3-compatible upload (e.g. Cloudflare R2, AWS S3, Backblaze B2)
    s3_bucket = os.environ.get("S3_BUCKET")
    if s3_bucket:
        return _upload_to_s3(video_bytes, job_id, s3_bucket)

    # Option 2: Runpod built-in temp storage
    return _upload_to_runpod_storage(video_bytes, job_id)


def _upload_to_runpod_storage(video_bytes: bytes, job_id: str) -> str:
    """Upload to Runpod's built-in temporary S3 storage and return the URL."""
    if not hasattr(runpod, "upload_file"):
        raise RuntimeError(
            "runpod.upload_file is not available in this SDK version. "
            "Upgrade to runpod>=1.7.7 or configure S3_BUCKET for external storage."
        )

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        # runpod.upload_file returns a presigned URL valid for ~1 hour.
        url = runpod.upload_file(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return url


def _upload_to_s3(video_bytes: bytes, job_id: str, bucket: str) -> str:
    """
    Upload to an S3-compatible bucket and return the object URL.

    Required env vars (set as Runpod secrets):
      S3_BUCKET              — bucket name
      S3_ACCESS_KEY_ID       — access key id
      S3_SECRET_ACCESS_KEY   — secret access key
      S3_ENDPOINT_URL        — endpoint (omit for AWS S3; required for R2/B2)
      S3_REGION              — region (default: us-east-1)
      S3_KEY_PREFIX          — object key prefix (default: ltx-2.5)
      PRESIGNED_URL_TTL_SECONDS — URL expiry in seconds (default: 86400)
    """
    import boto3  # deferred; boto3 is optional

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

    # Generate a presigned URL (default 24 h; override via PRESIGNED_URL_TTL_SECONDS).
    url = s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": object_key},
        ExpiresIn=ttl,
    )
    return url


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
