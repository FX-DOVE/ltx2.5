"""
src/handler.py
─────────────────────────────────────────────────────────────────────────────
RunPod Serverless entrypoint for Lightricks LTX-2.5 Video Generation API.

Workflow:
  1. Boot / Cold Start:
     - Check network volume and ensure weights are present.
     - Preload LTX-2.5 Diffusers pipeline into GPU memory.
     - Gracefully handle startup issues to prevent fatal worker crash loops.
  2. Request Handling:
     - Validate input payload with Pydantic.
     - Execute LTX-2.5 video generation (text2video, image2video, flf2video).
     - Encode generated frames to H.264 MP4 with faststart.
     - Upload to S3/Cloudflare R2 (or fallback to base64 / RunPod temp storage).
     - Return structured JSON response.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import base64
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import runpod
from loguru import logger
from pydantic import ValidationError

import inference as inference_module
import model_loader
from schema import InferenceInput


# ─────────────────────────────────────────────────────────────────────────────
# Cold-start Initialization
# ─────────────────────────────────────────────────────────────────────────────

logger.info("=" * 60)
logger.info("Starting LTX-2.5 RunPod Serverless Worker")
logger.info(f"MODEL_ID   : {model_loader.MODEL_ID}")
logger.info(f"DTYPE      : {model_loader.DTYPE_STR}")
logger.info(f"VOLUME_ROOT: {model_loader.VOLUME_ROOT}")
logger.info("=" * 60)

_PIPELINE: Optional[Any] = None
_INIT_ERROR: Optional[str] = None
_COLD_START_T0 = time.monotonic()

try:
    model_loader.ensure_weights_present()
    _PIPELINE = model_loader.load_pipeline()
    _COLD_START_ELAPSED = time.monotonic() - _COLD_START_T0
    logger.info(
        f"[handler] Cold start completed successfully in {_COLD_START_ELAPSED:.1f}s. "
        "Worker is ready to accept jobs."
    )
except Exception as _exc:
    _INIT_ERROR = str(_exc)
    logger.error(
        f"[handler] Non-fatal initialization warning: {_exc}\n"
        "Worker will start and attempt lazy initialization on first request."
    )


# Threshold for uploading vs inline base64
_MAX_BASE64_BYTES = 5 * 1024 * 1024  # 5 MB


# ─────────────────────────────────────────────────────────────────────────────
# Handler
# ─────────────────────────────────────────────────────────────────────────────


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """
    RunPod serverless job handler.
    """
    global _PIPELINE, _INIT_ERROR

    job_id = job.get("id", "local_job")
    logger.info(f"[handler] Processing job {job_id}")

    # ── 1. Pipeline Readiness Check / Lazy Loading ────────────────────────────
    if _PIPELINE is None:
        logger.info(f"[handler] Pipeline not initialized yet. Attempting lazy load for job {job_id}...")
        try:
            model_loader.ensure_weights_present()
            _PIPELINE = model_loader.load_pipeline()
            _INIT_ERROR = None
        except Exception as exc:
            _INIT_ERROR = str(exc)
            logger.exception(f"[handler] Lazy loading failed for job {job_id}: {exc}")
            return _error_response(
                error_code="model_initialization_failed",
                message=(
                    f"Model initialization failed: {exc}. "
                    f"Check that HF_TOKEN is configured and has access to '{model_loader.MODEL_ID}'."
                ),
                retryable=False,
            )

    # ── 2. Input Validation ───────────────────────────────────────────────────
    try:
        raw_input = job.get("input", {})
        params = InferenceInput.model_validate(raw_input)
    except ValidationError as exc:
        logger.warning(f"[handler] Job {job_id} validation error: {exc}")
        return _error_response(
            error_code="validation_error",
            message=str(exc),
            retryable=False,
        )

    # ── 3. Run Inference ──────────────────────────────────────────────────────
    t_inference_start = time.monotonic()
    try:
        frames_uint8, seed_used = inference_module.run_inference(_PIPELINE, params)
    except RuntimeError as exc:
        err_str = str(exc)
        if "out_of_memory" in err_str:
            logger.error(f"[handler] Job {job_id} CUDA OOM.")
            return _error_response(
                error_code="out_of_memory",
                message="CUDA out of memory. Try reducing width, height, or num_frames.",
                retryable=False,
            )
        logger.exception(f"[handler] Job {job_id} inference runtime error: {exc}")
        return _error_response(
            error_code="inference_error",
            message=err_str,
            retryable=False,
        )
    except Exception as exc:
        logger.exception(f"[handler] Job {job_id} unexpected error: {exc}")
        return _error_response(
            error_code="internal_error",
            message=f"{type(exc).__name__}: {exc}",
            retryable=False,
        )

    generation_time = time.monotonic() - t_inference_start

    # ── 4. Encode Video to MP4 ────────────────────────────────────────────────
    try:
        video_bytes = _encode_video(frames_uint8, fps=params.fps)
    except Exception as exc:
        logger.exception(f"[handler] Job {job_id} video encoding error: {exc}")
        return _error_response(
            error_code="encoding_error",
            message=f"Video encoding failed: {exc}",
            retryable=False,
        )

    # ── 5. Output Packaging (S3 / R2 / Base64) ────────────────────────────────
    duration_s = params.num_frames / params.fps
    video_url: Optional[str] = None
    video_b64: Optional[str] = None

    if len(video_bytes) > _MAX_BASE64_BYTES or os.environ.get("S3_BUCKET"):
        try:
            video_url = _upload_video(video_bytes, job_id)
        except Exception as exc:
            logger.warning(f"[handler] S3/R2 upload failed ({exc}). Returning base64 encoding.")
            video_b64 = base64.b64encode(video_bytes).decode("utf-8")
    else:
        video_b64 = base64.b64encode(video_bytes).decode("utf-8")

    response: dict[str, Any] = {
        "status": "success",
        "mode": params.mode.value,
        "duration_seconds": round(duration_s, 2),
        "generation_time_seconds": round(generation_time, 2),
        "seed_used": seed_used,
        "resolution": f"{params.width}x{params.height}",
        "num_frames": params.num_frames,
        "fps": params.fps,
    }
    if video_url:
        response["video_url"] = video_url
    if video_b64:
        response["video_base64"] = video_b64

    logger.info(
        f"[handler] Job {job_id} complete: {params.width}x{params.height} | "
        f"{params.num_frames}f | gen_time={generation_time:.1f}s | seed={seed_used}"
    )
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Internal Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _encode_video(frames: np.ndarray, fps: int) -> bytes:
    """Encode (T, H, W, 3) uint8 numpy frames to H.264 MP4."""
    import imageio

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        imageio.mimwrite(
            tmp_path,
            frames,
            fps=fps,
            quality=8,
            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            macro_block_size=None,
        )
        video_bytes = Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return video_bytes


def _upload_video(video_bytes: bytes, job_id: str) -> str:
    """Upload video to S3/Cloudflare R2 or RunPod temporary storage."""
    s3_bucket = os.environ.get("S3_BUCKET") or os.environ.get("R2_BUCKET")
    if s3_bucket:
        return _upload_to_s3(video_bytes, job_id, s3_bucket)
    return _upload_to_runpod_storage(video_bytes, job_id)


def _upload_to_runpod_storage(video_bytes: bytes, job_id: str) -> str:
    """Upload to RunPod temporary file storage."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        url = runpod.upload_file(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return url


def _upload_to_s3(video_bytes: bytes, job_id: str, bucket: str) -> str:
    """Upload to S3 / Cloudflare R2 bucket."""
    import boto3

    endpoint_url = os.environ.get("S3_ENDPOINT_URL") or os.environ.get("R2_ENDPOINT_URL")
    access_key = os.environ.get("S3_ACCESS_KEY_ID") or os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("S3_SECRET_ACCESS_KEY") or os.environ.get("R2_SECRET_ACCESS_KEY")
    region = os.environ.get("S3_REGION", "auto")
    key_prefix = os.environ.get("S3_KEY_PREFIX", "ltx-2.5")
    ttl = int(os.environ.get("PRESIGNED_URL_TTL_SECONDS", "86400"))

    s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )

    object_key = f"{key_prefix}/{job_id}.mp4"
    s3_client.put_object(
        Bucket=bucket,
        Key=object_key,
        Body=video_bytes,
        ContentType="video/mp4",
    )

    # If public R2 base URL is set, construct clean public URL
    r2_public_base = os.environ.get("R2_PUBLIC_BASE_URL")
    if r2_public_base:
        return f"{r2_public_base.rstrip('/')}/{object_key}"

    # Generate presigned URL
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
# RunPod Serverless Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
