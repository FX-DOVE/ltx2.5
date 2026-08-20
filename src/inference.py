"""
src/inference.py
─────────────────────────────────────────────────────────────────────────────
Inference wrapper for the LTX-2.5 pipeline.

This module is deliberately kept thin — it translates the validated
InferenceInput schema into pipeline kwargs and returns raw output (a tensor
of frames).  All I/O (image decoding, video encoding, upload) lives in
handler.py so this file stays easily testable in isolation.

Supported modes:
  • text2video   — prompt only
  • image2video  — prompt + first frame conditioning image
  • flf2video    — prompt + first frame + last frame (First-Last-Frame)

Pipeline backends
─────────────────
model_loader.py may load one of two pipeline types:

  1. DistilledPipeline (ltx_pipelines) — primary backend.
     • Uses seed (int), NOT a torch.Generator.
     • No output_type kwarg — always returns a tensor/list.
     • FLF uses first_frame / last_frame kwargs, not LTXVideoCondition.

  2. LTXVideoPipeline (diffusers) — fallback backend.
     • Uses generator (torch.Generator).
     • output_type="np" returns float32 numpy frames [0, 1].
     • FLF uses LTXVideoCondition list.

run_inference() detects the backend via _is_ltx_distilled_pipeline() and
dispatches to the correct call helper.

GPU memory notes (NVIDIA L40S, 48 GB VRAM):
  • 450p (768×448) / 241 frames / bfloat16 uses roughly 28–35 GB VRAM — safe.
  • 720p / 97 frames uses roughly 35–42 GB — fits with some headroom.
  • 1080p / 97 frames requires 55–65 GB — will OOM on L40S; do not use.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import base64
import io
import random
import time
from typing import Optional

import numpy as np
import torch
from loguru import logger
from PIL import Image, UnidentifiedImageError

from schema import RESOLUTION_MAP, GenerationMode, InferenceInput


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline-type helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_ltx_distilled_pipeline(pipeline: object) -> bool:
    """
    Return True when the loaded pipeline is a DistilledPipeline from the
    ltx_pipelines package (Lightricks' first-party API).

    We detect this by module prefix so we don't require the package to be
    importable just to check the type.
    """
    module = type(pipeline).__module__ or ""
    return module.startswith("ltx_pipelines") or module.startswith("ltx_core")


# ─────────────────────────────────────────────────────────────────────────────
# Lazy import helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_ltx_condition_cls():
    """
    Return LTXVideoCondition from diffusers (deferred import).

    Used only for the diffusers backend in flf2video mode.
    """
    try:
        from diffusers.pipelines.ltx.pipeline_ltx_video import LTXVideoCondition  # type: ignore
        return LTXVideoCondition
    except ImportError:
        from diffusers import LTXVideoCondition  # type: ignore
        return LTXVideoCondition


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def run_inference(
    pipeline: object,
    params: InferenceInput,
) -> tuple[np.ndarray, int]:
    """
    Run LTX-2.5 inference and return (frames_uint8, seed_used).

    Dispatches to the correct call helper based on the loaded pipeline backend.

    Returns:
        frames_uint8: uint8 numpy array of shape (T, H, W, 3).
        seed_used:    The integer seed that was actually used.
    """
    seed = params.seed if params.seed is not None else random.randint(0, 2**31 - 1)
    width, height = RESOLUTION_MAP[params.resolution]

    logger.info(
        f"[inference] Starting {params.mode.value} | "
        f"{width}x{height} | {params.num_frames} frames | "
        f"steps={params.num_inference_steps} | cfg={params.guidance_scale} | seed={seed}"
    )

    # Decode conditioning images (may raise ValueError on bad data — caught in handler)
    first_pil: Optional[Image.Image] = None
    last_pil: Optional[Image.Image] = None

    if params.mode in (GenerationMode.image2video, GenerationMode.flf2video):
        first_pil = _decode_image(params.first_frame_image, label="first_frame_image")  # type: ignore[arg-type]
    if params.mode == GenerationMode.flf2video:
        last_pil = _decode_image(params.last_frame_image, label="last_frame_image")    # type: ignore[arg-type]

    t0 = time.monotonic()
    try:
        if _is_ltx_distilled_pipeline(pipeline):
            frames = _run_ltx_distilled(pipeline, params, seed, width, height, first_pil, last_pil)
        else:
            frames = _run_diffusers(pipeline, params, seed, width, height, first_pil, last_pil)
    except torch.cuda.OutOfMemoryError as oom:  # type: ignore[attr-defined]
        torch.cuda.empty_cache()
        logger.error(f"[inference] CUDA OOM: {oom}")
        raise RuntimeError("out_of_memory") from oom
    except RuntimeError as exc:
        torch.cuda.empty_cache()
        # Re-raise OOM that was already wrapped (e.g. from model_loader)
        if "out_of_memory" in str(exc):
            raise
        logger.exception(f"[inference] RuntimeError during generation: {exc}")
        raise
    except Exception as exc:
        torch.cuda.empty_cache()
        logger.exception(f"[inference] Unexpected error during generation: {exc}")
        raise

    elapsed = time.monotonic() - t0
    logger.info(f"[inference] Completed in {elapsed:.1f}s.")

    # Normalise to uint8 (T, H, W, 3).
    frames_uint8 = _to_uint8(frames)

    # Validate output shape before handing off to encoder.
    if frames_uint8.ndim != 4 or frames_uint8.shape[-1] != 3:
        raise RuntimeError(
            f"Pipeline returned unexpected frame shape {frames_uint8.shape}. "
            "Expected (T, H, W, 3)."
        )
    if frames_uint8.shape[0] == 0:
        raise RuntimeError("Pipeline returned 0 frames.")

    # Release VRAM before the next request.
    torch.cuda.empty_cache()

    return frames_uint8, seed


# ─────────────────────────────────────────────────────────────────────────────
# Backend-specific call helpers
# ─────────────────────────────────────────────────────────────────────────────


def _run_ltx_distilled(
    pipeline: object,
    params: InferenceInput,
    seed: int,
    width: int,
    height: int,
    first_pil: Optional[Image.Image],
    last_pil: Optional[Image.Image],
) -> np.ndarray:
    """
    Call a Lightricks DistilledPipeline (ltx_pipelines backend).

    Key differences from diffusers:
      • seed is an int, not a torch.Generator.
      • No output_type kwarg.
      • FLF conditioning via first_frame / last_frame kwargs.
      • Output is a tensor or list — normalised by _to_uint8().
    """
    kwargs: dict = {
        "prompt": params.prompt,
        "negative_prompt": params.negative_prompt,
        "width": width,
        "height": height,
        "num_frames": params.num_frames,
        "num_inference_steps": params.num_inference_steps,
        "guidance_scale": params.guidance_scale,
        "seed": seed,
    }

    if params.mode == GenerationMode.image2video:
        kwargs["first_frame"] = first_pil
    elif params.mode == GenerationMode.flf2video:
        kwargs["first_frame"] = first_pil
        kwargs["last_frame"] = last_pil

    result = pipeline(**kwargs)  # type: ignore[operator]
    return _extract_frames(result)


def _run_diffusers(
    pipeline: object,
    params: InferenceInput,
    seed: int,
    width: int,
    height: int,
    first_pil: Optional[Image.Image],
    last_pil: Optional[Image.Image],
) -> np.ndarray:
    """
    Call a diffusers LTXVideoPipeline.

    Key differences from DistilledPipeline:
      • Uses torch.Generator (device-aware, not hardcoded to cuda).
      • output_type="np" → float32 frames in [0, 1].
      • FLF via LTXVideoCondition list.
    """
    # Use the correct device for the generator — crashes on CPU if hardcoded to "cuda".
    gen_device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(seed)

    kwargs: dict = {
        "prompt": params.prompt,
        "negative_prompt": params.negative_prompt,
        "width": width,
        "height": height,
        "num_frames": params.num_frames,
        "num_inference_steps": params.num_inference_steps,
        "guidance_scale": params.guidance_scale,
        "generator": generator,
        "output_type": "np",  # float32 [0, 1] numpy array
    }

    if params.mode == GenerationMode.image2video:
        logger.debug("[inference] diffusers: image2video — image kwarg set.")
        kwargs["image"] = first_pil

    elif params.mode == GenerationMode.flf2video:
        logger.debug(
            "[inference] diffusers: flf2video — building LTXVideoCondition list "
            "for frame 0 and frame %d.", params.num_frames - 1
        )
        LTXVideoCondition = _get_ltx_condition_cls()
        kwargs["conditions"] = [
            LTXVideoCondition(image=first_pil, frame_index=0, strength=1.0),
            LTXVideoCondition(image=last_pil, frame_index=params.num_frames - 1, strength=1.0),
        ]
        # Some diffusers builds also expect `image` as the seed frame.
        kwargs["image"] = first_pil

    result = pipeline(**kwargs)  # type: ignore[operator]
    return _extract_frames(result)


# ─────────────────────────────────────────────────────────────────────────────
# Output normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────


def _extract_frames(result: object) -> np.ndarray:
    """
    Extract a (T, H, W, 3) numpy array from whatever the pipeline returned.

    Handles all observed output formats:
      • result.frames[0]  — diffusers standard (batch → first item)
      • result.frames     — some builds return the array directly
      • result[0]         — list/tuple of per-batch outputs
      • result            — raw tensor or array
    """
    # diffusers: PipelineOutput with .frames attr
    if hasattr(result, "frames"):
        frames = result.frames
        # frames may be (B, T, H, W, 3) or (T, H, W, 3)
        if isinstance(frames, (list, tuple)):
            frames = frames[0]  # first batch item
        elif hasattr(frames, "shape") and frames.ndim == 5:
            frames = frames[0]  # index into batch dim
        return _to_numpy(frames)

    # DistilledPipeline / list output
    if isinstance(result, (list, tuple)):
        return _to_numpy(result[0])

    # Raw tensor or array
    raw = _to_numpy(result)
    if raw.ndim == 5:
        raw = raw[0]  # strip batch dim
    return raw


def _to_numpy(x: object) -> np.ndarray:
    """Convert tensor / list / ndarray to a numpy array."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "cpu"):  # torch.Tensor
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()  # type: ignore[union-attr]
    if isinstance(x, list):
        return np.array(x)
    raise TypeError(f"Cannot convert {type(x).__name__} to numpy array.")


def _to_uint8(frames: np.ndarray) -> np.ndarray:
    """
    Normalise frames to uint8 (T, H, W, 3) regardless of input dtype/range.

      • float32 / float16 in [0, 1]  → scale × 255
      • float32 in [0, 255]           → clip and cast (shouldn't happen, but safe)
      • uint8 already                 → return as-is
    """
    if frames.dtype == np.uint8:
        return frames
    if np.issubdtype(frames.dtype, np.floating):
        if frames.max() <= 1.0 + 1e-3:
            # Standard [0, 1] float output
            return (np.clip(frames, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            # Rare: float already in [0, 255] range
            return np.clip(frames, 0, 255).astype(np.uint8)
    # Integer types other than uint8
    return frames.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Image decoding helper
# ─────────────────────────────────────────────────────────────────────────────


def _decode_image(source: str, label: str = "image") -> Image.Image:
    """
    Decode a conditioning image from either:
      • A base64-encoded data URI / raw base64 string, or
      • An HTTP/HTTPS URL.

    Returns a PIL.Image in RGB mode.
    Raises ValueError with a clear message on any decoding failure.
    """
    try:
        if source.startswith("http://") or source.startswith("https://"):
            import requests  # local import — avoids loading at module level in tests

            try:
                response = requests.get(source, timeout=30)
                response.raise_for_status()
            except requests.exceptions.Timeout:
                raise ValueError(
                    f"{label}: HTTP request timed out after 30s fetching '{source}'."
                )
            except requests.exceptions.HTTPError as http_err:
                raise ValueError(
                    f"{label}: HTTP {http_err.response.status_code} fetching '{source}'."
                )
            except requests.exceptions.RequestException as req_err:
                raise ValueError(
                    f"{label}: Failed to fetch '{source}': {req_err}"
                )
            try:
                img = Image.open(io.BytesIO(response.content))
            except UnidentifiedImageError:
                raise ValueError(
                    f"{label}: URL did not return a recognisable image (got "
                    f"content-type: {response.headers.get('Content-Type', 'unknown')})."
                )
        else:
            # Strip data-URI prefix if present (e.g. "data:image/png;base64,...")
            if "," in source:
                source = source.split(",", 1)[1]
            try:
                img_bytes = base64.b64decode(source, validate=True)
            except Exception:
                raise ValueError(
                    f"{label}: Invalid base64 string — could not decode."
                )
            try:
                img = Image.open(io.BytesIO(img_bytes))
            except UnidentifiedImageError:
                raise ValueError(
                    f"{label}: Decoded base64 data is not a recognisable image format."
                )

        return img.convert("RGB")

    except ValueError:
        raise  # propagate our own clear errors unchanged
    except Exception as exc:
        raise ValueError(f"{label}: Unexpected error decoding image: {exc}") from exc
