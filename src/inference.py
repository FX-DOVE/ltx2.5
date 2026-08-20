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

GPU memory notes (NVIDIA L40S, 48 GB VRAM):
  • 450p (768×448) / 241 frames / bfloat16 uses roughly 28–35 GB VRAM — safe.
  • 720p / 97 frames uses roughly 35–42 GB — fits with some headroom.
  • 1080p / 97 frames requires 55–65 GB — will OOM on L40S; do not use.
  • The pipeline is kept in memory between requests (singleton in
    model_loader.py). torch.cuda.empty_cache() is called after each
    generation to return fragmented memory before the next request.
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
from PIL import Image

from schema import RESOLUTION_MAP, GenerationMode, InferenceInput


# ---------------------------------------------------------------------------
# Lazy import helpers for LTXVideoCondition (only needed at inference time)
# ---------------------------------------------------------------------------

def _get_ltx_condition_cls():
    """
    Return LTXVideoCondition from diffusers.  Deferred so unit tests that mock
    torch/diffusers still work without the real packages installed.

    LTXVideoCondition is the correct way to express first-last-frame constraints
    in the diffusers LTXVideoPipeline.  Each condition pins a specific video
    frame (by index) to a reference image tensor.
    """
    try:
        from diffusers.pipelines.ltx.pipeline_ltx_video import LTXVideoCondition  # type: ignore
        return LTXVideoCondition
    except ImportError:
        # Older diffusers builds may have a different import path
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
    Run LTX-2.5 inference and return (frames_array, seed_used).

    The mode is determined DYNAMICALLY:
      • text2video   — no conditioning images needed.
      • image2video  — first_frame_image provided; pipeline receives `image` kwarg.
      • flf2video    — both first and last frame provided; pipeline receives a
                       `conditions` list of LTXVideoCondition objects anchoring
                       the first and last frames by frame index.  This is the
                       correct diffusers API for first-last-frame conditioning.

    If mode is auto-detected (schema allows it), missing fields are caught at
    validation time in schema.py before we ever reach this function.

    Args:
        pipeline: The LTXVideoPipeline singleton from model_loader.load_pipeline().
        params:   Validated InferenceInput from schema.py.

    Returns:
        frames:     uint8 numpy array of shape (T, H, W, 3).
        seed_used:  The integer seed that was actually used.
    """
    seed = params.seed if params.seed is not None else random.randint(0, 2**31 - 1)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    width, height = RESOLUTION_MAP[params.resolution]

    # Build keyword arguments common to all modes.
    kwargs: dict = {
        "prompt": params.prompt,
        "negative_prompt": params.negative_prompt,
        "width": width,
        "height": height,
        "num_frames": params.num_frames,
        "num_inference_steps": params.num_inference_steps,
        "guidance_scale": params.guidance_scale,
        "generator": generator,
        "output_type": "np",   # return numpy uint8 frames, not PIL
    }

    # ── Dynamic mode dispatch ────────────────────────────────────────────────
    #
    # text2video:  no extra kwargs needed.
    #
    # image2video: pass PIL image as `image` kwarg.  The pipeline uses it as
    #              the first-frame conditioning signal.
    #
    # flf2video:   use LTXVideoCondition objects to anchor frame 0 (first) and
    #              frame (num_frames - 1) (last) to the respective input images.
    #              This is the correct diffusers API — not a `last_image` kwarg.
    #
    # Both image2video and flf2video accept base64-encoded images OR HTTPS URLs.
    # _decode_image() handles both transparently.
    # ────────────────────────────────────────────────────────────────────────

    if params.mode == GenerationMode.image2video:
        logger.debug("[inference] Mode: image2video — using first_frame_image as conditioning.")
        first_pil = _decode_image(params.first_frame_image)  # type: ignore[arg-type]
        kwargs["image"] = first_pil

    elif params.mode == GenerationMode.flf2video:
        logger.debug(
            "[inference] Mode: flf2video — building LTXVideoCondition list "
            "for first (frame 0) and last (frame %d).", params.num_frames - 1
        )
        LTXVideoCondition = _get_ltx_condition_cls()

        first_pil = _decode_image(params.first_frame_image)  # type: ignore[arg-type]
        last_pil  = _decode_image(params.last_frame_image)   # type: ignore[arg-type]

        # conditioning_strength=1.0 means the model strictly adheres to the
        # reference frames.  Lower values allow creative drift from the reference.
        kwargs["conditions"] = [
            LTXVideoCondition(
                image=first_pil,
                frame_index=0,
                strength=1.0,
            ),
            LTXVideoCondition(
                image=last_pil,
                frame_index=params.num_frames - 1,
                strength=1.0,
            ),
        ]
        # Also pass the first frame as `image` for pipelines that use it as
        # a seed/initialiser even in FLF mode (harmless if not used).
        kwargs["image"] = first_pil

    # else: text2video — no extra conditioning.

    logger.info(
        f"[inference] Starting {params.mode.value} | "
        f"{width}x{height} | {params.num_frames} frames | "
        f"steps={params.num_inference_steps} | cfg={params.guidance_scale} | seed={seed}"
    )
    t0 = time.monotonic()

    try:
        result = pipeline(**kwargs)  # type: ignore[operator]
        frames: np.ndarray = result.frames[0]  # shape (T, H, W, 3), float32 0–1
    except torch.cuda.OutOfMemoryError as oom:
        torch.cuda.empty_cache()
        logger.error(f"[inference] CUDA OOM: {oom}")
        raise RuntimeError("out_of_memory") from oom
    except Exception as exc:
        torch.cuda.empty_cache()
        logger.exception(f"[inference] Unexpected error: {exc}")
        raise

    elapsed = time.monotonic() - t0
    logger.info(f"[inference] Completed in {elapsed:.1f}s.")

    # Convert float32 [0, 1] → uint8 [0, 255] for video encoding.
    frames_uint8 = (np.clip(frames, 0.0, 1.0) * 255).astype(np.uint8)

    # Release intermediate tensors; helps allocator before next request.
    del result
    torch.cuda.empty_cache()

    return frames_uint8, seed


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _decode_image(source: str) -> Image.Image:
    """
    Decode a conditioning image from either:
      • A base64-encoded data URI / raw base64 string, or
      • An HTTPS URL.

    Returns a PIL.Image in RGB mode.
    """
    if source.startswith("http://") or source.startswith("https://"):
        import requests  # local import to avoid loading at module level in tests

        response = requests.get(source, timeout=30)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))
    else:
        # Strip data-URI prefix if present (e.g. "data:image/png;base64,...")
        if "," in source:
            source = source.split(",", 1)[1]
        img_bytes = base64.b64decode(source)
        img = Image.open(io.BytesIO(img_bytes))

    return img.convert("RGB")
