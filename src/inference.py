"""
src/inference.py
─────────────────────────────────────────────────────────────────────────────
Inference execution for the LTX-2.5 Diffusers pipeline.

Supports:
  • text2video   — prompt -> video
  • image2video  — prompt + first_frame_image -> video
  • flf2video    — prompt + first_frame_image + last_frame_image -> video

Optimized with torch.inference_mode(), automatic tensor conversion,
and GPU memory management for RTX PRO 6000 96GB.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import base64
import io
import random
import time
from typing import Any, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from PIL import Image

from schema import GenerationMode, InferenceInput


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def run_inference(
    pipeline: Any,
    params: InferenceInput,
) -> Tuple[np.ndarray, int]:
    """
    Run LTX-2.5 inference and return (frames_uint8, seed_used).

    Args:
        pipeline: The loaded Diffusers pipeline singleton.
        params:   Validated InferenceInput parameters.

    Returns:
        frames_uint8: (T, H, W, 3) uint8 numpy array.
        seed_used:    Integer seed used for generation.
    """
    seed = params.seed if params.seed is not None else random.randint(0, 2**31 - 1)
    generator_device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed)

    width = params.width or 1280
    height = params.height or 720

    # Build kwargs
    kwargs: dict[str, Any] = {
        "prompt": params.prompt,
        "negative_prompt": params.negative_prompt,
        "width": width,
        "height": height,
        "num_frames": params.num_frames,
        "generator": generator,
        "output_type": "np",
    }

    # Frame rate / FPS handling (LTX2Pipeline uses frame_rate, others use fps)
    kwargs["frame_rate"] = float(params.fps)

    # Denoising steps and guidance
    if params.num_inference_steps is not None:
        kwargs["num_inference_steps"] = params.num_inference_steps
    if params.guidance_scale is not None:
        kwargs["guidance_scale"] = params.guidance_scale

    # ── Conditioning Images ───────────────────────────────────────────────────
    if params.mode == GenerationMode.image2video and params.first_frame_image:
        logger.debug("[inference] Mode: image2video - decoding first frame image.")
        first_pil = _decode_image(params.first_frame_image)
        kwargs["image"] = first_pil

    elif params.mode == GenerationMode.flf2video and params.first_frame_image:
        logger.debug("[inference] Mode: flf2video - decoding first and last frame images.")
        first_pil = _decode_image(params.first_frame_image)
        last_pil = _decode_image(params.last_frame_image) if params.last_frame_image else None
        
        # Try to use diffusers LTX condition objects if supported
        condition_cls = _get_ltx_condition_cls()
        if condition_cls and last_pil:
            kwargs["conditions"] = [
                condition_cls(image=first_pil, frame_index=0, strength=1.0),
                condition_cls(image=last_pil, frame_index=params.num_frames - 1, strength=1.0),
            ]
        kwargs["image"] = first_pil

    logger.info(
        f"[inference] Executing LTX-2.5 ({params.mode.value}) | "
        f"{width}x{height} | {params.num_frames} frames @ {params.fps}fps | "
        f"steps={params.num_inference_steps} | cfg={params.guidance_scale} | seed={seed}"
    )

    t0 = time.monotonic()

    try:
        with torch.inference_mode():
            # Pass extra compatibility kwargs safely if pipeline requires
            try:
                result = pipeline(**kwargs)
            except TypeError as te:
                # Handle parameter name differences between diffusers pipeline versions
                err_msg = str(te)
                if "unexpected keyword argument 'frame_rate'" in err_msg:
                    kwargs.pop("frame_rate", None)
                    kwargs["fps"] = params.fps
                    result = pipeline(**kwargs)
                elif "unexpected keyword argument 'fps'" in err_msg:
                    kwargs.pop("fps", None)
                    kwargs["frame_rate"] = float(params.fps)
                    result = pipeline(**kwargs)
                else:
                    raise

        frames_uint8 = _extract_frames_as_uint8(result)

    except torch.cuda.OutOfMemoryError as oom:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.error(f"[inference] CUDA Out of Memory: {oom}")
        raise RuntimeError("out_of_memory") from oom
    except Exception as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.exception(f"[inference] Generation failed: {exc}")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.monotonic() - t0
    logger.info(f"[inference] Generation finished in {elapsed:.2f}s.")

    return frames_uint8, seed


# ─────────────────────────────────────────────────────────────────────────────
# Internal Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _get_ltx_condition_cls() -> Any:
    """Retrieve LTXVideoCondition if present in diffusers."""
    try:
        from diffusers.pipelines.ltx.pipeline_ltx_video import LTXVideoCondition
        return LTXVideoCondition
    except Exception:
        try:
            from diffusers import LTXVideoCondition
            return LTXVideoCondition
        except Exception:
            return None


def _decode_image(source: str) -> Image.Image:
    """Decode an image from URL or base64 string into a RGB PIL Image."""
    if source.startswith("http://") or source.startswith("https://"):
        import requests
        response = requests.get(source, timeout=30)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))
    else:
        if "," in source:
            source = source.split(",", 1)[1]
        img_bytes = base64.b64decode(source)
        img = Image.open(io.BytesIO(img_bytes))
    return img.convert("RGB")


def _extract_frames_as_uint8(result: Any) -> np.ndarray:
    """Extract frames from any Diffusers pipeline output and format as uint8 array."""
    video_data = None

    if hasattr(result, "frames"):
        frames_attr = result.frames
        if isinstance(frames_attr, list) and len(frames_attr) > 0:
            video_data = frames_attr[0]
        else:
            video_data = frames_attr
    elif isinstance(result, (tuple, list)):
        video_data = result[0]
        if isinstance(video_data, list) and len(video_data) > 0:
            video_data = video_data[0]
    elif isinstance(result, dict):
        video_data = result.get("videos", result.get("frames"))
        if isinstance(video_data, list) and len(video_data) > 0:
            video_data = video_data[0]
    else:
        video_data = result

    # Convert torch.Tensor to numpy
    if isinstance(video_data, torch.Tensor):
        video_data = video_data.detach().cpu().numpy()

    # If shape has batch dim (1, T, H, W, 3), squeeze batch dim
    if isinstance(video_data, np.ndarray) and video_data.ndim == 5:
        video_data = video_data[0]

    # Convert float [0.0, 1.0] to uint8 [0, 255]
    if isinstance(video_data, np.ndarray):
        if np.issubdtype(video_data.dtype, np.floating):
            return (np.clip(video_data, 0.0, 1.0) * 255.0).astype(np.uint8)
        return video_data.astype(np.uint8)

    raise ValueError(f"Unrecognized pipeline output format: {type(result)}")
