"""
src/inference.py
─────────────────────────────────────────────────────────────────────────────
Inference wrapper for the LTX-2.5 `DistilledPipeline`.

This module translates the validated InferenceInput schema into the exact
call signature of `ltx_pipelines.distilled.DistilledPipeline.__call__`:

    pipeline(
        prompt: str,
        seed: int,
        height: int,
        width: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        num_frames: int | AutoDuration = DEFAULT_AUTO_DURATION,
        tiling_config: TilingConfig | AutoTiling | None = AUTO_TILING,
        ...
    ) -> tuple[Iterator[torch.Tensor], Audio, int, TilingConfig | None]

Notes on what the pipeline does NOT accept
──────────────────────────────────────────
The distilled checkpoint is guidance-distilled with a baked-in sigma schedule
(8 steps in stage 1, 3 in stage 2), so there is no `negative_prompt`,
`num_inference_steps` or `guidance_scale` kwarg. Passing them raises
TypeError. The schema still accepts those fields for API compatibility; they
are ignored here.

Conditioning
────────────
`ImageConditioningInput` takes a *filesystem path*, not a PIL image, so
decoded images are written to temporary files for the duration of the call.
Per upstream `combined_image_conditionings()`:
  • frame_idx == 0  → replacing latent (the frame is pinned as frame 0)
  • frame_idx  > 0  → guiding keyframe (used for first-last-frame conditioning)

Resolution
──────────
`height`/`width` are the FINAL output size. The pipeline renders stage 1 at
half of it and upsamples ×2, so upstream `assert_resolution(..., is_two_stage
=True)` requires both to be divisible by 64. See schema.RESOLUTION_MAP.

Output
──────
run_inference() returns the raw pipeline result (a lazily-decoded chunk
iterator plus the audio track), NOT decoded frames — the caller hands it
straight to `ltx_pipelines.utils.media_io.encode_video`, which streams chunks
into libx264 and muxes the audio. Materialising frames as a numpy array would
defeat the chunked decoder and roughly double peak host memory.

GPU memory notes (NVIDIA L40S, 48 GB VRAM, fp8-cast quantization):
  • 450p (768×448) / 241 frames — comfortable.
  • 720p (1280×704) / 121 frames — fits with the chunked DiffVAE decoder.
  • 1080p (1920×1088) — only with short clips; prefer LTX_OFFLOAD_MODE=cpu.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import base64
import io
import random
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import torch
from loguru import logger
from PIL import Image, UnidentifiedImageError

from schema import RESOLUTION_MAP, GenerationMode, InferenceInput


@dataclass
class InferenceResult:
    """
    Everything the encoder needs, in the order `encode_video` wants it.

    Attributes:
        video:  Iterator of (F, H, W, C) float [0, 1] RGB chunks from the VAE.
        audio:  Generated audio track (waveform + sampling_rate), or None.
        num_frames: Frame count actually generated (the duration head may pick
            this when the caller does not).
        tiling_config: Tiling actually used, needed for the chunk count.
        seed: The seed that was used.
        width / height: Final output dimensions.
    """

    video: Iterator[torch.Tensor]
    audio: Any
    num_frames: int
    tiling_config: Any
    seed: int
    width: int
    height: int
    # Temp dir holding conditioning images; the caller must call cleanup()
    # after encoding finishes (the frame iterator is lazy).
    _tmpdir: Optional[str] = None

    def cleanup(self) -> None:
        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def run_inference(pipeline: object, params: InferenceInput) -> InferenceResult:
    """
    Run LTX-2.5 inference.

    Returns:
        InferenceResult — the lazy frame iterator, audio track, resolved frame
        count and tiling config, ready to hand to encode_video().

    Raises:
        RuntimeError("out_of_memory") on CUDA OOM.
        ValueError on unusable conditioning images.
    """
    from ltx_core.model.video_vae import AUTO_TILING
    from ltx_pipelines.utils.args import ImageConditioningInput

    seed = params.seed if params.seed is not None else random.randint(0, 2**31 - 1)
    width, height = RESOLUTION_MAP[params.resolution]

    logger.info(
        f"[inference] {params.mode.value} | {width}x{height} | "
        f"{params.num_frames} frames @ {params.fps} fps | seed={seed}"
    )

    # ── Conditioning images ──────────────────────────────────────────────────
    # ImageConditioningInput wants a path, so decoded images are staged on disk.
    tmpdir: Optional[str] = None
    images: list = []

    if params.mode in (GenerationMode.image2video, GenerationMode.flf2video):
        tmpdir = tempfile.mkdtemp(prefix="ltx-cond-")
        try:
            first_path = _stage_image(
                params.first_frame_image, Path(tmpdir) / "first.png", "first_frame_image"
            )
            images.append(
                ImageConditioningInput(
                    path=str(first_path),
                    frame_idx=0,  # frame 0 → replacing latent
                    strength=params.conditioning_strength,
                )
            )
            if params.mode == GenerationMode.flf2video:
                last_path = _stage_image(
                    params.last_frame_image,
                    Path(tmpdir) / "last.png",
                    "last_frame_image",
                )
                images.append(
                    ImageConditioningInput(
                        path=str(last_path),
                        # >0 → guiding keyframe. The last generated frame.
                        frame_idx=params.num_frames - 1,
                        strength=params.conditioning_strength,
                    )
                )
        except Exception:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise

    # ── Generate ─────────────────────────────────────────────────────────────
    t0 = time.monotonic()
    try:
        video, audio, num_frames, tiling_config = pipeline(  # type: ignore[operator]
            prompt=params.prompt,
            seed=seed,
            height=height,
            width=width,
            frame_rate=float(params.fps),
            images=images,
            num_frames=params.num_frames,
            tiling_config=AUTO_TILING,
        )
    except torch.cuda.OutOfMemoryError as oom:  # type: ignore[attr-defined]
        _discard(tmpdir)
        torch.cuda.empty_cache()
        logger.error(f"[inference] CUDA OOM: {oom}")
        raise RuntimeError("out_of_memory") from oom
    except Exception as exc:
        _discard(tmpdir)
        torch.cuda.empty_cache()
        if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
            logger.error(f"[inference] CUDA OOM: {exc}")
            raise RuntimeError("out_of_memory") from exc
        logger.exception(f"[inference] Generation failed: {exc}")
        raise

    # Note: `video` is a lazy iterator — the diffusion work above is done, but
    # VAE decode happens as the encoder pulls chunks.
    logger.info(
        f"[inference] Denoising complete in {time.monotonic() - t0:.1f}s "
        f"({num_frames} frames). Decode streams during encode."
    )

    return InferenceResult(
        video=video,
        audio=audio,
        num_frames=num_frames,
        tiling_config=tiling_config,
        seed=seed,
        width=width,
        height=height,
        _tmpdir=tmpdir,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Image staging / decoding helpers
# ─────────────────────────────────────────────────────────────────────────────


def _discard(tmpdir: Optional[str]) -> None:
    """Remove a staging directory if one was created."""
    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _stage_image(source: str, dest: Path, label: str) -> Path:
    """Decode a conditioning image and write it to `dest` as PNG."""
    img = _decode_image(source, label=label)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, format="PNG")
    return dest


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
                raise ValueError(f"{label}: Failed to fetch '{source}': {req_err}")
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
                raise ValueError(f"{label}: Invalid base64 string — could not decode.")
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
