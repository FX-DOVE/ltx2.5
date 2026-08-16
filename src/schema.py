"""
src/schema.py
─────────────────────────────────────────────────────────────────────────────
Pydantic v2 input schemas for the LTX-2.5 serverless handler.

Three generation modes are supported:
  • text2video  – prompt → video (no reference image required)
  • image2video – prompt + first frame image → video
  • flf2video   – prompt + first frame + last frame images → video
         (First-Last-Frame conditioning, a key LTX-2.5 feature)

All optional fields have sensible defaults that keep VRAM usage within the
RTX PRO 6000 96 GB budget.  Documented per-field so callers know what to
expect without reading inference.py.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class GenerationMode(str, Enum):
    text2video = "text2video"
    image2video = "image2video"
    flf2video = "flf2video"   # first-last-frame conditioning


class Resolution(str, Enum):
    # Common LTX-2.5 resolutions (width x height).
    # Odd multiples of 32 accepted by the pipeline but not exposed here.
    r480p = "480p"    # 848x480
    r720p = "720p"    # 1280x720
    r1080p = "1080p"  # 1920x1080 — requires plenty of VRAM, use with caution


# Pixel dimensions for each resolution token
RESOLUTION_MAP: dict[Resolution, tuple[int, int]] = {
    Resolution.r480p: (848, 480),
    Resolution.r720p: (1280, 720),
    Resolution.r1080p: (1920, 1080),
}


class InferenceInput(BaseModel):
    """
    Validated input for a single generation request.
    All fields except `prompt` have defaults so callers can send minimal JSON.
    """

    # ── Required ──────────────────────────────────────────────────────────────
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Text description of the video to generate.",
    )

    # ── Mode ──────────────────────────────────────────────────────────────────
    mode: GenerationMode = Field(
        default=GenerationMode.text2video,
        description=(
            "Generation mode. "
            "'text2video': prompt only. "
            "'image2video': prompt + first_frame_image. "
            "'flf2video': prompt + first_frame_image + last_frame_image."
        ),
    )

    # ── Conditioning images (base64-encoded JPEG/PNG or HTTPS URL) ────────────
    first_frame_image: Optional[str] = Field(
        default=None,
        description=(
            "Base64-encoded image (JPEG/PNG) OR an HTTPS URL for the first frame. "
            "Required when mode is 'image2video' or 'flf2video'."
        ),
    )
    last_frame_image: Optional[str] = Field(
        default=None,
        description=(
            "Base64-encoded image (JPEG/PNG) OR an HTTPS URL for the last frame. "
            "Required when mode is 'flf2video'."
        ),
    )

    # ── Negative prompt ───────────────────────────────────────────────────────
    negative_prompt: str = Field(
        default=(
            "low quality, worst quality, deformed, distorted, disfigured, "
            "motion smear, motion artifacts, fused fingers, bad anatomy, "
            "weird hand, ugly"
        ),
        max_length=500,
        description="Negative conditioning text.",
    )

    # ── Output dimensions ─────────────────────────────────────────────────────
    resolution: Resolution = Field(
        default=Resolution.r720p,
        description="Output video resolution.",
    )

    # ── Temporal settings ─────────────────────────────────────────────────────
    num_frames: int = Field(
        default=97,
        ge=9,
        le=257,
        description=(
            "Number of frames to generate. LTX-2.5 requires (N-1) divisible by 8. "
            "Values like 9, 17, 25, 33, ..., 97, 121, 145, 161, 193, 225, 257 are valid."
        ),
    )
    fps: int = Field(
        default=24,
        ge=8,
        le=30,
        description="Playback frame rate of the output video.",
    )

    # ── Sampling settings ─────────────────────────────────────────────────────
    num_inference_steps: int = Field(
        default=40,
        ge=10,
        le=100,
        description="Diffusion denoising steps. 40 is the recommended default.",
    )
    guidance_scale: float = Field(
        default=3.5,
        ge=1.0,
        le=10.0,
        description="Classifier-free guidance scale.",
    )
    seed: Optional[int] = Field(
        default=None,
        ge=0,
        le=2**31 - 1,
        description="RNG seed for reproducibility. Omit for random.",
    )

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("num_frames")
    @classmethod
    def frames_must_be_valid(cls, v: int) -> int:
        """LTX-2.5 requires (num_frames - 1) % 8 == 0."""
        if (v - 1) % 8 != 0:
            raise ValueError(
                f"num_frames={v} is invalid. (num_frames - 1) must be divisible by 8. "
                f"Try one of: 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, …"
            )
        return v

    @model_validator(mode="after")
    def check_image_requirements(self) -> "InferenceInput":
        """
        Dynamic mode resolution:
          1. If mode is explicitly set to image2video/flf2video, enforce that
             the required images are present.
          2. If mode is text2video (the default) but images ARE provided, auto-
             promote the mode so callers don't have to specify it explicitly:
               • first_frame_image only  → image2video
               • first + last frame      → flf2video
          3. If mode requires images but none are provided, raise a clear error.
        """
        # Auto-promote: text2video + images provided → pick the right mode
        if self.mode == GenerationMode.text2video:
            if self.first_frame_image and self.last_frame_image:
                self.mode = GenerationMode.flf2video
            elif self.first_frame_image:
                self.mode = GenerationMode.image2video

        # Enforce: image-based modes must have at least the first frame
        if self.mode in (GenerationMode.image2video, GenerationMode.flf2video):
            if not self.first_frame_image:
                raise ValueError(
                    f"mode='{self.mode.value}' requires 'first_frame_image' to be provided."
                )
        if self.mode == GenerationMode.flf2video:
            if not self.last_frame_image:
                raise ValueError(
                    "mode='flf2video' requires both 'first_frame_image' and 'last_frame_image'."
                )
        return self


class InferenceOutput(BaseModel):
    """Structured success response returned by the handler."""

    status: str = "success"
    video_url: Optional[str] = None      # presigned URL or runpod temp-storage URL
    video_base64: Optional[str] = None   # fallback for small outputs
    duration_seconds: float = Field(description="Duration of the generated video in seconds.")
    generation_time_seconds: float = Field(description="Wall-clock inference time.")
    seed_used: int = Field(description="The actual seed used (useful if seed was random).")
    resolution: str
    num_frames: int
    fps: int


class ErrorOutput(BaseModel):
    """Structured error response."""

    status: str = "error"
    error: str          # machine-readable error code, e.g. "out_of_memory"
    message: str        # human-readable description
    retryable: bool = False
