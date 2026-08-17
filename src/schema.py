"""
src/schema.py
─────────────────────────────────────────────────────────────────────────────
Pydantic v2 input/output schemas for the LTX-2.5 serverless handler.

Supported generation modes:
  • text2video  – prompt -> video (no reference image required)
  • image2video – prompt + first frame image -> video
  • flf2video   – prompt + first frame + last frame images -> video
                 (First-Last-Frame conditioning)

Supports:
  • Direct pixel dimensions (width, height) e.g. width=704, height=480
  • Named resolution presets (480p, 720p, 1080p)
  • Explicit or auto-detected mode
  • Aliases for field names (image, frame_rate, etc.)
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class GenerationMode(str, Enum):
    text2video = "text2video"
    image2video = "image2video"
    flf2video = "flf2video"


class Resolution(str, Enum):
    r480p = "480p"    # 848x480
    r720p = "720p"    # 1280x720
    r1080p = "1080p"  # 1920x1080


RESOLUTION_MAP: dict[Resolution, tuple[int, int]] = {
    Resolution.r480p: (848, 480),
    Resolution.r720p: (1280, 720),
    Resolution.r1080p: (1920, 1080),
}


class InferenceInput(BaseModel):
    """
    Validated input for a single video generation request.
    """

    # ── Required ──────────────────────────────────────────────────────────────
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=4000,
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

    # ── Conditioning images ───────────────────────────────────────────────────
    first_frame_image: Optional[str] = Field(
        default=None,
        alias="image",
        description="Base64-encoded image (JPEG/PNG) OR an HTTPS URL for the first frame.",
    )
    last_frame_image: Optional[str] = Field(
        default=None,
        description="Base64-encoded image (JPEG/PNG) OR an HTTPS URL for the last frame.",
    )

    # ── Negative prompt ───────────────────────────────────────────────────────
    negative_prompt: str = Field(
        default=(
            "low quality, worst quality, deformed, distorted, disfigured, "
            "motion smear, motion artifacts, fused fingers, bad anatomy, "
            "weird hand, ugly, blurry"
        ),
        max_length=1000,
        description="Negative conditioning text.",
    )

    # ── Output dimensions ─────────────────────────────────────────────────────
    resolution: Optional[Resolution] = Field(
        default=None,
        description="Named resolution preset ('480p', '720p', '1080p'). Used if width/height are omitted.",
    )
    width: Optional[int] = Field(
        default=None,
        ge=128,
        le=3840,
        description="Output video width in pixels. Must be divisible by 32.",
    )
    height: Optional[int] = Field(
        default=None,
        ge=128,
        le=2160,
        description="Output video height in pixels. Must be divisible by 32.",
    )

    # ── Temporal settings ─────────────────────────────────────────────────────
    num_frames: int = Field(
        default=97,
        ge=9,
        le=257,
        description="Number of frames to generate. Must satisfy (N-1) % 8 == 0 (e.g. 9, 49, 97, 121, 241).",
    )
    fps: int = Field(
        default=24,
        alias="frame_rate",
        ge=8,
        le=60,
        description="Playback frame rate of output video.",
    )

    # ── Sampling settings ─────────────────────────────────────────────────────
    num_inference_steps: Optional[int] = Field(
        default=40,
        ge=5,
        le=100,
        description="Diffusion denoising steps. Default is 40 for standard diffusion, or 8-10 for distilled models.",
    )
    guidance_scale: float = Field(
        default=3.0,
        ge=1.0,
        le=20.0,
        description="Classifier-free guidance scale.",
    )
    seed: Optional[int] = Field(
        default=None,
        ge=0,
        le=2**31 - 1,
        description="RNG seed for reproducibility. Omit for random seed.",
    )

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("num_frames")
    @classmethod
    def validate_frames(cls, v: int) -> int:
        if (v - 1) % 8 != 0:
            raise ValueError(
                f"num_frames={v} is invalid. (num_frames - 1) must be divisible by 8. "
                f"Valid examples: 9, 17, 25, 33, 49, 65, 97, 121, 145, 161, 193, 225, 241, 257."
            )
        return v

    @field_validator("width", "height")
    @classmethod
    def validate_dimensions(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v % 32 != 0:
            raise ValueError(
                f"Dimension {v} must be divisible by 32 (e.g. 704, 480, 768, 1280, 1920)."
            )
        return v

    @model_validator(mode="after")
    def resolve_dimensions_and_mode(self) -> "InferenceInput":
        # Resolve width and height
        if self.width is None or self.height is None:
            res_key = self.resolution or Resolution.r720p
            default_w, default_h = RESOLUTION_MAP[res_key]
            if self.width is None:
                self.width = default_w
            if self.height is None:
                self.height = default_h

        # Auto-promote mode if images are present
        if self.mode == GenerationMode.text2video:
            if self.first_frame_image and self.last_frame_image:
                self.mode = GenerationMode.flf2video
            elif self.first_frame_image:
                self.mode = GenerationMode.image2video

        # Check image requirements
        if self.mode in (GenerationMode.image2video, GenerationMode.flf2video):
            if not self.first_frame_image:
                raise ValueError(
                    f"mode='{self.mode.value}' requires 'first_frame_image' (or 'image') to be provided."
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
    video_url: Optional[str] = None
    video_base64: Optional[str] = None
    duration_seconds: float
    generation_time_seconds: float
    seed_used: int
    resolution: str
    num_frames: int
    fps: int


class ErrorOutput(BaseModel):
    """Structured error response."""

    status: str = "error"
    error: str
    message: str
    retryable: bool = False
