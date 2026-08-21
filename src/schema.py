"""
src/schema.py
─────────────────────────────────────────────────────────────────────────────
Pydantic v2 input schemas for the LTX-2.5 serverless handler.

Three generation modes are supported:
  • text2video  – prompt → video (no reference image required)
  • image2video – prompt + first frame image → video
  • flf2video   – prompt + first frame + last frame images → video
         (First-Last-Frame conditioning, a key LTX-2.5 feature)

Default target GPU: NVIDIA L40S (48 GB VRAM).
Default resolution: 450p (768×448) — fits comfortably in L40S VRAM.
Max duration:       10 seconds at 24 fps = 241 frames.
                    (241−1=240, 240÷8=30 ✅ satisfies LTX-2.5 constraint)
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
    """
    LTX-2.5 resolution tokens.

    The DistilledPipeline is a TWO-STAGE pipeline: stage 1 renders at half
    resolution and stage 2 upsamples ×2.  Upstream `assert_resolution(h, w,
    is_two_stage=True)` therefore requires BOTH width and height to be
    divisible by 64 — not 32.  The previous values (848×480, 1280×720,
    1920×1080) all failed that check and raised before a single step ran.
    """

    r450p = "450p"    # 768×448   — DEFAULT for L40S (48 GB)
    r480p = "480p"    # 896×512   — nearest legal size to 848×480
    r576p = "576p"    # 1024×576  — mid-size 16:9
    r720p = "720p"    # 1280×704  — comfortable on L40S for short clips
    r1080p = "1080p"  # 1920×1088 — upstream's own "1080p" HQ preset


# Pixel dimensions for each resolution token — (width, height), both %64 == 0.
RESOLUTION_MAP: dict[Resolution, tuple[int, int]] = {
    Resolution.r450p: (768, 448),
    Resolution.r480p: (896, 512),
    Resolution.r576p: (1024, 576),
    Resolution.r720p: (1280, 704),
    Resolution.r1080p: (1920, 1088),
}


class InferenceInput(BaseModel):
    """
    Validated input for a single generation request.
    All fields except `prompt` have defaults so callers can send minimal JSON.

    Designed for NVIDIA L40S (48 GB VRAM) serverless endpoint.
    Default resolution: 450p (768×448).  Max duration: 10 s (241 frames @ 24 fps).
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
    # NOTE: the distilled LTX-2.5 pipeline is guidance-distilled and has no
    # negative-prompt branch. Kept for API compatibility; it does not affect
    # the output.
    negative_prompt: str = Field(
        default=(
            "low quality, worst quality, deformed, distorted, disfigured, "
            "motion smear, motion artifacts, fused fingers, bad anatomy, "
            "weird hand, ugly"
        ),
        max_length=500,
        description=(
            "ACCEPTED BUT IGNORED. The distilled checkpoint has no negative "
            "conditioning branch."
        ),
    )

    # ── Output dimensions ─────────────────────────────────────────────────────
    resolution: Resolution = Field(
        default=Resolution.r450p,
        description=(
            "Output video resolution. Default '450p' (768×448) is optimised for "
            "the L40S 48 GB GPU. '480p'=896×512, '576p'=1024×576, "
            "'720p'=1280×704, '1080p'=1920×1088."
        ),
    )

    # ── Temporal settings ─────────────────────────────────────────────────────
    # Max 10 seconds at 24 fps = 241 frames (241−1=240, 240÷8=30 ✅).
    num_frames: int = Field(
        default=241,
        ge=9,
        le=257,
        description=(
            "Number of frames to generate. LTX-2.5 requires (N-1) divisible by 8. "
            "Valid values: 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, ..., 241, 257. "
            "Max 241-257 frames = ~10 s at 24 fps (L40S memory cap)."
        ),
    )
    fps: int = Field(
        default=24,
        ge=8,
        le=30,
        description="Playback frame rate of the output video.",
    )

    # ── Sampling settings ─────────────────────────────────────────────────────
    # NOTE: the distilled LTX-2.5 checkpoint uses BAKED-IN sigma schedules —
    # 8 steps in stage 1 and 3 steps in stage 2 — and is guidance-distilled, so
    # it takes no CFG scale.  `num_inference_steps` and `guidance_scale` are
    # accepted for backwards compatibility with existing callers but have NO
    # effect on the output.  They are echoed back in the response for clarity.
    num_inference_steps: int = Field(
        default=40,
        ge=10,
        le=100,
        description=(
            "ACCEPTED BUT IGNORED. The distilled checkpoint uses a fixed "
            "8-step (stage 1) + 3-step (stage 2) sigma schedule."
        ),
    )
    guidance_scale: float = Field(
        default=3.5,
        ge=1.0,
        le=10.0,
        description=(
            "ACCEPTED BUT IGNORED. The distilled checkpoint is "
            "guidance-distilled and does not run classifier-free guidance."
        ),
    )
    conditioning_strength: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description=(
            "How strongly the conditioning image(s) constrain the result. "
            "1.0 pins the frame exactly; lower values allow more deviation. "
            "Ignored in text2video mode."
        ),
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
        """LTX-2.5 requires (num_frames - 1) % 8 == 0. Max 241 (10 s @ 24 fps)."""
        if (v - 1) % 8 != 0:
            raise ValueError(
                f"num_frames={v} is invalid. (num_frames - 1) must be divisible by 8. "
                f"Try one of: 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, …, 241"
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
    has_audio: bool = Field(
        default=False,
        description="True when LTX-2.5 generated an audio track that was muxed into the MP4.",
    )


class ErrorOutput(BaseModel):
    """Structured error response."""

    status: str = "error"
    error: str          # machine-readable error code, e.g. "out_of_memory"
    message: str        # human-readable description
    retryable: bool = False
