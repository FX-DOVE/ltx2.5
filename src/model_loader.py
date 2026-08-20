"""
src/model_loader.py
─────────────────────────────────────────────────────────────────────────────
Network-volume weight management and model loading for LTX-2.5.

Responsibilities
────────────────
1. ensure_weights_present()
   Called once at worker cold-start. Checks whether the LTX-2.5 checkpoint
   files already exist on the Runpod network volume. If they do, it fast-
   paths past any network I/O. If they don't (first-ever cold start on a
   fresh volume), it downloads them from the gated Hugging Face repo using
   huggingface_hub snapshot_download.

2. load_pipeline()
   Loads all model components (VAE, text encoder, transformer, scheduler)
   into GPU memory and returns a ready-to-call LTXVideoPipeline. This is
   called once per worker lifetime and the result is kept warm in a module-
   level singleton.

Design decisions worth noting:
  • We use snapshot_download with local_dir=WEIGHTS_DIR and
    local_dir_use_symlinks=False so that the real files live on the volume
    and are not just symlinks into the HF cache. This is critical because
    the HF_HOME cache directory is also on the volume — if both pointed at
    the same location we'd get double storage usage.
  • Weight existence is checked by verifying individual sentinel files
    (not just the directory) to guard against partial downloads.
  • torch.cuda.empty_cache() is called before model load to maximise
    available VRAM — this matters if the handler process survived a previous
    request that fragmented the allocator.
  • We load in bfloat16 (the native LTX-2.5 dtype). int8 quantisation is
    available via quanto but is not the default because it adds 2–3 minutes
    to model load time with no measurable quality benefit at 48 GB VRAM.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import snapshot_download
from loguru import logger

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

VOLUME_PATH_OVERRIDE = os.environ.get("RUNPOD_VOLUME_PATH")

def get_volume_root() -> Path:
    """Detect mounted network volume across Serverless (/runpod-volume) and Pods (/workspace)."""
    if VOLUME_PATH_OVERRIDE:
        return Path(VOLUME_PATH_OVERRIDE)
    for candidate in [Path("/runpod-volume"), Path("/workspace")]:
        if (candidate / "models" / "ltx-2.5").exists():
            return candidate
    for candidate in [Path("/runpod-volume"), Path("/workspace")]:
        if candidate.exists():
            return candidate
    return Path("/runpod-volume")

VOLUME_ROOT = get_volume_root()
WEIGHTS_DIR = VOLUME_ROOT / "models" / "ltx-2.5"
HF_REPO_ID = "Lightricks/LTX-2.5"

# ─────────────────────────────────────────────────────────────────────────────
# Model file layout matching Lightricks/LTX-2.5 safetensors repository
# ─────────────────────────────────────────────────────────────────────────────
MODEL_FILES = {
    # Transformers
    "transformer_distilled_bf16": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "transformer_distilled_int8": "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
    "transformer_distilled_nvfp4": "diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors",
    "transformer_dev_bf16": "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
    "transformer_dev_int8": "diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors",
    # Text Encoders
    "text_encoder_bf16": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "text_encoder_int8": "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    # VAEs
    "video_vae_bf16": "vae/ltx-2.5-video-vae-bf16.safetensors",
    "audio_vae_bf16": "vae/ltx-2.5-audio-vae-bf16.safetensors",
    # Patches & Upscalers
    "duration_head_bf16": "model_patches/ltx-2.5-duration-head-bf16.safetensors",
    "spatial_upscaler_bf16": "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    "temporal_upscaler_bf16": "latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors",
    # LoRAs
    "lora_distilled_bf16": "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
}

# Sentinel files required for standard LTX-2.5 inference
SENTINEL_FILES = [
    MODEL_FILES["transformer_distilled_bf16"],
    MODEL_FILES["text_encoder_bf16"],
    MODEL_FILES["video_vae_bf16"],
    MODEL_FILES["spatial_upscaler_bf16"],
]

# Module-level pipeline singleton — loaded once per worker process.
_pipeline: Optional[object] = None  # type: ignore[assignment]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def ensure_weights_present() -> None:
    """
    Idempotent weight-availability check.  Safe to call on every cold start.

    Behaviour:
      • If all sentinel files exist on the volume → log and return immediately.
      • Otherwise → download the full snapshot to WEIGHTS_DIR via HF Hub.

    Raises:
      RuntimeError   if HF_TOKEN is missing or download fails.
      EnvironmentError if the network volume is not mounted / writable.
    """
    _assert_volume_mounted()

    t0 = time.monotonic()
    missing = _missing_sentinel_files()

    if not missing:
        elapsed = time.monotonic() - t0
        logger.info(
            f"[model_loader] Weights already present on volume at {WEIGHTS_DIR} "
            f"(checked in {elapsed:.2f}s) — skipping download."
        )
        return

    logger.info(
        f"[model_loader] Missing sentinel files: {missing}. "
        f"Starting download from '{HF_REPO_ID}' → {WEIGHTS_DIR}"
    )
    _download_weights()
    elapsed = time.monotonic() - t0
    logger.info(f"[model_loader] Download complete in {elapsed:.1f}s.")


def load_pipeline() -> object:
    """
    Load LTX-2.5 into GPU memory and return the pipeline singleton.

    Subsequent calls return the cached singleton without reloading.
    Supports both official ltx_pipelines (DistilledPipeline) and diffusers LTXVideoPipeline.
    """
    global _pipeline

    if _pipeline is not None:
        logger.debug("[model_loader] Pipeline already loaded — reusing singleton.")
        return _pipeline

    t0 = time.monotonic()
    logger.info(f"[model_loader] Loading LTX-2.5 pipeline from {WEIGHTS_DIR} ...")

    # Free any lingering allocations before loading new model weights.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        # First attempt: Official LTX-2.5 distilled / dev pipeline
        try:
            from ltx_pipelines.distilled import DistilledPipeline
            from ltx_pipelines.utils.model_paths import ModelPaths
            from ltx_core.model.video_vae.transformer import DiffVAEMode
            from ltx_pipelines.utils.types import OffloadMode

            # Prefer distilled bf16 on L40S (48GB), fallback to int8 or dev
            transformer_path = WEIGHTS_DIR / MODEL_FILES["transformer_distilled_bf16"]
            if not transformer_path.exists():
                transformer_path = WEIGHTS_DIR / MODEL_FILES["transformer_distilled_int8"]
            if not transformer_path.exists():
                transformer_path = WEIGHTS_DIR / MODEL_FILES["transformer_dev_bf16"]

            text_encoder_path = WEIGHTS_DIR / MODEL_FILES["text_encoder_bf16"]
            if not text_encoder_path.exists():
                text_encoder_path = WEIGHTS_DIR / MODEL_FILES["text_encoder_int8"]

            video_vae_path = WEIGHTS_DIR / MODEL_FILES["video_vae_bf16"]
            audio_vae_path = WEIGHTS_DIR / MODEL_FILES["audio_vae_bf16"]
            duration_head_path = WEIGHTS_DIR / MODEL_FILES["duration_head_bf16"]
            spatial_upscaler_path = WEIGHTS_DIR / MODEL_FILES["spatial_upscaler_bf16"]

            model_paths = ModelPaths.from_split(
                transformer_path=str(transformer_path),
                text_encoder_path=str(text_encoder_path),
                video_vae_path=str(video_vae_path),
                audio_vae_path=str(audio_vae_path) if audio_vae_path.exists() else None,
                duration_head_path=str(duration_head_path) if duration_head_path.exists() else None,
            )

            pipeline = DistilledPipeline(
                model_paths=model_paths,
                spatial_upsampler_path=str(spatial_upscaler_path) if spatial_upscaler_path.exists() else None,
                loras=(),
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                quantization=None,
                compilation_config=None,
                offload_mode=OffloadMode.NONE,
                diffvae_optimization=DiffVAEMode.CHUNKED_EAGER,
            )
            logger.info("[model_loader] DistilledPipeline loaded successfully via ltx_pipelines.")
            _pipeline = pipeline
            elapsed = time.monotonic() - t0
            logger.info(f"[model_loader] Pipeline ready in {elapsed:.1f}s.")
            return _pipeline

        except (ImportError, AttributeError) as exc:
            logger.debug(f"[model_loader] ltx_pipelines not used ({exc}), falling back to diffusers LTXVideoPipeline.")

        # Second attempt: diffusers LTXVideoPipeline
        try:
            from diffusers import LTXVideoPipeline  # type: ignore
        except ImportError:
            from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline  # type: ignore

        pipeline = LTXVideoPipeline.from_pretrained(
            str(WEIGHTS_DIR),
            torch_dtype=torch.bfloat16,
            device_map="balanced",
        )
        if torch.cuda.is_available():
            pipeline = pipeline.to("cuda")

        try:
            pipeline.enable_xformers_memory_efficient_attention()
            logger.info("[model_loader] xFormers memory-efficient attention enabled.")
        except Exception:
            logger.warning(
                "[model_loader] xFormers not available — using native SDPA attention."
            )

        _pipeline = pipeline
        elapsed = time.monotonic() - t0
        logger.info(f"[model_loader] Pipeline ready in {elapsed:.1f}s.")
        return _pipeline

    except torch.cuda.OutOfMemoryError as oom:
        logger.error(f"[model_loader] CUDA OOM during model load: {oom}")
        raise RuntimeError("out_of_memory_during_load") from oom
    except Exception as exc:
        logger.exception(f"[model_loader] Failed to load pipeline: {exc}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _assert_volume_mounted() -> None:
    """Fail fast if the network volume isn't mounted."""
    if not VOLUME_ROOT.exists():
        raise EnvironmentError(
            f"Runpod network volume not found at '{VOLUME_ROOT}'. "
            "Ensure a network volume is attached and mounted at /runpod-volume (or /workspace) "
            "in the Runpod configuration."
        )
    # Confirm write access
    test_file = VOLUME_ROOT / ".write_test"
    try:
        test_file.touch()
        test_file.unlink()
    except OSError as exc:
        raise EnvironmentError(
            f"Network volume at '{VOLUME_ROOT}' is not writable: {exc}"
        ) from exc


def _missing_sentinel_files() -> list[str]:
    """Return the list of expected sentinel files that are absent or zero-size."""
    missing = []
    for rel_path in SENTINEL_FILES:
        full = WEIGHTS_DIR / rel_path
        if not full.exists() or full.stat().st_size == 0:
            missing.append(rel_path)
    return missing


def _download_weights() -> None:
    """
    Download all LTX-2.5 model files to the network volume.

    Uses snapshot_download so that HF handles sharded files, retries,
    and partial-download resumption transparently.

    IMPORTANT: HF_TOKEN must be set in the environment (via Runpod secret)
    because Lightricks/LTX-2.5 is a gated repository.
    """
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. "
            "Add it as a Runpod secret and reference it in the endpoint env vars. "
            "The token must have 'read access to gated repos' scope and you must "
            "have accepted the model license at huggingface.co/Lightricks/LTX-2.5"
        )

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0") == "1":
        logger.info("[model_loader] hf_transfer enabled for accelerated download.")

    logger.info(
        f"[model_loader] Downloading '{HF_REPO_ID}' -> '{WEIGHTS_DIR}' ...\n"
        "This only happens once on a fresh volume."
    )

    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="model",
        local_dir=str(WEIGHTS_DIR),
        local_dir_use_symlinks=False,
        token=hf_token,
        ignore_patterns=[
            "*.msgpack",
            "*.h5",
            "flax_model*",
            "tf_model*",
            "rust_model*",
            "*.ot",
        ],
    )

    # Final sanity check: confirm sentinels are now present.
    still_missing = _missing_sentinel_files()
    if still_missing:
        raise RuntimeError(
            f"Download appeared to complete but sentinel files are still missing: "
            f"{still_missing}. Check disk space on the network volume."
        )
