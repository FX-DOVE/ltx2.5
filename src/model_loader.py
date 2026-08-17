"""
src/model_loader.py
─────────────────────────────────────────────────────────────────────────────
Network-volume weight management and model loading for LTX-2.5.

Supports:
  • MODEL_ID              (default: "Lightricks/LTX-2.5-Diffusers")
  • DTYPE                 (default: "bfloat16")
  • GPU_MEMORY_UTILIZATION (default: 0.95)
  • HF_TOKEN              (for gated repo download)
  • RUNPOD_VOLUME_PATH    (default: "/runpod-volume")
  • Optimized for NVIDIA RTX PRO 6000 96GB VRAM
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
from huggingface_hub import snapshot_download
from loguru import logger

# ─────────────────────────────────────────────────────────────────────────────
# Configuration & Constants
# ─────────────────────────────────────────────────────────────────────────────

VOLUME_ROOT = Path(os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume"))
MODEL_ID = os.environ.get("MODEL_ID", "Lightricks/LTX-2.5-Diffusers")
DTYPE_STR = os.environ.get("DTYPE", "bfloat16").lower()
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.95"))

# Local directory where weights are cached on the network volume
_sanitized_repo_name = MODEL_ID.replace("/", "--")
WEIGHTS_DIR = VOLUME_ROOT / "models" / _sanitized_repo_name

# Sentinel files confirming a complete Diffusers pipeline
SENTINEL_FILES = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "transformer/config.json",
    "vae/config.json",
    "text_encoder/config.json",
    "tokenizer/tokenizer.json",
]

# Torch dtype map
DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}

# Module-level pipeline singleton
_pipeline: Optional[Any] = None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def get_torch_dtype() -> torch.dtype:
    """Return the configured torch dtype (default bfloat16)."""
    return DTYPE_MAP.get(DTYPE_STR, torch.bfloat16)


def ensure_weights_present() -> None:
    """
    Check if the model checkpoint exists on the network volume.
    If not, download the snapshot from Hugging Face.
    """
    logger.info(f"[model_loader] Checking weights for '{MODEL_ID}' on volume at {WEIGHTS_DIR}")
    
    if not VOLUME_ROOT.exists():
        logger.warning(
            f"[model_loader] Volume root '{VOLUME_ROOT}' is not mounted. "
            "Will attempt direct loading / downloading to default cache if local storage allows."
        )
        return

    # Check volume write permission
    try:
        test_file = VOLUME_ROOT / ".write_test"
        test_file.touch()
        test_file.unlink()
    except OSError as exc:
        logger.warning(f"[model_loader] Volume root '{VOLUME_ROOT}' write test failed: {exc}")

    missing = _missing_sentinel_files()
    if not missing:
        logger.info(f"[model_loader] All sentinel files present at {WEIGHTS_DIR} - skipping download.")
        return

    logger.info(
        f"[model_loader] Missing components: {missing}. "
        f"Downloading '{MODEL_ID}' to {WEIGHTS_DIR}..."
    )
    _download_weights()


def load_pipeline() -> Any:
    """
    Load LTX-2.5 into GPU memory and return the pipeline singleton.
    Optimized for RTX PRO 6000 96GB VRAM.
    """
    global _pipeline

    if _pipeline is not None:
        logger.debug("[model_loader] Pipeline already loaded - returning cached singleton.")
        return _pipeline

    t0 = time.monotonic()
    dtype = get_torch_dtype()
    hf_token = os.environ.get("HF_TOKEN")

    logger.info(f"[model_loader] Initializing LTX-2.5 Pipeline (dtype={dtype})...")

    # Determine load source
    if (WEIGHTS_DIR / "model_index.json").exists():
        load_source = str(WEIGHTS_DIR)
        logger.info(f"[model_loader] Loading from local volume path: {load_source}")
    else:
        load_source = MODEL_ID
        logger.info(f"[model_loader] Local volume path empty or incomplete. Loading from HF repo: {load_source}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gpu_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"[model_loader] GPU: {gpu_name} ({total_vram_gb:.1f} GB VRAM)")
    else:
        gpu_name = "CPU"
        total_vram_gb = 0.0
        logger.warning("[model_loader] No CUDA GPU detected - running in CPU mode.")

    # Dynamically import pipeline class
    PipelineClass = _resolve_pipeline_class()

    try:
        logger.info(f"[model_loader] Loading {PipelineClass.__name__} from {load_source}...")
        pipeline = PipelineClass.from_pretrained(
            load_source,
            torch_dtype=dtype,
            token=hf_token,
        )

        # Device placement & VRAM optimization
        if torch.cuda.is_available():
            if total_vram_gb >= 70.0:
                logger.info(
                    f"[model_loader] High VRAM detected ({total_vram_gb:.1f} GB >= 70 GB). "
                    "Placing entire pipeline in VRAM (cuda:0) for fastest execution."
                )
                pipeline = pipeline.to("cuda")
            else:
                logger.info(
                    f"[model_loader] Standard VRAM detected ({total_vram_gb:.1f} GB < 70 GB). "
                    "Enabling model CPU offload."
                )
                pipeline.enable_model_cpu_offload()

            # Enable VAE tiling for memory-safe decoding of large videos/frames
            if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_tiling"):
                pipeline.vae.enable_tiling()
                logger.info("[model_loader] VAE tiling enabled.")
            if hasattr(pipeline, "diffusion_decoder") and hasattr(pipeline.diffusion_decoder, "enable_tiling"):
                pipeline.diffusion_decoder.enable_tiling()
                logger.info("[model_loader] Diffusion decoder tiling enabled.")

        # Set eval mode
        if hasattr(pipeline, "eval"):
            pipeline.eval()

        _pipeline = pipeline
        elapsed = time.monotonic() - t0
        logger.info(f"[model_loader] Pipeline successfully loaded and ready in {elapsed:.1f}s.")
        return _pipeline

    except Exception as exc:
        logger.exception(f"[model_loader] Failed to load pipeline: {exc}")
        raise RuntimeError(f"Pipeline loading error: {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# Internal Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_pipeline_class() -> Any:
    """Resolve the appropriate Diffusers pipeline class for LTX-2.5."""
    try:
        from diffusers import LTX2Pipeline
        logger.info("[model_loader] Using diffusers.LTX2Pipeline")
        return LTX2Pipeline
    except ImportError:
        pass

    try:
        from diffusers import LTXVideoPipeline
        logger.info("[model_loader] Using diffusers.LTXVideoPipeline")
        return LTXVideoPipeline
    except ImportError:
        pass

    try:
        from diffusers import AutoPipelineForText2Video
        logger.info("[model_loader] Using diffusers.AutoPipelineForText2Video")
        return AutoPipelineForText2Video
    except ImportError as e:
        logger.error(f"[model_loader] Failed to import Diffusers pipeline: {e}")
        raise ImportError("diffusers is required to run LTX-2.5.") from e


def _missing_sentinel_files() -> list[str]:
    """Check which sentinel files are missing from the weights directory."""
    missing = []
    for rel_path in SENTINEL_FILES:
        full = WEIGHTS_DIR / rel_path
        if not full.exists() or full.stat().st_size == 0:
            missing.append(rel_path)
    return missing


def _download_weights() -> None:
    """Download model weights using snapshot_download."""
    hf_token = os.environ.get("HF_TOKEN")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0") == "1":
        logger.info("[model_loader] HF Transfer acceleration active.")

    t0 = time.monotonic()
    logger.info(f"[model_loader] Starting snapshot_download of '{MODEL_ID}' -> {WEIGHTS_DIR}")

    try:
        snapshot_download(
            repo_id=MODEL_ID,
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
                "transformer_full/*",  # Exclude raw transformer_full unless requested to save disk
            ],
        )
        elapsed = time.monotonic() - t0
        logger.info(f"[model_loader] Download completed in {elapsed:.1f}s.")
    except Exception as exc:
        logger.error(
            f"[model_loader] Failed to download '{MODEL_ID}' from Hugging Face: {exc}\n"
            "If this is a gated model, verify that:\n"
            "1. You have requested and been granted access on https://huggingface.co/" + MODEL_ID + "\n"
            "2. Your HF_TOKEN has 'read access to gated repos' permissions."
        )
        raise
