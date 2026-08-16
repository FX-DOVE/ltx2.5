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
    to model load time with no measurable quality benefit at 96 GB VRAM.
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

VOLUME_ROOT = Path(os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume"))
WEIGHTS_DIR = VOLUME_ROOT / "models" / "ltx-2.5"
HF_REPO_ID = "Lightricks/LTX-Video-2-0-5B-Distilled"  # The public HF repo name
# NOTE: If Lightricks releases a newer repo ID or renames, update this constant.

# Sentinel files we check to confirm the download is complete.
# Checking a file list is more robust than checking dir existence (a partial
# download would leave the directory but miss some files).
SENTINEL_FILES = [
    "transformer/config.json",
    "vae/config.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "tokenizer/tokenizer.json",
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
    Thread-safety: Runpod workers are single-threaded per handler invocation,
    so we don't need a lock here.
    """
    global _pipeline

    if _pipeline is not None:
        logger.debug("[model_loader] Pipeline already loaded — reusing singleton.")
        return _pipeline

    t0 = time.monotonic()
    logger.info(f"[model_loader] Loading LTX-2.5 pipeline from {WEIGHTS_DIR} …")

    # Free any lingering allocations before loading new model weights.
    torch.cuda.empty_cache()

    try:
        # Deferred import: we only need this at load time, not at module import.
        # This keeps startup fast if load_pipeline() is never called (e.g. in tests).
        from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline  # type: ignore

        pipeline = LTXVideoPipeline.from_pretrained(
            str(WEIGHTS_DIR),
            torch_dtype=torch.bfloat16,
            # device_map="balanced" would split across GPUs; for a single GPU
            # this is equivalent to moving everything to cuda:0.
            device_map="balanced",
        )
        # Ensure everything is on the primary CUDA device.
        pipeline = pipeline.to("cuda")

        # Optional: enable memory-efficient attention if xFormers is available.
        # xFormers gives ~10-15% speedup with no quality loss on Ampere+ GPUs.
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
            "Ensure a network volume is attached and mounted at /runpod-volume "
            "in the Runpod endpoint configuration."
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
    because Lightricks/LTX-Video is a gated repository.
    """
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. "
            "Add it as a Runpod secret and reference it in the endpoint env vars. "
            "The token must have 'read access to gated repos' scope and you must "
            "have accepted the model license at huggingface.co/Lightricks/LTX-Video-2-0-5B-Distilled"
        )

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # Enable hf_transfer for faster multipart downloads if installed.
    # The ENV var is set in the Dockerfile; this is a belt-and-suspenders check.
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0") == "1":
        logger.info("[model_loader] hf_transfer enabled for accelerated download.")

    logger.info(
        f"[model_loader] Downloading '{HF_REPO_ID}' → '{WEIGHTS_DIR}' …\n"
        "This only happens once on a fresh volume. Estimated download size: ~25–30 GB."
    )

    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="model",
        local_dir=str(WEIGHTS_DIR),
        local_dir_use_symlinks=False,  # store real files on the volume, not symlinks
        token=hf_token,
        ignore_patterns=[
            "*.msgpack",       # Flax weights — not needed for PyTorch inference
            "*.h5",            # Keras/TF weights
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
