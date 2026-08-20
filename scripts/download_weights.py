#!/usr/bin/env python3
"""
scripts/download_weights.py
─────────────────────────────────────────────────────────────────────────────
Standalone script to pre-populate the network volume with LTX-2.5 weights.

Usage:
  # On a RunPod pod attached to the network volume:
  HF_TOKEN=hf_xxx python scripts/download_weights.py

  # With custom model repo or volume path:
  MODEL_ID=Lightricks/LTX-2.5-Diffusers RUNPOD_VOLUME_PATH=/runpod-volume python scripts/download_weights.py
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import sys

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from loguru import logger
import model_loader


def main() -> None:
    weights_dir = model_loader.get_weights_dir()
    logger.info("=== LTX-2.5 Weight Download & Verification ===")
    logger.info(f"Repo ID     : {model_loader.HF_REPO_ID}")
    logger.info(f"Weights Dir : {weights_dir}")
    logger.info(f"Volume Root : {weights_dir.parent.parent}")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning(
            "HF_TOKEN environment variable is not set. "
            "If downloading a gated model, provide HF_TOKEN=hf_xxx."
        )

    try:
        model_loader.ensure_weights_present()
        logger.success("Weights are downloaded and verified on the volume.")
    except Exception as exc:
        logger.error(f"Failed to prepare model weights: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
