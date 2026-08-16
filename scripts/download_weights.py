#!/usr/bin/env python3
"""
scripts/download_weights.py
─────────────────────────────────────────────────────────────────────────────
Standalone script to pre-populate the network volume with LTX-2.5 weights.

Usage:
  # On a Runpod pod attached to the same network volume:
  python scripts/download_weights.py

  # With an explicit token (if HF_TOKEN env var is not set):
  HF_TOKEN=hf_xxx python scripts/download_weights.py

  # Override the default download directory:
  RUNPOD_VOLUME_PATH=/my/volume python scripts/download_weights.py

This script is also used internally by the handler on the very first cold
start. Running it manually on a Pod (instead of a Serverless worker) lets
you pre-populate the volume cheaply before the Serverless endpoint is live,
avoiding the first-cold-start download penalty entirely.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import sys
import os

# Allow running from repo root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from loguru import logger
import model_loader


def main() -> None:
    logger.info("=== LTX-2.5 Weight Download Script ===")
    logger.info(f"Volume root : {model_loader.VOLUME_ROOT}")
    logger.info(f"Weights dir : {model_loader.WEIGHTS_DIR}")
    logger.info(f"HF repo     : {model_loader.HF_REPO_ID}")

    try:
        model_loader.ensure_weights_present()
        logger.success("✓ Weights are present and ready on the network volume.")
    except EnvironmentError as exc:
        logger.error(f"Volume error: {exc}")
        sys.exit(1)
    except RuntimeError as exc:
        logger.error(f"Download error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
