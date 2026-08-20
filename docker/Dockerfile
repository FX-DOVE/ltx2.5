# syntax=docker/dockerfile:1.7
# ─────────────────────────────────────────────────────────────────────────────
# LTX-2.5 Runpod Serverless – Dockerfile
#
# Base: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
# Official Runpod base image bundling CUDA 12.4.1, PyTorch 2.4.0, Python 3.11.
# ─────────────────────────────────────────────────────────────────────────────

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 AS runtime

LABEL org.opencontainers.image.title="LTX-2.5 Runpod Serverless"
LABEL org.opencontainers.image.description="Lightricks LTX-2.5 video generation serverless endpoint"
LABEL org.opencontainers.image.source="https://github.com/FX-DOVE/ltx2.5"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/

ENV MODEL_ID=Lightricks/LTX-2.5 \
    DTYPE=bfloat16 \
    GPU_MEMORY_UTILIZATION=0.95 \
    RUNPOD_VOLUME_PATH=/runpod-volume \
    HF_HOME=/runpod-volume/hf_cache \
    TRANSFORMERS_CACHE=/runpod-volume/hf_cache \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["python", "-u", "/app/src/handler.py"]
