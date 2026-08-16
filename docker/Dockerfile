# syntax=docker/dockerfile:1.7
# ─────────────────────────────────────────────────────────────────────────────
# LTX-2.5 Runpod Serverless – Dockerfile
#
# Base: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
# Official Runpod base image bundling CUDA 12.4.1, PyTorch 2.4.0, Python 3.11.
# Model weights are loaded dynamically from /runpod-volume on cold start.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 2: lean runtime image ──────────────────────────────────────────────
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 AS runtime

LABEL org.opencontainers.image.title="LTX-2.5 Runpod Serverless"
LABEL org.opencontainers.image.description="Lightricks LTX-2.5 text-to-video / image-to-video serverless endpoint"
LABEL org.opencontainers.image.source="https://github.com/FX-DOVE/ltx2.5"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app
COPY src/ ./src/
COPY scripts/ ./scripts/

ENV HF_HOME=/runpod-volume/.cache/huggingface \
    TRANSFORMERS_CACHE=/runpod-volume/.cache/huggingface/transformers \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["python", "-u", "/app/src/handler.py"]
