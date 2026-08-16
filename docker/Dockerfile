# syntax=docker/dockerfile:1.7
# ─────────────────────────────────────────────────────────────────────────────
# LTX-2.5 Runpod Serverless – Dockerfile
#
# Design decisions (comment for future maintainers):
#  • Base: runpod/pytorch:1.1.0-rc.154-cu1281-torch280-ubuntu2204
#    Runpod changed their image naming scheme in 2025. The new format is:
#      runpod/pytorch:<sdk-version>-cu<cuda>-torch<pytorch>-ubuntu<os>
#    Selected tag: CUDA 12.8.1 + PyTorch 2.8.0 + Ubuntu 22.04
#    Python 3.11 is the default in this image.
#    To find available tags: https://hub.docker.com/r/runpod/pytorch/tags
#    Fallback candidate: 1.1.0-rc.154-cu1290-torch291-ubuntu2204
#  • Model weights are NOT baked in. They live on the network volume at
#    /runpod-volume/models/ and are downloaded once on the very first
#    cold start. See src/model_loader.py for the logic.
#  • HF_TOKEN is injected at RUNTIME via a Runpod secret environment
#    variable — never as a build ARG committed to the image history.
#  • We install hf_transfer for faster HF downloads (multipart).
#  • ffmpeg is installed system-wide because imageio-ffmpeg's bundled
#    binary occasionally has codec issues inside containers.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder (keeps final layer cache efficient) ───────────
FROM runpod/pytorch:1.1.0-rc.154-cu1281-torch280-ubuntu2204 AS builder

WORKDIR /build

# System packages: ffmpeg for video encoding, git for any pip git+ deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Copy only requirements first so Docker layer-caches this step independently
# of source-code changes — avoids a full pip reinstall on every code push.
COPY requirements.txt .

# Upgrade pip, then install deps.
# NOTE: torch is already in the base image; pip will skip it (version pinned
#       by the base image is correct for CUDA 12.8).
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 2: lean runtime image ──────────────────────────────────────────────
FROM runpod/pytorch:1.1.0-rc.154-cu1281-torch280-ubuntu2204 AS runtime

LABEL org.opencontainers.image.title="LTX-2.5 Runpod Serverless"
LABEL org.opencontainers.image.description="Lightricks LTX-2.5 text-to-video / image-to-video serverless endpoint"
LABEL org.opencontainers.image.source="https://github.com/FX-DOVE/ltx2.5"

# ── System packages (runtime only) ───────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# ── Copy installed Python packages from builder ───────────────────────────────
# This avoids running pip again in the final stage.
# Copy Python site-packages — the new Runpod images use python3.11 but the
# exact path may include a dist-packages or site-packages dir depending on
# the base OS. We copy both to be safe; one will simply be empty.
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# ── Application source ────────────────────────────────────────────────────────
WORKDIR /app
COPY src/ ./src/
COPY scripts/ ./scripts/

# ── Runtime environment variables ─────────────────────────────────────────────
# HF_TOKEN:  injected by Runpod secret — never hardcode here.
# HF_HOME:   point HF cache to the network volume so the HF metadata cache
#            (not weights) also survives restarts.
# TRANSFORMERS_CACHE: kept in sync with HF_HOME.
# HF_HUB_ENABLE_HF_TRANSFER: enable multi-part download acceleration.
ENV HF_HOME=/runpod-volume/.cache/huggingface \
    TRANSFORMERS_CACHE=/runpod-volume/.cache/huggingface/transformers \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Runpod serverless: the handler registers itself via runpod.serverless.start().
CMD ["python", "-u", "/app/src/handler.py"]
