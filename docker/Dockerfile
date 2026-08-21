# syntax=docker/dockerfile:1.7
# ─────────────────────────────────────────────────────────────────────────────
# LTX-2.5 Runpod Serverless – Dockerfile
#
# Base: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
#   CUDA 12.8.1 · PyTorch 2.8.0 · Python 3.12 · Ubuntu 24.04
#
# Why not the previous 2.4.0/cu12.4 base: ltx-core requires `torch~=2.7`, so pip
# would have force-replaced torch with a multi-GB PyPI wheel on every build.
# ─────────────────────────────────────────────────────────────────────────────

FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 AS runtime

LABEL org.opencontainers.image.title="LTX-2.5 Runpod Serverless"
LABEL org.opencontainers.image.description="Lightricks LTX-2.5 video generation serverless endpoint"
LABEL org.opencontainers.image.source="https://github.com/FX-DOVE/ltx2.5"

WORKDIR /app

# git: required to install ltx-core / ltx-pipelines from the LTX-2 monorepo.
# ffmpeg: not used for encoding (PyAV bundles its own libs) but kept for
#         debugging / probing generated files inside the container.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Fail the build (not the first cold start) if the LTX packages did not land.
RUN python -c "\
import ltx_core, ltx_pipelines; \
from ltx_pipelines.distilled import DistilledPipeline; \
from ltx_pipelines.utils.model_paths import ModelPaths; \
from ltx_pipelines.utils.quantization_factory import QuantizationKind; \
from ltx_pipelines.utils.media_io import encode_video; \
import torch; \
print('ltx ok | torch', torch.__version__)"

COPY src/ ./src/
COPY scripts/ ./scripts/

ENV MODEL_ID=Lightricks/LTX-2.5 \
    DTYPE=bfloat16 \
    RUNPOD_VOLUME_PATH=/runpod-volume \
    HF_HOME=/runpod-volume/hf_cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Required by the LTX optimisation docs: without expandable segments the 22B
# transformer fragments the CUDA allocator and OOMs mid-decode on a 48 GB L40S.
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# fp8-cast is the only quantization usable on an L40S (sm_89): nvfp4 requires
# Blackwell (sm_100+) plus compiled ltx-kernels. ~19.6 GB vs 39.1 GB for bf16.
ENV LTX_QUANTIZATION=fp8-cast \
    LTX_OFFLOAD_MODE=none \
    LTX_DIFFVAE_MODE=chunked_eager \
    LTX_TRANSFORMER=distilled

CMD ["python", "-u", "/app/src/handler.py"]
