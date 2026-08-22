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
   Builds the Lightricks `DistilledPipeline` from the split ("Comfy-aligned")
   checkpoint layout on the volume — one .safetensors per component, wired
   through ModelPaths.from_split() — and returns it. This is called once per
   worker lifetime and the result is kept warm in a module-level singleton.

Design decisions worth noting:
  • We use snapshot_download with local_dir=WEIGHTS_DIR so the real files live
    on the volume rather than as symlinks into the HF cache. This is critical
    because HF_HOME is also on the volume — if both pointed at the same
    location we'd get double storage usage.
  • Only the files needed for distilled inference are downloaded
    (REQUIRED_FILES); pulling the full repo would exceed the volume.
  • Weight existence is checked by verifying individual sentinel files
    (not just the directory) to guard against partial downloads.
  • torch.cuda.empty_cache() is called before model load to maximise
    available VRAM — this matters if the handler process survived a previous
    request that fragmented the allocator.
  • We load in bfloat16 (the native LTX-2.5 dtype). Quantisation defaults to
    fp8-cast (see LTX_QUANTIZATION below) because the raw bf16 transformer is
    39.1 GB — too close to the 48 GB L40S limit for reliable two-stage decode.
  • Model paths are resolved dynamically at runtime using get_weights_dir(),
    which respects the MODEL_PATH environment variable if set.
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

HF_REPO_ID = "Lightricks/LTX-2.5"

# ─────────────────────────────────────────────────────────────────────────────
# Model file layout matching Lightricks/LTX-2.5 safetensors repository
# ─────────────────────────────────────────────────────────────────────────────
MODEL_FILES = {
    # ── Transformers (DiT) ───────────────────────────────────────────────────
    # Only the bf16 checkpoints are loadable by ltx-pipelines / PyTorch.
    # The *-comfy-int8-convrot files are ComfyUI-only (per the model card) and
    # are deliberately NOT used as fallbacks here.
    "transformer_distilled_bf16": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "transformer_distilled_nvfp4": "diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors",
    "transformer_dev_bf16": "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
    # ComfyUI-only — listed so we can detect them and emit a clear error.
    "transformer_distilled_int8": "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
    "transformer_dev_int8": "diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors",
    # ── Text encoders (Gemma-4 12B + LTX projections) ────────────────────────
    "text_encoder_bf16": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    # ComfyUI-only.
    "text_encoder_int8": "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    # ── VAEs ─────────────────────────────────────────────────────────────────
    "video_vae_bf16": "vae/ltx-2.5-video-vae-bf16.safetensors",           # DiffVAE: higher quality, heavier
    "video_vae_conv_bf16": "vae/ltx-2.5-video-vae-conv-bf16.safetensors",  # Conv VAE: faster, lighter
    "audio_vae_bf16": "vae/ltx-2.5-audio-vae-bf16.safetensors",           # audio VAE + vocoder
    # ── Patches & upscalers ──────────────────────────────────────────────────
    "duration_head_bf16": "model_patches/ltx-2.5-duration-head-bf16.safetensors",
    "spatial_upscaler_bf16": "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    "temporal_upscaler_bf16": "latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors",
    # ── LoRAs ────────────────────────────────────────────────────────────────
    "lora_distilled_bf16": "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
}

# ComfyUI-only checkpoints, keyed so _require_first can explain the failure.
COMFY_ONLY_KEYS = frozenset(
    {"transformer_distilled_int8", "transformer_dev_int8", "text_encoder_int8"}
)

# The minimal set of files needed for distilled text/image-to-video inference.
# Used as `allow_patterns` so a fresh volume does not pull all 185 GB of
# variants (bf16 + int8 + nvfp4 + dev + temporal upscaler + LoRA).
REQUIRED_FILES = [
    MODEL_FILES["transformer_distilled_bf16"],
    MODEL_FILES["text_encoder_bf16"],
    MODEL_FILES["video_vae_bf16"],
    MODEL_FILES["audio_vae_bf16"],
    MODEL_FILES["duration_head_bf16"],
    MODEL_FILES["spatial_upscaler_bf16"],
]

# Sentinel files required for standard LTX-2.5 distilled inference.
SENTINEL_FILES = list(REQUIRED_FILES)

# Module-level pipeline singleton — loaded once per worker process.
_pipeline: Optional[object] = None  # type: ignore[assignment]


# ─────────────────────────────────────────────────────────────────────────────
# Path Resolution
# ─────────────────────────────────────────────────────────────────────────────

def get_weights_dir() -> Path:
    """
    Determine the model storage path at runtime.
    Priority:
    1. MODEL_PATH env var (escape hatch)
    2. RUNPOD_VOLUME_PATH env var (Runpod specific)
    3. Default mounted paths (/runpod-volume, then /workspace)
    """
    if env_path := os.environ.get("MODEL_PATH"):
        path = Path(env_path)
    elif env_vol := os.environ.get("RUNPOD_VOLUME_PATH"):
        path = Path(env_vol) / "models" / "ltx-2.5"
    else:
        # Auto-detect standard Runpod volume locations
        for candidate in [Path("/runpod-volume"), Path("/workspace")]:
            if (candidate / "models" / "ltx-2.5").exists():
                path = candidate / "models" / "ltx-2.5"
                break
        else:
            path = Path("/runpod-volume/models/ltx-2.5")
    
    logger.debug(f"[model_loader] Resolved weights directory to: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def ensure_weights_present() -> None:
    """
    Idempotent weight-availability check.  Safe to call on every cold start.

    Behaviour:
      • If all sentinel files exist on the volume → log and return immediately.
      • Otherwise → download the full snapshot to the resolved weights directory.

    Raises:
      RuntimeError   if HF_TOKEN is missing or download fails.
      EnvironmentError if the network volume is not mounted / writable.
    """
    weights_dir = get_weights_dir()
    _assert_volume_writable(weights_dir.parent)

    t0 = time.monotonic()
    missing = _missing_sentinel_files(weights_dir)

    if not missing:
        elapsed = time.monotonic() - t0
        logger.info(
            f"[model_loader] Weights already present at {weights_dir} "
            f"(checked in {elapsed:.2f}s) — skipping download."
        )
        return

    logger.info(
        f"[model_loader] Missing sentinel files: {missing}. "
        f"Starting download from '{HF_REPO_ID}' → {weights_dir}"
    )
    _download_weights(weights_dir)
    elapsed = time.monotonic() - t0
    logger.info(f"[model_loader] Download complete in {elapsed:.1f}s.")


def load_pipeline() -> object:
    """
    Load LTX-2.5 into GPU memory and return the pipeline singleton.

    Subsequent calls return the cached singleton without reloading.

    Configuration (all optional, read from the environment):
      LTX_TRANSFORMER   distilled | distilled-nvfp4 | dev   (default: distilled)
      LTX_QUANTIZATION  none | fp8-cast | fp8-scaled-mm |
                        nvfp4-cast | nvfp4-prequant          (default: fp8-cast)
      LTX_OFFLOAD_MODE  auto | none | cpu | disk               (default: auto)
                        auto = none on a >= 44 GiB card, else cpu/disk.
      LTX_DIFFVAE_MODE  chunked_eager | chunked_compile |
                        combined_compile | blackwell_dsl      (default: chunked_eager)
      LTX_VIDEO_VAE     auto | diffusion | conv               (default: auto)
      LTX_WEIGHT_CACHE  auto | on | off                       (default: auto)
                        Keep checkpoint weights in host RAM between the ~8
                        model builds each request performs, instead of re-reading
                        them from the network volume. auto = on when the host has
                        the RAM for it.
      LTX_ALLOC_TRIM    auto | trim | defer                   (default: auto)
                        defer skips synchronize + empty_cache on every block
                        exit. auto = defer when weights are resident, else trim.
    """
    global _pipeline
    weights_dir = get_weights_dir()

    if _pipeline is not None:
        logger.debug("[model_loader] Pipeline already loaded - reusing singleton.")
        return _pipeline

    t0 = time.monotonic()
    logger.info(f"[model_loader] Loading LTX-2.5 pipeline from {weights_dir} ...")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
    from ltx_core.model.video_vae.transformer import DiffVAEMode
    from ltx_pipelines.distilled import DistilledPipeline
    from ltx_pipelines.utils.model_paths import ModelPaths
    from ltx_pipelines.utils.quantization_factory import QuantizationKind
    from ltx_pipelines.utils.types import OffloadMode

    try:
        # ── Resolve checkpoint files on the volume ───────────────────────────
        transformer_path = _resolve_transformer(weights_dir)
        text_encoder_path = _require_first(
            weights_dir,
            ["text_encoder_bf16"],
            "text encoder",
        )
        video_vae_path = _resolve_video_vae(weights_dir)
        # spatial_upsampler_path is a REQUIRED str on DistilledPipeline — the
        # two-stage pipeline cannot run without it, so fail loudly here rather
        # than passing None and getting an opaque TypeError deep in the stack.
        spatial_upscaler_path = _require_first(
            weights_dir, ["spatial_upscaler_bf16"], "latent spatial upsampler"
        )

        # Optional slots: absent → pipeline degrades gracefully (no audio track /
        # no automatic duration prediction).
        audio_vae_path = weights_dir / MODEL_FILES["audio_vae_bf16"]
        duration_head_path = weights_dir / MODEL_FILES["duration_head_bf16"]
        if not audio_vae_path.exists():
            logger.warning(
                "[model_loader] Audio VAE not found — output will be video-only."
            )
        if not duration_head_path.exists():
            logger.warning(
                "[model_loader] Duration head not found — num_frames must be explicit."
            )

        model_paths = ModelPaths.from_split(
            transformer_path=str(transformer_path),
            text_encoder_path=str(text_encoder_path),
            video_vae_path=str(video_vae_path),
            audio_vae_path=str(audio_vae_path) if audio_vae_path.exists() else None,
            duration_head_path=(
                str(duration_head_path) if duration_head_path.exists() else None
            ),
        )

        # ── Resolve runtime knobs ────────────────────────────────────────────
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        quantization_raw = os.environ.get("LTX_QUANTIZATION", "fp8-cast")
        quantization = _resolve_quantization(QuantizationKind, transformer_path)
        offload_mode = _resolve_offload_mode(OffloadMode, quantization_raw)
        diffvae_mode = _resolve_enum(
            DiffVAEMode,
            os.environ.get("LTX_DIFFVAE_MODE", "chunked_eager"),
            "LTX_DIFFVAE_MODE",
        )

        logger.info(
            "[model_loader] transformer={} | text_encoder={} | video_vae={}\n"
            "[model_loader] quantization={} | offload={} | diffvae={} | device={}"
            " | vram={:.1f} GiB".format(
                transformer_path.name,
                text_encoder_path.name,
                video_vae_path.name,
                quantization_raw,
                offload_mode.value,
                diffvae_mode.value,
                device,
                _total_vram_gib(),
            )
        )

        registry = _build_weight_cache(offload_mode)
        alloc_trim = _resolve_alloc_trim(AllocatorTrimStrategy, offload_mode)

        pipeline = DistilledPipeline(
            model_paths=model_paths,
            spatial_upsampler_path=str(spatial_upscaler_path),
            loras=[],
            device=device,
            quantization=quantization,
            registry=registry,
            compilation_config=None,
            offload_mode=offload_mode,
            alloc_trim_strategy=alloc_trim,
            diffvae_optimization=diffvae_mode,
        )

        # Per-phase timers, and the lazy VAE-encoder build that removes two
        # unused encoder builds from every text-to-video request.
        from perf import instrument_pipeline

        instrument_pipeline(pipeline)

        _pipeline = pipeline
        elapsed = time.monotonic() - t0
        logger.info(f"[model_loader] DistilledPipeline ready in {elapsed:.1f}s.")
        return _pipeline

    except torch.cuda.OutOfMemoryError as oom:
        logger.error(f"[model_loader] CUDA OOM during model load: {oom}")
        raise RuntimeError("out_of_memory_during_load") from oom
    except Exception as exc:
        logger.exception(f"[model_loader] Failed to load pipeline: {exc}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint / knob resolution
# ─────────────────────────────────────────────────────────────────────────────


def _require_first(weights_dir: Path, keys: list[str], label: str) -> Path:
    """
    Return the first existing file among `keys`, else raise a clear error.

    If a ComfyUI-only variant of the same component is present on the volume,
    say so explicitly — that is the most likely reason a user hits this.
    """
    for key in keys:
        candidate = weights_dir / MODEL_FILES[key]
        if candidate.exists():
            return candidate

    tried = ", ".join(MODEL_FILES[k] for k in keys)
    hint = ""
    present_comfy = [
        MODEL_FILES[k]
        for k in COMFY_ONLY_KEYS
        if (weights_dir / MODEL_FILES[k]).exists()
    ]
    if present_comfy:
        hint = (
            "\nThese ComfyUI-only checkpoints ARE on the volume but cannot be "
            "loaded by ltx-pipelines (the *-comfy-int8-convrot files are for "
            f"ComfyUI only): {', '.join(present_comfy)}"
        )
    raise RuntimeError(
        f"No {label} checkpoint found under {weights_dir}. Tried: {tried}{hint}"
    )


def _resolve_transformer(weights_dir: Path) -> Path:
    """
    Pick the transformer checkpoint according to LTX_TRANSFORMER.

    'distilled' (default) is the two-stage distilled 22B model — the one
    DistilledPipeline expects (fixed 8-step schedule, CFG=1). 'dev' is the full
    trainable DiT, provided as an escape hatch.

    Only bf16 (and the nvfp4 pre-quantised) checkpoints are candidates: per the
    model card the *-comfy-int8-convrot files are ComfyUI-only.
    """
    variant = os.environ.get("LTX_TRANSFORMER", "distilled").strip().lower()
    variants: dict[str, list[str]] = {
        "distilled": ["transformer_distilled_bf16"],
        "distilled-nvfp4": ["transformer_distilled_nvfp4"],
        "dev": ["transformer_dev_bf16"],
    }
    if variant not in variants:
        raise RuntimeError(
            f"LTX_TRANSFORMER='{variant}' is not recognised. "
            f"Valid values: {', '.join(sorted(variants))}"
        )
    return _require_first(weights_dir, variants[variant], f"{variant} transformer")


def _resolve_video_vae(weights_dir: Path) -> Path:
    """
    Pick the video VAE.

    Two decoders ship with LTX-2.5 and the decoder kind is read from the file's
    own metadata:
      • ...video-vae-bf16.safetensors      → diffusion decoder (higher quality,
        wants `natten` for full speed; falls back to Triton/eager without it)
      • ...video-vae-conv-bf16.safetensors → convolutional decoder (much faster,
        no natten needed)

    LTX_VIDEO_VAE=auto (default) prefers the conv decoder when it is on the
    volume, because `natten` cannot be installed alongside torch 2.8 (it pins
    torch==2.13.0), which makes the diffusion decoder the slow path here.
    """
    choice = os.environ.get("LTX_VIDEO_VAE", "auto").strip().lower()
    conv = weights_dir / MODEL_FILES["video_vae_conv_bf16"]
    diffusion = weights_dir / MODEL_FILES["video_vae_bf16"]

    if choice == "conv":
        if not conv.exists():
            raise RuntimeError(
                f"LTX_VIDEO_VAE=conv but {conv} is not on the volume. "
                "Download it or use LTX_VIDEO_VAE=diffusion."
            )
        return conv
    if choice == "diffusion":
        return _require_first(weights_dir, ["video_vae_bf16"], "video VAE")
    if choice != "auto":
        raise RuntimeError(
            f"LTX_VIDEO_VAE='{choice}' is not recognised. "
            "Valid values: auto, diffusion, conv"
        )

    if conv.exists():
        logger.info("[model_loader] Using the convolutional video VAE (faster decode).")
        return conv
    if diffusion.exists():
        logger.info(
            "[model_loader] Using the diffusion video VAE. `natten` is not "
            "installable on torch 2.8, so decode runs on the Triton/eager "
            "fallback — expect slower decoding."
        )
        return diffusion
    raise RuntimeError(f"No video VAE checkpoint found under {weights_dir}.")


def _resolve_quantization(quantization_kind_cls: type, transformer_path: Path):
    """
    Build the QuantizationPolicy from LTX_QUANTIZATION.

    Defaults to fp8-cast: the bf16 22B transformer is 39.1 GB, which leaves
    almost no headroom on a 48 GB L40S once the text encoder, VAEs and
    activations are resident. fp8-cast brings the transformer to ~19.6 GB and
    needs no extra dependencies on any FP8-capable GPU (sm_89+).

    nvfp4 is rejected on non-Blackwell hardware with an actionable message
    instead of failing later inside the kernel dispatch.
    """
    raw = os.environ.get("LTX_QUANTIZATION", "fp8-cast").strip().lower()
    if raw in ("", "none", "off", "bf16"):
        logger.warning(
            "[model_loader] Quantization disabled. The bf16 transformer needs "
            "~39 GB VRAM — expect OOM on a 48 GB GPU at anything above 450p."
        )
        return None

    try:
        kind = quantization_kind_cls(raw)
    except ValueError as exc:
        valid = ", ".join(k.value for k in quantization_kind_cls)
        raise RuntimeError(
            f"LTX_QUANTIZATION='{raw}' is not recognised. Valid values: none, {valid}"
        ) from exc

    if kind.value.startswith("nvfp4"):
        capability = (
            torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
        )
        if capability[0] < 10:
            raise RuntimeError(
                f"LTX_QUANTIZATION='{raw}' requires a Blackwell GPU (compute "
                f"capability >= 10.0) plus compiled ltx-kernels. This GPU reports "
                f"sm_{capability[0]}{capability[1]}. Use 'fp8-cast' instead."
            )

    # nvfp4-prequant reads its scales from the checkpoint, so the pre-quantised
    # file must be the one that was loaded.
    if kind.value == "nvfp4-prequant" and "nvfp4" not in transformer_path.name:
        raise RuntimeError(
            "LTX_QUANTIZATION='nvfp4-prequant' requires the pre-quantised "
            "checkpoint. Set LTX_TRANSFORMER=distilled-nvfp4."
        )
    if "nvfp4" in transformer_path.name and kind.value != "nvfp4-prequant":
        raise RuntimeError(
            f"Transformer '{transformer_path.name}' is pre-quantised to nvfp4 but "
            f"LTX_QUANTIZATION='{raw}'. Set LTX_QUANTIZATION=nvfp4-prequant."
        )

    return kind.to_policy(str(transformer_path))


def _resolve_enum(enum_cls: type, raw: str, env_name: str):
    """Map an env-var string onto an enum by value, with a clear error."""
    value = (raw or "").strip().lower()
    try:
        return enum_cls(value)
    except ValueError as exc:
        valid = ", ".join(str(m.value) for m in enum_cls)
        raise RuntimeError(
            f"{env_name}='{raw}' is not recognised. Valid values: {valid}"
        ) from exc


# Below this much total VRAM, `LTX_OFFLOAD_MODE=auto` streams weights instead of
# keeping them resident. A 48 GB L40S reports ~44.39 GiB, so the threshold sits
# just under it: with autograd off (see inference.py) the resident peak is the
# bf16 text encoder at ~25 GiB, which such a card holds comfortably.
_OFFLOAD_NONE_MIN_GIB = 44.0

# OffloadMode.CPU pins the full weight set in host RAM (upstream budgets ~36 GB
# for LTX-2). With less than this, DISK re-reads from the volume instead.
_OFFLOAD_CPU_MIN_HOST_GIB = 48.0

# `StreamingModelBuilder` rejects any quantization whose fuse rule is neither
# bf16 nor fp8-cast (ltx_pipelines.utils.blocks._build_streaming_builder), so
# offloading is incompatible with these. Caught here instead of deep in a build.
_NON_STREAMABLE_QUANTIZATIONS = ("fp8-scaled-mm", "nvfp4")


def _total_vram_gib() -> float:
    """Total VRAM on device 0 in GiB, or 0.0 when there is no CUDA device."""
    if not torch.cuda.is_available():
        return 0.0
    try:
        return torch.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception:
        return 0.0


def _total_host_ram_gib() -> float:
    """Total host RAM in GiB, or 0.0 when it cannot be determined."""
    try:
        import psutil

        return psutil.virtual_memory().total / 1024**3
    except Exception:
        return _meminfo_gib("MemTotal")


def _available_host_ram_gib() -> float:
    """Host RAM available right now, in GiB, or 0.0 when unknown."""
    try:
        import psutil

        return psutil.virtual_memory().available / 1024**3
    except Exception:
        return _meminfo_gib("MemAvailable")


def _meminfo_gib(field: str) -> float:
    """
    Fallback probe for a `/proc/meminfo` field, in GiB.

    psutil is declared in requirements.txt, but the RAM probes gate real
    decisions (CPU offload, weight cache), and returning 0.0 makes both take
    the pessimistic branch silently. The container is Linux, so /proc/meminfo
    is a reliable second source.
    """
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(f"{field}:"):
                    return int(line.split()[1]) / 1024**2  # kB → GiB
    except Exception:
        pass
    return 0.0


# ── Host-RAM weight cache ────────────────────────────────────────────────────
# Host RAM left untouched for the process, PyAV, and page cache.
_WEIGHT_CACHE_RESERVE_GIB = 24.0
# The working set that actually matters is Gemma-4-12B bf16 (~24.5 GiB) plus the
# fp8-cast transformer (~19.6 GiB); below this there is no point admitting the
# two dominant checkpoints and thrashing on the rest.
_WEIGHT_CACHE_MIN_BUDGET_GIB = 56.0
# Hard floor applied even to `LTX_WEIGHT_CACHE=on`. Roughly the fp8-cast
# transformer alone; under it the cache cannot hold a single useful checkpoint.
_WEIGHT_CACHE_FLOOR_GIB = 20.0


def _build_weight_cache(offload_mode: object):
    """
    Build the host-RAM weight cache, or return None to keep upstream's default.

    Every block in `ltx_pipelines.utils.blocks` defaults to
    `ModelRegistry(cache_models=True, cache_weights=False)`, so weights are
    re-read from the checkpoint on each of the ~8 builds a request performs —
    from a *network* volume, on RunPod. Retaining them in host RAM turns those
    reads into PCIe copies.

    `auto` only enables it with `offload=none`: the streaming builders used for
    CPU/DISK offload already hold or stream weights from the host, so a second
    host-side copy would compete with them for the same RAM.
    """
    raw = os.environ.get("LTX_WEIGHT_CACHE", "auto").strip().lower()
    if raw not in ("auto", "on", "off", "1", "0", "true", "false"):
        raise RuntimeError(
            f"LTX_WEIGHT_CACHE={raw!r} is not recognised. Use auto, on or off."
        )
    if raw in ("off", "0", "false"):
        logger.info("[model_loader] LTX_WEIGHT_CACHE=off — weights re-read per build.")
        return None

    resident = getattr(offload_mode, "value", str(offload_mode)) == "none"
    available = _available_host_ram_gib()
    budget = max(0.0, available - _WEIGHT_CACHE_RESERVE_GIB)

    if raw == "auto":
        if not resident:
            logger.info(
                "[model_loader] LTX_WEIGHT_CACHE=auto: offload is streaming from the "
                "host already — leaving the weight cache off to avoid a second copy."
            )
            return None
        if budget < _WEIGHT_CACHE_MIN_BUDGET_GIB:
            logger.info(
                f"[model_loader] LTX_WEIGHT_CACHE=auto: {available:.0f} GiB host RAM "
                f"available leaves a {budget:.0f} GiB budget, under the "
                f"{_WEIGHT_CACHE_MIN_BUDGET_GIB:.0f} GiB the text encoder plus "
                "transformer need — weight cache off."
            )
            return None
    elif not resident:
        logger.warning(
            "[model_loader] LTX_WEIGHT_CACHE=on with host offload active — the "
            "streaming builder already keeps weights host-side; watch RAM."
        )

    # `on` overrides the 56 GiB heuristic, but not physics: a cache that cannot
    # hold anything is worse than no cache, because it pays the device→host copy
    # and then refuses the entry.
    if budget < _WEIGHT_CACHE_FLOOR_GIB:
        logger.warning(
            f"[model_loader] weight cache off: {available:.0f} GiB host RAM available "
            f"leaves only {budget:.0f} GiB after the {_WEIGHT_CACHE_RESERVE_GIB:.0f} GiB "
            "reserve — not enough to retain a checkpoint."
        )
        return None

    from weight_cache import HostWeightCacheRegistry

    pin = os.environ.get("LTX_WEIGHT_CACHE_PIN", "0").strip().lower() in ("1", "true", "on")
    logger.info(
        f"[model_loader] weight cache on: {budget:.0f} GiB budget of "
        f"{available:.0f} GiB available host RAM (pinned={pin})."
    )
    return HostWeightCacheRegistry(budget_bytes=int(budget * 1024**3), pin=pin)


def _resolve_alloc_trim(strategy_cls: type, offload_mode: object):
    """
    Resolve LTX_ALLOC_TRIM.

    `AllocatorTrimStrategy.TRIM` — upstream's default — runs
    `synchronize_device()`, `dispose()` and `cleanup_memory()` on every
    `gpu_model()` exit, so a request that builds eight models pays eight full
    `empty_cache()` round trips and hands the freed blocks back to the driver
    only to ask for them again seconds later. `DEFER` keeps `dispose()` and
    skips the sync and the cache flush.

    `auto` defers only when weights are resident on a card with room to spare:
    with a smaller card the allocator wants the OS-level release, and
    `expandable_segments:True` is what makes keeping the pool safe here.
    """
    raw = os.environ.get("LTX_ALLOC_TRIM", "auto").strip().lower()
    if raw == "trim":
        return strategy_cls("trim")
    if raw == "defer":
        return strategy_cls("defer")
    if raw != "auto":
        raise RuntimeError(
            f"LTX_ALLOC_TRIM={raw!r} is not recognised. Use auto, trim or defer."
        )

    resident = getattr(offload_mode, "value", str(offload_mode)) == "none"
    vram = _total_vram_gib()
    if resident and vram >= _OFFLOAD_NONE_MIN_GIB:
        logger.info(
            f"[model_loader] LTX_ALLOC_TRIM=auto: {vram:.1f} GiB VRAM with weights "
            "resident — deferring allocator trims (keeps the CUDA pool warm across "
            "builds)."
        )
        return strategy_cls("defer")
    logger.info(
        "[model_loader] LTX_ALLOC_TRIM=auto: trimming the allocator on every block "
        "exit (streaming offload or a card without spare VRAM)."
    )
    return strategy_cls("trim")


def _resolve_offload_mode(offload_mode_cls: type, quantization_raw: str):
    """
    Resolve LTX_OFFLOAD_MODE, including the `auto` default.

    `auto` keeps every weight resident (OffloadMode.NONE — fastest) on cards with
    at least ~44 GiB of VRAM, and streams layer-by-layer from pinned host RAM
    (OffloadMode.CPU) below that, falling back to DISK when the host does not
    have the ~36-48 GB of RAM that CPU offload pins. Explicit values are always
    honoured; `auto` exists so a smaller card degrades instead of OOMing.

    Streaming rejects quantizations that are neither bf16 nor fp8-cast, so an
    incompatible pairing is rejected here with an actionable message.
    """
    raw = os.environ.get("LTX_OFFLOAD_MODE", "auto").strip().lower()

    if raw in ("", "auto"):
        vram = _total_vram_gib()
        if vram == 0.0:
            mode = offload_mode_cls("none")
            logger.info(
                "[model_loader] LTX_OFFLOAD_MODE=auto: no CUDA device detected — "
                "using offload=none (CPU inference)."
            )
        elif vram >= _OFFLOAD_NONE_MIN_GIB:
            mode = offload_mode_cls("none")
            logger.info(
                f"[model_loader] LTX_OFFLOAD_MODE=auto: {vram:.1f} GiB VRAM "
                f">= {_OFFLOAD_NONE_MIN_GIB:.0f} GiB — keeping weights resident "
                "(offload=none, fastest)."
            )
        else:
            host = _total_host_ram_gib()
            if 0.0 < host < _OFFLOAD_CPU_MIN_HOST_GIB:
                mode = offload_mode_cls("disk")
                logger.warning(
                    f"[model_loader] LTX_OFFLOAD_MODE=auto: {vram:.1f} GiB VRAM and "
                    f"only {host:.1f} GiB host RAM — streaming from disk "
                    "(offload=disk). Expect slow generation; every pass re-reads "
                    "the checkpoint."
                )
            else:
                mode = offload_mode_cls("cpu")
                logger.warning(
                    f"[model_loader] LTX_OFFLOAD_MODE=auto: {vram:.1f} GiB VRAM "
                    f"< {_OFFLOAD_NONE_MIN_GIB:.0f} GiB — streaming weights from "
                    "host RAM (offload=cpu). Slower than resident weights but it "
                    "fits; a >= 48 GB GPU avoids this."
                )
    else:
        mode = _resolve_enum(offload_mode_cls, raw, "LTX_OFFLOAD_MODE")

    if mode.value != "none":
        quant = (quantization_raw or "").strip().lower()
        for blocked in _NON_STREAMABLE_QUANTIZATIONS:
            if blocked in quant:
                raise RuntimeError(
                    f"LTX_OFFLOAD_MODE={mode.value!r} streams weights layer by "
                    f"layer, which only supports bf16 and fp8-cast fuse rules, "
                    f"but LTX_QUANTIZATION={quantization_raw!r}. Use "
                    "LTX_QUANTIZATION=fp8-cast, or LTX_OFFLOAD_MODE=none."
                )

    return mode


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _assert_volume_writable(parent_dir: Path) -> None:
    """Confirm the base directory is accessible and writable."""
    if not parent_dir.exists():
        try:
            parent_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise EnvironmentError(f"Cannot create base directory {parent_dir}: {exc}")
            
    test_file = parent_dir / ".write_test"
    try:
        test_file.touch()
        test_file.unlink()
    except OSError as exc:
        raise EnvironmentError(f"Directory '{parent_dir}' is not writable: {exc}") from exc


def _missing_sentinel_files(weights_dir: Path) -> list[str]:
    """Return the list of expected sentinel files that are absent or zero-size."""
    missing = []
    for rel_path in SENTINEL_FILES:
        full = weights_dir / rel_path
        if not full.exists() or full.stat().st_size == 0:
            missing.append(rel_path)
    return missing


def _download_weights(weights_dir: Path) -> None:
    """
    Download the LTX-2.5 model files needed for distilled inference.

    Only REQUIRED_FILES are fetched (via `allow_patterns`). Grabbing the whole
    repo would pull ~185 GB — every bf16 / ComfyUI-int8 / nvfp4 variant plus the
    dev transformer, the temporal upscaler and the LoRA — which does not fit
    alongside itself on a 200 GB volume and is not needed by this handler.

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

    weights_dir.mkdir(parents=True, exist_ok=True)

    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0") == "1":
        logger.info("[model_loader] hf_transfer enabled for accelerated download.")

    # Optionally also fetch the faster convolutional VAE.
    allow_patterns = list(REQUIRED_FILES)
    if os.environ.get("LTX_DOWNLOAD_CONV_VAE", "0") == "1":
        allow_patterns.append(MODEL_FILES["video_vae_conv_bf16"])

    logger.info(
        f"[model_loader] Downloading {len(allow_patterns)} file(s) from "
        f"'{HF_REPO_ID}' -> '{weights_dir}' ...\n"
        "This only happens once on a fresh volume."
    )

    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="model",
        local_dir=str(weights_dir),
        token=hf_token,
        allow_patterns=allow_patterns,
    )

    # Final sanity check: confirm sentinels are now present.
    still_missing = _missing_sentinel_files(weights_dir)
    if still_missing:
        raise RuntimeError(
            f"Download appeared to complete but sentinel files are still missing: "
            f"{still_missing}. Check disk space on the network volume."
        )
