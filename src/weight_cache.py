"""
src/weight_cache.py
─────────────────────────────────────────────────────────────────────────────
A `Registry` that keeps checkpoint weights in host RAM between builds.

The problem
───────────
`ltx_pipelines.utils.blocks` builds a model on every `__call__` and frees it
again, and every builder in it defaults to:

    ModelRegistry(cache_models=True, cache_weights=False)

Model *shells* are cached; weights are re-read from the safetensors file each
time. `DistilledPipeline.__call__` performs at least eight such builds per
request — Gemma-4-12B once, the 22B transformer twice (one per stage), the VAE
encoder up to three times, plus the spatial upsampler, video decoder, audio
decoder and vocoder. On RunPod the checkpoints live on a *network* volume, so
that is tens of gigabytes pulled over network storage for every single video.

Why not just flip `cache_weights=True`
──────────────────────────────────────
`ModelRegistry.add` stores the state dict exactly as handed to it, and
`helpers.load_state_dict` loads with `device=cuda`. So `cache_weights=True`
would retain the weights *in VRAM*: ~19.6 GB of fp8 transformer plus ~24.5 GB
of Gemma is already more than a 48 GB L40S reports (44.39 GiB), and because
`load_state_dict(..., assign=True)` makes the model's parameters alias those
cached tensors, `dispose()` cannot give the memory back. The result is an OOM,
which is exactly the failure this deployment already fixed once.

What this does instead
──────────────────────
`add()` rewrites the retained copy to host RAM and hands that back — the case
upstream documents in `helpers.load_state_dict`:

    # ``add`` returns the retained copy when the registry rewrites storage
    # (e.g. pin to CPU).

and confirms in `SingleGPUModelBuilder.keeps_gpu_resident_weights`:

    # Registry may retain a CPU SD; each build H2Ds into fresh GPU storages.

So each later build assigns CPU tensors onto the meta shell and `build()`'s
closing `meta_model.to(device)` copies them to fresh GPU storages. Repeat
builds become a PCIe transfer instead of a network-volume read, and `dispose()`
still frees all of the VRAM because nothing in the cache points at it.

The first build of each checkpoint costs one extra device→host copy, since the
loader has already put the tensors on the GPU by the time `add()` sees them.
That is paid once per worker, not once per request.

Budget
──────
Retention is capped. Once the cap is reached, new checkpoints are simply not
admitted — the pipeline still works, it just reloads those from the volume, so
running out of budget degrades speed rather than correctness. Nothing is
evicted, because the working set is the same handful of files on every request;
evicting would only guarantee a miss next time.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import dataclasses
import hashlib
import threading
from pathlib import Path
from typing import Any

import torch
from loguru import logger

_GIB = 1024**3


def _name_of(paths: list[str]) -> str:
    """Filename for logging, tolerant of an empty path list."""
    return Path(paths[0]).name if paths else "?"


def _sd_bytes(sd: dict) -> int:
    """Footprint of a flat tensor dict, ignoring anything that is not a tensor."""
    return sum(
        value.numel() * value.element_size()
        for value in sd.values()
        if isinstance(value, torch.Tensor)
    )


def _to_host(sd: dict, pin: bool) -> dict:
    """
    Copy every tensor in a state dict to host RAM.

    `pin=True` makes later host→device copies faster and lets them overlap, but
    pinned pages cannot be swapped, so pinning tens of gigabytes is its own
    risk. Off by default; `pin_memory()` failures degrade to ordinary pageable
    memory rather than failing the build.
    """
    host: dict = {}
    for key, value in sd.items():
        if not isinstance(value, torch.Tensor):
            host[key] = value
            continue
        tensor = value.detach().to("cpu", copy=not value.is_cpu)
        if pin and not tensor.is_pinned():
            try:
                tensor = tensor.pin_memory()
            except Exception:
                pass
        host[key] = tensor
    return host


class HostWeightCacheRegistry:
    """
    `Registry` that retains state dicts in host RAM and model shells in place.

    Satisfies `ltx_core.loader.registry.Registry` structurally — it is a
    Protocol, so no import from the container-only package is needed here and
    this module stays importable on a machine without LTX installed.
    """

    def __init__(self, *, budget_bytes: int, pin: bool = False) -> None:
        self._budget = max(0, int(budget_bytes))
        self._pin = pin
        self._lock = threading.Lock()
        self._state_dicts: dict[str, Any] = {}
        self._models: dict[str, Any] = {}
        self._retained = 0
        self._skipped: set[str] = set()
        self.hits = 0
        self.misses = 0

    # -- identity -----------------------------------------------------------
    @staticmethod
    def _key(paths: list[str], sd_ops: Any) -> str:
        digest = hashlib.sha256()
        parts = [str(Path(p).resolve()) for p in paths]
        if sd_ops is not None:
            parts.append(getattr(sd_ops, "name", str(sd_ops)))
        digest.update("\0".join(parts).encode("utf-8"))
        return digest.hexdigest()

    # -- state dicts --------------------------------------------------------
    def get(self, paths: list[str], sd_ops: Any) -> Any:
        with self._lock:
            cached = self._state_dicts.get(self._key(paths, sd_ops))
        if cached is None:
            self.misses += 1
        else:
            self.hits += 1
            logger.debug(f"[weight_cache] hit {_name_of(paths)}")
        return cached

    def add(self, paths: list[str], sd_ops: Any, state_dict: Any) -> Any:
        """
        Retain a host-RAM copy and return it, so the caller uses the copy.

        Returning the retained object is what makes the cache sound: the first
        build then assigns the same CPU tensors every later build will, and
        `build()`'s trailing `.to(device)` is the only thing holding VRAM.
        """
        key = self._key(paths, sd_ops)
        name = _name_of(paths)
        declared = getattr(state_dict, "size", None)
        size = int(declared) if isinstance(declared, int) and declared > 0 else _sd_bytes(
            getattr(state_dict, "sd", {}) or {}
        )

        with self._lock:
            if key in self._state_dicts:
                return self._state_dicts[key]
            # A non-positive budget refuses everything rather than meaning
            # "unlimited": the budget is derived from free host RAM, and an
            # exhausted host must degrade to upstream's re-read-per-build, not
            # to unbounded retention.
            if self._retained + size > self._budget:
                if key not in self._skipped:
                    self._skipped.add(key)
                    logger.warning(
                        f"[weight_cache] not caching {name} ({size / _GIB:.1f} GiB): "
                        f"budget {self._budget / _GIB:.1f} GiB, "
                        f"{self._retained / _GIB:.1f} GiB already retained. "
                        "It will be re-read from the volume on every build."
                    )
                return state_dict

        host_sd = _to_host(getattr(state_dict, "sd", {}) or {}, self._pin)
        try:
            retained = _rehost(state_dict, host_sd)
        except TypeError as exc:
            # Never fail a build over a caching decision.
            logger.warning(f"[weight_cache] not caching {name}: {exc}")
            return state_dict

        with self._lock:
            # Another thread may have won the race while the copy was running.
            if key in self._state_dicts:
                return self._state_dicts[key]
            self._state_dicts[key] = retained
            self._retained += size
            total = self._retained
        logger.info(
            f"[weight_cache] retained {name} {size / _GIB:.2f} GiB in host RAM "
            f"({total / _GIB:.2f} GiB of {self._budget / _GIB:.1f} GiB budget)"
        )
        return retained

    def pop(self, paths: list[str], sd_ops: Any) -> Any:
        with self._lock:
            return self._state_dicts.pop(self._key(paths, sd_ops), None)

    # -- model shells -------------------------------------------------------
    def get_model(self, key: str) -> Any:
        with self._lock:
            return self._models.get(key)

    def add_model(self, key: str, model: Any) -> Any:
        with self._lock:
            if key in self._models:
                return self._models[key]
            self._models[key] = model
        return model

    def pop_model(self, key: str) -> Any:
        with self._lock:
            return self._models.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._state_dicts.clear()
            self._models.clear()
            self._retained = 0
            self._skipped.clear()

    # -- diagnostics --------------------------------------------------------
    @property
    def retained_bytes(self) -> int:
        with self._lock:
            return self._retained

    def log_stats(self) -> None:
        with self._lock:
            entries, retained, hits, misses = (
                len(self._state_dicts),
                self._retained,
                self.hits,
                self.misses,
            )
        logger.info(
            f"[weight_cache] {entries} checkpoints, {retained / _GIB:.2f} GiB host RAM, "
            f"{hits} hits / {misses} misses"
        )


def _rehost(state_dict: Any, host_sd: dict) -> Any:
    """
    Rebuild the immutable `StateDict` container around host tensors.

    `StateDict` is a frozen dataclass (`sd`, `device`, `size`, `dtype`), so
    `dataclasses.replace` is the supported way to change it. Anything that is
    not a dataclass is returned as a plain dict-carrying shim so a future
    upstream change does not silently cache GPU tensors.
    """
    if dataclasses.is_dataclass(state_dict) and not isinstance(state_dict, type):
        fields = {f.name for f in dataclasses.fields(state_dict)}
        changes: dict[str, Any] = {"sd": host_sd}
        if "device" in fields:
            changes["device"] = torch.device("cpu")
        return dataclasses.replace(state_dict, **changes)
    raise TypeError(
        f"weight cache expected a StateDict dataclass, got {type(state_dict)!r}; "
        "refusing to retain it because its storage device cannot be verified"
    )
