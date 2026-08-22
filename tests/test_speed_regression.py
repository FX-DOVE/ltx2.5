#!/usr/bin/env python3
"""
tests/test_speed_regression.py
─────────────────────────────────────────────────────────────────────────────
Regression tests for the generation-speed work.

What is being locked down
─────────────────────────
A 10-second 450p request measured 268 s end to end. The cost is not the step
count — `ltx_pipelines.utils.blocks` states its own design as "Blocks build a
model on each `__call__`, use it, then free GPU memory", and every builder in it
defaults to `ModelRegistry(cache_models=True, cache_weights=False)`, so weights
are re-read from the checkpoint on each of the ~8 builds a single request
performs. On RunPod those checkpoints are on a network volume.

Three changes address that, and each has a failure mode worth a test:

  1. `HostWeightCacheRegistry` retains weights in *host* RAM. Retaining them
     where the loader put them — VRAM — would pin ~44 GB on a 44.39 GiB card and
     reproduce the OOM this deployment already fixed once, because
     `load_state_dict(..., assign=True)` makes parameters alias the cached
     tensors. So "the retained copy is on the host" is the assertion that
     matters most in this file.

  2. `AllocatorTrimStrategy.DEFER` skips the sync + `empty_cache()` upstream
     runs on every block exit — but only where there is VRAM headroom.

  3. The image conditioner builds the VAE encoder unconditionally, and
     `DistilledPipeline` calls it once per stage. On text-to-video,
     `combined_image_conditionings` gets `images=[]`, never touches the encoder
     and returns `[]`, so both builds are waste. The lazy proxy must remove them
     without changing behaviour when a conditioning image *is* supplied.

Run:  python tests/test_speed_regression.py      (or: pytest tests/ -v)
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import sys
import types
import unittest
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from unittest.mock import mock_open, patch

import torch
from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# The modules under test log a line per caching/trim/sigma decision. That is the
# point in production and noise here, where it buries the assertion that failed.
logger.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Doubles for the container-only LTX types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _StateDict:
    """Mirror of `ltx_core.loader.primitives.StateDict` (frozen dataclass)."""

    sd: dict
    device: torch.device
    size: int
    dtype: set = field(default_factory=set)


class _SDOps:
    def __init__(self, name: str) -> None:
        self.name = name


class _AllocTrim(str, Enum):
    """Stand-in for `ltx_core.allocator_trim_strategy.AllocatorTrimStrategy`."""

    TRIM = "trim"
    DEFER = "defer"


class _OffloadMode(str, Enum):
    NONE = "none"
    CPU = "cpu"
    DISK = "disk"


def _sd(nbytes: int = 4096, device: str = "cpu", keys: int = 2) -> _StateDict:
    """A small state dict whose declared size can be inflated for budget tests."""
    tensors = {f"w{i}": torch.ones(8, 8) for i in range(keys)}
    return _StateDict(sd=tensors, device=torch.device(device), size=nbytes)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Host-RAM weight cache
# ─────────────────────────────────────────────────────────────────────────────


class TestHostWeightCacheRetainsOnTheHost(unittest.TestCase):
    """
    The one property that must never regress.

    `SingleGPUModelBuilder._load_model_weights` does
    `load_state_dict(sd, strict=False, assign=True)`, so the model's parameters
    *alias* whatever the registry handed back. If the retained copy lived in
    VRAM, `dispose()` could not reclaim it and two checkpoints would exceed the
    44.39 GiB an L40S reports.
    """

    def setUp(self) -> None:
        from weight_cache import HostWeightCacheRegistry

        self.registry = HostWeightCacheRegistry(budget_bytes=64 * 1024**3)
        self.ops = _SDOps("bf16")

    def test_retained_copy_reports_the_cpu_device(self) -> None:
        # The loader hands over a CUDA-resident SD; the cache must not keep it.
        retained = self.registry.add(["/vol/transformer.safetensors"], self.ops, _sd(device="cuda"))
        self.assertEqual(retained.device, torch.device("cpu"))

    def test_every_retained_tensor_is_on_the_host(self) -> None:
        retained = self.registry.add(["/vol/gemma.safetensors"], self.ops, _sd(keys=3))
        self.assertEqual(len(retained.sd), 3)
        for name, tensor in retained.sd.items():
            self.assertTrue(tensor.is_cpu, f"{name} escaped to {tensor.device}")

    def test_add_returns_the_copy_not_the_original(self) -> None:
        # helpers.load_state_dict: "``add`` returns the retained copy when the
        # registry rewrites storage (e.g. pin to CPU)." The caller must use it.
        original = _sd(device="cuda")
        retained = self.registry.add(["/vol/a.safetensors"], self.ops, original)
        self.assertIsNot(retained, original)
        self.assertIs(retained, self.registry.get(["/vol/a.safetensors"], self.ops))

    def test_non_tensor_entries_survive_rehosting(self) -> None:
        mixed = _StateDict(
            sd={"w": torch.ones(4, 4), "meta": {"format": "pt"}},
            device=torch.device("cuda"),
            size=64,
        )
        retained = self.registry.add(["/vol/mixed.safetensors"], self.ops, mixed)
        self.assertEqual(retained.sd["meta"], {"format": "pt"})

    def test_declared_size_and_dtype_are_preserved(self) -> None:
        original = _StateDict(
            sd={"w": torch.ones(4, 4)},
            device=torch.device("cuda"),
            size=12345,
            dtype={torch.bfloat16},
        )
        retained = self.registry.add(["/vol/b.safetensors"], self.ops, original)
        self.assertEqual(retained.size, 12345)
        self.assertEqual(retained.dtype, {torch.bfloat16})


class TestHostWeightCacheRegistryContract(unittest.TestCase):
    """`get`/`add`/`pop`/`clear` and the model-shell half of the Protocol."""

    def setUp(self) -> None:
        from weight_cache import HostWeightCacheRegistry

        self.registry = HostWeightCacheRegistry(budget_bytes=64 * 1024**3)
        self.ops = _SDOps("bf16")

    def test_miss_returns_none_and_counts(self) -> None:
        self.assertIsNone(self.registry.get(["/vol/absent.safetensors"], self.ops))
        self.assertEqual((self.registry.hits, self.registry.misses), (0, 1))

    def test_hit_after_add(self) -> None:
        self.registry.add(["/vol/a.safetensors"], self.ops, _sd())
        self.assertIsNotNone(self.registry.get(["/vol/a.safetensors"], self.ops))
        self.assertEqual((self.registry.hits, self.registry.misses), (1, 0))

    def test_duplicate_add_returns_the_first_copy_without_raising(self) -> None:
        # Deliberately unlike `ModelRegistry.add`, which raises ValueError on a
        # duplicate key. A rebuild of the same checkpoint is the normal case
        # here, not a programming error.
        first = self.registry.add(["/vol/a.safetensors"], self.ops, _sd())
        second = self.registry.add(["/vol/a.safetensors"], self.ops, _sd())
        self.assertIs(first, second)

    def test_pop_removes_the_entry(self) -> None:
        self.registry.add(["/vol/a.safetensors"], self.ops, _sd())
        self.assertIsNotNone(self.registry.pop(["/vol/a.safetensors"], self.ops))
        self.assertIsNone(self.registry.get(["/vol/a.safetensors"], self.ops))

    def test_pop_of_an_absent_key_is_none(self) -> None:
        self.assertIsNone(self.registry.pop(["/vol/nope.safetensors"], self.ops))

    def test_key_separates_different_sd_ops(self) -> None:
        # The same file loaded under a different fuse/quantization rule is a
        # different tensor set, so it must not collide.
        self.registry.add(["/vol/t.safetensors"], _SDOps("bf16"), _sd())
        self.assertIsNone(self.registry.get(["/vol/t.safetensors"], _SDOps("fp8-cast")))

    def test_key_normalises_paths(self) -> None:
        here = Path(__file__)
        spelled_oddly = str(here.parent / "." / here.name)
        self.registry.add([str(here)], self.ops, _sd())
        self.assertIsNotNone(self.registry.get([spelled_oddly], self.ops))

    def test_key_covers_every_path_in_a_sharded_checkpoint(self) -> None:
        self.registry.add(["/vol/shard-1.safetensors", "/vol/shard-2.safetensors"], self.ops, _sd())
        self.assertIsNone(self.registry.get(["/vol/shard-1.safetensors"], self.ops))

    def test_empty_path_list_does_not_raise(self) -> None:
        self.assertIsNone(self.registry.get([], self.ops))
        self.registry.add([], self.ops, _sd())
        self.assertIsNotNone(self.registry.get([], self.ops))

    def test_model_shells_round_trip(self) -> None:
        shell = object()
        self.assertIs(self.registry.add_model("transformer", shell), shell)
        self.assertIs(self.registry.get_model("transformer"), shell)
        self.assertIs(self.registry.pop_model("transformer"), shell)
        self.assertIsNone(self.registry.get_model("transformer"))

    def test_add_model_keeps_the_first_shell(self) -> None:
        first, second = object(), object()
        self.registry.add_model("k", first)
        self.assertIs(self.registry.add_model("k", second), first)

    def test_clear_resets_entries_and_accounting(self) -> None:
        self.registry.add(["/vol/a.safetensors"], self.ops, _sd(nbytes=1024))
        self.registry.add_model("k", object())
        self.registry.clear()
        self.assertIsNone(self.registry.get(["/vol/a.safetensors"], self.ops))
        self.assertIsNone(self.registry.get_model("k"))
        self.assertEqual(self.registry.retained_bytes, 0)

    def test_log_stats_never_raises(self) -> None:
        self.registry.add(["/vol/a.safetensors"], self.ops, _sd())
        self.registry.log_stats()  # diagnostics only; must not throw


class TestHostWeightCacheBudget(unittest.TestCase):
    """Running out of budget must cost speed, never correctness."""

    def setUp(self) -> None:
        self.ops = _SDOps("bf16")

    def _registry(self, gib: float):
        from weight_cache import HostWeightCacheRegistry

        return HostWeightCacheRegistry(budget_bytes=int(gib * 1024**3))

    def test_oversized_checkpoint_is_returned_uncached(self) -> None:
        registry = self._registry(1.0)
        original = _sd(nbytes=8 * 1024**3)
        returned = registry.add(["/vol/huge.safetensors"], self.ops, original)
        self.assertIs(returned, original)
        self.assertIsNone(registry.get(["/vol/huge.safetensors"], self.ops))
        self.assertEqual(registry.retained_bytes, 0)

    def test_second_checkpoint_refused_once_the_budget_is_spent(self) -> None:
        registry = self._registry(2.0)
        first = _sd(nbytes=int(1.5 * 1024**3))
        second = _sd(nbytes=int(1.5 * 1024**3))
        self.assertIsNot(registry.add(["/vol/one.safetensors"], self.ops, first), first)
        self.assertIs(registry.add(["/vol/two.safetensors"], self.ops, second), second)
        self.assertEqual(registry.retained_bytes, int(1.5 * 1024**3))

    def test_zero_budget_refuses_everything_rather_than_meaning_unlimited(self) -> None:
        # `LTX_WEIGHT_CACHE=on` on a RAM-starved host used to compute a 0 byte
        # budget; a falsey-budget short-circuit would have made that unbounded.
        registry = self._registry(0.0)
        original = _sd()
        self.assertIs(registry.add(["/vol/a.safetensors"], self.ops, original), original)
        self.assertEqual(registry.retained_bytes, 0)

    def test_declared_size_drives_accounting(self) -> None:
        registry = self._registry(64.0)
        registry.add(["/vol/a.safetensors"], self.ops, _sd(nbytes=7 * 1024**3))
        self.assertEqual(registry.retained_bytes, 7 * 1024**3)

    def test_missing_declared_size_falls_back_to_measuring_tensors(self) -> None:
        registry = self._registry(64.0)
        sd = _StateDict(sd={"w": torch.ones(16, 16, dtype=torch.float32)}, device=torch.device("cpu"), size=0)
        registry.add(["/vol/a.safetensors"], self.ops, sd)
        self.assertEqual(registry.retained_bytes, 16 * 16 * 4)


class TestHostWeightCacheRefusesUnverifiableStorage(unittest.TestCase):
    """
    `_rehost` only knows how to rewrite the `StateDict` dataclass.

    If upstream ever changes that container, silently caching the object as-is
    would retain GPU tensors — the exact OOM this class exists to prevent. It
    must decline instead, and declining must not fail the build.
    """

    def setUp(self) -> None:
        from weight_cache import HostWeightCacheRegistry

        self.registry = HostWeightCacheRegistry(budget_bytes=64 * 1024**3)
        self.ops = _SDOps("bf16")

    def test_plain_object_is_not_cached_and_does_not_raise(self) -> None:
        class _Foreign:
            sd = {"w": torch.ones(2, 2)}
            size = 16

        original = _Foreign()
        self.assertIs(self.registry.add(["/vol/x.safetensors"], self.ops, original), original)
        self.assertIsNone(self.registry.get(["/vol/x.safetensors"], self.ops))

    def test_rehost_rejects_a_non_dataclass(self) -> None:
        from weight_cache import _rehost

        with self.assertRaises(TypeError):
            _rehost(object(), {"w": torch.ones(2, 2)})

    def test_rehost_rejects_a_dataclass_type_rather_than_an_instance(self) -> None:
        from weight_cache import _rehost

        with self.assertRaises(TypeError):
            _rehost(_StateDict, {"w": torch.ones(2, 2)})

    def test_rehost_tolerates_a_container_without_a_device_field(self) -> None:
        from weight_cache import _rehost

        @dataclass(frozen=True)
        class _NoDevice:
            sd: dict
            size: int

        rehosted = _rehost(_NoDevice(sd={"w": torch.zeros(1)}, size=4), {"w": torch.ones(1)})
        self.assertEqual(rehosted.sd["w"].item(), 1.0)
        self.assertEqual(rehosted.size, 4)


class TestToHost(unittest.TestCase):
    def test_pin_failure_degrades_to_pageable_memory(self) -> None:
        from weight_cache import _to_host

        with patch.object(torch.Tensor, "pin_memory", side_effect=RuntimeError("no pinning here")):
            host = _to_host({"w": torch.ones(4, 4)}, pin=True)
        self.assertTrue(host["w"].is_cpu)

    def test_detaches_so_the_cache_holds_no_graph(self) -> None:
        from weight_cache import _to_host

        leaf = torch.ones(4, 4, requires_grad=True)
        host = _to_host({"w": leaf * 2}, pin=False)
        self.assertFalse(host["w"].requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 2. The two DistilledPipeline levers this deployment had never set
# ─────────────────────────────────────────────────────────────────────────────


@contextmanager
def _env(**overrides: str):
    """
    Run with a clean LTX_* environment plus `overrides`.

    Every knob under test reads `os.environ` directly, so a stray LTX_* export
    in the developer's shell would otherwise decide the result.
    """
    import os

    saved = dict(os.environ)
    for key in [k for k in os.environ if k.startswith("LTX_")]:
        del os.environ[key]
    os.environ.update(overrides)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


class TestResolveAllocTrim(unittest.TestCase):
    """
    `AllocatorTrimStrategy.TRIM` is upstream's default and costs a
    `synchronize_device()` + `cleanup_memory()` on every `gpu_model()` exit —
    about eight per request here. `DEFER` keeps `dispose()` and leaves the
    allocator pool warm, which is only safe with headroom to spare.
    """

    def setUp(self) -> None:
        import model_loader

        self.model_loader = model_loader

    def _resolve(self, offload=_OffloadMode.NONE, vram: float = 44.39, **env: str):
        with _env(**env), patch.object(self.model_loader, "_total_vram_gib", return_value=vram):
            return self.model_loader._resolve_alloc_trim(_AllocTrim, offload)

    def test_auto_defers_on_a_48gb_card_with_resident_weights(self) -> None:
        # An L40S reports 44.39 GiB; _OFFLOAD_NONE_MIN_GIB is 44.0.
        self.assertIs(self._resolve(vram=44.39), _AllocTrim.DEFER)

    def test_auto_trims_just_below_the_threshold(self) -> None:
        self.assertIs(self._resolve(vram=43.9), _AllocTrim.TRIM)

    def test_auto_trims_when_offload_is_streaming(self) -> None:
        # CPU/DISK offload moves weights across the bus constantly; the pool has
        # to go back to the driver.
        self.assertIs(self._resolve(offload=_OffloadMode.CPU, vram=80.0), _AllocTrim.TRIM)
        self.assertIs(self._resolve(offload=_OffloadMode.DISK, vram=80.0), _AllocTrim.TRIM)

    def test_auto_trims_without_a_cuda_device(self) -> None:
        self.assertIs(self._resolve(vram=0.0), _AllocTrim.TRIM)

    def test_explicit_values_win_over_the_heuristic(self) -> None:
        self.assertIs(self._resolve(vram=0.0, LTX_ALLOC_TRIM="defer"), _AllocTrim.DEFER)
        self.assertIs(self._resolve(vram=80.0, LTX_ALLOC_TRIM="trim"), _AllocTrim.TRIM)

    def test_case_and_whitespace_are_tolerated(self) -> None:
        self.assertIs(self._resolve(vram=0.0, LTX_ALLOC_TRIM="  DEFER  "), _AllocTrim.DEFER)

    def test_unknown_value_is_rejected_at_load_time(self) -> None:
        # Better a refused cold start than a silent fall back to the slow path.
        with self.assertRaises(RuntimeError) as ctx:
            self._resolve(LTX_ALLOC_TRIM="aggressive")
        self.assertIn("LTX_ALLOC_TRIM", str(ctx.exception))


class TestBuildWeightCache(unittest.TestCase):
    """`LTX_WEIGHT_CACHE` resolution, including the `auto` RAM heuristics."""

    def setUp(self) -> None:
        import model_loader

        self.model_loader = model_loader

    def _build(self, offload=_OffloadMode.NONE, ram: float = 128.0, **env: str):
        with _env(**env), patch.object(
            self.model_loader, "_available_host_ram_gib", return_value=ram
        ):
            return self.model_loader._build_weight_cache(offload)

    def test_off_returns_none(self) -> None:
        for value in ("off", "0", "false"):
            self.assertIsNone(self._build(LTX_WEIGHT_CACHE=value), value)

    def test_auto_enables_the_cache_on_a_roomy_host(self) -> None:
        from weight_cache import HostWeightCacheRegistry

        registry = self._build(ram=128.0)
        self.assertIsInstance(registry, HostWeightCacheRegistry)
        # budget = available - 24 GiB reserve
        self.assertEqual(registry._budget, int((128.0 - 24.0) * 1024**3))

    def test_auto_declines_when_the_budget_cannot_hold_both_big_checkpoints(self) -> None:
        # 60 - 24 = 36 GiB, under the 56 GiB Gemma (~24.5) + fp8 transformer
        # (~19.6) actually need.
        self.assertIsNone(self._build(ram=60.0))

    def test_auto_declines_while_offload_streams_from_the_host(self) -> None:
        # The streaming builder already holds weights host-side; a second copy
        # would compete for the same RAM.
        self.assertIsNone(self._build(offload=_OffloadMode.CPU, ram=256.0))
        self.assertIsNone(self._build(offload=_OffloadMode.DISK, ram=256.0))

    def test_on_overrides_the_offload_heuristic(self) -> None:
        from weight_cache import HostWeightCacheRegistry

        registry = self._build(offload=_OffloadMode.CPU, ram=256.0, LTX_WEIGHT_CACHE="on")
        self.assertIsInstance(registry, HostWeightCacheRegistry)

    def test_on_overrides_the_56gb_minimum(self) -> None:
        from weight_cache import HostWeightCacheRegistry

        registry = self._build(ram=60.0, LTX_WEIGHT_CACHE="on")
        self.assertIsInstance(registry, HostWeightCacheRegistry)

    def test_on_still_declines_when_the_budget_would_be_useless(self) -> None:
        # The floor exists so `on` cannot construct a cache that pays the
        # device→host copy and then refuses every entry.
        self.assertIsNone(self._build(ram=30.0, LTX_WEIGHT_CACHE="on"))
        self.assertIsNone(self._build(ram=0.0, LTX_WEIGHT_CACHE="on"))

    def test_pinning_is_opt_in(self) -> None:
        self.assertFalse(self._build()._pin)
        self.assertTrue(self._build(LTX_WEIGHT_CACHE_PIN="1")._pin)

    def test_unknown_value_is_rejected_at_load_time(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            self._build(LTX_WEIGHT_CACHE="host")
        self.assertIn("LTX_WEIGHT_CACHE", str(ctx.exception))

    def test_a_plain_string_offload_mode_is_understood(self) -> None:
        # _resolve_offload_mode returns an enum, but the helpers read `.value`
        # with a str() fallback, so a string must not silently mean "streaming".
        from weight_cache import HostWeightCacheRegistry

        self.assertIsInstance(self._build(offload="none"), HostWeightCacheRegistry)
        self.assertIsNone(self._build(offload="cpu"))


class TestHostRamProbes(unittest.TestCase):
    """
    Both probes gate real decisions, so a missing psutil must not silently
    become "0 GiB" and take the pessimistic branch on a Linux container.
    """

    def setUp(self) -> None:
        import model_loader

        self.model_loader = model_loader

    def test_probes_return_a_non_negative_float(self) -> None:
        for probe in (self.model_loader._total_host_ram_gib, self.model_loader._available_host_ram_gib):
            value = probe()
            self.assertIsInstance(value, float)
            self.assertGreaterEqual(value, 0.0)

    def test_meminfo_fallback_is_used_when_psutil_is_missing(self) -> None:
        real_import = __import__

        def _no_psutil(name, *args, **kwargs):
            if name == "psutil":
                raise ImportError("no psutil")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", _no_psutil), patch.object(
            self.model_loader, "_meminfo_gib", return_value=123.0
        ):
            self.assertEqual(self.model_loader._total_host_ram_gib(), 123.0)
            self.assertEqual(self.model_loader._available_host_ram_gib(), 123.0)

    def test_meminfo_parses_kilobytes(self) -> None:
        content = "MemTotal:       131072000 kB\nMemAvailable:    65536000 kB\n"
        with patch("builtins.open", mock_open(read_data=content)):
            self.assertAlmostEqual(self.model_loader._meminfo_gib("MemTotal"), 125.0, places=3)

    def test_meminfo_returns_zero_when_unreadable(self) -> None:
        with patch("builtins.open", side_effect=OSError("no /proc here")):
            self.assertEqual(self.model_loader._meminfo_gib("MemTotal"), 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Sigma schedule override
# ─────────────────────────────────────────────────────────────────────────────


class TestSigmaParsing(unittest.TestCase):
    """
    `stage_1_sigmas` / `stage_2_sigmas` are real `__call__` kwargs, so the step
    count *is* reachable — but a distilled model asked for sigmas it never
    trained on degrades rather than merely running faster. Anything the sampler
    cannot walk must be rejected loudly at parse time, not produce mush.
    """

    def setUp(self) -> None:
        import inference

        self.inference = inference

    def test_accepts_the_trimmed_stage_1_schedule(self) -> None:
        # Upstream stage 1 spends four of eight steps between 1.0 and 0.975 —
        # 2.5% of the trajectory for half the stage's compute.
        values = self.inference._parse_sigmas("1.0,0.975,0.909375,0.725,0.421875,0.0", "X")
        self.assertEqual(values, [1.0, 0.975, 0.909375, 0.725, 0.421875, 0.0])

    def test_accepts_the_upstream_defaults_unchanged(self) -> None:
        stage1 = "1.0,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0.0"
        self.assertEqual(len(self.inference._parse_sigmas(stage1, "X")), 9)
        self.assertEqual(len(self.inference._parse_sigmas("0.909375,0.725,0.421875,0.0", "X")), 4)

    def test_spaces_are_ignored(self) -> None:
        self.assertEqual(self.inference._parse_sigmas(" 1.0 , 0.5 , 0.0 ", "X"), [1.0, 0.5, 0.0])

    def test_rejects_a_schedule_that_does_not_end_at_zero(self) -> None:
        with self.assertRaises(RuntimeError):
            self.inference._parse_sigmas("1.0,0.5,0.1", "X")

    def test_rejects_a_non_decreasing_schedule(self) -> None:
        with self.assertRaises(RuntimeError):
            self.inference._parse_sigmas("1.0,0.5,0.5,0.0", "X")
        with self.assertRaises(RuntimeError):
            self.inference._parse_sigmas("0.5,1.0,0.0", "X")

    def test_rejects_values_outside_the_unit_interval(self) -> None:
        with self.assertRaises(RuntimeError):
            self.inference._parse_sigmas("1.5,0.5,0.0", "X")
        with self.assertRaises(RuntimeError):
            self.inference._parse_sigmas("1.0,0.5,-0.1", "X")

    def test_rejects_a_single_value(self) -> None:
        with self.assertRaises(RuntimeError):
            self.inference._parse_sigmas("0.0", "X")

    def test_rejects_garbage(self) -> None:
        with self.assertRaises(RuntimeError):
            self.inference._parse_sigmas("fast", "X")

    def test_error_names_the_env_var(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            self.inference._parse_sigmas("nonsense", "LTX_STAGE1_SIGMAS")
        self.assertIn("LTX_STAGE1_SIGMAS", str(ctx.exception))


class TestSigmaOverrides(unittest.TestCase):
    """The override has to stay opt-in: unset means upstream's tuned schedule."""

    def setUp(self) -> None:
        import inference

        self.inference = inference

    def test_no_env_means_no_kwargs(self) -> None:
        with _env():
            self.assertEqual(self.inference._sigma_overrides(), {})

    def test_blank_env_is_treated_as_unset(self) -> None:
        with _env(LTX_STAGE1_SIGMAS="   "):
            self.assertEqual(self.inference._sigma_overrides(), {})

    def test_each_stage_maps_to_its_own_kwarg(self) -> None:
        with _env(LTX_STAGE1_SIGMAS="1.0,0.5,0.0", LTX_STAGE2_SIGMAS="0.9,0.4,0.0"):
            overrides = self.inference._sigma_overrides()
        self.assertEqual(set(overrides), {"stage_1_sigmas", "stage_2_sigmas"})
        self.assertTrue(torch.is_tensor(overrides["stage_1_sigmas"]))
        self.assertEqual(overrides["stage_1_sigmas"].dtype, torch.float32)
        self.assertEqual(overrides["stage_1_sigmas"].tolist(), [1.0, 0.5, 0.0])

    def test_one_stage_can_be_overridden_alone(self) -> None:
        with _env(LTX_STAGE2_SIGMAS="0.9,0.0"):
            self.assertEqual(list(self.inference._sigma_overrides()), ["stage_2_sigmas"])

    def test_a_bad_schedule_fails_the_request_rather_than_being_ignored(self) -> None:
        with _env(LTX_STAGE1_SIGMAS="1.0,2.0,0.0"), self.assertRaises(RuntimeError):
            self.inference._sigma_overrides()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Instrumentation and the lazy VAE encoder
# ─────────────────────────────────────────────────────────────────────────────


class _FakeVaeEncoder:
    """The object `combined_image_conditionings` calls as `video_encoder(img)`."""

    tile_size = 512

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, image: str) -> str:
        self.calls.append(image)
        return f"latent({image})"


class _RecordingGpuModel:
    """
    Stand-in for `ltx_pipelines.utils.gpu_model.gpu_model`.

    Records what was handed to it and when it was released, which is how these
    tests observe whether a build happened at all.
    """

    def __init__(self) -> None:
        self.entered: list[Any] = []
        self.exited: list[Any] = []
        self.trims: list[Any] = []

    @contextmanager
    def __call__(self, model: Any, alloc_trim_strategy: Any = None):
        self.entered.append(model)
        self.trims.append(alloc_trim_strategy)
        try:
            yield model
        finally:
            self.exited.append(model)


def _gpu_model_stubs(recorder: _RecordingGpuModel) -> dict:
    """`sys.modules` entries so `_LazyEncoder` can import upstream's context."""
    leaf = types.ModuleType("ltx_pipelines.utils.gpu_model")
    leaf.gpu_model = recorder  # type: ignore[attr-defined]
    return {
        "ltx_pipelines": types.ModuleType("ltx_pipelines"),
        "ltx_pipelines.utils": types.ModuleType("ltx_pipelines.utils"),
        "ltx_pipelines.utils.gpu_model": leaf,
    }


class _FakeImageConditioner:
    """
    Mirror of `ltx_pipelines.utils.blocks.ImageConditioner`.

    Upstream's `__call__` is, in full:

        with gpu_model(self._build_encoder(), alloc_trim_strategy=...) as encoder:
            return fn(encoder)

    so the build is unconditional — it happens before `fn` gets a say.
    """

    def __init__(self, gpu_model: _RecordingGpuModel) -> None:
        self.builds = 0
        self.resolve_crf = True  # a real attribute DistilledPipeline reads
        self._alloc_trim_strategy = _AllocTrim.DEFER
        self._gpu_model = gpu_model

    def _build_encoder(self) -> _FakeVaeEncoder:
        self.builds += 1
        return _FakeVaeEncoder()

    def __call__(self, fn):
        with self._gpu_model(
            self._build_encoder(), alloc_trim_strategy=self._alloc_trim_strategy
        ) as encoder:
            return fn(encoder)


def _conditioning_fn(images: list[str]):
    """
    Mirror of `helpers.combined_image_conditionings`.

    The encoder is touched only inside `for img in images:` — so with no
    conditioning image the callable never looks at it and returns `[]`.
    """

    def fn(video_encoder):
        return [video_encoder(img) for img in images]

    return fn


class TestLazyImageConditionerRemovesUnusedBuilds(unittest.TestCase):
    """
    `DistilledPipeline` calls `image_conditioner` once per stage, and a
    text-to-video request passes `images=[]` both times. Upstream still builds
    the VAE encoder twice and throws both away.
    """

    def setUp(self) -> None:
        import perf

        self.perf = perf
        self.gpu_model = _RecordingGpuModel()
        self.block = _FakeImageConditioner(self.gpu_model)
        self.wrapped = perf._LazyImageConditioner(self.block)
        self.stubs = patch.dict(sys.modules, _gpu_model_stubs(self.gpu_model))
        self.stubs.start()
        self.addCleanup(self.stubs.stop)

    def test_text_to_video_builds_no_encoder(self) -> None:
        self.assertEqual(self.wrapped(_conditioning_fn([])), [])
        self.assertEqual(self.block.builds, 0)
        self.assertEqual(self.gpu_model.entered, [])

    def test_both_stage_calls_are_free_on_text_to_video(self) -> None:
        self.wrapped(_conditioning_fn([]))
        self.wrapped(_conditioning_fn([]))
        self.assertEqual(self.block.builds, 0)

    def test_a_conditioning_image_still_gets_encoded(self) -> None:
        self.assertEqual(self.wrapped(_conditioning_fn(["first.png"])), ["latent(first.png)"])
        self.assertEqual(self.block.builds, 1)
        self.assertEqual(len(self.gpu_model.entered), 1)

    def test_first_last_frame_shares_one_encoder(self) -> None:
        result = self.wrapped(_conditioning_fn(["first.png", "last.png"]))
        self.assertEqual(result, ["latent(first.png)", "latent(last.png)"])
        self.assertEqual(self.block.builds, 1)

    def test_the_encoder_is_released_when_it_was_used(self) -> None:
        self.wrapped(_conditioning_fn(["first.png"]))
        self.assertEqual(len(self.gpu_model.exited), 1)

    def test_attribute_access_also_materialises_the_encoder(self) -> None:
        # Anything that reaches past __call__ must still get a real encoder.
        def fn(video_encoder):
            return video_encoder.tile_size

        self.assertEqual(self.wrapped(fn), 512)
        self.assertEqual(self.block.builds, 1)

    def test_the_alloc_trim_strategy_is_forwarded(self) -> None:
        self.wrapped(_conditioning_fn(["first.png"]))
        self.assertEqual(self.gpu_model.trims, [_AllocTrim.DEFER])

    def test_an_exception_inside_the_callable_still_releases_the_encoder(self) -> None:
        def fn(video_encoder):
            video_encoder("first.png")
            raise ValueError("conditioning blew up")

        with self.assertRaises(ValueError):
            self.wrapped(fn)
        self.assertEqual(len(self.gpu_model.exited), 1)

    def test_block_attributes_are_forwarded_both_ways(self) -> None:
        self.assertIs(self.wrapped.resolve_crf, True)
        self.wrapped.resolve_crf = False
        self.assertIs(self.block.resolve_crf, False)

    def test_falls_back_to_the_eager_path_if_upstream_moves_the_build_hook(self) -> None:
        # A rename of `_build_encoder` must degrade to upstream behaviour, not
        # to a crash or a silently skipped conditioning image.
        class _Renamed:
            def __init__(self) -> None:
                self.called = 0

            def __call__(self, fn):
                self.called += 1
                return fn(_FakeVaeEncoder())

        block = _Renamed()
        wrapped = self.perf._LazyImageConditioner(block)
        self.assertEqual(wrapped(_conditioning_fn(["a.png"])), ["latent(a.png)"])
        self.assertEqual(block.called, 1)


class _FakeBlock:
    """A pipeline block that returns an already-computed result."""

    def __init__(self, result: Any, label: str = "block") -> None:
        self.result = result
        self.label = label
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return self.result


class _FakeChunkDecoder:
    """
    Mirror of the `video_decoder` block: returns a *lazy* generator.

    `DistilledPipeline` also reads `checkpoint_path` and `diffvae_optimization`
    off this object, so a proxy that only forwards `__call__` breaks the
    pipeline.
    """

    def __init__(self, chunks: int = 3) -> None:
        self.chunks = chunks
        self.checkpoint_path = "/vol/video-vae.safetensors"
        self.diffvae_optimization = "conv"
        self.pulled = 0

    def __call__(self, *args: Any, **kwargs: Any):
        def _gen():
            for index in range(self.chunks):
                self.pulled += 1
                yield torch.zeros(1, 2, 2, 3) + index

        return _gen()


class _FakePipeline:
    """The subset of `DistilledPipeline`'s surface `instrument_pipeline` touches."""

    def __init__(self, gpu_model: _RecordingGpuModel, duration_predictor: Any = None) -> None:
        self.prompt_encoder = _FakeBlock(("embeddings", "mask"))
        self.image_conditioner = _FakeImageConditioner(gpu_model)
        self.stage = _FakeBlock("latents")
        self.upsampler = _FakeBlock("upsampled")
        self.video_decoder = _FakeChunkDecoder(3)
        self.audio_decoder = _FakeBlock("audio")
        self.duration_predictor = duration_predictor


class TestTimedBlock(unittest.TestCase):
    def setUp(self) -> None:
        import perf

        self.perf = perf
        self.ledger = perf.PhaseLedger()

    def test_records_one_entry_per_call(self) -> None:
        block = self.perf._TimedBlock(_FakeBlock("x"), "stage", self.ledger)
        block()
        block()
        label, count, seconds = self.ledger.snapshot()[0]
        self.assertEqual((label, count), ("stage", 2))
        self.assertGreaterEqual(seconds, 0.0)

    def test_returns_the_result_untouched(self) -> None:
        result = ("embeddings", "mask")
        block = self.perf._TimedBlock(_FakeBlock(result), "prompt_encoder", self.ledger)
        self.assertIs(block(), result)

    def test_arguments_are_passed_through(self) -> None:
        seen: dict = {}

        def target(*args, **kwargs):
            seen.update(kwargs)
            return args

        block = self.perf._TimedBlock(target, "stage", self.ledger)
        self.assertEqual(block(1, 2, seed=7), (1, 2))
        self.assertEqual(seen, {"seed": 7})

    def test_forwards_attribute_reads_and_writes(self) -> None:
        target = _FakeChunkDecoder()
        block = self.perf._TimedBlock(target, "video_decoder", self.ledger)
        self.assertEqual(block.checkpoint_path, "/vol/video-vae.safetensors")
        self.assertEqual(block.diffvae_optimization, "conv")
        block.diffvae_optimization = "bf16"
        self.assertEqual(target.diffvae_optimization, "bf16")

    def test_repr_does_not_recurse(self) -> None:
        block = self.perf._TimedBlock(_FakeBlock("x"), "stage", self.ledger)
        self.assertIn("stage", repr(block))

    def test_a_lazy_generator_is_billed_to_the_drain(self) -> None:
        target = _FakeChunkDecoder(3)
        block = self.perf._TimedBlock(target, "video_decoder", self.ledger)
        stream = block()
        # The call itself computed nothing, so only the setup is recorded yet.
        self.assertEqual([label for label, _c, _s in self.ledger.snapshot()], ["video_decoder:setup"])
        self.assertEqual(target.pulled, 0)
        chunks = list(stream)
        self.assertEqual(len(chunks), 3)
        labels = dict((label, count) for label, count, _s in self.ledger.snapshot())
        self.assertEqual(labels["video_decoder:drain"], 3)

    def test_the_drain_preserves_order_and_values(self) -> None:
        block = self.perf._TimedBlock(_FakeChunkDecoder(3), "video_decoder", self.ledger)
        values = [float(chunk.flatten()[0]) for chunk in block()]
        self.assertEqual(values, [0.0, 1.0, 2.0])

    def test_an_empty_generator_records_no_drain(self) -> None:
        block = self.perf._TimedBlock(_FakeChunkDecoder(0), "video_decoder", self.ledger)
        self.assertEqual(list(block()), [])
        self.assertNotIn("video_decoder:drain", [label for label, _c, _s in self.ledger.snapshot()])


class TestIsLazy(unittest.TestCase):
    """
    Only a real iterator defers work. A tensor and a list are iterable but
    already paid for, and mistaking either for lazy would move their cost into
    a `:drain` label — or worse, replace them with a generator.
    """

    def setUp(self) -> None:
        import perf

        self.perf = perf

    def test_generators_and_iterators_are_lazy(self) -> None:
        self.assertTrue(self.perf._is_lazy(iter([1, 2])))
        self.assertTrue(self.perf._is_lazy((index for index in range(2))))

    def test_settled_containers_and_tensors_are_not(self) -> None:
        for value in ([1, 2], (1, 2), {"a": 1}, "text", b"bytes", torch.zeros(4), 7, None):
            self.assertFalse(self.perf._is_lazy(value), repr(value))


class TestInstrumentPipeline(unittest.TestCase):
    def setUp(self) -> None:
        import perf

        self.perf = perf
        self.ledger = perf.PhaseLedger()
        self.gpu_model = _RecordingGpuModel()
        self.pipeline = _FakePipeline(self.gpu_model)
        self.originals = {
            name: getattr(self.pipeline, name) for name in self.perf._BLOCK_ATTRS
        }
        self.stubs = patch.dict(sys.modules, _gpu_model_stubs(self.gpu_model))
        self.stubs.start()
        self.addCleanup(self.stubs.stop)

    def _instrument(self, **env: str):
        with _env(**env):
            return self.perf.instrument_pipeline(self.pipeline, self.ledger)

    def test_returns_the_same_pipeline_object(self) -> None:
        self.assertIs(self._instrument(), self.pipeline)

    def test_every_present_block_is_wrapped(self) -> None:
        self._instrument()
        for name in ("prompt_encoder", "image_conditioner", "stage", "upsampler",
                     "video_decoder", "audio_decoder"):
            self.assertIsInstance(getattr(self.pipeline, name), self.perf._TimedBlock, name)

    def test_a_block_the_checkpoint_does_not_carry_is_skipped(self) -> None:
        # `duration_predictor` is None on checkpoints predating DurationHead.
        self._instrument()
        self.assertIsNone(self.pipeline.duration_predictor)

    def test_a_present_duration_predictor_is_wrapped(self) -> None:
        pipeline = _FakePipeline(self.gpu_model, duration_predictor=_FakeBlock(241))
        with _env():
            self.perf.instrument_pipeline(pipeline, self.ledger)
        self.assertIsInstance(pipeline.duration_predictor, self.perf._TimedBlock)

    def test_instrumenting_twice_does_not_double_wrap(self) -> None:
        self._instrument()
        once = self.pipeline.prompt_encoder
        self._instrument()
        self.assertIs(self.pipeline.prompt_encoder, once)
        self.assertIs(once._target, self.originals["prompt_encoder"])

    def test_attributes_the_pipeline_reads_survive_wrapping(self) -> None:
        # distilled.py reads all three through the block attribute.
        self._instrument()
        self.assertEqual(self.pipeline.video_decoder.checkpoint_path, "/vol/video-vae.safetensors")
        self.assertEqual(self.pipeline.video_decoder.diffvae_optimization, "conv")
        self.assertIs(self.pipeline.image_conditioner.resolve_crf, True)

    def test_calls_still_reach_the_real_blocks(self) -> None:
        self._instrument()
        self.assertEqual(self.pipeline.stage(), "latents")
        self.assertEqual(self.originals["stage"].calls, 1)
        self.assertEqual(dict((name, count) for name, count, _s in self.ledger.snapshot())["stage"], 1)

    def test_a_text_to_video_request_pays_for_no_encoder_build(self) -> None:
        # The whole point: DistilledPipeline calls this once per stage with
        # images=[], and upstream builds the VAE encoder both times.
        self._instrument()
        self.assertEqual(self.pipeline.image_conditioner(_conditioning_fn([])), [])
        self.assertEqual(self.pipeline.image_conditioner(_conditioning_fn([])), [])
        self.assertEqual(self.originals["image_conditioner"].builds, 0)
        self.assertEqual(self.gpu_model.entered, [])
        self.assertEqual(dict((name, count) for name, count, _s in self.ledger.snapshot())["image_conditioner"], 2)

    def test_an_image_request_is_unaffected(self) -> None:
        self._instrument()
        result = self.pipeline.image_conditioner(_conditioning_fn(["first.png"]))
        self.assertEqual(result, ["latent(first.png)"])
        self.assertEqual(self.originals["image_conditioner"].builds, 1)

    def test_timing_off_still_removes_the_wasted_builds(self) -> None:
        self._instrument(LTX_PERF_TIMING="0")
        self.assertIsInstance(self.pipeline.image_conditioner, self.perf._LazyImageConditioner)
        self.assertIs(self.pipeline.prompt_encoder, self.originals["prompt_encoder"])
        self.pipeline.image_conditioner(_conditioning_fn([]))
        self.assertEqual(self.originals["image_conditioner"].builds, 0)
        self.assertEqual(self.ledger.snapshot(), [])

    def test_lazy_off_restores_the_unconditional_build(self) -> None:
        # The escape hatch has to actually restore upstream behaviour.
        self._instrument(LTX_LAZY_IMAGE_ENCODER="0")
        self.assertIsInstance(self.pipeline.image_conditioner, self.perf._TimedBlock)
        self.assertIs(self.pipeline.image_conditioner._target, self.originals["image_conditioner"])
        self.pipeline.image_conditioner(_conditioning_fn([]))
        self.assertEqual(self.originals["image_conditioner"].builds, 1)

    def test_both_flags_off_leaves_the_pipeline_untouched(self) -> None:
        self._instrument(LTX_PERF_TIMING="0", LTX_LAZY_IMAGE_ENCODER="0")
        for name, original in self.originals.items():
            self.assertIs(getattr(self.pipeline, name), original, name)


class TestPhaseLedger(unittest.TestCase):
    def setUp(self) -> None:
        import perf

        self.ledger = perf.PhaseLedger()

    def test_first_call_order_is_kept(self) -> None:
        for label in ("prompt_encoder", "stage", "upsampler"):
            self.ledger.record(label, 1.0)
        self.ledger.record("stage", 1.0)
        self.assertEqual(
            [label for label, _c, _s in self.ledger.snapshot()],
            ["prompt_encoder", "stage", "upsampler"],
        )

    def test_totals_and_counts_accumulate(self) -> None:
        self.ledger.record("stage", 2.0)
        self.ledger.record("stage", 3.0)
        self.assertEqual(self.ledger.snapshot(), [("stage", 2, 5.0)])
        self.assertEqual(self.ledger.total(), 5.0)

    def test_reset_clears_everything(self) -> None:
        self.ledger.record("stage", 1.0)
        self.ledger.reset()
        self.assertEqual(self.ledger.snapshot(), [])
        self.assertEqual(self.ledger.total(), 0.0)

    def test_report_is_silent_when_nothing_was_recorded(self) -> None:
        self.ledger.report(268.08)  # must not raise on an empty ledger

    def test_report_handles_a_missing_or_zero_wall_time(self) -> None:
        self.ledger.record("stage", 1.0)
        self.ledger.record("stage", 1.0)
        self.ledger.report(None)
        self.ledger.report(0.0)  # would be a ZeroDivisionError if it divided
        self.ledger.report(268.08)


class TestTimePhase(unittest.TestCase):
    """Used for the libx264 encode, which drives the VAE decode as a side effect."""

    def setUp(self) -> None:
        import perf

        self.perf = perf
        self.ledger = perf.PhaseLedger()

    def test_records_the_labelled_phase(self) -> None:
        with self.perf.time_phase("encode_video(+vae decode)", self.ledger):
            pass
        label, count, _seconds = self.ledger.snapshot()[0]
        self.assertEqual((label, count), ("encode_video(+vae decode)", 1))

    def test_records_even_when_the_body_raises(self) -> None:
        # A failed encode is exactly when the breakdown is worth having.
        with self.assertRaises(RuntimeError):
            with self.perf.time_phase("encode_video(+vae decode)", self.ledger):
                raise RuntimeError("libx264 exploded")
        self.assertEqual(len(self.ledger.snapshot()), 1)

    def test_defaults_to_the_module_ledger(self) -> None:
        self.perf.LEDGER.reset()
        with self.perf.time_phase("phase"):
            pass
        self.assertEqual([label for label, _c, _s in self.perf.LEDGER.snapshot()], ["phase"])
        self.perf.LEDGER.reset()


class TestPerfFlags(unittest.TestCase):
    def setUp(self) -> None:
        import perf

        self.perf = perf

    def test_both_default_to_on(self) -> None:
        with _env():
            self.assertTrue(self.perf.timing_enabled())
            self.assertTrue(self.perf.lazy_encoder_enabled())

    def test_falsey_spellings_switch_them_off(self) -> None:
        for value in ("0", "false", "no", "off", ""):
            with _env(LTX_PERF_TIMING=value, LTX_LAZY_IMAGE_ENCODER=value):
                self.assertFalse(self.perf.timing_enabled(), value)
                self.assertFalse(self.perf.lazy_encoder_enabled(), value)

    def test_sync_is_a_no_op_without_cuda(self) -> None:
        with _env(LTX_PERF_SYNC="1"), patch.object(torch.cuda, "is_available", return_value=False):
            self.perf._sync()  # must not raise

    def test_sync_swallows_driver_errors(self) -> None:
        with _env(LTX_PERF_SYNC="1"), patch.object(
            torch.cuda, "is_available", return_value=True
        ), patch.object(torch.cuda, "synchronize", side_effect=RuntimeError("driver gone")):
            self.perf._sync()  # diagnostics must never break a request


if __name__ == "__main__":
    unittest.main(verbosity=2)
