#!/usr/bin/env python3
"""
tests/test_oom_regression.py
─────────────────────────────────────────────────────────────────────────────
Regression tests for the inference-time CUDA OOM.

The failure these lock down
───────────────────────────
A production run on a 48 GB L40S died with:

    Tried to allocate 1.44 GiB. GPU 0 has a total capacity of 44.39 GiB of
    which 105.38 MiB is free ... 43.37 GiB is allocated by PyTorch, and
    417.11 MiB is reserved by PyTorch but unallocated.

...immediately after `PromptEncoder` logged "Text encoder done, building
embeddings processor". The cause was autograd, not fragmentation (only 417 MiB
was reserved-unallocated, and expandable_segments was already on):

  * nothing in `ltx_pipelines.utils.blocks` disables gradient tracking — the
    only guard upstream ships is `@torch.inference_mode()` on the reference
    CLI's `distilled.main()`, which this repo's handler replaces;
  * with grad on, the graph hanging off Gemma's hidden states holds the saved
    activations and weight storages, so the `gpu_model()` exit's `dispose()` +
    `cleanup_memory()` reclaim nothing;
  * the next build then has ~1 GiB to work with instead of ~44.

These tests assert the guards are in place at both points where tensors are
produced: the eager pipeline call, and the *lazy* VAE chunk iterator that is
drained later by `encode_video`.

Run:  python tests/test_oom_regression.py       (or: pytest tests/ -v)
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import sys
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# ─────────────────────────────────────────────────────────────────────────────
# Stubs for the LTX packages, which are only installed inside the container.
# ─────────────────────────────────────────────────────────────────────────────

_LTX_STUBS = {
    "ltx_core": MagicMock(),
    "ltx_core.model": MagicMock(),
    "ltx_core.model.video_vae": MagicMock(AUTO_TILING=object()),
    "ltx_pipelines": MagicMock(),
    "ltx_pipelines.utils": MagicMock(),
    "ltx_pipelines.utils.args": MagicMock(),
}


class _RecordingPipeline:
    """
    Fake DistilledPipeline that records the grad state it was called under and
    the grad state each lazy chunk is produced under.

    Mirrors the real return contract: (chunk iterator, audio, num_frames,
    tiling_config), where the iterator is lazy — the whole point of the second
    guard is that these tensors materialise after `run_inference` has returned.
    """

    def __init__(self, chunks: int = 3) -> None:
        self.chunks = chunks
        self.grad_at_call: bool | None = None
        self.grad_at_chunk: list[bool] = []
        # A tiny real module so the recorded tensors go through a real autograd
        # decision rather than a mocked one.
        self.layer = torch.nn.Linear(4, 4)

    def __call__(self, **kwargs: object):
        self.grad_at_call = torch.is_grad_enabled()
        return self._decode(), None, 9, None

    def _decode(self):
        for _ in range(self.chunks):
            self.grad_at_chunk.append(torch.is_grad_enabled())
            yield self.layer(torch.ones(1, 4))


def _params(**overrides: object):
    from schema import InferenceInput

    payload: dict = {"prompt": "a cat", "num_frames": 9, "seed": 42}
    payload.update(overrides)
    return InferenceInput.model_validate(payload)


class TestInferenceRunsWithoutAutograd(unittest.TestCase):
    """The guards that keep the text encoder's 24 GB from surviving dispose()."""

    def _run(self, pipeline):
        import inference

        with patch.dict(sys.modules, _LTX_STUBS):
            return inference.run_inference(pipeline, _params())

    def test_pipeline_is_called_with_grad_disabled(self):
        pipeline = _RecordingPipeline()
        self._run(pipeline)
        self.assertIs(
            pipeline.grad_at_call,
            False,
            "run_inference must call the pipeline inside torch.no_grad(); with "
            "grad enabled the text encoder's graph pins ~24 GB past dispose() "
            "and the embeddings-processor build OOMs.",
        )

    def test_lazy_vae_chunks_are_pulled_with_grad_disabled(self):
        pipeline = _RecordingPipeline(chunks=3)
        result = self._run(pipeline)
        # Drain the iterator the way encode_video does — outside run_inference,
        # so only the iterator's own guard can be responsible.
        chunks = list(result.video)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(
            pipeline.grad_at_chunk,
            [False, False, False],
            "Each pull from the VAE chunk iterator must happen inside "
            "torch.no_grad(); decode runs lazily during encode_video.",
        )

    def test_decoded_chunks_carry_no_autograd_graph(self):
        pipeline = _RecordingPipeline(chunks=2)
        result = self._run(pipeline)
        for chunk in result.video:
            self.assertIsNone(
                chunk.grad_fn,
                "A decoded chunk with grad_fn keeps the decoder's activations "
                "alive for the lifetime of the chunk.",
            )
            self.assertFalse(chunk.requires_grad)

    def test_chunks_are_not_inference_tensors(self):
        """
        no_grad, not inference_mode.

        `encode_video` hands these tensors to PyAV/numpy after they have left
        the guard's scope. Inference-mode tensors raise on in-place mutation
        outside the mode that created them, so the guard must be `no_grad`,
        which sheds the graph just as completely without marking the tensor.
        """
        pipeline = _RecordingPipeline(chunks=1)
        result = self._run(pipeline)
        for chunk in result.video:
            self.assertFalse(
                chunk.is_inference(),
                "Chunks must not be inference tensors — encode_video consumes "
                "them outside the guard.",
            )

    def test_grad_state_is_restored_for_the_caller(self):
        pipeline = _RecordingPipeline()
        was_enabled = torch.is_grad_enabled()
        result = self._run(pipeline)
        list(result.video)
        self.assertEqual(torch.is_grad_enabled(), was_enabled)

    def test_negative_control_the_same_fake_records_grad_when_unguarded(self):
        """
        Guards against a vacuous suite.

        Calling the identical fake pipeline directly — no `run_inference`, no
        guard — must record grad enabled and produce chunks carrying `grad_fn`.
        If this ever reports False, the assertions above prove nothing because
        something else already disabled grad.
        """
        pipeline = _RecordingPipeline(chunks=2)
        chunks, _audio, _frames, _tiling = pipeline()
        materialised = list(chunks)
        self.assertIs(pipeline.grad_at_call, True)
        self.assertEqual(pipeline.grad_at_chunk, [True, True])
        self.assertIsNotNone(materialised[0].grad_fn)


class _OffloadMode(str, Enum):
    """Stand-in for ltx_pipelines.utils.types.OffloadMode."""

    NONE = "none"
    CPU = "cpu"
    DISK = "disk"


class TestOffloadModeAuto(unittest.TestCase):
    """
    `LTX_OFFLOAD_MODE=auto` must never leave a card on a setting it cannot run.

    The OOM run was configured `offload=none` on a 44.39 GiB card. With the grad
    fix that configuration is correct and fastest, so `auto` keeps it — but a
    smaller GPU has to degrade to streaming rather than fail.
    """

    def _resolve(self, vram: float, host: float = 1007.0, quant: str = "fp8-cast"):
        import model_loader

        with patch.object(model_loader, "_total_vram_gib", return_value=vram), \
             patch.object(model_loader, "_total_host_ram_gib", return_value=host), \
             patch.dict("os.environ", {"LTX_OFFLOAD_MODE": "auto"}):
            return model_loader._resolve_offload_mode(_OffloadMode, quant)

    def test_48gb_card_keeps_weights_resident(self):
        # The L40S from the failing run reports 44.39 GiB.
        self.assertEqual(self._resolve(44.39), _OffloadMode.NONE)

    def test_80gb_card_keeps_weights_resident(self):
        self.assertEqual(self._resolve(79.2), _OffloadMode.NONE)

    def test_24gb_card_streams_from_host_ram(self):
        # RTX 4090 / L4 class: cannot hold the fp8 transformer plus Gemma.
        self.assertEqual(self._resolve(23.6), _OffloadMode.CPU)

    def test_small_card_with_small_host_streams_from_disk(self):
        # CPU offload pins the whole weight set in RAM; without the RAM, DISK.
        self.assertEqual(self._resolve(23.6, host=16.0), _OffloadMode.DISK)

    def test_no_cuda_device_resolves_to_none(self):
        self.assertEqual(self._resolve(0.0), _OffloadMode.NONE)

    def test_explicit_value_overrides_auto(self):
        import model_loader

        with patch.object(model_loader, "_total_vram_gib", return_value=79.0), \
             patch.dict("os.environ", {"LTX_OFFLOAD_MODE": "cpu"}):
            mode = model_loader._resolve_offload_mode(_OffloadMode, "fp8-cast")
        self.assertEqual(mode, _OffloadMode.CPU)

    def test_unknown_value_is_rejected(self):
        import model_loader

        with patch.dict("os.environ", {"LTX_OFFLOAD_MODE": "gpu-please"}):
            with self.assertRaises(RuntimeError) as ctx:
                model_loader._resolve_offload_mode(_OffloadMode, "fp8-cast")
        self.assertIn("LTX_OFFLOAD_MODE", str(ctx.exception))


class TestStreamingQuantizationGuard(unittest.TestCase):
    """
    `StreamingModelBuilder` only accepts bf16 and fp8-cast fuse rules
    (ltx_pipelines.utils.blocks._build_streaming_builder raises otherwise), so
    an offload + quantization pairing that cannot work is rejected at load time
    with an actionable message instead of deep inside a model build.
    """

    def _resolve(self, offload: str, quant: str):
        import model_loader

        with patch.object(model_loader, "_total_vram_gib", return_value=23.6), \
             patch.object(model_loader, "_total_host_ram_gib", return_value=128.0), \
             patch.dict("os.environ", {"LTX_OFFLOAD_MODE": offload}):
            return model_loader._resolve_offload_mode(_OffloadMode, quant)

    def test_fp8_cast_streams_fine(self):
        self.assertEqual(self._resolve("cpu", "fp8-cast"), _OffloadMode.CPU)

    def test_bf16_streams_fine(self):
        self.assertEqual(self._resolve("cpu", "none"), _OffloadMode.CPU)

    def test_fp8_scaled_mm_is_rejected_when_streaming(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._resolve("cpu", "fp8-scaled-mm")
        self.assertIn("fp8-cast", str(ctx.exception))

    def test_nvfp4_is_rejected_when_streaming(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._resolve("disk", "nvfp4-prequant")
        self.assertIn("fp8-cast", str(ctx.exception))

    def test_incompatible_quantization_is_fine_without_offload(self):
        self.assertEqual(self._resolve("none", "fp8-scaled-mm"), _OffloadMode.NONE)

    def test_auto_on_small_card_rejects_incompatible_quantization(self):
        """auto must not silently pick a mode the quantization cannot support."""
        with self.assertRaises(RuntimeError):
            self._resolve("auto", "fp8-scaled-mm")


class TestVramProbesAreSafe(unittest.TestCase):
    """The probes feed logging and defaults; they must never raise."""

    def test_vram_probe_returns_a_float(self):
        import model_loader

        self.assertIsInstance(model_loader._total_vram_gib(), float)

    def test_host_ram_probe_returns_a_float(self):
        import model_loader

        self.assertIsInstance(model_loader._total_host_ram_gib(), float)

    def test_vram_probe_survives_a_broken_driver(self):
        import model_loader

        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(
                 torch.cuda, "get_device_properties", side_effect=RuntimeError("boom")
             ):
            self.assertEqual(model_loader._total_vram_gib(), 0.0)

    def test_log_vram_never_raises(self):
        import inference

        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(
                 torch.cuda, "memory_allocated", side_effect=RuntimeError("boom")
             ):
            inference._log_vram("in a test")  # must not propagate


if __name__ == "__main__":
    unittest.main(verbosity=2)
