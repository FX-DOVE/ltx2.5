"""
src/perf.py
─────────────────────────────────────────────────────────────────────────────
Per-phase timing and build-avoidance wrappers for the LTX-2.5 pipeline.

Why this exists
───────────────
A production 10-second 450p request took 268 s end to end and the handler
emitted exactly one number for it ("Denoising complete in …s"). That is not
something you can optimise against: `DistilledPipeline.__call__` performs at
least eight separate model builds, and upstream's design note at the top of
`ltx_pipelines/utils/blocks.py` is explicit — "Blocks build a model on each
`__call__`, use it, then free GPU memory." Every builder defaults to
`ModelRegistry(cache_models=True, cache_weights=False)`: shells are reused,
but *weights are re-read from the checkpoint on every build*. On RunPod the
checkpoints live on a network volume, so one request can re-read tens of
gigabytes before it does any arithmetic.

Whether that I/O or the denoising maths dominates decides which fix matters,
so this module measures rather than guesses.

What it does
────────────
`instrument_pipeline()` swaps each block attribute on the pipeline for a
transparent proxy that records wall time per call and forwards everything else,
so the pipeline cannot tell the difference. Two blocks need special handling:

  • `video_decoder` returns a *lazy* generator. The call itself is nearly free
    and the real cost lands later, while `encode_video` pulls chunks, so the
    iterator is wrapped and the drain is timed under a `…:drain` label.

  • `image_conditioner` builds the VAE encoder unconditionally — even for
    text-to-video, where `combined_image_conditionings` receives `images=[]`,
    never touches the encoder, and returns `[]`. `DistilledPipeline` calls it
    once per stage, so a plain text-to-video request pays for two complete
    encoder builds it cannot use. The proxy hands the callable a lazy encoder
    that materialises on first touch, which removes both builds when no
    conditioning image was supplied and behaves identically when one was.

Timings are wall clock, with an optional `torch.cuda.synchronize()` at each
boundary. Without it, work queued by one block is charged to whichever later
block hits a synchronisation point; `AllocatorTrimStrategy.TRIM` already syncs
on every block exit, so the added cost is one sync per block — about eight per
request.

Single request at a time
────────────────────────
The ledger is a module-level singleton behind a lock. A RunPod Serverless
worker handles one job at a time unless a `concurrency_modifier` is set, and
this repo does not set one. Give each request its own ledger if that changes.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Iterator

import torch
from loguru import logger

# Block attributes on DistilledPipeline, in the order __call__ reaches them.
# `duration_predictor` holds its head directly instead of rebuilding (upstream:
# "a few MB, so there's no memory pressure motivating the build-on-call
# pattern"), but it is timed anyway so the ledger accounts for all of __call__.
_BLOCK_ATTRS = (
    "prompt_encoder",
    "image_conditioner",
    "stage",
    "upsampler",
    "video_decoder",
    "audio_decoder",
    "duration_predictor",
)

_FALSEY = {"0", "false", "no", "off", ""}


def _flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in _FALSEY


def timing_enabled() -> bool:
    """`LTX_PERF_TIMING=0` silences per-phase timing."""
    return _flag("LTX_PERF_TIMING")


def lazy_encoder_enabled() -> bool:
    """`LTX_LAZY_IMAGE_ENCODER=0` restores upstream's eager encoder build."""
    return _flag("LTX_LAZY_IMAGE_ENCODER")


def _sync() -> None:
    """Drain the CUDA queue so a phase is charged for its own kernels."""
    if not _flag("LTX_PERF_SYNC") or not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:  # diagnostics must never break a request
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Ledger
# ─────────────────────────────────────────────────────────────────────────────


class PhaseLedger:
    """
    Wall-clock totals and call counts per phase, kept in first-call order.

    A phase may be entered more than once — `stage` runs twice (low-res then
    refine) and `image_conditioner` twice — so both the total and the count are
    reported, and the mean is what tells you whether the second visit is as
    expensive as the first.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._order: list[str] = []
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def record(self, label: str, seconds: float) -> None:
        with self._lock:
            if label not in self._totals:
                self._order.append(label)
                self._totals[label] = 0.0
                self._counts[label] = 0
            self._totals[label] += seconds
            self._counts[label] += 1

    def reset(self) -> None:
        with self._lock:
            self._order.clear()
            self._totals.clear()
            self._counts.clear()

    def snapshot(self) -> list[tuple[str, int, float]]:
        with self._lock:
            return [(label, self._counts[label], self._totals[label]) for label in self._order]

    def total(self) -> float:
        with self._lock:
            return sum(self._totals.values())

    def report(self, wall_seconds: float | None = None) -> None:
        """
        Log the breakdown, widest column aligned, plus the unaccounted balance.

        `wall_seconds` is the caller's own measurement of the whole request.
        The gap between it and the sum of the phases is not noise to hide — it
        is everything outside the blocks (latent init, tiling resolution, the
        guider, host-side bookkeeping), and if it is large that is the finding.
        """
        rows = self.snapshot()
        if not rows:
            return
        accounted = sum(seconds for _label, _count, seconds in rows)
        width = max(len(label) for label, _count, _seconds in rows)
        logger.info("[perf] ── phase breakdown ──────────────────────────────")
        for label, count, seconds in rows:
            share = f" | {seconds / wall_seconds * 100:5.1f}% of wall" if wall_seconds else ""
            mean = f" | mean {seconds / count:7.2f}s" if count > 1 else ""
            logger.info(f"[perf]   {label:<{width}}  {count}x {seconds:8.2f}s{mean}{share}")
        if wall_seconds:
            logger.info(
                f"[perf]   {'accounted':<{width}}  -- {accounted:8.2f}s of "
                f"{wall_seconds:.2f}s wall | unaccounted {wall_seconds - accounted:.2f}s"
            )
        else:
            logger.info(f"[perf]   {'accounted':<{width}}  -- {accounted:8.2f}s")


LEDGER = PhaseLedger()


# ─────────────────────────────────────────────────────────────────────────────
# Timing proxy
# ─────────────────────────────────────────────────────────────────────────────


def _timed_drain(iterator: Iterator[Any], label: str, ledger: PhaseLedger) -> Iterator[Any]:
    """
    Time each pull from a lazy iterator and charge it to `label`.

    Yields outside the timed region so the consumer's own work — libx264
    encoding, in this pipeline's case — is not billed to the decoder.
    """
    source = iter(iterator)
    while True:
        started = time.monotonic()
        try:
            item = next(source)
        except StopIteration:
            return
        _sync()
        ledger.record(label, time.monotonic() - started)
        yield item


class _TimedBlock:
    """
    Transparent timing proxy around one pipeline block.

    `DistilledPipeline.__call__` does more than call these objects: it reads
    `self.image_conditioner.resolve_crf`, `self.video_decoder.checkpoint_path`
    and `self.video_decoder.diffvae_optimization`, so the proxy has to forward
    attribute access as well as `__call__`.
    """

    __slots__ = ("_target", "_label", "_ledger")

    def __init__(self, target: Any, label: str, ledger: PhaseLedger) -> None:
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_label", label)
        object.__setattr__(self, "_ledger", ledger)

    # -- forwarding ---------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_target"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_target"), name, value)

    def __repr__(self) -> str:
        return f"<timed {object.__getattribute__(self, '_label')} {object.__getattribute__(self, '_target')!r}>"

    # -- the measurement ---------------------------------------------------
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        target = object.__getattribute__(self, "_target")
        label = object.__getattribute__(self, "_label")
        ledger = object.__getattribute__(self, "_ledger")

        started = time.monotonic()
        result = target(*args, **kwargs)
        if _is_lazy(result):
            # Nothing has been computed yet; bill the drain, not the call.
            ledger.record(f"{label}:setup", time.monotonic() - started)
            return _timed_drain(result, f"{label}:drain", ledger)
        _sync()
        ledger.record(label, time.monotonic() - started)
        return result


def _is_lazy(value: Any) -> bool:
    """A generator or bare iterator whose cost has not been paid yet."""
    if isinstance(value, (list, tuple, dict, str, bytes, torch.Tensor)):
        return False
    return hasattr(value, "__next__")


# ─────────────────────────────────────────────────────────────────────────────
# Lazy VAE encoder for the image conditioner
# ─────────────────────────────────────────────────────────────────────────────


class _LazyEncoder:
    """
    Stand-in for the VAE encoder that builds the real thing on first touch.

    `combined_image_conditionings` only ever uses the encoder as
    `video_encoder(image)`, inside `for img in images:` — so on a text-to-video
    request the loop never runs and this object is never touched. Attribute
    access is forwarded for anything else that might reach for it.

    Lifecycle is upstream's own `gpu_model()` context, entered late and closed
    on the way out, rather than a reimplementation of its dispose/trim rules.
    """

    __slots__ = ("_build", "_trim", "_stack", "_encoder")

    def __init__(self, build: Callable[[], Any], trim: Any) -> None:
        self._build = build
        self._trim = trim
        self._stack: Any = None
        self._encoder: Any = None

    def _materialise(self) -> Any:
        if self._encoder is None:
            from contextlib import ExitStack

            from ltx_pipelines.utils.gpu_model import gpu_model

            self._stack = ExitStack()
            self._encoder = self._stack.enter_context(
                gpu_model(self._build(), alloc_trim_strategy=self._trim)
            )
            logger.info("[perf] image conditioner: VAE encoder built on demand")
        return self._encoder

    def built(self) -> bool:
        return self._encoder is not None

    def release(self) -> None:
        if self._stack is not None:
            self._stack.close()
            self._stack = None
            self._encoder = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._materialise()(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._materialise(), name)


class _LazyImageConditioner:
    """
    `ImageConditioner` that does not build the encoder unless it is used.

    Upstream builds it unconditionally, and `DistilledPipeline` calls the block
    once per stage, so a text-to-video request pays for two full encoder builds
    whose result is discarded. Falls back to the untouched block if the private
    build hook it needs is not where this expects it.
    """

    __slots__ = ("_target",)

    def __init__(self, target: Any) -> None:
        object.__setattr__(self, "_target", target)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_target"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_target"), name, value)

    def __call__(self, fn: Callable[[Any], Any]) -> Any:
        target = object.__getattribute__(self, "_target")
        build = getattr(target, "_build_encoder", None)
        if build is None:  # upstream moved it — take the eager path unchanged
            return target(fn)
        encoder = _LazyEncoder(build, getattr(target, "_alloc_trim_strategy", None))
        try:
            return fn(encoder)
        finally:
            if not encoder.built():
                logger.info("[perf] image conditioner: no conditioning image, encoder build skipped")
            encoder.release()


# ─────────────────────────────────────────────────────────────────────────────
# Entry points
# ─────────────────────────────────────────────────────────────────────────────


def instrument_pipeline(pipeline: Any, ledger: PhaseLedger | None = None) -> Any:
    """
    Wrap a `DistilledPipeline`'s blocks in place; returns the same pipeline.

    Idempotent, so calling it twice on a cached pipeline is harmless. Blocks
    that a checkpoint does not carry (`duration_predictor` is `None` on
    checkpoints predating DurationHead) are skipped.
    """
    if getattr(pipeline, "_ltx_instrumented", False):
        return pipeline

    ledger = ledger or LEDGER
    lazy = lazy_encoder_enabled()
    timing = timing_enabled()
    wrapped: list[str] = []

    for name in _BLOCK_ATTRS:
        block = getattr(pipeline, name, None)
        if block is None:
            continue
        replaced = False
        if name == "image_conditioner" and lazy:
            block = _LazyImageConditioner(block)
            replaced = True
        if timing:
            block = _TimedBlock(block, name, ledger)
            replaced = True
        if not replaced:
            continue
        setattr(pipeline, name, block)
        wrapped.append(name)

    pipeline._ltx_instrumented = True
    logger.info(
        f"[perf] instrumented: {', '.join(wrapped) or 'nothing'} "
        f"(timing={'on' if timing else 'off'}, lazy_image_encoder={'on' if lazy else 'off'})"
    )
    return pipeline


class time_phase:
    """
    Context manager for timing a phase that is not a pipeline block.

    Used for the libx264 encode, which happens in `media_io.encode_video` and
    drives the VAE decode as a side effect.
    """

    __slots__ = ("_label", "_ledger", "_started")

    def __init__(self, label: str, ledger: PhaseLedger | None = None) -> None:
        self._label = label
        self._ledger = ledger or LEDGER

    def __enter__(self) -> "time_phase":
        self._started = time.monotonic()
        return self

    def __exit__(self, *exc: Any) -> None:
        _sync()
        self._ledger.record(self._label, time.monotonic() - self._started)

