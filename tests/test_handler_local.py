#!/usr/bin/env python3
"""
tests/test_handler_local.py
─────────────────────────────────────────────────────────────────────────────
Local test harness for the LTX-2.5 serverless handler.

Two modes:
  1. UNIT MODE (default, no GPU needed):
     Mocks model_loader and inference so you can validate the handler's
     request-routing, validation, error handling, and response shape
     without any GPU or network calls.

  2. INTEGRATION MODE (GPU + network volume required):
     Exercises the real pipeline end-to-end.  Run with --integration.
     Expects RUNPOD_VOLUME_PATH (or /runpod-volume) to be mounted and
     HF_TOKEN to be set.

Usage:
  # Unit tests (CI-safe, no GPU):
  python tests/test_handler_local.py

  # Full integration test (requires GPU + volume):
  python tests/test_handler_local.py --integration

  # Alternatively, use pytest (which auto-discovers):
  pytest tests/test_handler_local.py -v
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

# Allow running from repo root without installing as a package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_job(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Wrap an input dict in a Runpod-style job envelope."""
    return {"id": "test-job-001", "input": input_dict}


def _make_result(
    num_frames: int = 9,
    width: int = 768,
    height: int = 448,
    seed: int = 42,
    audio: Any = None,
) -> SimpleNamespace:
    """
    Stand-in for inference.InferenceResult.

    run_inference() returns the pipeline's LAZY chunk iterator plus the audio
    track, frame count and tiling config — not a materialised frame array — so
    the mock mirrors that shape.  cleanup() records that the handler released
    the conditioning temp dir.
    """
    result = SimpleNamespace(
        video=iter(()),          # encoder is mocked, so no chunks are needed
        audio=audio,
        num_frames=num_frames,
        tiling_config=None,
        seed=seed,
        width=width,
        height=height,
        cleaned_up=False,
    )
    def _cleanup() -> None:
        result.cleaned_up = True
    result.cleanup = _cleanup
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Unit Tests  (no GPU)
# ─────────────────────────────────────────────────────────────────────────────

class TestHandlerUnit(unittest.TestCase):
    """
    Handler logic tests using mocked model_loader and inference.

    These run in any environment — no GPU, no network volume, no HF token.
    """

    def _get_handler(self):
        """Import handler with all heavy dependencies mocked out."""
        mock_pipeline = MagicMock()

        with patch.dict("sys.modules", {
            "torch": MagicMock(),
            "loguru": MagicMock(logger=MagicMock(
                info=print, debug=print, warning=print,
                error=print, exception=print, critical=print,
            )),
        }):
            # Patch model_loader so the cold-start block doesn't actually run.
            mock_model_loader = MagicMock()
            mock_model_loader.ensure_weights_present = MagicMock(return_value=None)
            mock_model_loader.load_pipeline = MagicMock(return_value=mock_pipeline)

            # Patch inference so we return a fake InferenceResult.
            mock_inference = MagicMock()
            mock_inference.run_inference = MagicMock(return_value=_make_result())

            with patch.dict("sys.modules", {
                "model_loader": mock_model_loader,
                "inference": mock_inference,
                "runpod": MagicMock(),
            }):
                # Force re-import of handler so our mocks take effect.
                if "handler" in sys.modules:
                    del sys.modules["handler"]

                import handler as h

                # Patch internals that touch I/O.  _encode_video is where the
                # real code calls ltx_pipelines' PyAV encoder, which needs the
                # actual GPU-decoded chunks.
                h._PIPELINE = mock_pipeline
                h._encode_video = MagicMock(return_value=b"FAKEVIDEO" * 10)
                h._upload_video = MagicMock(return_value="https://example.com/video.mp4")

                return h

    # ── Basic text2video ──────────────────────────────────────────────────────

    def test_text2video_success(self):
        """Happy path: text2video with minimal input returns success response."""
        h = self._get_handler()
        job = _make_job({"prompt": "a scenic mountain sunrise timelapse"})

        result = h.handler(job)

        self.assertEqual(result["status"], "success")
        self.assertIn("generation_time_seconds", result)
        self.assertIn("num_frames", result)

    # ── Validation errors ─────────────────────────────────────────────────────

    def test_missing_prompt_returns_validation_error(self):
        """Handler should return validation_error when prompt is absent."""
        h = self._get_handler()
        job = _make_job({})  # no prompt

        result = h.handler(job)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"], "validation_error")
        self.assertFalse(result["retryable"])

    def test_empty_prompt_returns_validation_error(self):
        """Empty string prompt should fail Pydantic min_length=1."""
        h = self._get_handler()
        job = _make_job({"prompt": ""})

        result = h.handler(job)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"], "validation_error")

    def test_invalid_num_frames_returns_validation_error(self):
        """num_frames=10 is invalid (10-1=9, not divisible by 8)."""
        h = self._get_handler()
        job = _make_job({"prompt": "test", "num_frames": 10})

        result = h.handler(job)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"], "validation_error")

    def test_valid_num_frames_accepted(self):
        """num_frames=97 is valid (97-1=96, divisible by 8)."""
        h = self._get_handler()
        job = _make_job({"prompt": "test", "num_frames": 97})

        result = h.handler(job)
        self.assertEqual(result["status"], "success")

    def test_image2video_missing_image_returns_error(self):
        """image2video mode without first_frame_image should return validation_error."""
        h = self._get_handler()
        job = _make_job({"prompt": "test", "mode": "image2video"})

        result = h.handler(job)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"], "validation_error")

    def test_flf2video_missing_last_frame_returns_error(self):
        """flf2video without last_frame_image should return validation_error."""
        h = self._get_handler()
        job = _make_job({
            "prompt": "test",
            "mode": "flf2video",
            "first_frame_image": "data:image/png;base64,abc123",
            # last_frame_image intentionally absent
        })

        result = h.handler(job)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"], "validation_error")

    # ── OOM handling ──────────────────────────────────────────────────────────

    def test_oom_error_is_caught_and_returned_cleanly(self):
        """OOM during inference should return a clear error, not a raw exception."""
        h = self._get_handler()

        # handler.py does: `import inference as inference_module`
        # At test time, inference_module is the MagicMock we injected into
        # sys.modules["inference"].  We override run_inference on that mock
        # so the handler's call to `inference_module.run_inference(...)` raises.
        h.inference_module.run_inference.side_effect = RuntimeError("out_of_memory")

        job = _make_job({"prompt": "test"})
        result = h.handler(job)

        # Reset so subsequent tests aren't affected
        h.inference_module.run_inference.side_effect = None
        h.inference_module.run_inference.return_value = _make_result()

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"], "out_of_memory")
        self.assertFalse(result["retryable"])

    # ── Result lifecycle ──────────────────────────────────────────────────────

    def test_conditioning_tempdir_is_released(self):
        """
        The frame iterator is lazy, so the conditioning temp dir must survive
        until encoding finishes — and must be released once it has.
        """
        h = self._get_handler()
        fake = _make_result()
        h.inference_module.run_inference.return_value = fake

        h.handler(_make_job({"prompt": "test"}))

        self.assertTrue(fake.cleaned_up, "InferenceResult.cleanup() was never called")

    def test_cleanup_runs_even_when_encoding_fails(self):
        """A failed encode must not leak the conditioning temp dir."""
        h = self._get_handler()
        fake = _make_result()
        h.inference_module.run_inference.return_value = fake
        h._encode_video = MagicMock(side_effect=RuntimeError("libx264 exploded"))

        result = h.handler(_make_job({"prompt": "test"}))

        self.assertEqual(result["error"], "encoding_error")
        self.assertTrue(fake.cleaned_up, "cleanup() skipped on the failure path")

    def test_reported_frame_count_comes_from_the_pipeline(self):
        """
        The duration head can pick a different frame count than the request, so
        the response must echo what was generated, not what was asked for.
        """
        h = self._get_handler()
        h.inference_module.run_inference.return_value = _make_result(num_frames=121)

        result = h.handler(_make_job({"prompt": "test", "num_frames": 241}))

        self.assertEqual(result["num_frames"], 121)
        self.assertAlmostEqual(result["duration_seconds"], round(121 / 24, 2))

    def test_large_video_is_uploaded_not_base64(self):
        """Outputs above the base64 threshold go to object storage."""
        h = self._get_handler()
        h._encode_video = MagicMock(return_value=b"\x00" * (h._MAX_BASE64_BYTES + 1))

        result = h.handler(_make_job({"prompt": "test"}))

        self.assertEqual(result["video_url"], "https://example.com/video.mp4")
        self.assertNotIn("video_base64", result)

    def test_upload_failure_falls_back_to_base64(self):
        """A storage outage should degrade to base64 rather than fail the job."""
        h = self._get_handler()
        h._encode_video = MagicMock(return_value=b"\x00" * (h._MAX_BASE64_BYTES + 1))
        h._upload_video = MagicMock(side_effect=RuntimeError("no bucket configured"))

        result = h.handler(_make_job({"prompt": "test"}))

        self.assertEqual(result["status"], "success")
        self.assertIn("video_base64", result)
        self.assertNotIn("video_url", result)

    # ── Delivery mode (LTX_UPLOAD_MODE) ───────────────────────────────────────

    def test_small_video_is_inlined_under_auto(self):
        """`auto` is size-driven, so a sub-threshold clip must not be uploaded.

        This is why the R2 path looked broken: at 450p a 10-second clip encodes
        to under a megabyte, so `auto` never reached the uploader.
        """
        h = self._get_handler()
        h._encode_video = MagicMock(return_value=b"\x00" * 1024)
        h._upload_video = MagicMock(return_value="https://example.com/video.mp4")

        result = h.handler(_make_job({"prompt": "test"}))

        self.assertNotIn("video_url", result)
        self.assertIn("video_base64", result)
        h._upload_video.assert_not_called()

    def test_always_uploads_even_a_tiny_video(self):
        """`always` must reach storage regardless of size."""
        h = self._get_handler()
        h._UPLOAD_MODE = "always"
        h._encode_video = MagicMock(return_value=b"\x00" * 1024)
        h._upload_video = MagicMock(return_value="https://example.com/video.mp4")

        result = h.handler(_make_job({"prompt": "test"}))

        self.assertEqual(result["video_url"], "https://example.com/video.mp4")
        self.assertNotIn("video_base64", result)
        h._upload_video.assert_called_once()

    def test_never_inlines_even_a_large_video(self):
        """`never` must skip storage even above the threshold."""
        h = self._get_handler()
        h._UPLOAD_MODE = "never"
        h._encode_video = MagicMock(return_value=b"\x00" * (h._MAX_BASE64_BYTES + 1))
        h._upload_video = MagicMock(return_value="https://example.com/video.mp4")

        result = h.handler(_make_job({"prompt": "test"}))

        self.assertIn("video_base64", result)
        self.assertNotIn("video_url", result)
        h._upload_video.assert_not_called()

    def test_always_still_falls_back_when_storage_fails(self):
        """A forced upload must not lose a video we already paid GPU time for."""
        h = self._get_handler()
        h._UPLOAD_MODE = "always"
        h._encode_video = MagicMock(return_value=b"\x00" * 1024)
        h._upload_video = MagicMock(side_effect=RuntimeError("R2 unreachable"))

        result = h.handler(_make_job({"prompt": "test"}))

        self.assertEqual(result["status"], "success")
        self.assertIn("video_base64", result)
        self.assertNotIn("video_url", result)

    def test_response_reports_the_encoded_size(self):
        """`size_bytes` is what makes an `auto` delivery decision auditable."""
        h = self._get_handler()
        h._encode_video = MagicMock(return_value=b"\x00" * 4242)

        result = h.handler(_make_job({"prompt": "test"}))

        self.assertEqual(result["size_bytes"], 4242)

    def test_threshold_is_read_from_the_environment(self):
        """LTX_MAX_BASE64_MB has to move the auto cutoff, or it is not tunable."""
        with patch.dict(os.environ, {"LTX_MAX_BASE64_MB": "0.001"}):
            h = self._get_handler()
            self.assertEqual(h._MAX_BASE64_BYTES, int(0.001 * 1024 * 1024))
            h._encode_video = MagicMock(return_value=b"\x00" * 4096)
            h._upload_video = MagicMock(return_value="https://example.com/video.mp4")

            result = h.handler(_make_job({"prompt": "test"}))

        self.assertEqual(result["video_url"], "https://example.com/video.mp4")

    def test_an_unknown_upload_mode_degrades_to_auto(self):
        """A typo in the endpoint env must not silently disable delivery."""
        with patch.dict(os.environ, {"LTX_UPLOAD_MODE": "s3"}):
            h = self._get_handler()

        self.assertEqual(h._UPLOAD_MODE, "auto")

    def test_has_audio_reflects_the_generated_waveform(self):
        """LTX-2.5 emits audio alongside video; the response must report it."""
        h = self._get_handler()

        silent = _make_result(audio=None)
        h.inference_module.run_inference.return_value = silent
        self.assertFalse(h.handler(_make_job({"prompt": "test"}))["has_audio"])

        with_audio = _make_result(
            audio=SimpleNamespace(waveform=SimpleNamespace(numel=lambda: 48000),
                                  sampling_rate=48000)
        )
        h.inference_module.run_inference.return_value = with_audio
        self.assertTrue(h.handler(_make_job({"prompt": "test"}))["has_audio"])

    # ── Response shape ────────────────────────────────────────────────────────

    def test_response_includes_expected_fields(self):
        """Successful response must include all documented fields."""
        h = self._get_handler()
        job = _make_job({"prompt": "a colorful abstract animation"})

        result = h.handler(job)

        required_fields = {
            "status", "duration_seconds", "generation_time_seconds",
            "seed_used", "resolution", "num_frames", "fps", "has_audio",
        }
        self.assertTrue(required_fields.issubset(result.keys()),
                        f"Missing fields: {required_fields - result.keys()}")

    def test_response_has_video_url_or_base64(self):
        """Response must include at least one of video_url or video_base64."""
        h = self._get_handler()
        job = _make_job({"prompt": "test"})

        result = h.handler(job)

        has_video = ("video_url" in result) or ("video_base64" in result)
        self.assertTrue(has_video, "Response has neither video_url nor video_base64")

    def test_resolution_field_format(self):
        """resolution field should be in 'WxH' format."""
        h = self._get_handler()
        job = _make_job({"prompt": "test", "resolution": "480p"})

        result = h.handler(job)

        self.assertRegex(result.get("resolution", ""), r"^\d+x\d+$")


# ─────────────────────────────────────────────────────────────────────────────
# Schema / Pydantic Tests (no GPU)
# ─────────────────────────────────────────────────────────────────────────────

class TestSchema(unittest.TestCase):
    """Test Pydantic schema validation directly, bypassing the handler."""

    def setUp(self):
        from schema import InferenceInput
        self.InferenceInput = InferenceInput

    def test_minimal_valid_input(self):
        obj = self.InferenceInput(prompt="test prompt")
        self.assertEqual(obj.mode.value, "text2video")
        self.assertEqual(obj.resolution.value, "450p")
        self.assertEqual(obj.num_frames, 241)
        self.assertEqual(obj.fps, 24)

    def test_seed_accepted(self):
        obj = self.InferenceInput(prompt="test", seed=12345)
        self.assertEqual(obj.seed, 12345)

    def test_invalid_resolution_rejected(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            self.InferenceInput(prompt="test", resolution="4K")

    def test_invalid_mode_rejected(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            self.InferenceInput(prompt="test", mode="video2text")

    def test_guidance_scale_bounds(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            self.InferenceInput(prompt="test", guidance_scale=0.5)  # below 1.0
        with self.assertRaises(ValidationError):
            self.InferenceInput(prompt="test", guidance_scale=11.0)  # above 10.0

    def test_all_valid_num_frames(self):
        """All standard LTX-2.5 frame counts should pass validation."""
        valid_frames = [9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 121, 145, 161, 193, 225, 257]
        for n in valid_frames:
            with self.subTest(num_frames=n):
                obj = self.InferenceInput(prompt="test", num_frames=n)
                self.assertEqual(obj.num_frames, n)

    # ── Dynamic mode auto-detection ───────────────────────────────────────────

    def test_auto_promote_to_image2video(self):
        """Providing only first_frame_image without mode should auto-promote to image2video."""
        obj = self.InferenceInput(
            prompt="test",
            first_frame_image="https://example.com/frame.jpg",
            # mode NOT specified — should be auto-promoted
        )
        self.assertEqual(obj.mode.value, "image2video")

    def test_auto_promote_to_flf2video(self):
        """Providing both frames without mode should auto-promote to flf2video."""
        obj = self.InferenceInput(
            prompt="test",
            first_frame_image="https://example.com/first.jpg",
            last_frame_image="https://example.com/last.jpg",
            # mode NOT specified — should be auto-promoted
        )
        self.assertEqual(obj.mode.value, "flf2video")

    def test_explicit_text2video_ignores_no_images(self):
        """Explicit text2video with no images remains text2video (no auto-promote)."""
        obj = self.InferenceInput(prompt="test", mode="text2video")
        self.assertEqual(obj.mode.value, "text2video")

    def test_explicit_mode_image2video_missing_image_fails(self):
        """Explicit image2video without first_frame_image should still raise."""
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            self.InferenceInput(prompt="test", mode="image2video")
            # no first_frame_image provided

    def test_explicit_mode_flf2video_missing_last_frame_fails(self):
        """Explicit flf2video without last_frame_image should raise."""
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            self.InferenceInput(
                prompt="test",
                mode="flf2video",
                first_frame_image="https://example.com/first.jpg",
                # last_frame_image intentionally absent
            )


# ─────────────────────────────────────────────────────────────────────────────
# Integration test (real GPU + volume)
# ─────────────────────────────────────────────────────────────────────────────

class TestHandlerIntegration(unittest.TestCase):
    """
    Full end-to-end test.  Only runs when --integration flag is passed.
    Requires:
      • A CUDA GPU with ≥16 GB VRAM
      • HF_TOKEN set in environment (with gated-repo access to LTX-Video)
      • RUNPOD_VOLUME_PATH pointing to a writable directory with ≥100 GB free
        (or /runpod-volume mounted by default)
    """

    @classmethod
    def setUpClass(cls):
        """Skip all tests in this class unless --integration was requested."""
        if not getattr(cls, "_run_integration", False):
            raise unittest.SkipTest("Integration tests skipped (pass --integration to enable)")

    def test_full_text2video_pipeline(self):
        """Run a minimal text2video job end-to-end."""
        # Import after cold-start (handler.py runs ensure_weights + load_pipeline).
        import handler as h

        job = _make_job({
            "prompt": "a single red sphere rolling across a white table",
            "resolution": "480p",   # 896×512 — %64, legal for the two-stage pipeline
            "num_frames": 9,        # minimum frames — keeps the test fast
            # NOTE: num_inference_steps / guidance_scale are deliberately omitted.
            # The distilled checkpoint has a baked-in 8+3 step sigma schedule and
            # no CFG branch, so those fields are accepted but have no effect.
        })

        t0 = time.monotonic()
        result = h.handler(job)
        elapsed = time.monotonic() - t0

        print(f"\n[integration] text2video completed in {elapsed:.1f}s")
        print(f"[integration] result keys: {list(result.keys())}")

        self.assertEqual(result["status"], "success")
        self.assertIn("generation_time_seconds", result)
        self.assertTrue(
            ("video_url" in result) or ("video_base64" in result),
            "Expected video_url or video_base64 in response"
        )

        if "video_base64" in result:
            video_bytes = base64.b64decode(result["video_base64"])
            self.assertGreater(len(video_bytes), 1000, "Video is suspiciously small")
            print(f"[integration] Video size: {len(video_bytes)/1024:.1f} KB (base64)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LTX-2.5 handler test suite")
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run full integration tests (requires GPU + network volume).",
    )
    args, remaining = parser.parse_known_args()

    if args.integration:
        print("[integration] Integration mode enabled - running full pipeline tests.")
        TestHandlerIntegration._run_integration = True
    else:
        print("[unit] Unit mode - mocking GPU and model dependencies.")

    # Hand off to unittest runner, forwarding remaining args (e.g. -v).
    unittest.main(argv=[sys.argv[0]] + remaining, verbosity=2)


if __name__ == "__main__":
    main()
