#!/usr/bin/env python3
"""
tests/test_handler_local.py
─────────────────────────────────────────────────────────────────────────────
Local test harness for the LTX-2.5 serverless handler.

Runs unit tests without GPU/network dependencies.
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
from typing import Any
from unittest.mock import MagicMock, patch

# Allow running from repo root without installing as a package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))


def _make_job(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Wrap an input dict in a Runpod-style job envelope."""
    return {"id": "test-job-001", "input": input_dict}


def _make_black_frames(t: int = 9, h: int = 480, w: int = 704) -> Any:
    """Return a (T, H, W, 3) uint8 array of black frames for testing."""
    import numpy as np
    return (np.zeros((t, h, w, 3), dtype=np.uint8), 42)


class TestHandlerUnit(unittest.TestCase):
    """Handler logic unit tests using mocked model_loader and inference."""

    def _get_handler(self):
        import numpy as np

        mock_pipeline = MagicMock()

        with patch.dict("sys.modules", {
            "torch": MagicMock(),
            "loguru": MagicMock(logger=MagicMock(
                info=lambda *a, **k: None,
                debug=lambda *a, **k: None,
                warning=lambda *a, **k: None,
                error=lambda *a, **k: None,
                exception=lambda *a, **k: None,
                critical=lambda *a, **k: None,
            )),
        }):
            mock_model_loader = MagicMock()
            mock_model_loader.MODEL_ID = "Lightricks/LTX-2.5-Diffusers"
            mock_model_loader.DTYPE_STR = "bfloat16"
            mock_model_loader.VOLUME_ROOT = Path("/runpod-volume")
            mock_model_loader.ensure_weights_present = MagicMock(return_value=None)
            mock_model_loader.load_pipeline = MagicMock(return_value=mock_pipeline)

            mock_inference = MagicMock()
            mock_inference.run_inference = MagicMock(
                return_value=_make_black_frames()
            )

            with patch.dict("sys.modules", {
                "model_loader": mock_model_loader,
                "inference": mock_inference,
                "runpod": MagicMock(),
            }):
                if "handler" in sys.modules:
                    del sys.modules["handler"]

                import handler as h

                h._PIPELINE = mock_pipeline
                h._encode_video = MagicMock(return_value=b"FAKEVIDEO" * 10)
                h._upload_video = MagicMock(return_value="https://example.com/video.mp4")

                return h

    def test_text2video_success(self):
        h = self._get_handler()
        job = _make_job({"prompt": "a scenic mountain sunrise timelapse"})
        result = h.handler(job)

        self.assertEqual(result["status"], "success")
        self.assertIn("generation_time_seconds", result)
        self.assertIn("num_frames", result)

    def test_user_target_example(self):
        """Test user payload with explicit width, height, num_frames=241, fps=24."""
        h = self._get_handler()
        job = _make_job({
            "prompt": "A cinematic shot of a cute fox walking through a magical forest, realistic movement, high quality",
            "width": 704,
            "height": 480,
            "num_frames": 241,
            "fps": 24,
        })
        result = h.handler(job)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["resolution"], "704x480")
        self.assertEqual(result["num_frames"], 241)
        self.assertEqual(result["fps"], 24)
        self.assertAlmostEqual(result["duration_seconds"], 10.04, places=2)

    def test_missing_prompt_returns_validation_error(self):
        h = self._get_handler()
        job = _make_job({})
        result = h.handler(job)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"], "validation_error")

    def test_invalid_num_frames_returns_validation_error(self):
        h = self._get_handler()
        job = _make_job({"prompt": "test", "num_frames": 10})
        result = h.handler(job)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"], "validation_error")

    def test_invalid_dimension_returns_validation_error(self):
        h = self._get_handler()
        job = _make_job({"prompt": "test", "width": 700})  # 700 not divisible by 32
        result = h.handler(job)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"], "validation_error")

    def test_oom_error_is_caught_and_returned_cleanly(self):
        h = self._get_handler()
        h.inference_module.run_inference.side_effect = RuntimeError("out_of_memory")

        job = _make_job({"prompt": "test"})
        result = h.handler(job)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"], "out_of_memory")
        self.assertFalse(result["retryable"])


class TestSchema(unittest.TestCase):
    """Test Pydantic schema validation directly."""

    def setUp(self):
        from schema import InferenceInput
        self.InferenceInput = InferenceInput

    def test_minimal_valid_input(self):
        obj = self.InferenceInput(prompt="test prompt")
        self.assertEqual(obj.mode.value, "text2video")
        self.assertEqual(obj.width, 1280)
        self.assertEqual(obj.height, 720)
        self.assertEqual(obj.num_frames, 97)
        self.assertEqual(obj.fps, 24)

    def test_custom_dimensions(self):
        obj = self.InferenceInput(
            prompt="test",
            width=704,
            height=480,
            num_frames=241,
            fps=24,
        )
        self.assertEqual(obj.width, 704)
        self.assertEqual(obj.height, 480)
        self.assertEqual(obj.num_frames, 241)
        self.assertEqual(obj.fps, 24)

    def test_all_valid_num_frames(self):
        valid_frames = [9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 121, 145, 161, 193, 225, 241, 257]
        for n in valid_frames:
            with self.subTest(num_frames=n):
                obj = self.InferenceInput(prompt="test", num_frames=n)
                self.assertEqual(obj.num_frames, n)

    def test_image_alias(self):
        obj = self.InferenceInput(
            prompt="test",
            image="https://example.com/frame.jpg",
        )
        self.assertEqual(obj.mode.value, "image2video")
        self.assertEqual(obj.first_frame_image, "https://example.com/frame.jpg")


def main():
    parser = argparse.ArgumentParser(description="LTX-2.5 handler test suite")
    parser.add_argument("--integration", action="store_true")
    args, remaining = parser.parse_known_args()

    print("Running Unit Tests for LTX-2.5 Serverless Handler...")
    unittest.main(argv=[sys.argv[0]] + remaining, verbosity=2)


if __name__ == "__main__":
    main()
