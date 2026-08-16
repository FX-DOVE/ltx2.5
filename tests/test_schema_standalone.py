"""
tests/test_schema_standalone.py
─────────────────────────────────────────────────────────────────────────────
Pure-Python validation of schema.py with zero GPU/torch dependencies.
Runs in any Python 3.11+ environment with only pydantic installed.

Covers:
  • All three generation modes (text2video / image2video / flf2video)
  • Dynamic mode auto-detection logic
  • Resolution enum
  • num_frames validator (LTX-2.5 constraint: (N-1) % 8 == 0)
  • guidance_scale / seed bounds
  • Negative prompt defaults
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import unittest
from pydantic import ValidationError
from schema import InferenceInput, GenerationMode, Resolution, RESOLUTION_MAP


class TestGenerationModes(unittest.TestCase):
    """All three modes parse and route correctly."""

    def test_text2video_default(self):
        obj = InferenceInput(prompt="a sunrise over mountains")
        self.assertEqual(obj.mode, GenerationMode.text2video)
        self.assertIsNone(obj.first_frame_image)
        self.assertIsNone(obj.last_frame_image)

    def test_text2video_explicit(self):
        obj = InferenceInput(prompt="test", mode="text2video")
        self.assertEqual(obj.mode, GenerationMode.text2video)

    def test_image2video_explicit(self):
        obj = InferenceInput(
            prompt="test",
            mode="image2video",
            first_frame_image="https://example.com/frame.jpg",
        )
        self.assertEqual(obj.mode, GenerationMode.image2video)

    def test_flf2video_explicit(self):
        obj = InferenceInput(
            prompt="test",
            mode="flf2video",
            first_frame_image="https://example.com/first.jpg",
            last_frame_image="https://example.com/last.jpg",
        )
        self.assertEqual(obj.mode, GenerationMode.flf2video)

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValidationError):
            InferenceInput(prompt="test", mode="video2text")


class TestDynamicModeAutoDetection(unittest.TestCase):
    """
    Calling code should be able to just pass images and let the schema
    figure out the mode automatically — no need to set mode explicitly.
    """

    def test_no_images_stays_text2video(self):
        obj = InferenceInput(prompt="test")
        self.assertEqual(obj.mode, GenerationMode.text2video)

    def test_first_frame_only_promotes_to_image2video(self):
        obj = InferenceInput(
            prompt="test",
            first_frame_image="https://example.com/frame.jpg",
        )
        self.assertEqual(obj.mode, GenerationMode.image2video,
                         "Expected auto-promotion to image2video when only first_frame_image given")

    def test_both_frames_promotes_to_flf2video(self):
        obj = InferenceInput(
            prompt="test",
            first_frame_image="https://example.com/first.jpg",
            last_frame_image="https://example.com/last.jpg",
        )
        self.assertEqual(obj.mode, GenerationMode.flf2video,
                         "Expected auto-promotion to flf2video when both frames given")

    def test_last_frame_without_first_stays_text2video(self):
        obj = InferenceInput(
            prompt="test",
            last_frame_image="https://example.com/last.jpg",
        )
        self.assertEqual(obj.mode, GenerationMode.text2video)

    def test_explicit_image2video_missing_image_raises(self):
        with self.assertRaises(ValidationError) as ctx:
            InferenceInput(prompt="test", mode="image2video")
        self.assertIn("first_frame_image", str(ctx.exception).lower())

    def test_explicit_flf2video_missing_last_frame_raises(self):
        with self.assertRaises(ValidationError) as ctx:
            InferenceInput(
                prompt="test",
                mode="flf2video",
                first_frame_image="https://example.com/first.jpg",
            )
        self.assertIn("last_frame_image", str(ctx.exception).lower())

    def test_explicit_flf2video_missing_first_frame_raises(self):
        with self.assertRaises(ValidationError) as ctx:
            InferenceInput(
                prompt="test",
                mode="flf2video",
                last_frame_image="https://example.com/last.jpg",
            )
        self.assertIn("first_frame_image", str(ctx.exception).lower())


class TestResolution(unittest.TestCase):
    """Resolution enum and pixel dimension map."""

    def test_default_resolution_is_720p(self):
        obj = InferenceInput(prompt="test")
        self.assertEqual(obj.resolution, Resolution.r720p)

    def test_480p_dimensions(self):
        self.assertEqual(RESOLUTION_MAP[Resolution.r480p], (848, 480))

    def test_720p_dimensions(self):
        self.assertEqual(RESOLUTION_MAP[Resolution.r720p], (1280, 720))

    def test_1080p_dimensions(self):
        self.assertEqual(RESOLUTION_MAP[Resolution.r1080p], (1920, 1080))

    def test_all_resolutions_accepted(self):
        for res in ("480p", "720p", "1080p"):
            with self.subTest(resolution=res):
                obj = InferenceInput(prompt="test", resolution=res)
                self.assertEqual(obj.resolution.value, res)

    def test_invalid_resolution_rejected(self):
        with self.assertRaises(ValidationError):
            InferenceInput(prompt="test", resolution="2160p")


class TestNumFrames(unittest.TestCase):
    """LTX-2.5 frame count constraint: (N-1) must be divisible by 8."""

    VALID = [9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97,
             121, 145, 161, 193, 225, 257]
    INVALID = [8, 10, 11, 16, 18, 24, 26, 98, 100, 256, 258]

    def test_all_valid_frame_counts(self):
        for n in self.VALID:
            with self.subTest(num_frames=n):
                obj = InferenceInput(prompt="test", num_frames=n)
                self.assertEqual(obj.num_frames, n)

    def test_invalid_frame_counts_rejected(self):
        for n in self.INVALID:
            with self.subTest(num_frames=n):
                with self.assertRaises(ValidationError,
                                       msg=f"Expected ValidationError for num_frames={n}"):
                    InferenceInput(prompt="test", num_frames=n)

    def test_below_minimum_rejected(self):
        with self.assertRaises(ValidationError):
            InferenceInput(prompt="test", num_frames=1)

    def test_above_maximum_rejected(self):
        with self.assertRaises(ValidationError):
            InferenceInput(prompt="test", num_frames=999)

    def test_default_is_97(self):
        obj = InferenceInput(prompt="test")
        self.assertEqual(obj.num_frames, 97)


class TestSamplingParams(unittest.TestCase):
    """guidance_scale, num_inference_steps, fps, seed."""

    def test_default_values(self):
        obj = InferenceInput(prompt="test")
        self.assertEqual(obj.guidance_scale, 3.5)
        self.assertEqual(obj.num_inference_steps, 40)
        self.assertEqual(obj.fps, 24)
        self.assertIsNone(obj.seed)

    def test_guidance_scale_lower_bound(self):
        with self.assertRaises(ValidationError):
            InferenceInput(prompt="test", guidance_scale=0.9)

    def test_guidance_scale_upper_bound(self):
        with self.assertRaises(ValidationError):
            InferenceInput(prompt="test", guidance_scale=10.1)

    def test_valid_guidance_scale(self):
        obj = InferenceInput(prompt="test", guidance_scale=7.5)
        self.assertAlmostEqual(obj.guidance_scale, 7.5)

    def test_seed_set_and_retrieved(self):
        obj = InferenceInput(prompt="test", seed=42)
        self.assertEqual(obj.seed, 42)

    def test_seed_none_is_default(self):
        obj = InferenceInput(prompt="test")
        self.assertIsNone(obj.seed)

    def test_seed_upper_bound(self):
        with self.assertRaises(ValidationError):
            InferenceInput(prompt="test", seed=2**32)

    def test_fps_bounds(self):
        with self.assertRaises(ValidationError):
            InferenceInput(prompt="test", fps=7)
        with self.assertRaises(ValidationError):
            InferenceInput(prompt="test", fps=31)

    def test_num_steps_bounds(self):
        with self.assertRaises(ValidationError):
            InferenceInput(prompt="test", num_inference_steps=9)
        with self.assertRaises(ValidationError):
            InferenceInput(prompt="test", num_inference_steps=101)


class TestPromptValidation(unittest.TestCase):
    """Prompt field constraints."""

    def test_prompt_required(self):
        with self.assertRaises(ValidationError):
            InferenceInput()  # type: ignore[call-arg]

    def test_empty_prompt_rejected(self):
        with self.assertRaises(ValidationError):
            InferenceInput(prompt="")

    def test_max_length_prompt(self):
        InferenceInput(prompt="a" * 2000)

    def test_over_max_length_rejected(self):
        with self.assertRaises(ValidationError):
            InferenceInput(prompt="a" * 2001)

    def test_negative_prompt_has_default(self):
        obj = InferenceInput(prompt="test")
        self.assertIn("low quality", obj.negative_prompt)

    def test_custom_negative_prompt(self):
        obj = InferenceInput(prompt="test", negative_prompt="blurry")
        self.assertEqual(obj.negative_prompt, "blurry")


class TestImageInputFormats(unittest.TestCase):
    """Images can be provided as base64 data URIs, raw base64, or HTTPS URLs."""

    def test_base64_data_uri_accepted(self):
        obj = InferenceInput(
            prompt="test",
            first_frame_image="data:image/png;base64,iVBORw0KGgo=",
        )
        self.assertEqual(obj.mode, GenerationMode.image2video)

    def test_raw_base64_string_accepted(self):
        obj = InferenceInput(
            prompt="test",
            first_frame_image="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ",
        )
        self.assertEqual(obj.mode, GenerationMode.image2video)

    def test_https_url_accepted(self):
        obj = InferenceInput(
            prompt="test",
            first_frame_image="https://cdn.example.com/image.jpg",
        )
        self.assertEqual(obj.mode, GenerationMode.image2video)

    def test_flf_with_mixed_formats(self):
        """First frame as URL, last frame as base64 — both valid."""
        obj = InferenceInput(
            prompt="test",
            first_frame_image="https://example.com/first.jpg",
            last_frame_image="data:image/jpeg;base64,/9j/4AAQ==",
        )
        self.assertEqual(obj.mode, GenerationMode.flf2video)


if __name__ == "__main__":
    unittest.main(verbosity=2)
