"""Unit test: the Preprocessor denoises BEFORE text extraction / skew handling.

Requirement 6.2: WHILE processing a Scanned_Document_Image, THE Guardrail SHALL
apply image preprocessing that reduces noise before text extraction.

Two complementary checks:
    - Ordering (spy): monkeypatch ``Preprocessor._denoise`` and
      ``Preprocessor.estimate_skew_angle`` to record their invocation order, then
      assert the denoise step is recorded before the skew step. Since OCR runs in
      a separate component downstream of ``preprocess``, "before OCR" is verified
      by proving denoise happens before every in-``preprocess`` analysis step
      (skew estimation), which itself precedes OCR in the pipeline.
    - Metamorphic noise-metric check: add random noise to a smooth image, run the
      real ``_denoise``, and assert a noise metric (variance of the Laplacian, and
      std of pixel differences from the smooth baseline) decreases -- i.e. denoise
      actually reduces noise.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pii_guardrail.preprocessor import Preprocessor

pytestmark = pytest.mark.unit


class TestDenoiseRunsBeforeOCR:
    def test_denoise_recorded_before_skew_step(self, monkeypatch) -> None:
        """Denoise must be applied before the skew step (and thus before OCR)."""
        calls: list[str] = []

        real_denoise = Preprocessor._denoise

        def spy_denoise(image, *, upscaled=False):
            calls.append("denoise")
            return real_denoise(image, upscaled=upscaled)

        def spy_estimate_skew(self, image):
            calls.append("skew")
            # Return 0.0 so preprocess takes the deadband no-op path and does not
            # re-invoke skew estimation, keeping the recorded order unambiguous.
            return 0.0

        # _denoise is a staticmethod; patch with a plain function.
        monkeypatch.setattr(Preprocessor, "_denoise", staticmethod(spy_denoise))
        monkeypatch.setattr(Preprocessor, "estimate_skew_angle", spy_estimate_skew)

        image = np.full((64, 64, 3), 200, dtype=np.uint8)
        Preprocessor().preprocess(image)

        assert "denoise" in calls, "denoise step was never invoked"
        assert "skew" in calls, "skew step was never invoked"
        assert calls.index("denoise") < calls.index("skew"), (
            f"denoise must run before the skew/text step, got order: {calls}"
        )

    def test_denoise_reduces_noise_metric(self) -> None:
        """Metamorphic check: real _denoise lowers the image's noise level."""
        rng = np.random.default_rng(1234)

        # A smooth baseline: a gentle horizontal gradient, no high-freq content.
        h, w = 128, 128
        gradient = np.linspace(40, 210, w, dtype=np.float64)
        smooth = np.tile(gradient, (h, 1))
        smooth_u8 = smooth.astype(np.uint8)

        # Add strong random noise on top of the smooth baseline.
        noise = rng.normal(0.0, 25.0, size=(h, w))
        noisy_u8 = np.clip(smooth + noise, 0, 255).astype(np.uint8)

        denoised_u8 = Preprocessor._denoise(noisy_u8)

        # Metric 1: variance of the Laplacian (high-frequency energy / noise).
        def laplacian_var(img: np.ndarray) -> float:
            return float(cv2.Laplacian(img, cv2.CV_64F).var())

        noisy_lap = laplacian_var(noisy_u8)
        denoised_lap = laplacian_var(denoised_u8)
        assert denoised_lap < noisy_lap, (
            "denoise should reduce high-frequency energy: "
            f"noisy={noisy_lap:.2f} denoised={denoised_lap:.2f}"
        )

        # Metric 2: std of pixel differences vs the smooth baseline.
        noisy_dev = float(np.std(noisy_u8.astype(np.float64) - smooth))
        denoised_dev = float(
            np.std(denoised_u8.astype(np.float64) - smooth)
        )
        assert denoised_dev < noisy_dev, (
            "denoise should move pixels closer to the smooth baseline: "
            f"noisy_dev={noisy_dev:.2f} denoised_dev={denoised_dev:.2f}"
        )
