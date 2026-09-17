"""Unit tests: the Preprocessor auto-upscales low-resolution inputs (Option A).

When an input's estimated effective resolution is below the 150 DPI reliability
threshold, the Preprocessor enlarges it (up to ``MAX_UPSCALE_FACTOR``) so it can
still be OCR'd, rather than rejecting it. This trades a LARGER output image for
the ability to detect PII on low-resolution inputs (e.g. screenshots).

These example-based unit tests pin the behavior:
    - a low-resolution image is upscaled (upscaled=True, factor>1) and, when the
      upscale is enough to reach the threshold, quality_sufficient becomes True;
    - the output image is genuinely larger than the input;
    - an already-high-resolution image is NOT upscaled (upscaled=False);
    - the upscale factor is capped at ``MAX_UPSCALE_FACTOR``;
    - channel layout is preserved.
"""

from __future__ import annotations

import numpy as np
import pytest

from pii_guardrail.preprocessor import (
    ASSUMED_PAGE_HEIGHT_INCHES,
    DPI_RELIABILITY_THRESHOLD,
    MAX_UPSCALE_FACTOR,
    UPSCALE_TARGET_DPI,
    Preprocessor,
)

pytestmark = pytest.mark.unit

# Pixel height at exactly the reliability threshold (150 * 11 == 1650).
_THRESHOLD_HEIGHT_PX = int(DPI_RELIABILITY_THRESHOLD * ASSUMED_PAGE_HEIGHT_INCHES)


class TestPreprocessorAutoUpscale:
    def test_moderately_low_res_is_upscaled_to_sufficient(self) -> None:
        """A modestly-low-res image is enlarged until it clears the threshold.

        Height 1000 px -> estimated DPI ~90.9 (< 150). The needed factor is
        ~1.65 (< MAX_UPSCALE_FACTOR), so upscaling reaches the target DPI and
        quality_sufficient becomes True; the output is larger than the input.
        """
        h, w = 1000, 700
        image = np.full((h, w, 3), 255, dtype=np.uint8)

        result = Preprocessor().preprocess(image)

        assert result.upscaled is True
        assert result.upscale_factor > 1.0
        assert result.quality_sufficient is True
        # Output is genuinely larger than the input (Option A: bigger image out).
        assert result.image.shape[0] > h
        assert result.image.shape[1] > w
        # Reached (approximately) the reliability threshold height.
        assert result.image.shape[0] >= _THRESHOLD_HEIGHT_PX - 2
        # Estimated DPI reported is the POST-upscale value, at/above threshold.
        assert result.estimated_dpi is not None
        assert result.estimated_dpi >= DPI_RELIABILITY_THRESHOLD - 1.0
        # Channel layout preserved.
        assert result.image.ndim == 3 and result.image.shape[2] == 3

    def test_upscale_factor_is_capped(self) -> None:
        """A tiny image cannot exceed MAX_UPSCALE_FACTOR, so it stays insufficient.

        Height 100 px -> DPI ~9.1; reaching 150 DPI would need ~16.5x, far above
        the cap. The factor is clamped to MAX_UPSCALE_FACTOR and the image, still
        below threshold after the capped upscale, remains quality-insufficient.
        """
        h, w = 100, 200
        image = np.full((h, w, 3), 255, dtype=np.uint8)

        result = Preprocessor().preprocess(image)

        assert result.upscaled is True
        # Factor never exceeds the cap (allow tiny rounding slack).
        assert result.upscale_factor <= MAX_UPSCALE_FACTOR + 1e-6
        # Enlarged height is about the capped multiple of the original.
        assert result.image.shape[0] == pytest.approx(h * MAX_UPSCALE_FACTOR, abs=2)
        # Still below the threshold after the capped upscale -> insufficient.
        assert result.quality_sufficient is False

    def test_high_res_image_is_not_upscaled(self) -> None:
        """An image already at/above the threshold is left at its size."""
        h, w = _THRESHOLD_HEIGHT_PX + 200, 900  # comfortably >= 150 DPI
        image = np.full((h, w, 3), 255, dtype=np.uint8)

        result = Preprocessor().preprocess(image)

        assert result.upscaled is False
        assert result.upscale_factor == 1.0
        assert result.quality_sufficient is True
        # No enlargement: dimensions are unchanged by the (no-op) upscale step.
        assert result.image.shape[0] == h
        assert result.image.shape[1] == w

    def test_grayscale_low_res_is_upscaled_and_stays_2d(self) -> None:
        """A 2-D grayscale low-res image is upscaled and keeps its 2-D layout."""
        h, w = 1000, 700
        image = np.full((h, w), 255, dtype=np.uint8)

        result = Preprocessor().preprocess(image)

        assert result.upscaled is True
        assert result.image.ndim == 2  # grayscale stays single-channel/2-D
        assert result.image.shape[0] > h

    def test_maybe_upscale_helper_reaches_target_dpi(self) -> None:
        """``_maybe_upscale`` scales toward UPSCALE_TARGET_DPI when uncapped."""
        # Height 1100 -> DPI 100; target 150 needs factor 1.5 (< cap).
        image = np.full((1100, 400, 3), 255, dtype=np.uint8)
        out, upscaled, factor = Preprocessor._maybe_upscale(image)

        assert upscaled is True
        assert factor == pytest.approx(UPSCALE_TARGET_DPI / (1100 / ASSUMED_PAGE_HEIGHT_INCHES), rel=0.01)
        # New height reaches ~ the target-DPI height (150 * 11 == 1650).
        assert out.shape[0] == pytest.approx(1650, abs=3)
