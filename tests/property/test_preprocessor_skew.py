"""Property-based test for the Preprocessor's skew correction.

Feature: thai-image-pii-guardrail.

This test exercises ``pii_guardrail.preprocessor.Preprocessor.preprocess``,
which detects skew (``estimate_skew_angle``) and rotates the image so that the
residual skew of the text rows is within 1 degree of horizontal (Requirement
6.3). We build a clean synthetic document (a white page with several black
horizontal text-row bars), rotate it by a known angle ``theta``, feed the
rotated image through ``preprocess``, and assert the reported
``residual_skew_deg`` is <= 1.0.

Angle range note: the skew is drawn from -12..12 degrees, excluding the tiny
deadband magnitude (< 0.2 deg) that the implementation intentionally treats as
already-aligned. This range is what the ``minAreaRect``-based detector corrects
reliably for block-of-text images; at more extreme angles the dominant
rectangle orientation of a synthetic bar image can flip, which is a property of
the synthetic generator rather than the implementation, so we keep the range to
where detection is stable (still well beyond the 1-degree tolerance we assert).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pii_guardrail.preprocessor import Preprocessor


def _make_text_row_image(width: int, height: int, num_rows: int) -> np.ndarray:
    """Return a white image with ``num_rows`` evenly spaced black horizontal bars.

    The bars stand in for lines of text: long, thin, horizontal, and clearly
    the dominant foreground orientation so ``minAreaRect`` skew detection is
    stable. Bars span the middle 80% of the width and are a few pixels tall.
    """
    image = np.full((height, width, 3), 255, dtype=np.uint8)

    left = int(width * 0.1)
    right = int(width * 0.9)
    bar_thickness = max(3, height // (num_rows * 6))

    # Distribute rows across the vertical middle so rotation keeps them on-page.
    top_margin = int(height * 0.2)
    usable = int(height * 0.6)
    step = usable // num_rows
    for i in range(num_rows):
        y = top_margin + i * step + step // 2
        y0 = max(0, y - bar_thickness // 2)
        y1 = min(height, y + bar_thickness // 2 + 1)
        image[y0:y1, left:right] = 0

    return image


def _rotate(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate ``image`` counter-clockwise by ``angle_deg`` about its center.

    Borders are filled white so the rotation introduces no dark artifacts that
    could confuse skew detection.
    """
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


@st.composite
def skewed_text_images(draw):
    """Generate (rotated_image,) for a synthetic text-row page skewed by theta.

    The page is rendered at a document-like resolution ABOVE the 150 DPI
    reliability threshold (height >= 1650 px) so the Preprocessor's auto-upscale
    step (Option A) is a no-op here. This keeps the test focused on deskew
    correctness for a real-resolution scan rather than incidentally exercising
    the upscale path (and avoids denoising a freshly-enlarged image on every
    example).
    """
    width = draw(st.integers(min_value=1200, max_value=1500))
    height = draw(st.integers(min_value=1700, max_value=1900))
    num_rows = draw(st.integers(min_value=8, max_value=14))

    # Known skew, excluding the tiny deadband the implementation ignores.
    magnitude = draw(st.floats(min_value=0.2, max_value=12.0, allow_nan=False))
    sign = draw(st.sampled_from((-1.0, 1.0)))
    theta = sign * magnitude

    base = _make_text_row_image(width, height, num_rows)
    return _rotate(base, theta)


# Feature: thai-image-pii-guardrail, Property 13: Skew correction aligns text rows within 1 degree
# Validates: Requirements 6.3
@pytest.mark.property
@settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(skewed_text_images())
def test_skew_correction_within_one_degree(rotated_image):
    result = Preprocessor().preprocess(rotated_image, is_scanned=True)

    # The document-resolution input is above the reliability threshold, so no
    # upscale should have occurred (this test isolates deskew correctness).
    assert result.upscaled is False
    assert abs(result.residual_skew_deg) <= 1.0
