"""Property-based test that OCR confidences stay within [0.0, 1.0].

# Feature: thai-image-pii-guardrail, Property 5: Confidence values are within range

Validates: Requirements 2.7

Segments are generated with RAW confidences that may be out of range (including
negative, greater-than-one, NaN, and infinite values) and fed through the
``OCREngine`` via a :class:`FakeOCRBackend`. The engine clamps every confidence
into [0.0, 1.0] (mapping NaN to 0.0), so every resulting ``TextSegment`` -- and
every ``SensitiveRegion`` derived from those segments by ``Detector.detect`` --
must carry a confidence within range.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pii_guardrail.detector import Detector
from pii_guardrail.models import BoundingBox, TextSegment
from pii_guardrail.ocr import OCREngine
from tests.property.ocr_fakes import FakeOCRBackend


# Raw confidences intentionally span outside [0.0, 1.0], plus NaN and infinities,
# so the engine's clamping is actually exercised (Requirement 2.7).
_raw_confidence = st.one_of(
    st.floats(min_value=-5.0, max_value=5.0),
    st.sampled_from([float("nan"), float("inf"), float("-inf")]),
)


@st.composite
def _image_and_segments(draw: st.DrawFn) -> tuple[np.ndarray, list[TextSegment]]:
    """Generate an image plus segments whose boxes fit inside it.

    A ``BoundingBox`` requires a non-negative origin and strictly positive
    extent, so boxes are constructed relative to the drawn image size: pick
    ``x`` / ``y`` leaving room for at least one pixel of extent, then a width /
    height that keeps the box inside the image. Pixels are irrelevant (the
    ``FakeOCRBackend`` ignores them), so a zero-filled uint8 array suffices.
    """
    width = draw(st.integers(min_value=32, max_value=128))
    height = draw(st.integers(min_value=32, max_value=128))

    count = draw(st.integers(min_value=0, max_value=8))
    segments: list[TextSegment] = []
    for _ in range(count):
        x = draw(st.integers(min_value=0, max_value=width - 2))
        y = draw(st.integers(min_value=0, max_value=height - 2))
        box_w = draw(st.integers(min_value=1, max_value=width - x))
        box_h = draw(st.integers(min_value=1, max_value=height - y))
        text = draw(st.text())
        confidence = draw(_raw_confidence)
        segments.append(
            TextSegment(
                text=text,
                box=BoundingBox(x=x, y=y, width=box_w, height=box_h),
                confidence=confidence,
            )
        )

    image = np.zeros((height, width, 3), dtype=np.uint8)
    return image, segments


@pytest.mark.property
@settings(max_examples=200)
@given(data=_image_and_segments())
def test_confidence_values_within_range(
    data: tuple[np.ndarray, list[TextSegment]],
) -> None:
    """Engine-output and derived-region confidences are all in [0.0, 1.0].

    For segments carrying arbitrary (possibly out-of-range) raw confidences,
    every ``TextSegment`` returned by ``OCREngine.extract`` has a finite
    confidence within [0.0, 1.0], and every ``SensitiveRegion`` produced by
    ``Detector.detect`` over those segments carries a confidence in the same
    range.

    # Feature: thai-image-pii-guardrail, Property 5: Confidence values are within range
    Validates: Requirements 2.7
    """
    image, segments = data

    engine = OCREngine(FakeOCRBackend(segments))
    extracted = engine.extract(image)

    # Assert on the ENGINE OUTPUT: raw NaN/out-of-range inputs are clamped.
    for seg in extracted:
        assert math.isfinite(seg.confidence)
        assert 0.0 <= seg.confidence <= 1.0

    outcome = Detector().detect(extracted)
    for region in outcome.regions:
        assert math.isfinite(region.confidence)
        assert 0.0 <= region.confidence <= 1.0
