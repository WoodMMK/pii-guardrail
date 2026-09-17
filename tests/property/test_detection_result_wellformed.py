"""Property-based test for Detection_Result well-formedness and completeness.

# Feature: thai-image-pii-guardrail, Property 15: Every region in a Detection_Result is well-formed and complete

Validates: Requirements 7.1, 7.2, 7.3
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pii_guardrail.detector import Detector
from pii_guardrail.models import BoundingBox, DetectionResult, SensitiveRegion, TextSegment

# Known-sensitive seed strings spanning several categories so that the
# generated segment lists actually produce regions (not just empty results).
_SENSITIVE_STRINGS: tuple[str, ...] = (
    "user@example.com",
    "https://example.com/path",
    "$100",
    "50%",
    "1,000 บาท",
    "+66 81 234 5678",
    "081-234-5678",
    "sk-abcdefghijklmnop1234",
    "Bearer abcdef123456",
    "นายสมชาย ใจดี",
    "บริษัท เอซีเอ็มอี จำกัด",
    "Dr. Jane Smith",
    "Acme Corp",
)


@st.composite
def _segments_within_image(draw: st.DrawFn) -> tuple[list[TextSegment], int, int]:
    """Generate a list of TextSegments whose boxes fit inside an image (W, H).

    A mix of known-sensitive strings (so regions are produced) and arbitrary
    text (for variety). Every box satisfies the BoundingBox invariants
    (x >= 0, y >= 0, width > 0, height > 0) and fits within the generated image
    dimensions.
    """
    image_width = draw(st.integers(min_value=20, max_value=2000))
    image_height = draw(st.integers(min_value=20, max_value=2000))

    text_strategy = st.one_of(st.sampled_from(_SENSITIVE_STRINGS), st.text())

    def _draw_segment() -> TextSegment:
        width = draw(st.integers(min_value=1, max_value=image_width))
        height = draw(st.integers(min_value=1, max_value=image_height))
        x = draw(st.integers(min_value=0, max_value=image_width - width))
        y = draw(st.integers(min_value=0, max_value=image_height - height))
        text = draw(text_strategy)
        confidence = draw(st.floats(min_value=0.0, max_value=1.0))
        return TextSegment(
            text=text,
            box=BoundingBox(x=x, y=y, width=width, height=height),
            confidence=confidence,
        )

    count = draw(st.integers(min_value=0, max_value=12))
    segments = [_draw_segment() for _ in range(count)]
    return segments, image_width, image_height


@pytest.mark.property
@settings(max_examples=200)
@given(data=_segments_within_image())
def test_every_region_is_well_formed_and_complete(
    data: tuple[list[TextSegment], int, int],
) -> None:
    """Every region in a Detection_Result is well-formed and complete.

    For any list of text segments run through the Detector, a DetectionResult
    built from the detection output has regions matching the detection output,
    a count equal to the regions list length, every box satisfying the
    BoundingBox invariants, and every region carrying a non-empty category set.

    # Feature: thai-image-pii-guardrail, Property 15: Every region in a Detection_Result is well-formed and complete
    Validates: Requirements 7.1, 7.2, 7.3
    """
    segments, image_width, image_height = data

    outcome = Detector().detect(segments)
    result = DetectionResult(
        regions=outcome.regions,
        image_width=image_width,
        image_height=image_height,
    )

    # 7.1: the result's regions equal exactly the detection output.
    assert result.regions == outcome.regions

    # count matches the regions list length.
    assert result.count == len(result.regions)

    for region in result.regions:
        assert isinstance(region, SensitiveRegion)

        # 7.2: every box satisfies the BoundingBox invariants.
        assert region.box.x >= 0
        assert region.box.y >= 0
        assert region.box.width > 0
        assert region.box.height > 0

        # 7.3: every region has a non-empty set of categories.
        assert isinstance(region.categories, set)
        assert len(region.categories) >= 1
