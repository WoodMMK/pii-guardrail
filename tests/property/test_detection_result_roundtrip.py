"""Property-based test for DetectionResult serialization round-trips.

# Feature: thai-image-pii-guardrail, Property 16: Detection_Result serialization round-trips

Validates: Requirements 7.4
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pii_guardrail.models import (
    BoundingBox,
    DetectionResult,
    SensitiveCategory,
    SensitiveRegion,
    deserialize_detection_result,
    serialize_detection_result,
)


@st.composite
def bounding_boxes(draw: st.DrawFn) -> BoundingBox:
    """Generate a BoundingBox honoring its invariants (x,y>=0; width,height>=1).

    Sizes are kept modest and need not fit any image: this is a pure
    serialization property, so in-image bounds are irrelevant here.
    """
    return BoundingBox(
        x=draw(st.integers(min_value=0, max_value=4000)),
        y=draw(st.integers(min_value=0, max_value=4000)),
        width=draw(st.integers(min_value=1, max_value=4000)),
        height=draw(st.integers(min_value=1, max_value=4000)),
    )


@st.composite
def sensitive_regions(draw: st.DrawFn) -> SensitiveRegion:
    """Generate a SensitiveRegion with a non-empty category set."""
    return SensitiveRegion(
        box=draw(bounding_boxes()),
        categories=draw(
            st.sets(st.sampled_from(list(SensitiveCategory)), min_size=1)
        ),
        text=draw(st.text()),
        confidence=draw(
            st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
            )
        ),
    )


@st.composite
def detection_results(draw: st.DrawFn) -> DetectionResult:
    """Generate an arbitrary DetectionResult with 0..8 regions."""
    return DetectionResult(
        regions=draw(st.lists(sensitive_regions(), min_size=0, max_size=8)),
        image_width=draw(st.integers(min_value=1, max_value=4000)),
        image_height=draw(st.integers(min_value=1, max_value=4000)),
    )


@pytest.mark.property
@settings(max_examples=200)
@given(result=detection_results())
def test_detection_result_serialization_round_trips(result: DetectionResult) -> None:
    """deserialize(serialize(result)) reproduces an equivalent DetectionResult.

    Dimensions, count, and every region (boxes, category sets, text, and
    confidence) round-trip faithfully, with region order preserved.

    # Feature: thai-image-pii-guardrail, Property 16: Detection_Result serialization round-trips
    Validates: Requirements 7.4
    """
    restored = deserialize_detection_result(serialize_detection_result(result))

    assert restored.image_width == result.image_width
    assert restored.image_height == result.image_height
    assert restored.count == result.count
    assert len(restored.regions) == len(result.regions)

    # Order-sensitive comparison: serialize preserves list order.
    for restored_region, original_region in zip(restored.regions, result.regions):
        assert restored_region.box.x == original_region.box.x
        assert restored_region.box.y == original_region.box.y
        assert restored_region.box.width == original_region.box.width
        assert restored_region.box.height == original_region.box.height
        assert restored_region.categories == original_region.categories
        assert restored_region.text == original_region.text
        # confidence passes only through float() -> exact equality holds.
        assert restored_region.confidence == original_region.confidence
