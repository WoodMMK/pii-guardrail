"""Property-based tests for the Redactor's box-coverage guarantee.

Feature: thai-image-pii-guardrail, Property 17: Redaction covers each region's
full bounding box in black.
Validates: Requirements 8.1, 8.2.

These tests exercise ``pii_guardrail.redactor.Redactor.redact``. For any image
and any non-empty list of ``SensitiveRegion``s that fit inside the image, every
pixel within each region's ``BoundingBox`` (all four corners inclusive) must be
black (0 across every channel) in the redacted output.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pii_guardrail.models import BoundingBox, SensitiveCategory, SensitiveRegion
from pii_guardrail.redactor import Redactor


@st.composite
def images_with_regions(draw):
    """Generate (image, regions) with 1..4 boxes that fit inside the image.

    The image is an ``(H, W, 3)`` uint8 array with arbitrary pixel values
    (nonzero allowed, so a redacted box is a visible change). Each region's box
    satisfies x in [0, W-1], y in [0, H-1], width in [1, W-x], height in
    [1, H-y], guaranteeing the box lies fully inside the image.
    """
    width = draw(st.integers(min_value=8, max_value=64))
    height = draw(st.integers(min_value=8, max_value=64))

    # Arbitrary pixel values; bias toward nonzero so redaction is observable.
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(seed)
    image = rng.integers(1, 256, size=(height, width, 3), dtype=np.uint8)

    def a_box():
        x = draw(st.integers(min_value=0, max_value=width - 1))
        y = draw(st.integers(min_value=0, max_value=height - 1))
        w = draw(st.integers(min_value=1, max_value=width - x))
        h = draw(st.integers(min_value=1, max_value=height - y))
        return BoundingBox(x=x, y=y, width=w, height=h)

    n = draw(st.integers(min_value=1, max_value=4))
    regions = [
        SensitiveRegion(
            box=a_box(),
            categories={SensitiveCategory.EMAIL},
            text="",
            confidence=1.0,
        )
        for _ in range(n)
    ]
    return image, regions


# Feature: thai-image-pii-guardrail, Property 17: Redaction covers each region's full bounding box in black
# Validates: Requirements 8.1, 8.2
@pytest.mark.property
@settings(max_examples=200)
@given(images_with_regions())
def test_redaction_covers_each_box_in_black(case):
    image, regions = case

    result = Redactor().redact(image, regions)

    # Every pixel within each region's box (corners inclusive) is black across
    # all channels.
    for region in regions:
        box = region.box
        sub = result[box.y : box.y + box.height, box.x : box.x + box.width]
        assert np.all(sub == 0), (
            f"region box {box} not fully black in redacted output"
        )

    # Light shape/dtype check (fully covered by Property 19; a sanity check here).
    assert result.shape == image.shape
    assert result.dtype == image.dtype
