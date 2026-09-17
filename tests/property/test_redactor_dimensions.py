"""Property-based test: redaction preserves image dimensions.

Feature: thai-image-pii-guardrail, Property 19: Redaction preserves image
dimensions.
Validates: Requirements 8.5.

This test exercises ``pii_guardrail.redactor.Redactor.redact``, which returns a
copy of the input image with each ``SensitiveRegion``'s Bounding_Box covered by
solid black. The redacted output must keep the input's exact width, height, and
channel count (captured by ``shape``) as well as its pixel ``dtype``, whether
there are zero regions or several.
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
    """Generate ``(image, regions)`` for the redactor.

    The image is a ``uint8`` array in one of three channel layouts: grayscale
    ``(H, W)``, color ``(H, W, 3)``, or ``(H, W, 4)``. Between 0 and 4 regions
    are drawn; each region's box fits fully inside the image and carries a
    non-empty category set. Allowing a region count of 0 naturally covers the
    zero-region case as well as the non-empty case.
    """
    width = draw(st.integers(min_value=8, max_value=64))
    height = draw(st.integers(min_value=8, max_value=64))
    layout = draw(st.sampled_from(("gray", "rgb", "rgba")))

    if layout == "gray":
        shape = (height, width)
    elif layout == "rgb":
        shape = (height, width, 3)
    else:
        shape = (height, width, 4)

    image = draw(
        st.integers(min_value=0, max_value=255).map(
            lambda fill: np.full(shape, fill, dtype=np.uint8)
        )
    )

    @st.composite
    def in_bounds_region(sub_draw):
        x = sub_draw(st.integers(min_value=0, max_value=width - 1))
        y = sub_draw(st.integers(min_value=0, max_value=height - 1))
        box_width = sub_draw(st.integers(min_value=1, max_value=width - x))
        box_height = sub_draw(st.integers(min_value=1, max_value=height - y))
        categories = sub_draw(
            st.sets(st.sampled_from(list(SensitiveCategory)), min_size=1, max_size=3)
        )
        return SensitiveRegion(
            box=BoundingBox(x=x, y=y, width=box_width, height=box_height),
            categories=categories,
            text="secret",
            confidence=0.9,
        )

    regions = draw(st.lists(in_bounds_region(), min_size=0, max_size=4))
    return image, regions


# Feature: thai-image-pii-guardrail, Property 19: Redaction preserves image dimensions
# Validates: Requirements 8.5
@pytest.mark.property
@settings(max_examples=200)
@given(images_with_regions())
def test_redaction_preserves_dimensions(case):
    """Redacted output keeps the input's shape (width, height, channels) and dtype.

    Feature: thai-image-pii-guardrail, Property 19: Redaction preserves image
    dimensions. Validates: Requirements 8.5.
    """
    image, regions = case

    output = Redactor().redact(image, regions)

    # shape captures width, height, AND channel count in one comparison; dtype
    # confirms the pixel type is unchanged. Holds for both the zero-region and
    # non-empty-region cases (the generator covers both).
    assert output.shape == image.shape
    assert output.dtype == image.dtype
