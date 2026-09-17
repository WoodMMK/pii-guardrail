"""Property-based tests for bounding-box normalization geometry.

Feature: thai-image-pii-guardrail.

These tests exercise ``pii_guardrail.geometry.normalize_quad_to_box``, which
turns an OCR quadrilateral into an enclosing, in-bounds axis-aligned
``BoundingBox`` (top-left origin). The implementation clamps vertices into the
image, rounds min/max to pixels, and widens any degenerate (zero-extent) box by
1px while staying inside the image.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pii_guardrail.geometry import normalize_quad_to_box


@st.composite
def images_with_in_bounds_quads(draw):
    """Generate (quad, width, height) with a quad whose vertices lie in-bounds.

    Width and height are each >= 1. The quad has exactly four vertices; each
    vertex x is in [0, W] and each y is in [0, H], matching the top-left origin
    coordinate system used by the OCR engine.
    """
    width = draw(st.integers(min_value=1, max_value=2000))
    height = draw(st.integers(min_value=1, max_value=2000))

    coord = lambda hi: st.floats(  # noqa: E731 - concise per-axis coordinate strategy
        min_value=0.0,
        max_value=float(hi),
        allow_nan=False,
        allow_infinity=False,
    )
    vertex = st.tuples(coord(width), coord(height))
    quad = draw(st.lists(vertex, min_size=4, max_size=4))
    return quad, width, height


# Feature: thai-image-pii-guardrail, Property 3: Bounding-box normalization stays within the image and encloses its source
# Validates: Requirements 2.2
@pytest.mark.property
@settings(max_examples=200)
@given(images_with_in_bounds_quads())
def test_normalize_quad_stays_in_image_and_encloses_source(case):
    quad, width, height = case

    box = normalize_quad_to_box(quad, width, height)

    # Box is a valid, positive-extent rectangle.
    assert box.x >= 0
    assert box.y >= 0
    assert box.width > 0
    assert box.height > 0

    # Box lies fully inside the image (top-left origin).
    assert box.x + box.width <= width
    assert box.y + box.height <= height

    # Box encloses every source vertex. The implementation rounds vertex
    # coordinates to pixels, so we compare against the rounded coordinates and
    # use the box's inclusive bounds (a vertex on the far edge counts as
    # enclosed).
    for vx, vy in quad:
        rx = round(vx)
        ry = round(vy)
        assert box.x <= rx <= box.x + box.width
        assert box.y <= ry <= box.y + box.height
