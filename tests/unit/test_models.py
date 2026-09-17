"""Unit tests for core data model invariants.

Covers:
    - BoundingBox rejects non-positive width/height and negative origin.
    - A valid BoundingBox at origin (0, 0) with positive width/height works.
    - DetectionResult.count matches the length of its regions list.

Requirements: 7.1, 7.2, 7.3.
"""

from __future__ import annotations

import pytest

from pii_guardrail.models import (
    BoundingBox,
    DetectionResult,
    SensitiveCategory,
    SensitiveRegion,
)

pytestmark = pytest.mark.unit


class TestBoundingBoxInvariants:
    def test_negative_x_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            BoundingBox(x=-1, y=0, width=10, height=10)

    def test_negative_y_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            BoundingBox(x=0, y=-1, width=10, height=10)

    def test_zero_width_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            BoundingBox(x=0, y=0, width=0, height=10)

    def test_negative_width_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            BoundingBox(x=0, y=0, width=-5, height=10)

    def test_zero_height_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            BoundingBox(x=0, y=0, width=10, height=0)

    def test_negative_height_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            BoundingBox(x=0, y=0, width=10, height=-5)

    def test_valid_box_at_origin_is_accepted(self) -> None:
        box = BoundingBox(x=0, y=0, width=1, height=1)
        assert box.x == 0
        assert box.y == 0
        assert box.width == 1
        assert box.height == 1

    def test_valid_box_with_positive_offset_is_accepted(self) -> None:
        box = BoundingBox(x=5, y=7, width=20, height=15)
        assert (box.x, box.y, box.width, box.height) == (5, 7, 20, 15)


class TestDetectionResultCount:
    def test_count_is_zero_for_no_regions(self) -> None:
        result = DetectionResult(regions=[], image_width=100, image_height=100)
        assert result.count == 0
        assert result.count == len(result.regions)

    def test_count_matches_length_for_n_regions(self) -> None:
        regions = [
            SensitiveRegion(
                box=BoundingBox(x=i, y=i, width=10, height=10),
                categories={SensitiveCategory.EMAIL},
                text=f"user{i}@example.com",
                confidence=0.9,
            )
            for i in range(4)
        ]
        result = DetectionResult(
            regions=regions, image_width=200, image_height=200
        )
        assert result.count == 4
        assert result.count == len(result.regions)

    def test_count_tracks_default_empty_regions(self) -> None:
        result = DetectionResult()
        assert result.count == 0
        assert result.count == len(result.regions)
