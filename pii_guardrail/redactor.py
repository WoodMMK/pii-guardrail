"""Redactor: draw filled black rectangles over each Sensitive_Region.

Covers the full Bounding_Box of every ``SensitiveRegion`` with solid black,
preserving the input image's pixel dimensions and channel count. Never mutates
the input array.

Requirements: 8.1, 8.2, 8.4, 8.5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .models import SensitiveRegion


class Redactor:
    """Cover sensitive regions with solid black rectangles."""

    def redact(self, image: "NDArray", regions: "list[SensitiveRegion]") -> "NDArray":
        """Return a copy of ``image`` with a solid black rectangle covering the
        full Bounding_Box of each region.

        Every pixel within a region's box, including all four corners, is set to
        black (0) across all channels. With zero regions the returned image is a
        pixelwise-identical copy of the input. Output width, height, and channel
        count equal the input's. The input array is never mutated.

        Requirements: 8.1, 8.2, 8.4, 8.5.
        """
        # Copy so the input array is never mutated (satisfies 8.4 for the
        # zero-region case and keeps callers' data intact).
        result = image.copy()

        for region in regions:
            box = region.box
            x = box.x
            y = box.y
            # Slice end is exclusive, so x + width / y + height includes the
            # bottom-right corner pixel of the box (full coverage, 8.2).
            result[y : y + box.height, x : x + box.width] = 0

        return result
