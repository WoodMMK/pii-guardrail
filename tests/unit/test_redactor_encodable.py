"""Unit test: the Redactor returns an encodable Redacted_Image.

Requirement 8.3: WHEN redaction completes, THE Redactor SHALL produce a
Redacted_Image that the Backend_Service can return to the Web_Interface.

The Backend_Service returns the image by encoding it (PNG/JPEG) for transport.
So "can be returned" is verified here by asserting the redacted output encodes
successfully to both PNG and JPEG and yields non-empty byte buffers. Cases:
    - one or more in-bounds sensitive regions,
    - zero regions (encoding still works),
using a concrete 3-channel uint8 image (JPEG requires 3-channel or grayscale).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pii_guardrail.models import BoundingBox, SensitiveCategory, SensitiveRegion
from pii_guardrail.redactor import Redactor

pytestmark = pytest.mark.unit


def _make_image() -> np.ndarray:
    """A small, clearly non-black 3-channel uint8 image (40x60x3, value 200)."""
    return np.full((40, 60, 3), 200, dtype=np.uint8)


def _assert_encodable(image: np.ndarray) -> None:
    """Assert ``image`` encodes to non-empty PNG and JPEG byte buffers."""
    png_ok, png_buf = cv2.imencode(".png", image)
    assert png_ok is True, "PNG encoding of the redacted image failed"
    assert png_buf is not None and png_buf.size > 0, "PNG buffer was empty"

    jpg_ok, jpg_buf = cv2.imencode(".jpg", image)
    assert jpg_ok is True, "JPEG encoding of the redacted image failed"
    assert jpg_buf is not None and jpg_buf.size > 0, "JPEG buffer was empty"


class TestRedactorReturnsEncodableImage:
    def test_encodable_with_regions(self) -> None:
        """Redacted output (with sensitive regions) encodes to PNG and JPEG."""
        image = _make_image()
        regions = [
            SensitiveRegion(
                box=BoundingBox(x=5, y=5, width=20, height=10),
                categories={SensitiveCategory.EMAIL},
                text="user@example.com",
                confidence=0.9,
            ),
            SensitiveRegion(
                box=BoundingBox(x=30, y=20, width=25, height=15),
                categories={SensitiveCategory.PHONE_NUMBER, SensitiveCategory.URL},
                text="+66 12 345 6789",
                confidence=0.8,
            ),
        ]

        redacted = Redactor().redact(image, regions)

        # Sanity: shape preserved and encodable for transport (Requirement 8.3).
        assert redacted.shape == image.shape
        _assert_encodable(redacted)

    def test_encodable_with_zero_regions(self) -> None:
        """Redacted output with zero regions still encodes to PNG and JPEG."""
        image = _make_image()

        redacted = Redactor().redact(image, [])

        assert redacted.shape == image.shape
        _assert_encodable(redacted)
