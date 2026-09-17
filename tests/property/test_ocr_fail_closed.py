"""Property-based test for OCR fail-closed behavior on invalid input.

# Feature: thai-image-pii-guardrail, Property 6: Invalid input fails closed with no partial result

Validates: Requirements 2.8, 2.9

Two properties exercise the OCR_Engine's fail-closed contract:

- Invalid input (missing/corrupt/unsupported array shape) is rejected with
  :class:`~pii_guardrail.errors.InvalidImageError` *before* any backend runs,
  producing no result (Requirement 2.8).
- A valid image accepted by the engine but whose backend fails during
  extraction is surfaced as :class:`~pii_guardrail.errors.OCRProcessingError`
  with no partial result (Requirement 2.9).

In both cases ``extract`` raises rather than returning, so "no partial result"
is inherent: no value is ever bound.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pii_guardrail.errors import InvalidImageError, OCRProcessingError
from pii_guardrail.ocr import OCREngine
from tests.property.ocr_fakes import FakeOCRBackend, RaisingOCRBackend


# Small pixel dimensions keep generated arrays cheap while covering the space.
_DIM = st.integers(min_value=1, max_value=64)


@st.composite
def invalid_images(draw: st.DrawFn) -> object:
    """Generate a variety of inputs the OCR_Engine must reject (2.8).

    Covers: ``None``; a non-ndarray (list or string); an empty ndarray; a 1-D
    ndarray; a 3-D ndarray with an invalid channel count; and a 3-D ndarray with
    a zero-length spatial axis. Each of these should trigger
    :class:`InvalidImageError` before any backend runs.
    """
    kind = draw(
        st.sampled_from(
            [
                "none",
                "non_ndarray",
                "empty",
                "one_dim",
                "bad_channels",
                "zero_spatial",
            ]
        )
    )

    if kind == "none":
        return None

    if kind == "non_ndarray":
        # Either a Python list or a string -- neither is an ndarray.
        return draw(
            st.one_of(
                st.lists(st.integers(min_value=0, max_value=255), max_size=8),
                st.text(max_size=8),
            )
        )

    if kind == "empty":
        # Zero-size arrays: np.array([]) or np.empty((0, 0)).
        return draw(
            st.sampled_from(
                [np.array([]), np.empty((0, 0)), np.empty((0, 0, 3), dtype=np.uint8)]
            )
        )

    if kind == "one_dim":
        length = draw(st.integers(min_value=1, max_value=32))
        return np.zeros((length,), dtype=np.uint8)

    if kind == "bad_channels":
        # 3-D array whose channel count is not in {1, 3, 4}.
        h = draw(_DIM)
        w = draw(_DIM)
        channels = draw(st.sampled_from([2, 5, 6, 7]))
        return np.zeros((h, w, channels), dtype=np.uint8)

    # kind == "zero_spatial": a zero-length spatial axis, e.g. (0, W, 3).
    w = draw(_DIM)
    axis = draw(st.sampled_from(["height", "width"]))
    if axis == "height":
        return np.zeros((0, w, 3), dtype=np.uint8)
    return np.zeros((w, 0, 3), dtype=np.uint8)


@st.composite
def valid_images(draw: st.DrawFn) -> np.ndarray:
    """Generate a valid uint8 (H, W, 3) color image the engine accepts."""
    h = draw(st.integers(min_value=8, max_value=64))
    w = draw(st.integers(min_value=8, max_value=64))
    return np.zeros((h, w, 3), dtype=np.uint8)


@pytest.mark.property
@settings(max_examples=200)
@given(bad_input=invalid_images())
def test_invalid_input_raises_invalid_image_error(bad_input: object) -> None:
    """Invalid input fails closed with ``InvalidImageError`` and no result.

    # Feature: thai-image-pii-guardrail, Property 6: Invalid input fails closed with no partial result
    Validates: Requirements 2.8

    Missing/corrupt/unsupported input is rejected before the backend runs, so
    the (empty) FakeOCRBackend is never consulted and no value is bound.
    """
    backend = FakeOCRBackend([])
    engine = OCREngine(backend)

    with pytest.raises(InvalidImageError):
        engine.extract(bad_input)

    # No partial result: the backend was never even invoked.
    assert backend.calls == []


@pytest.mark.property
@settings(max_examples=200)
@given(image=valid_images())
def test_backend_failure_raises_ocr_processing_error(image: np.ndarray) -> None:
    """A backend failure on valid input raises ``OCRProcessingError``, no result.

    # Feature: thai-image-pii-guardrail, Property 6: Invalid input fails closed with no partial result
    Validates: Requirements 2.9

    ``RaisingOCRBackend`` accepts the valid image (``available`` is ``True``)
    but always raises in ``run``; the engine must translate that into an
    ``OCRProcessingError`` and produce no partial result -- ``extract`` raises
    rather than returning, so no value is ever bound.
    """
    engine = OCREngine(RaisingOCRBackend())

    with pytest.raises(OCRProcessingError):
        engine.extract(image)
