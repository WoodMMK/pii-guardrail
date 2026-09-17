"""Property-based test: redaction with zero regions is an identity.

# Feature: thai-image-pii-guardrail, Property 18: Redaction with zero regions is an identity

Validates: Requirements 8.4
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.typing import NDArray

from pii_guardrail.redactor import Redactor

# Valid uint8 images across the shapes the redactor must preserve:
# grayscale (H, W), color (H, W, 3), and RGBA (H, W, 4). H/W in a modest range.
_dim = st.integers(min_value=1, max_value=64)
_images = st.one_of(
    arrays(dtype=np.uint8, shape=st.tuples(_dim, _dim)),
    arrays(dtype=np.uint8, shape=st.tuples(_dim, _dim, st.just(3))),
    arrays(dtype=np.uint8, shape=st.tuples(_dim, _dim, st.just(4))),
)


@pytest.mark.property
@settings(max_examples=200)
@given(image=_images)
def test_zero_regions_is_identity(image: NDArray) -> None:
    """Redacting with an empty regions list yields a pixelwise-identical image.

    With zero regions there is nothing to cover, so ``Redactor.redact`` returns
    an image that is pixelwise-equivalent to the input across every channel and
    shape (grayscale, RGB, RGBA). The returned array is a copy (not the same
    object), so the input is never mutated.

    # Feature: thai-image-pii-guardrail, Property 18: Redaction with zero regions is an identity
    Validates: Requirements 8.4
    """
    result = Redactor().redact(image, [])

    # Pixelwise-equivalent to the input: the key property.
    assert np.array_equal(result, image)

    # It is a copy, not the same object (the input is never mutated).
    assert result is not image
