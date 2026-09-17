"""Property-based test for the totality of pattern-based classification.

# Feature: thai-image-pii-guardrail, Property 7: Classification is total and returns a set of categories

Validates: Requirements 3.1
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pii_guardrail.detector import classify_segment
from pii_guardrail.models import SensitiveCategory


@pytest.mark.property
@settings(max_examples=200)
@given(text=st.text())
def test_classification_is_total(text: str) -> None:
    """For any string, classify_segment returns a set of SensitiveCategory.

    The function is total: it never raises, always returns a ``set``, and every
    member of that set is a ``SensitiveCategory``.

    # Feature: thai-image-pii-guardrail, Property 7: Classification is total and returns a set of categories
    Validates: Requirements 3.1
    """
    result = classify_segment(text)

    assert isinstance(result, set)
    assert all(isinstance(category, SensitiveCategory) for category in result)
