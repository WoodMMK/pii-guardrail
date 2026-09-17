"""Property-based test for the disabled-classifier equivalence rule.

# Feature: thai-image-pii-guardrail, Property 11: Disabled classifier equals pattern-only classification

Validates: Requirements 5.2
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pii_guardrail.detector import Detector, classify_segment
from pii_guardrail.errors import ClassifierUnavailableError
from pii_guardrail.models import SensitiveCategory


class _SpyClassifier:
    """A classifier that would add a category if ever consulted.

    Used to prove that a Detector with a classifier *supplied* but
    ``use_classifier=False`` never consults it: its ``classify`` returns a
    category that pattern-based classification would not otherwise produce for
    plain generated text, so any leakage into the result would break the
    equivalence assertion.
    """

    available = True

    def classify(self, text: str) -> set[SensitiveCategory]:
        # A category that arbitrary st.text() will essentially never produce
        # via pattern matching, so consulting the model would be observable.
        raise AssertionError(
            "classifier must not be consulted when use_classifier is False"
        )


@pytest.mark.property
@settings(max_examples=100)
@given(text=st.text())
def test_disabled_classifier_equals_pattern_only(text: str) -> None:
    """For any string, a disabled classifier yields pattern-only results.

    Two disabled configurations must both equal the module-level
    ``classify_segment`` (pattern-based source of truth):
      - a Detector with no classifier and ``use_classifier=False`` (default), and
      - a Detector with a classifier supplied but ``use_classifier=False``.

    # Feature: thai-image-pii-guardrail, Property 11: Disabled classifier equals pattern-only classification
    Validates: Requirements 5.2
    """
    expected = classify_segment(text)

    # No classifier, disabled (default).
    assert Detector(use_classifier=False).classify_segment(text) == expected

    # Classifier supplied but disabled: it must never be consulted, so the
    # result still equals pattern-only classification.
    detector_with_disabled_classifier = Detector(
        classifier=_SpyClassifier(), use_classifier=False
    )
    assert detector_with_disabled_classifier.classify_segment(text) == expected
