"""Property-based test for the enabled-but-unavailable classifier fallback.

# Feature: thai-image-pii-guardrail, Property 12: Enabled-but-unavailable classifier falls back and warns

Validates: Requirements 5.3
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pii_guardrail.detector import Detector, classify_segment
from pii_guardrail.errors import ClassifierUnavailableError
from pii_guardrail.models import BoundingBox, SensitiveCategory, TextSegment


class FakeClassifierModel:
    """A ClassifierModel that is enabled but unavailable at runtime.

    Satisfies the ``ClassifierModel`` protocol structurally: exposes an
    ``available`` property (``False``) and a ``classify`` method that, per the
    protocol contract, raises :class:`ClassifierUnavailableError` when invoked
    while unavailable. The Detector should never actually call ``classify``
    because it checks ``available`` first, but the raising behavior models the
    protocol faithfully and guards the fallback path either way.
    """

    @property
    def available(self) -> bool:
        return False

    def classify(self, text: str) -> set[SensitiveCategory]:
        raise ClassifierUnavailableError(
            "FakeClassifierModel is unavailable at runtime."
        )


@pytest.mark.property
@settings(max_examples=100)
@given(text=st.text())
def test_enabled_but_unavailable_classifier_falls_back_and_warns(text: str) -> None:
    """Enabled-but-unavailable classifier falls back to patterns and warns.

    For any text string, when the Classifier_Model is enabled but unavailable
    at runtime, the Detector returns exactly the pattern-based categories, and
    when a region is produced (the pattern result is non-empty), the outcome's
    warnings are non-empty.

    # Feature: thai-image-pii-guardrail, Property 12: Enabled-but-unavailable classifier falls back and warns
    Validates: Requirements 5.3
    """
    fake = FakeClassifierModel()
    detector = Detector(classifier=fake, use_classifier=True)

    pattern_categories = classify_segment(text)

    # The Detector's per-segment classification equals pattern-only results.
    assert detector.classify_segment(text) == pattern_categories

    box = BoundingBox(x=0, y=0, width=10, height=10)
    segment = TextSegment(text=text, box=box, confidence=0.9)
    outcome = detector.detect([segment])

    # Classification came only from the patterns: each region's categories
    # equal the pattern-based result for its source text.
    for region in outcome.regions:
        assert region.categories == pattern_categories

    if pattern_categories:
        # A region was produced, so the fallback warning must be recorded.
        assert len(outcome.regions) == 1
        assert outcome.regions[0].categories == pattern_categories
        assert outcome.warnings
    else:
        # Non-sensitive text yields no region.
        assert outcome.regions == []
