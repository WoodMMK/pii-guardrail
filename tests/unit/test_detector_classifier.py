"""Unit tests for Detector integration with an enabled+available Classifier_Model.

Covers the "classifier consulted" path (Requirement 5.1): when the Detector is
created with an enabled, available Classifier_Model, it SHALL submit extracted
text segments to the model and merge the model's categories with the
pattern-based categories.

These tests use a spy `FakeClassifierModel` (available=True) that records whether
`classify` was called and returns a chosen category set, so we can assert both
that the model is consulted and that the returned categories are the UNION of
pattern-based results and the model's results.

Requirements: 5.1.
"""

from __future__ import annotations

import pytest

from pii_guardrail.detector import Detector, classify_segment
from pii_guardrail.models import SensitiveCategory, TextSegment
from pii_guardrail.models import BoundingBox

pytestmark = pytest.mark.unit


class FakeClassifierModel:
    """A spy ClassifierModel that records calls and returns a fixed category set.

    Satisfies the `ClassifierModel` protocol structurally: exposes an
    `available` property and a `classify` method. It records every text passed
    to `classify` (and a simple call counter) so tests can assert the Detector
    consulted the model.
    """

    def __init__(
        self,
        returns: set[SensitiveCategory] | None = None,
        available: bool = True,
    ) -> None:
        self._returns = set(returns) if returns is not None else set()
        self._available = available
        self.classify_calls: list[str] = []

    @property
    def available(self) -> bool:
        return self._available

    @property
    def classify_called(self) -> bool:
        return len(self.classify_calls) > 0

    def classify(self, text: str) -> set[SensitiveCategory]:
        self.classify_calls.append(text)
        return set(self._returns)


class TestEnabledAvailableClassifierIsConsulted:
    def test_classify_is_called_on_the_spy(self) -> None:
        spy = FakeClassifierModel(
            returns={SensitiveCategory.ORGANIZATION_NAME}, available=True
        )
        detector = Detector(classifier=spy, use_classifier=True)

        detector.classify_segment("some ambiguous text")

        assert spy.classify_called
        assert spy.classify_calls == ["some ambiguous text"]

    def test_returned_categories_are_union_of_patterns_and_model(self) -> None:
        # Text that pattern-based classification recognizes as an EMAIL, so the
        # pattern result is a known, non-empty set distinct from the model's.
        text = "contact me at user@example.com"
        pattern_categories = classify_segment(text)
        assert SensitiveCategory.EMAIL in pattern_categories

        # The model contributes a category the patterns do NOT produce here.
        model_category = SensitiveCategory.ORGANIZATION_NAME
        assert model_category not in pattern_categories

        spy = FakeClassifierModel(returns={model_category}, available=True)
        detector = Detector(classifier=spy, use_classifier=True)

        result = detector.classify_segment(text)

        assert spy.classify_called
        # The result is the UNION of pattern-based categories and the model's.
        assert result == pattern_categories | {model_category}
        assert SensitiveCategory.EMAIL in result
        assert model_category in result

    def test_model_adds_category_to_otherwise_non_sensitive_text(self) -> None:
        # Text the patterns treat as non-sensitive; the model supplies a category.
        text = "just some plain words"
        pattern_categories = classify_segment(text)

        spy = FakeClassifierModel(
            returns={SensitiveCategory.PERSON_NAME}, available=True
        )
        detector = Detector(classifier=spy, use_classifier=True)

        result = detector.classify_segment(text)

        assert spy.classify_called
        assert result == pattern_categories | {SensitiveCategory.PERSON_NAME}
        assert SensitiveCategory.PERSON_NAME in result

    def test_detect_consults_model_and_merges_into_region(self) -> None:
        # End-to-end through detect(): the region's categories should include
        # both the pattern-based category and the model-contributed category.
        text = "user@example.com"
        pattern_categories = classify_segment(text)
        assert SensitiveCategory.EMAIL in pattern_categories

        model_category = SensitiveCategory.ORGANIZATION_NAME
        spy = FakeClassifierModel(returns={model_category}, available=True)
        detector = Detector(classifier=spy, use_classifier=True)

        segment = TextSegment(
            text=text,
            box=BoundingBox(x=0, y=0, width=10, height=10),
            confidence=0.9,
        )
        outcome = detector.detect([segment])

        assert spy.classify_called
        assert len(outcome.regions) == 1
        assert outcome.regions[0].categories == pattern_categories | {model_category}
        # An available classifier means no fallback warning.
        assert outcome.warnings == []
