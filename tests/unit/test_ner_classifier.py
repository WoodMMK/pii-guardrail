"""Unit tests for the optional Thai NER classifier (:mod:`pii_guardrail.ner`).

These tests exercise the ``ThaiNerClassifier`` IOB-parsing / category-mapping
logic and its ``ClassifierModel`` contract WITHOUT requiring PyThaiNLP: a fake
tagger is injected into the instance's ``_ner`` slot, so the parsing and
availability behavior are tested deterministically. A separate, PyThaiNLP-backed
end-to-end check lives in the integration tests and is skipped when the package
is absent.

Requirements: 5.1, 5.2, 5.3 (the concrete optional ClassifierModel).
"""

from __future__ import annotations

import pytest

from pii_guardrail.classifier import ClassifierModel
from pii_guardrail.errors import ClassifierUnavailableError
from pii_guardrail.models import SensitiveCategory
from pii_guardrail.ner import ThaiNerClassifier

pytestmark = pytest.mark.unit


class _FakeNer:
    """Stand-in for pythainlp's NER: returns preset IOB tuples, or raises."""

    def __init__(self, tagged=None, raise_exc: Exception | None = None) -> None:
        self._tagged = tagged if tagged is not None else []
        self._raise = raise_exc
        self.calls: list[str] = []

    def tag(self, text, pos=False):  # signature mirrors pythainlp NER.tag
        self.calls.append(text)
        if self._raise is not None:
            raise self._raise
        return self._tagged


def _make_classifier(tagged=None, raise_exc: Exception | None = None) -> ThaiNerClassifier:
    """Build a ThaiNerClassifier with its model replaced by a fake tagger.

    The real constructor attempts to load PyThaiNLP; we bypass that and inject a
    controlled fake so the parsing/contract logic is tested in isolation.
    """
    clf = ThaiNerClassifier.__new__(ThaiNerClassifier)
    clf._engine_name = "fake"
    clf._ner = _FakeNer(tagged=tagged, raise_exc=raise_exc)
    clf._load_error = None
    return clf


class TestProtocolAndAvailability:
    def test_satisfies_classifier_model_protocol(self) -> None:
        clf = _make_classifier()
        assert isinstance(clf, ClassifierModel)

    def test_available_true_when_model_present(self) -> None:
        assert _make_classifier().available is True

    def test_unavailable_classifier_raises_on_classify(self) -> None:
        # Simulate a failed model load: _ner is None.
        clf = ThaiNerClassifier.__new__(ThaiNerClassifier)
        clf._engine_name = "fake"
        clf._ner = None
        clf._load_error = RuntimeError("load failed")
        assert clf.available is False
        with pytest.raises(ClassifierUnavailableError):
            clf.classify("ธิดารัตน์")

    def test_runtime_tag_failure_raises_unavailable(self) -> None:
        clf = _make_classifier(raise_exc=ValueError("boom"))
        with pytest.raises(ClassifierUnavailableError):
            clf.classify("some text")


class TestIobCategoryMapping:
    def test_person_tokens_map_to_person_name(self) -> None:
        tagged = [
            ("ทดสอบ", "O"),
            ("นาย", "B-PERSON"),
            ("ธิดา", "I-PERSON"),
            ("รัตน์", "I-PERSON"),
        ]
        cats = _make_classifier(tagged).classify("ทดสอบนายธิดารัตน์")
        assert cats == {SensitiveCategory.PERSON_NAME}

    def test_organization_maps(self) -> None:
        tagged = [("บริษัท", "B-ORGANIZATION"), ("ปตท", "I-ORGANIZATION")]
        cats = _make_classifier(tagged).classify("บริษัทปตท")
        assert SensitiveCategory.ORGANIZATION_NAME in cats

    def test_location_maps_to_thai_address(self) -> None:
        tagged = [("จังหวัด", "B-LOCATION"), ("สกลนคร", "I-LOCATION")]
        cats = _make_classifier(tagged).classify("จังหวัดสกลนคร")
        assert SensitiveCategory.THAI_ADDRESS in cats

    def test_multiple_entities_yield_all_categories(self) -> None:
        tagged = [
            ("นาย", "B-PERSON"),
            ("เอ", "I-PERSON"),
            ("อยู่", "O"),
            ("กรุงเทพ", "B-LOCATION"),
        ]
        cats = _make_classifier(tagged).classify("นายเอ อยู่ กรุงเทพ")
        assert SensitiveCategory.PERSON_NAME in cats
        assert SensitiveCategory.THAI_ADDRESS in cats

    def test_email_phone_url_date_money_map(self) -> None:
        tagged = [
            ("a@b.com", "B-EMAIL"),
            ("02-123-4567", "B-PHONE"),
            ("http://x.io", "B-URL"),
            ("2003", "B-DATE"),
            ("100", "B-MONEY"),
        ]
        cats = _make_classifier(tagged).classify("...")
        assert {
            SensitiveCategory.EMAIL,
            SensitiveCategory.PHONE_NUMBER,
            SensitiveCategory.URL,
            SensitiveCategory.DATE_OF_BIRTH,
            SensitiveCategory.MONEY_AMOUNT,
        } <= cats

    def test_unmapped_entity_types_ignored(self) -> None:
        # TIME/LAW have no sensitive-category mapping -> ignored.
        tagged = [("14:49", "B-TIME"), ("พ.ร.บ.", "B-LAW"), ("x", "O")]
        assert _make_classifier(tagged).classify("...") == set()

    def test_all_outside_tags_yield_empty(self) -> None:
        tagged = [("hello", "O"), ("world", "O")]
        assert _make_classifier(tagged).classify("hello world") == set()

    def test_whitespace_text_skips_model(self) -> None:
        clf = _make_classifier(tagged=[("x", "B-PERSON")])
        assert clf.classify("   ") == set()
        # The model must NOT be consulted for whitespace-only input.
        assert clf._ner.calls == []

    def test_three_tuple_token_tag_pos_supported(self) -> None:
        # When pos=True style triples appear, the tag is still at index 1.
        tagged = [("นาย", "B-PERSON", "NOUN"), ("เอ", "I-PERSON", "PROPN")]
        cats = _make_classifier(tagged).classify("นายเอ")
        assert SensitiveCategory.PERSON_NAME in cats

    def test_malformed_items_are_skipped(self) -> None:
        tagged = [("only-one",), None, 42, ("ok", "B-PERSON")]
        cats = _make_classifier(tagged).classify("...")
        assert cats == {SensitiveCategory.PERSON_NAME}


# ---------------------------------------------------------------------------
# Detector wiring: NER results are unioned with patterns; unavailable NER falls
# back to pattern-only with a warning.
# ---------------------------------------------------------------------------

from pii_guardrail.detector import Detector  # noqa: E402
from pii_guardrail.models import BoundingBox, TextSegment  # noqa: E402


class _AvailableNerSpy:
    """A ClassifierModel-shaped spy that returns a fixed category set."""

    def __init__(self, returns: set[SensitiveCategory]) -> None:
        self._returns = set(returns)
        self.available = True
        self.calls: list[str] = []

    def classify(self, text: str) -> set[SensitiveCategory]:
        self.calls.append(text)
        return set(self._returns)


class _UnavailableNer:
    """A ClassifierModel that is enabled but not available at runtime."""

    available = False

    def classify(self, text: str) -> set[SensitiveCategory]:  # pragma: no cover
        raise ClassifierUnavailableError("unavailable")


class TestDetectorNerWiring:
    def test_ner_categories_union_with_patterns(self) -> None:
        # Pattern classifies the email; NER additionally contributes PERSON_NAME.
        ner = _AvailableNerSpy({SensitiveCategory.PERSON_NAME})
        detector = Detector(classifier=ner, use_classifier=True)

        cats = detector.classify_segment("user@example.com")

        assert SensitiveCategory.EMAIL in cats  # from patterns
        assert SensitiveCategory.PERSON_NAME in cats  # from NER
        assert ner.calls == ["user@example.com"]

    def test_ner_finds_name_patterns_miss(self) -> None:
        # A bare Thai full name (no honorific/label) that patterns miss, but NER
        # tags as a person -> Detector surfaces PERSON_NAME via the model.
        ner = _AvailableNerSpy({SensitiveCategory.PERSON_NAME})
        detector = Detector(classifier=ner, use_classifier=True)

        cats = detector.classify_segment("ธิดารัตน์ พั่วบุญมี")
        assert SensitiveCategory.PERSON_NAME in cats

    def test_unavailable_ner_falls_back_to_patterns_with_warning(self) -> None:
        detector = Detector(classifier=_UnavailableNer(), use_classifier=True)

        segment = TextSegment(
            text="user@example.com",
            box=BoundingBox(x=0, y=0, width=10, height=10),
            confidence=1.0,
        )
        outcome = detector.detect([segment])

        # Pattern result still present.
        assert outcome.regions
        assert SensitiveCategory.EMAIL in outcome.regions[0].categories
        # Fallback recorded a warning (Requirement 5.3).
        assert outcome.warnings

    def test_disabled_classifier_is_pattern_only(self) -> None:
        # Even with a model present, use_classifier=False -> patterns only.
        ner = _AvailableNerSpy({SensitiveCategory.PERSON_NAME})
        detector = Detector(classifier=ner, use_classifier=False)

        cats = detector.classify_segment("45%")
        assert cats == {SensitiveCategory.PERCENT_VALUE}
        assert ner.calls == []  # model never consulted when disabled
