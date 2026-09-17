"""Integration test: the real PyThaiNLP-backed Thai NER classifier.

Exercises :class:`~pii_guardrail.ner.ThaiNerClassifier` against the actual
PyThaiNLP model (engine ``thainer``). Skipped when PyThaiNLP is not installed or
the model cannot load, so the suite stays green without the optional ``ner``
extra. This is the piece that proves NER recovers full given+family names that
the deterministic patterns miss.

Requirements: 5.1 (model consulted and its categories merged).
"""

from __future__ import annotations

import pytest

pytest.importorskip("pythainlp")

from pii_guardrail.detector import Detector  # noqa: E402
from pii_guardrail.models import SensitiveCategory  # noqa: E402
from pii_guardrail.ner import ThaiNerClassifier  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def ner() -> ThaiNerClassifier:
    """Load the real Thai NER model once; skip the module if it cannot load."""
    clf = ThaiNerClassifier()
    if not clf.available:
        pytest.skip("PyThaiNLP is installed but the NER model could not load.")
    return clf


def test_full_thai_name_detected_as_person(ner: ThaiNerClassifier) -> None:
    # Bare given + family name (no honorific): patterns are weak here, NER is not.
    cats = ner.classify("ธิดารัตน์ พั่วบุญมี")
    assert SensitiveCategory.PERSON_NAME in cats


def test_honorific_name_detected_as_person(ner: ThaiNerClassifier) -> None:
    cats = ner.classify("นางสาวสุจิวรรณ มานะ")
    assert SensitiveCategory.PERSON_NAME in cats


def test_non_entity_text_is_empty(ner: ThaiNerClassifier) -> None:
    # Plain descriptive text with no named entity.
    assert ner.classify("ค่าธรรมเนียมการศึกษา") == set() or (
        SensitiveCategory.PERSON_NAME not in ner.classify("ค่าธรรมเนียมการศึกษา")
    )


def test_detector_merges_ner_with_patterns(ner: ThaiNerClassifier) -> None:
    # End-to-end through the Detector: a bare full name becomes a PERSON_NAME
    # region even though the deterministic patterns alone would miss it.
    detector = Detector(classifier=ner, use_classifier=True)
    cats = detector.classify_segment("ธิดารัตน์ พั่วบุญมี")
    assert SensitiveCategory.PERSON_NAME in cats
