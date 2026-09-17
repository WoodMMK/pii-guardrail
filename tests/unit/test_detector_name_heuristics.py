"""Unit tests: the deterministic patterns do NOT classify person/organization names.

Design change: PERSON_NAME and ORGANIZATION_NAME are no longer produced by the
regex-based :func:`~pii_guardrail.detector.classify_segment`. Those categories
require context to judge and were high-recall-but-noisy as patterns (they
flagged ordinary headings/labels like "Tuition Fee" or "Grand Total" as names).
Name/organization detection is now owned by the whole-page LLM classifier
(:class:`~pii_guardrail.litellm_backend.LiteLLMClassifier`), which the Detector
unions with the patterns.

These tests lock that boundary: the patterns stay total and stay focused on
structured/Thai identifiers, and never emit PERSON_NAME/ORGANIZATION_NAME on
their own -- including on the OCR-garbled name lines that the old heuristics
used to (over-)match.
"""

from __future__ import annotations

import pytest

from pii_guardrail.detector import classify_segment
from pii_guardrail.models import SensitiveCategory

pytestmark = pytest.mark.unit


class TestPatternsDoNotEmitPersonName:
    """Patterns never produce PERSON_NAME -- the LLM owns names now."""

    def test_thai_honorific_name_not_person_from_patterns(self) -> None:
        assert SensitiveCategory.PERSON_NAME not in classify_segment("นาย สมชาย ใจดี")

    def test_thai_name_field_label_not_person_from_patterns(self) -> None:
        text = "ชื่อตัวและชื่อสกุล น.ส. ธิดารัตน์ พั่วบุญมี"
        assert SensitiveCategory.PERSON_NAME not in classify_segment(text)

    def test_latin_name_not_person_from_patterns(self) -> None:
        assert SensitiveCategory.PERSON_NAME not in classify_segment("John Smith")

    def test_latin_name_label_not_person_from_patterns(self) -> None:
        assert SensitiveCategory.PERSON_NAME not in classify_segment("Name Thidarat")


class TestPatternsDoNotEmitOrganizationName:
    """Patterns never produce ORGANIZATION_NAME -- the LLM owns orgs now."""

    def test_thai_org_not_tagged_from_patterns(self) -> None:
        assert (
            SensitiveCategory.ORGANIZATION_NAME
            not in classify_segment("บริษัท เอซีเอ็มอี จำกัด")
        )

    def test_latin_org_not_tagged_from_patterns(self) -> None:
        assert (
            SensitiveCategory.ORGANIZATION_NAME
            not in classify_segment("Acme Corp")
        )


class TestNonSensitiveHeadingsAreClean:
    """The headings/labels that the old heuristics false-flagged are now clean.

    These are the concrete regressions the change targets: ordinary document
    text must not be classified as a name/organization by the patterns.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "Tuition Fee",
            "Grand Total",
            "Payment Method",
            "Date Of Birth",
            "Prince Of Songkla University",
            "ค่าธรรมเนียมการศึกษา",
            "ศาสนา พุทธ",
            "รายการ",
            "hello world",
        ],
    )
    def test_heading_has_no_name_or_org(self, text: str) -> None:
        cats = classify_segment(text)
        assert SensitiveCategory.PERSON_NAME not in cats
        assert SensitiveCategory.ORGANIZATION_NAME not in cats


class TestStructuredDetectionStillWorks:
    """Patterns still detect the structured/Thai values they own."""

    def test_email_still_detected(self) -> None:
        assert SensitiveCategory.EMAIL in classify_segment("mail bob@example.com")

    def test_money_still_detected(self) -> None:
        assert SensitiveCategory.MONEY_AMOUNT in classify_segment("ยอดชำระ 1,500.00 บาท")

    def test_classification_stays_total(self) -> None:
        for odd in ("", "   ", "Name\n", "ชื่อ\t", "John Smith"):
            assert isinstance(classify_segment(odd), set)
