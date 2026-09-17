"""Unit tests for the OCR-tolerant person-name heuristics.

Scanned Thai ID cards produce garbled OCR: honorifics like "น.ส." degrade to
"น.", Latin titles like "Miss" become "Mies", and field labels get joined to
the value ("Last namePhuaboonmee"). The Detector's name heuristics were extended
to treat Thai/Latin *name-field labels* and *degraded honorifics/titles* as
person-name signals so these lines are still redacted.

These tests lock that behavior using representative garbled lines (the same
shapes PaddleOCR produced on the sample ID) plus regressions ensuring ordinary
names still classify and non-name lines (labels alone, religion field, orgs) do
not become false positives.
"""

from __future__ import annotations

import pytest

from pii_guardrail.detector import classify_segment
from pii_guardrail.models import SensitiveCategory

pytestmark = pytest.mark.unit


class TestThaiNameOcrTolerant:
    def test_garbled_honorific_and_label_line(self) -> None:
        # OCR of the ID's name line: label garbled, "น.ส." -> "น.", name joined.
        text = "ชื่ดวนละซือสกุ น. ธิดารัตน์พ่วบุญมี"
        assert SensitiveCategory.PERSON_NAME in classify_segment(text)

    def test_name_field_label_with_name(self) -> None:
        text = "ชื่อตัวและชื่อสกุล น.ส. ธิดารัตน์ พั่วบุญมี"
        assert SensitiveCategory.PERSON_NAME in classify_segment(text)

    def test_label_missing_final_char(self) -> None:
        # OCR frequently drops the final "ล" of "ชื่อสกุล".
        text = "ชื่อตัวและชื่อสกุ ธิดารัตน์"
        assert SensitiveCategory.PERSON_NAME in classify_segment(text)

    def test_degraded_honorific_variants(self) -> None:
        for prefix in ("น.ส", "นส.", "น."):
            text = f"{prefix} ธิดารัตน์"
            assert SensitiveCategory.PERSON_NAME in classify_segment(text), prefix

    def test_normal_honorific_still_matches(self) -> None:
        assert SensitiveCategory.PERSON_NAME in classify_segment("นาย สมชาย ใจดี")


class TestLatinNameOcrTolerant:
    def test_garbled_miss_title(self) -> None:
        # "Miss" misread as "Mies".
        assert SensitiveCategory.PERSON_NAME in classify_segment("Mies Thidarat")

    def test_last_name_label_joined_to_value(self) -> None:
        assert SensitiveCategory.PERSON_NAME in classify_segment("Last namePhuaboonmee")

    def test_name_label_with_value(self) -> None:
        assert SensitiveCategory.PERSON_NAME in classify_segment("Name Thidarat")

    def test_namo_ocr_misread_label(self) -> None:
        # "Name" misread as "Namo" on the card.
        assert SensitiveCategory.PERSON_NAME in classify_segment("Namo Thidarat")

    def test_normal_latin_name_still_matches(self) -> None:
        assert SensitiveCategory.PERSON_NAME in classify_segment("John Smith")


class TestNameHeuristicNoFalsePositives:
    def test_bare_latin_label_alone_is_not_a_name(self) -> None:
        # A label word with no accompanying value must not be a person name.
        assert SensitiveCategory.PERSON_NAME not in classify_segment("Name")
        assert SensitiveCategory.PERSON_NAME not in classify_segment("Last name")

    def test_bare_thai_label_alone_is_not_a_name(self) -> None:
        assert SensitiveCategory.PERSON_NAME not in classify_segment("ชื่อ")

    def test_thai_religion_field_not_a_name(self) -> None:
        assert SensitiveCategory.PERSON_NAME not in classify_segment("ศาสนา พุทธ")

    def test_plain_word_not_a_name(self) -> None:
        assert SensitiveCategory.PERSON_NAME not in classify_segment("hello")

    def test_organization_not_tagged_person(self) -> None:
        cats = classify_segment("บริษัท เอซีเอ็มอี จำกัด")
        assert SensitiveCategory.ORGANIZATION_NAME in cats
        assert SensitiveCategory.PERSON_NAME not in cats

    def test_classification_stays_total(self) -> None:
        for odd in ("", "   ", "Name\n", "ชื่อ\t"):
            assert isinstance(classify_segment(odd), set)
