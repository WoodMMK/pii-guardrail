"""Unit tests for the Thailand-specific PII patterns in the Detector.

Covers the categories added for Thai identity documents:
    - THAI_NATIONAL_ID (mod-11 checksum-validated)
    - DATE_OF_BIRTH (Thai month, Latin month, numeric forms)
    - THAI_ADDRESS (full-word keywords and abbreviated markers)
    - LICENSE_PLATE (old and newer 1-digit-prefixed plates)
    - BANK_ACCOUNT (grouped account numbers)
    - PASSPORT_NUMBER

Plus regressions ensuring the new patterns do not disturb existing behavior
(a phone number stays a phone number; a valid national ID is NOT also tagged as
a phone number; classification stays total).

National ID numbers used here are SYNTHETIC: they are constructed to satisfy the
official mod-11 checksum via :func:`make_valid_thai_id`, not taken from any real
person.
"""

from __future__ import annotations

import pytest

from pii_guardrail.detector import classify_segment
from pii_guardrail.models import SensitiveCategory

pytestmark = pytest.mark.unit


def make_valid_thai_id(prefix12: str) -> str:
    """Return a 13-digit ID: the given 12 digits + correct mod-11 check digit.

    The check digit is ``(11 - (sum(d_i * (13 - i) for i in 0..11) % 11)) % 10``.
    """
    assert len(prefix12) == 12 and prefix12.isdigit()
    total = sum(int(prefix12[i]) * (13 - i) for i in range(12))
    check = (11 - (total % 11)) % 10
    return prefix12 + str(check)


# A couple of distinct synthetic, checksum-valid IDs.
_VALID_ID = make_valid_thai_id("110170050100")
_VALID_ID_2 = make_valid_thai_id("312345678901")


class TestThaiNationalId:
    def test_plain_valid_id_detected(self) -> None:
        cats = classify_segment(_VALID_ID)
        assert SensitiveCategory.THAI_NATIONAL_ID in cats

    def test_grouped_valid_id_detected(self) -> None:
        grouped = (
            f"{_VALID_ID[0]}-{_VALID_ID[1:5]}-{_VALID_ID[5:10]}-"
            f"{_VALID_ID[10:12]}-{_VALID_ID[12]}"
        )
        cats = classify_segment(grouped)
        assert SensitiveCategory.THAI_NATIONAL_ID in cats

    def test_space_grouped_valid_id_detected(self) -> None:
        spaced = (
            f"{_VALID_ID[0]} {_VALID_ID[1:5]} {_VALID_ID[5:10]} "
            f"{_VALID_ID[10:12]} {_VALID_ID[12]}"
        )
        cats = classify_segment(spaced)
        assert SensitiveCategory.THAI_NATIONAL_ID in cats

    def test_invalid_checksum_not_detected(self) -> None:
        # Flip the check digit so the checksum fails.
        bad = _VALID_ID[:12] + str((int(_VALID_ID[12]) + 1) % 10)
        cats = classify_segment(bad)
        assert SensitiveCategory.THAI_NATIONAL_ID not in cats

    def test_valid_id_not_also_tagged_phone(self) -> None:
        # A checksum-valid national ID must NOT also be labeled a phone number.
        cats = classify_segment(_VALID_ID)
        assert SensitiveCategory.PHONE_NUMBER not in cats

    def test_valid_id_not_tagged_api_key(self) -> None:
        # All-digit IDs never satisfy the mixed alnum API-key fallback anyway,
        # but assert it explicitly as a guard.
        cats = classify_segment(_VALID_ID_2)
        assert SensitiveCategory.THAI_NATIONAL_ID in cats
        assert SensitiveCategory.API_KEY not in cats


class TestDateOfBirth:
    def test_thai_month_date(self) -> None:
        cats = classify_segment("18 ต.ค. 2546")
        assert SensitiveCategory.DATE_OF_BIRTH in cats

    def test_thai_full_month_date(self) -> None:
        cats = classify_segment("18 ตุลาคม 2546")
        assert SensitiveCategory.DATE_OF_BIRTH in cats

    def test_latin_month_date(self) -> None:
        cats = classify_segment("Date of Birth 18 Oct. 2003")
        assert SensitiveCategory.DATE_OF_BIRTH in cats

    def test_numeric_date(self) -> None:
        cats = classify_segment("12/05/2540")
        assert SensitiveCategory.DATE_OF_BIRTH in cats

    def test_thai_abbrev_month_not_mislabeled_address(self) -> None:
        # "ต.ค." is October, NOT the address marker "ต." (subdistrict).
        cats = classify_segment("18 ต.ค. 2546")
        assert SensitiveCategory.THAI_ADDRESS not in cats


class TestThaiAddress:
    def test_full_address_detected(self) -> None:
        text = "ที่อยู่ 166 หมู่ที่ 2 ต.ปลาโหล อ.วาริชภูมิ จ.สกลนคร"
        cats = classify_segment(text)
        assert SensitiveCategory.THAI_ADDRESS in cats

    def test_keyword_only_detected(self) -> None:
        assert SensitiveCategory.THAI_ADDRESS in classify_segment("ถนนสุขุมวิท")
        assert SensitiveCategory.THAI_ADDRESS in classify_segment("จังหวัดเชียงใหม่")

    def test_bare_five_digits_not_address(self) -> None:
        # A lone 5-digit postal code without any keyword is too ambiguous.
        assert SensitiveCategory.THAI_ADDRESS not in classify_segment("50200")


class TestLicensePlate:
    def test_old_style_plate(self) -> None:
        assert SensitiveCategory.LICENSE_PLATE in classify_segment("กท 1234")

    def test_new_style_plate_with_leading_digit(self) -> None:
        assert SensitiveCategory.LICENSE_PLATE in classify_segment("1กก 2345")

    def test_plain_number_not_plate(self) -> None:
        assert SensitiveCategory.LICENSE_PLATE not in classify_segment("1234")


class TestBankAccount:
    def test_grouped_account_detected(self) -> None:
        assert SensitiveCategory.BANK_ACCOUNT in classify_segment("123-4-56789-0")

    def test_grouped_six_digit_block(self) -> None:
        assert SensitiveCategory.BANK_ACCOUNT in classify_segment("123-4-567890-1")


class TestPassport:
    def test_one_letter_passport(self) -> None:
        assert SensitiveCategory.PASSPORT_NUMBER in classify_segment("A1234567")

    def test_two_letter_passport(self) -> None:
        assert SensitiveCategory.PASSPORT_NUMBER in classify_segment("AA123456")


class TestRegressionsUnaffected:
    def test_phone_still_phone(self) -> None:
        cats = classify_segment("081-234-5678")
        assert SensitiveCategory.PHONE_NUMBER in cats
        assert SensitiveCategory.THAI_NATIONAL_ID not in cats

    def test_percent_still_percent(self) -> None:
        assert SensitiveCategory.PERCENT_VALUE in classify_segment("45%")

    def test_email_still_email(self) -> None:
        assert SensitiveCategory.EMAIL in classify_segment("user@example.com")

    def test_plain_text_non_sensitive(self) -> None:
        assert classify_segment("hello world") == set()

    def test_thirteen_random_digits_not_id(self) -> None:
        # 13 digits that fail the checksum are not a national ID.
        assert SensitiveCategory.THAI_NATIONAL_ID not in classify_segment(
            "9999999999999"
        )

    def test_classification_stays_total(self) -> None:
        # Never raises, always returns a set (spot-check a few odd inputs).
        for odd in ("", "   ", "\n\t", "๙๙๙", "!@#$%^&*()"):
            result = classify_segment(odd)
            assert isinstance(result, set)
