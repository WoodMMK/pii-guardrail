"""Detector: pattern-based classification with an optional Classifier_Model.

This module implements the deterministic, pattern-based classification core used
by the ``Detector`` component. Task 4.1 wires :func:`classify_segment` into a
``Detector`` class (with the optional ``Classifier_Model``); this task (3.1)
provides the reusable, importable classification function itself.

Design notes:
    - :func:`classify_segment` is **total**: it never raises for any string
      input and always returns a ``set[SensitiveCategory]``.
    - Empty or whitespace-only text yields the empty set (non-sensitive).
    - A single segment may match multiple categories; every matching category
      is returned.
    - Classification is purely deterministic and pattern/heuristic based, with
      small seed gazetteers for Thai/Latin names and organizations so that
      category-shaped strings classify correctly without an external model.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10, 3.11,
4.1, 4.2, 4.3.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pii_guardrail.errors import ClassifierUnavailableError
from pii_guardrail.models import (
    SensitiveCategory,
    SensitiveRegion,
    TextSegment,
)

if TYPE_CHECKING:
    from pii_guardrail.classifier import ClassifierModel

# ---------------------------------------------------------------------------
# Seed gazetteers and lexical heuristics
#
# These are intentionally small: a handful of representative entries plus
# prefix/suffix heuristics. They exist so that category-shaped strings (Thai and
# Latin person names / organization names) classify correctly for pattern-based
# detection and for the property-based generators seeded from the same shapes.
# ---------------------------------------------------------------------------

# Thai honorific / title prefixes that introduce a person name.
_THAI_PERSON_PREFIXES: tuple[str, ...] = (
    "นาย",  # Mr.
    "นางสาว",  # Miss  (checked before "นาง" — longer match first)
    "นาง",  # Mrs.
    "เด็กชาย",
    "เด็กหญิง",
    "ด.ช.",
    "ด.ญ.",
    "ดร.",  # Dr.
    "คุณ",  # polite "Khun"
)

# Thai "name field" labels found on documents/ID cards. When a segment contains
# one of these labels it is a strong signal that the segment (a document line)
# carries a person name -- even when OCR garbles the honorific or joins tokens
# together (e.g. "ชื่อตัวและชื่อสกุล น. ธิดารัตน์พ่วบุญมี"). Redacting the whole
# labeled line is the safe choice on identity documents.
_THAI_NAME_FIELD_LABELS: tuple[str, ...] = (
    "ชื่อตัวและชื่อสกุล",
    "ชื่อตัวและชื่อสกุ",  # OCR often drops the final "ล"
    "ชื่อตัว",
    "ชื่อสกุล",
    "ชื่อ-สกุล",
    "ชื่อ-นามสกุล",
    "นามสกุล",
    "ชื่อ",  # broad; only used as a name-field signal alongside a following token
)

# OCR frequently mangles the "น.ส." / "นาย" honorifics on scanned Thai IDs. We
# accept a small set of degraded honorific forms so a garbled prefix still
# signals a following person name. These are matched as a token followed by a
# name-like token (see _match_thai_person).
_THAI_HONORIFIC_FUZZY: tuple[str, ...] = (
    "น.ส.", "น.ส", "นส.", "น.ส ", "น. ส", "น.",  # Miss (นางสาว) degraded forms
    "นาย", "นาง", "ด.ช.", "ด.ญ.",
)

# Thai organization keywords (prefixes/suffixes) indicating an organization name.
_THAI_ORG_KEYWORDS: tuple[str, ...] = (
    "บริษัท",  # company
    "ห้างหุ้นส่วนจำกัด",
    "ห้างหุ้นส่วน",  # partnership
    "จำกัด",  # limited
    "มหาชน",  # public (company)
    "องค์การ",  # organization
    "มูลนิธิ",  # foundation
    "สมาคม",  # association
    "ธนาคาร",  # bank
)

# Latin organization suffixes/keywords. Matched case-insensitively as whole
# tokens (allowing a trailing period, e.g. "Inc." / "Co.").
_LATIN_ORG_KEYWORDS: tuple[str, ...] = (
    "inc",
    "ltd",
    "llc",
    "llp",
    "plc",
    "corp",
    "corporation",
    "co",
    "company",
    "gmbh",
    "group",
    "holdings",
    "limited",
    "incorporated",
)

# Job-title / position keywords (Thai + Latin). Presence of one of these in a
# segment marks it as a JOB_TITLE. These are matched as substrings (Thai) or as
# whole tokens/phrases (Latin, case-insensitive). Titles are considered PII here
# because, combined with a name/organization, they help re-identify a person.
_THAI_JOB_TITLE_KEYWORDS: tuple[str, ...] = (
    "ผู้อำนวยการ",
    "รองผู้อำนวยการ",
    "ผู้ช่วยผู้อำนวยการ",
    "กรรมการผู้จัดการ",
    "รองกรรมการผู้จัดการ",
    "ผู้จัดการ",
    "รองผู้จัดการ",
    "ผู้ช่วยผู้จัดการ",
    "ประธานกรรมการ",
    "ประธานเจ้าหน้าที่บริหาร",
    "ประธาน",
    "รองประธาน",
    "กรรมการ",
    "เลขานุการ",
    "เหรัญญิก",
    "หัวหน้าฝ่าย",
    "หัวหน้างาน",
    "หัวหน้า",
    "ผู้บริหาร",
    "ผู้อำนวยการฝ่าย",
    "นักวิชาการเงินและบัญชี",
    "นักวิชาการ",
    "เจ้าหน้าที่",
    "พนักงาน",
    "อธิการบดี",
    "รองอธิการบดี",
    "คณบดี",
    "รองคณบดี",
    "ศาสตราจารย์",
    "รองศาสตราจารย์",
    "ผู้ช่วยศาสตราจารย์",
    "อาจารย์",
    "ปลัด",
    "รองปลัด",
    "อธิบดี",
    "รองอธิบดี",
    "ผู้ว่าราชการ",
    "นายกเทศมนตรี",
    "กำนัน",
    "ผู้ใหญ่บ้าน",
)

# Latin job-title keywords/phrases. Matched case-insensitively; multi-word
# phrases (e.g. "vice president") are checked as substrings, single words as
# whole tokens to avoid matching inside unrelated words.
_LATIN_JOB_TITLE_PHRASES: tuple[str, ...] = (
    "vice president",
    "senior vice president",
    "executive vice president",
    "assistant vice president",
    "vice-president",
    "managing director",
    "deputy managing director",
    "executive director",
    "board of directors",
    "chief executive officer",
    "chief financial officer",
    "chief operating officer",
    "chief technology officer",
    "chief marketing officer",
    "general manager",
    "deputy manager",
    "assistant manager",
    "vice chancellor",
    "vice-chancellor",
)
_LATIN_JOB_TITLE_WORDS: frozenset[str] = frozenset(
    {
        "director",
        "president",
        "manager",
        "ceo",
        "cfo",
        "coo",
        "cto",
        "cmo",
        "chairman",
        "chairperson",
        "secretary",
        "treasurer",
        "supervisor",
        "officer",
        "executive",
        "dean",
        "rector",
        "chancellor",
        "governor",
        "mayor",
        "commissioner",
        "professor",
        "lecturer",
    }
)

# Small seed gazetteer of Latin given names for the person-name heuristic. The
# heuristic also accepts capitalized multi-word tokens (see _looks_like_latin_name)
# so it is not limited to this list; the list anchors common examples.
_LATIN_GIVEN_NAMES: frozenset[str] = frozenset(
    {
        "john",
        "jane",
        "james",
        "mary",
        "robert",
        "michael",
        "william",
        "david",
        "richard",
        "susan",
        "sarah",
        "emma",
        "olivia",
        "liam",
        "noah",
        "somchai",
        "somsak",
        "wichai",
        "anong",
        "malee",
    }
)

# Latin honorific titles that introduce a person name. Includes a few common
# OCR misreads (e.g. "miss" -> "mies"/"mis", "mr" -> "mr") so a garbled title
# on a scanned ID still signals a following name.
_LATIN_PERSON_TITLES: frozenset[str] = frozenset(
    {
        "mr", "mrs", "ms", "miss", "dr", "prof", "sir", "madam", "mx",
        # OCR-degraded variants of "miss".
        "mies", "mis", "miass", "rniss",
    }
)

# Latin "name field" labels printed on documents/ID cards. Their presence marks
# the line as carrying a person name, even when OCR garbles the actual name or
# joins the label to it (e.g. "Last namePhuaboonmee", "Namo"). Matched as
# lowercased substrings (see _match_latin_person).
_LATIN_NAME_FIELD_LABELS: tuple[str, ...] = (
    "last name",
    "lastname",
    "first name",
    "firstname",
    "full name",
    "given name",
    "name",
    "namo",  # OCR misread of "Name"
)


# ---------------------------------------------------------------------------
# Compiled regular expressions for structured categories
# ---------------------------------------------------------------------------

# EMAIL: pragmatic RFC-ish local@domain.tld pattern.
_EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
)

# URL: http(s):// or protocol-relative, or a bare www./domain with a known-ish
# TLD. Kept broad but anchored to avoid matching arbitrary dotted numbers.
_URL_RE = re.compile(
    r"""(?xi)
    \b(
        (?:https?://|ftp://|www\.)          # scheme or www.
        [^\s]+                               # rest of the URL
        |
        (?:[a-z0-9](?:[a-z0-9\-]*[a-z0-9])?\.)+   # domain labels
        (?:com|net|org|io|co|gov|edu|info|biz|dev|app|ai|th|uk|us)  # TLD
        (?:/[^\s]*)?                          # optional path
    )
    """,
)

# PHONE_NUMBER: Thai mobile/landline and international formats. Requires enough
# digits to look like a phone number and avoids matching plain long integers by
# requiring either a leading + / 0 grouping or separators.
_PHONE_RE = re.compile(
    r"""(?x)
    (?<![\w])                                  # not mid-word
    (?:
        \+\d{1,3}[\s\-.]?                       # international prefix e.g. +66
        (?:\(?\d{1,4}\)?[\s\-.]?)?              # optional area/group
        \d{2,4}(?:[\s\-.]?\d{2,4}){1,3}         # grouped digits
        |
        0\d{1,2}[\s\-.]?\d{3}[\s\-.]?\d{3,4}    # Thai local 08x-xxx-xxxx / 0x-xxx-xxxx
    )
    (?![\w])
    """,
)

# PERCENT_VALUE: a number immediately (optionally spaced) followed by %.
_PERCENT_RE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?\s?%")

# MONEY_AMOUNT: currency symbol/code adjacent to a number, in either order.
# Symbols: ฿ $ € £ ¥.  Codes: THB, USD, EUR, GBP, JPY.  Thai word: บาท.
_CURRENCY_SYMBOL = r"[฿$€£¥]"
_CURRENCY_CODE = r"(?:THB|USD|EUR|GBP|JPY)"
_NUMBER = r"\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?"
_MONEY_RE = re.compile(
    rf"""(?xi)
    (?:
        {_CURRENCY_SYMBOL}\s?(?:{_NUMBER})           # ฿1,000  $19.99
        |
        (?:{_NUMBER})\s?{_CURRENCY_SYMBOL}           # 1,000฿
        |
        \b{_CURRENCY_CODE}\s?(?:{_NUMBER})\b         # THB 1000
        |
        \b(?:{_NUMBER})\s?{_CURRENCY_CODE}\b         # 1000 THB
        |
        (?:{_NUMBER})\s?บาท                          # 1,000 บาท
    )
    """,
)

# --- Credentials / secrets ---------------------------------------------------

# JWT / access token shaped as three base64url segments separated by dots,
# conventionally starting with the "eyJ" header prefix.
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+")

# GitHub-style personal/OAuth tokens: ghp_, gho_, ghu_, ghs_, ghr_ + 30+ chars.
_GITHUB_TOKEN_RE = re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")

# Generic bearer token.
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}")

# Explicit "access_token"/"token" assignments.
_TOKEN_ASSIGN_RE = re.compile(
    r"(?i)\b(?:access[_\-]?token|auth[_\-]?token|token)\b\s*[:=]\s*\S{6,}"
)

# OpenAI-style secret keys: sk-... (also handles sk-proj-...).
_OPENAI_KEY_RE = re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9]{16,}")

# AWS access key id: AKIA / ASIA followed by 16 uppercase alphanumerics.
_AWS_KEY_RE = re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")

# Google API key style: AIza + 35 chars.
_GOOGLE_KEY_RE = re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")

# Explicit "api_key"/"apikey"/"client_secret" assignments.
_APIKEY_ASSIGN_RE = re.compile(
    r"(?i)\b(?:api[_\-]?key|apikey|client[_\-]?secret|secret[_\-]?key)\b\s*[:=]\s*\S{6,}"
)

# Any ``*_KEY = value`` / ``KEY: value`` assignment, e.g. env-file entries like
# ``LITELLM_MASTER_KEY=<secret>`` or ``MASTER_KEY: abc123``. Environment
# variable names use underscores (no spaces), so the identifier ending in
# ``KEY`` is matched directly rather than relying on a word boundary between
# ``MASTER`` and ``KEY``. The value must be non-empty and non-whitespace.
_KEY_ASSIGN_RE = re.compile(
    r"(?i)\b[A-Z0-9_]*KEY\b\s*[:=]\s*\S{3,}"
)

# Private-key PEM headers.
_PEM_RE = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH |PGP |DSA )?PRIVATE KEY-----"
)

# Explicit generic secret/password assignments. Matches the bare words
# (secret/password/...) OR an identifier that ENDS in one of them, so env-file
# names like ``DB_PASSWORD``/``LITELLM_MASTER_SECRET`` are caught even though
# the underscore glues the words together.
_SECRET_ASSIGN_RE = re.compile(
    r"(?i)\b[A-Z0-9_]*(?:secret|password|passwd|pwd|passphrase|token)\b\s*[:=]\s*\S{4,}"
)

# Credentials embedded in a connection-string URL: ``scheme://user:password@host``.
# Common in DATABASE_URL / REDIS_URL / AMQP entries. The password segment
# between ':' and '@' is the sensitive part. Requires a non-empty password.
_URL_CREDENTIALS_RE = re.compile(
    r"(?i)\b[a-z][a-z0-9+.\-]*://[^\s:/@]+:[^\s:/@]+@"
)

# A long, mixed-content high-entropy-ish token used as a fallback for API keys
# (letters + digits, no spaces, sufficiently long). Used only as a heuristic.
_LONG_ALNUM_RE = re.compile(r"\b(?=[A-Za-z0-9_\-]*[A-Za-z])(?=[A-Za-z0-9_\-]*\d)[A-Za-z0-9_\-]{24,}\b")

# Thai script detection (any character in the Thai Unicode block).
_THAI_CHAR_RE = re.compile(r"[\u0E00-\u0E7F]")


# ---------------------------------------------------------------------------
# Thailand-specific PII patterns
#
# These target information that commonly appears on Thai identity documents,
# bank slips, and vehicle registrations. Where a structural check exists (the
# national ID checksum), it is applied to keep false positives low.
# ---------------------------------------------------------------------------

# THAI_NATIONAL_ID: 13 digits, optionally grouped as X-XXXX-XXXXX-XX-X with
# spaces or hyphens between groups. The pattern only finds *candidates*; the
# mod-11 checksum in _match_thai_national_id confirms a real ID.
_THAI_ID_CANDIDATE_RE = re.compile(
    r"""(?x)
    (?<!\d)                                     # not part of a longer number
    \d[\s\-]?                                    #  1 digit
    \d{4}[\s\-]?                                 #  4 digits
    \d{5}[\s\-]?                                 #  5 digits
    \d{2}[\s\-]?                                 #  2 digits
    \d                                           #  1 digit  (13 total)
    (?!\d)
    """,
)

# PASSPORT_NUMBER: 1-2 uppercase letters followed by 6-7 digits (covers Thai
# "AA1234567" and common international shapes). Anchored on word boundaries.
_PASSPORT_RE = re.compile(r"(?<![A-Za-z0-9])[A-Z]{1,2}\d{6,7}(?![A-Za-z0-9])")

# LICENSE_PLATE (Thailand): 1-2 Thai consonants (optionally a leading digit for
# the newer 1-Thai-digit series) + 1-4 digits, e.g. "กท 1234", "1กก 234",
# "ขข 1", often with a province name on the next line (not required here).
_THAI_PLATE_RE = re.compile(
    r"(?<![\u0E00-\u0E7F\w])\d?[\u0E01-\u0E2E]{1,2}\s?\d{1,4}(?![\u0E00-\u0E7F\w])"
)

# BANK_ACCOUNT (Thailand): commonly 10-12 digits, frequently grouped as
# XXX-X-XXXXX-X or XXX-X-XXXXXX-X. Match a grouped form to avoid catching every
# long integer; a bare 10-12 digit run is accepted only when clearly grouped.
_BANK_ACCOUNT_RE = re.compile(
    r"""(?x)
    (?<!\d)
    \d{3}[\s\-]\d[\s\-]\d{5,6}[\s\-]\d          # XXX-X-XXXXX(X)-X grouped account
    (?!\d)
    """,
)

# DATE_OF_BIRTH / dates: numeric and Thai/Latin month-name forms.
# Numeric: DD/MM/YYYY or DD-MM-YYYY (year 2 or 4 digits; Buddhist or Gregorian).
_DATE_NUMERIC_RE = re.compile(
    r"(?<!\d)(?:[0-3]?\d)[/\-.](?:[01]?\d)[/\-.](?:\d{4}|\d{2})(?!\d)"
)

# Thai abbreviated month tokens (ม.ค. ก.พ. ... ธ.ค.) and full names.
_THAI_MONTHS = (
    "ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.",
    "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค.",
    "มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
    "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม",
)
# Latin month names/abbreviations.
_LATIN_MONTHS = (
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "january", "february", "march", "april", "june",
    "july", "august", "september", "october", "november", "december",
)
# Date with a Latin month name: "18 Oct 2003", "18 Oct. 2003".
_LATIN_MONTH_DATE_RE = re.compile(
    r"(?xi)(?<!\d)[0-3]?\d\s+(?:" + "|".join(_LATIN_MONTHS) + r")\.?\s+\d{4}(?!\d)"
)

# Thai postal code: exactly 5 digits. Only treated as an address signal when a
# Thai address keyword is also present (see _match_thai_address), to avoid
# matching arbitrary 5-digit numbers.
_THAI_POSTCODE_RE = re.compile(r"(?<!\d)\d{5}(?!\d)")

# Thai address keywords that are unambiguous on their own (full words).
_THAI_ADDRESS_KEYWORDS: tuple[str, ...] = (
    "บ้านเลขที่", "เลขที่", "หมู่ที่", "หมู่บ้าน", "ซอย", "ถนน",
    "ตำบล", "แขวง", "อำเภอ", "เขต", "จังหวัด", "หมู่",
)

# Abbreviated address markers (ต. อ. จ.) share a prefix with Thai month
# abbreviations (ต.ค. = October, etc.). To avoid mislabeling a date as an
# address, an abbreviation counts as an address signal only when it is NOT part
# of a month token. We detect it as "<marker>" immediately followed by a Thai
# place character that is not the second letter of a month abbreviation.
_THAI_ADDRESS_ABBREV_RE = re.compile(
    r"(?<![\u0E00-\u0E7F])(?:ต|อ|จ)\.\s?[\u0E01-\u0E2E]"
)
# Month abbreviations to exclude (their second segment starts with these).
_THAI_MONTH_ABBREVS = ("ต.ค.", "จ.")  # "ต.ค." (Oct); "จ." alone is rare, guarded below


# ---------------------------------------------------------------------------
# Heuristic helpers
# ---------------------------------------------------------------------------


def _has_thai(text: str) -> bool:
    return bool(_THAI_CHAR_RE.search(text))


def _strip_structured(text: str) -> str:
    """Remove email and URL substrings so name heuristics don't match inside them.

    Emails and URLs frequently contain name-like tokens (e.g. the local part of
    ``john@example.com``). Those are already classified by their own patterns;
    blanking them out here prevents a spurious PERSON_NAME/ORGANIZATION_NAME
    match derived from characters that belong to a structured value.
    """
    cleaned = _EMAIL_RE.sub(" ", text)
    cleaned = _URL_RE.sub(" ", cleaned)
    return cleaned


def _match_phone(text: str) -> bool:
    for m in _PHONE_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        # Require a plausible phone-length digit count.
        if 7 <= len(digits) <= 15:
            return True
    return False


def _match_thai_person(text: str) -> bool:
    """True when the text looks like a Thai person name.

    Robust to OCR errors common on scanned Thai IDs:
      1. A Thai *name-field label* ("ชื่อตัวและชื่อสกุล", "นามสกุล", ...) followed
         by any Thai token -> the line carries a name (redact the whole line).
      2. A (possibly OCR-degraded) honorific ("น.ส.", "น.ส", "นส.", "น.", "นาย",
         ...) followed by a name-like token.
    Redacting a labeled/honorific-led line is the safe choice on identity
    documents, where a missed name is worse than an over-redacted label.
    """
    # 1. Name-field label + a following Thai token anywhere after the label.
    for label in _THAI_NAME_FIELD_LABELS:
        idx = text.find(label)
        if idx == -1:
            continue
        rest = text[idx + len(label):]
        if _THAI_CHAR_RE.search(rest):
            return True

    # 2. Degraded honorific followed by a name-like token.
    for prefix in _THAI_HONORIFIC_FUZZY:
        idx = text.find(prefix)
        if idx == -1:
            continue
        rest = text[idx + len(prefix):].lstrip()
        if rest and (_THAI_CHAR_RE.match(rest) or rest[0].isalpha()):
            return True

    # 3. Original (strict) honorific prefixes as a final fallback.
    for prefix in _THAI_PERSON_PREFIXES:
        idx = text.find(prefix)
        if idx == -1:
            continue
        rest = text[idx + len(prefix):].lstrip()
        if rest and (_THAI_CHAR_RE.match(rest) or rest[0].isalpha()):
            return True
    return False


def _match_thai_org(text: str) -> bool:
    """True when a Thai organization keyword appears in the text."""
    return any(keyword in text for keyword in _THAI_ORG_KEYWORDS)


def _tokenize_latin(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z.\-']*", text)


def _match_latin_org(text: str) -> bool:
    """True when a Latin organization keyword appears as a token."""
    for tok in _tokenize_latin(text):
        cleaned = tok.rstrip(".").lower()
        if cleaned in _LATIN_ORG_KEYWORDS:
            return True
    return False


def _match_latin_person(text: str) -> bool:
    """Heuristic for Latin person names.

    Accepts either:
      - a known title (Mr., Dr., ...) followed by a capitalized word, or
      - a capitalized token from the seed given-name gazetteer, or
      - two or more consecutive capitalized alphabetic words (First Last),
        excluding strings that look like organizations.
    """
    # A Latin name-field label marks the line as carrying a name, even when OCR
    # joins it to the name ("Last namePhuaboonmee") or leaves only the label
    # with a garbled name. Strip EVERY known label out first, then require an
    # alphabetic remainder -- so a bare label ("Name", "Last name") with no
    # accompanying value does not fire.
    lowered_text = text.lower()
    has_label = any(label in lowered_text for label in _LATIN_NAME_FIELD_LABELS)
    if has_label:
        remainder = lowered_text
        for label in _LATIN_NAME_FIELD_LABELS:
            remainder = remainder.replace(label, " ")
        if re.search(r"[a-z]", remainder):
            return True

    if _match_latin_org(text):
        # Organization keywords dominate; person-name heuristic should not fire
        # on e.g. "Acme Corp" purely from the capitalization rule.
        # (A titled/gazetteer name can still match below.)
        pass

    tokens = _tokenize_latin(text)
    if not tokens:
        return False

    lowered = [t.rstrip(".").lower() for t in tokens]

    # Title + following capitalized word.
    for i, tok in enumerate(lowered):
        if tok in _LATIN_PERSON_TITLES and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt[:1].isupper():
                return True

    # Seed given-name gazetteer hit.
    if any(tok in _LATIN_GIVEN_NAMES for tok in lowered):
        return True

    # Two or more consecutive capitalized alphabetic words -> "First Last".
    if _match_latin_org(text):
        return False
    capitalized_run = 0
    for tok in tokens:
        core = tok.rstrip(".").replace("-", "").replace("'", "")
        if core[:1].isupper() and core[1:].islower() and core.isalpha() and len(core) >= 2:
            capitalized_run += 1
            if capitalized_run >= 2:
                return True
        else:
            capitalized_run = 0
    return False


def _match_api_key(text: str) -> bool:
    if _OPENAI_KEY_RE.search(text):
        return True
    if _AWS_KEY_RE.search(text):
        return True
    if _GOOGLE_KEY_RE.search(text):
        return True
    if _APIKEY_ASSIGN_RE.search(text):
        return True
    if _KEY_ASSIGN_RE.search(text):
        return True
    return False


def _match_access_token(text: str) -> bool:
    if _JWT_RE.search(text):
        return True
    if _GITHUB_TOKEN_RE.search(text):
        return True
    if _BEARER_RE.search(text):
        return True
    if _TOKEN_ASSIGN_RE.search(text):
        return True
    return False


def _match_secret(text: str) -> bool:
    if _PEM_RE.search(text):
        return True
    if _SECRET_ASSIGN_RE.search(text):
        return True
    if _URL_CREDENTIALS_RE.search(text):
        return True
    return False


# --- Thailand-specific matchers ---------------------------------------------


def _is_valid_thai_id_checksum(digits: str) -> bool:
    """Validate a 13-digit Thai national ID via its mod-11 check digit.

    The first 12 digits are weighted by 13, 12, ..., 2; the check digit is
    ``(11 - (weighted_sum % 11)) % 10`` and must equal the 13th digit. This is
    the official algorithm, so a passing string is almost certainly a real ID
    rather than a coincidental 13-digit number.
    """
    if len(digits) != 13 or not digits.isdigit():
        return False
    total = sum(int(digits[i]) * (13 - i) for i in range(12))
    check = (11 - (total % 11)) % 10
    return check == int(digits[12])


def _match_thai_national_id(text: str) -> bool:
    """True when the text contains a checksum-valid 13-digit Thai national ID."""
    for m in _THAI_ID_CANDIDATE_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if _is_valid_thai_id_checksum(digits):
            return True
    return False


def _match_date(text: str) -> bool:
    """True when the text contains a date (numeric or Thai/Latin month-name)."""
    if _DATE_NUMERIC_RE.search(text):
        return True
    if _LATIN_MONTH_DATE_RE.search(text):
        return True
    # Thai month token adjacent to a year-like number (e.g. "18 ต.ค. 2546").
    for month in _THAI_MONTHS:
        if month in text:
            # Require a nearby 4-digit year so a bare month word isn't a date.
            if re.search(r"\d{4}", text):
                return True
    return False


def _match_passport(text: str) -> bool:
    """True when the text contains a passport-number-shaped token."""
    return bool(_PASSPORT_RE.search(text))


def _match_license_plate(text: str) -> bool:
    """True when the text contains a Thai vehicle registration plate.

    Requires at least one Thai consonant adjacent to digits, which is the
    defining shape of a Thai plate and avoids matching plain numbers.
    """
    return bool(_THAI_PLATE_RE.search(text))


def _match_bank_account(text: str) -> bool:
    """True when the text contains a grouped Thai bank account number."""
    return bool(_BANK_ACCOUNT_RE.search(text))


def _match_job_title(text: str) -> bool:
    """True when the text contains a job-title / position keyword.

    Thai keywords are matched as substrings (Thai script has no word spacing),
    while Latin titles are matched as multi-word phrases (substring, e.g.
    "vice president") or as whole lowercased tokens (e.g. "director") to avoid
    matching inside unrelated words. Titles are treated as PII because, combined
    with a name or organization, they help re-identify a person.
    """
    # Thai job-title keywords: substring match (no word boundaries in Thai).
    if any(keyword in text for keyword in _THAI_JOB_TITLE_KEYWORDS):
        return True

    lowered = text.lower()

    # Latin multi-word phrases: substring match on the lowercased text.
    if any(phrase in lowered for phrase in _LATIN_JOB_TITLE_PHRASES):
        return True

    # Latin single-word titles: whole-token match (strip trailing punctuation).
    for tok in _tokenize_latin(text):
        if tok.rstrip(".").lower() in _LATIN_JOB_TITLE_WORDS:
            return True
    return False


def _match_thai_address(text: str) -> bool:
    """True when the text looks like a Thai postal address.

    Signalled by a full-word address keyword (บ้านเลขที่/ถนน/ตำบล/อำเภอ/จังหวัด/
    ...), OR an abbreviated marker (ต. อ. จ.) followed by a Thai place name --
    but NOT when that abbreviation is actually a month token such as "ต.ค."
    (October). A bare 5-digit postal code alone is NOT treated as an address.
    """
    if any(kw in text for kw in _THAI_ADDRESS_KEYWORDS):
        return True

    # Abbreviated markers, excluding month abbreviations like "ต.ค.".
    for m in _THAI_ADDRESS_ABBREV_RE.finditer(text):
        span = m.group(0)
        # Reject if this is the head of a month abbreviation (e.g. "ต.ค").
        if span.replace(" ", "").startswith("ต.ค"):
            continue
        return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_segment(text: str) -> set[SensitiveCategory]:
    """Classify a single text segment into zero or more sensitive categories.

    This is the deterministic, pattern-based classifier. It is **total**: it
    never raises for any string input and always returns a set (empty when the
    text is non-sensitive, whitespace-only, or empty). A segment may match more
    than one category, in which case every matching category is included.

    Args:
        text: The recognized text segment to classify. Any string is accepted;
            non-``str`` inputs are coerced defensively so the function stays
            total.

    Returns:
        A ``set[SensitiveCategory]`` of every category the text matches. The
        empty set denotes non-sensitive text.

    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10, 3.11,
    4.1, 4.2, 4.3.
    """
    # Defensive coercion keeps the function total even if a caller passes a
    # non-string (Requirement 3.1: classification must never raise).
    if text is None:
        return set()
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return set()

    # Empty or whitespace-only text is non-sensitive (Requirement 3.11).
    if not text.strip():
        return set()

    categories: set[SensitiveCategory] = set()

    # --- Thailand-specific identifiers (checked early) ---
    # The national ID is checksum-validated, so it is highly reliable. It is
    # detected before PHONE_NUMBER so a 13-digit ID is not additionally mislabeled
    # as a phone number (a valid ID is not a phone number).
    is_thai_id = _match_thai_national_id(text)
    if is_thai_id:
        categories.add(SensitiveCategory.THAI_NATIONAL_ID)

    if _match_bank_account(text):
        categories.add(SensitiveCategory.BANK_ACCOUNT)

    if _match_passport(text):
        categories.add(SensitiveCategory.PASSPORT_NUMBER)

    if _match_license_plate(text):
        categories.add(SensitiveCategory.LICENSE_PLATE)

    if _match_date(text):
        categories.add(SensitiveCategory.DATE_OF_BIRTH)

    if _match_thai_address(text):
        categories.add(SensitiveCategory.THAI_ADDRESS)

    if _match_job_title(text):
        categories.add(SensitiveCategory.JOB_TITLE)

    # --- Structured personal-information patterns (Requirements 3.4-3.8) ---
    if _EMAIL_RE.search(text):
        categories.add(SensitiveCategory.EMAIL)

    if _URL_RE.search(text):
        categories.add(SensitiveCategory.URL)

    # A checksum-valid national ID should not also be tagged as a phone number.
    if not is_thai_id and _match_phone(text):
        categories.add(SensitiveCategory.PHONE_NUMBER)

    if _MONEY_RE.search(text):
        categories.add(SensitiveCategory.MONEY_AMOUNT)

    if _PERCENT_RE.search(text):
        categories.add(SensitiveCategory.PERCENT_VALUE)

    # --- Credentials / security tokens (Requirements 4.1-4.3) ---
    if _match_access_token(text):
        categories.add(SensitiveCategory.ACCESS_TOKEN)

    if _match_api_key(text):
        categories.add(SensitiveCategory.API_KEY)

    if _match_secret(text):
        categories.add(SensitiveCategory.SECRET)

    # Fallback high-entropy long token -> treat as an API key when nothing more
    # specific matched it (avoids double-tagging JWTs/emails/urls, and Thai
    # identifiers such as a national ID or bank account).
    if (
        SensitiveCategory.API_KEY not in categories
        and SensitiveCategory.ACCESS_TOKEN not in categories
        and SensitiveCategory.EMAIL not in categories
        and SensitiveCategory.URL not in categories
        and SensitiveCategory.THAI_NATIONAL_ID not in categories
        and SensitiveCategory.BANK_ACCOUNT not in categories
        and SensitiveCategory.PASSPORT_NUMBER not in categories
    ):
        for m in _LONG_ALNUM_RE.finditer(text):
            categories.add(SensitiveCategory.API_KEY)
            break

    # --- Names and organizations: delegated to the LLM classifier ---
    # PERSON_NAME / ORGANIZATION_NAME are intentionally NOT produced by the
    # deterministic patterns anymore. The old regex heuristics (Thai honorific
    # prefixes, Latin "two capitalized words = a name", org keyword lists) were
    # high-recall but noisy: they flagged ordinary headings/labels such as
    # "Tuition Fee", "Grand Total", or "Payment Method" as person names. Names
    # and organizations require context to judge, which the whole-page LLM
    # classifier (LiteLLMClassifier) does far more accurately. Patterns now stay
    # focused on what they detect reliably -- structured values (email, phone,
    # money, dates), credentials, and Thai identifiers (national ID, bank
    # account, license plate, address) -- and the LLM owns names/orgs.
    return categories


# ---------------------------------------------------------------------------
# Detector: pattern-based classification with an optional Classifier_Model
# ---------------------------------------------------------------------------

#: Warning appended to a DetectionOutcome when an enabled classifier is
#: unavailable (or raised) and the Detector fell back to pattern-based
#: classification. (Requirement 5.3)
_CLASSIFIER_FALLBACK_WARNING = (
    "Classifier_Model was enabled but unavailable at runtime; "
    "fell back to pattern-based classification."
)


@dataclass
class DetectionOutcome:
    """The result of running the Detector over a list of text segments.

    Attributes:
        regions: One :class:`~pii_guardrail.models.SensitiveRegion` per segment
            that classified into at least one category, preserving the source
            box, text, and confidence. Segments that classify as non-sensitive
            (including empty/whitespace-only segments) produce no region.
        warnings: Human-readable warning messages accumulated during detection,
            e.g. the classifier-unavailable fallback notice. Non-empty only when
            something noteworthy occurred.
        segment_categories: The categories assigned to EACH input segment, in
            input order (one entry per segment). An empty set means the segment
            was classified as non-sensitive and produced no region. This exposes
            the full per-segment classification -- including the non-sensitive
            segments that ``regions`` omits -- so callers can show WHY each
            segment was or was not redacted.
        segment_sources: A per-segment breakdown of WHICH classifier layer
            assigned WHICH categories, in input order (one entry per segment).
            Each entry maps a source label (``"pattern"`` for the deterministic
            regex, plus each model layer's ``source_name`` such as ``"llm"`` /
            ``"presidio"`` / ``"detect-secrets"``) to that layer's categories
            for the segment. Lets a debug view attribute each detection to its
            source instead of showing one merged set.
    """

    regions: list[SensitiveRegion] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    segment_categories: list[set[SensitiveCategory]] = field(default_factory=list)
    segment_sources: list[dict[str, set[SensitiveCategory]]] = field(
        default_factory=list
    )


class Detector:
    """Classifies text segments into sensitive categories and builds regions.

    Pattern-based by default. When ``use_classifier`` is enabled and a supplied
    :class:`~pii_guardrail.classifier.ClassifierModel` is available, the model's
    results are unioned with the pattern-based results. When the classifier is
    enabled but unavailable at runtime (its ``available`` property is ``False``)
    or a call to it raises
    :class:`~pii_guardrail.errors.ClassifierUnavailableError`, the Detector
    falls back to pattern-based classification and records a warning that
    :meth:`detect` surfaces on the :class:`DetectionOutcome`.

    The module-level :func:`classify_segment` remains the pattern-based source
    of truth; the Detector's :meth:`classify_segment` method calls it.

    Requirements: 5.1, 5.2, 5.3, 3.1, 3.10, 3.11.
    """

    def __init__(
        self,
        classifier: "ClassifierModel | None" = None,
        use_classifier: bool = False,
    ) -> None:
        """Create a Detector.

        Args:
            classifier: An optional model implementing the ``ClassifierModel``
                protocol. Ignored unless ``use_classifier`` is ``True``.
            use_classifier: When ``True``, consult ``classifier`` in addition to
                pattern-based classification (subject to availability). When
                ``False`` (the default), classification is pattern-only.
        """
        self._classifier = classifier
        self._use_classifier = use_classifier
        # Set by classify_segment when an enabled classifier is unavailable or
        # raises, so detect() can append the fallback warning exactly once.
        self._classifier_fallback_occurred = False

    def classify_segment(self, text: str) -> set[SensitiveCategory]:
        """Classify one text segment into zero or more sensitive categories.

        Behavior:
            - When the classifier is disabled (or none was supplied), the result
              equals pattern-based classification alone (Requirement 5.2).
            - When enabled and the model is available, the result is the union
              of the pattern-based categories and the model's categories
              (Requirement 5.1).
            - When enabled but the model is unavailable, or a call to the model
              raises :class:`ClassifierUnavailableError`, the Detector falls back
              to pattern-based categories and flags that a fallback occurred so
              :meth:`detect` can record a warning (Requirement 5.3).

        This method never raises for a classifier availability problem; the
        underlying :func:`classify_segment` is itself total.
        """
        # Pattern-based classification is always the source of truth.
        categories = classify_segment(text)

        if not self._use_classifier or self._classifier is None:
            return categories

        # Enabled: consult the model, guarding both the availability flag and a
        # raised ClassifierUnavailableError (either triggers fallback + warning).
        try:
            if not self._classifier.available:
                self._classifier_fallback_occurred = True
                return categories
            model_categories = self._classifier.classify(text)
        except ClassifierUnavailableError:
            self._classifier_fallback_occurred = True
            return categories

        # Union pattern results with model results (model is advisory).
        return categories | set(model_categories)

    def _whole_page_categories(
        self, segments: list[TextSegment]
    ) -> dict[int, set[SensitiveCategory]]:
        """Consult a whole-page-capable classifier once for all segments.

        A classifier MAY expose an optional ``classify_segments`` method that
        receives the full ordered list of segments and returns a mapping of
        ``segment index -> set[SensitiveCategory]``. This lets a context-aware
        model (e.g. an LLM) reason over the ENTIRE page at once -- so it can tell
        that a bare name in the "Received from" row is a person name -- and be
        invoked a single time per image rather than once per segment.

        Returns an empty mapping when the classifier is disabled, absent, does
        not implement ``classify_segments``, is unavailable, or raises. In the
        unavailable/raise cases the fallback flag is set so :meth:`detect`
        records the standard warning and relies on patterns (+ any per-segment
        classifier path) instead.
        """
        if not self._use_classifier or self._classifier is None:
            return {}

        classify_segments = getattr(self._classifier, "classify_segments", None)
        if not callable(classify_segments):
            return {}

        try:
            if not self._classifier.available:
                self._classifier_fallback_occurred = True
                return {}
            raw = classify_segments(list(segments))
        except ClassifierUnavailableError:
            self._classifier_fallback_occurred = True
            return {}
        except Exception:  # noqa: BLE001 - a misbehaving model must not break detection
            self._classifier_fallback_occurred = True
            return {}

        # Coerce defensively: accept a dict[int, iterable-of-SensitiveCategory].
        result: dict[int, set[SensitiveCategory]] = {}
        if not isinstance(raw, dict):
            return {}
        for index, categories in raw.items():
            try:
                idx = int(index)
            except (TypeError, ValueError):
                continue
            try:
                cats = {c for c in categories if isinstance(c, SensitiveCategory)}
            except TypeError:
                continue
            if cats:
                result[idx] = cats
        return result

    def _whole_page_by_source(
        self, segments: list[TextSegment]
    ) -> dict[int, dict[str, set[SensitiveCategory]]]:
        """Per-SOURCE whole-page categories, when the classifier can provide them.

        Calls the classifier's optional ``classify_segments_by_source`` (exposed
        by :class:`~pii_guardrail.composite.CompositeClassifier`) so the Detector
        can attribute each detection to the layer that produced it. Returns an
        empty mapping when the classifier is disabled, absent, lacks the method,
        is unavailable, or raises -- in the unavailable/raise cases the fallback
        flag is set (mirroring :meth:`_whole_page_categories`).
        """
        if not self._use_classifier or self._classifier is None:
            return {}

        by_source = getattr(self._classifier, "classify_segments_by_source", None)
        if not callable(by_source):
            return {}

        try:
            if not self._classifier.available:
                self._classifier_fallback_occurred = True
                return {}
            raw = by_source(list(segments))
        except ClassifierUnavailableError:
            self._classifier_fallback_occurred = True
            return {}
        except Exception:  # noqa: BLE001 - a misbehaving model must not break detection
            self._classifier_fallback_occurred = True
            return {}

        # Coerce defensively into dict[int, dict[str, set[SensitiveCategory]]].
        result: dict[int, dict[str, set[SensitiveCategory]]] = {}
        if not isinstance(raw, dict):
            return {}
        for index, sources in raw.items():
            try:
                idx = int(index)
            except (TypeError, ValueError):
                continue
            if not isinstance(sources, dict):
                continue
            clean: dict[str, set[SensitiveCategory]] = {}
            for src, cats in sources.items():
                try:
                    valid = {c for c in cats if isinstance(c, SensitiveCategory)}
                except TypeError:
                    continue
                if valid and isinstance(src, str):
                    clean[src] = valid
            if clean:
                result[idx] = clean
        return result

    def detect(self, segments: list[TextSegment]) -> DetectionOutcome:
        """Build sensitive regions for all sensitive segments.

        For each segment, classify its text. A segment whose category set is
        non-empty yields a :class:`~pii_guardrail.models.SensitiveRegion` that
        preserves the segment's box, text, and confidence. Segments that
        classify as non-sensitive (including empty/whitespace-only segments,
        Requirement 3.11) produce no region.

        When the configured classifier supports WHOLE-PAGE classification (it
        exposes ``classify_segments``), it is consulted ONCE with all segments
        and its per-index categories are unioned with the pattern-based (and any
        per-segment classifier) results. Otherwise the per-segment path is used.

        Any classifier-unavailable fallback that occurred while classifying the
        segments is surfaced as a single warning on the returned outcome
        (Requirement 5.3).
        """
        # Reset per-call so warnings reflect only this detect() invocation.
        self._classifier_fallback_occurred = False

        # Whole-page pass (single call) when the classifier supports it. We also
        # request the per-SOURCE breakdown (which layer flagged what) when the
        # classifier can provide it, so debug views can attribute detections.
        classifier_is_whole_page = callable(
            getattr(self._classifier, "classify_segments", None)
        )
        by_source = self._whole_page_by_source(segments)
        # Merged whole-page categories (union across layers), reused for regions.
        whole_page: dict[int, set[SensitiveCategory]]
        if by_source:
            whole_page = {
                idx: set().union(*sources.values()) if sources else set()
                for idx, sources in by_source.items()
            }
        else:
            whole_page = self._whole_page_categories(segments)

        regions: list[SensitiveRegion] = []
        segment_categories: list[set[SensitiveCategory]] = []
        segment_sources: list[dict[str, set[SensitiveCategory]]] = []
        for index, segment in enumerate(segments):
            pattern_cats = classify_segment(segment.text)
            if classifier_is_whole_page:
                # Patterns are the source of truth; union the whole-page model's
                # categories for this segment index (empty if none / unavailable).
                categories = pattern_cats | whole_page.get(index, set())
            else:
                categories = self.classify_segment(segment.text)

            # Per-source breakdown: always include the deterministic "pattern"
            # layer, plus each model layer's contribution for this segment.
            sources: dict[str, set[SensitiveCategory]] = {}
            if pattern_cats:
                sources["pattern"] = set(pattern_cats)
            for src, cats in by_source.get(index, {}).items():
                if cats:
                    sources[src] = set(cats)
            segment_sources.append(sources)

            # Record the categories for EVERY segment (aligned by index), so the
            # non-sensitive segments -- which produce no region -- are still
            # visible to callers that want the full classification breakdown.
            segment_categories.append(set(categories))
            if not categories:
                continue
            regions.append(
                SensitiveRegion(
                    box=segment.box,
                    categories=categories,
                    text=segment.text,
                    confidence=segment.confidence,
                )
            )

        warnings: list[str] = []
        if self._classifier_fallback_occurred:
            warnings.append(_CLASSIFIER_FALLBACK_WARNING)

        return DetectionOutcome(
            regions=regions,
            warnings=warnings,
            segment_sources=segment_sources,
            segment_categories=segment_categories,
        )
