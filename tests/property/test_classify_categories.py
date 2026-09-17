"""Property test for :func:`pii_guardrail.detector.classify_segment`.

Feature: thai-image-pii-guardrail, Property 8: Category-bearing text is
classified into its category.

Validates: Requirements 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.1, 4.2, 4.3.

The strategies below are built to conform *exactly* to the deterministic
matchers in :mod:`pii_guardrail.detector` (regexes, gazetteers, and lexical
heuristics), so every generated example is guaranteed to be recognized. When a
shape is not matched, the generator is adjusted here -- never the implementation.
"""

from __future__ import annotations

import string

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pii_guardrail.detector import (
    _LATIN_GIVEN_NAMES,
    _LATIN_ORG_KEYWORDS,
    _LATIN_PERSON_TITLES,
    _THAI_ORG_KEYWORDS,
    _THAI_PERSON_PREFIXES,
    classify_segment,
)
from pii_guardrail.models import SensitiveCategory as Cat

pytestmark = pytest.mark.property

# ---------------------------------------------------------------------------
# Alphabets constrained to what each matcher accepts.
# ---------------------------------------------------------------------------

_DIGITS = st.sampled_from(string.digits)
_ALNUM = st.sampled_from(string.ascii_letters + string.digits)
_UPPER_ALNUM = st.sampled_from(string.ascii_uppercase + string.digits)
_B64URL = st.sampled_from(string.ascii_letters + string.digits + "_-")
_GOOGLE_CHARS = st.sampled_from(string.ascii_letters + string.digits + "_-")

# A handful of common single Thai consonants/vowels for building name tokens.
_THAI_NAME_CHARS = "กขคงจฉชญฐณดตถทนบปผฝพฟภมยรลวสหอ"


def _digits(n: int) -> st.SearchStrategy[str]:
    return st.lists(_DIGITS, min_size=n, max_size=n).map("".join)


def _alnum(min_size: int, max_size: int) -> st.SearchStrategy[str]:
    return st.lists(_ALNUM, min_size=min_size, max_size=max_size).map("".join)


# ---------------------------------------------------------------------------
# Category-shaped strategies. Each yields (category, conforming_text).
# ---------------------------------------------------------------------------


@st.composite
def _phone(draw) -> tuple[Cat, str]:
    """Thai local ``0xx-xxx-xxxx`` or international ``+66 ...`` forms."""
    if draw(st.booleans()):
        # Thai local: 0\d{1,2}[sep]\d{3}[sep]\d{3,4}; total digits 7..15.
        lead = "0" + draw(_digits(draw(st.integers(min_value=1, max_value=2))))
        mid = draw(_digits(3))
        tail = draw(_digits(draw(st.integers(min_value=3, max_value=4))))
        sep = draw(st.sampled_from(["-", " ", ".", ""]))
        text = f"{lead}{sep}{mid}{sep}{tail}"
    else:
        # International: +<1-3 digit cc> then 3-4 groups of 2-4 digits.
        # Guarantee the total digit count lands in the matcher's 7..15 window.
        cc = draw(_digits(draw(st.integers(min_value=1, max_value=3))))
        groups = draw(
            st.lists(
                st.integers(min_value=2, max_value=4).flatmap(_digits),
                min_size=3,
                max_size=4,
            )
        )
        sep = draw(st.sampled_from(["-", " ", "."]))
        text = "+" + cc + sep + sep.join(groups)
        # Clamp so we never exceed the 15-digit ceiling.
        while len(text.replace("+", "").replace(sep, "")) > 15:
            groups = groups[:-1]
            text = "+" + cc + sep + sep.join(groups)
    return Cat.PHONE_NUMBER, text


@st.composite
def _url(draw) -> tuple[Cat, str]:
    label = draw(st.text(alphabet=st.sampled_from(string.ascii_lowercase + string.digits), min_size=1, max_size=8))
    kind = draw(st.sampled_from(["scheme", "www", "tld"]))
    if kind == "scheme":
        scheme = draw(st.sampled_from(["http://", "https://", "ftp://"]))
        return Cat.URL, f"{scheme}{label}.example.com/path"
    if kind == "www":
        return Cat.URL, f"www.{label}.example.com"
    # bare domain + known TLD
    tld = draw(st.sampled_from(["com", "net", "org", "io", "co", "dev", "ai", "th"]))
    return Cat.URL, f"{label}.example.{tld}"


@st.composite
def _email(draw) -> tuple[Cat, str]:
    local = draw(st.text(alphabet=st.sampled_from(string.ascii_lowercase + string.digits + "._%+-"), min_size=1, max_size=10))
    # Ensure local part isn't only separator chars (needs the regex class match).
    local = "u" + local
    domain = draw(st.text(alphabet=st.sampled_from(string.ascii_lowercase + string.digits), min_size=1, max_size=8))
    tld = draw(st.sampled_from(["com", "net", "org", "io", "co"]))
    return Cat.EMAIL, f"{local}@{domain}.{tld}"


@st.composite
def _money(draw) -> tuple[Cat, str]:
    # Use a plain integer 1..3 digits (matches _NUMBER without needing groups).
    number = draw(st.integers(min_value=1, max_value=999)).__str__()
    form = draw(st.sampled_from(["symbol_pre", "symbol_post", "code_pre", "code_post", "baht"]))
    space = draw(st.sampled_from(["", " "]))
    if form == "symbol_pre":
        sym = draw(st.sampled_from(["฿", "$", "€", "£", "¥"]))
        return Cat.MONEY_AMOUNT, f"{sym}{space}{number}"
    if form == "symbol_post":
        sym = draw(st.sampled_from(["฿", "$", "€", "£", "¥"]))
        return Cat.MONEY_AMOUNT, f"{number}{space}{sym}"
    if form == "code_pre":
        code = draw(st.sampled_from(["THB", "USD", "EUR", "GBP", "JPY"]))
        return Cat.MONEY_AMOUNT, f"{code}{space}{number}"
    if form == "code_post":
        code = draw(st.sampled_from(["THB", "USD", "EUR", "GBP", "JPY"]))
        return Cat.MONEY_AMOUNT, f"{number}{space}{code}"
    return Cat.MONEY_AMOUNT, f"{number}{space}บาท"


@st.composite
def _percent(draw) -> tuple[Cat, str]:
    number = draw(st.integers(min_value=0, max_value=1000)).__str__()
    if draw(st.booleans()):
        frac = draw(st.integers(min_value=0, max_value=99)).__str__()
        sep = draw(st.sampled_from([".", ","]))
        number = f"{number}{sep}{frac}"
    space = draw(st.sampled_from(["", " "]))
    return Cat.PERCENT_VALUE, f"{number}{space}%"


@st.composite
def _api_key(draw) -> tuple[Cat, str]:
    kind = draw(st.sampled_from(["openai", "aws", "google", "assign"]))
    if kind == "openai":
        proj = "proj-" if draw(st.booleans()) else ""
        body = draw(_alnum(16, 40))
        return Cat.API_KEY, f"sk-{proj}{body}"
    if kind == "aws":
        prefix = draw(st.sampled_from(["AKIA", "ASIA"]))
        body = draw(st.lists(_UPPER_ALNUM, min_size=16, max_size=16).map("".join))
        return Cat.API_KEY, f"{prefix}{body}"
    if kind == "google":
        body = draw(st.lists(_GOOGLE_CHARS, min_size=35, max_size=35).map("".join))
        return Cat.API_KEY, f"AIza{body}"
    # explicit assignment: api_key: <6+ non-space>
    key = draw(st.sampled_from(["api_key", "apikey", "api-key", "client_secret", "secret_key"]))
    sep = draw(st.sampled_from([": ", "=", ":", " = "]))
    value = draw(st.text(alphabet=st.sampled_from(string.ascii_letters + string.digits + "-_"), min_size=6, max_size=20))
    return Cat.API_KEY, f"{key}{sep}{value}"


@st.composite
def _access_token(draw) -> tuple[Cat, str]:
    kind = draw(st.sampled_from(["jwt", "github", "bearer", "assign"]))
    if kind == "jwt":
        seg = lambda: draw(st.lists(_B64URL, min_size=1, max_size=12).map("".join))
        # Header must start with the literal "eyJ".
        header = "eyJ" + "".join(draw(st.lists(_B64URL, min_size=1, max_size=8)))
        return Cat.ACCESS_TOKEN, f"{header}.{seg()}.{seg()}"
    if kind == "github":
        prefix = draw(st.sampled_from(["ghp_", "gho_", "ghu_", "ghs_", "ghr_"]))
        body = draw(_alnum(20, 40))
        return Cat.ACCESS_TOKEN, f"{prefix}{body}"
    if kind == "bearer":
        word = draw(st.sampled_from(["bearer", "Bearer", "BEARER"]))
        value = draw(st.text(alphabet=st.sampled_from(string.ascii_letters + string.digits + "._-"), min_size=8, max_size=24))
        return Cat.ACCESS_TOKEN, f"{word} {value}"
    key = draw(st.sampled_from(["access_token", "auth_token", "token", "access-token"]))
    sep = draw(st.sampled_from([": ", "=", ":", " = "]))
    value = draw(st.text(alphabet=st.sampled_from(string.ascii_letters + string.digits + "-_"), min_size=6, max_size=20))
    return Cat.ACCESS_TOKEN, f"{key}{sep}{value}"


@st.composite
def _thai_person(draw) -> tuple[Cat, str]:
    prefix = draw(st.sampled_from(list(_THAI_PERSON_PREFIXES)))
    name = draw(st.lists(st.sampled_from(_THAI_NAME_CHARS), min_size=2, max_size=6).map("".join))
    space = draw(st.sampled_from(["", " "]))
    return Cat.PERSON_NAME, f"{prefix}{space}{name}"


@st.composite
def _thai_org(draw) -> tuple[Cat, str]:
    keyword = draw(st.sampled_from(list(_THAI_ORG_KEYWORDS)))
    extra = draw(st.lists(st.sampled_from(_THAI_NAME_CHARS), min_size=0, max_size=5).map("".join))
    # Keyword alone is sufficient; adding a name-ish suffix is still valid.
    return Cat.ORGANIZATION_NAME, f"{keyword}{extra}"


def _cap_word(draw, min_len: int = 3, max_len: int = 8) -> str:
    first = draw(st.sampled_from(string.ascii_uppercase))
    rest = draw(st.lists(st.sampled_from(string.ascii_lowercase), min_size=min_len - 1, max_size=max_len - 1).map("".join))
    return first + rest


@st.composite
def _latin_person(draw) -> tuple[Cat, str]:
    kind = draw(st.sampled_from(["title", "gazetteer", "two_words"]))
    if kind == "title":
        title = draw(st.sampled_from(list(_LATIN_PERSON_TITLES)))
        # Title may be lower/upper; heuristic lowercases it. Following word Cap.
        title_text = title.capitalize() if draw(st.booleans()) else title
        name = _cap_word(draw)
        dot = "." if draw(st.booleans()) else ""
        return Cat.PERSON_NAME, f"{title_text}{dot} {name}"
    if kind == "gazetteer":
        name = draw(st.sampled_from(list(_LATIN_GIVEN_NAMES)))
        return Cat.PERSON_NAME, name.capitalize()
    # Two consecutive capitalized words -> "First Last".
    first = _cap_word(draw)
    last = _cap_word(draw)
    return Cat.PERSON_NAME, f"{first} {last}"


@st.composite
def _latin_org(draw) -> tuple[Cat, str]:
    keyword = draw(st.sampled_from(list(_LATIN_ORG_KEYWORDS)))
    # Match as a whole token; capitalization is normalized in the matcher.
    kw_text = keyword.capitalize() if draw(st.booleans()) else keyword
    name = _cap_word(draw)
    trailing_dot = "." if draw(st.booleans()) else ""
    return Cat.ORGANIZATION_NAME, f"{name} {kw_text}{trailing_dot}"


_CATEGORY_STRATEGIES = st.one_of(
    _phone(),
    _url(),
    _email(),
    _money(),
    _percent(),
    _api_key(),
    _access_token(),
    _thai_person(),
    _thai_org(),
    _latin_person(),
    _latin_org(),
)


# Feature: thai-image-pii-guardrail, Property 8: Category-bearing text is classified into its category
@settings(max_examples=100)
@given(sample=_CATEGORY_STRATEGIES)
def test_category_bearing_text_is_classified_into_its_category(
    sample: tuple[Cat, str],
) -> None:
    """A string shaped like a category is classified into (at least) that category.

    Validates: Requirements 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.1, 4.2, 4.3.
    """
    expected_category, text = sample
    result = classify_segment(text)
    assert isinstance(result, set)
    assert expected_category in result, (
        f"expected {expected_category!r} in classify_segment({text!r}) but got {result!r}"
    )
