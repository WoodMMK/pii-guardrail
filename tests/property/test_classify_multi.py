"""Property-based test for multi-category classification.

# Feature: thai-image-pii-guardrail, Property 9: Multi-category text yields all matching categories

Validates Requirement 3.10: a text segment that combines tokens from N distinct
sensitive categories must classify into (at least) all N of those categories.

Strategy design
---------------
We build one generator per *structured* category whose shape the deterministic
classifier in :mod:`pii_guardrail.detector` matches reliably and whose token does
not suppress the other tokens when concatenated with spaces:

    - EMAIL          -> ``local@domain.tld``
    - PHONE_NUMBER   -> Thai/international grouped digits (e.g. ``+66 2 123 4567``)
    - PERCENT_VALUE  -> ``<number>%``
    - MONEY_AMOUNT   -> currency symbol/code + number (e.g. ``THB 1200``)
    - API_KEY        -> AWS-style ``AKIA`` + 16 uppercase alphanumerics
    - ACCESS_TOKEN   -> GitHub-style ``ghp_`` + >=20 alphanumerics

These structured categories are independent of the name/organization heuristics,
so combining them cannot trigger the name-suppression logic. We deliberately
avoid the URL category because its broad pattern also matches the domain inside
an email token, which would make per-category isolation ambiguous; superset
assertions stay clean without it.

For each example we pick N (>= 2) distinct categories, generate one token per
category, join the tokens with a non-numeric separator, and assert that
``classify_segment(combined)`` is a superset of the chosen categories.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pii_guardrail.detector import classify_segment
from pii_guardrail.models import SensitiveCategory

# ---------------------------------------------------------------------------
# Per-category token strategies (structured, non-interfering shapes)
# ---------------------------------------------------------------------------

_EMAIL_LOCAL = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
    min_size=1,
    max_size=10,
)
_DOMAIN_LABEL = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
    min_size=1,
    max_size=8,
)
_EMAIL_TLD = st.sampled_from(["com", "org", "net", "io", "co"])


@st.composite
def _email_token(draw: st.DrawFn) -> str:
    local = draw(_EMAIL_LOCAL)
    domain = draw(_DOMAIN_LABEL)
    tld = draw(_EMAIL_TLD)
    return f"{local}@{domain}.{tld}"


@st.composite
def _phone_token(draw: st.DrawFn) -> str:
    # International grouped form using hyphen separators so the token stays a
    # single self-contained unit that cannot blend with a space-separated
    # neighbour (e.g. a money amount). Digit count lands in the 9-12 range.
    cc = draw(st.integers(min_value=1, max_value=99))
    g1 = draw(st.integers(min_value=100, max_value=999))
    g2 = draw(st.integers(min_value=100, max_value=999))
    g3 = draw(st.integers(min_value=1000, max_value=9999))
    return f"+{cc}-{g1}-{g2}-{g3}"


@st.composite
def _percent_token(draw: st.DrawFn) -> str:
    value = draw(st.integers(min_value=0, max_value=100))
    frac = draw(st.one_of(st.none(), st.integers(min_value=0, max_value=99)))
    spacer = draw(st.sampled_from(["", " "]))
    body = f"{value}" if frac is None else f"{value}.{frac}"
    return f"{body}{spacer}%"


@st.composite
def _money_token(draw: st.DrawFn) -> str:
    # Currency symbol/code adjacent to the number with NO space between the
    # currency marker and the digits, so the money token is self-contained and
    # its digits cannot be claimed by a neighbouring numeric token. We avoid the
    # bare "<number> บาท" form because its detached leading number can merge with
    # adjacent digit groups.
    amount = draw(st.integers(min_value=1, max_value=999_999))
    style = draw(st.sampled_from(["symbol", "code"]))
    if style == "symbol":
        sym = draw(st.sampled_from(["฿", "$", "€", "£", "¥"]))
        return f"{sym}{amount}"
    code = draw(st.sampled_from(["THB", "USD", "EUR", "GBP", "JPY"]))
    return f"{code}{amount}"


_UPPER_ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_TOKEN_ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


@st.composite
def _api_key_token(draw: st.DrawFn) -> str:
    # AWS access key id: AKIA/ASIA + exactly 16 uppercase alphanumerics.
    prefix = draw(st.sampled_from(["AKIA", "ASIA"]))
    body = draw(st.text(alphabet=_UPPER_ALNUM, min_size=16, max_size=16))
    return f"{prefix}{body}"


@st.composite
def _access_token_token(draw: st.DrawFn) -> str:
    # GitHub-style token: gh[pousr]_ + >=20 alphanumerics.
    kind = draw(st.sampled_from(["ghp", "gho", "ghu", "ghs", "ghr"]))
    body = draw(st.text(alphabet=_TOKEN_ALNUM, min_size=20, max_size=36))
    return f"{kind}_{body}"


# Map each category to the strategy that produces a matching token for it.
_CATEGORY_TOKENS: dict[SensitiveCategory, st.SearchStrategy[str]] = {
    SensitiveCategory.EMAIL: _email_token(),
    SensitiveCategory.PHONE_NUMBER: _phone_token(),
    SensitiveCategory.PERCENT_VALUE: _percent_token(),
    SensitiveCategory.MONEY_AMOUNT: _money_token(),
    SensitiveCategory.API_KEY: _api_key_token(),
    SensitiveCategory.ACCESS_TOKEN: _access_token_token(),
}

_ALL_CATEGORIES = list(_CATEGORY_TOKENS)


@st.composite
def _multi_category(draw: st.DrawFn) -> tuple[list[SensitiveCategory], str]:
    """Pick N (>= 2) distinct categories and build a combined text of one token each."""
    chosen = draw(
        st.lists(
            st.sampled_from(_ALL_CATEGORIES),
            min_size=2,
            max_size=len(_ALL_CATEGORIES),
            unique=True,
        )
    )
    tokens = [draw(_CATEGORY_TOKENS[cat]) for cat in chosen]
    # Join with a non-numeric, non-word separator (" | ") rather than a bare
    # space. A plain space lets a numeric token (e.g. a trailing phone group)
    # bleed into an adjacent numeric token (e.g. a following percent value),
    # which can defeat per-token pattern matching. The pipe separator keeps each
    # structured token self-contained without introducing any new category.
    combined = " | ".join(tokens)
    return chosen, combined


@pytest.mark.property
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(_multi_category())
def test_multi_category_text_yields_all_categories(
    payload: tuple[list[SensitiveCategory], str],
) -> None:
    # Feature: thai-image-pii-guardrail, Property 9: Multi-category text yields all matching categories
    chosen, combined = payload
    result = classify_segment(combined)
    assert result.issuperset(set(chosen)), (
        f"Expected all of {set(chosen)} in classification of {combined!r}, "
        f"got {result}"
    )
