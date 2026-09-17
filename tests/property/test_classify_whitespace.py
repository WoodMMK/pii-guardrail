"""Property-based test that whitespace-only text is non-sensitive.

# Feature: thai-image-pii-guardrail, Property 10: Whitespace-only text is non-sensitive

Validates: Requirements 3.11
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pii_guardrail.detector import Detector, classify_segment
from pii_guardrail.models import BoundingBox, TextSegment

# Characters that are whitespace-only: ASCII space, tab, newline, carriage
# return, vertical tab, form feed, plus a few representative Unicode whitespace
# code points (no-break space, em space, line separator).
_WHITESPACE_CHARS = [
    " ",
    "\t",
    "\n",
    "\r",
    "\x0b",  # vertical tab
    "\x0c",  # form feed
    "\u00a0",  # no-break space
    "\u2003",  # em space
    "\u2028",  # line separator
]

# Whitespace-only strings, including the empty string (min_size=0).
_whitespace_text = st.text(
    alphabet=st.sampled_from(_WHITESPACE_CHARS),
    min_size=0,
)


@pytest.mark.property
@settings(max_examples=200)
@given(text=_whitespace_text)
def test_whitespace_only_text_is_non_sensitive(text: str) -> None:
    """Whitespace-only text yields no categories and produces no region.

    For any whitespace-only string (including the empty string):
      - ``classify_segment`` returns the empty set, and
      - ``Detector.detect`` produces no region for a segment carrying that text.

    # Feature: thai-image-pii-guardrail, Property 10: Whitespace-only text is non-sensitive
    Validates: Requirements 3.11
    """
    # Classification must return the empty set (non-sensitive).
    assert classify_segment(text) == set()

    # A segment carrying whitespace-only text must produce no region.
    segment = TextSegment(
        text=text,
        box=BoundingBox(x=0, y=0, width=10, height=10),
        confidence=1.0,
    )
    outcome = Detector().detect([segment])
    assert outcome.regions == []
