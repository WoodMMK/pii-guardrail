"""Unit tests for PaddleOCR 3.x word-level box parsing.

These test ``PaddleOCRBackend._parse_v3_result`` (and its ``_word_segments_for_line``
helper) directly with FAKE PaddleOCR result objects, so no PyTorch/PaddleOCR is
required. They pin the behavior that makes redaction boxes tight:

    - When a line exposes per-word text (``text_word``) + regions
      (``text_word_region``), the line is split into ONE TextSegment per word,
      each with its own tight box (fixes the "whole-line black box" problem).
    - Whitespace-only words are dropped.
    - When word data is absent or malformed, the parser falls back to a single
      line-level segment (the pre-existing behavior), so nothing regresses.

The fake mirrors the real 3.x ``OCRResult``: a dict-like object exposing
``rec_texts`` / ``rec_scores`` / ``rec_polys`` and optionally ``text_word`` /
``text_word_region``.
"""

from __future__ import annotations

import pytest

from pii_guardrail.models import TextSegment
from pii_guardrail.paddle_backend import PaddleOCRBackend

pytestmark = pytest.mark.unit


def _box_quad(x0, y0, x1, y1):
    """A rectangular quad (4 vertices) spanning (x0,y0)-(x1,y1)."""
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


class _FakeV3Result(dict):
    """Dict-like stand-in for a PaddleOCR 3.x per-image OCRResult."""


def _parse(result_dict, width=1000, height=400) -> list[TextSegment]:
    """Run the 3.x parser over a single fake per-image result."""
    return PaddleOCRBackend._parse_v3_result(
        [_FakeV3Result(result_dict)], width, height
    )


class TestWordLevelSplitting:
    def test_line_split_into_per_word_segments(self) -> None:
        # One recognized line, but three words with individual regions.
        result = {
            "rec_texts": ["Received นายเอ 6310710025"],
            "rec_scores": [0.9],
            "rec_polys": [_box_quad(10, 40, 900, 90)],
            "text_word": [["Received", " นายเอ ", "6310710025"]],
            "text_word_region": [
                [
                    _box_quad(10, 40, 160, 90),
                    _box_quad(170, 40, 620, 90),
                    _box_quad(630, 40, 890, 90),
                ]
            ],
        }
        segs = _parse(result)

        # Three word segments (not one wide line segment).
        assert len(segs) == 3
        texts = [s.text for s in segs]
        assert texts == ["Received", " นายเอ ", "6310710025"]

        # Each box is tight: none spans the whole line width (900px).
        for s in segs:
            assert s.box.width < 700  # each word is far narrower than the line
        # The first word starts near x=10 and is short.
        first = segs[0]
        assert first.box.x <= 12
        assert first.box.width <= 170

    def test_whitespace_words_are_dropped(self) -> None:
        result = {
            "rec_texts": ["A   B"],
            "rec_scores": [0.8],
            "rec_polys": [_box_quad(0, 0, 500, 50)],
            "text_word": [["A", "   ", "B"]],
            "text_word_region": [
                [
                    _box_quad(0, 0, 40, 50),
                    _box_quad(45, 0, 120, 50),  # whitespace-only -> dropped
                    _box_quad(125, 0, 165, 50),
                ]
            ],
        }
        segs = _parse(result)
        assert [s.text for s in segs] == ["A", "B"]

    def test_confidence_carried_to_each_word(self) -> None:
        result = {
            "rec_texts": ["x y"],
            "rec_scores": [0.77],
            "rec_polys": [_box_quad(0, 0, 200, 40)],
            "text_word": [["x", "y"]],
            "text_word_region": [
                [_box_quad(0, 0, 40, 40), _box_quad(50, 0, 90, 40)]
            ],
        }
        segs = _parse(result)
        assert len(segs) == 2
        assert all(abs(s.confidence - 0.77) < 1e-6 for s in segs)


class TestFallbackToLineLevel:
    def test_no_word_data_falls_back_to_line(self) -> None:
        # No text_word / text_word_region -> single line-level segment.
        result = {
            "rec_texts": ["whole line here"],
            "rec_scores": [0.95],
            "rec_polys": [_box_quad(10, 40, 900, 90)],
        }
        segs = _parse(result)
        assert len(segs) == 1
        assert segs[0].text == "whole line here"
        # Line-level box spans most of the width.
        assert segs[0].box.width > 800

    def test_mismatched_word_counts_fall_back(self) -> None:
        # words and regions length mismatch -> unusable -> line fallback.
        result = {
            "rec_texts": ["a b c"],
            "rec_scores": [0.9],
            "rec_polys": [_box_quad(0, 0, 300, 40)],
            "text_word": [["a", "b", "c"]],
            "text_word_region": [[_box_quad(0, 0, 40, 40)]],  # only 1 region
        }
        segs = _parse(result)
        assert len(segs) == 1
        assert segs[0].text == "a b c"

    def test_malformed_word_region_falls_back(self) -> None:
        # A region that cannot form a quad -> fall back to the line box.
        result = {
            "rec_texts": ["a b"],
            "rec_scores": [0.9],
            "rec_polys": [_box_quad(0, 0, 300, 40)],
            "text_word": [["a", "b"]],
            "text_word_region": [["not-a-quad", "also-bad"]],
        }
        segs = _parse(result)
        assert len(segs) == 1
        assert segs[0].text == "a b"

    def test_empty_result_yields_no_segments(self) -> None:
        assert _parse({"rec_texts": []}) == []


class TestMultipleLines:
    def test_mixed_word_and_line_fallback_across_lines(self) -> None:
        # Line 0 has word data (split); line 1 has none (line fallback).
        result = {
            "rec_texts": ["p q", "plain line"],
            "rec_scores": [0.9, 0.8],
            "rec_polys": [_box_quad(0, 0, 200, 40), _box_quad(0, 60, 800, 100)],
            "text_word": [["p", "q"], None],
            "text_word_region": [
                [_box_quad(0, 0, 40, 40), _box_quad(50, 0, 90, 40)],
                None,
            ],
        }
        segs = _parse(result)
        texts = [s.text for s in segs]
        assert texts == ["p", "q", "plain line"]
