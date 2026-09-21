"""Unit tests for sentence-level bounding box union and grouping logic.

Validates:
1. Converting word-level bounding boxes into single sentence-level bounding boxes (union).
2. Grouping segments on the same horizontal line into unified sentence segments when close.
3. Splitting segments on the same line when horizontal gap is large (e.g. table columns, UI elements).
4. Clamping bounding boxes to image dimensions.
"""

from __future__ import annotations

import pytest

from pii_guardrail.geometry import normalize_quad_to_box
from pii_guardrail.models import BoundingBox, TextSegment


def union_word_boxes(words: list[dict]) -> BoundingBox | None:
    """Python reference implementation of the JS unionBoxes function."""
    if not words:
        return None
    min_x = min(w["x"] for w in words)
    min_y = min(w["y"] for w in words)
    max_x = max(w["x"] + w["width"] for w in words)
    max_y = max(w["y"] + w["height"] for w in words)
    return BoundingBox(
        x=round(min_x),
        y=round(min_y),
        width=round(max_x - min_x),
        height=round(max_y - min_y),
    )


def group_words_into_sentences(
    words: list[dict],
    image_width: int,
    image_height: int,
    max_gap_ratio: float = 1.2,
) -> list[dict]:
    """Python reference of the JS groupWordsIntoSentences function with column margin alignment & gap checking."""
    if not words:
        return []

    # Step 1: Cluster into horizontal lines by vertical overlap/alignment
    sorted_words = sorted(words, key=lambda w: (w["box"]["y"], w["box"]["x"]))
    lines: list[list[dict]] = []

    for w in sorted_words:
        placed = False
        for line in lines:
            ref = line[0]
            avg_h = (ref["box"]["height"] + w["box"]["height"]) / 2
            center_y_diff = abs(
                (ref["box"]["y"] + ref["box"]["height"] / 2)
                - (w["box"]["y"] + w["box"]["height"] / 2)
            )
            if center_y_diff <= avg_h * 0.5:
                line.append(w)
                placed = True
                break
        if not placed:
            lines.append([w])

    # Step 2: Detect vertical column alignment guides across lines (tables/forms/multi-column)
    # Strict tolerance (+/- 6px) detects true vertical columns while preventing false matches on adjacent words
    col_x_starts = [seg["box"]["x"] for line in lines for seg in line]
    column_margins: list[int] = []
    for x in col_x_starts:
        match_count = sum(1 for line in lines if any(abs(s["box"]["x"] - x) <= 6 for s in line))
        if match_count >= 2 and not any(abs(cm - x) <= 6 for cm in column_margins):
            column_margins.append(x)

    # Step 3: For each line, sort left-to-right and split into separate columns/phrases
    sentence_groups: list[list[dict]] = []
    for line in lines:
        line.sort(key=lambda w: w["box"]["x"])
        current_group = [line[0]]

        for i in range(1, len(line)):
            prev = current_group[-1]
            curr = line[i]
            prev_right = prev["box"]["x"] + prev["box"]["width"]
            gap = curr["box"]["x"] - prev_right
            avg_h = (prev["box"]["height"] + curr["box"]["height"]) / 2
            max_gap = max(16, avg_h * max_gap_ratio)

            # Check if curr aligns with an established column boundary across lines
            is_column_boundary = any(
                abs(curr["box"]["x"] - cm) <= 6 and curr["box"]["x"] > current_group[0]["box"]["x"] + 30
                for cm in column_margins
            )

            # Normal word space within sentence -> keep in same group
            # Wide gap or column margin alignment -> start a new sentence/column group
            if gap <= max_gap and not is_column_boundary:
                current_group.append(curr)
            else:
                sentence_groups.append(current_group)
                current_group = [curr]

        if current_group:
            sentence_groups.append(current_group)

    # Step 4: Union bounding boxes for each sentence group
    sentences = []
    for group in sentence_groups:
        box = union_word_boxes([w["box"] for w in group])
        if not box:
            continue

        clamped = BoundingBox(
            x=max(0, min(image_width - 1, box.x)),
            y=max(0, min(image_height - 1, box.y)),
            width=min(image_width - box.x, box.width),
            height=min(image_height - box.y, box.height),
        )

        text = " ".join(w["text"] for w in group)
        conf = sum(w.get("confidence", 1.0) for w in group) / len(group)

        sentences.append(
            {
                "text": text,
                "box": clamped,
                "confidence": round(conf, 3),
                "words": group,
            }
        )

    return sentences


def get_redaction_boxes(sentence: dict, pii_targets: list[str]) -> list[BoundingBox]:
    """Extract word-level bounding boxes for specific PII substrings within a sentence.

    Enables fine-grained word-level redaction so labels and surrounding text remain visible.
    """
    if not sentence or not pii_targets:
        return []

    normalized_targets = [t.strip().lower() for t in pii_targets if t and t.strip()]
    if not normalized_targets:
        return []

    words = sentence.get("words", [])
    if not words:
        return [sentence["box"]] if "box" in sentence else []

    matched_boxes = []
    for w in words:
        w_text = w.get("text", "").strip().lower()
        if not w_text:
            continue
        is_pii = any(target in w_text or w_text in target for target in normalized_targets)
        if is_pii:
            b = w["box"]
            matched_boxes.append(
                BoundingBox(x=b["x"], y=b["y"], width=b["width"], height=b["height"])
                if isinstance(b, dict)
                else b
            )

    if not matched_boxes and "box" in sentence:
        sentence_text = sentence.get("text", "").lower()
        if any(target in sentence_text for target in normalized_targets):
            b = sentence["box"]
            matched_boxes.append(
                BoundingBox(x=b["x"], y=b["y"], width=b["width"], height=b["height"])
                if isinstance(b, dict)
                else b
            )

    return matched_boxes


def test_union_word_boxes_somchai_sample():
    """Verify that multiple word boxes unite into the expected sentence box."""
    words = [
        {"x": 20, "y": 36, "width": 46, "height": 24},
        {"x": 67, "y": 36, "width": 30, "height": 24},
        {"x": 97, "y": 36, "width": 47, "height": 24},
        {"x": 146, "y": 36, "width": 42, "height": 24},
    ]
    box = union_word_boxes(words)
    assert box is not None
    assert box.x == 20
    assert box.y == 36
    assert box.width == (146 + 42) - 20  # 168
    assert box.height == 24


def test_group_words_into_multiple_sentences():
    """Verify multiple lines are correctly grouped into distinct sentences."""
    words = [
        {"text": "นาย", "box": {"x": 20, "y": 36, "width": 46, "height": 24}, "confidence": 0.95},
        {"text": "สมชาย", "box": {"x": 70, "y": 36, "width": 60, "height": 24}, "confidence": 0.96},
        {"text": "อีเมล", "box": {"x": 20, "y": 84, "width": 50, "height": 28}, "confidence": 0.98},
        {"text": "somchai@example.com", "box": {"x": 80, "y": 84, "width": 250, "height": 28}, "confidence": 0.99},
    ]

    sentences = group_words_into_sentences(words, image_width=800, image_height=600)
    assert len(sentences) == 2
    assert sentences[0]["text"] == "นาย สมชาย"
    assert sentences[1]["text"] == "อีเมล somchai@example.com"


def test_separated_words_not_merged_when_gap_large():
    """Verify words with a large horizontal gap on the same line are NOT merged."""
    # User's case: Sandbox (x=40..155) vs API keys (x=501..677)
    words = [
        {"text": "Sandbox", "box": {"x": 40, "y": 24, "width": 115, "height": 23}},
        {"text": "0-", "box": {"x": 230, "y": 22, "width": 64, "height": 27}},
        {"text": "API keys", "box": {"x": 501, "y": 24, "width": 176, "height": 33}},
    ]

    sentences = group_words_into_sentences(words, image_width=1000, image_height=500, max_gap_ratio=1.0)
    # Each should be its own separate sentence/phrase because gaps are 75px and 207px (much larger than height ~24px)
    assert len(sentences) == 3
    assert sentences[0]["text"] == "Sandbox"
    assert sentences[1]["text"] == "0-"
    assert sentences[2]["text"] == "API keys"


def test_table_headers_separated_into_columns():
    """Verify table header columns with wide gaps are preserved as separate column boxes."""
    # User's case: T (x=641), Mode (x=721), API Key (x=916), API Secret (x=1559)
    words = [
        {"text": "T", "box": {"x": 641, "y": 198, "width": 20, "height": 25}},
        {"text": "Mode", "box": {"x": 721, "y": 200, "width": 64, "height": 23}},
        {"text": "API Key", "box": {"x": 916, "y": 200, "width": 92, "height": 25}},
        {"text": "API Secret", "box": {"x": 1559, "y": 202, "width": 127, "height": 21}},
    ]

    sentences = group_words_into_sentences(words, image_width=2000, image_height=1000, max_gap_ratio=1.0)
    assert len(sentences) == 4
    assert [s["text"] for s in sentences] == ["T", "Mode", "API Key", "API Secret"]


def test_phrase_merged_but_distant_column_separated():
    """Verify close words form a phrase ('G) Wink Widget') while distant items ('Tenant') are separated."""
    # User's case: G) (x=61..97), Wink Widget (x=110..265), Tenant (x=529..612)
    words = [
        {"text": "G)", "box": {"x": 61, "y": 181, "width": 36, "height": 23}},
        {"text": "Wink Widget", "box": {"x": 110, "y": 181, "width": 155, "height": 23}},
        {"text": "Tenant", "box": {"x": 529, "y": 181, "width": 83, "height": 23}},
    ]

    sentences = group_words_into_sentences(words, image_width=1000, image_height=500, max_gap_ratio=1.0)
    assert len(sentences) == 2
    assert sentences[0]["text"] == "G) Wink Widget"
    assert sentences[0]["box"].x == 61
    assert sentences[0]["box"].width == (265 - 61)  # 204
    assert sentences[1]["text"] == "Tenant"
    assert sentences[1]["box"].x == 529


def test_multi_line_table_columns_separated_by_alignment():
    """Verify multiple table rows are split by column alignment (x=524, 715/711, 867)."""
    # User's table row 1 & row 2 data
    words = [
        # Row 1
        {"text": "yourTenantID", "box": {"x": 524, "y": 323, "width": 159, "height": 23}},
        {"text": "SANDBOX", "box": {"x": 715, "y": 321, "width": 127, "height": 23}},
        {"text": "sak-123abc123abc", "box": {"x": 867, "y": 321, "width": 595, "height": 21}},
        # Row 2
        {"text": "yourTenantID", "box": {"x": 524, "y": 439, "width": 159, "height": 23}},
        {"text": "PRODUCTION", "box": {"x": 711, "y": 443, "width": 170, "height": 15}},
        {"text": "pak-123abc123abc", "box": {"x": 867, "y": 439, "width": 597, "height": 21}},
    ]

    sentences = group_words_into_sentences(words, image_width=2000, image_height=800, max_gap_ratio=1.2)
    # Both rows must be cleanly split into 3 distinct column boxes (total 6 column segments)
    assert len(sentences) == 6
    texts = [s["text"] for s in sentences]
    assert "yourTenantID" in texts
    assert "SANDBOX" in texts
    assert "PRODUCTION" in texts
    assert "sak-123abc123abc" in texts
    assert "pak-123abc123abc" in texts


def test_get_redaction_boxes_fine_grained():
    """Verify word-level redaction only returns boxes for PII words, leaving labels untouched."""
    sentence = {
        "text": "อีเมล somchai@example.com",
        "box": {"x": 20, "y": 84, "width": 332, "height": 32},
        "words": [
            {"text": "อีเมล", "box": {"x": 20, "y": 84, "width": 60, "height": 32}},
            {"text": "somchai@example.com", "box": {"x": 90, "y": 84, "width": 262, "height": 32}},
        ],
    }

    # Only redact the email address
    boxes = get_redaction_boxes(sentence, ["somchai@example.com"])
    assert len(boxes) == 1
    assert boxes[0].x == 90
    assert boxes[0].width == 262
    # The label 'อีเมล' (x=20) is NOT in the redaction boxes!


def count_horizontal_glyphs(text: str) -> int:
    count = 0
    for ch in text:
        code = ord(ch)
        is_combining = code == 0x0E31 or (0x0E34 <= code <= 0x0E3A) or (0x0E47 <= code <= 0x0E4E)
        if not is_combining:
            count += 1
    return max(1, count)


def segment_into_words(text: str, box: BoundingBox, confidence: float = 1.0) -> list[dict]:
    import pythainlp.tokenize

    raw_tokens = [t.strip() for t in pythainlp.tokenize.word_tokenize(text) if t.strip()]
    if len(raw_tokens) <= 1:
        return [{"text": text, "box": box, "confidence": confidence}]

    total_glyphs = count_horizontal_glyphs(text)
    px_per_glyph = box.width / total_glyphs

    words = []
    current_x = box.x
    for t in raw_tokens:
        t_glyphs = count_horizontal_glyphs(t)
        t_w = max(8, round(t_glyphs * px_per_glyph))
        words.append(
            {
                "text": t,
                "box": BoundingBox(
                    x=current_x,
                    y=box.y,
                    width=min(box.x + box.width - current_x, t_w),
                    height=box.height,
                ),
                "confidence": confidence,
            }
        )
        current_x += t_w
    return words


def test_thai_word_decomposition_with_boxes():
    """Verify continuous Thai announcement line decomposes into individual word boxes."""
    line_text = "ประกาศมหาวิทยาลัยราชภัฎสวนสุนันทา"
    line_box = BoundingBox(x=848, y=650, width=840, height=44)

    words = segment_into_words(line_text, line_box, confidence=0.717)
    # Should decompose into multiple words (not just 1 big line)
    assert len(words) >= 4
    word_texts = [w["text"] for w in words]
    assert "ประกาศ" in word_texts
    assert "มหาวิทยาลัย" in word_texts

    # Every word has its own distinct, non-zero width bounding box within the line bounds
    for w in words:
        assert w["box"].x >= 848
        assert w["box"].width > 0
        assert w["box"].x + w["box"].width <= 848 + 840 + 5



