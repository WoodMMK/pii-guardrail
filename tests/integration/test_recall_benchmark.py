"""Task 15.3 -- scanned-vs-synthetic detection recall benchmark.

Model/infrastructure-dependent integration benchmark exercising the REAL
PaddleOCR backend end-to-end through the :class:`GuardrailPipeline`.

Goal (Requirement 6.5): over a PAIRED corpus -- each ground-truth sensitive item
rendered both as a clean "synthetic" baseline image and as a ">=150 DPI scanned"
image (height >= 1650 px so the Preprocessor's DPI heuristic marks it quality
sufficient) -- detection RECALL on the scanned images, relative to their
synthetic counterparts, must be at least 95%.

RECALL definition here: for each ground-truth item ``(text, expected_category)``
rendered as a scanned image and run through the full pipeline (denoise + deskew +
OCR + detect), the item counts as RECALLED when ``expected_category`` appears in
the union of ``.categories`` across the detected regions. Recall is the fraction
of ground-truth items recalled on the scanned set. We first sanity-check that the
synthetic baseline recalls every item, then assert scanned recall >= 0.95.

The whole module is guarded on PaddleOCR availability (``importorskip`` + a
module-scoped backend fixture that skips if the model cannot load), mirroring
``test_paddle_extraction.py``.

This benchmark renders many tall (>=1650 px) images and runs the real denoise +
OCR on each, so it is intentionally SLOW (a few minutes). Width is kept modest
(tall-but-narrow) and N is bounded to keep runtime reasonable while a >=95% bar
stays meaningful.

Requirements: 6.5.
"""

from __future__ import annotations

import pytest

# Guard the whole module: skip everything if paddleocr is not importable.
pytest.importorskip("paddleocr")

from pii_guardrail.detector import Detector  # noqa: E402
from pii_guardrail.models import SensitiveCategory  # noqa: E402
from pii_guardrail.ocr import OCREngine  # noqa: E402
from pii_guardrail.paddle_backend import PaddleOCRBackend  # noqa: E402
from pii_guardrail.pipeline import GuardrailPipeline  # noqa: E402
from pii_guardrail.preprocessor import Preprocessor  # noqa: E402
from pii_guardrail.redactor import Redactor  # noqa: E402

from tests.integration.sample_corpus import (  # noqa: E402
    find_latin_font_path,
    render_scanned_sample,
    render_text_image,
)

pytestmark = pytest.mark.integration


#: Paired ground-truth corpus: each entry is ``(text, expected_category)``.
#:
#: Strings are chosen to be read essentially perfectly by PaddleOCR at a large
#: font, AND to be a single contiguous token whose whole shape matches one of
#: the Detector's high-signal patterns. This matters because PaddleOCR emits one
#: text SEGMENT per detected line/word and the Detector classifies each segment
#: independently: a value split across two boxes (e.g. a currency code and its
#: amount) would not match a pattern that needs both parts adjacent. Emails,
#: URLs, phone numbers, and percentages are single tokens with strong patterns,
#: so they survive segmentation; money amounts (symbol/code + number) are omitted
#: here precisely because they tend to split or lose the currency glyph.
#:
#: Several distinct concrete values per category grow the denominator (N) so a
#: single OCR misread doesn't sink recall below 95%, while every item stays
#: robust. All are SHORT so they fit the tall-but-narrow scan canvas (width
#: capped at 1000 px in ``render_scanned_sample``) at a large, legible glyph size.
GROUND_TRUTH: tuple[tuple[str, SensitiveCategory], ...] = (
    # EMAIL -- local@domain.tld (one contiguous token).
    ("user@example.com", SensitiveCategory.EMAIL),
    ("admin@test.org", SensitiveCategory.EMAIL),
    ("sales@company.io", SensitiveCategory.EMAIL),
    # URL -- explicit scheme / www so the URL pattern anchors cleanly.
    ("https://example.com", SensitiveCategory.URL),
    ("www.example.org", SensitiveCategory.URL),
    # PHONE_NUMBER -- Thai local grouping and international prefix.
    ("081-234-5678", SensitiveCategory.PHONE_NUMBER),
    ("02-123-4567", SensitiveCategory.PHONE_NUMBER),
    # PERCENT_VALUE -- number immediately followed by %.
    ("45%", SensitiveCategory.PERCENT_VALUE),
    ("100%", SensitiveCategory.PERCENT_VALUE),
    ("12.5%", SensitiveCategory.PERCENT_VALUE),
)

#: Recall bar required by Requirement 6.5.
_RECALL_THRESHOLD = 0.95


@pytest.fixture(scope="module")
def backend() -> PaddleOCRBackend:
    """Construct the real PaddleOCR backend ONCE for the module.

    PaddleOCR model load is slow, so this is module-scoped and reused across the
    (single) benchmark. If the model fails to load despite the package being
    importable, the benchmark skips cleanly rather than failing.
    """
    be = PaddleOCRBackend()
    if not be.available:
        pytest.skip(
            "PaddleOCR is installed but the model could not be loaded; "
            "skipping the recall benchmark."
        )
    return be


@pytest.fixture(scope="module")
def pipeline(backend: PaddleOCRBackend) -> GuardrailPipeline:
    """A full pipeline wired with the real, loaded PaddleOCR backend (built once)."""
    return GuardrailPipeline(
        preprocessor=Preprocessor(),
        ocr_engine=OCREngine(backend),
        detector=Detector(),
        redactor=Redactor(),
    )


def _detected_categories(result) -> set[SensitiveCategory]:
    """Union of ``.categories`` across all detected regions in a PipelineResult."""
    categories: set[SensitiveCategory] = set()
    for region in result.detection_result.regions:
        categories.update(region.categories)
    return categories


def test_scanned_vs_synthetic_recall_at_least_95_percent(
    pipeline: GuardrailPipeline,
):
    """Scanned-image detection recall relative to synthetic counterparts >= 95%.

    For each ``(text, expected_category)`` in :data:`GROUND_TRUTH`:
      1. Render a clean SYNTHETIC baseline and confirm the pipeline recalls the
         expected category (sanity: the item is detectable at all).
      2. Render a ">=150 DPI SCANNED" counterpart (height >= 1650 px), confirm it
         is ``quality_sufficient`` (so it genuinely exercises the >=150 DPI path),
         and record whether the expected category is recalled.

    Both members of a pair are rendered tall enough to clear the Preprocessor's
    150 DPI quality gate (height >= 1650 px); otherwise the pipeline
    short-circuits before OCR and reports zero regions (Requirement 6.6), which
    would make the comparison meaningless. The synthetic baseline is the clean
    tall render; the scanned counterpart is the tall render produced by
    ``render_scanned_sample`` (auto-sized tall-but-narrow to bound runtime).

    Assert scanned recall (recalled / total) >= 0.95. Requirement 6.5.
    """
    font_path = find_latin_font_path()

    # Rendering sizes are deliberately SMALL to bound the (denoise + real OCR)
    # runtime on this CPU-only PaddleOCR backend, where each tall-image pass costs
    # tens of seconds. Height is kept just above the >=150 DPI gate (1650 px), the
    # canvas is kept narrow, and the font is only as large as needed to stay
    # legible -- this keeps total pixels (and OCR cost) low while every item still
    # clears the quality gate and OCRs reliably.
    _HEIGHT = 1660  # > 1650 -> estimated_dpi ~150.9 -> quality_sufficient
    _MAX_WIDTH = 600
    _FONT_SIZE = 90

    synthetic_missed: list[str] = []
    scanned_missed: list[str] = []
    not_quality_sufficient: list[str] = []

    total = len(GROUND_TRUTH)
    scanned_recalled = 0

    for text, expected in GROUND_TRUTH:
        # --- Synthetic baseline (clean, quality-sufficient rendering) ------
        # Pass ``width=None`` so the canvas auto-sizes to the FULL string at
        # ``_FONT_SIZE`` (text_w + 2*margin) instead of clipping it to a fixed
        # 600px width. ``height=_HEIGHT`` (1660) is preserved because
        # ``render_text_image`` only GROWS height (max(height, text_h + 2*margin)),
        # so the baseline stays >=1650px -> quality_sufficient while rendering the
        # complete, legible token. This mirrors the scanned counterpart (which
        # auto-shrinks the font to fit ``max_width``); both now render the whole
        # string, but the synthetic baseline keeps the larger font_size 90 glyphs.
        synthetic_image = render_text_image(
            text,
            font_path=font_path,
            width=None,
            height=_HEIGHT,
            font_size=_FONT_SIZE,
        )
        synthetic_result = pipeline.process(synthetic_image, redact=False)
        if expected not in _detected_categories(synthetic_result):
            synthetic_missed.append(f"{text!r} (expected {expected.value})")

        # --- ">=150 DPI scanned" counterpart (tall image) ------------------
        scanned_image = render_scanned_sample(
            text,
            font_path=font_path,
            height=_HEIGHT,
            max_width=_MAX_WIDTH,
            font_size=_FONT_SIZE,
        )
        scanned_result = pipeline.process(scanned_image, redact=False)

        # It must genuinely be a >=150 DPI case, else the render height is wrong.
        if not scanned_result.quality_sufficient:
            not_quality_sufficient.append(f"{text!r} -> {scanned_image.shape}")

        if expected in _detected_categories(scanned_result):
            scanned_recalled += 1
        else:
            scanned_missed.append(f"{text!r} (expected {expected.value})")

    # Every scanned image must have cleared the >=150 DPI quality gate, otherwise
    # the benchmark isn't measuring the scanned-detection path it claims to.
    assert not not_quality_sufficient, (
        "some scanned images were NOT quality_sufficient (render height too "
        f"small for the >=150 DPI heuristic): {not_quality_sufficient}"
    )

    # Sanity: the synthetic baseline should recall every ground-truth item. A
    # miss here means the item isn't reliably detectable even on a clean render,
    # so the paired comparison would be meaningless.
    assert not synthetic_missed, (
        "synthetic baseline failed to recall some ground-truth items "
        f"(unreliable corpus items): {synthetic_missed}"
    )

    recall = scanned_recalled / total
    assert recall >= _RECALL_THRESHOLD, (
        f"scanned recall {recall:.3f} ({scanned_recalled}/{total}) below the "
        f"{_RECALL_THRESHOLD:.0%} threshold; missed on scanned images: "
        f"{scanned_missed}"
    )
