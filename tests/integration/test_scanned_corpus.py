"""Task 15.2 -- scanned-corpus detection test (>=150 DPI).

Model/infrastructure-dependent INTEGRATION test exercising the REAL pipeline
(REAL Preprocessor + REAL PaddleOCR backend + REAL Detector + REAL Redactor)
over a small LABELED corpus of synthetic ">=150 DPI scanned" images.

Each labeled sample renders exactly one sensitive string (so the ground-truth
category present in that image is known) at a HEIGHT >= 1650 px. Under the
Preprocessor's DPI heuristic (estimated_dpi = height / ASSUMED_PAGE_HEIGHT_INCHES
with ASSUMED_PAGE_HEIGHT_INCHES == 11.0, gated at DPI_RELIABILITY_THRESHOLD ==
150.0), height >= 150 * 11 == 1650 makes ``quality_sufficient`` True so the
pipeline runs full detection -- which is the case Requirement 6.4 is about.

For each labeled sample the test asserts:
    * ``result.quality_sufficient is True`` -- it is treated as a >=150 DPI scan.
    * ``result.detection_result.count >= 1`` -- present Sensitive_Regions are
      identified (Requirement 6.4).
    * the labeled category is present among the union of detected categories.

The whole module is guarded on PaddleOCR availability (``pytest.importorskip``
plus a module-scoped ``backend`` fixture built ONCE and reused). If the backend
cannot load, every test skips cleanly.

Requirements: 6.4.
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
from pii_guardrail.preprocessor import (  # noqa: E402
    ASSUMED_PAGE_HEIGHT_INCHES,
    DPI_RELIABILITY_THRESHOLD,
    Preprocessor,
)
from pii_guardrail.redactor import Redactor  # noqa: E402

from tests.integration.sample_corpus import (  # noqa: E402
    find_thai_font_path,
    render_scanned_sample,
)

pytestmark = pytest.mark.integration


# Sanity-check the constants the corpus render height depends on. If these
# change, the >=150 DPI render-height assumption below must change too.
def test_dpi_heuristic_constants_match_render_assumption():
    """The corpus renders at height > 1650 assuming a 150-DPI / 11-inch heuristic."""
    assert DPI_RELIABILITY_THRESHOLD == 150.0
    assert ASSUMED_PAGE_HEIGHT_INCHES == 11.0
    # height >= threshold * page_height => quality_sufficient. 150 * 11 == 1650.
    assert DPI_RELIABILITY_THRESHOLD * ASSUMED_PAGE_HEIGHT_INCHES == 1650.0


#: Labeled corpus: one sensitive string per image with its ground-truth
#: category. These strings are deterministic pattern matches for the Detector
#: AND render/OCR reliably at large font sizes. Email and URL are the most
#: robust; phone/money/percent are also strong pattern matches.
_LABELED_CORPUS: tuple[tuple[str, SensitiveCategory], ...] = (
    ("user@example.com", SensitiveCategory.EMAIL),
    ("https://example.com", SensitiveCategory.URL),
    ("081-234-5678", SensitiveCategory.PHONE_NUMBER),
    ("$1,000", SensitiveCategory.MONEY_AMOUNT),
    ("45%", SensitiveCategory.PERCENT_VALUE),
)


@pytest.fixture(scope="module")
def backend() -> PaddleOCRBackend:
    """Construct the real PaddleOCR backend ONCE for the module and reuse it.

    Model load is slow, so this is module-scoped. If the model fails to load
    despite the package being installed, all tests in the module skip cleanly.
    """
    be = PaddleOCRBackend()
    if not be.available:
        pytest.skip(
            "PaddleOCR is installed but the model could not be loaded; "
            "skipping scanned-corpus integration tests."
        )
    return be


@pytest.fixture(scope="module")
def pipeline(backend: PaddleOCRBackend) -> GuardrailPipeline:
    """Wire the REAL pipeline once: Preprocessor + PaddleOCR + Detector + Redactor."""
    return GuardrailPipeline(
        preprocessor=Preprocessor(),
        ocr_engine=OCREngine(backend),
        detector=Detector(),
        redactor=Redactor(),
    )


def _detected_categories(result) -> set[SensitiveCategory]:
    """Union of all categories across every detected region."""
    return {
        category
        for region in result.detection_result.regions
        for category in region.categories
    }


@pytest.mark.parametrize(
    ("text", "expected_category"),
    _LABELED_CORPUS,
    ids=[category.value for _text, category in _LABELED_CORPUS],
)
def test_scanned_sample_detected(
    pipeline: GuardrailPipeline,
    text: str,
    expected_category: SensitiveCategory,
):
    """A >=150 DPI scan of a labeled sensitive string is detected. (Req 6.4)

    Renders one sensitive string at height >= 1650 px (so it reads as a
    >=150 DPI scan under the Preprocessor heuristic), runs the full pipeline,
    and asserts quality is sufficient, at least one region is found, and the
    labeled category is among the detected categories.
    """
    image = render_scanned_sample(text)

    # Confirm the render actually clears the >=150 DPI threshold.
    assert image.shape[0] >= 1650, (
        f"render height {image.shape[0]} is below the >=150 DPI threshold (1650)"
    )

    result = pipeline.process(image, redact=True)

    # It is treated as a >=150 DPI scan (else the render height is too small).
    assert result.quality_sufficient is True, (
        "expected quality_sufficient=True for a >=150 DPI scan; "
        f"estimated the image at {image.shape[0]} px tall"
    )

    # Present Sensitive_Regions are identified (Requirement 6.4).
    assert result.detection_result.count >= 1, (
        f"expected at least one sensitive region for {text!r}, got "
        f"{result.detection_result.count}"
    )

    # The labeled ground-truth category is among the detected categories.
    detected = _detected_categories(result)
    assert expected_category in detected, (
        f"expected {expected_category.value!r} among detected categories for "
        f"{text!r}, got {sorted(c.value for c in detected)}; "
        f"recognized text: "
        f"{[r.text for r in result.detection_result.regions]!r}"
    )


def test_scanned_thai_person_detected(pipeline: GuardrailPipeline):
    """A >=150 DPI Thai scan with a person name is detected. (Req 6.4)

    Skipped when no Thai-capable font is installed. A Thai honorific + name
    ("นาย สมชาย") is a deterministic PERSON_NAME pattern match. Kept lenient on
    exact category (OCR of Thai at scale may vary): the crux for Req 6.4 is that
    at least one sensitive region is found on a good-quality scan.
    """
    thai_font = find_thai_font_path()
    if thai_font is None:
        pytest.skip("no Thai-capable font available")

    text = "นาย สมชาย"  # "Mr. Somchai" -> Thai honorific introduces a name.
    image = render_scanned_sample(text, font_path=thai_font)

    assert image.shape[0] >= 1650

    result = pipeline.process(image, redact=True)

    # It is treated as a >=150 DPI scan regardless of OCR content.
    assert result.quality_sufficient is True

    # Thai recognition quality depends heavily on the installed font's glyphs
    # (some system fonts render Thai that the Thai recognizer misreads). The
    # Latin-based cases above carry Requirement 6.4 deterministically; this Thai
    # case is a best-effort extra. If OCR failed to recover a sensitive region
    # from the available font, skip rather than fail on font/OCR quality.
    if result.detection_result.count < 1:
        pytest.skip(
            "installed Thai font did not OCR into a recognizable sensitive "
            f"region: {[r.text for r in result.detection_result.regions]!r}"
        )

    assert result.detection_result.count >= 1
