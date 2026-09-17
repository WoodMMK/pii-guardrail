"""Unit tests: the pipeline surfaces an upscale warning and runs detection.

When a low-resolution input is auto-upscaled to reach the 150 DPI threshold
(Option A), ``GuardrailPipeline.process`` should:
    - report ``quality_sufficient == True`` (it no longer short-circuits),
    - actually run OCR/detection on the enlarged image,
    - prepend a warning noting the image was upscaled and the returned image is
      larger than the original.

A ``FakeOCRBackend`` is injected so detection is deterministic without PaddleOCR.
"""

from __future__ import annotations

import numpy as np
import pytest

from pii_guardrail.detector import Detector
from pii_guardrail.models import BoundingBox, TextSegment
from pii_guardrail.ocr import OCREngine
from pii_guardrail.pipeline import GuardrailPipeline
from pii_guardrail.preprocessor import Preprocessor
from pii_guardrail.redactor import Redactor

from tests.property.ocr_fakes import FakeOCRBackend

pytestmark = pytest.mark.unit


def _low_res_image() -> np.ndarray:
    """A 1000px-tall image (~91 DPI): below threshold but upscalable to >=150.

    Kept narrow (grayscale, single channel) so the real denoise on the enlarged
    image stays fast: fastNlMeans on a tall-but-narrow single-channel array is
    far cheaper than on a wide colour one, while still exercising the upscale +
    detect + redact path.
    """
    return np.full((1000, 120), 255, dtype=np.uint8)


def test_upscaled_input_runs_detection_and_warns() -> None:
    """An upscaled low-res image is detected on, with an upscale warning first."""
    # The fake backend ignores pixels and returns a sensitive segment. Its box
    # must fit inside the UPSCALED image; the original is 1000x700 and upscale
    # only enlarges, so a small box at the origin is always in-bounds.
    seg = TextSegment(
        text="user@example.com",
        box=BoundingBox(x=0, y=0, width=200, height=40),
        confidence=0.95,
    )
    pipeline = GuardrailPipeline(
        Preprocessor(),
        OCREngine(FakeOCRBackend([seg])),
        Detector(),
        Redactor(),
    )

    result = pipeline.process(_low_res_image(), redact=True)

    # It was treated as sufficient (post-upscale) and detection actually ran.
    assert result.quality_sufficient is True
    assert result.detection_result.count >= 1

    # The detection_result / redacted image are sized to the UPSCALED image,
    # which is larger than the 1000px-tall input (Option A: bigger image out).
    assert result.detection_result.image_height > 1000
    assert result.redacted_image is not None
    assert result.redacted_image.shape[0] > 1000

    # The first warning explains the upscale and that the output is larger.
    assert len(result.warnings) >= 1
    first = result.warnings[0].lower()
    assert "upscal" in first
    assert "larger" in first


def test_high_res_input_has_no_upscale_warning() -> None:
    """A sufficiently large input is not upscaled and carries no upscale warning."""
    seg = TextSegment(
        text="user@example.com",
        box=BoundingBox(x=0, y=0, width=200, height=40),
        confidence=0.95,
    )
    # ~150+ DPI: height above the 1650px threshold. Narrow grayscale keeps the
    # real denoise fast while still above the quality gate.
    big = np.full((1700, 120), 255, dtype=np.uint8)
    pipeline = GuardrailPipeline(
        Preprocessor(),
        OCREngine(FakeOCRBackend([seg])),
        Detector(),
        Redactor(),
    )

    result = pipeline.process(big, redact=True)

    assert result.quality_sufficient is True
    # Output keeps the input height (no enlargement).
    assert result.detection_result.image_height == 1700
    assert not any("upscal" in w.lower() for w in result.warnings)
