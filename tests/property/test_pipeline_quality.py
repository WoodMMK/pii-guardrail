"""Property-based test for the pipeline's insufficient-quality behavior.

# Feature: thai-image-pii-guardrail, Property 14: Insufficient quality yields no reliable regions and a quality indication

Validates: Requirements 6.6
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pii_guardrail.detector import Detector
from pii_guardrail.ocr import OCREngine
from pii_guardrail.pipeline import GuardrailPipeline
from pii_guardrail.preprocessor import (
    ASSUMED_PAGE_HEIGHT_INCHES,
    DPI_RELIABILITY_THRESHOLD,
    MAX_UPSCALE_FACTOR,
    Preprocessor,
)
from pii_guardrail.redactor import Redactor

from tests.property.ocr_fakes import FakeOCRBackend

# The real Preprocessor derives effective DPI from image HEIGHT:
#   estimated_dpi = height / ASSUMED_PAGE_HEIGHT_INCHES
# and flags quality_sufficient = estimated_dpi >= DPI_RELIABILITY_THRESHOLD.
# So any image with height strictly below this pixel count trips the gate.
_INSUFFICIENT_HEIGHT_PX = int(DPI_RELIABILITY_THRESHOLD * ASSUMED_PAGE_HEIGHT_INCHES)

# Auto-upscale (Option A) enlarges a low-resolution input by up to
# MAX_UPSCALE_FACTOR to try to reach the threshold. An image STILL below the
# threshold AFTER the maximum upscale genuinely cannot be made reliable, so the
# pipeline still short-circuits. That happens when the enlarged height stays
# below the threshold, i.e. original height < _INSUFFICIENT_HEIGHT_PX / MAX_UPSCALE_FACTOR.
_UNRECOVERABLE_HEIGHT_PX = int(_INSUFFICIENT_HEIGHT_PX / MAX_UPSCALE_FACTOR)


@pytest.mark.property
@settings(max_examples=100, deadline=None)
@given(
    # Heights so low that even the maximum upscale (MAX_UPSCALE_FACTOR) cannot
    # lift them to the 150 DPI threshold, so the quality gate still trips
    # deterministically. Kept small so the real denoise step stays fast.
    height=st.integers(min_value=8, max_value=64),
    width=st.integers(min_value=8, max_value=64),
)
def test_insufficient_quality_yields_no_regions_and_indication(
    height: int, width: int
) -> None:
    """Unrecoverably-low-resolution images yield no regions plus a quality indication.

    Auto-upscale (Option A) tries to enlarge a low-resolution input to reach the
    150 DPI reliability threshold. When even the maximum upscale cannot reach it,
    ``GuardrailPipeline.process`` must still short-circuit: report
    ``quality_sufficient == False``, produce zero sensitive regions, and surface
    a non-empty insufficient-quality warning.

    # Feature: thai-image-pii-guardrail, Property 14: Insufficient quality yields no reliable regions and a quality indication
    Validates: Requirements 6.6
    """
    # Sanity: even after the maximum upscale these heights stay below the
    # threshold (height * MAX_UPSCALE_FACTOR < _INSUFFICIENT_HEIGHT_PX), so the
    # gate is guaranteed to flag them.
    assert height <= _UNRECOVERABLE_HEIGHT_PX

    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Real Preprocessor (so the quality gate is exercised for real); a fake OCR
    # backend is injected for determinism even though the pipeline short-circuits
    # before OCR runs on an insufficient-quality image.
    pipeline = GuardrailPipeline(
        Preprocessor(),
        OCREngine(FakeOCRBackend([])),
        Detector(),
        Redactor(),
    )

    result = pipeline.process(image)

    # No reliable detection on an insufficient-quality image.
    assert result.quality_sufficient is False
    assert result.detection_result.count == 0
    assert result.detection_result.regions == []

    # A non-empty indication that quality was insufficient.
    assert len(result.warnings) >= 1
    assert any(
        "quality" in w.lower() or "insufficient" in w.lower()
        for w in result.warnings
    )
