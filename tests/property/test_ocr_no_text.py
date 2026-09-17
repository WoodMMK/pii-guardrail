"""Property-based test: no recognized text yields zero sensitive regions.

# Feature: thai-image-pii-guardrail, Property 4: No recognized text yields zero sensitive regions

Validates: Requirements 2.6
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.typing import NDArray

from pii_guardrail.detector import Detector
from pii_guardrail.ocr import OCREngine
from pii_guardrail.pipeline import GuardrailPipeline, PipelineResult
from pii_guardrail.preprocessor import PreprocessResult, Preprocessor
from pii_guardrail.redactor import Redactor
from tests.property.ocr_fakes import FakeOCRBackend


class _PassthroughPreprocessor(Preprocessor):
    """Preprocessor double that reports sufficient quality without altering the image.

    The real Preprocessor gates quality on an estimated DPI derived from image
    height, so a modest generated image would short-circuit the pipeline before
    OCR runs. To keep this property focused on the "no recognized text -> zero
    regions" flow (Requirement 2.6) and still drive the image through OCR +
    detection, this double returns the input image unchanged with
    ``quality_sufficient=True``. It changes only the quality gate, not the
    detection logic under test.
    """

    def preprocess(
        self, image: NDArray, is_scanned: bool | None = None
    ) -> PreprocessResult:
        self._validate(image)
        return PreprocessResult(
            image=image,
            residual_skew_deg=0.0,
            estimated_dpi=300.0,
            quality_sufficient=True,
        )


# Small, valid RGB uint8 images: ndim 3, 3 channels, modest H/W.
_images = arrays(
    dtype=np.uint8,
    shape=st.tuples(
        st.integers(min_value=8, max_value=64),
        st.integers(min_value=8, max_value=64),
        st.just(3),
    ),
)


@pytest.mark.property
@settings(max_examples=100)
@given(image=_images)
def test_no_text_yields_zero_regions(image: NDArray) -> None:
    """No recognized text yields a DetectionResult with zero regions.

    For any image where the OCR_Engine returns no text segments (modeled by a
    ``FakeOCRBackend`` returning ``[]``), driving the full pipeline yields a
    ``DetectionResult`` with ``count == 0`` and no sensitive regions.

    # Feature: thai-image-pii-guardrail, Property 4: No recognized text yields zero sensitive regions
    Validates: Requirements 2.6
    """
    backend = FakeOCRBackend([])  # No text found: a normal, empty result.
    pipeline = GuardrailPipeline(
        preprocessor=_PassthroughPreprocessor(),
        ocr_engine=OCREngine(backend=backend),
        detector=Detector(),
        redactor=Redactor(),
    )

    result = pipeline.process(image, redact=True)

    assert isinstance(result, PipelineResult)
    # The empty backend means OCR was consulted and returned no segments, so
    # zero sensitive regions are produced.
    assert backend.calls, "OCR backend should have been invoked"
    assert result.detection_result.count == 0
    assert result.detection_result.regions == []
