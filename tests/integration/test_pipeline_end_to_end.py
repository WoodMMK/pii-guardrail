"""End-to-end integration test for the GuardrailPipeline core flow.

Drives :meth:`pii_guardrail.pipeline.GuardrailPipeline.process` through the full
preprocess -> OCR -> detect -> redact flow using test doubles (no PaddleOCR / no
real classifier model), asserting that a coherent ``DetectionResult`` and
redacted image are produced together.

This is an example-based integration test (not a Hypothesis property test).

Validates: Requirements 2.6, 7.1, 8.1.
    2.6 -- no recognized text yields a Detection_Result with zero regions.
    7.1 -- detection produces a Detection_Result listing each Sensitive_Region.
    8.1 -- when regions exist, redaction covers each with a black rectangle.
"""

from __future__ import annotations

import numpy as np
import pytest

from pii_guardrail.detector import Detector
from pii_guardrail.models import (
    BoundingBox,
    SensitiveCategory,
    TextSegment,
)
from pii_guardrail.ocr import OCREngine
from pii_guardrail.pipeline import GuardrailPipeline
from pii_guardrail.preprocessor import PreprocessResult, Preprocessor
from pii_guardrail.redactor import Redactor
from tests.property.ocr_fakes import FakeOCRBackend

pytestmark = pytest.mark.integration


class _PassthroughPreprocessor(Preprocessor):
    """A Preprocessor that validates then returns the image unchanged.

    The real Preprocessor runs a heavy non-local-means denoise and marks small
    images as insufficient quality (its heuristic effective DPI is
    ``height / 11``, so a 64px-tall image is well below the 150 DPI threshold).
    For a fast end-to-end flow we keep the real validation but skip the heavy
    work and force ``quality_sufficient=True`` so OCR + detect + redact run.
    """

    def preprocess(
        self, image: np.ndarray, is_scanned: bool | None = None
    ) -> PreprocessResult:
        # Reuse the real (cheap) validation so invalid arrays still fail closed.
        self._validate(image)
        return PreprocessResult(
            image=image,
            residual_skew_deg=0.0,
            estimated_dpi=300.0,
            quality_sufficient=True,
        )


class _FakeClassifierModel:
    """A minimal available ClassifierModel returning a fixed category set.

    Mirrors the spy used in the Detector unit tests: exposes ``available`` and a
    ``classify`` method whose categories the Detector unions with its
    pattern-based results.
    """

    def __init__(
        self,
        returns: set[SensitiveCategory] | None = None,
        available: bool = True,
    ) -> None:
        self._returns = set(returns) if returns is not None else set()
        self._available = available
        self.classify_calls: list[str] = []

    @property
    def available(self) -> bool:
        return self._available

    def classify(self, text: str) -> set[SensitiveCategory]:
        self.classify_calls.append(text)
        return set(self._returns)


def _blank_image() -> np.ndarray:
    """A small 64x64x3 uint8 image; boxes below fit within these bounds."""
    return np.full((64, 64, 3), 255, dtype=np.uint8)


def _build_pipeline(
    segments: list[TextSegment],
    *,
    classifier: _FakeClassifierModel | None = None,
    use_classifier: bool = False,
) -> GuardrailPipeline:
    preprocessor = _PassthroughPreprocessor()
    ocr_engine = OCREngine(FakeOCRBackend(segments=segments, available=True))
    detector = Detector(classifier=classifier, use_classifier=use_classifier)
    redactor = Redactor()
    return GuardrailPipeline(preprocessor, ocr_engine, detector, redactor)


class TestEndToEndCoreFlow:
    def test_sensitive_segments_produce_regions_and_redacted_image(self) -> None:
        # One clearly-sensitive segment (an email) with an in-bounds box, plus a
        # second, non-overlapping segment. The email segment is classified as
        # EMAIL by the patterns; the enabled+available model additionally
        # contributes PERSON_NAME to every segment it is consulted about.
        email_box = BoundingBox(x=4, y=8, width=40, height=12)
        other_box = BoundingBox(x=2, y=40, width=30, height=10)
        segments = [
            TextSegment(text="user@example.com", box=email_box, confidence=0.95),
            TextSegment(text="plain words here", box=other_box, confidence=0.9),
        ]
        # Enabled + available model contributes PERSON_NAME, which the Detector
        # unions with the pattern-based categories per segment (Req 5.1).
        model = _FakeClassifierModel(
            returns={SensitiveCategory.PERSON_NAME}, available=True
        )
        pipeline = _build_pipeline(
            segments, classifier=model, use_classifier=True
        )

        image = _blank_image()
        result = pipeline.process(image, redact=True)

        # Quality gate passed, so the full flow ran (Req 6.6 short-circuit avoided).
        assert result.quality_sufficient is True

        # The model was consulted for the segments' text.
        assert "user@example.com" in model.classify_calls

        # Both segments carry the model's PERSON_NAME category, so both become
        # regions; at least one region exists (Req 7.1).
        detection = result.detection_result
        assert detection.count >= 1

        # Locate the region derived from the email segment.
        email_regions = [r for r in detection.regions if r.text == "user@example.com"]
        assert len(email_regions) == 1
        region = email_regions[0]
        assert region.box == email_box
        # Pattern EMAIL unioned with the model's PERSON_NAME (Req 7.1, 5.1).
        assert SensitiveCategory.EMAIL in region.categories
        assert SensitiveCategory.PERSON_NAME in region.categories

        # detection_result dimensions match the (preprocessed == input) image.
        assert detection.image_width == image.shape[1]
        assert detection.image_height == image.shape[0]

        # Redaction ran (Req 8.1): same shape, and every pixel inside the email
        # region's box is black across all channels.
        redacted = result.redacted_image
        assert redacted is not None
        assert redacted.shape == image.shape
        b = region.box
        covered = redacted[b.y : b.y + b.height, b.x : b.x + b.width]
        assert np.all(covered == 0)

        # A pixel well outside every detected box is untouched (still white).
        assert np.all(redacted[60, 60] == 255)

    def test_no_recognized_text_yields_zero_regions(self) -> None:
        # Empty OCR result -> zero regions is a normal outcome (Req 2.6).
        pipeline = _build_pipeline(segments=[])

        image = _blank_image()
        result = pipeline.process(image, redact=True)

        assert result.quality_sufficient is True
        assert result.detection_result.count == 0
        assert result.detection_result.regions == []

        # Per pipeline.py, redact=True still redacts the preprocessed image even
        # with zero regions; the Redactor returns an equivalent copy, so the
        # redacted image is NOT None and is pixelwise-equivalent to the input.
        assert result.redacted_image is not None
        assert result.redacted_image.shape == image.shape
        assert np.array_equal(result.redacted_image, image)

    def test_redact_false_returns_no_image_but_populates_detection(self) -> None:
        email_box = BoundingBox(x=4, y=8, width=40, height=12)
        segments = [
            TextSegment(text="user@example.com", box=email_box, confidence=0.95),
        ]
        pipeline = _build_pipeline(segments)

        image = _blank_image()
        result = pipeline.process(image, redact=False)

        assert result.quality_sufficient is True
        # Detection still runs and populates the result (Req 7.1)...
        assert result.detection_result.count == 1
        assert SensitiveCategory.EMAIL in result.detection_result.regions[0].categories
        # ...but no redacted image is produced when redaction is not requested.
        assert result.redacted_image is None
