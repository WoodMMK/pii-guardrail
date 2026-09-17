"""GuardrailPipeline: the primary reuse surface wiring the core together.

Orchestrates a single image through the full flow:

    preprocess -> OCR -> detect -> (optional) redact

This is the framework-agnostic entry point the Backend_Service (and any
independent reuse project) calls. It has no HTTP/web dependency and works on
in-memory image arrays, returning a plain :class:`PipelineResult`.

Design reference: the "GuardrailPipeline (core entry point)" section of
design.md.

Requirements:
    2.6 -- no recognized text yields a Detection_Result with zero regions.
    6.4 -- a >=150 DPI scan produces a Detection_Result identifying its regions.
    6.6 -- below-threshold / unprocessable quality yields no reliable regions
           plus a quality-insufficient indication (no OCR/detect is run).
    7.1 -- detection produces a Detection_Result listing each Sensitive_Region.
    8.1 -- when regions exist, redaction covers each with a black rectangle.
    8.4 -- zero regions redacts to an image equivalent to the input.

Only the core components and models are imported here; no PaddleOCR/FastAPI
dependency is introduced at import time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pii_guardrail.models import DetectionResult

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pii_guardrail.detector import Detector
    from pii_guardrail.models import TextSegment
    from pii_guardrail.ocr import OCREngine
    from pii_guardrail.preprocessor import Preprocessor
    from pii_guardrail.redactor import Redactor

__all__ = ["GuardrailPipeline", "PipelineResult"]

#: Warning surfaced when preprocessing reports the image quality is below the
#: 150 DPI reliability threshold (or the image is otherwise unprocessable). In
#: that case no OCR/detection is run and no reliable regions are asserted.
#: (Requirement 6.6)
_INSUFFICIENT_QUALITY_WARNING = (
    "Image quality is insufficient for reliable detection "
    "(estimated resolution below the 150 DPI threshold); "
    "no sensitive regions were identified."
)

#: Warning surfaced when a low-resolution input was auto-upscaled to reach the
#: reliability threshold (Option A). The returned Redacted_Image is the UPSCALED
#: image, so its pixel dimensions are LARGER than the uploaded image, and OCR ran
#: on interpolated (not genuinely higher-resolution) pixels.
_UPSCALED_IMAGE_WARNING = (
    "Input resolution was below the 150 DPI threshold, so the image was "
    "upscaled {factor:.1f}x before detection. The returned image is larger "
    "than the original ({width}x{height}), and detection accuracy may be "
    "reduced because upscaling does not recover detail the source lacked."
)


@dataclass
class PipelineResult:
    """The outcome of running :meth:`GuardrailPipeline.process` on one image.

    Attributes:
        detection_result: The :class:`~pii_guardrail.models.DetectionResult`
            listing every located :class:`~pii_guardrail.models.SensitiveRegion`
            together with the (preprocessed) image dimensions. Contains zero
            regions when no text was found or when quality was insufficient.
        redacted_image: The redacted image when ``redact`` was requested and the
            image had sufficient quality; ``None`` when redaction was skipped
            (``redact=False``) or the quality was insufficient (no reliable
            regions to redact).
        quality_sufficient: ``False`` when preprocessing flagged the image as
            below the reliability threshold (Requirement 6.6); ``True``
            otherwise.
        warnings: Accumulated human-readable warnings, e.g. the
            classifier-fallback notice from the Detector or the
            insufficient-quality indication.
        ocr_segments: Every raw :class:`~pii_guardrail.models.TextSegment` the
            OCR engine recognized on the (preprocessed) image, in reading order,
            BEFORE classification. Useful for debugging what the OCR actually
            read -- including non-sensitive text that never becomes a region.
            Empty when no text was found or when quality was insufficient (OCR
            is not run in that case).
        processed_image: The image the pipeline actually operated on -- i.e. the
            preprocessed (denoised/deskewed and, when applicable, UPSCALED)
            image. This is the coordinate space that ``detection_result``'s
            boxes and ``image_width``/``image_height`` refer to, and the space
            ``redacted_image`` (when present) is in. Callers that need to return
            or overlay something aligned with the detection boxes -- e.g. when
            ``redacted_image`` is ``None`` (``redact=False`` or insufficient
            quality) -- MUST use this image rather than the original upload,
            whose dimensions differ after an upscale.
    """

    detection_result: DetectionResult
    redacted_image: "NDArray | None"
    quality_sufficient: bool
    warnings: list[str] = field(default_factory=list)
    ocr_segments: "list[TextSegment]" = field(default_factory=list)
    processed_image: "NDArray | None" = None
    #: Categories assigned to each OCR segment, aligned by index with
    #: ``ocr_segments``. An empty set means that segment was classified
    #: non-sensitive (and thus not redacted). Exposes the full per-segment
    #: classification for debugging why a segment was or was not redacted.
    segment_categories: "list[set]" = field(default_factory=list)
    #: Per-segment, per-SOURCE classification breakdown, aligned by index with
    #: ``ocr_segments``. Each entry maps a layer label ("pattern", "llm",
    #: "presidio", "detect-secrets") to the categories that layer assigned, so
    #: a debug view can attribute each detection to the classifier that made it.
    segment_sources: "list[dict]" = field(default_factory=list)


class GuardrailPipeline:
    """Wire the core components into a single reusable ``process`` entry point."""

    def __init__(
        self,
        preprocessor: "Preprocessor",
        ocr_engine: "OCREngine",
        detector: "Detector",
        redactor: "Redactor",
    ) -> None:
        """Create a pipeline from its four core components.

        Args:
            preprocessor: Denoise/deskew/quality-estimation stage.
            ocr_engine: Text-extraction stage (injectable OCR backend).
            detector: Classification stage producing sensitive regions.
            redactor: Black-box overlay stage.
        """
        self._preprocessor = preprocessor
        self._ocr_engine = ocr_engine
        self._detector = detector
        self._redactor = redactor

    def process(self, image: "NDArray", *, redact: bool = True) -> PipelineResult:
        """Run one image through preprocess -> OCR -> detect -> (optional) redact.

        Steps:
            1. Preprocess (denoise, deskew, estimate quality). Raises
               :class:`~pii_guardrail.errors.InvalidImageError` on an
               empty/corrupt array, which propagates to the caller.
            2. If quality is insufficient (Requirement 6.6), short-circuit: no
               OCR/detection is run on an unreliable image. Return a
               :class:`PipelineResult` with ``quality_sufficient=False``, a
               zero-region :class:`~pii_guardrail.models.DetectionResult` sized
               to the preprocessed image, ``redacted_image=None``, and a
               non-empty insufficient-quality warning.
            3. Otherwise, extract text (Requirement 2.6: an empty result is
               normal), detect sensitive regions, and carry any detector
               warnings through.
            4. If ``redact`` is requested, black-box every region on the
               preprocessed image so the returned boxes align with the returned
               redacted image (Requirements 8.1, 8.4).

        Args:
            image: The input image as a 2-D (grayscale) or 3-D (color) NumPy
                array.
            redact: When ``True`` (the default), also produce the redacted
                image. When ``False``, ``redacted_image`` is ``None``.

        Returns:
            A :class:`PipelineResult`.

        Raises:
            InvalidImageError: Missing/corrupt/unsupported input (from
                preprocessing or OCR). Propagated; no result is produced.
            OCRProcessingError: OCR failed after accepting a valid image.
                Propagated; no partial result is produced (Requirement 2.9).
        """
        # 1. Preprocess. InvalidImageError propagates to the caller.
        preprocessed = self._preprocessor.preprocess(image)
        work_image = preprocessed.image
        height, width = work_image.shape[0], work_image.shape[1]

        # 2. Quality gate (Requirement 6.6): do not run OCR/detect on an image
        #    we cannot detect reliably. Short-circuit with zero regions and a
        #    quality-insufficient indication.
        if not preprocessed.quality_sufficient:
            return PipelineResult(
                detection_result=DetectionResult(
                    regions=[],
                    image_width=int(width),
                    image_height=int(height),
                ),
                redacted_image=None,
                quality_sufficient=False,
                warnings=[_INSUFFICIENT_QUALITY_WARNING],
                # Carry the processed image so callers return something in the
                # SAME coordinate space as detection_result (see field docs).
                processed_image=work_image,
            )

        # 3. OCR then detect on the preprocessed image. OCRProcessingError /
        #    InvalidImageError propagate; an empty segment list is a normal
        #    result yielding zero regions (Requirement 2.6).
        segments = self._ocr_engine.extract(work_image)
        outcome = self._detector.detect(segments)

        detection_result = DetectionResult(
            regions=outcome.regions,
            image_width=int(width),
            image_height=int(height),
        )

        # 4. Optional redaction on the SAME (preprocessed) image so region boxes
        #    align with the returned redacted image (Requirements 8.1, 8.4).
        redacted_image: "NDArray | None" = None
        if redact:
            redacted_image = self._redactor.redact(work_image, outcome.regions)

        # Carry detector warnings (e.g. classifier-fallback) through, and when
        # the input was auto-upscaled, prepend a warning noting that the returned
        # image is larger than the upload and accuracy may be reduced (Option A).
        warnings = list(outcome.warnings)
        if getattr(preprocessed, "upscaled", False):
            warnings.insert(
                0,
                _UPSCALED_IMAGE_WARNING.format(
                    factor=getattr(preprocessed, "upscale_factor", 1.0),
                    width=int(width),
                    height=int(height),
                ),
            )

        return PipelineResult(
            detection_result=detection_result,
            redacted_image=redacted_image,
            quality_sufficient=True,
            warnings=warnings,
            ocr_segments=list(segments),
            processed_image=work_image,
            segment_categories=list(outcome.segment_categories),
            segment_sources=list(outcome.segment_sources),
        )
