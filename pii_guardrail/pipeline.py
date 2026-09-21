"""OCRPipeline: Pipeline combining Preprocessor and OCREngine.

Provides a unified preprocess -> OCR extraction pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pii_guardrail.models import TextSegment
    from pii_guardrail.ocr import OCREngine
    from pii_guardrail.preprocessor import Preprocessor

__all__ = ["OCRPipeline", "GuardrailPipeline", "PipelineResult"]

_INSUFFICIENT_QUALITY_WARNING = (
    "Image quality is insufficient for reliable text extraction "
    "(estimated resolution below the 150 DPI threshold)."
)


@dataclass
class PipelineResult:
    """Outcome of running the OCR pipeline on one image."""

    ocr_segments: list[TextSegment] = field(default_factory=list)
    processed_image: NDArray | None = None
    quality_sufficient: bool = True
    warnings: list[str] = field(default_factory=list)


class OCRPipeline:
    """Wire Preprocessor and OCREngine into a single OCR pipeline."""

    def __init__(
        self,
        preprocessor: Preprocessor,
        ocr_engine: OCREngine,
        **_kwargs,
    ) -> None:
        self._preprocessor = preprocessor
        self._ocr_engine = ocr_engine

    def process(self, image: NDArray, **_kwargs) -> PipelineResult:
        preprocessed = self._preprocessor.preprocess(image)
        work_image = preprocessed.image

        if not preprocessed.quality_sufficient:
            return PipelineResult(
                ocr_segments=[],
                processed_image=work_image,
                quality_sufficient=False,
                warnings=[_INSUFFICIENT_QUALITY_WARNING],
            )

        segments = self._ocr_engine.extract_text(work_image)
        return PipelineResult(
            ocr_segments=segments,
            processed_image=work_image,
            quality_sufficient=True,
            warnings=list(preprocessed.warnings),
        )


GuardrailPipeline = OCRPipeline
