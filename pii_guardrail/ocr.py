"""OCR_Engine and the injectable OCRBackend abstraction.

Responsibility: extract recognized text segments and their bounding boxes (with
confidence) from an image, using a PaddleOCR-family model when available. The
real PaddleOCR-backed backend lives in a separate module (task 13) and is
imported lazily so this core stays importable without PaddleOCR installed.

Design reference: the "OCR_Engine" section of design.md.

Requirements:
    2.1 -- extract recognized text segments, each carrying its text string.
    2.2 -- produce an axis-aligned Bounding_Box (top-left origin) for each
           segment, normalized from the backend's quadrilateral output.
    2.6 -- no recognized text is a normal result: return ``[]`` (not an error).
    2.7 -- each segment carries a confidence value in [0.0, 1.0] inclusive.
    2.8 -- missing/corrupt/unsupported input is rejected with ``InvalidImageError``
           and produces no result.
    2.9 -- a failure during extraction after a valid image was accepted raises
           ``OCRProcessingError`` and produces no partial result.

Only NumPy is used at import time here; both NumPy and the backend abstraction
keep the core free of any PaddleOCR/FastAPI dependency at import time.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from pii_guardrail.errors import InvalidImageError, OCRProcessingError
from pii_guardrail.geometry import normalize_quad_to_box
from pii_guardrail.models import BoundingBox, TextSegment

__all__ = ["OCRBackend", "OCREngine"]


@runtime_checkable
class OCRBackend(Protocol):
    """Structural contract for a text-extraction backend.

    Abstracts the concrete OCR implementation (PaddleOCR by default) so it can
    be swapped or replaced by a test double. Any object exposing ``run`` and an
    ``available`` property satisfies this protocol without an explicit base
    class or an ML dependency, keeping the core importable without PaddleOCR.

    Requirement 2.5 (the engine uses a PaddleOCR-family model *where available*)
    motivates the injectable backend: availability is a runtime concern.
    """

    def run(self, image: NDArray) -> list[TextSegment]:
        """Return recognized text segments for ``image``.

        Implementations receive a validated image array and return zero or more
        :class:`~pii_guardrail.models.TextSegment` values. Returning an empty
        list means "no text found" and is a normal result, not an error
        (Requirement 2.6). Implementations may raise on genuine processing
        failures; :class:`OCREngine` translates those into
        :class:`~pii_guardrail.errors.OCRProcessingError`.
        """
        ...

    @property
    def available(self) -> bool:
        """Whether the backend is loaded and ready to extract text."""
        ...


class OCREngine:
    """Extract text segments from an image via an injectable :class:`OCRBackend`.

    The engine owns the framework-agnostic contract around extraction:
    validating input, normalizing each segment's box to an in-bounds
    axis-aligned :class:`~pii_guardrail.models.BoundingBox` (Requirement 2.2),
    clamping confidence into [0.0, 1.0] (Requirement 2.7), treating an empty
    result as a normal success (Requirement 2.6), and failing closed with no
    partial result on error (Requirements 2.8, 2.9).

    The default backend is the PaddleOCR-backed one (task 13); it is injected
    rather than imported here so the core imports cleanly without PaddleOCR.
    """

    def __init__(self, backend: OCRBackend | None = None) -> None:
        """Create an OCREngine.

        Args:
            backend: The OCR backend to delegate extraction to. When ``None``,
                the engine attempts to construct the default PaddleOCR-backed
                backend lazily on first use; if that backend (or its optional
                dependency) is unavailable, :meth:`extract` raises
                :class:`~pii_guardrail.errors.OCRProcessingError`.
        """
        self._backend = backend

    @property
    def backend(self) -> OCRBackend | None:
        """The configured backend, or ``None`` when no default could be loaded."""
        return self._backend

    def extract(self, image: NDArray) -> list[TextSegment]:
        """Extract recognized text segments from ``image``.

        - Recognizes Thai and Latin script (via the backend).
        - Each returned segment has its text, an axis-aligned
          :class:`~pii_guardrail.models.BoundingBox` in pixel coordinates with a
          top-left origin, and a confidence clamped into [0.0, 1.0].
        - Returns ``[]`` when the backend finds no text (a normal result, not an
          error).

        Args:
            image: The image to read, as a 2-D (grayscale) or 3-D (color) NumPy
                array. This is typically the preprocessed (denoised/deskewed)
                image.

        Returns:
            A list of normalized :class:`~pii_guardrail.models.TextSegment`
            values, possibly empty.

        Raises:
            InvalidImageError: If ``image`` is missing, corrupt, or an
                unsupported array shape (Requirement 2.8). No result is produced.
            OCRProcessingError: If the backend is unavailable, or extraction
                fails after a valid image was accepted (Requirement 2.9). No
                partial result is produced.
        """
        # 1. Reject bad input up front, before any backend runs (Requirement 2.8).
        self._validate(image)
        height, width = image.shape[0], image.shape[1]

        backend = self._resolve_backend()

        # A backend that reports itself unavailable cannot extract; this is a
        # processing failure after accepting valid input (Requirement 2.9).
        try:
            available = backend.available
        except Exception as exc:  # pragma: no cover - defensive
            raise OCRProcessingError(
                "OCR backend failed while reporting availability."
            ) from exc
        if not available:
            raise OCRProcessingError("OCR backend is not available.")

        # 2. Delegate extraction. Any failure here is a processing failure, not
        #    bad input, and must produce no partial result (Requirement 2.9).
        try:
            raw_segments = backend.run(image)
        except InvalidImageError:
            # The backend rejected the input as invalid; surface as-is (2.8).
            raise
        except OCRProcessingError:
            # Already the right type; propagate without wrapping.
            raise
        except Exception as exc:
            raise OCRProcessingError(
                "OCR backend failed during text extraction."
            ) from exc

        # 3. Normalize every segment into the canonical, in-bounds form. Any
        #    malformed segment is a processing failure; we build the whole list
        #    first and only then return, so no partial result escapes on error.
        try:
            segments = [
                self._normalize_segment(seg, width, height) for seg in raw_segments
            ]
        except OCRProcessingError:
            raise
        except Exception as exc:
            raise OCRProcessingError(
                "OCR backend returned a malformed text segment."
            ) from exc

        return segments

    # -- internals ----------------------------------------------------------

    def _resolve_backend(self) -> OCRBackend:
        """Return the configured backend, lazily loading the default if needed.

        The default PaddleOCR-backed backend (task 13) is imported lazily so the
        core imports without PaddleOCR. If it cannot be constructed (missing
        optional dependency), extraction fails closed with
        :class:`~pii_guardrail.errors.OCRProcessingError`.
        """
        if self._backend is not None:
            return self._backend
        try:  # Lazy import: keeps PaddleOCR optional (task 13 provides this).
            from pii_guardrail.paddle_backend import PaddleOCRBackend
        except Exception as exc:
            raise OCRProcessingError(
                "No OCR backend configured and the default PaddleOCR backend "
                "is unavailable."
            ) from exc
        self._backend = PaddleOCRBackend()
        return self._backend

    def _normalize_segment(
        self, segment: TextSegment, width: int, height: int
    ) -> TextSegment:
        """Return a segment with an in-bounds box and a clamped confidence.

        Ensures the box is an axis-aligned, in-image :class:`BoundingBox`
        (Requirement 2.2) and the confidence lies in [0.0, 1.0] (Requirement
        2.7). A box already inside the image is re-clamped defensively via its
        corner vertices so backend boxes that overrun the image are corrected.
        """
        box = self._normalize_box(segment.box, width, height)
        confidence = self._clamp_confidence(segment.confidence)
        return TextSegment(text=segment.text, box=box, confidence=confidence)

    @staticmethod
    def _normalize_box(box: BoundingBox, width: int, height: int) -> BoundingBox:
        """Clamp an axis-aligned box inside the image via geometry normalization.

        Reuses :func:`~pii_guardrail.geometry.normalize_quad_to_box` (task 2.1)
        by treating the box's four corners as the source quadrilateral, so the
        result is guaranteed in-bounds with a strictly positive extent.
        """
        left = box.x
        top = box.y
        right = box.x + box.width
        bottom = box.y + box.height
        corners = ((left, top), (right, top), (right, bottom), (left, bottom))
        return normalize_quad_to_box(corners, width, height)

    @staticmethod
    def _clamp_confidence(confidence: float) -> float:
        """Clamp ``confidence`` into [0.0, 1.0], mapping NaN to 0.0 (2.7)."""
        try:
            value = float(confidence)
        except (TypeError, ValueError):
            return 0.0
        if value != value:  # NaN
            return 0.0
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    @staticmethod
    def _validate(image: NDArray) -> None:
        """Raise ``InvalidImageError`` for a missing/empty/corrupt array (2.8).

        Mirrors the Preprocessor's validation so the OCR stage independently
        rejects bad input even when called directly (not via the pipeline).
        """
        if image is None:
            raise InvalidImageError("Input image is None.")
        if not isinstance(image, np.ndarray):
            raise InvalidImageError(
                f"Input image must be a NumPy array, got {type(image).__name__}."
            )
        if image.size == 0:
            raise InvalidImageError("Input image is empty (zero-size array).")
        if image.ndim not in (2, 3):
            raise InvalidImageError(
                f"Input image must be 2-D (grayscale) or 3-D (color), got "
                f"{image.ndim} dimensions."
            )
        if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
            raise InvalidImageError(
                f"Color image must have 1, 3, or 4 channels, got {image.shape[2]}."
            )
        if image.shape[0] < 1 or image.shape[1] < 1:
            raise InvalidImageError("Input image has a zero-length spatial axis.")
