"""Core error hierarchy for the PII guardrail.

Framework-agnostic and dependency-free: this module MUST import cleanly without
PaddleOCR or FastAPI installed. The Backend_Service maps these exceptions to HTTP
responses; the core only raises them. See the design document's "Error Handling"
section for the authoritative hierarchy.

Requirements: 2.8, 2.9, 1.3, 1.5, 5.3.
"""

from __future__ import annotations

__all__ = [
    "GuardrailError",
    "InvalidImageError",
    "UnsupportedFormatError",
    "FileTooLargeError",
    "OCRProcessingError",
    "ClassifierUnavailableError",
]


class GuardrailError(Exception):
    """Base class for all errors raised by the PII guardrail core.

    Callers (e.g. the Backend_Service) can catch this single type to handle any
    guardrail-originated failure and map it to an appropriate response.
    """


class InvalidImageError(GuardrailError):
    """Raised when the input image is missing, corrupt, or otherwise unusable.

    Covers the general "bad input" case: an empty buffer, a file that cannot be
    decoded into a valid image, or an unsupported input that is rejected before
    processing begins.

    Requirement 2.8.
    """


class UnsupportedFormatError(InvalidImageError):
    """Raised when the input image is not a supported format (non-PNG/JPEG).

    A specialization of :class:`InvalidImageError` for the specific case where
    the file decodes/identifies as a format the guardrail does not accept. The
    optional ``received_format`` is woven into a helpful message when provided.

    Requirement 1.3.
    """

    #: Formats the guardrail accepts. Referenced when building error messages.
    SUPPORTED_FORMATS: tuple[str, ...] = ("PNG", "JPEG")

    def __init__(
        self,
        message: str | None = None,
        *,
        received_format: str | None = None,
    ) -> None:
        self.received_format = received_format
        if message is None:
            supported = ", ".join(self.SUPPORTED_FORMATS)
            if received_format:
                message = (
                    f"Unsupported image format {received_format!r}; "
                    f"supported formats are: {supported}."
                )
            else:
                message = f"Unsupported image format; supported formats are: {supported}."
        super().__init__(message)


class FileTooLargeError(GuardrailError):
    """Raised when the input file exceeds the configured size limit.

    The optional ``size_limit_bytes`` and ``actual_size_bytes`` are woven into a
    helpful message when provided, so callers can surface exactly how far over
    the limit the upload was.

    Requirement 1.5.
    """

    def __init__(
        self,
        message: str | None = None,
        *,
        size_limit_bytes: int | None = None,
        actual_size_bytes: int | None = None,
    ) -> None:
        self.size_limit_bytes = size_limit_bytes
        self.actual_size_bytes = actual_size_bytes
        if message is None:
            if size_limit_bytes is not None and actual_size_bytes is not None:
                message = (
                    f"File is too large: {actual_size_bytes} bytes exceeds the "
                    f"limit of {size_limit_bytes} bytes."
                )
            elif size_limit_bytes is not None:
                message = (
                    f"File is too large: exceeds the limit of "
                    f"{size_limit_bytes} bytes."
                )
            elif actual_size_bytes is not None:
                message = f"File is too large: {actual_size_bytes} bytes exceeds the size limit."
            else:
                message = "File is too large: exceeds the configured size limit."
        super().__init__(message)


class OCRProcessingError(GuardrailError):
    """Raised when text extraction fails after valid input was accepted.

    Distinct from :class:`InvalidImageError`: the input was a valid, supported
    image, but the OCR engine failed while processing it. This separates
    "bad input" from "processing failure".

    Requirement 2.9.
    """


class ClassifierUnavailableError(GuardrailError):
    """Raised when the ML classifier is invoked while unavailable.

    This is caught internally by the Detector, which falls back to pattern-based
    detection and appends a warning; it is not surfaced to the user flow as a
    hard failure.

    Requirement 5.3.
    """
