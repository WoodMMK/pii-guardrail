"""Upload validation and image loading for the Backend_Service.

This module is the transport-edge gate that runs BEFORE the reusable core
(:mod:`pii_guardrail`) ever sees an upload. It answers three questions about a
set of uploaded bytes:

    1. Is it small enough?        -> :func:`validate_size`
    2. Is it a supported format?  -> :func:`validate_format`
    3. Can it be decoded into an image array the core can consume?
                                  -> :func:`load_image`

Failures are mapped onto the core's canonical error hierarchy so the endpoint
(tasks 12.4/12.5) can translate a single exception type to an HTTP response:

    * over the size limit          -> :class:`~pii_guardrail.errors.FileTooLargeError`
    * not PNG/JPEG                  -> :class:`~pii_guardrail.errors.UnsupportedFormatError`
    * corrupt bytes that sniff OK   -> :class:`~pii_guardrail.errors.InvalidImageError`

Design references:
    * "Pipeline stages -> 1. Upload & validate" and the error-mapping table in
      design.md (415 UNSUPPORTED_FORMAT / 413 FILE_TOO_LARGE / 400 INVALID_IMAGE).
    * "Max upload size is a configured constant (default 10 MB)".

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5.

Dependency-light on purpose: only Pillow + NumPy (both core deps) are imported,
so this module is importable and testable without FastAPI installed. Format is
detected from the actual bytes (magic bytes via Pillow), never trusting a
client-supplied content-type or filename, which can be spoofed.
"""

from __future__ import annotations

import io

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

from pii_guardrail.errors import (
    FileTooLargeError,
    InvalidImageError,
    UnsupportedFormatError,
)

__all__ = [
    "DEFAULT_MAX_UPLOAD_SIZE_BYTES",
    "SUPPORTED_FORMATS",
    "sniff_format",
    "validate_size",
    "validate_format",
    "load_image",
    "validate_and_load",
]


# --- Configuration -----------------------------------------------------------

#: Default maximum accepted upload size, in bytes (10 MB). Surfaced in the
#: :class:`FileTooLargeError` when an upload is rejected. (Requirement 1.5,
#: design.md default recommendation.) Callers may override per request.
DEFAULT_MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024

#: Formats the guardrail accepts. Mirrors
#: :attr:`pii_guardrail.errors.UnsupportedFormatError.SUPPORTED_FORMATS`
#: (Requirement 1.4). Pillow reports JPEGs with the format tag ``"JPEG"``.
SUPPORTED_FORMATS: tuple[str, ...] = ("PNG", "JPEG")


# --- Format sniffing ---------------------------------------------------------

def sniff_format(data: bytes) -> str | None:
    """Return Pillow's format tag for ``data`` by inspecting the bytes.

    Detection is by content (magic bytes), not by any client-supplied
    content-type or filename, both of which can be spoofed (Requirement 1.3
    "identifying the unsupported format"). Returns the Pillow format string
    (e.g. ``"PNG"``, ``"JPEG"``, ``"GIF"``) when the bytes identify as a known
    image, or ``None`` when the bytes are not a recognizable image at all.

    This only sniffs the header; it does not fully decode the image, so it is
    cheap and does not raise on truncated-but-recognizable content.
    """
    if not data:
        return None
    try:
        with Image.open(io.BytesIO(data)) as img:
            return img.format
    except (UnidentifiedImageError, OSError, ValueError):
        return None


# --- Individual checks -------------------------------------------------------

def validate_size(
    data: bytes,
    *,
    max_size_bytes: int = DEFAULT_MAX_UPLOAD_SIZE_BYTES,
) -> None:
    """Raise :class:`FileTooLargeError` when ``data`` exceeds the size limit.

    The limit and the actual size are woven into the error so the caller can
    surface exactly how far over the limit the upload was (Requirement 1.5).
    An upload exactly at the limit is accepted.

    Args:
        data: The raw uploaded bytes.
        max_size_bytes: The maximum accepted size in bytes (default 10 MB).

    Raises:
        FileTooLargeError: When ``len(data)`` exceeds ``max_size_bytes``.
    """
    actual = len(data)
    if actual > max_size_bytes:
        raise FileTooLargeError(
            size_limit_bytes=max_size_bytes,
            actual_size_bytes=actual,
        )


def validate_format(data: bytes) -> str:
    """Verify ``data`` is a supported (PNG/JPEG) image and return its format.

    The format is determined from the bytes themselves (magic bytes via
    Pillow), never from a content-type header or filename, so a spoofed
    content-type cannot slip an unsupported format past the gate
    (Requirements 1.3, 1.4).

    Args:
        data: The raw uploaded bytes.

    Returns:
        The canonical supported format tag: ``"PNG"`` or ``"JPEG"``.

    Raises:
        UnsupportedFormatError: When the bytes are not a supported format. The
            ``received_format`` carries the identified format when the bytes are
            a recognizable-but-unsupported image (e.g. ``"GIF"``), or ``None``
            when the bytes are not a recognizable image at all.
    """
    detected = sniff_format(data)
    if detected not in SUPPORTED_FORMATS:
        raise UnsupportedFormatError(received_format=detected)
    return detected


# --- Image loading -----------------------------------------------------------

def load_image(data: bytes) -> NDArray[np.uint8]:
    """Decode supported image ``data`` into a BGR NumPy array for the core.

    The returned array is a 3-channel, ``uint8``, BGR-ordered image, matching
    what OpenCV / PaddleOCR expect and what the Preprocessor accepts (2-D or 3-D
    arrays with 1/3/4 channels). Alpha is composited onto white and dropped,
    grayscale is expanded to 3 channels, and channel order is converted from
    Pillow's RGB to OpenCV's BGR.

    This performs a FULL decode (unlike :func:`validate_format`, which only
    sniffs the header), so it will reject bytes that pass the format sniff but
    are actually truncated or corrupt.

    Args:
        data: The raw uploaded bytes, already validated as PNG/JPEG.

    Returns:
        A ``(H, W, 3)`` ``uint8`` BGR image array.

    Raises:
        InvalidImageError: When the bytes cannot be decoded into a valid image
            (corrupt/truncated), or decode to an empty image.
    """
    try:
        with Image.open(io.BytesIO(data)) as img:
            # verify() detects truncation/corruption but consumes the file, so
            # a fresh handle is needed afterwards to actually load the pixels.
            img.verify()
        with Image.open(io.BytesIO(data)) as img:
            has_alpha = img.mode in ("RGBA", "LA", "P") and (
                "transparency" in img.info or img.mode in ("RGBA", "LA")
            )
            if has_alpha:
                # Composite onto a white background so transparent regions do
                # not become black artifacts the OCR could mistake for content.
                rgba = img.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                rgb = Image.alpha_composite(background, rgba).convert("RGB")
            else:
                rgb = img.convert("RGB")
            arr = np.asarray(rgb, dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise InvalidImageError(
            "Uploaded bytes could not be decoded into a valid image "
            "(corrupt or truncated file)."
        ) from exc

    if arr.size == 0 or arr.ndim != 3 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise InvalidImageError("Decoded image is empty.")

    # Pillow yields RGB; the core/OpenCV/PaddleOCR expect BGR. Reverse the last
    # axis (no OpenCV dependency needed here) and return a contiguous array.
    bgr = np.ascontiguousarray(arr[:, :, ::-1])
    return bgr


# --- Orchestration -----------------------------------------------------------

def validate_and_load(
    data: bytes,
    *,
    max_size_bytes: int = DEFAULT_MAX_UPLOAD_SIZE_BYTES,
) -> NDArray[np.uint8]:
    """Validate size + format, then decode ``data`` into a BGR image array.

    Runs the full transport-edge gate in order (size, then format, then decode)
    and returns the loaded image ready for :meth:`GuardrailPipeline.process`.

    Args:
        data: The raw uploaded bytes.
        max_size_bytes: The maximum accepted size in bytes (default 10 MB).

    Returns:
        A ``(H, W, 3)`` ``uint8`` BGR image array.

    Raises:
        FileTooLargeError: When the upload exceeds the size limit (1.5).
        UnsupportedFormatError: When the bytes are not PNG/JPEG (1.3, 1.4).
        InvalidImageError: When supported-looking bytes fail to decode (2.8).
    """
    validate_size(data, max_size_bytes=max_size_bytes)
    validate_format(data)
    return load_image(data)
