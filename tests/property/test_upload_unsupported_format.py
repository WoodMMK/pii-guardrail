"""Property-based test: unsupported upload formats are rejected with the format identified.

# Feature: thai-image-pii-guardrail, Property 1: Unsupported formats are rejected with the format identified

Validates: Requirements 1.3, 1.4

The transport-edge gate (:func:`backend_service.validation.validate_format`)
must reject any upload that is not PNG/JPEG and surface *which* format it saw,
so the Web_Interface can tell the user exactly what was received (Req 1.3). The
set of accepted formats is exactly PNG and JPEG (Req 1.4).

We exercise the rejection path with real bytes: a tiny image is generated with
Pillow and re-encoded into a variety of recognizable-but-unsupported formats
(GIF/BMP/TIFF/WEBP), plus a garbage-bytes case that is not a recognizable image
at all. Supported PNG/JPEG bytes are included as a control that must NOT raise.
"""

from __future__ import annotations

import io

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image

from backend_service.validation import SUPPORTED_FORMATS, validate_format
from pii_guardrail.errors import UnsupportedFormatError

# Formats Pillow can both write and sniff back to the same tag in this venv,
# and that the guardrail does NOT accept. Verified writable at authoring time.
UNSUPPORTED_FORMATS: tuple[str, ...] = ("GIF", "BMP", "TIFF", "WEBP")


@st.composite
def unsupported_image_bytes(draw: st.DrawFn) -> tuple[str, bytes]:
    """Build ``(expected_format_tag, bytes)`` for a real unsupported image.

    Varies the format, dimensions, and fill colour so the property is exercised
    across a family of inputs rather than a single fixed image.
    """
    fmt = draw(st.sampled_from(UNSUPPORTED_FORMATS))
    width = draw(st.integers(min_value=1, max_value=16))
    height = draw(st.integers(min_value=1, max_value=16))
    color = (
        draw(st.integers(min_value=0, max_value=255)),
        draw(st.integers(min_value=0, max_value=255)),
        draw(st.integers(min_value=0, max_value=255)),
    )
    image = Image.new("RGB", (width, height), color)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return fmt, buffer.getvalue()


@st.composite
def supported_image_bytes(draw: st.DrawFn) -> tuple[str, bytes]:
    """Build ``(expected_format_tag, bytes)`` for a real PNG or JPEG image."""
    fmt = draw(st.sampled_from(SUPPORTED_FORMATS))
    width = draw(st.integers(min_value=1, max_value=16))
    height = draw(st.integers(min_value=1, max_value=16))
    color = (
        draw(st.integers(min_value=0, max_value=255)),
        draw(st.integers(min_value=0, max_value=255)),
        draw(st.integers(min_value=0, max_value=255)),
    )
    image = Image.new("RGB", (width, height), color)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return fmt, buffer.getvalue()


@pytest.mark.property
@settings(max_examples=200)
@given(sample=unsupported_image_bytes())
def test_recognizable_unsupported_formats_rejected_with_format_identified(
    sample: tuple[str, bytes],
) -> None:
    """Recognizable non-PNG/JPEG images are rejected, naming the format.

    For any tiny image re-encoded as GIF/BMP/TIFF/WEBP, ``validate_format``
    raises :class:`UnsupportedFormatError` whose ``received_format`` equals the
    sniffed format tag, and that tag appears in the error message so the caller
    can identify what was received.

    # Feature: thai-image-pii-guardrail, Property 1: Unsupported formats are rejected with the format identified
    Validates: Requirements 1.3, 1.4
    """
    expected_format, data = sample

    # The format is not one the guardrail accepts (Req 1.4 boundary).
    assert expected_format not in SUPPORTED_FORMATS

    with pytest.raises(UnsupportedFormatError) as exc_info:
        validate_format(data)

    exc = exc_info.value
    # The received format is identified on the exception (Req 1.3)...
    assert exc.received_format == expected_format
    # ...and echoed in the human-readable message.
    assert expected_format in str(exc)


@pytest.mark.property
@settings(max_examples=200)
@given(data=st.binary(min_size=0, max_size=64))
def test_unrecognizable_bytes_rejected_as_unknown_format(data: bytes) -> None:
    """Bytes that are not a recognizable image are rejected as unknown format.

    Garbage bytes that Pillow cannot identify as any image sniff to ``None``;
    ``validate_format`` still rejects them with :class:`UnsupportedFormatError`
    whose ``received_format`` is ``None``, and the message indicates the format
    is unsupported (Req 1.3 "identifying the unsupported format").

    # Feature: thai-image-pii-guardrail, Property 1: Unsupported formats are rejected with the format identified
    Validates: Requirements 1.3, 1.4
    """
    # Guard against the astronomically unlikely case that random bytes happen to
    # form a valid PNG/JPEG header; if they do, this input is out of scope for
    # the "unrecognizable" property.
    try:
        with Image.open(io.BytesIO(data)) as probe:
            sniffed = probe.format
    except Exception:
        sniffed = None
    if sniffed in SUPPORTED_FORMATS:
        return

    with pytest.raises(UnsupportedFormatError) as exc_info:
        validate_format(data)

    exc = exc_info.value
    if sniffed is None:
        assert exc.received_format is None
    else:
        # A recognizable-but-unsupported format still names itself.
        assert exc.received_format == sniffed
        assert sniffed in str(exc)
    assert "nsupported" in str(exc)  # "Unsupported"/"unsupported" appears.


@pytest.mark.property
@settings(max_examples=100)
@given(sample=supported_image_bytes())
def test_supported_formats_are_accepted(sample: tuple[str, bytes]) -> None:
    """Control: PNG/JPEG bytes are accepted and their format is returned.

    Confirms the property is about *unsupported* formats specifically: the
    supported set (PNG, JPEG) passes without raising and echoes the format tag
    (Req 1.4).

    # Feature: thai-image-pii-guardrail, Property 1: Unsupported formats are rejected with the format identified
    Validates: Requirements 1.3, 1.4
    """
    expected_format, data = sample
    assert validate_format(data) == expected_format
