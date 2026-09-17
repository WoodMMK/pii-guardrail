"""Property-based tests for upload size validation.

# Feature: thai-image-pii-guardrail, Property 2: Oversized uploads are rejected with the limit identified

Validates: Requirements 1.5

These tests exercise ``backend_service.validation.validate_size`` at the
transport edge. To keep examples fast we do NOT allocate real 10 MB buffers;
instead we draw a small configurable ``max_size_bytes`` and construct byte
buffers of an exact length via ``b"\\x00" * length``.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from backend_service.validation import (
    DEFAULT_MAX_UPLOAD_SIZE_BYTES,
    validate_size,
)
from pii_guardrail.errors import FileTooLargeError


def test_default_max_upload_size_is_ten_megabytes() -> None:
    """Sanity check that the documented default limit is 10 MB.

    Validates: Requirements 1.5
    """
    assert DEFAULT_MAX_UPLOAD_SIZE_BYTES == 10 * 1024 * 1024


@pytest.mark.property
@settings(max_examples=200)
@given(
    max_size_bytes=st.integers(min_value=1, max_value=4096),
    delta=st.integers(min_value=1, max_value=4096),
)
def test_oversized_upload_rejected_with_limit_identified(
    max_size_bytes: int, delta: int
) -> None:
    """Over-limit uploads are rejected and the limit is identified in the error.

    # Feature: thai-image-pii-guardrail, Property 2: Oversized uploads are rejected with the limit identified
    Validates: Requirements 1.5
    """
    length = max_size_bytes + delta
    data = b"\x00" * length

    with pytest.raises(FileTooLargeError) as excinfo:
        validate_size(data, max_size_bytes=max_size_bytes)

    # The limit is identified on the exception ...
    assert excinfo.value.size_limit_bytes == max_size_bytes
    # ... and surfaced in the human-readable message (Req 1.5).
    assert str(max_size_bytes) in str(excinfo.value)
    # The actual size is also reported for context.
    assert excinfo.value.actual_size_bytes == length


@pytest.mark.property
@settings(max_examples=200)
@given(data=st.data(), max_size_bytes=st.integers(min_value=1, max_value=4096))
def test_within_limit_upload_passes_size_check(
    data: st.DataObject, max_size_bytes: int
) -> None:
    """Uploads at or under the limit pass the size check (return None).

    Includes the exactly-at-limit boundary (``length == max_size_bytes``), which
    must be accepted.

    # Feature: thai-image-pii-guardrail, Property 2: Oversized uploads are rejected with the limit identified
    Validates: Requirements 1.5
    """
    length = data.draw(st.integers(min_value=0, max_value=max_size_bytes))
    buf = b"\x00" * length

    # Must not raise; validate_size returns None on success.
    assert validate_size(buf, max_size_bytes=max_size_bytes) is None
