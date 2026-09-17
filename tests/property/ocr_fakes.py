"""Shared OCR test doubles for the OCR_Engine tests.

Two backends satisfying the :class:`~pii_guardrail.ocr.OCRBackend` protocol:

- :class:`FakeOCRBackend` -- returns a controlled, caller-supplied list of
  ``TextSegment`` values (including the empty list) and reports a toggleable
  ``available`` flag. Used to drive deterministic OCR_Engine behavior in tests
  without any real OCR model (Properties 4 and 5).
- :class:`RaisingOCRBackend` -- accepts a valid image but always raises during
  extraction, modeling a processing failure after valid input so the engine's
  fail-closed behavior can be exercised (Property 6).

These live in a shared module (rather than inline in one test file) so the
OCR property tests (tasks 6.2, 6.3, 6.4) can all reuse them.
"""

from __future__ import annotations

from numpy.typing import NDArray

from pii_guardrail.models import TextSegment


class FakeOCRBackend:
    """An OCRBackend returning a fixed list of segments.

    Structurally satisfies the ``OCRBackend`` protocol: exposes ``run`` and an
    ``available`` property. Construct with the exact segments ``run`` should
    return (defaults to an empty list, i.e. "no text found"), and optionally
    mark the backend unavailable to exercise the engine's availability gate.
    """

    def __init__(
        self,
        segments: list[TextSegment] | None = None,
        *,
        available: bool = True,
    ) -> None:
        self._segments = list(segments) if segments is not None else []
        self._available = available
        #: Records the images passed to ``run`` for assertions/spying.
        self.calls: list[NDArray] = []

    @property
    def available(self) -> bool:
        return self._available

    def run(self, image: NDArray) -> list[TextSegment]:
        self.calls.append(image)
        # Return a fresh copy so callers cannot mutate the configured list.
        return list(self._segments)


class RaisingOCRBackend:
    """An OCRBackend that accepts valid input but always fails in ``run``.

    Models an OCR processing failure that occurs *after* a valid image is
    accepted (Requirement 2.9): ``available`` is ``True`` so the engine proceeds
    to call ``run``, which then raises. The engine is expected to translate this
    into an :class:`~pii_guardrail.errors.OCRProcessingError` and produce no
    partial result.
    """

    def __init__(self, *, available: bool = True, exc: Exception | None = None) -> None:
        self._available = available
        self._exc = exc if exc is not None else RuntimeError("simulated OCR failure")

    @property
    def available(self) -> bool:
        return self._available

    def run(self, image: NDArray) -> list[TextSegment]:
        raise self._exc
