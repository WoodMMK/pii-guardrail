"""Task 15.1 -- PaddleOCR extraction integration tests (Thai + Latin samples).

Model/infrastructure-dependent integration tests exercising the REAL PaddleOCR
backend against synthetic sample images generated at test time (see
``sample_corpus.py``). These assert:

    * The default ``OCREngine`` (no injected backend) resolves to a
      ``PaddleOCRBackend`` that reports itself available (Requirement 2.5).
    * Text + boxes are extracted on a representative Latin sample
      (Requirements 2.1, 2.4, 6.1).
    * Text + boxes are extracted on a representative Thai sample, with at least
      one Thai codepoint recognized (Requirement 2.3). Skipped when no
      Thai-capable font is installed.
    * ``GET /api/health`` reports ``ocr_backend_available == True`` when the
      pipeline is wired with a loaded PaddleOCR backend (Requirement 2.5
      surfacing).

The whole module is guarded on PaddleOCR availability: PaddleOCR is imported via
``pytest.importorskip`` and a module-scoped ``backend`` fixture is built ONCE and
reused. If the backend cannot load (native lib issue despite being installed),
every test skips cleanly rather than failing.

Requirements: 2.1, 2.3, 2.4, 2.5, 6.1.
"""

from __future__ import annotations

import pytest

# Guard the whole module: skip everything if paddleocr is not importable.
pytest.importorskip("paddleocr")

from pii_guardrail.detector import Detector  # noqa: E402
from pii_guardrail.models import BoundingBox, TextSegment  # noqa: E402
from pii_guardrail.ocr import OCREngine  # noqa: E402
from pii_guardrail.paddle_backend import PaddleOCRBackend  # noqa: E402
from pii_guardrail.pipeline import GuardrailPipeline  # noqa: E402
from pii_guardrail.preprocessor import Preprocessor  # noqa: E402
from pii_guardrail.redactor import Redactor  # noqa: E402

from tests.integration.sample_corpus import (  # noqa: E402
    make_latin_sample,
    make_thai_sample,
)

pytestmark = pytest.mark.integration


# Thai codepoint range (Unicode Thai block).
_THAI_LOW = 0x0E00
_THAI_HIGH = 0x0E7F


def _has_thai(text: str) -> bool:
    """Return True if ``text`` contains at least one Thai codepoint."""
    return any(_THAI_LOW <= ord(ch) <= _THAI_HIGH for ch in text)


@pytest.fixture(scope="module")
def backend() -> PaddleOCRBackend:
    """Construct the real PaddleOCR backend ONCE for the module.

    PaddleOCR model load is slow (first construction loads/downloads models), so
    this is module-scoped and reused across tests. If the model fails to load
    (e.g. a native dependency issue despite the package being installed), all
    tests in the module skip cleanly.
    """
    be = PaddleOCRBackend()
    if not be.available:
        pytest.skip(
            "PaddleOCR is installed but the model could not be loaded; "
            "skipping PaddleOCR extraction integration tests."
        )
    return be


def _assert_valid_segments(segments: list[TextSegment], width: int, height: int) -> None:
    """Assert a segment list is non-empty and every segment is well-formed."""
    assert len(segments) >= 1, "expected at least one recognized text segment"
    for seg in segments:
        assert isinstance(seg.text, str)
        box = seg.box
        assert isinstance(box, BoundingBox)
        # Positive extent (BoundingBox invariants guarantee > 0, assert anyway).
        assert box.width > 0 and box.height > 0
        # In-bounds against the image dimensions (top-left origin).
        assert box.x >= 0 and box.y >= 0
        assert box.x + box.width <= width
        assert box.y + box.height <= height
        # Confidence in [0, 1].
        assert 0.0 <= seg.confidence <= 1.0


def test_default_engine_uses_paddle_backend_when_available(backend: PaddleOCRBackend):
    """The default OCREngine resolves to an available PaddleOCRBackend. (Req 2.5)"""
    engine = OCREngine()  # no backend injected -> lazily resolves the default
    assert engine.backend is None  # not yet resolved

    # Trigger backend resolution by extracting on a simple generated image.
    image = make_latin_sample()
    engine.extract(image)

    assert isinstance(engine.backend, PaddleOCRBackend)
    assert engine.backend.available is True


def test_latin_extraction(backend: PaddleOCRBackend):
    """Latin sample yields >=1 segment with valid boxes and alphanumeric text.

    Requirements 2.1, 2.4, 6.1.
    """
    image = make_latin_sample("Invoice 2024")
    height, width = image.shape[0], image.shape[1]

    engine = OCREngine(backend)
    segments = engine.extract(image)

    _assert_valid_segments(segments, width, height)

    # Lenient content assertion: OCR may misread, but at least one alphanumeric
    # character should be recognized across the segments.
    combined = "".join(seg.text for seg in segments)
    assert combined.strip() != "", "expected non-empty recognized text"
    assert any(ch.isalnum() for ch in combined), (
        f"expected at least one alphanumeric character, got {combined!r}"
    )


def test_thai_extraction(backend: PaddleOCRBackend):
    """Thai sample yields >=1 segment with valid boxes; expect a Thai codepoint.

    Skipped when no Thai-capable font is available on the system. Requirement 2.3.
    """
    image = make_thai_sample()
    if image is None:
        pytest.skip("no Thai-capable font available")

    height, width = image.shape[0], image.shape[1]

    engine = OCREngine(backend)
    segments = engine.extract(image)

    _assert_valid_segments(segments, width, height)

    combined = "".join(seg.text for seg in segments)
    assert combined.strip() != "", "expected non-empty recognized text"
    # Ideally at least one Thai codepoint is recognized. Keep robust to OCR
    # error: assert Thai presence, which is the crux of Requirement 2.3.
    assert _has_thai(combined), (
        f"expected at least one Thai codepoint in recognized text, got {combined!r}"
    )


def test_health_reports_ocr_backend_available(backend: PaddleOCRBackend):
    """/api/health reports ocr_backend_available == True with a loaded backend.

    The default pipeline resolves PaddleOCR lazily, so ``engine.backend`` is
    ``None`` until first use and health would report False. We therefore inject a
    pipeline whose OCREngine already holds the loaded PaddleOCRBackend so health
    deterministically reports True (Requirement 2.5 surfacing). No production
    code is modified. Requirement 2.5.
    """
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from backend_service.app import create_app

    pipeline = GuardrailPipeline(
        preprocessor=Preprocessor(),
        ocr_engine=OCREngine(backend),  # already-resolved, available backend
        detector=Detector(),
        redactor=Redactor(),
    )

    client = fastapi_testclient.TestClient(create_app(pipeline))
    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["ocr_backend_available"] is True
