"""Unit test for the ``POST /api/redact`` success response shape.

Asserts that a successful redaction response carries every field the Web API
Contract promises: the serialized ``detection_result``, the base64-encoded
``redacted_image``, the ``warnings`` array, and the ``quality_sufficient`` flag.

This is an example-based unit test (no Hypothesis ``@given``). A pipeline wired
with a :class:`~tests.property.ocr_fakes.FakeOCRBackend` is injected via
``create_app`` so no PaddleOCR model is needed, and a passthrough preprocessor
forces ``quality_sufficient=True`` so a small in-memory PNG still runs the full
detect + redact flow and produces populated regions.

Validates: Requirements 9.1.
    9.1 -- a successful response includes the detection result and the
    (redacted) image, alongside the warnings and quality flag.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest

# FastAPI/TestClient are installed here; importorskip keeps collection robust
# if the web extra is ever absent.
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from backend_service.app import create_app  # noqa: E402
from pii_guardrail.detector import Detector  # noqa: E402
from pii_guardrail.models import BoundingBox, TextSegment  # noqa: E402
from pii_guardrail.ocr import OCREngine  # noqa: E402
from pii_guardrail.pipeline import GuardrailPipeline  # noqa: E402
from pii_guardrail.preprocessor import PreprocessResult, Preprocessor  # noqa: E402
from pii_guardrail.redactor import Redactor  # noqa: E402
from tests.property.ocr_fakes import FakeOCRBackend  # noqa: E402

pytestmark = pytest.mark.unit


class _PassthroughPreprocessor(Preprocessor):
    """Validate the image, then return it unchanged with quality forced True.

    The real Preprocessor marks small images as insufficient quality (its
    effective-DPI heuristic is well below threshold for a 64px-tall image),
    which would short-circuit the pipeline before detection. To keep this a
    fast, deterministic web-layer test we reuse the real (cheap) validation but
    skip the heavy denoise and force ``quality_sufficient=True``.
    """

    def preprocess(
        self, image: np.ndarray, is_scanned: bool | None = None
    ) -> PreprocessResult:
        self._validate(image)
        return PreprocessResult(
            image=image,
            residual_skew_deg=0.0,
            estimated_dpi=300.0,
            quality_sufficient=True,
        )


def _png_bytes(width: int = 64, height: int = 64) -> bytes:
    """A small valid white PNG whose bytes pass ``validate_and_load``."""
    image = Image.new("RGB", (width, height), (255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _build_injected_pipeline() -> GuardrailPipeline:
    """A pipeline with a fake OCR backend returning one sensitive segment.

    The email segment's box fits within the 64x64 upload, so the region is
    in-bounds and the detector classifies it as EMAIL via its patterns.
    """
    email_box = BoundingBox(x=4, y=8, width=40, height=12)
    segments = [
        TextSegment(text="user@example.com", box=email_box, confidence=0.95),
    ]
    return GuardrailPipeline(
        preprocessor=_PassthroughPreprocessor(),
        ocr_engine=OCREngine(FakeOCRBackend(segments=segments, available=True)),
        detector=Detector(),
        redactor=Redactor(),
    )


def test_redact_response_contains_detection_result_and_redacted_image() -> None:
    client = TestClient(create_app(pipeline=_build_injected_pipeline()))

    response = client.post(
        "/api/redact",
        files={"image": ("test.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()

    # -- detection_result present and shaped per the contract (Req 9.1) -----
    assert "detection_result" in body
    detection = body["detection_result"]
    for key in ("image_width", "image_height", "count", "regions"):
        assert key in detection
    assert isinstance(detection["regions"], list)
    assert detection["image_width"] == 64
    assert detection["image_height"] == 64

    # A sensitive segment was provided, so at least one region is detected and
    # the email region carries the EMAIL category.
    assert detection["count"] >= 1
    assert any("email" in region["categories"] for region in detection["regions"])

    # -- redacted_image present with format + non-empty base64 (Req 9.1) ----
    assert "redacted_image" in body
    redacted = body["redacted_image"]
    assert "format" in redacted
    assert "base64" in redacted
    assert isinstance(redacted["base64"], str)
    assert redacted["base64"] != ""

    # The base64 payload decodes to a valid PNG.
    decoded = base64.b64decode(redacted["base64"])
    with Image.open(io.BytesIO(decoded)) as decoded_image:
        assert decoded_image.format == "PNG"

    # -- warnings array present (Req 9.1) -----------------------------------
    assert "warnings" in body
    assert isinstance(body["warnings"], list)

    # -- quality_sufficient present and True here ---------------------------
    assert "quality_sufficient" in body
    assert body["quality_sufficient"] is True
