"""FastAPI application exposing the guardrail over HTTP.

This module is the web adapter's endpoint layer. FastAPI is imported HERE (not
in :mod:`backend_service.__init__`) so the package's low-level building blocks
(:mod:`backend_service.validation`) stay importable without the ``web`` extra.

Responsibilities (task 12.4):
    * Assemble a default :class:`~pii_guardrail.pipeline.GuardrailPipeline`
      (Preprocessor + OCREngine + Detector + Redactor). The default OCREngine
      uses PaddleOCR lazily; it may raise ``OCRProcessingError`` at OCR time,
      which is fine and handled by the error mapping in task 12.5.
    * Provide a ``create_app(pipeline=None)`` factory so tests can inject a
      pipeline wired with a fake OCR backend, plus a module-level default
      ``app``.
    * ``POST /api/redact``: parse the multipart upload, run edge validation via
      :func:`backend_service.validation.validate_and_load`, invoke
      ``pipeline.process``, and serialize the :class:`DetectionResult`, the
      base64-encoded redacted image, ``quality_sufficient``, and ``warnings``
      per the Web API Contract in design.md.

This module also maps the core error hierarchy to HTTP responses and exposes
``GET /api/health`` (task 12.5):

    * ``UnsupportedFormatError`` -> 415 ``UNSUPPORTED_FORMAT`` (detail identifies
      the received format, Requirement 1.3).
    * ``FileTooLargeError``      -> 413 ``FILE_TOO_LARGE`` (detail identifies the
      size limit, Requirement 1.5).
    * ``InvalidImageError``      -> 400 ``INVALID_IMAGE`` (Requirement 2.8).
    * ``OCRProcessingError``     -> 422 ``OCR_FAILED`` (Requirement 2.9).
    * any other ``GuardrailError`` / unexpected ``Exception`` -> 500
      ``INTERNAL_ERROR`` (Requirement 9.6).

Requirements: 1.3, 1.5, 2.8, 2.9, 7.1, 7.2, 7.3, 7.4, 8.3, 9.1, 9.6.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import TYPE_CHECKING

# Load environment variables from a local .env file (repo root) at import time,
# BEFORE any env var is read, so keys can live in .env instead of the shell.
import mimetypes

# Ensure .onnx and .wasm static files are served with correct MIME types
mimetypes.add_type("application/octet-stream", ".onnx")
mimetypes.add_type("application/wasm", ".wasm")

# Optional: if python-dotenv is not installed the app still works with real env
# vars. Existing environment values take precedence over the file (override=False).
try:
    from dotenv import load_dotenv

    _ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=_ENV_PATH, override=False)
except Exception:  # noqa: BLE001 - .env loading is a convenience, never fatal
    pass

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend_service.validation import validate_and_load
from pii_guardrail.detector import Detector
from pii_guardrail.errors import (
    FileTooLargeError,
    GuardrailError,
    InvalidImageError,
    OCRProcessingError,
    UnsupportedFormatError,
)
from pii_guardrail.ocr import OCREngine
from pii_guardrail.pipeline import GuardrailPipeline
from pii_guardrail.preprocessor import Preprocessor
from pii_guardrail.redactor import Redactor

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pii_guardrail.models import DetectionResult, SensitiveRegion, TextSegment

__all__ = ["build_default_pipeline", "create_app", "app"]

#: Directory holding the browser frontend (index.html, app.js, styles.css).
#: ``app.py`` lives in ``backend_service/`` so ``parent.parent`` is the repo
#: root; the frontend is served from ``<repo root>/web_interface`` on the SAME
#: ORIGIN as the API, which removes the need for CORS.
_WEB_DIR = Path(__file__).resolve().parent.parent / "web_interface"


def build_default_pipeline() -> GuardrailPipeline:
    """Assemble the default :class:`GuardrailPipeline`.

    Wires the four core components with their production defaults:

    * OCR: :func:`_build_ocr_engine` selects the backend. By default it uses the
      remote OCR.space API (fast on CPU-only machines, ~1-2s/page), falling back
      to the local PaddleOCR backend when ``PII_GUARDRAIL_OCR_BACKEND=paddle``.
    * Detection: pattern-based classification unioned with the remote
      HuggingFace Thai NER classifier (see
      :func:`_build_detector_with_optional_classifiers`).

    Both remote layers are optional and degrade gracefully: a failure at OCR or
    NER surfaces as an OCR error or a pattern-only fallback respectively, so a
    missing key or network hiccup never silently corrupts a result. Importing
    this module requires neither PaddleOCR nor any remote credential.

    Returns:
        A ready-to-use :class:`GuardrailPipeline`.
    """
    # Cloud OCR (the default) scales and orients internally, so the preprocessor
    # runs in passthrough mode: denoise only, NO upscale/deskew, no quality gate.
    # That keeps OCR boxes in the original image space (aligned redaction) and
    # lets the cloud engine handle low-resolution input. The local PaddleOCR path
    # keeps the full preprocessor (upscale/deskew/quality gate).
    use_cloud_ocr = (
        os.environ.get(_OCR_BACKEND_ENV) or "ocrspace"
    ).strip().lower() != "paddle"

    return GuardrailPipeline(
        preprocessor=Preprocessor(passthrough=use_cloud_ocr),
        ocr_engine=_build_ocr_engine(),
        detector=_build_detector_with_optional_classifiers(),
        redactor=Redactor(),
    )


#: Selects the OCR backend. ``ocrspace`` (default) uses the remote OCR.space API;
#: ``paddle`` uses the local PaddleOCR backend (slow on CPU, no network needed).
_OCR_BACKEND_ENV = "PII_GUARDRAIL_OCR_BACKEND"


def _build_ocr_engine() -> OCREngine:
    """Build the OCR engine, choosing the backend via ``PII_GUARDRAIL_OCR_BACKEND``.

    Default (``ocrspace`` or unset): the remote OCR.space backend -- fast on
    CPU-only machines and no local model to load. Set the env var to ``paddle``
    to use the local PaddleOCR backend instead (e.g. offline / privacy).

    Any failure constructing the OCR.space backend degrades to the lazily-loaded
    default (PaddleOCR), so OCR is never left unconfigured.
    """
    choice = (os.environ.get(_OCR_BACKEND_ENV) or "ocrspace").strip().lower()
    if choice == "paddle":
        # Explicit local backend: OCREngine() lazily loads PaddleOCR on first use.
        return OCREngine()
    try:
        from pii_guardrail.ocrspace_backend import OCRSpaceBackend

        return OCREngine(backend=OCRSpaceBackend())
    except Exception:  # noqa: BLE001 - fall back to the lazy local default
        return OCREngine()


def _build_detector_with_optional_classifiers() -> Detector:
    """Build a Detector wired with a composite of optional advisory classifiers.

    Three complementary layers are combined in a
    :class:`~pii_guardrail.composite.CompositeClassifier`, each optional and
    degrading gracefully:

    * :class:`~pii_guardrail.litellm_backend.LiteLLMClassifier` -- a remote LLM
      (via a LiteLLM OpenAI-compatible proxy) that reasons over the WHOLE page
      at once to catch names / organizations / addresses, robust to OCR errors.
    * :class:`~pii_guardrail.presidio_pattern_backend.PresidioPatternClassifier`
      -- Presidio's regex recognizers (NO NER) for internationally-structured
      PII: credit cards, IP addresses, IBANs, crypto wallets, email/phone/URL.
    * :class:`~pii_guardrail.secrets_backend.SecretsClassifier` -- detect-secrets
      (high-precision plugins only) for credentials: AWS/GitHub/GitLab keys,
      JWTs, private keys, etc.

    The composite exposes the whole-page ``classify_segments`` hook: the LLM is
    called once with full-page context while the per-segment pattern layers run
    on each segment, all merged by index. The deterministic patterns in
    :func:`~pii_guardrail.detector.classify_segment` remain the source of truth
    and are unioned with every layer's results. When no layer is available the
    Detector transparently falls back to pattern-only (Requirement 5.3). Any
    unexpected error degrades to a plain pattern-only Detector, so the optional
    layers never break the pipeline.
    """
    layers: list[object] = []

    # Whole-page contextual classifier (names/orgs/addresses). Optional.
    try:
        from pii_guardrail.litellm_backend import LiteLLMClassifier

        layers.append(LiteLLMClassifier())
    except Exception:  # noqa: BLE001 - never let one optional layer break the rest
        pass

    # Pattern-only Presidio (credit card / IP / IBAN / crypto / email / phone).
    try:
        from pii_guardrail.presidio_pattern_backend import PresidioPatternClassifier

        layers.append(PresidioPatternClassifier())
    except Exception:  # noqa: BLE001
        pass

    # detect-secrets (AWS/GitHub/JWT/private-key credentials).
    try:
        from pii_guardrail.secrets_backend import SecretsClassifier

        layers.append(SecretsClassifier())
    except Exception:  # noqa: BLE001
        pass

    if not layers:
        return Detector()

    try:
        from pii_guardrail.composite import CompositeClassifier

        return Detector(
            classifier=CompositeClassifier(layers), use_classifier=True
        )
    except Exception:  # noqa: BLE001 - degrade to pattern-only on any failure
        return Detector()


def _encode_png_base64(image: "NDArray") -> str:
    """Encode a BGR image array to a base64-encoded PNG string.

    Uses OpenCV's ``imencode`` (Requirement 8.3: the redacted image must be
    encodable) and base64-encodes the resulting bytes for JSON transport.

    Args:
        image: The image to encode (2-D grayscale or 3-D BGR ``uint8`` array).

    Returns:
        The base64-encoded PNG bytes as an ASCII string.

    Raises:
        RuntimeError: When OpenCV fails to encode the image.
    """
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Failed to encode redacted image as PNG.")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def _serialize_region(region: "SensitiveRegion") -> dict:
    """Serialize one :class:`SensitiveRegion` to the JSON contract shape.

    Categories are emitted as a sorted list of their string VALUES (the enum is
    a ``str`` Enum) so the output is deterministic and JSON-serializable.
    """
    box = region.box
    return {
        "box": {
            "x": int(box.x),
            "y": int(box.y),
            "width": int(box.width),
            "height": int(box.height),
        },
        "categories": sorted(category.value for category in region.categories),
        "confidence": float(region.confidence),
    }


def _serialize_detection_result(result: "DetectionResult") -> dict:
    """Serialize a :class:`DetectionResult` to the JSON contract shape."""
    return {
        "image_width": int(result.image_width),
        "image_height": int(result.image_height),
        "count": int(result.count),
        "regions": [_serialize_region(region) for region in result.regions],
    }


def _serialize_ocr_segment(
    segment: "TextSegment",
    categories: "set | None" = None,
    sources: "dict | None" = None,
) -> dict:
    """Serialize one raw OCR :class:`TextSegment` for the debug view.

    Emits the recognized text, its bounding box, OCR confidence, the merged
    categories, and a per-SOURCE breakdown. ``categories`` is the union across
    all layers (empty when non-sensitive); ``sources`` maps each layer label
    ("pattern", "llm", "presidio", "detect-secrets") to the categories THAT
    layer assigned. Together with the ``redacted`` flag this lets the frontend
    debug panel show WHY each segment was redacted and WHICH layer flagged it.
    """
    box = segment.box
    category_values = (
        sorted(c.value for c in categories) if categories else []
    )
    source_values = {}
    if sources:
        for src, cats in sources.items():
            if cats:
                source_values[src] = sorted(c.value for c in cats)
    return {
        "text": segment.text,
        "box": {
            "x": int(box.x),
            "y": int(box.y),
            "width": int(box.width),
            "height": int(box.height),
        },
        "confidence": float(segment.confidence),
        "categories": category_values,
        "redacted": bool(category_values),
        "sources": source_values,
    }


def _error_response(
    status_code: int,
    code: str,
    message: str,
    detail: str | None = None,
) -> JSONResponse:
    """Build the canonical error JSON response per the Web API Contract.

    The body shape is ``{"error": {"code", "message", "detail"}}`` (design.md),
    with ``detail`` omitted when ``None`` so responses stay tidy.
    """
    error: dict = {"code": code, "message": message}
    if detail is not None:
        error["detail"] = detail
    return JSONResponse(status_code=status_code, content={"error": error})


def _resolve_ocr_backend_available(pipeline: GuardrailPipeline) -> bool:
    """Report whether the pipeline's OCR backend is loaded and ready.

    Resolved defensively so ``/api/health`` never fails: if the OCR engine has
    no backend, cannot construct/resolve one (e.g. PaddleOCR is absent), or
    raises while reporting availability, this returns ``False`` rather than
    propagating. This intentionally does NOT force-load PaddleOCR in a way that
    raises -- it only inspects an already-configured backend.
    """
    try:
        engine = getattr(pipeline, "_ocr_engine", None)
        if engine is None:
            return False
        backend = engine.backend
        if backend is None:
            return False
        return bool(backend.available)
    except Exception:
        return False


def create_app(pipeline: GuardrailPipeline | None = None) -> FastAPI:
    """Build the FastAPI application.

    Args:
        pipeline: The :class:`GuardrailPipeline` to serve requests with. When
            ``None`` (the default), :func:`build_default_pipeline` is used.
            Tests inject a pipeline wired with a fake OCR backend so no real
            PaddleOCR model is required.

    Returns:
        A configured :class:`fastapi.FastAPI` instance.
    """
    active_pipeline = pipeline if pipeline is not None else build_default_pipeline()
    application = FastAPI(title="Thai Image PII Guardrail")

    # -- Error -> HTTP mapping (task 12.5) ---------------------------------
    #
    # Handlers are registered most-specific-first. FastAPI/Starlette resolves
    # an exception to the handler registered for its most specific type, so
    # ``UnsupportedFormatError`` (a subclass of ``InvalidImageError``) maps to
    # 415, while a plain ``InvalidImageError`` maps to 400.

    @application.exception_handler(UnsupportedFormatError)
    async def _handle_unsupported_format(
        _request: Request, exc: UnsupportedFormatError
    ) -> JSONResponse:
        # Detail identifies the received format when known (Requirement 1.3).
        detail = (
            f"received: {exc.received_format}"
            if getattr(exc, "received_format", None)
            else "received: unknown"
        )
        return _error_response(
            415, "UNSUPPORTED_FORMAT", str(exc), detail=detail
        )

    @application.exception_handler(FileTooLargeError)
    async def _handle_file_too_large(
        _request: Request, exc: FileTooLargeError
    ) -> JSONResponse:
        # Message/detail identifies the size limit (Requirement 1.5).
        limit = getattr(exc, "size_limit_bytes", None)
        detail = f"size_limit_bytes: {limit}" if limit is not None else None
        return _error_response(
            413, "FILE_TOO_LARGE", str(exc), detail=detail
        )

    @application.exception_handler(InvalidImageError)
    async def _handle_invalid_image(
        _request: Request, exc: InvalidImageError
    ) -> JSONResponse:
        return _error_response(400, "INVALID_IMAGE", str(exc))

    @application.exception_handler(OCRProcessingError)
    async def _handle_ocr_failed(
        _request: Request, exc: OCRProcessingError
    ) -> JSONResponse:
        return _error_response(422, "OCR_FAILED", str(exc))

    @application.exception_handler(GuardrailError)
    async def _handle_guardrail_error(
        _request: Request, exc: GuardrailError
    ) -> JSONResponse:
        # Any guardrail error not otherwise mapped is an unexpected failure.
        return _error_response(500, "INTERNAL_ERROR", str(exc))

    @application.exception_handler(Exception)
    async def _handle_unexpected(
        _request: Request, exc: Exception
    ) -> JSONResponse:
        # Any other unexpected failure (Requirement 9.6). Avoid leaking internal
        # details; a generic message is returned.
        return _error_response(
            500, "INTERNAL_ERROR", "An unexpected internal error occurred."
        )

    @application.get("/api/health")
    async def health() -> dict:
        """Report readiness and PaddleOCR/OCR backend availability.

        Always returns 200 with ``ocr_backend_available`` resolved defensively:
        ``false`` when no OCR backend is loaded (e.g. PaddleOCR is absent),
        ``true`` when a loaded backend reports itself available (design.md,
        Requirement 2.5 surfacing).
        """
        return {
            "status": "ok",
            "ocr_backend_available": _resolve_ocr_backend_available(active_pipeline),
        }

    @application.post("/api/redact")
    async def redact(
        image: UploadFile = File(...),
        redact: bool = Form(True),
    ) -> dict:
        """Detect and (optionally) redact sensitive regions in an uploaded image.

        Reads the multipart upload, validates it at the transport edge, runs it
        through the pipeline, and returns the serialized detection result plus a
        base64-encoded PNG redacted image (Requirements 7.1-7.4, 8.3, 9.1).
        """
        data = await image.read()

        # Transport-edge validation + decode into a BGR array. May raise the
        # core's canonical errors; HTTP mapping is added in task 12.5.
        image_array = validate_and_load(data)

        result = active_pipeline.process(image_array, redact=redact)

        # Choose the image to return. It MUST be in the same coordinate space as
        # detection_result's boxes/dimensions, which are in the pipeline's
        # PROCESSED (possibly upscaled) space. Prefer the redacted image (already
        # that space). When it is None -- redact=False, insufficient quality, or
        # zero regions -- fall back to the pipeline's processed_image, NOT the
        # original upload: after an auto-upscale the original has different
        # dimensions than the boxes, so returning it would misalign any overlay
        # and report mismatched dimensions. Only if processed_image is somehow
        # unavailable do we fall back to the original input (Requirements 8.3,
        # 8.4).
        image_to_encode: "NDArray | None" = result.redacted_image
        if image_to_encode is None:
            image_to_encode = (
                result.processed_image
                if result.processed_image is not None
                else image_array
            )

        response: dict = {
            "detection_result": _serialize_detection_result(result.detection_result),
            "redacted_image": {
                "format": "png",
                "base64": _encode_png_base64(np.asarray(image_to_encode)),
            },
            "quality_sufficient": bool(result.quality_sufficient),
            "warnings": list(result.warnings),
            # Raw OCR segments (debug view): everything the OCR read, each with
            # the categories the classifier assigned (empty => not redacted).
            # Lets the frontend show what text was recognized AND why each
            # segment was or was not redacted. Categories are aligned by index
            # with result.segment_categories; missing entries default to none.
            "ocr_segments": [
                _serialize_ocr_segment(
                    segment,
                    result.segment_categories[i]
                    if i < len(result.segment_categories)
                    else None,
                    result.segment_sources[i]
                    if i < len(result.segment_sources)
                    else None,
                )
                for i, segment in enumerate(result.ocr_segments)
            ],
        }
        return response

    # -- Static frontend (same-origin serving) -----------------------------
    #
    # Mount the browser frontend at the ROOT path so opening "/" serves
    # index.html and the UI can POST to same-origin "/api/redact" without CORS.
    # This mount is done LAST -- after the /api/* routes and exception handlers
    # are registered -- because a mount at "/" catches every path. Starlette
    # matches routes in registration order, so the explicitly-registered
    # /api/health and /api/redact routes (declared above) take precedence over
    # this catch-all mount. ``html=True`` makes "/" resolve to index.html.
    # Guard on directory existence so create_app still works for API-only
    # deployments and tests where web_interface/ may be absent.
    if _WEB_DIR.is_dir():
        application.mount(
            "/", StaticFiles(directory=str(_WEB_DIR), html=True), name="web"
        )

    return application


#: Module-level default application for ``uvicorn backend_service.app:app``.
app = create_app()
