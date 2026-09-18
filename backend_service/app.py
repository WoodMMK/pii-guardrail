"""FastAPI application exposing OCR inspection and web interface.

Serves the in-browser OCR web interface and provides endpoints for OCR health
and server-side text extraction.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING
import mimetypes

# Ensure .onnx and .wasm static files are served with correct MIME types
mimetypes.add_type("application/octet-stream", ".onnx")
mimetypes.add_type("application/wasm", ".wasm")

try:
    from dotenv import load_dotenv

    _ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=_ENV_PATH, override=False)
except Exception:  # noqa: BLE001
    pass

import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend_service.validation import validate_and_load
from pii_guardrail.errors import (
    FileTooLargeError,
    GuardrailError,
    InvalidImageError,
    OCRProcessingError,
    UnsupportedFormatError,
)
from pii_guardrail.ocr import OCREngine
from pii_guardrail.pipeline import OCRPipeline, GuardrailPipeline
from pii_guardrail.preprocessor import Preprocessor

if TYPE_CHECKING:
    from pii_guardrail.models import TextSegment

__all__ = ["build_default_pipeline", "create_app", "app"]

#: Directory holding the browser frontend (index.html, app.js, styles.css).
_WEB_DIR = Path(__file__).resolve().parent.parent / "web_interface"

#: Selects the OCR backend. ``ocrspace`` (default) uses remote OCR.space API;
#: ``paddle`` uses local PaddleOCR backend.
_OCR_BACKEND_ENV = "PII_GUARDRAIL_OCR_BACKEND"


def _build_ocr_engine() -> OCREngine:
    """Build the OCR engine, choosing backend via ``PII_GUARDRAIL_OCR_BACKEND``."""
    choice = (os.environ.get(_OCR_BACKEND_ENV) or "ocrspace").strip().lower()
    if choice == "paddle":
        return OCREngine()
    try:
        from pii_guardrail.ocrspace_backend import OCRSpaceBackend

        return OCREngine(backend=OCRSpaceBackend())
    except Exception:  # noqa: BLE001
        return OCREngine()


def build_default_pipeline() -> OCRPipeline:
    """Assemble default OCR pipeline with Preprocessor and OCREngine."""
    use_cloud_ocr = (
        os.environ.get(_OCR_BACKEND_ENV) or "ocrspace"
    ).strip().lower() != "paddle"

    return OCRPipeline(
        preprocessor=Preprocessor(passthrough=use_cloud_ocr),
        ocr_engine=_build_ocr_engine(),
    )


def _serialize_ocr_segment(segment: "TextSegment") -> dict:
    """Serialize one OCR TextSegment for API and debug inspection."""
    box = segment.box
    return {
        "text": segment.text,
        "box": {
            "x": int(box.x),
            "y": int(box.y),
            "width": int(box.width),
            "height": int(box.height),
        },
        "confidence": float(segment.confidence),
    }


def _error_response(
    status_code: int,
    code: str,
    message: str,
    detail: str | None = None,
) -> JSONResponse:
    """Build canonical error JSON response."""
    error: dict = {"code": code, "message": message}
    if detail is not None:
        error["detail"] = detail
    return JSONResponse(status_code=status_code, content={"error": error})


def _resolve_ocr_backend_available(pipeline: OCRPipeline) -> bool:
    """Report whether pipeline's OCR backend is loaded and ready."""
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


def create_app(pipeline: OCRPipeline | None = None) -> FastAPI:
    """Build the FastAPI application."""
    active_pipeline = pipeline if pipeline is not None else build_default_pipeline()
    application = FastAPI(title="Thai Document OCR Inspector")

    @application.middleware("http")
    async def _add_cross_origin_isolation_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
        return response

    @application.exception_handler(UnsupportedFormatError)
    async def _handle_unsupported_format(
        _request: Request, exc: UnsupportedFormatError
    ) -> JSONResponse:
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
        return _error_response(500, "INTERNAL_ERROR", str(exc))

    @application.exception_handler(Exception)
    async def _handle_unexpected(
        _request: Request, exc: Exception
    ) -> JSONResponse:
        return _error_response(
            500, "INTERNAL_ERROR", "An unexpected internal error occurred."
        )

    @application.get("/api/health")
    async def health() -> dict:
        """Report readiness and OCR backend availability."""
        return {
            "status": "ok",
            "ocr_backend_available": _resolve_ocr_backend_available(active_pipeline),
        }

    @application.post("/api/ocr")
    @application.post("/api/extract")
    @application.post("/api/redact")
    async def extract_ocr(
        image: UploadFile = File(...),
        redact: bool = Form(False),
    ) -> dict:
        """Extract text segments and bounding boxes from uploaded image."""
        data = await image.read()
        image_array = validate_and_load(data)
        result = active_pipeline.process(image_array)

        return {
            "quality_sufficient": bool(result.quality_sufficient),
            "warnings": list(result.warnings),
            "ocr_segments": [
                _serialize_ocr_segment(segment)
                for segment in result.ocr_segments
            ],
        }

    # Mount static web frontend at root
    if _WEB_DIR.is_dir():
        application.mount(
            "/", StaticFiles(directory=str(_WEB_DIR), html=True), name="web"
        )

    return application


app = create_app()
