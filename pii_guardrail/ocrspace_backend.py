"""Optional cloud OCR backend using the OCR.space API (remote text extraction).

Implements the :class:`~pii_guardrail.ocr.OCRBackend` protocol by sending the
image to the OCR.space REST API instead of running a local PaddleOCR model. This
is dramatically faster on CPU-only machines (a full page in ~1-2s vs. ~60-90s
locally) and uses no local RAM/CPU for the model.

Workflow position: it is the OCR stage in ``OCR -> NER -> redactor``. It returns
LINE-LEVEL segments -- one :class:`~pii_guardrail.models.TextSegment` per
recognized line, whose box is the union of that line's word boxes -- because
whole-line text gives the downstream NER better context and line-level boxes
redact more safely than tight per-word boxes.

Contract (mirrors the other optional/remote backends):
    - Dependency-free at import time: the HTTP call uses the standard library
      (:mod:`urllib.request`) with a hand-built multipart body and OpenCV (a
      core dep) to PNG-encode the image, so ``import pii_guardrail`` needs
      nothing extra.
    - Fails closed: ``available`` is ``False`` without an API key, and
      :meth:`run` raises :class:`~pii_guardrail.errors.OCRProcessingError` on a
      transport error, a non-200 status, or an API-reported processing error, so
      :class:`~pii_guardrail.ocr.OCREngine` surfaces it as an OCR failure rather
      than a partial result.

Privacy note: the image is sent to OCR.space (a third party). Fine for testing;
for production PII a self-hosted OCR keeps data in-house.

Configuration (environment variables):
    - ``OCRSPACE_API_KEY`` -- API key. Defaults to the public demo key
      ``helloworld`` (heavily rate-limited; get a free key at ocr.space/ocrapi).
    - ``OCRSPACE_LANGUAGE`` -- OCR.space language code, default ``tha`` (Thai;
      the Thai model also reads Latin script).
    - ``OCRSPACE_URL`` -- API endpoint, default ``https://api.ocr.space/parse/image``.
"""

from __future__ import annotations

import json
import os
import urllib.request
import uuid

import cv2
from numpy.typing import NDArray

from pii_guardrail.errors import OCRProcessingError
from pii_guardrail.models import BoundingBox, TextSegment

__all__ = ["OCRSpaceBackend"]


_DEFAULT_URL = "https://api.ocr.space/parse/image"
_DEFAULT_LANGUAGE = "tha"
_DEMO_KEY = "helloworld"
_API_KEY_ENV = "OCRSPACE_API_KEY"
_LANGUAGE_ENV = "OCRSPACE_LANGUAGE"
_URL_ENV = "OCRSPACE_URL"

# OCR.space "OCR Engine 2" handles mixed Thai/Latin and returns word overlays.
_OCR_ENGINE = "2"

# Confidence: OCR.space does not return a per-line score, so we assign a fixed
# high confidence (the box/text are what downstream stages use). 0.99 keeps
# every recognized line above any reasonable confidence gate.
_FIXED_CONFIDENCE = 0.99

_TIMEOUT_SECONDS = 120.0


class OCRSpaceBackend:
    """An :class:`~pii_guardrail.ocr.OCRBackend` backed by the OCR.space API.

    Structurally satisfies the protocol (``run`` + ``available``). Construct once
    and reuse. Network I/O happens only in :meth:`run`.
    """

    def __init__(
        self,
        api_key: str | None = None,
        language: str | None = None,
        url: str | None = None,
    ) -> None:
        """Configure from explicit args or environment variables (no network I/O).

        Args:
            api_key: OCR.space API key. Defaults to ``OCRSPACE_API_KEY`` or the
                public demo key ``helloworld``.
            language: OCR.space language code. Defaults to ``OCRSPACE_LANGUAGE``
                or ``tha``.
            url: API endpoint. Defaults to ``OCRSPACE_URL`` or the public URL.
        """
        self._api_key = api_key or os.environ.get(_API_KEY_ENV) or _DEMO_KEY
        self._language = language or os.environ.get(_LANGUAGE_ENV) or _DEFAULT_LANGUAGE
        self._url = url or os.environ.get(_URL_ENV) or _DEFAULT_URL

    @property
    def available(self) -> bool:
        """``True`` when an API key is configured (a key always defaults to demo)."""
        return bool(self._api_key)

    def run(self, image: NDArray) -> list[TextSegment]:
        """Extract line-level text segments from ``image`` via OCR.space.

        Args:
            image: The (preprocessed) image as a NumPy array.

        Returns:
            One :class:`~pii_guardrail.models.TextSegment` per recognized line,
            with a line-level box (union of the line's word boxes). Returns
            ``[]`` when no text is found.

        Raises:
            OCRProcessingError: On encode failure, transport error, non-200
                status, or an API-reported processing error.
        """
        if not self.available:
            raise OCRProcessingError("OCR.space backend is not configured.")

        png_bytes = self._encode_png(image)
        try:
            body = self._post_image(png_bytes)
        except OCRProcessingError:
            raise
        except Exception as exc:  # noqa: BLE001 - any transport failure -> OCR error
            raise OCRProcessingError("OCR.space request failed.") from exc

        return self._parse_segments(body)

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _encode_png(image: NDArray) -> bytes:
        """PNG-encode the image for upload; raise OCRProcessingError on failure."""
        success, buffer = cv2.imencode(".png", image)
        if not success:
            raise OCRProcessingError("Failed to encode image for OCR.space upload.")
        return buffer.tobytes()

    def _post_image(self, png_bytes: bytes) -> dict:
        """POST the image as multipart/form-data and return the parsed JSON.

        Builds the multipart body by hand (stdlib only) so no ``requests`` /
        ``httpx`` dependency is added to the core.
        """
        boundary = f"----pii-guardrail-{uuid.uuid4().hex}"
        fields = {
            "apikey": self._api_key,
            "language": self._language,
            "isOverlayRequired": "true",
            "OCREngine": _OCR_ENGINE,
            "scale": "true",
        }

        parts: list[bytes] = []
        for name, value in fields.items():
            parts.append(f"--{boundary}\r\n".encode())
            parts.append(
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
            )
            parts.append(f"{value}\r\n".encode())
        # The image file part.
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(
            b'Content-Disposition: form-data; name="file"; filename="image.png"\r\n'
        )
        parts.append(b"Content-Type: image/png\r\n\r\n")
        parts.append(png_bytes)
        parts.append(f"\r\n--{boundary}--\r\n".encode())
        payload = b"".join(parts)

        request = urllib.request.Request(  # noqa: S310 - fixed HTTPS API URL
            self._url,
            data=payload,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
        )
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw)

    def _parse_segments(self, body: dict) -> list[TextSegment]:
        """Turn an OCR.space response into line-level TextSegments.

        Raises OCRProcessingError when the API reports a processing error;
        returns ``[]`` when parsing succeeds but no lines were found.
        """
        if not isinstance(body, dict):
            raise OCRProcessingError("OCR.space returned an unexpected response.")

        # The API signals errors via IsErroredOnProcessing / OCRExitCode.
        if body.get("IsErroredOnProcessing"):
            message = body.get("ErrorMessage")
            if isinstance(message, list):
                message = "; ".join(str(m) for m in message)
            raise OCRProcessingError(
                f"OCR.space processing error: {message or 'unknown error'}."
            )

        results = body.get("ParsedResults")
        if not isinstance(results, list):
            return []

        segments: list[TextSegment] = []
        for result in results:
            if not isinstance(result, dict):
                continue
            overlay = result.get("TextOverlay") or {}
            lines = overlay.get("Lines") if isinstance(overlay, dict) else None
            if not isinstance(lines, list):
                continue
            for line in lines:
                segment = self._line_to_segment(line)
                if segment is not None:
                    segments.append(segment)
        return segments

    @staticmethod
    def _line_to_segment(line: object) -> TextSegment | None:
        """Build one line-level TextSegment from an OCR.space ``Line`` object.

        The line box is the union of its words' ``Left/Top/Width/Height`` rects.
        Returns ``None`` for an empty/malformed line (no text or no usable box).
        """
        if not isinstance(line, dict):
            return None
        text = line.get("LineText")
        if not isinstance(text, str) or not text.strip():
            return None

        words = line.get("Words")
        if not isinstance(words, list) or not words:
            return None

        # Union of word rectangles -> line-level box.
        lefts: list[int] = []
        tops: list[int] = []
        rights: list[int] = []
        bottoms: list[int] = []
        for word in words:
            if not isinstance(word, dict):
                continue
            try:
                left = int(word["Left"])
                top = int(word["Top"])
                w = int(word["Width"])
                h = int(word["Height"])
            except (KeyError, TypeError, ValueError):
                continue
            lefts.append(left)
            tops.append(top)
            rights.append(left + w)
            bottoms.append(top + h)

        if not lefts:
            return None

        x = min(lefts)
        y = min(tops)
        width = max(rights) - x
        height = max(bottoms) - y
        # Guard against degenerate rects; BoundingBox requires positive extent.
        if width <= 0 or height <= 0:
            return None

        # Clamp origin to be non-negative (BoundingBox invariant); OCREngine
        # re-normalizes against the real image dimensions afterwards.
        x = max(x, 0)
        y = max(y, 0)
        return TextSegment(
            text=text,
            box=BoundingBox(x=x, y=y, width=width, height=height),
            confidence=_FIXED_CONFIDENCE,
        )
