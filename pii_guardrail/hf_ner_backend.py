"""Optional HuggingFace-hosted Thai NER classifier (remote inference).

Wraps the HuggingFace Inference router so the
:class:`~pii_guardrail.detector.Detector` can consult the
``pythainlp/thainer-corpus-v2-base-model`` transformer NER model WITHOUT
installing PyTorch/transformers locally. The model runs on HuggingFace's
servers; this process only makes an HTTP call, keeping the local footprint tiny
(important on memory-constrained machines where a local transformer is too
heavy).

Workflow position: ``OCR -> NER -> redactor``. Each recognized OCR
:class:`~pii_guardrail.models.TextSegment` text is classified here; the Detector
turns any sensitive segment into a region using the segment's OCR box, and the
Redactor blacks out that box. This module only decides categories from text; it
never handles pixel coordinates.

Contract (mirrors :mod:`pii_guardrail.ner` / the other optional layers):
    - Dependency-free at import time: the HTTP call uses the standard library
      (:mod:`urllib.request`), so ``import pii_guardrail`` needs nothing extra.
    - ADVISORY: the Detector unions these categories with the deterministic
      pattern results and keeps patterns as the source of truth.
    - Fails safe: a missing token, a network error, a non-200 status (including
      the 503 "model loading" warm-up), or an unparseable body yields an empty
      set (or :class:`~pii_guardrail.errors.ClassifierUnavailableError`), so the
      Detector falls back to pattern-only classification.

Privacy note: segment text is sent to HuggingFace (a third party). This is fine
for testing, but for production PII a self-hosted model keeps data in-house.

Configuration (environment variables):
    - ``HF_TOKEN`` (or ``HUGGINGFACE_TOKEN``) -- HF access token. When absent,
      the classifier reports ``available=False``.
    - ``PII_GUARDRAIL_HF_NER_MODEL`` -- model id, default
      ``pythainlp/thainer-corpus-v2-base-model``.

Requirements: 5.1, 5.2, 5.3 (a concrete, optional ``ClassifierModel``).
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING

from pii_guardrail.errors import ClassifierUnavailableError
from pii_guardrail.models import SensitiveCategory

if TYPE_CHECKING:
    from pii_guardrail.models import TextSegment

__all__ = ["HFNerClassifier"]

logger = logging.getLogger(__name__)

# Separator joining segment texts into one page-level string for whole-page NER.
# A space avoids the newline-triggered instabilities seen on the HF serverless
# endpoint while still separating tokens; offsets still map back to segments.
_SEGMENT_SEPARATOR = " "

# The HF serverless endpoint for a community model can return transient 5xx /
# "model loading" errors, or a 503 during warm-up. Retry a few times with a
# short backoff before giving up (and falling back to patterns).
_MAX_RETRIES = 3
_RETRY_BACKOFF_SECONDS = 2.0
_RETRYABLE_STATUS = frozenset({500, 502, 503, 504, 429})


# --- Configuration defaults --------------------------------------------------

_DEFAULT_MODEL = "pythainlp/thainer-corpus-v2-base-model"
_ROUTER_BASE = "https://router.huggingface.co/hf-inference/models"
_TOKEN_ENVS = ("HF_TOKEN", "HUGGINGFACE_TOKEN")
_MODEL_ENV = "PII_GUARDRAIL_HF_NER_MODEL"

# Minimum entity score to accept. HF NER scores are in [0, 1]; 0.5 keeps recall
# high while dropping the weakest guesses. Over-redaction is preferable to
# leaking PII, so this stays moderate.
_SCORE_THRESHOLD = 0.5

# HTTP timeout (seconds). The FIRST request may be slow while HF warms the model
# (we pass wait_for_model=true); subsequent calls are fast.
_TIMEOUT_SECONDS = 90.0

#: Map the model's ``entity_group`` (upper-cased) to our
#: :class:`SensitiveCategory`. Shares the same mapping intent as the local
#: PyThaiNLP NER. Unmapped groups (TIME, LAW, ...) are ignored.
_ENTITY_GROUP_TO_CATEGORY: dict[str, SensitiveCategory] = {
    "PERSON": SensitiveCategory.PERSON_NAME,
    "ORGANIZATION": SensitiveCategory.ORGANIZATION_NAME,
    "ORG": SensitiveCategory.ORGANIZATION_NAME,
    "LOCATION": SensitiveCategory.THAI_ADDRESS,
    "LOC": SensitiveCategory.THAI_ADDRESS,
    "EMAIL": SensitiveCategory.EMAIL,
    "PHONE": SensitiveCategory.PHONE_NUMBER,
    "URL": SensitiveCategory.URL,
    "ZIP": SensitiveCategory.THAI_ADDRESS,
    "DATE": SensitiveCategory.DATE_OF_BIRTH,
    "TIME": SensitiveCategory.DATE_OF_BIRTH,
    "MONEY": SensitiveCategory.MONEY_AMOUNT,
}


class HFNerClassifier:
    """A :class:`~pii_guardrail.classifier.ClassifierModel` backed by HF Inference.

    Structurally satisfies the protocol (``available`` + ``classify``) by
    calling the HuggingFace Inference router over HTTP. Construct once and reuse.
    """

    def __init__(
        self,
        token: str | None = None,
        model: str | None = None,
    ) -> None:
        """Configure from explicit args or environment variables (no network I/O).

        Args:
            token: HF access token. Defaults to ``HF_TOKEN`` /
                ``HUGGINGFACE_TOKEN``. When neither is set, :attr:`available` is
                ``False``.
            model: Model id. Defaults to ``PII_GUARDRAIL_HF_NER_MODEL`` or
                ``pythainlp/thainer-corpus-v2-base-model``.

        Readiness is based on having a token configured; actual reachability is
        exercised lazily on the first :meth:`classify` call and degrades
        gracefully on failure.
        """
        env_token = next(
            (os.environ[name] for name in _TOKEN_ENVS if os.environ.get(name)),
            None,
        )
        self._token = token if token is not None else env_token
        self._model = model or os.environ.get(_MODEL_ENV) or _DEFAULT_MODEL
        self._url = f"{_ROUTER_BASE}/{self._model}"

    @property
    def available(self) -> bool:
        """``True`` when an HF token is configured (network probed at call time)."""
        return bool(self._token)

    def classify(self, text: str) -> set[SensitiveCategory]:
        """Return the sensitive categories the HF NER model finds in ``text``.

        Args:
            text: The recognized OCR text segment to classify.

        Returns:
            A ``set[SensitiveCategory]`` for the entity groups detected above the
            score threshold (possibly empty). The Detector unions this with the
            pattern-based results.

        Raises:
            ClassifierUnavailableError: If unconfigured, or the HTTP call fails
                (network error, non-200, warm-up timeout). The Detector catches
                this to fall back to pattern-based classification.
        """
        if not self.available:
            raise ClassifierUnavailableError("HF NER classifier is not configured.")

        # Empty/whitespace text has no entities; skip the network call.
        if not text or not text.strip():
            return set()

        try:
            entities = self._call_inference(text)
        except ClassifierUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001 - any transport failure -> fallback
            raise ClassifierUnavailableError(
                "HF NER inference request failed."
            ) from exc

        return self._categories_from_entities(entities)

    def classify_segments(
        self, segments: "list[TextSegment]"
    ) -> dict[int, set[SensitiveCategory]]:
        """Whole-page classification: one NER call for all segments (approach B).

        This is the WHOLE-PAGE hook the Detector prefers. Instead of classifying
        each short OCR segment in isolation -- where a transformer NER often
        misses entities for lack of context (e.g. a bare person name) -- it joins
        every segment's text into one page-level string, sends a SINGLE request,
        and maps each detected entity's character span back to the segment(s) it
        overlaps.

        Overlap handling: an entity that straddles a segment boundary (e.g. a
        name OCR split across two boxes) flags EVERY overlapping segment, so no
        part of the entity is left un-redacted (over-redaction is the safe
        choice for a PII guardrail).

        Args:
            segments: The ordered OCR segments (their ``.text`` is used).

        Returns:
            ``{segment_index: set[SensitiveCategory]}`` for segments that overlap
            a detected entity. Segments with no entity simply do not appear.

        Raises:
            ClassifierUnavailableError: If unconfigured or the HTTP call fails,
                so the Detector falls back to pattern-only classification.
        """
        if not self.available:
            raise ClassifierUnavailableError("HF NER classifier is not configured.")

        texts = [
            (seg.text if seg is not None and isinstance(seg.text, str) else "")
            for seg in segments
        ]
        if not any(t.strip() for t in texts):
            return {}

        # Build the joined page text and record each segment's [start, end) span
        # within it, so entity offsets can be mapped back to segment indices.
        spans: list[tuple[int, int]] = []
        cursor = 0
        parts: list[str] = []
        for i, text in enumerate(texts):
            start = cursor
            spans.append((start, start + len(text)))
            parts.append(text)
            cursor += len(text)
            if i < len(texts) - 1:
                cursor += len(_SEGMENT_SEPARATOR)
        page_text = _SEGMENT_SEPARATOR.join(parts)

        try:
            entities = self._call_inference(page_text)
        except ClassifierUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001 - any transport failure -> fallback
            raise ClassifierUnavailableError(
                "HF NER inference request failed."
            ) from exc

        return self._map_entities_to_segments(entities, spans)

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _map_entities_to_segments(
        entities: list, spans: list[tuple[int, int]]
    ) -> dict[int, set[SensitiveCategory]]:
        """Map entities (with char ``start``/``end``) to overlapping segments.

        For each entity above the score threshold, every segment whose span
        overlaps the entity's ``[start, end)`` is tagged with the mapped
        category. Entities missing usable offsets are skipped.
        """
        result: dict[int, set[SensitiveCategory]] = {}
        if not isinstance(entities, list):
            return result

        for item in entities:
            if not isinstance(item, dict):
                continue
            group = item.get("entity_group") or item.get("entity")
            score = item.get("score", 1.0)
            if not isinstance(group, str):
                continue
            try:
                if float(score) < _SCORE_THRESHOLD:
                    continue
            except (TypeError, ValueError):
                continue
            category = _ENTITY_GROUP_TO_CATEGORY.get(group.upper())
            if category is None:
                continue

            e_start = item.get("start")
            e_end = item.get("end")
            if not isinstance(e_start, int) or not isinstance(e_end, int):
                continue
            if e_end <= e_start:
                continue

            # Half-open overlap test: seg [s, e) overlaps entity [e_start, e_end)
            # iff s < e_end and e_start < e.
            for index, (s, e) in enumerate(spans):
                if s < e_end and e_start < e:
                    result.setdefault(index, set()).add(category)
        return result

    def _call_inference(self, text: str) -> list:
        """POST the text to the HF router and return the parsed entity list.

        Sends a UTF-8 encoded JSON body (Thai text must not be mangled) with
        ``wait_for_model`` so the first call waits for warm-up instead of
        erroring. Raises on non-200 so the caller degrades gracefully.
        """
        payload = {
            "inputs": text,
            "options": {"wait_for_model": True},
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        last_error: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            request = urllib.request.Request(  # noqa: S310 - fixed HTTPS router URL
                self._url,
                data=data,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(  # noqa: S310
                    request, timeout=_TIMEOUT_SECONDS
                ) as resp:
                    body = resp.read().decode("utf-8", errors="replace")
                parsed = json.loads(body)
                # A successful token-classification response is a list of entity
                # dicts. A dict body usually wraps a warning/error -> no entities.
                return parsed if isinstance(parsed, list) else []
            except urllib.error.HTTPError as exc:
                # Read the server's error body for logging (may be a warm-up or
                # a transient 5xx from the HF serverless endpoint).
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:200]
                except Exception:  # noqa: BLE001
                    detail = ""
                last_error = exc
                logger.warning(
                    "HF NER request failed (attempt %d/%d): HTTP %s %s",
                    attempt,
                    _MAX_RETRIES,
                    exc.code,
                    detail,
                )
                # Only retry transient statuses; a 4xx (auth/bad request) is
                # permanent, so stop immediately.
                if exc.code not in _RETRYABLE_STATUS:
                    break
            except Exception as exc:  # noqa: BLE001 - network/parse error
                last_error = exc
                logger.warning(
                    "HF NER request error (attempt %d/%d): %s: %s",
                    attempt,
                    _MAX_RETRIES,
                    type(exc).__name__,
                    exc,
                )

            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_BACKOFF_SECONDS * attempt)

        # Every attempt failed: surface the last error so the caller degrades to
        # pattern-only and the reason is visible in the logs above.
        raise ClassifierUnavailableError(
            f"HF NER inference failed after {_MAX_RETRIES} attempts."
        ) from last_error

    def _categories_from_entities(self, entities: list) -> set[SensitiveCategory]:
        """Map HF entity dicts to a set of sensitive categories.

        ``entities`` is a list of dicts with ``entity_group`` (str) and
        ``score`` (float). Entities below the score threshold are dropped; the
        remaining groups are upper-cased and mapped to a
        :class:`SensitiveCategory` when one exists.
        """
        categories: set[SensitiveCategory] = set()
        if not isinstance(entities, list):
            return categories

        for item in entities:
            if not isinstance(item, dict):
                continue
            group = item.get("entity_group") or item.get("entity")
            score = item.get("score", 1.0)
            if not isinstance(group, str):
                continue
            try:
                if float(score) < _SCORE_THRESHOLD:
                    continue
            except (TypeError, ValueError):
                continue
            category = _ENTITY_GROUP_TO_CATEGORY.get(group.upper())
            if category is not None:
                categories.add(category)
        return categories
