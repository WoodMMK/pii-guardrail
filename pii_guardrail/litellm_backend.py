"""Optional LiteLLM-backed WHOLE-PAGE PII classifier.

Wraps an OpenAI-compatible chat endpoint (a LiteLLM proxy) so the
:class:`~pii_guardrail.detector.Detector` can consult a language model that
reasons over an ENTIRE page of OCR text at once. Unlike the per-segment
classifiers (:mod:`pii_guardrail.ner`, :mod:`pii_guardrail.presidio_backend`),
this layer implements the optional ``classify_segments`` hook: it receives every
recognized :class:`~pii_guardrail.models.TextSegment` together, so the model can
use surrounding context (e.g. a bare name in a "Received from" row is a person
name) and is invoked a SINGLE time per image.

Contract / design (mirrors the other optional layers):
    - No heavy dependency at import time: the HTTP call uses the standard
      library (:mod:`urllib.request`), so ``import pii_guardrail`` needs nothing
      extra. The MODEL runs remotely on the LiteLLM host, so this process stays
      light -- important on memory-constrained machines.
    - The model is ADVISORY: the Detector unions its categories with the
      deterministic pattern results and always keeps patterns as the source of
      truth.
    - Fails safe: any misconfiguration, network error, non-200 response, or
      unparseable body results in an empty mapping (or
      :class:`~pii_guardrail.errors.ClassifierUnavailableError`), so the Detector
      falls back to the other layers rather than crashing.

Configuration (environment variables):
    - ``LITELLM_BASE_URL``   -- OpenAI-compatible base, default
      ``http://localhost:4000/v1``.
    - ``LITELLM_API_KEY``    -- bearer token for the proxy (required; when
      absent the classifier reports ``available=False``).
    - ``PII_GUARDRAIL_LLM_MODEL`` -- model name to call, default ``Model_A``.

Requirements: 5.1, 5.2, 5.3 (a concrete, optional context-aware classifier).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from pii_guardrail.errors import ClassifierUnavailableError
from pii_guardrail.models import SensitiveCategory, TextSegment

__all__ = ["LiteLLMClassifier"]


# --- Configuration defaults --------------------------------------------------

_DEFAULT_BASE_URL = "http://localhost:4000/v1"
# Model_B is a non-reasoning model: it answers the JSON directly (finish=stop),
# unlike reasoning models (Model_A/C, claude-A/C) that burn the token budget on
# a "thinking" preamble and return truncated/empty content for this task.
_DEFAULT_MODEL = "Model_B"
_BASE_URL_ENV = "LITELLM_BASE_URL"
_API_KEY_ENV = "LITELLM_API_KEY"
_MODEL_ENV = "PII_GUARDRAIL_LLM_MODEL"

# Reasoning-capable models (e.g. Model_A) spend tokens "thinking" before they
# emit the answer. A generous cap ensures the JSON answer is not truncated
# (which would surface as an empty/`null` message content, finish_reason=length).
_MAX_TOKENS = 1200

# Deterministic classification: temperature 0 so the same page yields the same
# labels run to run.
_TEMPERATURE = 0.0

# HTTP timeout (seconds) for a single whole-page call. Reasoning models are
# slower, so this is comfortably long; a timeout degrades to pattern-only.
_TIMEOUT_SECONDS = 120.0

# The set of category string values the model is allowed to emit. Kept in sync
# with :class:`SensitiveCategory` (that enum is the single source of truth); the
# prompt lists exactly these and results are filtered to them.
_ALLOWED_CATEGORY_VALUES: frozenset[str] = frozenset(c.value for c in SensitiveCategory)


def _build_system_prompt() -> str:
    """Return the system prompt describing the task and the strict output shape."""
    categories = ", ".join(sorted(_ALLOWED_CATEGORY_VALUES))
    return (
        "You are a PII detection engine for a redaction guardrail. You receive a "
        "numbered list of text segments extracted by OCR from a single document "
        "image (Thai and/or English). Decide, for EACH segment, whether it "
        "contains sensitive or personally identifiable information that should "
        "be redacted before the document is shared with an external AI service.\n\n"
        "Use the whole page as context (labels, table headers, adjacent rows) to "
        "judge each segment. Treat as sensitive: personal names, organizations "
        "that identify an individual, addresses, phone numbers, emails, national "
        "IDs, passport/driver/student/document/receipt numbers, bank accounts, "
        "credit/debit cards, dates of birth, job titles tied to a person, "
        "monetary amounts on personal documents, credentials/keys/secrets, "
        "usernames, and ages.\n\n"
        "Respond with ONLY a compact JSON object, no prose and no markdown "
        "fences. The object maps the segment index (as a string) to an array of "
        "category labels. OMIT segments that are not sensitive. Use ONLY these "
        f"category labels: {categories}.\n"
        'Example: {"0": ["person_name"], "3": ["thai_national_id"]}'
    )


class LiteLLMClassifier:
    """A whole-page classifier backed by an OpenAI-compatible LiteLLM endpoint.

    Structurally satisfies the ``ClassifierModel`` protocol (``available`` +
    ``classify``) AND exposes the optional whole-page ``classify_segments``
    hook the Detector prefers. Construct once and reuse.
    """

    #: Stable label identifying this layer in per-source debug breakdowns.
    source_name = "llm"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        """Configure the classifier from explicit args or environment variables.

        Args:
            base_url: OpenAI-compatible base URL. Defaults to ``LITELLM_BASE_URL``
                or ``http://localhost:4000/v1``.
            api_key: Bearer token. Defaults to ``LITELLM_API_KEY``. When neither
                is set, :attr:`available` is ``False``.
            model: Model name to call. Defaults to ``PII_GUARDRAIL_LLM_MODEL`` or
                ``Model_A``.

        Constructing this object performs NO network I/O; readiness is based on
        having an API key configured, and actual reachability is exercised
        lazily on the first classify call (a failure there degrades gracefully).
        """
        self._base_url = (
            base_url or os.environ.get(_BASE_URL_ENV) or _DEFAULT_BASE_URL
        ).rstrip("/")
        self._api_key = api_key if api_key is not None else os.environ.get(_API_KEY_ENV)
        self._model = model or os.environ.get(_MODEL_ENV) or _DEFAULT_MODEL
        self._system_prompt = _build_system_prompt()

    @property
    def available(self) -> bool:
        """``True`` when an API key is configured (a base URL always exists).

        Network reachability is NOT probed here (that would add latency and I/O
        at construction). A runtime failure during a call is handled by
        returning an empty result, so the Detector still falls back cleanly.
        """
        return bool(self._api_key)

    def classify(self, text: str) -> set[SensitiveCategory]:
        """Classify a single text segment (per-segment protocol compatibility).

        Delegates to :meth:`classify_segments` with a one-element list so the
        object also satisfies the plain ``ClassifierModel`` protocol. The
        Detector prefers :meth:`classify_segments` (whole page) when present.

        Raises:
            ClassifierUnavailableError: If the classifier is not configured.
        """
        if not self.available:
            raise ClassifierUnavailableError("LiteLLM classifier is not configured.")
        if not text or not text.strip():
            return set()
        return self._classify_texts([text]).get(0, set())

    def classify_segments(
        self, segments: list[TextSegment]
    ) -> dict[int, set[SensitiveCategory]]:
        """Classify ALL segments in one call, returning ``index -> categories``.

        Builds a numbered list of the segment texts, sends it to the LiteLLM
        chat endpoint with a strict-JSON system prompt, and parses the reply
        into a mapping of segment index to its sensitive categories. Segments
        the model omits (or maps to unknown labels) simply do not appear.

        Fails safe: returns an empty mapping on an empty input, and raises
        :class:`ClassifierUnavailableError` when unconfigured or when the HTTP
        call / parse fails, so the Detector records its fallback warning and
        relies on the deterministic patterns.
        """
        if not self.available:
            raise ClassifierUnavailableError("LiteLLM classifier is not configured.")

        texts = [
            (seg.text if seg is not None and isinstance(seg.text, str) else "")
            for seg in segments
        ]
        return self._classify_texts(texts)

    # -- internals ----------------------------------------------------------

    def _classify_texts(self, texts: list[str]) -> dict[int, set[SensitiveCategory]]:
        """Classify a list of raw text strings in one call (index -> categories).

        Shared core for :meth:`classify` and :meth:`classify_segments`. Skips the
        network call for all-empty input; raises
        :class:`ClassifierUnavailableError` on a transport/parse failure so the
        Detector falls back to patterns.
        """
        # Nothing to classify: skip the network call entirely.
        if not any(t.strip() for t in texts):
            return {}

        user_content = self._format_segments(texts)
        try:
            content = self._call_chat(user_content)
        except ClassifierUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001 - any transport failure -> fallback
            raise ClassifierUnavailableError(
                "LiteLLM classifier request failed."
            ) from exc

        return self._parse_response(content, num_segments=len(texts))

    @staticmethod
    def _format_segments(texts: list[str]) -> str:
        """Render the segments as a stable, numbered list for the prompt."""
        lines = [f"[{i}] {text}" for i, text in enumerate(texts)]
        return "Segments:\n" + "\n".join(lines)

    def _call_chat(self, user_content: str) -> str:
        """POST a chat-completion request and return the assistant message text.

        Raises on any non-200 status or transport error; the caller converts
        that into a graceful fallback.
        """
        url = f"{self._base_url}/chat/completions"
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": _MAX_TOKENS,
            "temperature": _TEMPERATURE,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(  # noqa: S310 - fixed https/http proxy URL
            url,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as resp:  # noqa: S310
            body = resp.read().decode("utf-8", errors="replace")

        parsed = json.loads(body)
        choices = parsed.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        return content if isinstance(content, str) else ""

    def _parse_response(
        self, content: str, *, num_segments: int
    ) -> dict[int, set[SensitiveCategory]]:
        """Parse the model's JSON reply into an ``index -> categories`` mapping.

        Tolerant of markdown code fences and leading/trailing prose: the first
        ``{...}`` JSON object in the text is extracted. Unknown category labels
        and out-of-range indices are dropped. Never raises -- an unparseable
        reply yields an empty mapping.
        """
        obj = self._extract_json_object(content)
        if not isinstance(obj, dict):
            return {}

        result: dict[int, set[SensitiveCategory]] = {}
        for key, value in obj.items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                continue
            if index < 0 or index >= num_segments:
                continue
            if not isinstance(value, (list, tuple, set)):
                continue
            cats: set[SensitiveCategory] = set()
            for label in value:
                if not isinstance(label, str):
                    continue
                normalized = label.strip().lower()
                if normalized in _ALLOWED_CATEGORY_VALUES:
                    cats.add(SensitiveCategory(normalized))
            if cats:
                result[index] = cats
        return result

    @staticmethod
    def _extract_json_object(content: str) -> object:
        """Best-effort extraction of the first JSON object from model output.

        Handles a bare JSON object, one wrapped in ```json fences, or one
        embedded in surrounding prose. Returns ``None`` when no object parses.
        """
        if not content or not isinstance(content, str):
            return None

        text = content.strip()

        # Fast path: the whole thing is JSON.
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass

        # Strip a leading ```json / ``` fence if present.
        if "```" in text:
            fenced = text.split("```")
            for chunk in fenced:
                candidate = chunk.strip()
                if candidate.lower().startswith("json"):
                    candidate = candidate[4:].strip()
                if candidate.startswith("{"):
                    try:
                        return json.loads(candidate)
                    except (json.JSONDecodeError, ValueError):
                        continue

        # Last resort: grab the substring from the first '{' to the last '}'.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            try:
                return json.loads(snippet)
            except (json.JSONDecodeError, ValueError):
                return None
        return None
