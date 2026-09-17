"""Optional Presidio PATTERN-ONLY classifier (no NER / no spaCy model download).

Wraps Microsoft Presidio's predefined **regex/pattern recognizers** so the
:class:`~pii_guardrail.detector.Detector` can detect internationally-structured
PII that the hand-written Thai patterns don't cover -- credit-card numbers
(Luhn), IP addresses, IBANs, crypto wallet addresses, etc.

Why pattern-only:
    Presidio can also run an NLP (NER) engine for names/locations, but that
    pulls a heavy spaCy model (~500MB+ RAM). We DON'T need it here -- the LLM
    classifier handles names/organizations/addresses. So this backend builds the
    analyzer with a BLANK spaCy pipeline (tokenizer only, near-zero RAM) and
    uses ONLY the pattern recognizers. Measured footprint: ~110MB, no model
    download required.

Contract (mirrors the other optional backends):
    - Lazy import: ``import pii_guardrail`` works without Presidio installed.
    - ADVISORY: the Detector unions these categories with the pattern + LLM
      results; the deterministic patterns remain the source of truth.
    - Fails safe: any construction/analysis error -> ``available=False`` or an
      empty set, so the Detector falls back cleanly.

Requirements: 5.1, 5.2, 5.3 (a concrete, optional ``ClassifierModel``).
"""

from __future__ import annotations

from pii_guardrail.errors import ClassifierUnavailableError
from pii_guardrail.models import SensitiveCategory

__all__ = ["PresidioPatternClassifier"]


#: Map Presidio entity types (pattern-based ones) to our SensitiveCategory.
#: NER-only entity types (PERSON, LOCATION, NRP, DATE_TIME) are intentionally
#: OMITTED -- the LLM classifier owns those, and we run Presidio without NER.
_PRESIDIO_ENTITY_TO_CATEGORY: dict[str, SensitiveCategory] = {
    "CREDIT_CARD": SensitiveCategory.CREDIT_CARD,
    "IP_ADDRESS": SensitiveCategory.IP_ADDRESS,
    "IBAN_CODE": SensitiveCategory.IBAN,
    "CRYPTO": SensitiveCategory.CRYPTO_WALLET,
    "EMAIL_ADDRESS": SensitiveCategory.EMAIL,
    "PHONE_NUMBER": SensitiveCategory.PHONE_NUMBER,
    "URL": SensitiveCategory.URL,
    "US_BANK_NUMBER": SensitiveCategory.BANK_ACCOUNT,
    "US_SSN": SensitiveCategory.THAI_NATIONAL_ID,  # generic national-id bucket
    "US_PASSPORT": SensitiveCategory.PASSPORT_NUMBER,
}

#: Entity types Presidio may emit that we DROP (handled elsewhere / not wanted).
_IGNORED_ENTITY_TYPES: frozenset[str] = frozenset(
    {"PERSON", "LOCATION", "NRP", "DATE_TIME"}
)

#: Minimum Presidio score for a result to count (0..1). 0.4 keeps recall high
#: while dropping the weakest matches.
_DEFAULT_SCORE_THRESHOLD = 0.4

_LANGUAGE = "en"


class PresidioPatternClassifier:
    """A pattern-only :class:`~pii_guardrail.classifier.ClassifierModel` (Presidio).

    Exposes ``available`` + ``classify`` (per-segment). Construct once and reuse:
    building the registry + blank NLP engine is done once.
    """

    def __init__(self, score_threshold: float = _DEFAULT_SCORE_THRESHOLD) -> None:
        """Build the pattern-only analyzer; never raise on failure.

        Args:
            score_threshold: Minimum result score in [0, 1] to report an entity.

        On any failure (Presidio/spaCy not installed, engine build error), the
        classifier records the error and reports ``available`` as ``False`` so
        the Detector falls back to the other layers.
        """
        self._score_threshold = score_threshold
        self._analyzer: object | None = None
        self._load_error: Exception | None = None
        try:
            self._analyzer = self._build_analyzer()
        except Exception as exc:  # noqa: BLE001 - degrade gracefully, never raise
            self._analyzer = None
            self._load_error = exc

    @staticmethod
    def _build_analyzer() -> object:
        """Construct an AnalyzerEngine with predefined patterns + a BLANK spaCy.

        The blank pipeline (tokenizer only) satisfies Presidio's requirement for
        an NLP engine WITHOUT downloading or loading any NER model, keeping RAM
        low. Only the regex/pattern recognizers contribute results.
        """
        import spacy  # type: ignore[import-not-found]
        from presidio_analyzer import (  # type: ignore[import-not-found]
            AnalyzerEngine,
            RecognizerRegistry,
        )
        from presidio_analyzer.nlp_engine import (  # type: ignore[import-not-found]
            SpacyNlpEngine,
        )
        from presidio_analyzer.nlp_engine.ner_model_configuration import (  # type: ignore[import-not-found]
            NerModelConfiguration,
        )

        class _BlankSpacyNlpEngine(SpacyNlpEngine):
            """SpacyNlpEngine backed by a blank pipeline (no model, no NER)."""

            def __init__(self) -> None:
                self.ner_model_configuration = NerModelConfiguration()
                self.nlp = {"en": spacy.blank("en")}

        registry = RecognizerRegistry()
        registry.load_predefined_recognizers()
        return AnalyzerEngine(
            registry=registry,
            nlp_engine=_BlankSpacyNlpEngine(),
            supported_languages=[_LANGUAGE],
        )

    @property
    def available(self) -> bool:
        """``True`` only when the pattern analyzer built successfully."""
        return self._analyzer is not None

    def classify(self, text: str) -> set[SensitiveCategory]:
        """Return the pattern-detected sensitive categories in ``text``.

        Raises:
            ClassifierUnavailableError: If the analyzer is unavailable. The
                Detector catches this to fall back to the other layers.
        """
        if self._analyzer is None:
            raise ClassifierUnavailableError(
                "Presidio pattern analyzer is not available."
            ) from self._load_error

        if not text or not text.strip():
            return set()

        try:
            results = self._analyzer.analyze(  # type: ignore[attr-defined]
                text=text, language=_LANGUAGE
            )
        except Exception as exc:  # noqa: BLE001
            raise ClassifierUnavailableError(
                "Presidio pattern analysis failed."
            ) from exc

        return self._categories_from_results(results)

    def _categories_from_results(self, results: object) -> set[SensitiveCategory]:
        """Map Presidio results to categories, dropping NER-only entity types."""
        categories: set[SensitiveCategory] = set()
        try:
            iterator = iter(results)  # type: ignore[call-overload]
        except TypeError:
            return categories

        for result in iterator:
            entity_type = getattr(result, "entity_type", None)
            score = getattr(result, "score", 1.0)
            if not isinstance(entity_type, str):
                continue
            upper = entity_type.upper()
            if upper in _IGNORED_ENTITY_TYPES:
                continue
            try:
                if float(score) < self._score_threshold:
                    continue
            except (TypeError, ValueError):
                continue
            category = _PRESIDIO_ENTITY_TO_CATEGORY.get(upper)
            if category is not None:
                categories.add(category)
        return categories
