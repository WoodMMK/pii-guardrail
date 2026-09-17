"""Optional Presidio-backed classifier implementing the ``ClassifierModel`` protocol.

Wraps Microsoft Presidio's :class:`AnalyzerEngine` so the
:class:`~pii_guardrail.detector.Detector` can consult it for English-language
PII that the deterministic pattern rules and the Thai NER model miss -- in
particular free-form English person names, locations, and structured entities
such as credit-card numbers.

Design / contract (mirrors :mod:`pii_guardrail.ner`):
    - Presidio is an OPTIONAL dependency (the ``presidio`` extra in
      pyproject.toml). The reusable core MUST import cleanly without it, so this
      module imports ``presidio_analyzer`` LAZILY inside the constructor,
      wrapped in ``try/except``. A missing dependency or a model-load failure
      sets an internal "not available" state instead of raising;
      :attr:`available` reflects that.
    - The model is ADVISORY: the Detector unions its categories with the
      pattern-based ones and always retains patterns as the source of truth.
    - Runs LOCALLY (Presidio + its spaCy NLP engine execute in-process); the
      analyzed text never leaves the machine.

Presidio returns a list of ``RecognizerResult`` objects, each carrying an
``entity_type`` (e.g. ``PERSON``, ``EMAIL_ADDRESS``) and a ``score``. We map the
entity types onto :class:`SensitiveCategory` values, apply a minimum score
threshold to keep false positives low, and return the set of categories present
anywhere in the given text.

Requirements: 5.1, 5.2, 5.3 (a concrete, optional ``ClassifierModel``).
"""

from __future__ import annotations

import os

from pii_guardrail.errors import ClassifierUnavailableError
from pii_guardrail.models import SensitiveCategory

__all__ = ["PresidioClassifier"]


#: Map Presidio entity types to our :class:`SensitiveCategory`. Only entity
#: types that correspond to a sensitive category we model are mapped; anything
#: else Presidio may emit (e.g. ``NRP``, ``DATE_TIME`` handled separately) is
#: ignored. Presidio entity-type strings are upper-cased before lookup.
_PRESIDIO_ENTITY_TO_CATEGORY: dict[str, SensitiveCategory] = {
    "PERSON": SensitiveCategory.PERSON_NAME,
    "ORGANIZATION": SensitiveCategory.ORGANIZATION_NAME,
    "ORG": SensitiveCategory.ORGANIZATION_NAME,
    "EMAIL_ADDRESS": SensitiveCategory.EMAIL,
    "PHONE_NUMBER": SensitiveCategory.PHONE_NUMBER,
    "URL": SensitiveCategory.URL,
    "LOCATION": SensitiveCategory.THAI_ADDRESS,
    "GPE": SensitiveCategory.THAI_ADDRESS,
    "DATE_TIME": SensitiveCategory.DATE_OF_BIRTH,
    "CREDIT_CARD": SensitiveCategory.BANK_ACCOUNT,
    "IBAN_CODE": SensitiveCategory.BANK_ACCOUNT,
    "US_BANK_NUMBER": SensitiveCategory.BANK_ACCOUNT,
    "US_PASSPORT": SensitiveCategory.PASSPORT_NUMBER,
    "US_SSN": SensitiveCategory.THAI_NATIONAL_ID,
    "CRYPTO": SensitiveCategory.SECRET,
}

#: Minimum Presidio confidence score for a result to count. Presidio scores are
#: in [0, 1]; 0.4 keeps recall high while discarding the weakest matches.
_DEFAULT_SCORE_THRESHOLD = 0.4

#: Default language passed to Presidio's analyzer. English only: Thai names are
#: handled by :class:`~pii_guardrail.ner.ThaiNerClassifier`, and Thai/structured
#: PII by the deterministic patterns.
_DEFAULT_LANGUAGE = "en"

#: Default spaCy model Presidio's NLP engine loads. The SMALL English model
#: (~12MB on disk, ~50MB RAM) is used deliberately instead of Presidio's own
#: default of ``en_core_web_lg`` (~400MB / ~600MB RAM): the large model roughly
#: TRIPLES the server's memory footprint for a modest NER accuracy gain we do
#: not need here (structured/Thai PII is handled by patterns + Thai NER).
#: Override with the ``PII_GUARDRAIL_SPACY_MODEL`` environment variable, e.g.
#: ``en_core_web_md`` or ``en_core_web_lg``, when higher accuracy is worth the
#: extra memory.
_DEFAULT_SPACY_MODEL = "en_core_web_sm"
_SPACY_MODEL_ENV_VAR = "PII_GUARDRAIL_SPACY_MODEL"


class PresidioClassifier:
    """A :class:`~pii_guardrail.classifier.ClassifierModel` backed by Presidio.

    Structurally satisfies the protocol (exposes ``available`` and ``classify``)
    without inheriting from it, keeping the core free of a Presidio/spaCy
    dependency at import time. Construct once and reuse: building the analyzer
    (which loads a spaCy model) is expensive.
    """

    def __init__(
        self,
        language: str = _DEFAULT_LANGUAGE,
        score_threshold: float = _DEFAULT_SCORE_THRESHOLD,
        model_name: str | None = None,
    ) -> None:
        """Attempt to build the Presidio analyzer; never raise on failure.

        Args:
            language: Language code passed to Presidio (``"en"`` by default).
                Must match a language the configured spaCy NLP engine supports.
            score_threshold: Minimum result score in [0, 1] for a detected
                entity to be reported. Lower values increase recall (and false
                positives); higher values are more conservative.
            model_name: spaCy model to load. Defaults to the
                ``PII_GUARDRAIL_SPACY_MODEL`` environment variable when set,
                otherwise the small English model (``en_core_web_sm``) to keep
                the memory footprint low. Pass ``en_core_web_md`` /
                ``en_core_web_lg`` (and install it) for higher accuracy at the
                cost of more RAM.

        On success :attr:`available` is ``True``. On any failure
        (``presidio_analyzer`` not installed, spaCy model missing, engine build
        error), the classifier records the error and reports :attr:`available`
        as ``False`` so the Detector falls back to the other layers.
        """
        self._language = language
        self._score_threshold = score_threshold
        self._model_name = (
            model_name
            or os.environ.get(_SPACY_MODEL_ENV_VAR)
            or _DEFAULT_SPACY_MODEL
        )
        self._analyzer: object | None = None
        self._load_error: Exception | None = None
        try:
            # Lazy import keeps Presidio optional: `import pii_guardrail` (and
            # this module) works without the `presidio` extra installed.
            from presidio_analyzer import AnalyzerEngine  # type: ignore[import-not-found]
            from presidio_analyzer.nlp_engine import (  # type: ignore[import-not-found]
                NlpEngineProvider,
            )

            # Build the NLP engine on the SMALL model explicitly. Presidio's own
            # default targets ``en_core_web_lg``; configuring the provider lets
            # us pick a lighter model and cut the memory footprint by ~500MB.
            nlp_configuration = {
                "nlp_engine_name": "spacy",
                "models": [
                    {"lang_code": self._language, "model_name": self._model_name}
                ],
            }
            nlp_engine = NlpEngineProvider(
                nlp_configuration=nlp_configuration
            ).create_engine()

            self._analyzer = AnalyzerEngine(
                nlp_engine=nlp_engine,
                supported_languages=[self._language],
            )
        except Exception as exc:  # noqa: BLE001 - degrade gracefully, never raise
            self._analyzer = None
            self._load_error = exc

    @property
    def available(self) -> bool:
        """``True`` only when the Presidio analyzer built successfully."""
        return self._analyzer is not None

    def classify(self, text: str) -> set[SensitiveCategory]:
        """Return the sensitive categories Presidio finds in ``text``.

        Args:
            text: The recognized text segment to classify.

        Returns:
            A ``set[SensitiveCategory]`` for the entity types Presidio detected
            above the score threshold (possibly empty). The Detector unions this
            with the pattern-based and Thai-NER results.

        Raises:
            ClassifierUnavailableError: If invoked while the analyzer is
                unavailable. The Detector catches this to fall back to the other
                classification layers (Requirement 5.3).
        """
        if self._analyzer is None:
            raise ClassifierUnavailableError(
                "Presidio analyzer is not available."
            ) from self._load_error

        # Empty/whitespace text has no entities; skip the analyzer call.
        if not text or not text.strip():
            return set()

        try:
            results = self._analyzer.analyze(  # type: ignore[attr-defined]
                text=text,
                language=self._language,
            )
        except Exception as exc:  # noqa: BLE001
            # A runtime failure inside Presidio is treated as "unavailable" so
            # the Detector falls back to the other layers rather than crashing.
            raise ClassifierUnavailableError(
                "Presidio analyzer failed during analysis."
            ) from exc

        return self._categories_from_results(results)

    # -- internals ----------------------------------------------------------

    def _categories_from_results(self, results: object) -> set[SensitiveCategory]:
        """Map Presidio ``RecognizerResult`` objects to sensitive categories.

        ``results`` is an iterable of objects exposing ``entity_type`` (str) and
        ``score`` (float). Results below the configured score threshold are
        discarded; the remaining entity types are upper-cased and mapped to a
        :class:`SensitiveCategory` when one exists.
        """
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
            try:
                if float(score) < self._score_threshold:
                    continue
            except (TypeError, ValueError):
                continue
            category = _PRESIDIO_ENTITY_TO_CATEGORY.get(entity_type.upper())
            if category is not None:
                categories.add(category)
        return categories
