"""Optional Thai NER classifier implementing the ``ClassifierModel`` protocol.

Wraps PyThaiNLP's named-entity recognizer so the
:class:`~pii_guardrail.detector.Detector` can consult it to catch person and
organization names that the deterministic pattern rules miss -- in particular
full names (given + family) that OCR emits without a clean honorific/label.

Design / contract:
    - PyThaiNLP is an OPTIONAL dependency (the ``ner`` extra in pyproject.toml).
      The reusable core MUST import cleanly without it, so this module imports
      ``pythainlp`` LAZILY inside the constructor, wrapped in ``try/except``. A
      missing dependency or a model-load failure sets an internal "not
      available" state instead of raising; :attr:`available` reflects that.
    - The model is ADVISORY: the Detector unions its categories with the
      pattern-based ones and always retains patterns as the source of truth.
    - Runs LOCALLY (no network); uploaded text never leaves the machine.

The NER model returns IOB-tagged tokens (e.g. ``B-PERSON``/``I-PERSON``). We map
the entity types PyThaiNLP produces onto :class:`SensitiveCategory` values and
return the set of categories present anywhere in the given text.

Requirements: 5.1, 5.2, 5.3 (this is the concrete, optional ``ClassifierModel``).
"""

from __future__ import annotations

from pii_guardrail.errors import ClassifierUnavailableError
from pii_guardrail.models import SensitiveCategory

__all__ = ["ThaiNerClassifier"]


#: Map PyThaiNLP NER entity types (the part after the ``B-``/``I-`` prefix) to
#: our :class:`SensitiveCategory`. Only entity types that correspond to a
#: sensitive category are mapped; anything else (e.g. ``TIME``, ``LAW``) is
#: ignored. PyThaiNLP tags are upper-cased before lookup.
_NER_TAG_TO_CATEGORY: dict[str, SensitiveCategory] = {
    "PERSON": SensitiveCategory.PERSON_NAME,
    "ORGANIZATION": SensitiveCategory.ORGANIZATION_NAME,
    "LOCATION": SensitiveCategory.THAI_ADDRESS,
    "EMAIL": SensitiveCategory.EMAIL,
    "PHONE": SensitiveCategory.PHONE_NUMBER,
    "URL": SensitiveCategory.URL,
    "ZIP": SensitiveCategory.THAI_ADDRESS,
    "DATE": SensitiveCategory.DATE_OF_BIRTH,
    "MONEY": SensitiveCategory.MONEY_AMOUNT,
}


class ThaiNerClassifier:
    """A :class:`~pii_guardrail.classifier.ClassifierModel` backed by PyThaiNLP.

    Structurally satisfies the protocol (exposes ``available`` and ``classify``)
    without inheriting from it, keeping the core free of an NLP dependency at
    import time. Construct once and reuse: loading the NER model is expensive.
    """

    def __init__(self, engine: str = "thainer") -> None:
        """Attempt to load the PyThaiNLP NER model; never raise on failure.

        Args:
            engine: PyThaiNLP NER engine name. ``"thainer"`` (CRF-based) is the
                default: it is lightweight and CPU-friendly. ``"thainer-v2"``
                is more accurate but pulls in heavy transformer dependencies.

        On success :attr:`available` is ``True``. On any failure (``pythainlp``
        not installed, model/data download failure), the classifier records the
        error and reports :attr:`available` as ``False`` so the Detector falls
        back to pattern-based classification.
        """
        self._engine_name = engine
        self._ner: object | None = None
        self._load_error: Exception | None = None
        try:
            # Lazy import keeps PyThaiNLP optional: `import pii_guardrail` (and
            # this module) works without the `ner` extra installed.
            from pythainlp.tag import NER  # type: ignore[import-not-found]

            self._ner = NER(engine)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully, never raise
            self._ner = None
            self._load_error = exc

    @property
    def available(self) -> bool:
        """``True`` only when the PyThaiNLP NER model loaded successfully."""
        return self._ner is not None

    def classify(self, text: str) -> set[SensitiveCategory]:
        """Return the sensitive categories the NER model finds in ``text``.

        Args:
            text: The recognized text segment to classify.

        Returns:
            A ``set[SensitiveCategory]`` for the entity types the model tagged
            (possibly empty). The Detector unions this with pattern results.

        Raises:
            ClassifierUnavailableError: If invoked while the model is
                unavailable. The Detector catches this to fall back to
                pattern-based classification (Requirement 5.3).
        """
        if self._ner is None:
            raise ClassifierUnavailableError(
                "Thai NER model is not available."
            ) from self._load_error

        # Empty/whitespace text has no entities; skip the model call.
        if not text or not text.strip():
            return set()

        try:
            tagged = self._ner.tag(text, pos=False)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            # A runtime failure inside the model is treated as "unavailable" so
            # the Detector falls back to patterns rather than crashing.
            raise ClassifierUnavailableError(
                "Thai NER model failed during tagging."
            ) from exc

        return self._categories_from_iob(tagged)

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _categories_from_iob(tagged: object) -> set[SensitiveCategory]:
        """Map IOB-tagged NER output to a set of sensitive categories.

        ``tagged`` is a sequence of ``(token, ner_tag)`` tuples where ``ner_tag``
        is ``"O"`` or an IOB tag like ``"B-PERSON"`` / ``"I-PERSON"``. We take
        the entity type (the part after ``B-``/``I-``), upper-case it, and add
        the mapped :class:`SensitiveCategory` when one exists.
        """
        categories: set[SensitiveCategory] = set()
        try:
            iterator = iter(tagged)  # type: ignore[call-overload]
        except TypeError:
            return categories

        for item in iterator:
            # Each item is (token, tag) or (token, tag, pos). Take the tag at [1].
            if not isinstance(item, (tuple, list)) or len(item) < 2:
                continue
            tag = item[1]
            if not isinstance(tag, str) or tag == "O" or "-" not in tag:
                continue
            entity_type = tag.split("-", 1)[1].upper()
            category = _NER_TAG_TO_CATEGORY.get(entity_type)
            if category is not None:
                categories.add(category)
        return categories
