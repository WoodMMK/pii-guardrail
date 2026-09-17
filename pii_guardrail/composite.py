"""Composite classifier: fan out to several ``ClassifierModel`` layers and union.

The :class:`~pii_guardrail.detector.Detector` accepts a single
:class:`~pii_guardrail.classifier.ClassifierModel`. To run MULTIPLE advisory
layers side by side -- e.g. the Thai NER model
(:class:`~pii_guardrail.ner.ThaiNerClassifier`) for Thai names and Presidio
(:class:`~pii_guardrail.presidio_backend.PresidioClassifier`) for English PII --
we wrap them in one :class:`CompositeClassifier` that itself satisfies the
protocol.

Semantics:
    - :attr:`available` is ``True`` when AT LEAST ONE wrapped classifier is
      available. This lets the Detector consult the composite whenever any layer
      can contribute; the deterministic patterns in
      :func:`~pii_guardrail.detector.classify_segment` remain the source of
      truth regardless.
    - :meth:`classify` returns the UNION of categories from every currently
      available layer. Layers that are unavailable -- or that raise
      :class:`~pii_guardrail.errors.ClassifierUnavailableError` or any other
      error at call time -- are skipped silently so one broken layer never
      suppresses the others.
    - :meth:`classify` raises
      :class:`~pii_guardrail.errors.ClassifierUnavailableError` ONLY when the
      composite has no available layers at all, so the Detector records its
      single fallback warning and relies on patterns alone (Requirement 5.3).

This keeps each layer independent and optional: a deployment with only Presidio
installed, only PyThaiNLP installed, both, or neither all behave sensibly.

Requirements: 5.1, 5.2, 5.3.
"""

from __future__ import annotations

from collections.abc import Iterable

from pii_guardrail.errors import ClassifierUnavailableError
from pii_guardrail.models import SensitiveCategory

__all__ = ["CompositeClassifier"]


class CompositeClassifier:
    """Combine multiple ``ClassifierModel`` layers behind one protocol object.

    Structurally satisfies the ``ClassifierModel`` protocol (exposes
    ``available`` and ``classify``) by delegating to a list of wrapped
    classifiers and unioning their results.
    """

    def __init__(self, classifiers: Iterable[object]) -> None:
        """Create a composite over the given classifier layers.

        Args:
            classifiers: An iterable of objects that each satisfy the
                ``ClassifierModel`` protocol (expose an ``available`` property
                and a ``classify(text) -> set[SensitiveCategory]`` method).
                ``None`` entries are ignored so callers can pass optional
                layers without pre-filtering.
        """
        self._classifiers: list[object] = [c for c in classifiers if c is not None]

    @property
    def available(self) -> bool:
        """``True`` when at least one wrapped classifier is available.

        Availability is probed defensively: a layer whose ``available`` property
        raises is treated as unavailable rather than propagating the error.
        """
        for classifier in self._classifiers:
            try:
                if classifier.available:  # type: ignore[attr-defined]
                    return True
            except Exception:  # noqa: BLE001 - a broken layer is just unavailable
                continue
        return False

    def classify(self, text: str) -> set[SensitiveCategory]:
        """Union the categories from every currently available layer.

        Each wrapped classifier is consulted independently. A layer that is
        unavailable, raises :class:`ClassifierUnavailableError`, or fails for any
        other reason is skipped without affecting the others.

        Returns:
            The union of all available layers' categories (possibly empty when
            every available layer returns nothing).

        Raises:
            ClassifierUnavailableError: Only when NO wrapped layer is available,
                so the Detector falls back to pattern-based classification and
                records its fallback warning (Requirement 5.3).
        """
        categories: set[SensitiveCategory] = set()
        any_available = False

        for classifier in self._classifiers:
            try:
                if not classifier.available:  # type: ignore[attr-defined]
                    continue
            except Exception:  # noqa: BLE001 - treat a broken probe as unavailable
                continue

            any_available = True
            try:
                result = classifier.classify(text)  # type: ignore[attr-defined]
            except ClassifierUnavailableError:
                # This layer became unavailable at call time; skip it but keep
                # consulting the remaining layers.
                continue
            except Exception:  # noqa: BLE001 - never let one layer break the union
                continue

            if result:
                categories |= set(result)

        if not any_available:
            # No layer could contribute: signal fallback so the Detector records
            # its single warning and relies on the deterministic patterns.
            raise ClassifierUnavailableError(
                "No classifier layer is available."
            )

        return categories
