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

    def classify_segments(
        self, segments: list
    ) -> dict[int, set[SensitiveCategory]]:
        """Whole-page classification across mixed layer types.

        The Detector prefers this hook when present. It combines two kinds of
        wrapped layers:

        * WHOLE-PAGE layers (expose ``classify_segments``, e.g. the LLM) are
          called ONCE with all segments so they keep full-page context, and
          their ``{index: categories}`` mapping is merged in.
        * PER-SEGMENT layers (expose only ``classify``, e.g. the Presidio
          pattern and detect-secrets backends) are called once per segment and
          their result is merged at that segment's index.

        Every layer is consulted independently and defensively: an unavailable
        or failing layer is skipped without affecting the others.

        Returns:
            ``{segment_index: set[SensitiveCategory]}`` for segments that any
            layer flagged (empty entries omitted).

        Raises:
            ClassifierUnavailableError: Only when NO wrapped layer is available,
                so the Detector falls back to pattern-only classification.
        """
        seg_list = list(segments)
        result: dict[int, set[SensitiveCategory]] = {}
        any_available = False

        for classifier in self._classifiers:
            try:
                if not classifier.available:  # type: ignore[attr-defined]
                    continue
            except Exception:  # noqa: BLE001 - treat a broken probe as unavailable
                continue
            any_available = True

            whole_page = getattr(classifier, "classify_segments", None)
            if callable(whole_page):
                # Whole-page layer: one call for all segments.
                try:
                    mapping = whole_page(seg_list)
                except (ClassifierUnavailableError, Exception):  # noqa: BLE001
                    continue
                if isinstance(mapping, dict):
                    for idx, cats in mapping.items():
                        try:
                            i = int(idx)
                        except (TypeError, ValueError):
                            continue
                        if cats:
                            result.setdefault(i, set()).update(cats)
            else:
                # Per-segment layer: classify each segment's text.
                for i, seg in enumerate(seg_list):
                    text = getattr(seg, "text", None)
                    if not isinstance(text, str) or not text.strip():
                        continue
                    try:
                        cats = classifier.classify(text)  # type: ignore[attr-defined]
                    except (ClassifierUnavailableError, Exception):  # noqa: BLE001
                        continue
                    if cats:
                        result.setdefault(i, set()).update(cats)

        if not any_available:
            raise ClassifierUnavailableError(
                "No classifier layer is available."
            )

        return result

    def classify_segments_by_source(
        self, segments: list
    ) -> dict[int, dict[str, set[SensitiveCategory]]]:
        """Like :meth:`classify_segments` but KEEPS each layer's result separate.

        Returns ``{segment_index: {source_name: set[categories]}}`` so callers
        (e.g. a debug view) can see WHICH layer flagged WHAT on each segment,
        rather than one merged set. Layers are labelled by their ``source_name``
        attribute (falling back to the class name). Same defensive semantics as
        :meth:`classify_segments`: failing/unavailable layers are skipped.

        Raises:
            ClassifierUnavailableError: Only when NO wrapped layer is available.
        """
        seg_list = list(segments)
        result: dict[int, dict[str, set[SensitiveCategory]]] = {}
        any_available = False

        def _record(index: int, source: str, cats: set[SensitiveCategory]) -> None:
            if not cats:
                return
            result.setdefault(index, {}).setdefault(source, set()).update(cats)

        for classifier in self._classifiers:
            try:
                if not classifier.available:  # type: ignore[attr-defined]
                    continue
            except Exception:  # noqa: BLE001
                continue
            any_available = True

            source = getattr(classifier, "source_name", type(classifier).__name__)
            whole_page = getattr(classifier, "classify_segments", None)
            if callable(whole_page):
                try:
                    mapping = whole_page(seg_list)
                except (ClassifierUnavailableError, Exception):  # noqa: BLE001
                    continue
                if isinstance(mapping, dict):
                    for idx, cats in mapping.items():
                        try:
                            i = int(idx)
                        except (TypeError, ValueError):
                            continue
                        _record(i, source, set(cats) if cats else set())
            else:
                for i, seg in enumerate(seg_list):
                    text = getattr(seg, "text", None)
                    if not isinstance(text, str) or not text.strip():
                        continue
                    try:
                        cats = classifier.classify(text)  # type: ignore[attr-defined]
                    except (ClassifierUnavailableError, Exception):  # noqa: BLE001
                        continue
                    _record(i, source, set(cats) if cats else set())

        if not any_available:
            raise ClassifierUnavailableError(
                "No classifier layer is available."
            )

        return result
