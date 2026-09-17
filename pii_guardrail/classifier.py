"""Classifier_Model protocol (optional, with fallback).

Defines the structural contract for an optional small language / classifier
model that the :class:`~pii_guardrail.detector.Detector` may consult to assist
pattern-based classification (especially for ambiguous Thai names/orgs).

The model is *strictly optional* and *advisory*:

    - The Detector always retains pattern-based classification as the source of
      truth. When the model is enabled and available, its results are unioned
      with the pattern results.
    - When the model is enabled but unavailable at runtime (``available`` is
      ``False``), or a call to :meth:`ClassifierModel.classify` raises
      :class:`~pii_guardrail.errors.ClassifierUnavailableError`, the Detector
      falls back to patterns and records a warning.

This module is dependency-free at import time: it defines only a
:class:`typing.Protocol` (structural typing), so any object exposing the
``available`` property and the ``classify`` method satisfies it without an
explicit base class or an ML dependency.

Requirements: 5.1, 5.2, 5.3.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pii_guardrail.models import SensitiveCategory

__all__ = ["ClassifierModel"]


@runtime_checkable
class ClassifierModel(Protocol):
    """Structural contract for an optional classification model.

    Any object exposing an ``available`` property and a ``classify`` method with
    the signatures below satisfies this protocol. Decorated with
    :func:`typing.runtime_checkable` so ``isinstance(obj, ClassifierModel)`` can
    be used for defensive runtime checks, though the Detector relies on duck
    typing rather than isinstance.
    """

    @property
    def available(self) -> bool:
        """Whether the model is loaded and ready to classify text.

        The Detector consults this before invoking :meth:`classify`. When it is
        ``False`` and the classifier is enabled, the Detector falls back to
        pattern-based classification and records a warning (Requirement 5.3).
        """
        ...

    def classify(self, text: str) -> set[SensitiveCategory]:
        """Return the categories the model believes apply to ``text``.

        Args:
            text: The recognized text segment to classify.

        Returns:
            A ``set[SensitiveCategory]`` the model believes apply (possibly
            empty). The Detector unions this with pattern-based results.

        Raises:
            ClassifierUnavailableError: If invoked while the model is
                unavailable. The Detector catches this to fall back to
                pattern-based classification (Requirement 5.3).
        """
        ...
