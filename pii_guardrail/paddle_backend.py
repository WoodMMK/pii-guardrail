"""Default PaddleOCR-backed :class:`OCRBackend` implementation.

Wraps a PaddleOCR-family model (Thai + Latin recognition, text-detection
quadrilaterals, and angle classification) behind the :class:`OCRBackend`
protocol defined in :mod:`pii_guardrail.ocr`. This module is the concrete
default backend the :class:`~pii_guardrail.ocr.OCREngine` loads lazily when no
backend is injected.

Requirements:
    2.3 -- recognize Thai-language text.
    2.4 -- recognize Latin-script text.
    2.5 -- WHERE a PaddleOCR-family model is available, use it for extraction.

Optional-dependency contract (design.md, "Technology Choices"):
    PaddleOCR is an OPTIONAL dependency (the ``ocr`` extra in pyproject.toml).
    The reusable core MUST import cleanly without PaddleOCR installed. Therefore
    this module imports ``paddleocr`` LAZILY inside the constructor, wrapped in a
    ``try/except``. A missing dependency or a model-load failure sets an internal
    "not available" state instead of raising, and :attr:`available` reflects
    whether the model actually loaded. Extraction (:meth:`run`) fails closed with
    :class:`~pii_guardrail.errors.OCRProcessingError` when the model is
    unavailable or a call fails.
"""

from __future__ import annotations

from numpy.typing import NDArray

from pii_guardrail.errors import OCRProcessingError
from pii_guardrail.geometry import normalize_quad_to_box
from pii_guardrail.models import TextSegment

__all__ = ["PaddleOCRBackend"]


class PaddleOCRBackend:
    """A :class:`~pii_guardrail.ocr.OCRBackend` backed by a PaddleOCR model.

    The constructor attempts to load a PaddleOCR model configured for Thai +
    Latin recognition with angle classification. The PaddleOCR Thai recognition
    model also handles Latin script, so a single ``lang='th'`` model satisfies
    Requirements 2.3 and 2.4. The import and instantiation happen inside a
    ``try/except`` so a missing optional dependency (Requirement 2.5's "WHERE
    ... available") or a load failure degrades gracefully: :attr:`available`
    becomes ``False`` and no exception escapes construction.

    This class satisfies the :class:`~pii_guardrail.ocr.OCRBackend` protocol
    structurally (it exposes ``run`` and an ``available`` property); it does not
    inherit from it, keeping the core free of an ML dependency at import time.
    """

    def __init__(self) -> None:
        """Attempt to load the PaddleOCR model; never raise on failure.

        On success, :attr:`available` is ``True`` and the model is used by
        :meth:`run`. On any failure (``paddleocr`` not installed, native
        dependency missing, model download/load error), the backend records the
        failure and reports :attr:`available` as ``False``.

        The PaddleOCR constructor signature changed substantially between the
        2.x and 3.x lines: 3.x renamed ``use_angle_cls`` to
        ``use_textline_orientation`` and REMOVED ``show_log`` entirely (passing
        it raises ``ValueError``). We therefore try a sequence of
        progressively-more-compatible argument sets, catching a BROAD
        ``Exception`` between attempts (2.x raised ``TypeError`` for unknown
        kwargs, 3.x raises ``ValueError``), and only mark the backend
        unavailable if every attempt fails.
        """
        self._model: object | None = None
        self._load_error: Exception | None = None
        self._api_generation: str | None = None

        # Each entry is (api-generation label, kwargs) tried in order. The 3.x
        # variant additionally requests ``enable_mkldnn=False``: on some
        # Windows/PaddlePaddle builds the default oneDNN (MKL-DNN) execution
        # path raises an unimplemented-attribute error during inference, so we
        # disable it up front. The kwarg is forwarded through PaddleOCR 3.x's
        # ``**kwargs`` to the underlying predictor and is harmless when ignored.
        # Each entry is (api-generation label, kwargs) tried in order. The 3.x
        # variant additionally requests ``enable_mkldnn=False``: on some
        # Windows/PaddlePaddle builds the default oneDNN (MKL-DNN) execution
        # path raises an unimplemented-attribute error during inference, so we
        # disable it up front. The kwarg is forwarded through PaddleOCR 3.x's
        # ``**kwargs`` to the underlying predictor and is harmless when ignored.
        #
        # NOTE on speed: an attempt to speed up by forcing the mobile detector
        # and disabling doc-orientation/unwarping was measured to be SLOWER and
        # LESS accurate on this build (it also silently dropped the Thai
        # recognizer when a detector model name was pinned), so it was reverted.
        # The real cost seen during testing was reloading the model each run;
        # keeping a warm server process (one load, many requests) is the win.
        attempts: tuple[tuple[str, dict[str, object]], ...] = (
            # 3.x with word boxes (tight per-word redaction) + Thai recognizer.
            ("3.x", {"use_textline_orientation": True, "lang": "th",
                     "enable_mkldnn": False, "return_word_box": True}),
            # 3.x without word boxes (older 3.x that rejects return_word_box).
            ("3.x-noword", {"use_textline_orientation": True, "lang": "th",
                            "enable_mkldnn": False}),
            # 2.x style: angle classifier + silenced logs.
            ("2.x", {"use_angle_cls": True, "lang": "th", "show_log": False}),
            # Minimal: just the language; lets PaddleOCR pick every default.
            ("minimal", {"lang": "th"}),
        )

        for generation, kwargs in attempts:
            try:
                # Lazy import inside the try: keeps PaddleOCR optional so
                # `import pii_guardrail` (and this module) works without the
                # `ocr` extra installed.
                from paddleocr import PaddleOCR  # type: ignore[import-not-found]

                self._model = PaddleOCR(**kwargs)  # type: ignore[arg-type]
                self._api_generation = generation
                self._load_error = None
                break
            except Exception as exc:  # noqa: BLE001 - broad on purpose (see docstring)
                # A missing optional dependency, an unknown kwarg (TypeError on
                # 2.x, ValueError on 3.x), or a genuine load failure: remember
                # the last error and try the next, more-compatible, arg set.
                self._model = None
                self._load_error = exc

    @property
    def available(self) -> bool:
        """``True`` only when the PaddleOCR model loaded successfully."""
        return self._model is not None

    def run(self, image: NDArray) -> list[TextSegment]:
        """Extract text segments from ``image`` using the loaded PaddleOCR model.

        Args:
            image: The (preprocessed) image as a NumPy array.

        Returns:
            A list of :class:`~pii_guardrail.models.TextSegment`, one per detected
            line, each with its text, an axis-aligned in-bounds
            :class:`~pii_guardrail.models.BoundingBox` (the enclosing rectangle of
            PaddleOCR's quadrilateral), and a confidence clamped into [0.0, 1.0].
            Returns ``[]`` when PaddleOCR finds no text.

        Raises:
            OCRProcessingError: If the backend is unavailable, or the PaddleOCR
                call fails for any reason. No partial result is produced.
        """
        if self._model is None:
            raise OCRProcessingError(
                "PaddleOCR backend is not available."
            ) from self._load_error

        height, width = int(image.shape[0]), int(image.shape[1])

        try:
            raw = self._invoke_model(image)
        except Exception as exc:
            raise OCRProcessingError(
                "PaddleOCR failed during text extraction."
            ) from exc

        try:
            return self._parse_result(raw, width, height)
        except OCRProcessingError:
            raise
        except Exception as exc:
            raise OCRProcessingError(
                "PaddleOCR returned a result in an unexpected format."
            ) from exc

    # -- internals ----------------------------------------------------------

    def _invoke_model(self, image: NDArray) -> object:
        """Call the underlying PaddleOCR model, tolerating API differences.

        PaddleOCR 3.x exposes ``predict(input)`` as the supported inference
        entry point and returns a list of ``OCRResult`` objects. Its ``ocr``
        method still exists but is deprecated and returns a different shape, so
        we PREFER ``predict`` whenever it is present. On 2.x (no ``predict``) we
        use ``ocr(image, cls=True)``, falling back to ``ocr(image)`` when the
        ``cls`` keyword is not accepted.
        """
        model = self._model
        assert model is not None  # guarded by caller

        # Prefer the 3.x predict() surface when available.
        predict = getattr(model, "predict", None)
        if callable(predict):
            return predict(image)

        ocr = getattr(model, "ocr", None)
        if ocr is None:  # pragma: no cover - depends on paddleocr API
            # No known inference entry point; try calling the model directly.
            return model(image)  # type: ignore[operator]
        try:
            return ocr(image, cls=True)
        except TypeError:  # pragma: no cover - depends on paddleocr version
            return ocr(image)

    def _parse_result(
        self, raw: object, width: int, height: int
    ) -> list[TextSegment]:
        """Convert PaddleOCR's raw output into normalized ``TextSegment``s.

        PaddleOCR result formats handled defensively:

        - Classic (PaddleOCR <= 2.x) ``ocr(img, cls=True)`` returns a list with
          one entry per image; for a single image that entry is a list of lines,
          each line being ``[quad, (text, confidence)]`` where ``quad`` is four
          ``[x, y]`` vertices. So the shape is::

              [[ [quad, (text, conf)], [quad, (text, conf)], ... ]]

        - Some versions return the per-image list directly (no outer wrapper),
          i.e. ``[[quad, (text, conf)], ...]``.

        - No text: PaddleOCR may return ``None``, ``[None]``, or ``[]``.

        We unwrap a single-image outer list when present, then read each line's
        quadrilateral and ``(text, confidence)`` pair, normalizing the quad to an
        in-bounds :class:`BoundingBox` via
        :func:`~pii_guardrail.geometry.normalize_quad_to_box`.
        """
        if raw is None:
            return []
        if not isinstance(raw, (list, tuple)):
            raise OCRProcessingError("PaddleOCR result is not a list.")

        # PaddleOCR 3.x: predict() returns a list with one dict-like OCRResult
        # per input image, exposing parallel `rec_texts`/`rec_scores`/`rec_polys`
        # lists. Detect and parse that shape first; it is structurally distinct
        # from the 2.x nested-list-of-lines format handled below.
        if self._looks_like_v3_result(raw):
            return self._parse_v3_result(list(raw), width, height)

        lines = self._unwrap_lines(list(raw))
        segments: list[TextSegment] = []
        for line in lines:
            if line is None:
                continue
            quad, text, confidence = self._parse_line(line)
            box = normalize_quad_to_box(quad, width, height)
            segments.append(
                TextSegment(
                    text=text,
                    box=box,
                    confidence=self._clamp_confidence(confidence),
                )
            )
        return segments

    @staticmethod
    def _v3_field(res: object, key: str) -> object | None:
        """Read ``key`` from a 3.x per-image result object, defensively.

        The per-image element is typically a dict-like ``OCRResult`` supporting
        ``res[key]``, but we also fall back to attribute access and to a
        ``res.json``/``res.res`` mapping so we tolerate minor version drift.
        """
        # dict-like: OCRResult supports __getitem__ + keys().
        try:
            keys = res.keys()  # type: ignore[attr-defined]
        except Exception:
            keys = None
        if keys is not None:
            try:
                if key in keys:
                    return res[key]  # type: ignore[index]
            except Exception:
                pass
        # Plain attribute (e.g. res.rec_texts).
        value = getattr(res, key, None)
        if value is not None:
            return value
        # Nested json/res mappings sometimes wrap the payload.
        for container_attr in ("json", "res"):
            container = getattr(res, container_attr, None)
            if isinstance(container, dict) and key in container:
                return container[key]
            if isinstance(container, dict) and "res" in container:
                inner = container["res"]
                if isinstance(inner, dict) and key in inner:
                    return inner[key]
        return None

    @classmethod
    def _looks_like_v3_result(cls, raw) -> bool:
        """Heuristic: is ``raw`` a PaddleOCR 3.x list of ``OCRResult`` objects?

        True when the first non-``None`` element exposes ``rec_texts`` (via
        dict-like access or an attribute). This is unambiguous versus the 2.x
        format, whose elements are lists of ``[quad, (text, conf)]`` lines.
        """
        for element in raw:
            if element is None:
                continue
            return cls._v3_field(element, "rec_texts") is not None
        return False

    @classmethod
    def _parse_v3_result(
        cls, raw: list, width: int, height: int
    ) -> list[TextSegment]:
        """Parse the PaddleOCR 3.x ``predict`` output into ``TextSegment``s.

        Each element is a per-image ``OCRResult`` exposing parallel lists:
        ``rec_texts`` (str), ``rec_scores`` (float), and ``rec_polys`` (4x2
        point arrays; ``dt_polys`` as a fallback). We zip them per detected
        line, normalize each polygon to an in-bounds box, and clamp confidence.
        The no-text case (empty ``rec_texts``) yields ``[]``.
        """
        segments: list[TextSegment] = []
        for res in raw:
            if res is None:
                continue
            texts = cls._v3_field(res, "rec_texts")
            if texts is None:
                continue
            scores = cls._v3_field(res, "rec_scores")
            polys = cls._v3_field(res, "rec_polys")
            if polys is None:
                polys = cls._v3_field(res, "dt_polys")

            # Optional per-word text + regions (present when the model was
            # constructed with return_word_box=True). Used to emit tight
            # word-level boxes so redaction does not cover a whole line.
            words = cls._v3_field(res, "text_word")
            word_regions = cls._v3_field(res, "text_word_region")

            texts_list = list(texts)
            scores_list = list(scores) if scores is not None else []
            polys_list = list(polys) if polys is not None else []
            words_list = list(words) if words is not None else []
            word_regions_list = (
                list(word_regions) if word_regions is not None else []
            )

            for index, text in enumerate(texts_list):
                poly = polys_list[index] if index < len(polys_list) else None
                if poly is None:
                    # Without a polygon we cannot place a box; skip the line.
                    continue
                score = scores_list[index] if index < len(scores_list) else 0.0
                try:
                    confidence = cls._clamp_confidence(float(score))
                except (TypeError, ValueError):
                    confidence = 0.0

                # Prefer WORD-level segments for this line when available: one
                # TextSegment per word, each with its own tight box. Fall back to
                # a single line-level segment when word data is absent/malformed.
                line_words = words_list[index] if index < len(words_list) else None
                line_word_regions = (
                    word_regions_list[index]
                    if index < len(word_regions_list)
                    else None
                )
                word_segments = cls._word_segments_for_line(
                    line_words, line_word_regions, confidence, width, height
                )
                if word_segments:
                    segments.extend(word_segments)
                    continue

                # Fallback: whole-line box.
                quad = cls._to_quad(poly)
                box = normalize_quad_to_box(quad, width, height)
                text_str = text if isinstance(text, str) else (
                    "" if text is None else str(text)
                )
                segments.append(
                    TextSegment(text=text_str, box=box, confidence=confidence)
                )
        return segments

    @classmethod
    def _word_segments_for_line(
        cls,
        words: object,
        word_regions: object,
        confidence: float,
        width: int,
        height: int,
    ) -> list[TextSegment]:
        """Build per-word ``TextSegment``s from a line's word text + regions.

        ``words`` is a list of word strings and ``word_regions`` a parallel list
        of quadrilaterals (each 4 ``(x, y)`` vertices), as produced by PaddleOCR
        when ``return_word_box=True`` (fields ``text_word`` / ``text_word_region``).
        Returns one segment per non-empty word with its own tight, in-bounds box.
        Returns ``[]`` when the data is missing or unusable, so the caller falls
        back to a single line-level segment.
        """
        if not isinstance(words, (list, tuple)) or not isinstance(
            word_regions, (list, tuple)
        ):
            return []
        if not words or len(words) != len(word_regions):
            return []

        out: list[TextSegment] = []
        for word, region in zip(words, word_regions):
            text_str = word if isinstance(word, str) else (
                "" if word is None else str(word)
            )
            # Skip whitespace-only words: no visible glyphs to redact, and the
            # detector treats them as non-sensitive anyway.
            if not text_str.strip():
                continue
            try:
                quad = cls._to_quad(region)
                box = normalize_quad_to_box(quad, width, height)
            except Exception:
                # A malformed word region invalidates word-level parsing for
                # this line; signal fallback by returning nothing.
                return []
            out.append(TextSegment(text=text_str, box=box, confidence=confidence))
        return out

    @staticmethod
    def _to_quad(poly) -> list[tuple[float, float]]:
        """Convert a 3.x polygon (4x2, possibly numpy) into plain (x, y) tuples.

        Accepts numpy arrays or nested sequences of vertices; each vertex is an
        ``(x, y)`` pair whose components are coerced to plain Python floats so
        the geometry helper receives no numpy scalars.
        """
        quad: list[tuple[float, float]] = []
        for vertex in poly:
            quad.append((float(vertex[0]), float(vertex[1])))
        if not quad:
            raise OCRProcessingError("PaddleOCR 3.x polygon has no vertices.")
        return quad

    @staticmethod
    def _unwrap_lines(raw: list) -> list:
        """Return the flat list of per-line entries from a PaddleOCR result.

        Handles the single-image outer wrapper (``[[line, line, ...]]``) as well
        as the already-flat form (``[line, line, ...]``) and the no-text forms
        (``[]`` / ``[None]``).
        """
        if not raw:
            return []
        # Single-image outer wrapper: one element that is itself a list of lines
        # (or None for "no text on this image").
        if len(raw) == 1:
            only = raw[0]
            if only is None:
                return []
            if isinstance(only, (list, tuple)) and PaddleOCRBackend._is_line_list(
                only
            ):
                return list(only)
        # Otherwise assume the top level is already the list of lines.
        return raw

    @staticmethod
    def _is_line_list(candidate) -> bool:
        """Heuristic: is ``candidate`` a list of ``[quad, (text, conf)]`` lines?

        A line is a 2-element sequence whose first element is itself a sequence
        of vertices (the quad). This distinguishes a per-image list-of-lines from
        a single line that happened to be nested.
        """
        if not isinstance(candidate, (list, tuple)) or not candidate:
            return False
        first = candidate[0]
        if not isinstance(first, (list, tuple)) or len(first) < 2:
            return False
        quad = first[0]
        # The quad is a sequence of vertices; each vertex is an (x, y) pair.
        return (
            isinstance(quad, (list, tuple))
            and len(quad) >= 1
            and isinstance(quad[0], (list, tuple))
        )

    @staticmethod
    def _parse_line(line) -> tuple[object, str, float]:
        """Extract ``(quad, text, confidence)`` from one PaddleOCR line entry.

        A line is ``[quad, (text, confidence)]``. Some variants store the
        recognition result as ``[text, confidence]`` instead of a tuple; both are
        handled.
        """
        if not isinstance(line, (list, tuple)) or len(line) < 2:
            raise OCRProcessingError(f"Malformed PaddleOCR line: {line!r}")
        quad = line[0]
        rec = line[1]
        if not isinstance(rec, (list, tuple)) or len(rec) < 2:
            raise OCRProcessingError(
                f"Malformed PaddleOCR recognition entry: {rec!r}"
            )
        text = rec[0]
        confidence = rec[1]
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        return quad, text, confidence

    @staticmethod
    def _clamp_confidence(confidence: float) -> float:
        """Clamp ``confidence`` into [0.0, 1.0], mapping NaN to 0.0."""
        value = float(confidence)
        if value != value:  # NaN
            return 0.0
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value
