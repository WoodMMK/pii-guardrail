"""Geometry helpers for the PII guardrail core.

Pure, framework-agnostic geometry: no PaddleOCR, FastAPI, OpenCV, or NumPy
dependency, so this module imports cleanly wherever the core does.

The OCR engine (PaddleOCR family) returns text regions as quadrilaterals: four
[x, y] vertices in image pixel coordinates with a top-left origin (0, 0). The
rest of the pipeline works with axis-aligned ``BoundingBox`` rectangles, so this
module converts a quadrilateral into the enclosing axis-aligned box, clamped to
the image bounds.

Requirements: 2.2 (Property 3).
"""

from __future__ import annotations

from collections.abc import Sequence

from pii_guardrail.models import BoundingBox

# A single vertex: (x, y). A quad is a sequence of such vertices.
Point = Sequence[float]
Quad = Sequence[Point]


def normalize_quad_to_box(
    quad: Quad,
    image_width: int,
    image_height: int,
) -> BoundingBox:
    """Convert an OCR quadrilateral into an enclosing axis-aligned ``BoundingBox``.

    The result is the smallest axis-aligned rectangle that encloses every vertex
    of ``quad`` (``x = min xs``, ``y = min ys``, ``right = max xs``,
    ``bottom = max ys``), clamped so it lies fully inside an image of the given
    dimensions with a top-left origin at (0, 0).

    Guarantees (Property 3 / Requirement 2.2), for a positive image size::

        x >= 0, y >= 0, width > 0, height > 0,
        x + width <= image_width, y + height <= image_height,

    and the box contains every source vertex (up to the image clamp: vertices
    are assumed to lie inside the image, so clamping does not exclude them).

    Coordinates are rounded to integers to match the pixel-based ``BoundingBox``.
    Degenerate quads (a point or an axis-aligned line, or a box that collapses to
    zero extent after rounding) are widened to a minimum of 1px in each
    dimension while staying inside the image, since ``BoundingBox`` requires
    ``width > 0`` and ``height > 0``.

    Args:
        quad: An iterable of at least one ``(x, y)`` vertex.
        image_width: Image width in pixels; must be >= 1.
        image_height: Image height in pixels; must be >= 1.

    Returns:
        The enclosing, in-bounds ``BoundingBox``.

    Raises:
        ValueError: If ``quad`` has no vertices, a vertex is malformed, or the
            image dimensions are not positive.
    """
    if image_width <= 0:
        raise ValueError(f"image_width must be >= 1, got {image_width}")
    if image_height <= 0:
        raise ValueError(f"image_height must be >= 1, got {image_height}")

    xs: list[float] = []
    ys: list[float] = []
    for vertex in quad:
        # Each vertex must be an (x, y) pair.
        try:
            vx, vy = vertex[0], vertex[1]
        except (TypeError, IndexError) as exc:
            raise ValueError(f"quad vertex must be an (x, y) pair, got {vertex!r}") from exc
        xs.append(float(vx))
        ys.append(float(vy))

    if not xs:
        raise ValueError("quad must contain at least one vertex")

    # Enclosing rectangle of the raw (float) vertices.
    min_x = min(xs)
    min_y = min(ys)
    max_x = max(xs)
    max_y = max(ys)

    # Clamp the enclosing rectangle into the image, then round to pixels.
    # Rounding after clamping keeps every coordinate inside [0, image_*].
    left = int(round(_clamp(min_x, 0.0, float(image_width))))
    top = int(round(_clamp(min_y, 0.0, float(image_height))))
    right = int(round(_clamp(max_x, 0.0, float(image_width))))
    bottom = int(round(_clamp(max_y, 0.0, float(image_height))))

    # Ensure a strictly positive extent. If the box collapsed (a point/line, or
    # rounding erased a sub-pixel extent), widen by 1px, preferring to grow the
    # far edge and falling back to shifting the near edge when we are flush
    # against the image's right/bottom boundary.
    if right <= left:
        if left + 1 <= image_width:
            right = left + 1
        else:
            left = image_width - 1
            right = image_width
    if bottom <= top:
        if top + 1 <= image_height:
            bottom = top + 1
        else:
            top = image_height - 1
            bottom = image_height

    return BoundingBox(x=left, y=top, width=right - left, height=bottom - top)


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into the inclusive range [``low``, ``high``]."""
    if value < low:
        return low
    if value > high:
        return high
    return value
