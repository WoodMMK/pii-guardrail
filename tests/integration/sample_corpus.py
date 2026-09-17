"""Synthetic sample-corpus helpers for the task-15 integration tests.

These helpers GENERATE sample images at test time (no image fixtures are stored
in the repo) by rendering known Thai and Latin strings onto white backgrounds
with Pillow. The result is a BGR ``uint8`` NumPy array suitable for feeding to
PaddleOCR / OpenCV. A white background with black text works for detection
regardless of whether the backend treats the array as RGB or BGR, since the
text is achromatic.

Reused by:
    * ``test_paddle_extraction.py`` (task 15.1)
    * future task-15 tests (15.2 scanned-corpus, 15.3 recall benchmark)

Thai rendering requires a font that actually contains Thai glyphs. We probe a
list of candidate system font paths (Windows ships several Thai-capable fonts)
and use the first that exists. When NO Thai-capable font can be found, the Thai
sample generator returns ``None`` so callers can ``pytest.skip`` cleanly rather
than rendering tofu boxes and asserting on garbage.
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

__all__ = [
    "LATIN_SAMPLE_TEXT",
    "THAI_SAMPLE_TEXT",
    "find_thai_font_path",
    "find_latin_font_path",
    "render_text_image",
    "make_latin_sample",
    "make_thai_sample",
    "render_scanned_sample",
]

#: Default known strings rendered by the sample generators. The Latin sample
#: mixes letters and digits so tests can look for any alphanumeric token. The
#: Thai sample is a common greeting/phrase entirely within the Thai Unicode
#: block (U+0E00..U+0E7F).
LATIN_SAMPLE_TEXT = "Invoice 2024"
THAI_SAMPLE_TEXT = "สวัสดีครับ"

#: Candidate Thai-capable font files. Windows commonly ships Tahoma, Leelawadee
#: UI, and Angsana/Cordia families, all of which include Thai glyphs. The list
#: also includes common Linux locations so the corpus is portable.
_THAI_FONT_CANDIDATES: tuple[str, ...] = (
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\LeelaUIb.ttf",
    r"C:\Windows\Fonts\LeelawUI.ttf",
    r"C:\Windows\Fonts\leelawui.ttf",
    r"C:\Windows\Fonts\upcdl.ttf",
    r"C:\Windows\Fonts\cordia.ttf",
    r"C:\Windows\Fonts\angsa.ttf",
    r"C:\Windows\Fonts\THSarabun.ttf",
    r"C:\Windows\Fonts\Norasi.ttf",
    "/usr/share/fonts/truetype/tlwg/Norasi.ttf",
    "/usr/share/fonts/truetype/tlwg/Loma.ttf",
    "/usr/share/fonts/truetype/tlwg/Sawasdee.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansThai-Regular.ttf",
    "/Library/Fonts/Thonburi.ttf",
    "/System/Library/Fonts/Thonburi.ttf",
)

#: Candidate Latin fonts. If none are found we fall back to Pillow's built-in
#: bitmap font (small, but adequate for a couple of tokens on a large image).
_LATIN_FONT_CANDIDATES: tuple[str, ...] = (
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
)


def find_thai_font_path() -> str | None:
    """Return the first existing Thai-capable font path, or ``None``.

    When ``None`` is returned no system font with Thai glyphs was located, so
    Thai text cannot be rendered as real glyphs and Thai-specific assertions
    should be skipped by the caller.
    """
    for path in _THAI_FONT_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


def find_latin_font_path() -> str | None:
    """Return the first existing Latin TrueType font path, or ``None``.

    ``None`` means callers should fall back to :func:`PIL.ImageFont.load_default`.
    """
    for path in _LATIN_FONT_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


def _load_font(font_path: str | None, font_size: int) -> ImageFont.ImageFont:
    """Load a TrueType font at ``font_size``, or Pillow's default font.

    Falls back to the built-in bitmap font when no path is given or the file
    cannot be opened as a TrueType font.
    """
    if font_path is not None:
        try:
            return ImageFont.truetype(font_path, font_size)
        except OSError:
            pass
    return ImageFont.load_default()


def _measure(draw: "ImageDraw.ImageDraw", text: str, font) -> tuple[int, int]:
    """Return the (width, height) of ``text`` rendered with ``font``.

    Uses ``textbbox`` (Pillow >= 8) and falls back to ``textlength``/font metrics
    for older Pillow builds.
    """
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return int(right - left), int(bottom - top)
    except AttributeError:  # pragma: no cover - very old Pillow
        width = int(draw.textlength(text, font=font))
        # Approximate height from the font size where bbox is unavailable.
        return width, int(getattr(font, "size", 16) * 1.4)


def render_text_image(
    text: str,
    *,
    font_path: str | None = None,
    width: int | None = None,
    height: int = 200,
    font_size: int = 72,
    margin: int = 40,
) -> np.ndarray:
    """Render ``text`` as dark text on a white background, returning a BGR array.

    The image is deliberately rendered large (tall, big glyphs) so OCR has an
    easy time: black text on a white field. The returned array is a 3-channel
    ``uint8`` BGR image (OpenCV/PaddleOCR consume BGR/RGB ndarrays; an achromatic
    black-on-white image is identical under either interpretation).

    Args:
        text: The string to render.
        font_path: Optional path to a TrueType font. When ``None`` a Latin font
            is auto-located, falling back to Pillow's default bitmap font.
        width: Image width in pixels. When ``None`` it is sized to the rendered
            text plus margins.
        height: Image height in pixels.
        font_size: Glyph size in points.
        margin: Padding (pixels) around the text when auto-sizing width and for
            the text's top-left placement.

    Returns:
        A ``(height, width, 3)`` ``uint8`` BGR NumPy array.
    """
    resolved_font_path = font_path if font_path is not None else find_latin_font_path()
    font = _load_font(resolved_font_path, font_size)

    # Measure on a scratch image to size the canvas when width is not given.
    scratch = Image.new("RGB", (10, 10), color=(255, 255, 255))
    scratch_draw = ImageDraw.Draw(scratch)
    text_w, text_h = _measure(scratch_draw, text, font)

    if width is None:
        width = max(text_w + 2 * margin, 2 * margin + 1)
    height = max(height, text_h + 2 * margin, 1)

    image = Image.new("RGB", (int(width), int(height)), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    # Vertically center the text; left-align with a margin.
    y = max((height - text_h) // 2, 0)
    draw.text((margin, y), text, fill=(0, 0, 0), font=font)

    # PIL RGB -> NumPy RGB -> BGR (OpenCV convention). For black-on-white the
    # channel order does not affect detection, but we stay consistent with the
    # rest of the pipeline (OpenCV/BGR).
    rgb = np.asarray(image, dtype=np.uint8)
    bgr = rgb[:, :, ::-1].copy()
    return bgr


def make_latin_sample(text: str = LATIN_SAMPLE_TEXT) -> np.ndarray:
    """Return a rendered Latin sample image (BGR ``uint8``)."""
    return render_text_image(text, font_path=find_latin_font_path())


#: Minimum image HEIGHT (pixels) that renders as ">=150 DPI" under the
#: Preprocessor's DPI heuristic (estimated_dpi = height / ASSUMED_PAGE_HEIGHT_INCHES,
#: with ASSUMED_PAGE_HEIGHT_INCHES == 11.0 and DPI_RELIABILITY_THRESHOLD == 150.0).
#: 150 * 11 == 1650, so a height >= 1650 yields quality_sufficient == True.
_MIN_SCANNED_HEIGHT_PX = 150 * 11  # 1650


def render_scanned_sample(
    text: str,
    *,
    font_path: str | None = None,
    height: int = _MIN_SCANNED_HEIGHT_PX + 30,
    max_width: int = 1000,
    font_size: int = 90,
    margin: int = 40,
) -> np.ndarray:
    """Render ``text`` as a tall image that reads as a ">=150 DPI" scan.

    The Preprocessor estimates effective DPI as ``height / 11`` and marks quality
    sufficient at >= 150 DPI, i.e. height >= 1650 px. This helper renders at a
    height comfortably above that threshold so
    :meth:`GuardrailPipeline.process` runs full detection (Requirement 6.4).

    To keep the (denoise + OCR) runtime reasonable on such a tall image, the
    WIDTH is kept modest: it is auto-sized to the rendered text plus margins but
    capped at ``max_width``. Denoise cost scales with total pixels, so a
    tall-but-narrow image is cheaper than a tall-and-wide one.

    Args:
        text: The (sensitive) string to render, one per image so the ground-truth
            category present in the image is known.
        font_path: Optional TrueType font path (auto-located Latin font by default).
        height: Image height in pixels. Defaults just above the >=150 DPI threshold.
        max_width: Upper bound on the auto-sized width (keeps runtime down).
        font_size: Glyph size in points (large so text is legible at this scale).
        margin: Padding (pixels) around the text.

    Returns:
        A ``(height, width, 3)`` ``uint8`` BGR NumPy array with ``height`` >= 1650.
    """
    # Ensure the height clears the >=150 DPI heuristic even if a caller lowers it.
    effective_height = max(int(height), _MIN_SCANNED_HEIGHT_PX + 1)

    resolved_font_path = font_path if font_path is not None else find_latin_font_path()
    scratch_draw = ImageDraw.Draw(Image.new("RGB", (10, 10), color=(255, 255, 255)))

    # Shrink the glyph size until the whole string (plus margins) fits within
    # ``max_width``. Clipping the text would drop characters and defeat the
    # ground-truth label (e.g. "user@example.com" clipped to "user@examp" is no
    # longer a valid EMAIL), so we scale the font DOWN rather than crop.
    effective_font_size = int(font_size)
    while effective_font_size > 12:
        font = _load_font(resolved_font_path, effective_font_size)
        text_w, _text_h = _measure(scratch_draw, text, font)
        if text_w + 2 * margin <= int(max_width):
            break
        effective_font_size -= 6
    else:
        font = _load_font(resolved_font_path, effective_font_size)
        text_w, _text_h = _measure(scratch_draw, text, font)

    # Auto-size width to the (now-fitting) text, capped at max_width as a guard.
    width = min(max(text_w + 2 * margin, 2 * margin + 1), int(max_width))

    return render_text_image(
        text,
        font_path=resolved_font_path,
        width=width,
        height=effective_height,
        font_size=effective_font_size,
        margin=margin,
    )


def make_thai_sample(text: str = THAI_SAMPLE_TEXT) -> np.ndarray | None:
    """Return a rendered Thai sample image, or ``None`` if no Thai font exists.

    ``None`` signals the caller to ``pytest.skip`` because Thai glyphs cannot be
    rendered without a Thai-capable font.
    """
    thai_font = find_thai_font_path()
    if thai_font is None:
        return None
    return render_text_image(text, font_path=thai_font)
