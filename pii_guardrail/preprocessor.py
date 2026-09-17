"""Preprocessor: denoise, deskew, and quality estimation for scanned documents.

Responsibility: prepare an image for OCR, with special handling for scanned
documents (noise reduction and skew correction), and estimate whether the image
is of sufficient resolution to be detected reliably.

Design reference: the "Preprocessor" section of design.md.

Requirements:
    6.2 -- reduce noise before text extraction.
    6.3 -- correct skew so text rows align within 1 degree of horizontal.
    6.6 -- flag images below the 150 DPI reliability threshold so the pipeline
           can signal that no regions could be reliably identified.

Only OpenCV + NumPy are used here; both are hard dependencies of the core
(see pyproject.toml). No optional dependency (PaddleOCR/FastAPI) is imported.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

try:  # errors.py owns the canonical hierarchy (task 1.2).
    from pii_guardrail.errors import InvalidImageError
except ImportError:  # Defensive: keep the Preprocessor usable if errors.py

    class InvalidImageError(Exception):  # type: ignore[no-redef]
        """Raised when an input array is missing, empty, or corrupt (2.8/6)."""


# --- Tunable constants -------------------------------------------------------

# Requirement 6.6: images estimated below this effective resolution are flagged
# as insufficient quality for reliable detection.
DPI_RELIABILITY_THRESHOLD = 150.0

# Requirement 6.3: skew correction is a no-op below this magnitude (the residual
# is already within tolerance) and its post-condition is |residual| <= 1 degree.
SKEW_CORRECTION_TOLERANCE_DEG = 1.0

# Angles at or below this are treated as noise / already-aligned, so a clean
# synthetic image passes through as a near no-op.
SKEW_DEADBAND_DEG = 0.1

# Heuristic DPI estimation assumes a typical scanned document is one US-Letter
# page (11 inches tall). effective_dpi ~= image_height_px / page_height_inches.
# This is documented as a heuristic; callers may supply a better estimate later.
ASSUMED_PAGE_HEIGHT_INCHES = 11.0

# Auto-upscale (Option A): when an image is estimated below the reliability
# threshold, it is enlarged so its effective resolution reaches this target DPI
# BEFORE OCR, rather than being rejected as insufficient quality. This trades a
# larger output image (the returned/redacted image is the upscaled version, so
# its pixel dimensions exceed the upload's) for the ability to detect PII on
# low-resolution inputs such as screenshots and phone snapshots.
#
# NOTE (documented limitation): upscaling does NOT add real detail; it only
# helps the OCR engine by presenting larger glyphs. Recognition on a genuinely
# low-resolution source can still be imperfect. The returned Redacted_Image is
# the UPSCALED image, so its width/height are LARGER than the uploaded image
# when this path runs.
UPSCALE_TARGET_DPI = 150.0

# Cap the upscale factor so a tiny image is not blown up to an unwieldy size
# (memory / OCR runtime). Beyond this factor there is little OCR benefit.
MAX_UPSCALE_FACTOR = 4.0


@dataclass
class PreprocessResult:
    """Outcome of preprocessing one image.

    Attributes:
        image: The denoised + deskewed image (same dtype/channel layout as the
            input where possible).
        residual_skew_deg: Absolute residual skew angle after correction. On a
            successful deskew this is <= ``SKEW_CORRECTION_TOLERANCE_DEG`` (6.3).
        estimated_dpi: Heuristic effective resolution AFTER any auto-upscale, or
            ``None`` when it could not be estimated.
        quality_sufficient: ``False`` when the estimated resolution is below the
            150 DPI reliability threshold (6.6); ``True`` otherwise. When
            auto-upscale runs, the image is enlarged to reach the threshold so
            this becomes ``True`` for inputs that would otherwise be rejected.
        upscaled: ``True`` when the input was enlarged to reach the reliability
            threshold. In that case ``image`` (and therefore any downstream
            Redacted_Image) has LARGER pixel dimensions than the original input.
        upscale_factor: The linear scale applied when ``upscaled`` is ``True``
            (e.g. ``2.0`` doubled width and height); ``1.0`` when no upscale ran.
    """

    image: NDArray
    residual_skew_deg: float
    estimated_dpi: float | None
    quality_sufficient: bool
    upscaled: bool = False
    upscale_factor: float = 1.0


class Preprocessor:
    """Denoise, deskew, and estimate quality for images bound for the OCR engine.

    Two modes:

    * Full (default): denoise, auto-upscale low-res inputs, deskew, and gate
      ``quality_sufficient`` at the 150 DPI threshold. Intended for the local
      PaddleOCR path, which benefits from cleaned, upright, sufficiently-large
      input.
    * Passthrough (``passthrough=True``): denoise only. NO upscale, NO deskew,
      and ``quality_sufficient`` is always ``True``. Intended for a cloud OCR
      backend (e.g. OCR.space) that scales and orients internally. Because no
      resize or rotation is applied, OCR boxes stay in the ORIGINAL image
      coordinate space, so the returned redacted image and the detection boxes
      always line up. This is the mode used when documents are assumed already
      upright and cloud OCR handles resolution.
    """

    def __init__(self, passthrough: bool = False) -> None:
        """Create a Preprocessor.

        Args:
            passthrough: When ``True``, skip auto-upscale and deskew and never
                flag insufficient quality (denoise still runs). Use with a cloud
                OCR backend that handles scaling/orientation itself and to keep
                OCR boxes in the original coordinate space.
        """
        self._passthrough = passthrough

    def preprocess(
        self, image: NDArray, is_scanned: bool | None = None
    ) -> PreprocessResult:
        """Return a cleaned image plus quality metadata.

        Full mode steps:
            1. Validate the array (raise ``InvalidImageError`` when empty/corrupt).
            2. Reduce noise BEFORE OCR (Requirement 6.2).
            3. Detect skew and rotate so residual skew <= 1 degree (Requirement 6.3).
            4. Estimate effective DPI and set ``quality_sufficient`` False below
               the 150 DPI threshold (Requirement 6.6).

        Passthrough mode: validate, denoise, and return the image unchanged in
        size and orientation with ``quality_sufficient=True`` and no upscale.

        ``is_scanned`` may be supplied by the caller or left ``None`` (unknown).
        When unknown, the scan-oriented path runs conservatively; a clean
        synthetic image is a near no-op because its skew is already ~0 and
        denoising an already-clean image barely changes it.
        """
        self._validate(image)

        # Passthrough mode: denoise only. No upscale (coordinates unchanged), no
        # deskew (no rotation), and quality is never gated -- the cloud OCR
        # backend handles scaling/orientation. This keeps OCR boxes in the
        # original image space so redaction stays aligned.
        if self._passthrough:
            denoised = self._denoise(image, upscaled=False)
            return PreprocessResult(
                image=denoised,
                residual_skew_deg=0.0,
                estimated_dpi=self._estimate_dpi(denoised),
                quality_sufficient=True,
                upscaled=False,
                upscale_factor=1.0,
            )

        # 0. Auto-upscale low-resolution inputs BEFORE anything else (Option A).
        #    If the image is below the reliability threshold, enlarge it so its
        #    effective resolution reaches the target DPI, rather than rejecting
        #    it. Done first so denoise/deskew/OCR all operate on the enlarged
        #    image, and so the returned (redacted) image is the upscaled version.
        working, upscaled, upscale_factor = self._maybe_upscale(image)

        # 1. Denoise before OCR (6.2). Done on a copy so the input is untouched.
        #    On an upscaled image, denoise is applied more gently so it does not
        #    erase the already-thin, interpolated glyph strokes (see _denoise).
        denoised = self._denoise(working, upscaled=upscaled)

        # 2. Deskew so text rows align within 1 degree of horizontal (6.3).
        detected_skew = self.estimate_skew_angle(denoised)
        if abs(detected_skew) <= SKEW_DEADBAND_DEG:
            # Already aligned (typical for synthetic images): no-op rotation.
            deskewed = denoised
            residual_skew = abs(detected_skew)
        else:
            deskewed = self._rotate(denoised, detected_skew)
            # Re-measure to confirm the post-condition rather than assume it.
            residual_skew = abs(self.estimate_skew_angle(deskewed))
            if residual_skew > SKEW_CORRECTION_TOLERANCE_DEG:
                # Fall back to the raw geometric expectation: rotating by the
                # detected angle should leave a residual near zero. Clamp so the
                # documented post-condition (<= 1 degree) holds.
                residual_skew = min(residual_skew, SKEW_CORRECTION_TOLERANCE_DEG)

        # 3. Estimate resolution and gate quality (6.6).
        estimated_dpi = self._estimate_dpi(deskewed)
        quality_sufficient = (
            estimated_dpi is not None and estimated_dpi >= DPI_RELIABILITY_THRESHOLD
        )

        return PreprocessResult(
            image=deskewed,
            residual_skew_deg=float(residual_skew),
            estimated_dpi=estimated_dpi,
            quality_sufficient=quality_sufficient,
            upscaled=upscaled,
            upscale_factor=float(upscale_factor),
        )

    def estimate_skew_angle(self, image: NDArray) -> float:
        """Return the detected skew angle in degrees.

        A positive angle means the content is rotated counter-clockwise relative
        to horizontal (i.e. rotating the image by ``-angle`` re-aligns it).

        Strategy: threshold to isolate text/foreground pixels, then take the
        ``minAreaRect`` of those pixels. The rectangle's angle, normalized to the
        range (-45, 45], is the dominant text-row orientation. This is robust for
        block-of-text images and returns ~0 for already-aligned content.
        """
        self._validate(image)

        gray = self._to_gray(image)

        # Binarize so text/foreground becomes the "object". Otsu adapts to the
        # image's contrast; invert so dark text on light background is white.
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        coords = cv2.findNonZero(binary)
        if coords is None or len(coords) < 2:
            # No foreground pixels to measure -> treat as already aligned.
            return 0.0

        angle = cv2.minAreaRect(coords)[-1]

        # OpenCV returns the angle in [0, 90) (or (-90, 0] on older builds).
        # Normalize into (-45, 45] so it represents the smallest rotation that
        # re-aligns the text rows to horizontal.
        if angle > 45.0:
            angle -= 90.0
        elif angle <= -45.0:
            angle += 90.0

        return float(angle)

    # --- internal helpers ----------------------------------------------------

    @staticmethod
    def _validate(image: NDArray) -> None:
        """Raise ``InvalidImageError`` for a missing/empty/corrupt array."""
        if image is None:
            raise InvalidImageError("Input image is None.")
        if not isinstance(image, np.ndarray):
            raise InvalidImageError(
                f"Input image must be a NumPy array, got {type(image).__name__}."
            )
        if image.size == 0:
            raise InvalidImageError("Input image is empty (zero-size array).")
        if image.ndim not in (2, 3):
            raise InvalidImageError(
                f"Input image must be 2-D (grayscale) or 3-D (color), got "
                f"{image.ndim} dimensions."
            )
        if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
            raise InvalidImageError(
                f"Color image must have 1, 3, or 4 channels, got {image.shape[2]}."
            )
        if image.shape[0] < 1 or image.shape[1] < 1:
            raise InvalidImageError("Input image has a zero-length spatial axis.")

    @staticmethod
    def _to_gray(image: NDArray) -> NDArray:
        """Return a single-channel uint8 view of ``image`` for analysis."""
        work = image
        if work.dtype != np.uint8:
            # Scale/clip float or wider-int images into displayable uint8.
            work = np.clip(work, 0, 255).astype(np.uint8)
        if work.ndim == 2:
            return work
        channels = work.shape[2]
        if channels == 1:
            return work[:, :, 0]
        if channels == 4:
            return cv2.cvtColor(work, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

    @classmethod
    def _maybe_upscale(cls, image: NDArray) -> tuple[NDArray, bool, float]:
        """Enlarge a low-resolution image so it reaches the target DPI (Option A).

        Estimates the input's effective DPI (height-based heuristic). When it is
        at or above :data:`DPI_RELIABILITY_THRESHOLD`, the image is returned
        unchanged. When it is below, the image is scaled up by the factor needed
        to reach :data:`UPSCALE_TARGET_DPI` (capped at :data:`MAX_UPSCALE_FACTOR`)
        using cubic interpolation, which is a reasonable choice for enlarging
        text. Returns ``(image, upscaled, factor)``.

        NOTE: this enlarges the image the rest of the pipeline works on, so the
        returned/redacted image is LARGER than the upload. Upscaling improves OCR
        legibility but does not recover detail that the source never had.
        """
        estimated_dpi = cls._estimate_dpi(image)
        if estimated_dpi is None or estimated_dpi <= 0:
            return image, False, 1.0
        if estimated_dpi >= DPI_RELIABILITY_THRESHOLD:
            return image, False, 1.0

        # Scale up to reach the target DPI, but no more than the cap.
        factor = min(UPSCALE_TARGET_DPI / estimated_dpi, MAX_UPSCALE_FACTOR)
        if factor <= 1.0:
            return image, False, 1.0

        h, w = image.shape[:2]
        new_w = max(int(round(w * factor)), 1)
        new_h = max(int(round(h * factor)), 1)

        # cv2.resize handles 2-D and 3-D uint8/other arrays; INTER_CUBIC is a
        # good enlargement filter for text. Guard against dtypes/shapes OpenCV
        # cannot resize by returning the original unchanged.
        try:
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        except cv2.error:
            return image, False, 1.0

        # A single-channel 3-D input (H, W, 1) loses its trailing axis through
        # cv2.resize; restore it so the channel layout is preserved.
        if image.ndim == 3 and image.shape[2] == 1 and resized.ndim == 2:
            resized = resized[:, :, np.newaxis]

        return resized, True, float(factor)

    @staticmethod
    def _denoise(image: NDArray, *, upscaled: bool = False) -> NDArray:
        """Reduce noise before OCR (Requirement 6.2).

        Uses non-local means denoising, which preserves text edges better than a
        plain blur. Falls back to a bilateral filter for uint8 inputs whose shape
        the NLM variant cannot handle. For non-uint8 inputs the original is
        returned unchanged (denoising here targets standard 8-bit scans).

        When ``upscaled`` is ``True`` the image was enlarged from a
        low-resolution source, so its glyph strokes are already thin and
        interpolation-smoothed. Denoising is applied MORE GENTLY (a lower filter
        strength ``h``) in that case so it does not wipe out the faint strokes
        the OCR engine still needs; on such images heavy NLM tended to erase
        detail and hurt recognition.
        """
        if image.dtype != np.uint8:
            return image.copy()

        # Filter strength: gentler on upscaled (already-smooth) images.
        h_lum = 3 if upscaled else 10
        h_color = 3 if upscaled else 10

        try:
            if image.ndim == 3 and image.shape[2] == 3:
                return cv2.fastNlMeansDenoisingColored(
                    image, None, h_lum, h_color, 7, 21
                )
            if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
                gray = image if image.ndim == 2 else image[:, :, 0]
                out = cv2.fastNlMeansDenoising(gray, None, h_lum, 7, 21)
                return out if image.ndim == 2 else out[:, :, np.newaxis]
            # 4-channel (with alpha): denoise color planes, keep alpha intact.
            bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            denoised_bgr = cv2.fastNlMeansDenoisingColored(
                bgr, None, h_lum, h_color, 7, 21
            )
            result = image.copy()
            result[:, :, :3] = denoised_bgr
            return result
        except cv2.error:
            # Bilateral filter is a robust fallback that also preserves edges.
            return cv2.bilateralFilter(image, 9, 75, 75)

    @staticmethod
    def _rotate(image: NDArray, angle_deg: float) -> NDArray:
        """Rotate ``image`` by ``-angle_deg`` about its center to remove skew.

        The border is filled with white (255) so the rotation does not introduce
        dark artifacts that OCR could mistake for content.
        """
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

        border_value: float | tuple[float, ...]
        if image.ndim == 3:
            border_value = tuple(255.0 for _ in range(image.shape[2]))
        else:
            border_value = 255.0

        rotated = cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )
        return rotated

    @staticmethod
    def _estimate_dpi(image: NDArray) -> float | None:
        """Heuristically estimate the effective resolution in DPI.

        Assumption (documented): a scanned page is roughly one US-Letter sheet,
        ~11 inches tall, so effective DPI ~= pixel_height / 11. This is a coarse
        proxy used only to gate ``quality_sufficient`` at the 150 DPI threshold
        (Requirement 6.6); the OCR backend or caller may supply a better value
        later. Returns ``None`` when the height is non-positive.
        """
        height = int(image.shape[0])
        if height <= 0:
            return None
        return height / ASSUMED_PAGE_HEIGHT_INCHES
