"""Core data models for the PII guardrail.

All models here are framework-agnostic plain dataclasses with NO web dependency
and NO dependency on PaddleOCR/FastAPI. See the design document's "Data Models"
section for the authoritative definitions.

Requirements: 7.1, 7.2, 7.3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SensitiveCategory(str, Enum):
    """The categories of sensitive information the guardrail can detect.

    Inherits from ``str`` so members serialize directly to their string value
    (e.g. ``SensitiveCategory.EMAIL == "email"``).
    """

    PERSON_NAME = "person_name"
    ORGANIZATION_NAME = "organization_name"
    PHONE_NUMBER = "phone_number"
    URL = "url"
    EMAIL = "email"
    MONEY_AMOUNT = "money_amount"
    PERCENT_VALUE = "percent_value"
    API_KEY = "api_key"
    ACCESS_TOKEN = "access_token"
    SECRET = "secret"

    # --- Thailand-specific personal identifiers ---
    # These categories cover PII that commonly appears on Thai documents such as
    # national ID cards, bank slips, and vehicle registrations. Several are
    # verifiable by structure (e.g. the national ID's mod-11 checksum), keeping
    # false positives low.
    THAI_NATIONAL_ID = "thai_national_id"  # 13-digit citizen ID (checksum-validated)
    DATE_OF_BIRTH = "date_of_birth"  # Thai/Latin dates (e.g. 18 ต.ค. 2546 / 18 Oct 2003)
    THAI_ADDRESS = "thai_address"  # Thai postal address / postal code
    BANK_ACCOUNT = "bank_account"  # Thai bank account number
    LICENSE_PLATE = "license_plate"  # Thai vehicle registration plate
    PASSPORT_NUMBER = "passport_number"  # passport number (1-2 letters + 6-7 digits)
    JOB_TITLE = "job_title"  # job title / position (e.g. director, ผู้อำนวยการ)

    # --- International structured identifiers (detected by Presidio patterns) ---
    # These have well-defined formats (and often checksums) that regex/Presidio
    # recognize reliably without any NER model.
    CREDIT_CARD = "credit_card"  # 13-19 digit card number (Luhn-validated)
    IP_ADDRESS = "ip_address"  # IPv4 / IPv6 address
    IBAN = "iban"  # International Bank Account Number
    CRYPTO_WALLET = "crypto_wallet"  # e.g. Bitcoin wallet address


@dataclass(frozen=True)
class BoundingBox:
    """An axis-aligned rectangle in pixel coordinates, top-left origin (0, 0).

    Frozen (immutable and hashable) so boxes can be safely shared between a
    ``TextSegment`` and any ``SensitiveRegion`` derived from it.

    Invariants (enforced in ``__post_init__``):
        x >= 0, y >= 0, width > 0, height > 0.

    The additional in-image bound (x + width <= image_width, y + height <=
    image_height) is enforced by the normalization step (task 2.1) that has
    access to the image dimensions, not here.
    """

    x: int  # left, pixels from top-left origin
    y: int  # top, pixels from top-left origin
    width: int  # > 0
    height: int  # > 0

    def __post_init__(self) -> None:
        if self.x < 0:
            raise ValueError(f"BoundingBox.x must be >= 0, got {self.x}")
        if self.y < 0:
            raise ValueError(f"BoundingBox.y must be >= 0, got {self.y}")
        if self.width <= 0:
            raise ValueError(f"BoundingBox.width must be > 0, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"BoundingBox.height must be > 0, got {self.height}")


@dataclass
class TextSegment:
    """One recognized text segment produced by the OCR engine."""

    text: str
    box: BoundingBox
    confidence: float  # 0.0 <= confidence <= 1.0


@dataclass
class SensitiveRegion:
    """A text region classified as sensitive, with its matching categories.

    ``categories`` is non-empty for a genuine sensitive region.
    """

    box: BoundingBox
    categories: set[SensitiveCategory]  # non-empty for a sensitive region
    text: str  # source text (for verification/debug)
    confidence: float  # OCR confidence of the source segment


@dataclass
class DetectionResult:
    """The complete set of sensitive regions detected in one image."""

    regions: list[SensitiveRegion] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0

    @property
    def count(self) -> int:
        """Number of sensitive regions in this result."""
        return len(self.regions)


# ---------------------------------------------------------------------------
# Serialization (framework-agnostic, JSON-compatible)
#
# These functions map a ``DetectionResult`` to and from a plain dict of only
# JSON-native types (str, int, float, list, dict). They live here (rather than
# in the web layer) so the core stays self-contained and a round-trip is
# faithful: ``deserialize_detection_result(serialize_detection_result(x))``
# reproduces an equivalent ``DetectionResult`` (same regions, boxes, category
# sets, dimensions, text, and confidence). See design.md's Detection_Result
# contract. Requirements: 7.4.
#
# Note on ``categories``: they serialize as a deterministically sorted list of
# the enum string VALUES (``SensitiveCategory`` is a ``str`` Enum) so output is
# stable across runs and comparisons/round-trips are order-independent. On
# deserialize they are rebuilt into a ``set[SensitiveCategory]``.
#
# Note on ``count``: it is a derived property (``len(regions)``). It is emitted
# on serialize for the machine-readable contract but is NOT required on
# deserialize; a present ``count`` is tolerated and ignored (regions are the
# source of truth).
# ---------------------------------------------------------------------------


def serialize_bounding_box(box: BoundingBox) -> dict:
    """Serialize a :class:`BoundingBox` to ``{x, y, width, height}``."""
    return {
        "x": int(box.x),
        "y": int(box.y),
        "width": int(box.width),
        "height": int(box.height),
    }


def deserialize_bounding_box(data: dict) -> BoundingBox:
    """Reconstruct a :class:`BoundingBox` from a ``{x, y, width, height}`` dict.

    Invariants (x >= 0, y >= 0, width > 0, height > 0) are enforced by
    ``BoundingBox.__post_init__``.
    """
    return BoundingBox(
        x=int(data["x"]),
        y=int(data["y"]),
        width=int(data["width"]),
        height=int(data["height"]),
    )


def serialize_sensitive_region(region: SensitiveRegion) -> dict:
    """Serialize one :class:`SensitiveRegion` to the structured dict shape.

    ``categories`` are emitted as a sorted list of their string values so the
    output is deterministic. ``text`` and ``confidence`` are preserved so a
    round-trip is faithful.
    """
    return {
        "box": serialize_bounding_box(region.box),
        "categories": sorted(category.value for category in region.categories),
        "text": region.text,
        "confidence": float(region.confidence),
    }


def deserialize_sensitive_region(data: dict) -> SensitiveRegion:
    """Reconstruct a :class:`SensitiveRegion` from its structured dict shape.

    ``categories`` are mapped back to a ``set[SensitiveCategory]`` via the enum
    constructor.
    """
    return SensitiveRegion(
        box=deserialize_bounding_box(data["box"]),
        categories={SensitiveCategory(value) for value in data["categories"]},
        text=data["text"],
        confidence=float(data["confidence"]),
    )


def serialize_detection_result(result: DetectionResult) -> dict:
    """Serialize a :class:`DetectionResult` to a JSON-compatible dict.

    Shape::

        {
          "image_width": int,
          "image_height": int,
          "count": int,
          "regions": [ {"box": {...}, "categories": [...],
                        "text": str, "confidence": float}, ... ]
        }

    ``count`` is derived from ``len(regions)`` and included for the
    machine-readable contract. The empty-regions case yields ``regions: []``
    and ``count: 0``.
    """
    return {
        "image_width": int(result.image_width),
        "image_height": int(result.image_height),
        "count": int(result.count),
        "regions": [serialize_sensitive_region(region) for region in result.regions],
    }


def deserialize_detection_result(data: dict) -> DetectionResult:
    """Reconstruct a :class:`DetectionResult` from its serialized dict.

    Regions and image dimensions are rebuilt from ``data``. ``count`` is a
    derived property, so it is not read from ``data`` (a present ``count`` is
    tolerated and ignored). The empty-regions case is handled naturally.
    """
    return DetectionResult(
        regions=[
            deserialize_sensitive_region(region) for region in data.get("regions", [])
        ],
        image_width=int(data["image_width"]),
        image_height=int(data["image_height"]),
    )
