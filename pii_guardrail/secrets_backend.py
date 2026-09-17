"""Optional credential/secret classifier backed by detect-secrets (Yelp).

Wraps `detect-secrets <https://github.com/Yelp/detect-secrets>`_ so the
:class:`~pii_guardrail.detector.Detector` can flag credentials/keys/tokens more
thoroughly than the hand-written regex -- AWS/Azure/GitHub/GitLab keys, JWTs,
private keys, Slack/Stripe/Twilio/SendGrid tokens, etc.

Precision note: detect-secrets ships high-recall ENTROPY plugins (Base64/Hex
"High Entropy String") that false-positive heavily on ordinary OCR text (they
flag normal sentences). We therefore enable ONLY the specific-pattern plugins
and DROP the entropy ones, so a segment is flagged only when it matches a known
credential shape.

Contract (mirrors the other optional backends):
    - Lazy import: ``import pii_guardrail`` works without detect-secrets.
    - ADVISORY: unioned with the pattern + LLM results; patterns remain source
      of truth.
    - Fails safe: any error -> ``available=False`` or empty set.
    - Runs LOCALLY, tiny footprint (~30MB), no ML model.

Requirements: 4.1, 4.2, 4.3, 5.1, 5.2, 5.3.
"""

from __future__ import annotations

from pii_guardrail.errors import ClassifierUnavailableError
from pii_guardrail.models import SensitiveCategory

__all__ = ["SecretsClassifier"]


#: High-precision detect-secrets plugins (specific credential shapes). The
#: entropy plugins (Base64/Hex High Entropy String) are intentionally EXCLUDED
#: to avoid false positives on normal text.
_PLUGINS: tuple[dict[str, str], ...] = (
    {"name": "AWSKeyDetector"},
    {"name": "AzureStorageKeyDetector"},
    {"name": "GitHubTokenDetector"},
    {"name": "GitLabTokenDetector"},
    {"name": "JwtTokenDetector"},
    {"name": "PrivateKeyDetector"},
    {"name": "SlackDetector"},
    {"name": "StripeDetector"},
    {"name": "TwilioKeyDetector"},
    {"name": "SendGridDetector"},
    {"name": "NpmDetector"},
    {"name": "BasicAuthDetector"},
    {"name": "IbmCloudIamDetector"},
    {"name": "SquareOAuthDetector"},
    {"name": "ArtifactoryDetector"},
    {"name": "DiscordBotTokenDetector"},
    {"name": "MailchimpDetector"},
)

#: Map a detect-secrets ``.type`` label to our SensitiveCategory.
#: - JWT / OAuth-ish bearer tokens -> ACCESS_TOKEN
#: - private keys / basic-auth creds -> SECRET
#: - everything else (API/service keys) -> API_KEY
_TYPE_TO_CATEGORY: dict[str, SensitiveCategory] = {
    "JSON Web Token": SensitiveCategory.ACCESS_TOKEN,
    "Square OAuth Secret": SensitiveCategory.ACCESS_TOKEN,
    "Discord Bot Token": SensitiveCategory.ACCESS_TOKEN,
    "Private Key": SensitiveCategory.SECRET,
    "Basic Auth Credentials": SensitiveCategory.SECRET,
}


class SecretsClassifier:
    """A :class:`~pii_guardrail.classifier.ClassifierModel` backed by detect-secrets.

    Exposes ``available`` + ``classify`` (per-segment). Construct once and reuse.
    """

    #: Stable label identifying this layer in per-source debug breakdowns.
    source_name = "detect-secrets"

    def __init__(self) -> None:
        """Prepare the detect-secrets scanner; never raise on failure.

        On any import error, ``available`` is ``False`` and the Detector falls
        back to the other layers.
        """
        self._scan_line = None
        self._transient_settings = None
        self._load_error: Exception | None = None
        try:
            from detect_secrets.core.scan import scan_line  # type: ignore[import-not-found]
            from detect_secrets.settings import (  # type: ignore[import-not-found]
                transient_settings,
            )

            self._scan_line = scan_line
            self._transient_settings = transient_settings
        except Exception as exc:  # noqa: BLE001 - degrade gracefully, never raise
            self._load_error = exc

    @property
    def available(self) -> bool:
        """``True`` when detect-secrets imported successfully."""
        return self._scan_line is not None and self._transient_settings is not None

    def classify(self, text: str) -> set[SensitiveCategory]:
        """Return credential categories found in ``text``.

        Raises:
            ClassifierUnavailableError: If detect-secrets is unavailable, so the
                Detector falls back to the other layers.
        """
        if not self.available:
            raise ClassifierUnavailableError(
                "detect-secrets is not available."
            ) from self._load_error

        if not text or not text.strip():
            return set()

        try:
            with self._transient_settings({"plugins_used": list(_PLUGINS)}):
                secrets = list(self._scan_line(text))
        except Exception as exc:  # noqa: BLE001
            raise ClassifierUnavailableError(
                "detect-secrets scan failed."
            ) from exc

        categories: set[SensitiveCategory] = set()
        for secret in secrets:
            secret_type = getattr(secret, "type", None)
            if not isinstance(secret_type, str):
                continue
            # Default any unmapped detected credential to API_KEY (a key/token).
            categories.add(
                _TYPE_TO_CATEGORY.get(secret_type, SensitiveCategory.API_KEY)
            )
        return categories
