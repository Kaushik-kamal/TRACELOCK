"""The provider contract every image host implements.

Two rules shape this interface:

CAPABILITIES ARE DECLARED, NOT ASSUMED
    `supports_deletion` and `expires_after_seconds` are properties of the
    provider, surfaced to the operator BEFORE they consent. Catbox cannot
    delete an anonymous upload; ImgBB expires but does not offer a REST
    delete; Cloudinary genuinely destroys on request. Presenting those three
    as equivalent would make a consent screen a lie.

CLEANUP IS REPORTED, NEVER CLAIMED
    `delete()` returns whether it actually worked. A provider that cannot
    delete returns False rather than pretending, and the result carries that
    to the UI. "We deleted it" when we did not is exactly the kind of
    reassurance this system exists to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

# Refuse anything implausible as a probe photograph, before any network call.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MIN_UPLOAD_BYTES = 64


class HostingError(Exception):
    """Upload or deletion failed.

    Deliberately distinct from DiscoveryError: "the host rejected us" and "the
    search engine found nothing" are different failures needing different
    responses, and collapsing them would hide which one occurred.
    """


class Retention(str, Enum):
    """How long the host keeps the file, when it offers a choice."""

    ONE_HOUR = "1h"
    TWELVE_HOURS = "12h"
    ONE_DAY = "24h"
    THREE_DAYS = "72h"

    @property
    def seconds(self) -> int:
        return {
            Retention.ONE_HOUR: 3600,
            Retention.TWELVE_HOURS: 43200,
            Retention.ONE_DAY: 86400,
            Retention.THREE_DAYS: 259200,
        }[self]


@dataclass(frozen=True, slots=True)
class HostingResult:
    """A completed upload. Everything the evidence record needs, no secrets."""

    url: str
    provider: str
    asset_id: str = ""
    delete_token: str = field(default="", repr=False)
    expires_after_seconds: int | None = None
    deletion_supported: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Safe to write into an artifact and safe to send to the browser.

        `delete_token` is deliberately omitted: it is a credential for
        destroying the asset, and it has no place in evidence or in a page.
        """
        payload: dict[str, Any] = {
            "url": self.url,
            "provider": self.provider,
            "deletion_supported": self.deletion_supported,
        }
        if self.asset_id:
            payload["asset_id"] = self.asset_id
        if self.expires_after_seconds is not None:
            payload["expires_after_seconds"] = self.expires_after_seconds
        if self.note:
            payload["note"] = self.note
        return payload


@runtime_checkable
class ImageHostProvider(Protocol):
    """What a host must be able to answer for itself."""

    key: str
    display_name: str

    @property
    def configured(self) -> bool:
        """True when this provider can actually be used right now.

        A key-based provider with no key is NOT configured, and must say so
        before a consent screen offers it.
        """

    @property
    def supports_deletion(self) -> bool:
        """True only when `delete()` genuinely removes the asset."""

    @property
    def retention_note(self) -> str:
        """One sentence an operator reads before consenting."""

    def missing_configuration(self) -> tuple[str, ...]:
        """Environment variables this provider still needs."""

    def upload(self, data: bytes, filename: str, *, retention: Retention) -> HostingResult:
        """Publish `data` and return its public URL. Raises HostingError."""

    def delete(self, result: HostingResult) -> bool:
        """Best-effort removal. Returns whether it actually succeeded."""


def validate_payload(data: bytes) -> None:
    """Guard before any network call. Cheap, and keeps junk off third parties."""
    if not data:
        raise HostingError("refusing to upload an empty file")
    if len(data) < MIN_UPLOAD_BYTES:
        raise HostingError(
            "refusing to upload {0} bytes: too small to be a photograph".format(len(data))
        )
    if len(data) > MAX_UPLOAD_BYTES:
        raise HostingError(
            "file is {0} bytes, over the {1} byte cap".format(len(data), MAX_UPLOAD_BYTES)
        )

    from tracelock.acquisition.validation import sniff_format

    fmt = sniff_format(data[:64])
    if not fmt:
        raise HostingError(
            "refusing to upload: the bytes are not a recognised image format"
        )


def validate_returned_url(url: str, provider: str) -> str:
    """A host must return a usable public http(s) URL, not an error page."""
    from urllib.parse import urlsplit

    url = (url or "").strip()
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise HostingError(
            "{0} did not return a usable URL. Body was: {1!r}".format(
                provider, url[:200]
            )
        )
    return url
