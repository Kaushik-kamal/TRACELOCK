"""Platform adapters and the honest capability matrix.

EVERY piece of per-platform knowledge lives in this file. The resolver, the
runner, the API and the UI all branch on adapter OUTPUT, never on the platform
itself. That is the rule that stops five parallel pipelines from growing.

WHAT AN ADAPTER MAY DO
----------------------
Read metadata a site voluntarily serves to an anonymous visitor -- the same
OpenGraph and oEmbed tags every chat app's link preview consumes -- and rewrite
a URL into another PUBLIC url the same site already serves (a share link into
its canonical form, a Drive share link into its published thumbnail endpoint).

WHAT AN ADAPTER MAY NOT DO
--------------------------
Authenticate, carry cookies or tokens, impersonate a signed-in session, solve a
CAPTCHA, hit a private/internal API, or work around a platform that has said
no. A refusal is a RESULT, reported honestly -- never an obstacle to route
around. `Support.expectation` is what we tell the operator BEFORE we try, so a
failure is never a surprise and never a lie.

Nothing here bypasses `fetch_media`, so the SSRF guard, redirect cap, size cap
and timeout apply to every request an adapter causes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any
from urllib.parse import parse_qs, urlsplit

from tracelock.ingest.classify import Platform


class Support(str, Enum):
    """How likely anonymous public resolution is -- stated up front."""

    RELIABLE = "reliable"       # public metadata, works consistently
    BEST_EFFORT = "best_effort"  # sometimes public, often not
    AUTH_REQUIRED = "auth_required"  # normally needs sign-in; we do not sign in

    @property
    def expectation(self) -> str:
        return {
            Support.RELIABLE: "Public image expected to resolve.",
            Support.BEST_EFFORT: "May resolve, depending on the post's privacy.",
            Support.AUTH_REQUIRED: (
                "Usually requires sign-in. TRACELOCK does not authenticate, so "
                "this often cannot be resolved."
            ),
        }[self]

    @property
    def badge(self) -> str:
        return {
            Support.RELIABLE: "supported",
            Support.BEST_EFFORT: "best effort",
            Support.AUTH_REQUIRED: "usually blocked",
        }[self]


@dataclass(frozen=True, slots=True)
class PlatformAdapter:
    """Declared capability plus optional public-URL rewriting."""

    platform: Platform
    support: Support
    note: str
    guidance: str = ""

    def rewrite(self, url: str) -> str:
        """Rewrite to an equivalent PUBLIC url the same site already serves.

        Default is identity. Overrides live in `_REWRITERS` so this stays a
        frozen data class rather than a class hierarchy.
        """
        rewriter = _REWRITERS.get(self.platform)
        return rewriter(url) if rewriter else url

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform.value,
            "display": self.platform.display,
            "icon": self.platform.icon,
            "support": self.support.value,
            "badge": self.support.badge,
            "expectation": self.support.expectation,
            "note": self.note,
            "guidance": self.guidance,
        }


# --------------------------------------------------------------------------
# URL rewriters -- public form to public form, no credentials involved
# --------------------------------------------------------------------------

_DRIVE_ID = (
    re.compile(r"/file/d/([A-Za-z0-9_-]{10,})"),
    re.compile(r"/d/([A-Za-z0-9_-]{10,})"),
)


def _rewrite_drive(url: str) -> str:
    """Drive share link -> the public thumbnail endpoint.

    Only works when the file is already shared publicly ("anyone with the
    link"). A private file returns an HTML sign-in page, which the sniffer
    rejects and the operator is told about. No credential is involved either
    way -- this is the same endpoint a browser hits for a public preview.
    """
    parts = urlsplit(url)
    file_id = ""

    for pattern in _DRIVE_ID:
        found = pattern.search(parts.path)
        if found:
            file_id = found.group(1)
            break

    if not file_id:
        file_id = (parse_qs(parts.query).get("id") or [""])[0]

    if not file_id:
        return url
    return "https://drive.google.com/thumbnail?id={0}&sz=w2000".format(file_id)


def _rewrite_x(url: str) -> str:
    """twitter.com -> x.com, and drop tracking query parameters.

    Canonicalisation only: same public resource, fewer trackers.
    """
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if host.endswith("twitter.com"):
        return "https://x.com{0}".format(parts.path)
    if parts.query:
        return "https://{0}{1}".format(host, parts.path)
    return url


def _rewrite_reddit(url: str) -> str:
    """Strip Reddit tracking params; the bare permalink is public."""
    parts = urlsplit(url)
    return "https://{0}{1}".format((parts.hostname or "").lower(), parts.path)


_REWRITERS = {
    Platform.GOOGLE_DRIVE: _rewrite_drive,
    Platform.X_TWITTER: _rewrite_x,
    Platform.REDDIT: _rewrite_reddit,
}


# --------------------------------------------------------------------------
# The capability matrix
# --------------------------------------------------------------------------
#
# These ratings describe what anonymous access ACTUALLY yields as observed,
# not what would be convenient for a demo. A platform rated AUTH_REQUIRED is
# still ACCEPTED as input and still attempted -- we simply say beforehand that
# it will probably not work, so a refusal is expected rather than a failure.

ADAPTERS: dict[Platform, PlatformAdapter] = {
    Platform.GENERIC: PlatformAdapter(
        Platform.GENERIC, Support.RELIABLE,
        "Reads the page's public preview metadata (og:image, twitter:image, "
        "schema.org) and any in-page images.",
    ),
    Platform.GITHUB: PlatformAdapter(
        Platform.GITHUB, Support.RELIABLE,
        "GitHub serves avatars and raw content publicly.",
    ),
    Platform.IMGUR: PlatformAdapter(
        Platform.IMGUR, Support.RELIABLE,
        "Imgur publishes OpenGraph metadata for public posts.",
    ),
    Platform.YOUTUBE: PlatformAdapter(
        Platform.YOUTUBE, Support.RELIABLE,
        "Uses YouTube's public oEmbed endpoint, which needs no access token.",
        guidance="Resolves to the video thumbnail, which may not contain a face.",
    ),
    Platform.PINTEREST: PlatformAdapter(
        Platform.PINTEREST, Support.BEST_EFFORT,
        "Public pins usually publish OpenGraph metadata.",
    ),
    Platform.REDDIT: PlatformAdapter(
        Platform.REDDIT, Support.BEST_EFFORT,
        "Public posts usually publish a preview image.",
    ),
    Platform.GOOGLE_DRIVE: PlatformAdapter(
        Platform.GOOGLE_DRIVE, Support.BEST_EFFORT,
        "Works only when the file is shared with 'anyone with the link'.",
        guidance="Set the file to 'Anyone with the link' before pasting it.",
    ),
    Platform.X_TWITTER: PlatformAdapter(
        Platform.X_TWITTER, Support.AUTH_REQUIRED,
        "X restricts anonymous access to most post content.",
        guidance="Open the image directly and paste its pbs.twimg.com URL instead.",
    ),
    Platform.INSTAGRAM: PlatformAdapter(
        Platform.INSTAGRAM, Support.AUTH_REQUIRED,
        "Instagram requires authentication for most post pages, and its oEmbed "
        "endpoint needs an app token TRACELOCK does not use.",
        guidance="Save the image and upload it, or paste a direct image URL.",
    ),
    Platform.FACEBOOK: PlatformAdapter(
        Platform.FACEBOOK, Support.AUTH_REQUIRED,
        "Almost all Facebook content requires a signed-in session.",
        guidance="Save the image and upload it instead.",
    ),
    Platform.LINKEDIN: PlatformAdapter(
        Platform.LINKEDIN, Support.AUTH_REQUIRED,
        "LinkedIn blocks anonymous access to most profile and post content.",
        guidance="Save the image and upload it instead.",
    ),
}


def adapter_for(platform: Platform) -> PlatformAdapter:
    return ADAPTERS.get(platform, ADAPTERS[Platform.GENERIC])


def capability_matrix() -> list[dict[str, Any]]:
    """The full support table, for the UI and `/api/resolver/info`.

    Ordered most-capable first so the UI can show what works before what does
    not, without re-sorting.
    """
    order = {Support.RELIABLE: 0, Support.BEST_EFFORT: 1, Support.AUTH_REQUIRED: 2}
    rows = sorted(ADAPTERS.values(), key=lambda a: (order[a.support], a.platform.display))
    return [adapter.to_dict() for adapter in rows]
