"""Public URL resolver.

Turns a URL a person would actually paste -- a social post, an article, a
direct image link -- into a publicly reachable IMAGE url, using only metadata
the site already publishes to anonymous visitors.

    https://site.com/photo.jpg      -> direct
    https://news.site/article       -> og:image / twitter:image / schema.org
    https://x.com/user/status/123   -> public oEmbed thumbnail, where offered
    https://instagram.com/p/ABC/    -> whatever the page publicly exposes

WHAT THIS IS NOT
----------------
It is not a scraper built to defeat websites. It:

  * sends no credentials, cookies or tokens, and never signs in
  * solves no CAPTCHA and evades no bot detection
  * reads only metadata a site voluntarily serves to an anonymous request --
    the same tags every link-preview in every chat app consumes
  * treats a refusal as a refusal: if a platform declines to serve a public
    preview, that is reported honestly, never worked around

Instagram in particular increasingly requires authentication for post pages.
When that happens the resolver says so plainly rather than pretending.

SECURITY POSTURE
----------------
Every fetch -- the HTML page AND the resolved image -- goes through the same
hardened `fetch_media`: SSRF guard, redirect limit, size cap, timeouts. A page
whose `og:image` points at 169.254.169.254 is blocked exactly like a candidate
URL would be, because the resolved URL is re-validated rather than trusted.

There is deliberately NO separate validation path. The resolver's only job is
to produce a URL; Stage 2.5 then validates it as it validates everything else.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlsplit

# Byte cap for an HTML page. Metadata lives in <head>; anything beyond this is
# body content we do not need, and refusing to buffer it bounds the work.
MAX_HTML_BYTES = 3 * 1024 * 1024

DIRECT_IMAGE_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".avif",
)


class UrlType(str, Enum):
    DIRECT_IMAGE = "direct_image"
    SOCIAL_POST = "social_post"
    WEBPAGE = "webpage"


class Platform(str, Enum):
    INSTAGRAM = "instagram"
    X_TWITTER = "x"
    FACEBOOK = "facebook"
    LINKEDIN = "linkedin"
    YOUTUBE = "youtube"
    GENERIC = "generic"

    @property
    def label(self) -> str:
        return {
            Platform.INSTAGRAM: "Instagram post",
            Platform.X_TWITTER: "X post",
            Platform.FACEBOOK: "Facebook post",
            Platform.LINKEDIN: "LinkedIn post",
            Platform.YOUTUBE: "YouTube video",
            Platform.GENERIC: "Public webpage",
        }[self]

    @property
    def icon(self) -> str:
        return {
            Platform.INSTAGRAM: "📷",
            Platform.X_TWITTER: "𝕏",
            Platform.FACEBOOK: "f",
            Platform.LINKEDIN: "in",
            Platform.YOUTUBE: "▶",
            Platform.GENERIC: "🔗",
        }[self]


class ResolutionMethod(str, Enum):
    DIRECT = "direct"
    OPENGRAPH = "opengraph"
    TWITTER_CARD = "twitter_card"
    SCHEMA_ORG = "schema_org"
    OEMBED = "oembed"
    NONE = "none"


class ResolutionStatus(str, Enum):
    RESOLVED = "resolved"
    UNAVAILABLE = "unavailable"


# Host -> platform. Matched against the registrable-ish host suffix so
# www./m./mobile. prefixes and regional LinkedIn hosts all work.
_PLATFORM_HOSTS: tuple[tuple[str, Platform], ...] = (
    ("instagram.com", Platform.INSTAGRAM),
    ("instagr.am", Platform.INSTAGRAM),
    ("twitter.com", Platform.X_TWITTER),
    ("x.com", Platform.X_TWITTER),
    ("t.co", Platform.X_TWITTER),
    ("facebook.com", Platform.FACEBOOK),
    ("fb.watch", Platform.FACEBOOK),
    ("linkedin.com", Platform.LINKEDIN),
    ("lnkd.in", Platform.LINKEDIN),
    ("youtube.com", Platform.YOUTUBE),
    ("youtu.be", Platform.YOUTUBE),
)

# oEmbed endpoints that are PUBLIC and require no access token.
# Instagram and Facebook oEmbed now require an app token, so they are
# deliberately absent: using them would mean authenticating, which this
# resolver does not do.
_PUBLIC_OEMBED: dict[Platform, str] = {
    Platform.YOUTUBE: "https://www.youtube.com/oembed?format=json&url={0}",
}

# Why a given platform commonly fails. Used to give an honest, specific
# message instead of a generic "not an image".
_PLATFORM_FAILURE_NOTE: dict[Platform, str] = {
    Platform.INSTAGRAM: (
        "Instagram post detected, but its image could not be retrieved "
        "publicly by TRACELOCK. The post may require authentication, or the "
        "platform may restrict automated retrieval."
    ),
    Platform.FACEBOOK: (
        "Facebook post detected, but no public preview image was available. "
        "Most Facebook content requires a signed-in session, which TRACELOCK "
        "does not use."
    ),
    Platform.LINKEDIN: (
        "LinkedIn post detected, but no public preview image was available. "
        "LinkedIn restricts anonymous access to most post content."
    ),
    Platform.X_TWITTER: (
        "X post detected, but no public preview image was available. X limits "
        "anonymous access to post content."
    ),
    Platform.YOUTUBE: (
        "YouTube video detected, but no public thumbnail could be retrieved."
    ),
    Platform.GENERIC: (
        "The page loaded, but it does not publish a preview image "
        "(no og:image, twitter:image or schema.org image)."
    ),
}


@dataclass(frozen=True, slots=True)
class ResolvedUrl:
    """What the resolver did, and what it found. Never overstates."""

    input_url: str
    input_type: UrlType
    platform: Platform
    resolution_method: ResolutionMethod
    resolution_status: ResolutionStatus
    resolved_image_url: str | None = None
    reason: str = ""
    http_status: int | None = None
    page_title: str = ""

    @property
    def ok(self) -> bool:
        return (
            self.resolution_status is ResolutionStatus.RESOLVED
            and bool(self.resolved_image_url)
        )

    @property
    def provenance_type(self) -> str:
        """Platform-qualified input type for the provenance record.

        `classify()` returns the coarse UrlType (direct_image / social_post /
        webpage) because that is what drives the resolution branch. The
        provenance record is more specific -- "instagram_post" rather than
        "social_post" -- so a reader of the artifact can tell WHICH platform
        was involved without cross-referencing another field.
        """
        if (
            self.input_type is UrlType.SOCIAL_POST
            and self.platform is not Platform.GENERIC
        ):
            return "{0}_post".format(self.platform.value)
        return self.input_type.value

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "input_url": self.input_url,
            "input_type": self.provenance_type,
            "url_class": self.input_type.value,
            "platform": self.platform.value,
            "platform_label": self.platform.label,
            "platform_icon": self.platform.icon,
            "resolution_method": self.resolution_method.value,
            "resolution_status": self.resolution_status.value,
        }
        if self.ok:
            payload["resolved_image_url"] = self.resolved_image_url
        else:
            # Never emit a resolved_image_url we did not actually obtain.
            payload["reason"] = self.reason
        if self.http_status is not None:
            payload["http_status"] = self.http_status
        if self.page_title:
            payload["page_title"] = self.page_title
        return payload


# ==========================================================================
# Classification
# ==========================================================================


def classify(url: str) -> tuple[UrlType, Platform]:
    """Classify a URL by shape alone. No network access."""
    parts = urlsplit((url or "").strip())
    host = (parts.hostname or "").lower()
    path = (parts.path or "").lower()

    platform = Platform.GENERIC
    for suffix, known in _PLATFORM_HOSTS:
        if host == suffix or host.endswith("." + suffix):
            platform = known
            break

    # A direct-image extension wins regardless of host: a .jpg on a CDN is a
    # direct image even when the CDN belongs to a social platform.
    if any(path.endswith(extension) for extension in DIRECT_IMAGE_EXTENSIONS):
        return UrlType.DIRECT_IMAGE, platform

    if platform is not Platform.GENERIC:
        return UrlType.SOCIAL_POST, platform

    return UrlType.WEBPAGE, platform


# ==========================================================================
# HTML metadata extraction
# ==========================================================================


class _MetaExtractor(HTMLParser):
    """Pull preview metadata out of a page head.

    Stdlib parser rather than a third-party HTML library: the job is reading a
    handful of meta tags, and the project should not gain a dependency for it.
    Malformed markup is tolerated -- convert_charrefs handles entities and
    unknown tags are simply ignored.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.json_ld: list[str] = []
        self.title: str = ""
        self._in_title = False
        self._in_ld = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        attributes = {k.lower(): (v or "") for k, v in attrs}

        if tag == "meta":
            key = (attributes.get("property") or attributes.get("name") or "").lower()
            content = attributes.get("content", "").strip()
            if key and content and key not in self.meta:
                self.meta[key] = content

        elif tag == "link":
            # <link rel="image_src"> is an older but still-used preview hint.
            if "image_src" in (attributes.get("rel") or "").lower():
                href = attributes.get("href", "").strip()
                if href:
                    self.meta.setdefault("link:image_src", href)

        elif tag == "title":
            self._in_title = True

        elif tag == "script":
            if "ld+json" in (attributes.get("type") or "").lower():
                self._in_ld = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag == "script":
            self._in_ld = False

    def handle_data(self, data: str) -> None:
        if self._in_title and not self.title:
            self.title = data.strip()[:200]
        elif self._in_ld:
            self.json_ld.append(data)


def _schema_org_image(blocks: list[str]) -> str | None:
    """First usable image URL from any JSON-LD block."""
    for block in blocks:
        try:
            data = json.loads(block)
        except (json.JSONDecodeError, ValueError):
            continue

        for node in data if isinstance(data, list) else [data]:
            if not isinstance(node, dict):
                continue
            found = _image_from_node(node)
            if found:
                return found
    return None


def _image_from_node(node: dict) -> str | None:
    """schema.org `image` is polymorphic: string, object, or list of either."""
    image = node.get("image") or node.get("thumbnailUrl")

    if isinstance(image, str) and image.strip():
        return image.strip()
    if isinstance(image, dict):
        url = image.get("url") or image.get("contentUrl")
        if isinstance(url, str) and url.strip():
            return url.strip()
    if isinstance(image, list):
        for entry in image:
            if isinstance(entry, str) and entry.strip():
                return entry.strip()
            if isinstance(entry, dict):
                url = entry.get("url") or entry.get("contentUrl")
                if isinstance(url, str) and url.strip():
                    return url.strip()

    # Some pages nest the useful node under @graph.
    graph = node.get("@graph")
    if isinstance(graph, list):
        for entry in graph:
            if isinstance(entry, dict):
                found = _image_from_node(entry)
                if found:
                    return found
    return None


# Preview-metadata keys, in the order the brief specifies.
_META_ORDER: tuple[tuple[str, ResolutionMethod], ...] = (
    ("og:image:secure_url", ResolutionMethod.OPENGRAPH),
    ("og:image:url", ResolutionMethod.OPENGRAPH),
    ("og:image", ResolutionMethod.OPENGRAPH),
    ("twitter:image:src", ResolutionMethod.TWITTER_CARD),
    ("twitter:image", ResolutionMethod.TWITTER_CARD),
    ("link:image_src", ResolutionMethod.SCHEMA_ORG),
)


def extract_image_from_html(
    html: str, base_url: str
) -> tuple[str | None, ResolutionMethod, str]:
    """Find a preview image in page HTML. Returns (url, method, page_title)."""
    parser = _MetaExtractor()
    try:
        parser.feed(html)
    except Exception:  # malformed markup must not raise past here
        pass

    for key, method in _META_ORDER:
        value = parser.meta.get(key)
        if value:
            return urljoin(base_url, value), method, parser.title

    schema_image = _schema_org_image(parser.json_ld)
    if schema_image:
        return urljoin(base_url, schema_image), ResolutionMethod.SCHEMA_ORG, parser.title

    return None, ResolutionMethod.NONE, parser.title


# ==========================================================================
# Resolution
# ==========================================================================


def _unavailable(
    url: str, url_type: UrlType, platform: Platform, reason: str,
    *, status: int | None = None,
    method: ResolutionMethod = ResolutionMethod.NONE,
) -> ResolvedUrl:
    return ResolvedUrl(
        input_url=url,
        input_type=url_type,
        platform=platform,
        resolution_method=method,
        resolution_status=ResolutionStatus.UNAVAILABLE,
        reason=reason,
        http_status=status,
    )


def resolve_public_url(url: str, *, fetch_policy=None) -> ResolvedUrl:
    """Resolve any public URL to an image URL, or explain why it could not be.

    Never raises for an unreachable or hostile URL: the failure is the result.
    """
    from tracelock.acquisition.fetcher import FetchPolicy, fetch_media
    from tracelock.acquisition.validation import sniff_format

    url = (url or "").strip()
    url_type, platform = classify(url)

    if not url:
        return _unavailable(url, url_type, platform, "No link was provided.")

    policy = fetch_policy or FetchPolicy(max_bytes=MAX_HTML_BYTES)

    # --- direct image: nothing to resolve --------------------------------
    if url_type is UrlType.DIRECT_IMAGE:
        return ResolvedUrl(
            input_url=url,
            input_type=UrlType.DIRECT_IMAGE,
            platform=platform,
            resolution_method=ResolutionMethod.DIRECT,
            resolution_status=ResolutionStatus.RESOLVED,
            resolved_image_url=url,
        )

    # --- fetch the page --------------------------------------------------
    # Same hardened fetcher as everything else: SSRF guard, redirect limit,
    # size cap, timeout. A page is untrusted input like any other.
    response = fetch_media(url, policy=policy)

    if not response.ok:
        reason = _fetch_failure_reason(response, platform)
        return _unavailable(
            url, url_type, platform, reason, status=response.status_code
        )

    body = response.content or b""

    # A URL with no image extension may still SERVE an image. Sniff before
    # assuming it is a page -- extensions are a hint, bytes are the truth.
    if sniff_format(body):
        return ResolvedUrl(
            input_url=url,
            input_type=UrlType.DIRECT_IMAGE,
            platform=platform,
            resolution_method=ResolutionMethod.DIRECT,
            resolution_status=ResolutionStatus.RESOLVED,
            resolved_image_url=response.final_url or url,
            http_status=response.status_code,
        )

    html = body.decode("utf-8", errors="replace")
    base = response.final_url or url

    image_url, method, title = extract_image_from_html(html, base)

    if image_url:
        return ResolvedUrl(
            input_url=url,
            input_type=url_type,
            platform=platform,
            resolution_method=method,
            resolution_status=ResolutionStatus.RESOLVED,
            resolved_image_url=image_url,
            http_status=response.status_code,
            page_title=title,
        )

    # --- public oEmbed, where the platform offers it without a token -----
    oembed_url = _try_public_oembed(url, platform, policy)
    if oembed_url:
        return ResolvedUrl(
            input_url=url,
            input_type=url_type,
            platform=platform,
            resolution_method=ResolutionMethod.OEMBED,
            resolution_status=ResolutionStatus.RESOLVED,
            resolved_image_url=oembed_url,
            http_status=response.status_code,
            page_title=title,
        )

    return _unavailable(
        url, url_type, platform,
        _PLATFORM_FAILURE_NOTE.get(platform, _PLATFORM_FAILURE_NOTE[Platform.GENERIC]),
        status=response.status_code,
    )


def _fetch_failure_reason(response, platform: Platform) -> str:
    """Explain a page fetch failure in platform-aware terms."""
    status = response.status_code
    reason = response.reason.value if response.reason else ""

    if status in (401, 403):
        if platform is not Platform.GENERIC:
            return _PLATFORM_FAILURE_NOTE[platform]
        return (
            "That site refused an anonymous request (HTTP {0}). It may require "
            "sign-in or block automated access.".format(status)
        )
    if status == 404:
        return "That page was not found (HTTP 404). The link may be wrong or deleted."
    if reason == "DOWNLOAD_TIMEOUT":
        return "That page did not respond in time."
    if reason == "BLOCKED_URL_TARGET":
        return "That link points to a private network address and was blocked."
    if reason == "INVALID_URL":
        return "That does not look like a valid web link."
    if reason == "CONTENT_TOO_LARGE":
        return "That page is too large to inspect."
    if status:
        return "That page returned HTTP {0}.".format(status)
    return "That page could not be reached."


def _try_public_oembed(url: str, platform: Platform, policy) -> str | None:
    """Ask a platform's PUBLIC oEmbed endpoint for a thumbnail.

    Only endpoints that work without an access token are listed. Instagram and
    Facebook oEmbed now require an app token; using them would mean
    authenticating, which this resolver does not do.
    """
    from urllib.parse import quote

    from tracelock.acquisition.fetcher import fetch_media

    template = _PUBLIC_OEMBED.get(platform)
    if not template:
        return None

    response = fetch_media(template.format(quote(url, safe="")), policy=policy)
    if not response.ok:
        return None

    try:
        payload = json.loads((response.content or b"").decode("utf-8", errors="replace"))
    except (json.JSONDecodeError, ValueError):
        return None

    thumbnail = payload.get("thumbnail_url")
    return thumbnail if isinstance(thumbnail, str) and thumbnail.strip() else None


SUPPORT_STATEMENT = (
    "TRACELOCK supports direct images and attempts to resolve publicly "
    "available images from supported social posts and web pages."
)
