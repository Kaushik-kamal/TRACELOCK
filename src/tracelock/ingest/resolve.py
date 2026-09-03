"""The universal resolver: any public URL in, a ranked image list out.

    direct image      -> itself, no page fetch
    webpage           -> preview metadata + ranked in-page images
    social post       -> adapter rewrite, then public metadata
    cloud share link  -> adapter rewrite to the public preview endpoint
    private / blocked -> an honest refusal that names the reason

SECURITY
--------
Both hops go through `fetch_media`: the ORIGINAL page URL and the RESOLVED
image URL. There is no path in this module that reaches the network any other
way, so the SSRF guard, redirect cap, byte cap and timeout cannot be skipped by
a page that points its `og:image` at an internal address. The resolver returns
a URL; it never hands downstream code bytes that dodged validation.

TRUTHFULNESS
------------
`ResolvedInput.ok` is false unless an image URL was genuinely obtained. When it
is false there is no `image_url` key at all -- structurally absent, not None,
so no downstream template can render a fabricated value. A platform that
refuses is reported as refusing, with what to do instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import quote

from tracelock.ingest.adapters import PlatformAdapter, Support, adapter_for
from tracelock.ingest.candidates import ImageCandidate, rank_candidates
from tracelock.ingest.classify import (
    Classification,
    InputType,
    Platform,
    classify_url,
    refine_with_response,
)

MAX_HTML_BYTES = 3 * 1024 * 1024

# oEmbed endpoints that are PUBLIC and need no access token. Instagram and
# Facebook oEmbed require an app token, so they are deliberately absent:
# using them would mean authenticating, which this resolver does not do.
_PUBLIC_OEMBED: dict[Platform, str] = {
    Platform.YOUTUBE: "https://www.youtube.com/oembed?format=json&url={0}",
}


class Method(str, Enum):
    DIRECT = "direct"
    SNIFFED = "sniffed"
    OPENGRAPH = "opengraph"
    TWITTER_CARD = "twitter_card"
    SCHEMA_ORG = "schema_org"
    IN_PAGE_IMAGE = "in_page_image"
    OEMBED = "oembed"
    ADAPTER_REWRITE = "adapter_rewrite"
    NONE = "none"

    @property
    def explanation(self) -> str:
        return {
            Method.DIRECT: "The link is already a direct image.",
            Method.SNIFFED: "The link returned image bytes directly.",
            Method.OPENGRAPH: "Read the page's public og:image preview tag.",
            Method.TWITTER_CARD: "Read the page's public twitter:image tag.",
            Method.SCHEMA_ORG: "Read the page's public schema.org image data.",
            Method.IN_PAGE_IMAGE: "Selected the highest-ranked image on the page.",
            Method.OEMBED: "Used the platform's public oEmbed endpoint.",
            Method.ADAPTER_REWRITE: "Rewrote the share link to its public preview URL.",
            Method.NONE: "No public image could be obtained.",
        }[self]


_SOURCE_TO_METHOD = {
    "og:image": Method.OPENGRAPH,
    "twitter:image": Method.TWITTER_CARD,
    "link:image_src": Method.SCHEMA_ORG,
    "schema.org": Method.SCHEMA_ORG,
    "img": Method.IN_PAGE_IMAGE,
}


@dataclass(frozen=True, slots=True)
class ResolvedInput:
    """What we found, how, and -- when we failed -- why, in plain words."""

    input_url: str
    classification: Classification
    adapter: PlatformAdapter
    method: Method
    image_url: str | None = None
    candidates: tuple[ImageCandidate, ...] = field(default_factory=tuple)
    reason: str = ""
    guidance: str = ""
    http_status: int | None = None
    page_title: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.image_url)

    @property
    def alternatives(self) -> tuple[ImageCandidate, ...]:
        """Ranked candidates other than the one selected."""
        return tuple(c for c in self.candidates if c.url != self.image_url)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "input_url": self.input_url,
            "ok": self.ok,
            "method": self.method.value,
            "method_explanation": self.method.explanation,
            "platform": self.classification.platform.value,
            "platform_display": self.classification.platform.display,
            "platform_icon": self.classification.platform.icon,
            "input_type": self.classification.qualifier,
            "url_class": self.classification.input_type.value,
            "label": self.classification.label,
            "support": self.adapter.support.value,
            "support_badge": self.adapter.support.badge,
        }
        if self.ok:
            payload["image_url"] = self.image_url
            payload["candidate_count"] = len(self.candidates)
            payload["alternatives"] = [c.to_dict() for c in self.alternatives[:8]]
        else:
            # No image_url key at all when we did not get one. A missing key
            # cannot be rendered as a plausible-looking value by mistake.
            payload["reason"] = self.reason
            if self.guidance:
                payload["guidance"] = self.guidance
        if self.http_status is not None:
            payload["http_status"] = self.http_status
        if self.page_title:
            payload["page_title"] = self.page_title
        return payload


def _failed(
    url: str, classification: Classification, adapter: PlatformAdapter,
    reason: str, *, status: int | None = None, title: str = "",
    candidates: tuple[ImageCandidate, ...] = (),
) -> ResolvedInput:
    return ResolvedInput(
        input_url=url,
        classification=classification,
        adapter=adapter,
        method=Method.NONE,
        reason=reason,
        guidance=adapter.guidance,
        http_status=status,
        page_title=title,
        candidates=candidates,
    )


def resolve(url: str, *, fetch_policy=None) -> ResolvedInput:
    """Resolve any public URL. Never raises -- a failure is a result."""
    from urllib.parse import urlsplit

    from tracelock.acquisition.fetcher import (
        FetchPolicy,
        fetch_media,
        is_obviously_private,
    )
    from tracelock.acquisition.validation import sniff_format

    url = (url or "").strip()
    classification = classify_url(url)
    adapter = adapter_for(classification.platform)

    if not url:
        return _failed(url, classification, adapter, "No link was provided.")

    if classification.input_type is InputType.UNKNOWN_URL:
        return _failed(
            url, classification, adapter,
            "That is not a valid public http(s) link.",
        )

    # Refuse obviously-private targets BEFORE any branch below, including the
    # direct-image short-circuit that does no fetch of its own. Otherwise
    # resolve() would answer "resolved: http://169.254.169.254/x.jpg" and leave
    # the refusal to whatever ran next.
    #
    # This is the offline pre-filter, not the authoritative guard: every actual
    # fetch -- the page here, and the resolved image URL wherever it is used --
    # still passes through `fetch_media`, which resolves the name and rechecks
    # every address it returns.
    blocked, why = is_obviously_private(urlsplit(url).hostname or "")
    if blocked:
        return _failed(
            url, classification, adapter,
            "That link points to a private or internal network address "
            "and was blocked ({0}).".format(why),
        )

    policy = fetch_policy or FetchPolicy(max_bytes=MAX_HTML_BYTES)

    # --- direct image: nothing to resolve --------------------------------
    if classification.input_type is InputType.DIRECT_IMAGE_URL:
        return ResolvedInput(
            input_url=url,
            classification=classification,
            adapter=adapter,
            method=Method.DIRECT,
            image_url=url,
        )

    # --- adapter rewrite: public share form -> public canonical form -----
    target = adapter.rewrite(url)
    rewritten = target != url

    # Rewrites must be re-validated like any other URL. An adapter is code we
    # wrote, but the URL it produces still gets the full SSRF treatment below
    # because it flows through the same fetch_media call.
    response = fetch_media(target, policy=policy)

    if not response.ok:
        return _failed(
            url, classification, adapter,
            _failure_reason(response, adapter, classification),
            status=response.status_code,
        )

    body = response.content or b""
    final_url = response.final_url or target

    # A URL with no extension may still SERVE an image. Bytes outrank paths.
    if sniff_format(body):
        refined = refine_with_response(
            classification, content_type=None, body_prefix=body[:64],
            final_url=final_url,
        )
        return ResolvedInput(
            input_url=url,
            classification=refined,
            adapter=adapter,
            method=Method.ADAPTER_REWRITE if rewritten else Method.SNIFFED,
            image_url=final_url,
            http_status=response.status_code,
        )

    # --- it is a document: harvest and rank every image on it ------------
    html = body.decode("utf-8", errors="replace")
    ranked, title = rank_candidates(html, final_url)

    if not ranked:
        schema_image = _schema_org_image(html, final_url)
        if schema_image:
            ranked = (
                ImageCandidate(schema_image, 80, "schema.org"),
            )

    if ranked:
        best = ranked[0]
        return ResolvedInput(
            input_url=url,
            classification=classification,
            adapter=adapter,
            method=_SOURCE_TO_METHOD.get(best.source, Method.IN_PAGE_IMAGE),
            image_url=best.url,
            candidates=tuple(ranked),
            http_status=response.status_code,
            page_title=title,
        )

    # --- last resort: a platform's PUBLIC, token-free oEmbed -------------
    thumbnail = _public_oembed(target, classification.platform, policy)
    if thumbnail:
        return ResolvedInput(
            input_url=url,
            classification=classification,
            adapter=adapter,
            method=Method.OEMBED,
            image_url=thumbnail,
            http_status=response.status_code,
            page_title=title,
        )

    return _failed(
        url, classification, adapter,
        _no_image_reason(adapter, classification),
        status=response.status_code, title=title,
    )


def _no_image_reason(adapter: PlatformAdapter, classification: Classification) -> str:
    if adapter.support is Support.AUTH_REQUIRED:
        return (
            "{0} detected, but no public image was available. {1}".format(
                classification.label.replace(" detected", ""), adapter.note
            )
        )
    return (
        "The page loaded but publishes no usable image "
        "(no og:image, twitter:image, schema.org image, or in-page photo)."
    )


def _failure_reason(
    response, adapter: PlatformAdapter, classification: Classification
) -> str:
    """Explain a fetch failure in terms the operator can act on."""
    status = response.status_code
    reason = response.reason.value if response.reason else ""

    if reason == "BLOCKED_URL_TARGET":
        return "That link points to a private or internal network address and was blocked."
    if reason == "INVALID_URL":
        return "That does not look like a valid web link."
    if reason == "DOWNLOAD_TIMEOUT":
        return "That page did not respond in time."
    if reason == "CONTENT_TOO_LARGE":
        return "That page is too large to inspect."

    # A platform we already declared as auth-gated gets its own explanation for
    # ANY refusal, not just 401/403. LinkedIn answers anonymous requests with
    # HTTP 999, which would otherwise surface as a meaningless number.
    if adapter.support is Support.AUTH_REQUIRED and status and status >= 400:
        if status != 404:
            return "{0} {1}".format(adapter.note, adapter.guidance).strip()

    # Drive answers a private or missing file with a bare 400/403/404. "That
    # page returned HTTP 400" is useless to the person holding the link; the
    # actionable fact is almost always the sharing setting.
    if classification.platform is Platform.GOOGLE_DRIVE and status and status >= 400:
        return (
            "This Google Drive file is not publicly accessible. Open it in "
            "Drive, choose Share, and set General access to "
            "'Anyone with the link'. (Drive returned HTTP {0}.)".format(status)
        )

    if status in (401, 403):
        return (
            "That site refused an anonymous request (HTTP {0}). It may require "
            "sign-in or block automated access.".format(status)
        )
    if status == 404:
        return "That page was not found (HTTP 404). The link may be wrong or deleted."
    if status == 429:
        return "That site is rate-limiting requests right now. Try again shortly."
    if status:
        return "That page returned HTTP {0}.".format(status)
    return "That page could not be reached."


def _schema_org_image(html: str, base_url: str) -> str | None:
    """Fallback to JSON-LD when no meta tag or <img> yielded anything."""
    from urllib.parse import urljoin

    from tracelock.service.url_resolver import _MetaExtractor, _schema_org_image as _dig

    parser = _MetaExtractor()
    try:
        parser.feed(html)
    except Exception:
        return None

    found = _dig(parser.json_ld)
    return urljoin(base_url, found) if found else None


def _public_oembed(url: str, platform: Platform, policy) -> str | None:
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
