"""Normalized data models shared across the pipeline.

Normalization happens HERE, once, at the boundary.  Every provider emits
wildly different JSON; downstream code must only ever see `Candidate`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Unambiguous tracking parameters only.
# Deliberately NOT stripping bare `ref` -- GitHub and others use it as a real
# routing param, and over-stripping mangles URLs, which is worse than a
# duplicate.  Dedup tolerates false negatives; broken URLs it does not.
TRACKING_PARAMS = frozenset(
    {
        "utm_source", "utm_medium", "utm_campaign", "utm_term",
        "utm_content", "utm_id", "utm_name", "utm_reader",
        "fbclid", "gclid", "dclid", "msclkid", "yclid", "igshid",
        "mc_cid", "mc_eid", "_ga", "_gl", "ref_src", "ref_url",
        "si", "spm", "vero_id", "wickedid",
    }
)

_DEFAULT_PORTS = {"http": "80", "https": "443"}


def normalize_url(raw: str | None) -> str | None:
    """Canonicalize a URL for comparison and dedup.

    Lowercases scheme/host, drops default ports and fragments, strips known
    tracking params, and sorts the remaining query.  Returns None for empty or
    unusable input rather than raising -- providers routinely omit fields, and
    a missing URL is data, not an error.
    """
    if not raw or not raw.strip():
        return None

    candidate = raw.strip()
    if candidate.startswith("//"):
        candidate = f"https:{candidate}"

    try:
        parts = urlsplit(candidate)
    except ValueError:
        return None

    if parts.scheme not in ("http", "https") or not parts.netloc:
        return None

    host = (parts.hostname or "").lower()
    if not host:
        return None

    port = parts.port
    netloc = host
    if port is not None and str(port) != _DEFAULT_PORTS.get(parts.scheme):
        netloc = f"{host}:{port}"

    kept = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in TRACKING_PARAMS
    ]
    query = urlencode(sorted(kept))

    path = parts.path or "/"

    return urlunsplit((parts.scheme, netloc, path, query, ""))


def registrable_host(url: str | None) -> str | None:
    """Approximate the registrable domain (eTLD+1) by stripping a `www.` prefix.

    NOTE: this is a stdlib approximation, correct for the common case and wrong
    for multi-label suffixes (`foo.co.uk`, `bar.github.io`).  That is acceptable
    now because nothing depends on exactness yet.  Phase 2 replaces this with
    `tldextract` when the corroboration signal starts counting *independent*
    domains, where over-counting would inflate the trust score.
    """
    if not url:
        return None
    host = (urlsplit(url).hostname or "").lower()
    if not host:
        return None
    return host[4:] if host.startswith("www.") else host


class ProbeRef(BaseModel):
    """The input face image, in the forms different providers actually need.

    DESIGN NOTE -- deviation from `search(image_path)`:
    No production reverse-image API accepts a local file; Google Lens and
    Yandex both fetch a URL you supply.  A future Bluesky provider needs
    neither -- it queries text.  Passing a bare path would force every provider
    to re-solve hosting internally.  ProbeRef carries all three forms, and each
    provider declares which it requires via `Requirement`.
    """

    model_config = ConfigDict(frozen=True)

    local_path: str
    sha256: str
    public_url: str | None = None
    hint_text: str | None = None  # for text-query providers, e.g. Bluesky

    @field_validator("public_url")
    @classmethod
    def _normalize(cls, value: str | None) -> str | None:
        return normalize_url(value)


class Candidate(BaseModel):
    """One discovered result, normalized. Providers emit only this type.

    A Candidate is UNVERIFIED by construction.  It records what a search engine
    claimed, nothing more.  Verification (Phase 2) re-derives identity from the
    downloaded bytes and may reject it.
    """

    model_config = ConfigDict(populate_by_name=True)

    # Which search engine surfaced it -- distinct from `source`, which is the
    # website it lives on.  Both are needed: corroboration counts independent
    # SOURCES, while provider attribution audits the discovery layer.
    provider: str
    source: str | None = None

    # Which result array within the provider response this came from.
    # Optional and defaulted, so artifacts written before this field existed
    # still load unchanged.
    #
    # Not cosmetic: for yandex_images, `image_results` carries a REAL source
    # page while `similar_images` carries Yandex's own CDN cache with no
    # original host. Phase 3 must be able to weight those differently, and
    # collapsing them would let 40 cached thumbnails masquerade as 40
    # independent corroborating sources.
    result_group: str | None = None

    post_url: str | None = None
    image_url: str | None = None
    thumbnail_url: str | None = None

    title: str | None = None
    text: str | None = None
    author: str | None = None
    timestamp: str | None = None

    rank: int | None = None
    discovered_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Verbatim provider payload. Never parsed downstream -- kept so a schema
    # change can be diagnosed after the fact, and so the raw observation can be
    # committed to the evidence bundle in a later phase.
    raw_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("post_url", "image_url", "thumbnail_url", mode="before")
    @classmethod
    def _normalize_urls(cls, value: Any) -> str | None:
        return normalize_url(value) if isinstance(value, str) else None

    @field_validator("title", "text", "author", "source", mode="before")
    @classmethod
    def _clean_text(cls, value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        cleaned = " ".join(value.split())
        return cleaned or None

    def model_post_init(self, _context: Any) -> None:
        # Fall back to the post URL's host when the provider omits a site name.
        if self.source is None:
            object.__setattr__(self, "source", registrable_host(self.post_url))

    @property
    def has_media(self) -> bool:
        """True when something fetchable exists. The gate's PASS criterion."""
        return bool(self.image_url or self.thumbnail_url)

    @property
    def dedup_key(self) -> str:
        """Stable identity for dedup: normalized post URL, else image URL."""
        return self.post_url or self.image_url or self.thumbnail_url or ""
