"""Extract profile hints from metadata TRACELOCK already has. Zero network calls.

Every function in this module is a pure string transform over data that
already exists on a `VerificationResult` dict: the candidate's own verified
URL, and whatever title/author/text SerpAPI returned with it at discovery
time (`discovery_metadata`, see `verification.verifier._discovery_metadata`).
Nothing here makes an HTTP request, resolves a DNS name, or visits a page --
a test in `tests/test_social_profile.py` asserts this module imports no
networking library at all, so that guarantee cannot silently rot.

TWO WAYS A HANDLE REACHES A URL, AND ONE WAY IT DOESN'T
--------------------------------------------------------
A handle extracted from a platform's OWN url structure (x.com/<handle>/status
or tiktok.com/@<handle>/video) is URL-safe by construction -- it was already
sitting in a URL. Building `https://x.com/<handle>` from it is a string slice,
not a guess.

A DISPLAY NAME extracted from free text ("Jane Q. Doe | LinkedIn") is not
URL-safe. Slugifying "Jane Q. Doe" into a guessed profile path would very
often be wrong, and a wrong-but-plausible-looking URL is worse than none: it
invites a reader to click through to the wrong page. So a text-derived
DISPLAY NAME never becomes a constructed `profile_url` -- `handle` carries
the name, `profile_url` stays `None`, and the evidence chain says why.

PLATFORM COVERAGE IS DELIBERATELY UNEVEN
-----------------------------------------
Instagram and Reddit post URLs are opaque (a shortcode, a submission id) and
never embed the author's handle -- that is a real, structural limitation of
those platforms' URL schemes, not a gap in this code. Only a profile-shaped
URL or a text hint can ever surface a handle for those two.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qs, urlsplit

from tracelock.ingest.social import SourceCategory, classify_source
from tracelock.social_profile.models import EvidenceChainStep, ExtractionMethod, ProfileHint

# ---------------------------------------------------------------------------
# Per-platform URL shape. Conservative on purpose: an unmatched URL produces
# no hint from this method rather than a guessed one.
# ---------------------------------------------------------------------------

_INSTAGRAM_RESERVED = (
    "p", "reel", "reels", "tv", "stories", "explore", "accounts", "direct",
    "about", "developer", "legal",
)
_X_RESERVED = (
    "i", "home", "search", "settings", "messages", "notifications",
    "explore", "compose", "hashtag", "intent",
)
_FACEBOOK_RESERVED = (
    "photo.php", "permalink.php", "story.php", "groups", "pages", "watch",
    "events", "marketplace", "gaming", "help", "login", "plugins",
    "sharer.php", "profile.php", "reel",
)

_PROFILE_PATTERNS: dict[str, re.Pattern[str]] = {
    "Instagram": re.compile(
        r"^/(?!(?:{0})(?:/|$))([A-Za-z0-9._]{{1,30}})/?$".format(
            "|".join(_INSTAGRAM_RESERVED)
        )
    ),
    "X": re.compile(
        r"^/(?!(?:{0})(?:/|$))([A-Za-z0-9_]{{1,15}})/?$".format("|".join(_X_RESERVED))
    ),
    "LinkedIn": re.compile(r"^/in/([A-Za-z0-9\-]+)/?"),
    "Facebook": re.compile(
        r"^/(?!(?:{0})(?:/|$))([A-Za-z0-9.]{{5,50}})/?$".format(
            "|".join(_FACEBOOK_RESERVED)
        )
    ),
    "YouTube": re.compile(r"^/(?:channel/|@|c/|user/)([A-Za-z0-9_.\-]+)/?"),
    "TikTok": re.compile(r"^/@([A-Za-z0-9_.]{1,24})/?$"),
    "Reddit": re.compile(r"^/(?:user|u)/([A-Za-z0-9_\-]{3,20})/?$"),
}

# Platforms whose OWN post-url convention embeds the author's handle.
# Instagram and Reddit are deliberately ABSENT -- see module docstring.
_POST_AUTHOR_PATTERNS: dict[str, re.Pattern[str]] = {
    "X": re.compile(r"^/([A-Za-z0-9_]{1,15})/status/\d+"),
    "TikTok": re.compile(r"^/@([A-Za-z0-9_.]{1,24})/video/\d+"),
    "Facebook": re.compile(
        r"^/(?!(?:{0})(?:/|$))([A-Za-z0-9.]{{5,50}})/posts/".format(
            "|".join(_FACEBOOK_RESERVED)
        )
    ),
    # LinkedIn's own convention: /posts/<author-slug>_<title-slug>-activity-<id>
    # The author segment is a best-effort read of that convention, not a
    # guarantee -- flagged explicitly in the evidence chain text below.
    "LinkedIn": re.compile(r"^/posts/([a-z0-9\-]+?)_"),
}

_PROFILE_URL_TEMPLATE: dict[str, str] = {
    "Instagram": "https://www.instagram.com/{handle}/",
    "X": "https://x.com/{handle}",
    "LinkedIn": "https://www.linkedin.com/in/{handle}/",
    "Facebook": "https://www.facebook.com/{handle}",
    "YouTube": "https://www.youtube.com/{handle}",
    "TikTok": "https://www.tiktok.com/@{handle}",
    "Reddit": "https://www.reddit.com/user/{handle}/",
}


def _facebook_numeric_profile(url: str) -> ProfileHint | None:
    """facebook.com/profile.php?id=<digits> -- a numeric-id profile, common
    when a page has no chosen username. Handled separately: the id lives in
    the query string, not the path, so the regex table above cannot see it.
    """
    parts = urlsplit(url)
    if parts.path.rstrip("/") != "/profile.php":
        return None
    ids = parse_qs(parts.query).get("id")
    if not ids or not ids[0].isdigit():
        return None
    handle = ids[0]
    return ProfileHint(
        candidate_id="",  # filled in by the caller
        platform="Facebook",
        method=ExtractionMethod.URL_IS_PROFILE_SHAPED,
        handle=handle,
        profile_url="https://www.facebook.com/profile.php?id={0}".format(handle),
        quoted_text=url,
        source_field="source_url",
    )


def _url_shape_hint(platform: str, url: str) -> ProfileHint | None:
    """Method 1: is the verified candidate's OWN url profile-shaped?"""
    if platform == "Facebook":
        numeric = _facebook_numeric_profile(url)
        if numeric is not None:
            return numeric

    pattern = _PROFILE_PATTERNS.get(platform)
    if pattern is None:
        return None
    match = pattern.match(urlsplit(url).path)
    if not match:
        return None
    handle = match.group(1)
    return ProfileHint(
        candidate_id="",
        platform=platform,
        method=ExtractionMethod.URL_IS_PROFILE_SHAPED,
        handle=handle,
        profile_url=url,
        quoted_text=url,
        source_field="source_url",
    )


def _post_author_hint(platform: str, url: str) -> ProfileHint | None:
    """Method 2: does this platform embed the author's handle in post URLs?"""
    pattern = _POST_AUTHOR_PATTERNS.get(platform)
    if pattern is None:
        return None
    match = pattern.match(urlsplit(url).path)
    if not match:
        return None
    handle = match.group(1)
    template = _PROFILE_URL_TEMPLATE.get(platform)
    return ProfileHint(
        candidate_id="",
        platform=platform,
        method=ExtractionMethod.HANDLE_IN_POST_URL,
        handle=handle,
        profile_url=template.format(handle=handle) if template else None,
        quoted_text=url,
        source_field="source_url",
    )


# ---------------------------------------------------------------------------
# Text patterns against discovery_metadata. A HANDLE match is URL-safe and
# gets a constructed profile_url; a DISPLAY NAME match never does (see
# module docstring).
# ---------------------------------------------------------------------------

_HANDLE_IN_PARENS = re.compile(r"\(@([A-Za-z0-9_.]{1,30})\)")

_DISPLAY_NAME_BEFORE_PLATFORM: dict[str, re.Pattern[str]] = {
    "LinkedIn": re.compile(r"^(.{1,120}?)\s*[|\-–]\s*LinkedIn\b"),
    "YouTube": re.compile(r"^(.{1,120}?)\s*[|\-–]\s*YouTube\b"),
    "Facebook": re.compile(r"^(.{1,120}?)\s*[|\-–]\s*Facebook\b"),
}


def _title_hint(platform: str, title: str) -> ProfileHint | None:
    """Method 3: does the discovery provider's page title match a known
    per-platform naming convention?"""
    handle_match = _HANDLE_IN_PARENS.search(title)
    if handle_match:
        handle = handle_match.group(1)
        template = _PROFILE_URL_TEMPLATE.get(platform)
        return ProfileHint(
            candidate_id="",
            platform=platform,
            method=ExtractionMethod.TITLE_TEXT_PATTERN,
            handle=handle,
            profile_url=template.format(handle=handle) if template else None,
            quoted_text=title,
            source_field="discovery_metadata.title",
        )

    name_pattern = _DISPLAY_NAME_BEFORE_PLATFORM.get(platform)
    if name_pattern:
        name_match = name_pattern.match(title)
        if name_match:
            display_name = name_match.group(1).strip()
            if display_name:
                return ProfileHint(
                    candidate_id="",
                    platform=platform,
                    method=ExtractionMethod.TITLE_TEXT_PATTERN,
                    handle=display_name,
                    # A free-text display name is not a URL-safe handle --
                    # no profile_url is constructed from it.
                    profile_url=None,
                    quoted_text=title,
                    source_field="discovery_metadata.title",
                )
    return None


def _author_field_hint(platform: str, author: str) -> ProfileHint:
    """Method 4: the provider populated author/channel/uploader directly.

    Treated the same as a text-derived display name: attributed, quoted, but
    never turned into a constructed URL, because a free-text author string
    is not guaranteed to be URL-safe.
    """
    return ProfileHint(
        candidate_id="",
        platform=platform,
        method=ExtractionMethod.AUTHOR_FIELD,
        handle=author,
        profile_url=None,
        quoted_text=author,
        source_field="discovery_metadata.author",
    )


def extract_hints(result: dict[str, Any]) -> list[ProfileHint]:
    """Every hint extractable from ONE verification result. No network call.

    Callers decide what a candidate's `status` must be before calling this --
    this function only extracts; it does not gate on verification. `relate.py`
    is where that gate lives, deliberately, so the gate is auditable in one
    place rather than duplicated in every extraction method here.
    """
    source_url = result.get("source_url") or ""
    classification = classify_source(source_url)
    if classification.category is not SourceCategory.SOCIAL:
        return []
    platform = classification.platform

    hints: list[ProfileHint] = []

    shape_hint = _url_shape_hint(platform, source_url)
    if shape_hint is not None:
        hints.append(shape_hint)

    # A url that IS the profile page is not also a "post by" that profile --
    # only look for an embedded post-author handle when the url was not
    # already classified as the profile itself.
    if shape_hint is None:
        author_hint = _post_author_hint(platform, source_url)
        if author_hint is not None:
            hints.append(author_hint)

    metadata = result.get("discovery_metadata") or {}
    title = metadata.get("title")
    if title:
        title_hint = _title_hint(platform, title)
        if title_hint is not None:
            hints.append(title_hint)

    author = metadata.get("author")
    if author:
        hints.append(_author_field_hint(platform, author))

    candidate_id = result.get("candidate_id") or ""
    return [
        ProfileHint(
            candidate_id=candidate_id,
            platform=h.platform,
            method=h.method,
            handle=h.handle,
            profile_url=h.profile_url,
            quoted_text=h.quoted_text,
            source_field=h.source_field,
        )
        for h in hints
    ]
