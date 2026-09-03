"""Is a discovered candidate genuinely a social-media post?

The hackathon requirement asks for "at least one real, matching social media
post". That makes the distinction load-bearing rather than cosmetic: a news
article carrying the same photograph is real, useful corroboration, but it is
NOT a social media post, and calling it one would be the exact kind of
overclaim this system is built to avoid.

THREE STATES, NEVER COLLAPSED
-----------------------------
    DISCOVERED      a live search returned this URL
    SOCIAL          the URL belongs to a genuine social platform
    FACE_VERIFIED   we downloaded the image and the calibrated policy
                    classified it as the same person

A candidate can be SOCIAL without being FACE_VERIFIED -- Instagram serves most
post pages only to signed-in users, so we may know a post exists and be unable
to confirm who is in it. Reporting that as a verified match would be a lie, and
reporting it as nothing would throw away a true finding. So it is reported as
exactly what it is.

The domain list below is evidence that a URL is social, never evidence that a
match is real. Only the face engine decides the second thing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any
from urllib.parse import urlsplit


class SourceCategory(str, Enum):
    """What KIND of place a candidate came from."""

    SOCIAL = "social"
    MEDIA = "media"           # news, magazines, broadcasters
    IMAGE_HOST = "image_host"  # CDNs, image boards, stock
    ENCYCLOPEDIA = "encyclopedia"
    OTHER = "other"

    @property
    def display(self) -> str:
        return {
            SourceCategory.SOCIAL: "Social media",
            SourceCategory.MEDIA: "News / media",
            SourceCategory.IMAGE_HOST: "Image host",
            SourceCategory.ENCYCLOPEDIA: "Encyclopedia",
            SourceCategory.OTHER: "Website",
        }[self]

    @property
    def satisfies_social_requirement(self) -> bool:
        return self is SourceCategory.SOCIAL


# Genuine social platforms: places where a PERSON publishes about themselves or
# others. Registrable domains, matched on suffix so regional and mobile hosts
# (m.facebook.com, in.pinterest.com) resolve correctly.
SOCIAL_DOMAINS: dict[str, str] = {
    "instagram.com": "Instagram",
    "cdninstagram.com": "Instagram",
    "instagr.am": "Instagram",
    "facebook.com": "Facebook",
    "fbcdn.net": "Facebook",
    "fb.com": "Facebook",
    "fb.watch": "Facebook",
    "twitter.com": "X",
    "x.com": "X",
    "twimg.com": "X",
    "t.co": "X",
    "tiktok.com": "TikTok",
    "tiktokcdn.com": "TikTok",
    "linkedin.com": "LinkedIn",
    "licdn.com": "LinkedIn",
    "lnkd.in": "LinkedIn",
    "reddit.com": "Reddit",
    "redd.it": "Reddit",
    "redditmedia.com": "Reddit",
    "pinterest.com": "Pinterest",
    "pinimg.com": "Pinterest",
    "pin.it": "Pinterest",
    "threads.net": "Threads",
    "threads.com": "Threads",
    "youtube.com": "YouTube",
    "youtu.be": "YouTube",
    "ytimg.com": "YouTube",
    "tumblr.com": "Tumblr",
    "vk.com": "VK",
    "weibo.com": "Weibo",
    "mastodon.social": "Mastodon",
    "bsky.app": "Bluesky",
    "flickr.com": "Flickr",
    "staticflickr.com": "Flickr",
    "snapchat.com": "Snapchat",
    "quora.com": "Quora",
    "telegram.org": "Telegram",
    "t.me": "Telegram",
}

# Deliberately NOT social. Real corroboration, but a different claim.
ENCYCLOPEDIA_DOMAINS = (
    "wikipedia.org", "wikimedia.org", "wikidata.org", "britannica.com",
)

IMAGE_HOST_DOMAINS = (
    "imgur.com", "gettyimages.com", "shutterstock.com", "alamy.com",
    "istockphoto.com", "dreamstime.com", "catbox.moe", "tmpfiles.org",
    "imgbb.com", "ibb.co", "cloudinary.com", "githubusercontent.com",
)

# Signals a domain publishes journalism. Not exhaustive -- anything unmatched
# falls to OTHER rather than being guessed into a category.
MEDIA_MARKERS = (
    "news", "times", "post", "herald", "tribune", "gazette", "daily",
    "press", "media", "journal", "reuters", "/bbc", "cnn", "ndtv", "aajtak",
    "indiatoday", "hindustantimes", "thehindu", "guardian", "telegraph",
)


@dataclass(frozen=True, slots=True)
class SourceClassification:
    """Where a candidate came from, and why we say so."""

    category: SourceCategory
    platform: str = ""
    host: str = ""
    reason: str = ""

    @property
    def is_social(self) -> bool:
        return self.category.satisfies_social_requirement

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "category": self.category.value,
            "category_display": self.category.display,
            "is_social": self.is_social,
        }
        if self.platform:
            payload["platform"] = self.platform
        if self.host:
            payload["host"] = self.host
        if self.reason:
            payload["reason"] = self.reason
        return payload


def classify_source(url: str | None) -> SourceClassification:
    """Categorise a candidate URL. Pure, offline, no network."""
    if not url:
        return SourceClassification(SourceCategory.OTHER, reason="no URL")

    host = (urlsplit(url).hostname or "").lower()
    if not host:
        return SourceClassification(SourceCategory.OTHER, reason="no host")

    def matches(domain: str) -> bool:
        return host == domain or host.endswith("." + domain)

    for domain, platform in SOCIAL_DOMAINS.items():
        if matches(domain):
            return SourceClassification(
                SourceCategory.SOCIAL, platform=platform, host=host,
                reason="{0} is a social platform domain".format(domain),
            )

    for domain in ENCYCLOPEDIA_DOMAINS:
        if matches(domain):
            return SourceClassification(
                SourceCategory.ENCYCLOPEDIA, host=host,
                reason="encyclopedia, not a social post",
            )

    for domain in IMAGE_HOST_DOMAINS:
        if matches(domain):
            return SourceClassification(
                SourceCategory.IMAGE_HOST, host=host,
                reason="image host or CDN, not a social post",
            )

    if any(marker in host for marker in MEDIA_MARKERS):
        return SourceClassification(
            SourceCategory.MEDIA, host=host,
            reason="domain name indicates a news publisher",
        )

    return SourceClassification(
        SourceCategory.OTHER, host=host, reason="no category matched",
    )


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Answer the requirement's question about a finished run, honestly.

    Separates three counts that are easy to blur and must not be:

        social_discovered      social URLs a live search returned
        social_face_verified   of those, how many WE confirmed as the same
                               person by downloading and re-measuring

    A run can execute perfectly and find nothing. "Search completed" and "match
    found" are different outcomes, and only the second is a claim about a
    person.
    """
    from tracelock.core.reasons import VerificationStatus

    social_discovered = 0
    social_verified: list[dict[str, Any]] = []
    categories: dict[str, int] = {}

    for result in results:
        provenance = result.get("provenance") or {}
        url = (
            provenance.get("post_url")
            or provenance.get("url")
            or result.get("media_url")
            or ""
        )
        classification = classify_source(url or provenance.get("host"))
        categories[classification.category.value] = (
            categories.get(classification.category.value, 0) + 1
        )
        if not classification.is_social:
            continue

        social_discovered += 1
        if result.get("status") == VerificationStatus.VERIFIED_CANDIDATE.value:
            social_verified.append({
                "platform": classification.platform,
                "host": classification.host,
                "url": url,
                "similarity": result.get("face_similarity"),
                "identity_probability": result.get("identity_probability"),
                "candidate_id": result.get("candidate_id"),
            })

    return {
        "categories": categories,
        "social_discovered": social_discovered,
        "social_face_verified": len(social_verified),
        "verified_social_posts": social_verified,
        # The requirement's exact question, answered without ambiguity.
        "requirement_met": bool(social_verified),
        "statement": (
            "{0} social-media post(s) were discovered by live search and "
            "independently face-verified.".format(len(social_verified))
            if social_verified
            else (
                "Live search completed. {0} social-media URL(s) were "
                "discovered but none could be independently face-verified."
                .format(social_discovered)
                if social_discovered
                else "Live search completed. No social-media post was discovered."
            )
        ),
    }
