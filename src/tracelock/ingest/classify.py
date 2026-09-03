"""Universal input classification.

One classifier for every source a judge might paste or upload. Downstream code
branches on the RESULT, never on where the input came from -- which is what
stops `instagram_pipeline()` and `upload_pipeline()` growing side by side with
duplicated logic underneath.

CLASSIFICATION IS PROGRESSIVE
-----------------------------
`classify_url` is a pure, offline, structural guess from the URL alone. It is
cheap and always available, so the UI can label a link the moment it is typed.

`refine_with_response` upgrades that guess once bytes arrive: a URL with no
extension that serves `image/jpeg` IS a direct image, whatever its path looked
like. Structure is a hint; bytes are the truth.

PLATFORM LOGIC LIVES IN adapters.py
-----------------------------------
This module knows which host maps to which platform and nothing else. It holds
no resolution strategy, no oEmbed endpoint, no per-site parsing. Adding a
platform is a row in a table here plus an adapter there.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any
from urllib.parse import urlsplit

IMAGE_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff",
    ".avif", ".heic", ".jfif",
)


class InputType(str, Enum):
    """What kind of thing the operator gave us."""

    LOCAL_UPLOAD = "local_upload"
    WEBCAM_CAPTURE = "webcam_capture"
    DEMO_EXAMPLE = "demo_example"
    DIRECT_IMAGE_URL = "direct_image_url"
    WEBPAGE_URL = "webpage_url"
    SOCIAL_POST = "social_post"
    SOCIAL_PROFILE = "social_profile"
    CLOUD_SHARE_LINK = "cloud_share_link"
    UNKNOWN_URL = "unknown_url"

    @property
    def is_local(self) -> bool:
        return self in (
            InputType.LOCAL_UPLOAD,
            InputType.WEBCAM_CAPTURE,
            InputType.DEMO_EXAMPLE,
        )

    @property
    def needs_resolution(self) -> bool:
        """True when we must look at the page before we have an image URL."""
        return self in (
            InputType.WEBPAGE_URL,
            InputType.SOCIAL_POST,
            InputType.SOCIAL_PROFILE,
            InputType.CLOUD_SHARE_LINK,
            InputType.UNKNOWN_URL,
        )


class Platform(str, Enum):
    INSTAGRAM = "instagram"
    FACEBOOK = "facebook"
    X_TWITTER = "x"
    LINKEDIN = "linkedin"
    REDDIT = "reddit"
    PINTEREST = "pinterest"
    YOUTUBE = "youtube"
    GOOGLE_DRIVE = "google_drive"
    IMGUR = "imgur"
    GITHUB = "github"
    GENERIC = "generic"

    @property
    def display(self) -> str:
        return {
            Platform.INSTAGRAM: "Instagram",
            Platform.FACEBOOK: "Facebook",
            Platform.X_TWITTER: "X",
            Platform.LINKEDIN: "LinkedIn",
            Platform.REDDIT: "Reddit",
            Platform.PINTEREST: "Pinterest",
            Platform.YOUTUBE: "YouTube",
            Platform.GOOGLE_DRIVE: "Google Drive",
            Platform.IMGUR: "Imgur",
            Platform.GITHUB: "GitHub",
            Platform.GENERIC: "Webpage",
        }[self]

    @property
    def icon(self) -> str:
        return {
            Platform.INSTAGRAM: "📷",
            Platform.FACEBOOK: "f",
            Platform.X_TWITTER: "𝕏",
            Platform.LINKEDIN: "in",
            Platform.REDDIT: "👽",
            Platform.PINTEREST: "📌",
            Platform.YOUTUBE: "▶",
            Platform.GOOGLE_DRIVE: "☁",
            Platform.IMGUR: "🖼",
            Platform.GITHUB: "⌘",
            Platform.GENERIC: "🔗",
        }[self]


# host suffix -> platform. Matched on suffix so www./m./mobile./regional
# prefixes all resolve correctly.
_HOSTS: tuple[tuple[str, Platform], ...] = (
    ("instagram.com", Platform.INSTAGRAM),
    ("instagr.am", Platform.INSTAGRAM),
    ("cdninstagram.com", Platform.INSTAGRAM),
    ("facebook.com", Platform.FACEBOOK),
    ("fb.watch", Platform.FACEBOOK),
    ("fbcdn.net", Platform.FACEBOOK),
    ("twitter.com", Platform.X_TWITTER),
    ("x.com", Platform.X_TWITTER),
    ("t.co", Platform.X_TWITTER),
    ("twimg.com", Platform.X_TWITTER),
    ("linkedin.com", Platform.LINKEDIN),
    ("lnkd.in", Platform.LINKEDIN),
    ("licdn.com", Platform.LINKEDIN),
    ("reddit.com", Platform.REDDIT),
    ("redd.it", Platform.REDDIT),
    ("redditmedia.com", Platform.REDDIT),
    ("pinterest.com", Platform.PINTEREST),
    ("pin.it", Platform.PINTEREST),
    ("pinimg.com", Platform.PINTEREST),
    ("youtube.com", Platform.YOUTUBE),
    ("youtu.be", Platform.YOUTUBE),
    ("ytimg.com", Platform.YOUTUBE),
    ("drive.google.com", Platform.GOOGLE_DRIVE),
    ("docs.google.com", Platform.GOOGLE_DRIVE),
    ("googleusercontent.com", Platform.GOOGLE_DRIVE),
    ("imgur.com", Platform.IMGUR),
    ("i.imgur.com", Platform.IMGUR),
    ("github.com", Platform.GITHUB),
    ("githubusercontent.com", Platform.GITHUB),
)

# Path shapes that mark a POST rather than a profile. Checked per platform so
# "instagram.com/nasa" is a profile while "instagram.com/p/ABC" is a post.
_POST_PATTERNS: dict[Platform, tuple[re.Pattern, ...]] = {
    Platform.INSTAGRAM: (re.compile(r"^/(p|reel|reels|tv)/"),),
    Platform.FACEBOOK: (
        re.compile(r"/(posts|photo|permalink|videos|story\.php)"),
        re.compile(r"^/share/"),
    ),
    Platform.X_TWITTER: (re.compile(r"/status(es)?/\d+"),),
    Platform.LINKEDIN: (
        re.compile(r"^/(posts|feed/update|pulse)/"),
    ),
    Platform.REDDIT: (re.compile(r"/comments/"),),
    Platform.PINTEREST: (re.compile(r"^/pin/"),),
    Platform.YOUTUBE: (
        re.compile(r"^/watch"), re.compile(r"^/shorts/"), re.compile(r"^/embed/"),
        re.compile(r"^/live/"),
    ),
}

# Path shapes that mark a PROFILE / channel / feed.
_PROFILE_PATTERNS: dict[Platform, tuple[re.Pattern, ...]] = {
    Platform.INSTAGRAM: (re.compile(r"^/[A-Za-z0-9._]+/?$"),),
    Platform.X_TWITTER: (re.compile(r"^/[A-Za-z0-9_]+/?$"),),
    Platform.LINKEDIN: (re.compile(r"^/(in|company|school)/"),),
    Platform.REDDIT: (re.compile(r"^/(u|user|r)/[^/]+/?$"),),
    Platform.PINTEREST: (re.compile(r"^/[A-Za-z0-9_]+/?$"),),
    Platform.YOUTUBE: (
        re.compile(r"^/@"), re.compile(r"^/(c|channel|user)/"),
    ),
    Platform.FACEBOOK: (re.compile(r"^/[A-Za-z0-9.]+/?$"),),
}

CLOUD_PLATFORMS = (Platform.GOOGLE_DRIVE,)


@dataclass(frozen=True, slots=True)
class Classification:
    """The classifier's answer, plus how confident and why."""

    input_type: InputType
    platform: Platform
    confidence: str          # "structural" | "confirmed"
    reason: str = ""

    @property
    def qualifier(self) -> str:
        """Platform-qualified label for provenance: `instagram_post`, etc.

        The coarse `input_type` drives the code path; this names the specific
        thing for a human reading an artifact, without a second lookup.
        """
        if self.input_type is InputType.DIRECT_IMAGE_URL:
            return "direct_image"

        if self.platform is Platform.GENERIC:
            if self.input_type is InputType.WEBPAGE_URL:
                return "generic_webpage"
            return self.input_type.value

        if self.input_type is InputType.SOCIAL_POST:
            return "{0}_post".format(self.platform.value)
        if self.input_type is InputType.SOCIAL_PROFILE:
            return "{0}_profile".format(self.platform.value)
        if self.input_type is InputType.CLOUD_SHARE_LINK:
            return self.platform.value
        if self.input_type is InputType.DIRECT_IMAGE_URL:
            return "direct_image"
        if self.input_type is InputType.WEBPAGE_URL:
            return "{0}_page".format(self.platform.value)
        return self.input_type.value

    @property
    def label(self) -> str:
        """One line a non-technical person can read."""
        if self.input_type is InputType.DIRECT_IMAGE_URL:
            return "Direct image detected"
        if self.input_type is InputType.SOCIAL_POST:
            return "{0} post detected".format(self.platform.display)
        if self.input_type is InputType.SOCIAL_PROFILE:
            return "{0} profile detected".format(self.platform.display)
        if self.input_type is InputType.CLOUD_SHARE_LINK:
            return "{0} link detected".format(self.platform.display)
        if self.input_type is InputType.LOCAL_UPLOAD:
            return "Uploaded image"
        if self.input_type is InputType.WEBCAM_CAPTURE:
            return "Webcam capture"
        if self.input_type is InputType.DEMO_EXAMPLE:
            return "Demo example"
        if self.platform is not Platform.GENERIC:
            return "{0} page detected".format(self.platform.display)
        return "Public webpage detected"

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_type": self.input_type.value,
            "qualifier": self.qualifier,
            "platform": self.platform.value,
            "platform_display": self.platform.display,
            "platform_icon": self.platform.icon,
            "label": self.label,
            "confidence": self.confidence,
            "reason": self.reason,
        }


def platform_for_host(host: str) -> Platform:
    host = (host or "").lower()
    for suffix, platform in _HOSTS:
        if host == suffix or host.endswith("." + suffix):
            return platform
    return Platform.GENERIC


def looks_like_image_path(path: str) -> bool:
    return any(path.lower().split("?")[0].endswith(ext) for ext in IMAGE_EXTENSIONS)


def classify_url(url: str) -> Classification:
    """Structural classification from the URL alone. Pure, offline, instant."""
    raw = (url or "").strip()
    parts = urlsplit(raw)
    host = (parts.hostname or "").lower()
    path = parts.path or "/"

    if not raw or parts.scheme not in ("http", "https") or not host:
        return Classification(
            InputType.UNKNOWN_URL, Platform.GENERIC, "structural",
            "not a well-formed http(s) link",
        )

    platform = platform_for_host(host)

    # An image extension wins over everything: a .jpg on a platform CDN is a
    # direct image, not a post.
    if looks_like_image_path(path):
        return Classification(
            InputType.DIRECT_IMAGE_URL, platform, "structural",
            "path ends in an image extension",
        )

    if platform in CLOUD_PLATFORMS:
        return Classification(
            InputType.CLOUD_SHARE_LINK, platform, "structural",
            "cloud storage share link",
        )

    # Short-link hosts put the item id in the whole path, so a path pattern
    # shared with the main domain would read them as profiles.
    for short_host, short_platform, what in (
        ("youtu.be", Platform.YOUTUBE, "video"),
        ("pin.it", Platform.PINTEREST, "pin"),
        ("fb.watch", Platform.FACEBOOK, "video"),
        ("lnkd.in", Platform.LINKEDIN, "post"),
        ("redd.it", Platform.REDDIT, "post"),
        ("t.co", Platform.X_TWITTER, "link"),
    ):
        if host.endswith(short_host) and len(path.strip("/")) >= 3:
            return Classification(
                InputType.SOCIAL_POST, short_platform, "structural",
                "{0} short link to a {1}".format(short_host, what),
            )

    for pattern in _POST_PATTERNS.get(platform, ()):
        if pattern.search(path):
            return Classification(
                InputType.SOCIAL_POST, platform, "structural",
                "path matches a {0} post".format(platform.display),
            )

    for pattern in _PROFILE_PATTERNS.get(platform, ()):
        if pattern.search(path):
            return Classification(
                InputType.SOCIAL_PROFILE, platform, "structural",
                "path matches a {0} profile".format(platform.display),
            )

    if platform is not Platform.GENERIC:
        # Known platform, unrecognised path shape. Treat as a page and let
        # metadata resolution decide -- better than guessing wrong.
        return Classification(
            InputType.WEBPAGE_URL, platform, "structural",
            "known platform, unrecognised path",
        )

    return Classification(
        InputType.WEBPAGE_URL, Platform.GENERIC, "structural", "generic web page"
    )


def refine_with_response(
    classification: Classification,
    *,
    content_type: str | None,
    body_prefix: bytes,
    final_url: str | None = None,
) -> Classification:
    """Upgrade a structural guess using what the server actually returned.

    Magic bytes outrank the declared content type, which outranks the path --
    the same precedence the acquisition layer uses, for the same reason.
    """
    from tracelock.acquisition.validation import sniff_format

    if sniff_format(body_prefix):
        platform = classification.platform
        if final_url:
            platform = platform_for_host(urlsplit(final_url).hostname or "")
            if platform is Platform.GENERIC:
                platform = classification.platform
        return Classification(
            InputType.DIRECT_IMAGE_URL, platform, "confirmed",
            "response body is image bytes",
        )

    if content_type and content_type.lower().startswith("image/"):
        # Header says image but bytes disagree -- trust the bytes, and say so.
        return Classification(
            classification.input_type, classification.platform, "confirmed",
            "server declared {0} but the bytes are not an image".format(content_type),
        )

    return Classification(
        classification.input_type, classification.platform, "confirmed",
        "response is a document, not an image",
    )
