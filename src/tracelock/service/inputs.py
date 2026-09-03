"""TraceInput -- the single normalization point for every input source.

    Webcam ────────┐
    Upload ────────┤
    Image URL ─────┼──> TraceInput ──> the existing TRACELOCK pipeline
    Examples ──────┤
    Google Drive ──┘

THE ARCHITECTURAL RULE THIS ENFORCES
------------------------------------
There is exactly ONE pipeline. Five ways in, one object out. Nothing
downstream of this module knows or cares where an image came from, which is
what stops "the webcam path" and "the upload path" drifting into two systems
that behave differently.

The pipeline reads `local_path` and nothing else. Every source resolves to a
file on disk before it becomes a TraceInput.

DEMO PROVENANCE IS NOT DISCOVERY PROVENANCE
-------------------------------------------
`kind` marks how the INPUT was obtained, never how the EVIDENCE was found.
A demo example is a staged INPUT whose discovery is still entirely live: the
search runs against the real internet and the candidates are genuinely
discovered. Conflating the two would let staged input be read as staged
evidence, which is the one thing this project must never do.

That is also why this enum is deliberately separate from
`calibration.DatasetKind` (REAL / FIXTURE). That one governs whether a
calibration result may be cited. This one governs how an image reached us.
They answer different questions and must not be merged.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlsplit

if TYPE_CHECKING:  # pragma: no cover
    from tracelock.acquisition.fetcher import FetchPolicy

SUPPORTED_MIME = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class SourceType(str, Enum):
    WEBCAM = "webcam"
    UPLOAD = "upload"
    URL = "url"
    EXAMPLE = "example"
    GOOGLE_DRIVE = "google_drive"

    @property
    def label(self) -> str:
        return {
            SourceType.WEBCAM: "Webcam capture",
            SourceType.UPLOAD: "Uploaded file",
            SourceType.URL: "Public image URL",
            SourceType.EXAMPLE: "Demo example",
            SourceType.GOOGLE_DRIVE: "Google Drive",
        }[self]


class InputKind(str, Enum):
    """How the INPUT was obtained. Says nothing about the evidence."""

    ORGANIC = "ORGANIC"          # supplied by the operator for a real investigation
    DEMO_EXAMPLE = "DEMO_EXAMPLE"  # a bundled example image

    @property
    def notice(self) -> str:
        return {
            InputKind.ORGANIC: "",
            InputKind.DEMO_EXAMPLE: (
                "This is a bundled DEMO INPUT image. The investigation it "
                "triggers is fully live: discovery queries the real internet "
                "and every candidate is genuinely found and independently "
                "verified. Only the starting image is pre-selected."
            ),
        }[self]


class InputError(Exception):
    """Input could not be normalized. Carries a message fit for a non-engineer."""

    def __init__(self, message: str, *, hint: str = ""):
        super().__init__(message)
        self.message = message
        self.hint = hint

    def to_dict(self) -> dict[str, Any]:
        return {"error": self.message, "hint": self.hint}


@dataclass(frozen=True, slots=True)
class TraceInput:
    """A normalized pipeline input. The only thing the pipeline accepts."""

    source_type: SourceType
    kind: InputKind
    local_path: str
    filename: str
    mime_type: str
    sha256: str
    byte_size: int
    image_url: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def is_demo(self) -> bool:
        return self.kind is InputKind.DEMO_EXAMPLE

    @property
    def needs_hosting(self) -> bool:
        """True when discovery has no public URL to hand a search provider.

        Reverse-image search fetches a URL; it cannot receive a local file.
        A webcam frame or an upload therefore has no discovery path until it
        is published somewhere reachable.
        """
        return not self.image_url

    def to_dict(self) -> dict[str, Any]:
        """Safe for the browser. Deliberately omits `local_path`.

        Internal filesystem paths are not shown in the UI: they leak server
        layout and mean nothing to a viewer.
        """
        return {
            "source_type": self.source_type.value,
            "source_label": self.source_type.label,
            "kind": self.kind.value,
            "kind_notice": self.kind.notice,
            "filename": self.filename,
            "mime_type": self.mime_type,
            "sha256": self.sha256,
            "byte_size": self.byte_size,
            "image_url": self.image_url,
            "needs_hosting": self.needs_hosting,
            "provenance": self.provenance,
            "created_at": self.created_at,
        }


# ==========================================================================
# Construction
# ==========================================================================


def _sniff_and_validate(data: bytes, filename: str) -> tuple[str, str]:
    """Return (mime_type, extension), rejecting anything that is not an image.

    Reuses the Phase 2 validator so the browser path applies exactly the same
    magic-byte rules as candidate acquisition -- the filename and any declared
    content type are ignored.
    """
    from tracelock.acquisition.validation import validate_image_bytes

    if not data:
        raise InputError(
            "That file is empty.", hint="Choose an image file and try again."
        )
    if len(data) > MAX_UPLOAD_BYTES:
        raise InputError(
            "That image is larger than {0} MB.".format(MAX_UPLOAD_BYTES // 1024 // 1024),
            hint="Try a smaller image.",
        )

    result = validate_image_bytes(data)
    if not result.ok:
        raise InputError(
            _friendly_validation_message(result), hint=_validation_hint(result)
        )

    fmt = (result.detected_format or "").upper()
    mime = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
    }.get(fmt)

    if mime is None:
        raise InputError(
            "{0} images are not supported.".format(fmt or "These"),
            hint="Use a JPG, PNG or WebP image.",
        )

    return mime, SUPPORTED_MIME[mime]


def _friendly_validation_message(result) -> str:
    """Translate a validator reason into something a judge can read."""
    reason = result.reason.value if result.reason else ""
    return {
        "NOT_AN_IMAGE": "That is not an image file.",
        "CORRUPT_IMAGE": "That image file appears to be damaged or incomplete.",
        "IMAGE_TOO_SMALL": "That image is too small to contain a usable face.",
        "UNSUPPORTED_IMAGE_FORMAT": "That image format is not supported.",
        "DECOMPRESSION_BOMB": "That image declares an unreasonable size and was rejected.",
        "EMPTY_CONTENT": "That file is empty.",
    }.get(reason, "That image could not be read.")


def _validation_hint(result) -> str:
    reason = result.reason.value if result.reason else ""
    if reason == "NOT_AN_IMAGE" and "HTML" in (result.detail or ""):
        return (
            "The link returned a web page, not an image. Open the image "
            "directly and copy that address instead."
        )
    if reason == "IMAGE_TOO_SMALL":
        return "Use an image at least 32 pixels on each side."
    return "Use a JPG, PNG or WebP image."


def _store(data: bytes, extension: str, work_dir: Path) -> tuple[Path, str]:
    """Write bytes to a content-addressed scratch file. Returns (path, sha256)."""
    digest = hashlib.sha256(data).hexdigest()
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / "{0}{1}".format(digest[:32], extension)
    if not path.exists():
        path.write_bytes(data)
    return path, digest


def from_bytes(
    data: bytes,
    *,
    source_type: SourceType,
    filename: str,
    work_dir: Path,
    kind: InputKind = InputKind.ORGANIC,
    image_url: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> TraceInput:
    """Normalize raw bytes from any source into a TraceInput."""
    mime, extension = _sniff_and_validate(data, filename)
    path, digest = _store(data, extension, work_dir)

    return TraceInput(
        source_type=source_type,
        kind=kind,
        local_path=str(path),
        filename=filename or "{0}{1}".format(digest[:12], extension),
        mime_type=mime,
        sha256=digest,
        byte_size=len(data),
        image_url=image_url,
        provenance=provenance or {},
    )


def from_upload(data: bytes, filename: str, work_dir: Path) -> TraceInput:
    return from_bytes(
        data,
        source_type=SourceType.UPLOAD,
        filename=filename or "upload",
        work_dir=work_dir,
        provenance={"origin": "browser file upload"},
    )


def from_webcam(data: bytes, work_dir: Path) -> TraceInput:
    return from_bytes(
        data,
        source_type=SourceType.WEBCAM,
        filename="webcam-capture.jpg",
        work_dir=work_dir,
        provenance={"origin": "browser webcam capture"},
    )


def from_url(
    url: str,
    work_dir: Path,
    *,
    source_type: SourceType = SourceType.URL,
    fetch_policy: "FetchPolicy | None" = None,
) -> TraceInput:
    """Fetch a public image URL through the hardened Phase 2 fetcher.

    The URL is untrusted input, so it goes through the same SSRF guard, size
    cap and redirect limit that candidate media does. Nothing here is a
    relaxed browser-facing shortcut.
    """
    from tracelock.acquisition.fetcher import FetchPolicy, fetch_media

    url = (url or "").strip()
    if not url:
        raise InputError("No link was provided.", hint="Paste a direct image link.")

    result = fetch_media(url, policy=fetch_policy or FetchPolicy())
    if not result.ok:
        raise InputError(
            _friendly_fetch_message(result), hint=_fetch_hint(result, url)
        )

    trace = from_bytes(
        result.content or b"",
        source_type=source_type,
        filename=Path(urlsplit(result.final_url or url).path).name or "image",
        work_dir=work_dir,
        image_url=result.final_url or url,
        provenance={
            "origin": source_type.label,
            "requested_url": url,
            "final_url": result.final_url,
            "http_status": result.status_code,
        },
    )
    return trace


def _friendly_fetch_message(result) -> str:
    reason = result.reason.value if result.reason else ""
    return {
        "HTTP_ERROR": "That link returned an error ({0}).".format(
            result.status_code or "no response"
        ),
        "DOWNLOAD_TIMEOUT": "That link took too long to respond.",
        "DOWNLOAD_FAILED": "That link could not be reached.",
        "INVALID_URL": "That does not look like a valid web link.",
        "BLOCKED_URL_TARGET": "That link points to a private network address and was blocked.",
        "CONTENT_TOO_LARGE": "That image is too large.",
        "EMPTY_CONTENT": "That link returned an empty response.",
        "TOO_MANY_REDIRECTS": "That link redirected too many times.",
    }.get(reason, "That link could not be loaded.")


def _fetch_hint(result, url: str) -> str:
    status = result.status_code
    if status == 403:
        return (
            "The site is blocking automated access. Many sites do. Try an "
            "image hosted somewhere more permissive, or upload the file directly."
        )
    if status == 404:
        return "The image may have been moved or deleted."
    if "drive.google.com" in url:
        return (
            "For Google Drive, set the file to 'Anyone with the link' and "
            "paste the share link -- it will be converted automatically."
        )
    return "Make sure the link points directly at an image file."


# ==========================================================================
# Google Drive
# ==========================================================================

_DRIVE_ID_PATTERNS = (
    re.compile(r"/file/d/([a-zA-Z0-9_-]{10,})"),
    re.compile(r"/document/d/([a-zA-Z0-9_-]{10,})"),
    re.compile(r"[?&]id=([a-zA-Z0-9_-]{10,})"),
)


def extract_drive_file_id(url: str) -> str | None:
    """Pull the file id out of any common Google Drive link shape."""
    if "drive.google.com" not in (url or "") and "docs.google.com" not in (url or ""):
        return None
    for pattern in _DRIVE_ID_PATTERNS:
        match = pattern.search(url)
        if match:
            return match.group(1)
    query = parse_qs(urlsplit(url).query)
    ids = query.get("id")
    return ids[0] if ids else None


def drive_direct_url(file_id: str) -> str:
    return "https://drive.google.com/uc?export=download&id={0}".format(file_id)


def from_google_drive(
    share_url: str, work_dir: Path, *, fetch_policy: "FetchPolicy | None" = None
) -> TraceInput:
    """Import a PUBLIC Google Drive file from its share link.

    This is a real, working import path -- not a placeholder. It converts a
    share link to Drive's direct-download form and fetches it through the
    same hardened path as any other URL.

    What it is NOT: the Google Picker API. Picker needs an OAuth client id and
    a consent flow that this build does not ship, so a private file will fail
    with a clear message rather than appearing to work.
    """
    file_id = extract_drive_file_id(share_url)
    if not file_id:
        raise InputError(
            "That does not look like a Google Drive link.",
            hint="Copy the share link from Drive (Share -> Copy link).",
        )

    try:
        return from_url(
            drive_direct_url(file_id),
            work_dir,
            source_type=SourceType.GOOGLE_DRIVE,
            fetch_policy=fetch_policy,
        )
    except InputError as exc:
        raise InputError(
            "That Drive file could not be opened. It is probably not shared publicly.",
            hint=(
                "In Drive: Share -> General access -> 'Anyone with the link'. "
                "This build imports public files by link; it does not sign in "
                "to your Google account."
            ),
        ) from exc
