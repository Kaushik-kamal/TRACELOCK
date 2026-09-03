"""Image validation: are these bytes actually a usable image?

THE THREAT
----------
A candidate URL is untrusted. The response can lie in three independent ways,
and each needs its own check:

  1. The FILENAME lies         ->  ignored entirely
  2. The CONTENT-TYPE lies     ->  advisory only; recorded, never trusted
  3. The BYTES lie             ->  caught by magic-byte sniffing + real decode

The single most common real-world case is an HTML error or login page served
with HTTP 200 and `Content-Type: image/jpeg`. That is why the magic bytes are
authoritative and the header is only evidence about what the server claimed.

ORIGINAL EVIDENCE != MODEL INPUT
--------------------------------
This module NEVER modifies the bytes. It decodes to inspect, then discards the
decoded array. The original bytes go to the CAS untouched and are what gets
hashed and later notarized. Any resizing the face model performs internally is
a derived representation of the evidence, not the evidence itself, and that
distinction has to survive into Phase 4.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

from tracelock.core.reasons import RejectionReason

# Magic-byte signatures. The authoritative test for "is this an image".
MAGIC_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "JPEG"),
    (b"\x89PNG\r\n\x1a\n", "PNG"),
    (b"GIF87a", "GIF"),
    (b"GIF89a", "GIF"),
    (b"BM", "BMP"),
    (b"II*\x00", "TIFF"),
    (b"MM\x00*", "TIFF"),
)

# WEBP and HEIF are container formats: the signature sits at a byte offset.
_RIFF_WEBP = (b"RIFF", b"WEBP")
_HEIF_BRANDS = (b"ftypheic", b"ftypheix", b"ftyphevc", b"ftypmif1", b"ftypavif")

# Markers that identify a text/HTML payload wearing an image Content-Type.
HTML_MARKERS: tuple[bytes, ...] = (
    b"<!doctype", b"<html", b"<head", b"<body", b"<?xml", b"<script",
)

SUPPORTED_FORMATS: frozenset[str] = frozenset(
    {"JPEG", "PNG", "WEBP", "BMP", "GIF", "TIFF"}
)

# A face below the recognition model's 112px input cannot be read reliably;
# an image smaller than this cannot contain one.
MIN_IMAGE_DIMENSION = 32

# Decompression-bomb ceiling. A few KB of crafted PNG can declare a
# 50000x50000 canvas and exhaust memory on decode.
MAX_IMAGE_PIXELS = 80_000_000  # ~80 MP


@dataclass(frozen=True, slots=True)
class ImageValidation:
    """Outcome of validating downloaded bytes."""

    ok: bool
    detected_format: str | None = None
    declared_content_type: str | None = None
    width: int | None = None
    height: int | None = None
    mode: str | None = None
    byte_size: int = 0
    reason: RejectionReason | None = None
    detail: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def pixels(self) -> int:
        return (self.width or 0) * (self.height or 0)

    @property
    def content_type_was_honest(self) -> bool:
        """Did the server's Content-Type agree with the actual bytes?

        Recorded because a mismatch is a provenance signal in its own right,
        even when the image turns out to be valid.
        """
        if not self.declared_content_type or not self.detected_format:
            return False
        return self.detected_format.lower() in self.declared_content_type.lower()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "detected_format": self.detected_format,
            "declared_content_type": self.declared_content_type,
            "content_type_was_honest": self.content_type_was_honest,
            "width": self.width,
            "height": self.height,
            "mode": self.mode,
            "byte_size": self.byte_size,
            "pixels": self.pixels,
            "reason": self.reason.value if self.reason else None,
            "detail": self.detail,
            "warnings": list(self.warnings),
        }


def sniff_format(data: bytes) -> str | None:
    """Identify an image format from its magic bytes. None if not an image."""
    if len(data) < 4:
        return None

    for signature, name in MAGIC_SIGNATURES:
        if data.startswith(signature):
            return name

    if data[:4] == _RIFF_WEBP[0] and data[8:12] == _RIFF_WEBP[1]:
        return "WEBP"

    if len(data) >= 16 and any(brand in data[4:16] for brand in _HEIF_BRANDS):
        return "HEIF"

    return None


def looks_like_html(data: bytes) -> bool:
    """Does this payload begin like an HTML/XML document?

    Checked explicitly so the rejection can say *HTML served as an image*
    rather than the uninformative *not an image*.
    """
    head = data[:512].lstrip()[:200].lower()
    return any(marker in head for marker in HTML_MARKERS)


def validate_image_bytes(
    data: bytes, *, declared_content_type: str | None = None
) -> ImageValidation:
    """Validate downloaded bytes as a usable image. Never mutates `data`."""
    size = len(data)

    if size == 0:
        return ImageValidation(
            ok=False,
            byte_size=0,
            declared_content_type=declared_content_type,
            reason=RejectionReason.EMPTY_CONTENT,
            detail="response body was empty",
        )

    # Magic bytes first -- authoritative, and cheap.
    detected = sniff_format(data)

    if detected is None:
        if looks_like_html(data):
            return ImageValidation(
                ok=False,
                byte_size=size,
                declared_content_type=declared_content_type,
                reason=RejectionReason.NOT_AN_IMAGE,
                detail=(
                    "payload is HTML, not an image"
                    + (
                        " (server declared Content-Type: {0})".format(
                            declared_content_type
                        )
                        if declared_content_type
                        else ""
                    )
                ),
            )
        return ImageValidation(
            ok=False,
            byte_size=size,
            declared_content_type=declared_content_type,
            reason=RejectionReason.NOT_AN_IMAGE,
            detail="no recognised image signature in the first bytes: {0}".format(
                data[:8].hex()
            ),
        )

    if detected not in SUPPORTED_FORMATS:
        return ImageValidation(
            ok=False,
            detected_format=detected,
            byte_size=size,
            declared_content_type=declared_content_type,
            reason=RejectionReason.UNSUPPORTED_IMAGE_FORMAT,
            detail="{0} is a real image format but is not in the supported "
            "set {1}".format(detected, sorted(SUPPORTED_FORMATS)),
        )

    # Now decode for real. A valid signature does not mean a decodable file.
    from PIL import Image, UnidentifiedImageError

    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    try:
        try:
            with Image.open(io.BytesIO(data)) as probe:
                probe.verify()  # structural check; consumes the file object

            # verify() leaves the image unusable, so reopen to read properties
            # and force a real decode.
            with Image.open(io.BytesIO(data)) as image:
                width, height = image.size
                mode = image.mode
                pil_format = image.format or detected
                image.load()

        except Image.DecompressionBombError as exc:
            return ImageValidation(
                ok=False,
                detected_format=detected,
                byte_size=size,
                declared_content_type=declared_content_type,
                reason=RejectionReason.DECOMPRESSION_BOMB,
                detail="declared pixel count exceeds the {0} limit: {1}".format(
                    MAX_IMAGE_PIXELS, exc
                ),
            )
        except UnidentifiedImageError:
            return ImageValidation(
                ok=False,
                detected_format=detected,
                byte_size=size,
                declared_content_type=declared_content_type,
                reason=RejectionReason.CORRUPT_IMAGE,
                detail="signature says {0} but PIL cannot identify it".format(detected),
            )
        except (OSError, SyntaxError, ValueError) as exc:
            return ImageValidation(
                ok=False,
                detected_format=detected,
                byte_size=size,
                declared_content_type=declared_content_type,
                reason=RejectionReason.CORRUPT_IMAGE,
                detail="decode failed, likely truncated or damaged: {0}".format(exc),
            )
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit

    if min(width, height) < MIN_IMAGE_DIMENSION:
        return ImageValidation(
            ok=False,
            detected_format=pil_format,
            byte_size=size,
            declared_content_type=declared_content_type,
            width=width,
            height=height,
            mode=mode,
            reason=RejectionReason.IMAGE_TOO_SMALL,
            detail="{0}x{1} is below the {2}px minimum".format(
                width, height, MIN_IMAGE_DIMENSION
            ),
        )

    warnings: list[str] = []
    if declared_content_type and detected.lower() not in declared_content_type.lower():
        warnings.append(
            "server declared Content-Type {0!r} but the bytes are {1}".format(
                declared_content_type, detected
            )
        )
    if pil_format and pil_format != detected:
        warnings.append(
            "magic bytes say {0}, PIL reports {1}".format(detected, pil_format)
        )

    return ImageValidation(
        ok=True,
        detected_format=pil_format,
        declared_content_type=declared_content_type,
        width=width,
        height=height,
        mode=mode,
        byte_size=size,
        warnings=tuple(warnings),
    )
