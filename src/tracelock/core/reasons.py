"""Pipeline stages and the rejection taxonomy.

Lives in `core/` because it is SHARED VOCABULARY: `acquisition` raises these
reasons and `verification` classifies on them. Putting it in either package
would create an import cycle between them -- which is exactly the signal that
it belongs to neither. `core/` depends on nothing internal, by design.

This module is the spine of Phase 2. Every candidate that does not survive must
carry a machine-readable answer to three questions:

    WHAT HAPPENED   -> RejectionReason
    WHY             -> the reason's `explanation`
    AT WHICH STAGE  -> the reason's `stage`

A candidate is never silently dropped. Failures are results, not absences.

WHY REJECTIONS ARE THE PRODUCT
------------------------------
A search engine returned these URLs claiming a relationship. Phase 2 exists to
test that claim independently. A verifier that accepts everything has tested
nothing, so the taxonomy below is deliberately detailed on the failure side: it
is the part a reviewer should be able to audit.
"""

from __future__ import annotations

from enum import Enum


class Stage(str, Enum):
    """Ordered pipeline stages. A candidate advances until it fails or finishes."""

    DISCOVERED = "DISCOVERED"   # handed to us by a Phase 0 provider
    ACQUIRED = "ACQUIRED"       # bytes re-downloaded by us, not the provider
    VALIDATED = "VALIDATED"     # bytes proven to be a decodable image
    ANALYZED = "ANALYZED"       # face independently detected and embedded
    COMPARED = "COMPARED"       # measured against the probe
    CLASSIFIED = "CLASSIFIED"   # policy applied

    @property
    def order(self) -> int:
        return _STAGE_ORDER[self]


_STAGE_ORDER: dict[Stage, int] = {
    Stage.DISCOVERED: 0,
    Stage.ACQUIRED: 1,
    Stage.VALIDATED: 2,
    Stage.ANALYZED: 3,
    Stage.COMPARED: 4,
    Stage.CLASSIFIED: 5,
}


class StageOutcome(str, Enum):
    OK = "OK"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"  # an earlier stage failed; this one never ran


class VerificationStatus(str, Enum):
    """Terminal classification of a candidate.

    Three states, deliberately. A binary accept/reject would force a decision
    the evidence does not support: with no calibration data, "I measured this
    and cannot conclude" is the honest answer for a large middle band.
    """

    VERIFIED_CANDIDATE = "VERIFIED_CANDIDATE"
    INCONCLUSIVE = "INCONCLUSIVE"
    REJECTED = "REJECTED"


class RejectionReason(str, Enum):
    """Why a candidate failed. Each maps to exactly one stage."""

    # --- ACQUIRED -------------------------------------------------------
    NO_MEDIA_URL = "NO_MEDIA_URL"
    INVALID_URL = "INVALID_URL"
    BLOCKED_URL_TARGET = "BLOCKED_URL_TARGET"
    DOWNLOAD_TIMEOUT = "DOWNLOAD_TIMEOUT"
    DOWNLOAD_FAILED = "DOWNLOAD_FAILED"
    HTTP_ERROR = "HTTP_ERROR"
    TOO_MANY_REDIRECTS = "TOO_MANY_REDIRECTS"
    CONTENT_TOO_LARGE = "CONTENT_TOO_LARGE"
    EMPTY_CONTENT = "EMPTY_CONTENT"

    # --- VALIDATED ------------------------------------------------------
    NOT_AN_IMAGE = "NOT_AN_IMAGE"
    UNSUPPORTED_IMAGE_FORMAT = "UNSUPPORTED_IMAGE_FORMAT"
    CORRUPT_IMAGE = "CORRUPT_IMAGE"
    IMAGE_TOO_SMALL = "IMAGE_TOO_SMALL"
    DECOMPRESSION_BOMB = "DECOMPRESSION_BOMB"

    # --- ANALYZED -------------------------------------------------------
    NO_FACE_DETECTED = "NO_FACE_DETECTED"
    FACE_ANALYSIS_FAILED = "FACE_ANALYSIS_FAILED"
    AMBIGUOUS_MULTIPLE_FACES = "AMBIGUOUS_MULTIPLE_FACES"
    LOW_FACE_QUALITY = "LOW_FACE_QUALITY"

    # --- COMPARED -------------------------------------------------------
    EMBEDDING_INCOMPATIBLE = "EMBEDDING_INCOMPATIBLE"
    LOW_FACE_SIMILARITY = "LOW_FACE_SIMILARITY"

    # --- cross-cutting --------------------------------------------------
    DUPLICATE_CONTENT = "DUPLICATE_CONTENT"

    @property
    def stage(self) -> Stage:
        return _REASON_STAGE[self]

    @property
    def explanation(self) -> str:
        return _REASON_EXPLANATION[self]


_REASON_STAGE: dict[RejectionReason, Stage] = {
    RejectionReason.NO_MEDIA_URL: Stage.DISCOVERED,
    RejectionReason.INVALID_URL: Stage.ACQUIRED,
    RejectionReason.BLOCKED_URL_TARGET: Stage.ACQUIRED,
    RejectionReason.DOWNLOAD_TIMEOUT: Stage.ACQUIRED,
    RejectionReason.DOWNLOAD_FAILED: Stage.ACQUIRED,
    RejectionReason.HTTP_ERROR: Stage.ACQUIRED,
    RejectionReason.TOO_MANY_REDIRECTS: Stage.ACQUIRED,
    RejectionReason.CONTENT_TOO_LARGE: Stage.ACQUIRED,
    RejectionReason.EMPTY_CONTENT: Stage.ACQUIRED,
    RejectionReason.NOT_AN_IMAGE: Stage.VALIDATED,
    RejectionReason.UNSUPPORTED_IMAGE_FORMAT: Stage.VALIDATED,
    RejectionReason.CORRUPT_IMAGE: Stage.VALIDATED,
    RejectionReason.IMAGE_TOO_SMALL: Stage.VALIDATED,
    RejectionReason.DECOMPRESSION_BOMB: Stage.VALIDATED,
    RejectionReason.NO_FACE_DETECTED: Stage.ANALYZED,
    RejectionReason.FACE_ANALYSIS_FAILED: Stage.ANALYZED,
    RejectionReason.AMBIGUOUS_MULTIPLE_FACES: Stage.ANALYZED,
    RejectionReason.LOW_FACE_QUALITY: Stage.ANALYZED,
    RejectionReason.EMBEDDING_INCOMPATIBLE: Stage.COMPARED,
    RejectionReason.LOW_FACE_SIMILARITY: Stage.COMPARED,
    RejectionReason.DUPLICATE_CONTENT: Stage.VALIDATED,
}

_REASON_EXPLANATION: dict[RejectionReason, str] = {
    RejectionReason.NO_MEDIA_URL: (
        "the discovery provider returned no image or thumbnail URL, so there "
        "is nothing to re-acquire"
    ),
    RejectionReason.INVALID_URL: (
        "the URL is not a well-formed http(s) address"
    ),
    RejectionReason.BLOCKED_URL_TARGET: (
        "the URL resolves to a private, loopback or link-local address. "
        "Provider URLs are untrusted input; fetching them would be an SSRF "
        "vector against our own network"
    ),
    RejectionReason.DOWNLOAD_TIMEOUT: (
        "the host did not respond within the configured timeout"
    ),
    RejectionReason.DOWNLOAD_FAILED: (
        "a transport-level failure occurred: DNS, TLS or connection reset"
    ),
    RejectionReason.HTTP_ERROR: (
        "the server returned a non-200 status. The URL may have expired, or "
        "the host may be blocking automated clients"
    ),
    RejectionReason.TOO_MANY_REDIRECTS: (
        "the redirect limit was exceeded, which also covers redirect loops"
    ),
    RejectionReason.CONTENT_TOO_LARGE: (
        "the response exceeded the maximum download size and was aborted "
        "mid-stream"
    ),
    RejectionReason.EMPTY_CONTENT: (
        "the server returned HTTP 200 with a zero-length body. A successful "
        "status with no content usually means an expired or revoked asset URL"
    ),
    RejectionReason.NOT_AN_IMAGE: (
        "the downloaded bytes are not an image. Commonly an HTML error page "
        "served with HTTP 200 and an image Content-Type -- which is exactly "
        "why the magic bytes are authoritative and the header is not"
    ),
    RejectionReason.UNSUPPORTED_IMAGE_FORMAT: (
        "the bytes decode as an image, but in a format outside the supported set"
    ),
    RejectionReason.CORRUPT_IMAGE: (
        "the file claims a known image format but fails to decode; likely "
        "truncated or damaged in transit"
    ),
    RejectionReason.IMAGE_TOO_SMALL: (
        "the image is below the minimum usable dimensions; too small to "
        "contain a face the recognition model can read"
    ),
    RejectionReason.DECOMPRESSION_BOMB: (
        "the declared pixel count exceeds the safety limit -- a small file "
        "that expands to an enormous bitmap and would exhaust memory"
    ),
    RejectionReason.NO_FACE_DETECTED: (
        "we re-ran face detection on the bytes we downloaded and found no "
        "face. The provider's visual match did not involve a detectable face"
    ),
    RejectionReason.FACE_ANALYSIS_FAILED: (
        "the face engine raised an error while analyzing the image"
    ),
    RejectionReason.AMBIGUOUS_MULTIPLE_FACES: (
        "several faces scored too closely for the primary-face policy to "
        "choose between them; attributing the match to one of them would be "
        "an unjustified guess"
    ),
    RejectionReason.LOW_FACE_QUALITY: (
        "the detected face is below the configured quality floor. A degraded "
        "face produces an unreliable embedding, and blur in particular pulls "
        "embeddings toward the population mean, INFLATING similarity"
    ),
    RejectionReason.EMBEDDING_INCOMPATIBLE: (
        "probe and candidate embeddings have different dimensions or model "
        "ids and must not be compared"
    ),
    RejectionReason.LOW_FACE_SIMILARITY: (
        "cosine similarity fell below the PROVISIONAL floor. This is an "
        "uncalibrated operational filter, not a calibrated identity claim"
    ),
    RejectionReason.DUPLICATE_CONTENT: (
        "these exact bytes were already acquired from another candidate URL. "
        "Both source references are preserved in the CAS"
    ),
}


def reasons_for_stage(stage: Stage) -> tuple[RejectionReason, ...]:
    """Every rejection reason that can arise at a given stage."""
    return tuple(r for r, s in _REASON_STAGE.items() if s is stage)
