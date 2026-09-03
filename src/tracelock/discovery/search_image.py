"""Stage 2.5 -- search-image guard.

WHY THIS EXISTS
---------------
A real failure: the gate was run with

    --image     data/probes/me.jpg                        (a good face)
    --image-url https://avatars.githubusercontent.com/...  (a GitHub identicon)

The gate validated the LOCAL file, found a face, and passed. But the provider
never sees the local file -- it fetches the URL. Yandex was handed a flat
two-colour geometric pattern and dutifully returned twenty other people's
GitHub avatars. Every downstream stage worked perfectly on input that was
meaningless.

    The guard therefore validates THE URL, not the local file.
    Validating the local file is precisely the bug it exists to catch.

WHAT IT GATES ON (hard failures)
--------------------------------
  URL_UNFETCHABLE       the provider could not have fetched it either
  NOT_AN_IMAGE          the bytes are not an image
  NO_FACE_DETECTED      the headline case -- searching a faceless image
  FACE_NOT_USABLE       a face was found but is icon-scale or too degraded to
                        seed a meaningful search (SCRFD fires spurious
                        detections on logos and textures)
  FACE_ANALYSIS_FAILED  the engine errored

WHAT IT WILL NOT DO
-------------------
It does NOT decide whether the searched image and the probe show the same
person. That is an identity claim, and identity claims require a calibrated
mapping fitted to labelled pairs, which this project does not yet have.

It DOES report the probe relationship as raw measurements -- a cosine
similarity and a pHash distance, both clearly labelled as observations with no
verdict attached. `ProbeRelationship` carries `is_identity_claim = False` and
there is no threshold anywhere in this module.

The one comparison it makes confidently is FACTUAL, not evaluative: whether the
searched image is byte-different from the probe. "You searched image A and will
verify against image B" is a provenance statement, not an identity judgement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from tracelock.acquisition.fetcher import FetchPolicy, fetch_media
from tracelock.acquisition.validation import validate_image_bytes


class SearchImageIssue(str, Enum):
    """Hard failures. The search must not proceed."""

    URL_UNFETCHABLE = "URL_UNFETCHABLE"
    NOT_AN_IMAGE = "NOT_AN_IMAGE"
    NO_FACE_DETECTED = "NO_FACE_DETECTED"
    FACE_NOT_USABLE = "FACE_NOT_USABLE"
    FACE_ANALYSIS_FAILED = "FACE_ANALYSIS_FAILED"

    @property
    def explanation(self) -> str:
        return _ISSUE_EXPLANATION[self]


_ISSUE_EXPLANATION: dict[SearchImageIssue, str] = {
    SearchImageIssue.URL_UNFETCHABLE: (
        "the search image URL could not be downloaded. The search provider "
        "fetches this same URL, so it would fail there too"
    ),
    SearchImageIssue.NOT_AN_IMAGE: (
        "the URL returned bytes that are not a decodable image. A provider "
        "given this would either error or return meaningless results"
    ),
    SearchImageIssue.NO_FACE_DETECTED: (
        "no face was found in the image the provider will actually receive. "
        "A face search seeded with a faceless image returns visually similar "
        "NON-FACES -- results that look plausible and mean nothing"
    ),
    SearchImageIssue.FACE_NOT_USABLE: (
        "a face was detected but is too small or too degraded to be a usable "
        "search seed. SCRFD reports spurious low-confidence detections on "
        "face-like patterns -- logos, icons, textures -- and a 10px 'face' in "
        "an icon will return results about the icon, not about a person. This "
        "is a judgement about IMAGE FITNESS, not about who is depicted"
    ),
    SearchImageIssue.FACE_ANALYSIS_FAILED: (
        "the face engine raised an error analyzing the search image"
    ),
}


class SearchImageWarning(str, Enum):
    """Non-fatal observations. The search proceeds."""

    MULTIPLE_FACES = "MULTIPLE_FACES"
    AMBIGUOUS_PRIMARY_FACE = "AMBIGUOUS_PRIMARY_FACE"
    LOW_FACE_QUALITY = "LOW_FACE_QUALITY"
    SMALL_FACE = "SMALL_FACE"
    DIFFERS_FROM_PROBE = "DIFFERS_FROM_PROBE"


# ---------------------------------------------------------------------------
# INPUT-FITNESS FLOORS -- these GATE.
#
# NOT identity thresholds. They answer "is this image a usable search seed",
# never "who is in it". No comparison against the probe influences them.
#
# They exist because a face-PRESENCE check alone is not enough. Measured
# against the real GitHub logo favicon (32x32):
#
#     det_score 0.511, quality 0.002, face 10px on its shorter side
#
# SCRFD fired a spurious detection on an icon, and a presence-only guard let
# it through -- the search then returned twenty pages about the GitHub logo.
# Phase 1 documented this failure mode: "SCRFD occasionally reports high
# confidence on face-like patterns".
#
# Values are provisional and configurable. 50px is comfortably under the
# recognition model's 112px input while still excluding icon-scale artefacts;
# 0.15 sits well below Phase 1's POOR band (0.35) so only genuinely unusable
# images are rejected.
# ---------------------------------------------------------------------------
MIN_SEARCH_FACE_PX = 50
MIN_SEARCH_QUALITY = 0.15

# Working defaults for WARNINGS only. These never gate.
WARN_QUALITY_BELOW = 0.35
WARN_FACE_PX_BELOW = 80
PHASH_SAME_IMAGE_MAX_DISTANCE = 10


@dataclass(frozen=True, slots=True)
class ProbeRelationship:
    """How the searched image relates to the local probe. MEASUREMENTS ONLY.

    Deliberately verdict-free. `is_identity_claim` is a permanent False and
    exists so a reader of the artifact cannot mistake these numbers for a
    conclusion about who is depicted.
    """

    cosine_similarity: float | None
    phash_distance: int | None
    is_same_image: bool | None
    is_identity_claim: bool = False

    @property
    def note(self) -> str:
        return (
            "Raw measurements between the searched image and the local probe. "
            "NO identity conclusion is drawn: converting a cosine similarity "
            "into an identity claim requires calibration this project does "
            "not yet have (see tracelock.calibration)."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cosine_similarity": (
                round(self.cosine_similarity, 6)
                if self.cosine_similarity is not None
                else None
            ),
            "phash_distance": self.phash_distance,
            "is_same_image": self.is_same_image,
            "is_identity_claim": self.is_identity_claim,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class SearchImageCheck:
    """Result of the Stage 2.5 guard."""

    url: str
    ok: bool
    issue: SearchImageIssue | None = None
    detail: str = ""

    http_status: int | None = None
    byte_size: int = 0
    image_format: str | None = None
    width: int | None = None
    height: int | None = None
    content_sha256: str | None = None

    faces_detected: int | None = None
    det_score: float | None = None
    quality_aggregate: float | None = None
    quality_band: str | None = None
    face_min_side_px: float | None = None

    probe_relationship: ProbeRelationship | None = None
    warnings: tuple[SearchImageWarning, ...] = field(default_factory=tuple)
    warning_details: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "ok": self.ok,
            "issue": self.issue.value if self.issue else None,
            "issue_explanation": self.issue.explanation if self.issue else None,
            "detail": self.detail,
            "http_status": self.http_status,
            "byte_size": self.byte_size,
            "image_format": self.image_format,
            "width": self.width,
            "height": self.height,
            "content_sha256": self.content_sha256,
            "faces_detected": self.faces_detected,
            "det_score": round(self.det_score, 4) if self.det_score is not None else None,
            "quality_aggregate": (
                round(self.quality_aggregate, 4)
                if self.quality_aggregate is not None
                else None
            ),
            "quality_band": self.quality_band,
            "face_min_side_px": (
                round(self.face_min_side_px, 1)
                if self.face_min_side_px is not None
                else None
            ),
            "probe_relationship": (
                self.probe_relationship.to_dict() if self.probe_relationship else None
            ),
            "warnings": [w.value for w in self.warnings],
            "warning_details": list(self.warning_details),
        }


def _fail(url: str, issue: SearchImageIssue, detail: str, **extra) -> SearchImageCheck:
    return SearchImageCheck(url=url, ok=False, issue=issue, detail=detail, **extra)


def check_search_image(
    url: str,
    face_engine,
    *,
    probe_analysis=None,
    probe_bytes: bytes | None = None,
    fetch_policy: FetchPolicy | None = None,
    work_dir: str | Path | None = None,
    min_face_px: float = MIN_SEARCH_FACE_PX,
    min_quality: float = MIN_SEARCH_QUALITY,
) -> SearchImageCheck:
    """Validate that `url` is fit to be handed to a search provider.

    `face_engine` is injected so tests can exercise every branch without
    loading a 275 MB model.

    `probe_analysis` and `probe_bytes` are optional. When supplied, the probe
    relationship is MEASURED and reported -- never judged.
    """
    import tempfile

    acquisition = fetch_media(url, policy=fetch_policy or FetchPolicy())
    if not acquisition.ok:
        return _fail(
            url,
            SearchImageIssue.URL_UNFETCHABLE,
            "{0} ({1})".format(
                acquisition.detail,
                acquisition.reason.value if acquisition.reason else "unknown",
            ),
            http_status=acquisition.status_code,
        )

    content = acquisition.content or b""
    validation = validate_image_bytes(
        content, declared_content_type=acquisition.declared_content_type
    )
    if not validation.ok:
        return _fail(
            url,
            SearchImageIssue.NOT_AN_IMAGE,
            validation.detail,
            http_status=acquisition.status_code,
            byte_size=len(content),
            content_sha256=acquisition.sha256,
        )

    base = dict(
        http_status=acquisition.status_code,
        byte_size=len(content),
        image_format=validation.detected_format,
        width=validation.width,
        height=validation.height,
        content_sha256=acquisition.sha256,
    )

    # Write to a temp file: FaceEngine.analyze() takes a path, and this is a
    # transient inspection copy -- it is NOT evidence and never enters the CAS.
    directory = Path(work_dir) if work_dir else Path(tempfile.gettempdir())
    directory.mkdir(parents=True, exist_ok=True)
    scratch = directory / "tracelock_search_image_{0}.bin".format(
        (acquisition.sha256 or "unknown")[:16]
    )
    scratch.write_bytes(content)

    from tracelock.face.errors import FaceEngineError, NoFaceDetectedError

    try:
        analysis = face_engine.analyze(scratch)
    except NoFaceDetectedError as exc:
        return _fail(
            url,
            SearchImageIssue.NO_FACE_DETECTED,
            "{0}. The provider would be seeded with a faceless image.".format(exc),
            **base,
        )
    except FaceEngineError as exc:
        return _fail(
            url, SearchImageIssue.FACE_ANALYSIS_FAILED, str(exc), **base
        )
    finally:
        scratch.unlink(missing_ok=True)

    # --- input-fitness gate -------------------------------------------
    # A detection is not automatically a usable seed. Checked BEFORE any
    # warnings so an unusable image fails rather than merely complains.
    quality = analysis.primary.quality.aggregate
    min_side = analysis.primary.bbox.min_side

    if min_side < min_face_px or quality < min_quality:
        return _fail(
            url,
            SearchImageIssue.FACE_NOT_USABLE,
            "detected face is {0:.0f}px on its shorter side with quality "
            "{1:.3f} (det {2:.3f}); floors are {3}px and {4:.2f}. Too "
            "degraded to seed a meaningful search.".format(
                min_side, quality, float(analysis.primary.det_score),
                min_face_px, min_quality,
            ),
            faces_detected=analysis.faces_detected,
            det_score=float(analysis.primary.det_score),
            quality_aggregate=quality,
            quality_band=analysis.primary.quality.band.value,
            face_min_side_px=min_side,
            **base,
        )

    warnings: list[SearchImageWarning] = []
    details: list[str] = []

    if analysis.faces_detected > 1:
        warnings.append(SearchImageWarning.MULTIPLE_FACES)
        details.append(
            "{0} faces in the search image; the provider cannot be told which "
            "one to match".format(analysis.faces_detected)
        )

    if analysis.selection.ambiguous:
        warnings.append(SearchImageWarning.AMBIGUOUS_PRIMARY_FACE)
        details.append(
            "primary-face selection margin {0:.4f} is below the ambiguity "
            "threshold".format(analysis.selection.margin)
        )

    if quality < WARN_QUALITY_BELOW:
        warnings.append(SearchImageWarning.LOW_FACE_QUALITY)
        details.append(
            "search image face quality {0:.3f} is low; expect weak results".format(quality)
        )

    if min_side < WARN_FACE_PX_BELOW:
        warnings.append(SearchImageWarning.SMALL_FACE)
        details.append(
            "face is only {0:.0f}px on its shorter side in the search "
            "image".format(min_side)
        )

    relationship = _observe_probe_relationship(
        analysis, content, probe_analysis, probe_bytes
    )
    if relationship and relationship.is_same_image is False:
        # FACTUAL, not evaluative: a different image is a different image.
        warnings.append(SearchImageWarning.DIFFERS_FROM_PROBE)
        details.append(
            "the searched image is not the same image as the local probe "
            "(pHash distance {0}/64); discovery and verification will use "
            "different pictures".format(relationship.phash_distance)
        )

    return SearchImageCheck(
        url=url,
        ok=True,
        faces_detected=analysis.faces_detected,
        det_score=float(analysis.primary.det_score),
        quality_aggregate=quality,
        quality_band=analysis.primary.quality.band.value,
        face_min_side_px=min_side,
        probe_relationship=relationship,
        warnings=tuple(warnings),
        warning_details=tuple(details),
        **base,
    )


def _observe_probe_relationship(
    search_analysis, search_bytes: bytes, probe_analysis, probe_bytes: bytes | None
) -> ProbeRelationship | None:
    """Measure search-image vs probe. Returns numbers, never a verdict."""
    if probe_analysis is None and probe_bytes is None:
        return None

    similarity = None
    if probe_analysis is not None:
        try:
            from tracelock.face.similarity import cosine_similarity

            similarity = cosine_similarity(
                probe_analysis.embedding, search_analysis.embedding
            )
        except Exception:
            similarity = None

    distance = None
    same_image = None
    if probe_bytes:
        try:
            from tracelock.verification.phash import phash_from_bytes

            distance = phash_from_bytes(probe_bytes).distance(
                phash_from_bytes(search_bytes)
            )
            # Purely visual and purely factual: is this the same picture?
            same_image = distance <= PHASH_SAME_IMAGE_MAX_DISTANCE
        except ValueError:
            distance = None

    return ProbeRelationship(
        cosine_similarity=similarity,
        phash_distance=distance,
        is_same_image=same_image,
    )
