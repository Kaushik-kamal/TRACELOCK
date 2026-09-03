"""Phase 3 -- evidence aggregation.

Turns a list of per-candidate verification results into a structured body of
evidence about ONE subject, without double-counting anything.

THE TWO WAYS AGGREGATION GOES WRONG
-----------------------------------
Both are easy, both inflate confidence, and this module exists to prevent them:

  1. COUNTING THE SAME IMAGE TWICE
     A widely syndicated photograph appears on ten sites. That is ONE
     photograph, republished ten times. Counting ten pieces of evidence would
     multiply a single observation into a false consensus.

  2. COUNTING THE SAME PUBLISHER TWICE
     Four URLs on narendramodi.in are four pages from ONE publisher. A
     publisher agreeing with itself is not corroboration.

So evidence is grouped twice, on different keys:

     by CONTENT      sha256 of the downloaded bytes  -> unique images
     by PUBLISHER    registrable domain (eTLD+1)     -> independent sources

and corroboration counts only the second, restricted to candidates whose
relation is genuinely independent (see below).

WHAT COUNTS AS INDEPENDENT CORROBORATION
----------------------------------------
Phase 2 classifies each candidate into the 2x2 relation. Only
SAME_PERSON_DIFFERENT_PHOTO is independent evidence of identity:

  SAME_PHOTO_REPUBLISHED       provenance -- proves the image travelled, adds
                               NO new identity information, because it is the
                               same pixels the probe already contained
  SAME_PERSON_DIFFERENT_PHOTO  a different camera, pose and moment producing a
                               matching face. This is the only quadrant that
                               tells us something new
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from tracelock.core.reasons import RejectionReason, VerificationStatus
from tracelock.verification.relation import EvidenceRelation

# Corroboration saturation rate. Going from one independent publisher to two is
# a large epistemic jump; four to five is nearly nothing. lambda=0.5 gives
# 1->0.00, 2->0.39, 3->0.63, 5->0.86, 10->0.99.
CORROBORATION_LAMBDA = 0.5


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """One unique piece of verified evidence, after de-duplication."""

    content_sha256: str
    candidate_ids: tuple[str, ...]
    domains: tuple[str, ...]
    source_urls: tuple[str, ...]
    media_urls: tuple[str, ...]

    identity_probability: float | None
    face_similarity: float
    face_quality: float
    relation: str
    phash_distance: int | None
    cas_path: str | None

    @property
    def is_independent_corroboration(self) -> bool:
        return self.relation == EvidenceRelation.SAME_PERSON_DIFFERENT_PHOTO.value

    @property
    def publisher_count(self) -> int:
        return len(set(self.domains))

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_sha256": self.content_sha256,
            "candidate_ids": list(self.candidate_ids),
            "domains": list(self.domains),
            "publisher_count": self.publisher_count,
            "source_urls": list(self.source_urls),
            "media_urls": list(self.media_urls),
            "identity_probability": (
                round(self.identity_probability, 6)
                if self.identity_probability is not None
                else None
            ),
            "face_similarity": round(self.face_similarity, 6),
            "face_quality": round(self.face_quality, 4),
            "relation": self.relation,
            "is_independent_corroboration": self.is_independent_corroboration,
            "phash_distance": self.phash_distance,
            "cas_path": self.cas_path,
        }


@dataclass(frozen=True, slots=True)
class FunnelCounts:
    """The pipeline funnel. Every candidate is in exactly one terminal bucket."""

    discovered: int
    downloaded: int
    validated: int
    analysed: int
    verified: int
    inconclusive: int
    rejected: int
    duplicates: int

    # Post-aggregation
    unique_images: int
    independent_publishers: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "discovered": self.discovered,
            "downloaded": self.downloaded,
            "validated": self.validated,
            "analysed": self.analysed,
            "verified": self.verified,
            "inconclusive": self.inconclusive,
            "rejected": self.rejected,
            "duplicates": self.duplicates,
            "unique_images": self.unique_images,
            "independent_publishers": self.independent_publishers,
        }


@dataclass(frozen=True, slots=True)
class AggregatedEvidence:
    """Everything Phase 3 needs to score, with the double-counting removed."""

    items: tuple[EvidenceItem, ...]
    funnel: FunnelCounts
    rejection_breakdown: dict[str, int]
    independent_domains: tuple[str, ...]
    all_verified_domains: tuple[str, ...]
    duplicate_groups: dict[str, list[str]] = field(default_factory=dict)

    @property
    def has_evidence(self) -> bool:
        return bool(self.items)

    @property
    def corroborating_items(self) -> tuple[EvidenceItem, ...]:
        return tuple(i for i in self.items if i.is_independent_corroboration)

    @property
    def independent_publisher_count(self) -> int:
        return len(self.independent_domains)

    def corroboration_factor(self, lam: float = CORROBORATION_LAMBDA) -> float:
        """C in [0, 1). Saturating in the number of INDEPENDENT publishers.

        One publisher yields exactly 0: a single source is an assertion, not
        corroboration. The curve saturates so a large number of sources cannot
        run away with the score.
        """
        n = self.independent_publisher_count
        if n <= 1:
            return 0.0
        return 1.0 - math.exp(-lam * (n - 1))

    def strongest(self) -> EvidenceItem | None:
        """Highest identity probability among unique verified images."""
        if not self.items:
            return None
        return max(
            self.items,
            key=lambda i: (i.identity_probability or 0.0, i.face_quality),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "funnel": self.funnel.to_dict(),
            "rejection_breakdown": self.rejection_breakdown,
            "independent_domains": list(self.independent_domains),
            "all_verified_domains": list(self.all_verified_domains),
            "corroboration_factor": round(self.corroboration_factor(), 6),
            "duplicate_groups": self.duplicate_groups,
            "items": [i.to_dict() for i in self.items],
        }


def aggregate(results: Iterable[dict[str, Any]]) -> AggregatedEvidence:
    """Aggregate serialized VerificationResult dicts into scored-ready evidence.

    Takes dicts rather than objects so it can run directly on a persisted
    Phase 2 artifact -- the aggregation is reproducible from the JSON alone,
    with no need to re-download or re-analyse anything.
    """
    results = list(results)

    verified = [r for r in results if r["status"] == VerificationStatus.VERIFIED_CANDIDATE.value]
    inconclusive = [r for r in results if r["status"] == VerificationStatus.INCONCLUSIVE.value]
    rejected = [r for r in results if r["status"] == VerificationStatus.REJECTED.value]

    downloaded = [r for r in results if (r.get("acquisition") or {}).get("ok")]
    validated = [r for r in results if (r.get("validation") or {}).get("ok")]
    analysed = [r for r in results if r.get("face")]

    duplicates = [
        r for r in rejected
        if any(x["reason"] == RejectionReason.DUPLICATE_CONTENT.value
               for x in r.get("rejection_reasons", []))
    ]

    breakdown: dict[str, int] = {}
    for result in rejected:
        for reason in result.get("rejection_reasons", []):
            key = reason["reason"]
            breakdown[key] = breakdown.get(key, 0) + 1

    # --- group VERIFIED evidence by content hash --------------------------
    # This is where republication collapses into one observation.
    by_content: dict[str, list[dict[str, Any]]] = {}
    for result in verified:
        digest = result.get("content_sha256")
        if digest:
            by_content.setdefault(digest, []).append(result)

    # Duplicates rejected at VALIDATED still contribute their PUBLISHER to the
    # group they duplicated: the same photo on a second domain is a second
    # place it was published, even though it is not a second image.
    for result in duplicates:
        digest = result.get("content_sha256")
        if digest and digest in by_content:
            by_content[digest].append(result)

    items: list[EvidenceItem] = []
    duplicate_groups: dict[str, list[str]] = {}

    for digest, group in by_content.items():
        primary = next(
            (g for g in group if g["status"] == VerificationStatus.VERIFIED_CANDIDATE.value),
            group[0],
        )
        domains = tuple(
            d for d in (
                (g.get("provenance") or {}).get("registrable_domain") for g in group
            ) if d
        )
        if len(group) > 1:
            duplicate_groups[digest] = [g["candidate_id"] for g in group]

        relation = (primary.get("relation") or {}).get("relation") or "UNDETERMINED"

        items.append(
            EvidenceItem(
                content_sha256=digest,
                candidate_ids=tuple(g["candidate_id"] for g in group),
                domains=domains,
                source_urls=tuple(g.get("source_url") or "" for g in group),
                media_urls=tuple(g.get("media_url") or "" for g in group),
                identity_probability=primary.get("identity_probability"),
                face_similarity=float(primary.get("face_similarity") or 0.0),
                face_quality=float((primary.get("face") or {}).get("quality_aggregate") or 0.0),
                relation=relation,
                phash_distance=(primary.get("relation") or {}).get("phash_distance"),
                cas_path=primary.get("cas_path"),
            )
        )

    items.sort(key=lambda i: (i.identity_probability or 0.0), reverse=True)

    # --- independent publishers -------------------------------------------
    # Only unique images whose relation is genuinely independent contribute.
    independent_domains = sorted(
        {d for item in items if item.is_independent_corroboration for d in item.domains}
    )
    all_domains = sorted({d for item in items for d in item.domains})

    funnel = FunnelCounts(
        discovered=len(results),
        downloaded=len(downloaded),
        validated=len(validated),
        analysed=len(analysed),
        verified=len(verified),
        inconclusive=len(inconclusive),
        rejected=len(rejected),
        duplicates=len(duplicates),
        unique_images=len(items),
        independent_publishers=len(independent_domains),
    )

    return AggregatedEvidence(
        items=tuple(items),
        funnel=funnel,
        rejection_breakdown=breakdown,
        independent_domains=tuple(independent_domains),
        all_verified_domains=tuple(all_domains),
        duplicate_groups=duplicate_groups,
    )
