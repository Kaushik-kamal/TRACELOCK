"""The 2x2 evidence relation.

Two INDEPENDENT measurements, deliberately never collapsed into one number:

    face similarity    ArcFace cosine  -- is this the same person?
    pHash similarity   DCT Hamming     -- is this the same photograph?

                    LOW pHash                  HIGH pHash
                    (different image)          (same image)
    HIGH face  |  SAME_PERSON_DIFFERENT_PHOTO | SAME_PHOTO_REPUBLISHED
    LOW  face  |  UNRELATED                   | VISUAL_MATCH_FACE_MISMATCH

WHY THIS MATTERS MORE THAN IT LOOKS
-----------------------------------
The two "high face similarity" quadrants carry very different evidentiary
weight, and averaging them into a single score destroys exactly the information
that makes the finding useful:

  SAME_PHOTO_REPUBLISHED
      Provenance. Proves the image travelled -- who reposted it and where.
      It contributes NO independent identity information, because it is
      literally the same pixels the probe already contained.

  SAME_PERSON_DIFFERENT_PHOTO
      Genuine corroboration, and much stronger. A different camera, pose and
      moment producing a matching face is independent evidence of identity.

  VISUAL_MATCH_FACE_MISMATCH
      The anomaly quadrant, and the most interesting one. Near-identical
      imagery whose faces do NOT match means a crop, a different subject in a
      shared template, a stock photo, or a deliberately altered image. Worth
      surfacing rather than discarding.

The Phase 3 trust score is expected to weight these differently. Phase 2's job
is to preserve the distinction, not to price it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class EvidenceRelation(str, Enum):
    SAME_PHOTO_REPUBLISHED = "SAME_PHOTO_REPUBLISHED"
    SAME_PERSON_DIFFERENT_PHOTO = "SAME_PERSON_DIFFERENT_PHOTO"
    VISUAL_MATCH_FACE_MISMATCH = "VISUAL_MATCH_FACE_MISMATCH"
    UNRELATED = "UNRELATED"
    UNDETERMINED = "UNDETERMINED"

    @property
    def explanation(self) -> str:
        return _RELATION_EXPLANATION[self]

    @property
    def is_independent_corroboration(self) -> bool:
        """Does this relation add identity evidence beyond the probe itself?

        Only a different photograph does. A republished copy of the same image
        is provenance, not corroboration -- and Phase 3 must not double-count
        it as though it were.
        """
        return self is EvidenceRelation.SAME_PERSON_DIFFERENT_PHOTO


_RELATION_EXPLANATION: dict[EvidenceRelation, str] = {
    EvidenceRelation.SAME_PHOTO_REPUBLISHED: (
        "high face similarity AND high visual similarity: the same photograph "
        "appearing elsewhere. Evidence of distribution, not of identity"
    ),
    EvidenceRelation.SAME_PERSON_DIFFERENT_PHOTO: (
        "high face similarity with LOW visual similarity: a different "
        "photograph whose face matches. The strongest single-candidate "
        "evidence this system can produce"
    ),
    EvidenceRelation.VISUAL_MATCH_FACE_MISMATCH: (
        "near-identical imagery whose faces do NOT match. An anomaly worth "
        "investigating: a crop, a shared template, a stock image, or an "
        "altered picture"
    ),
    EvidenceRelation.UNRELATED: (
        "neither the face nor the image matches. The provider's suggested "
        "relationship is not reproducible from the bytes we downloaded"
    ),
    EvidenceRelation.UNDETERMINED: (
        "one or both signals could not be measured, so no relation can be "
        "assigned"
    ),
}


@dataclass(frozen=True, slots=True)
class RelationAssessment:
    """The 2x2 classification, with both raw signals preserved."""

    relation: EvidenceRelation
    face_similarity: float | None
    phash_similarity: float | None
    phash_distance: int | None
    face_signal_high: bool | None
    phash_signal_high: bool | None
    provisional: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation": self.relation.value,
            "explanation": self.relation.explanation,
            "is_independent_corroboration": self.relation.is_independent_corroboration,
            # Both raw signals are always emitted separately. Downstream code
            # must be able to re-derive the quadrant rather than trust it.
            "face_similarity": (
                round(self.face_similarity, 6) if self.face_similarity is not None else None
            ),
            "phash_similarity": (
                round(self.phash_similarity, 6) if self.phash_similarity is not None else None
            ),
            "phash_distance": self.phash_distance,
            "face_signal_high": self.face_signal_high,
            "phash_signal_high": self.phash_signal_high,
            "provisional": self.provisional,
        }


def assess_relation(
    *,
    face_similarity: float | None,
    phash_distance: int | None,
    face_high_threshold: float,
    phash_near_duplicate_max_distance: int,
) -> RelationAssessment:
    """Place a candidate in the 2x2. Thresholds are injected, never assumed.

    Both thresholds are PROVISIONAL and supplied by the caller's policy, so the
    run artifact records exactly which values produced the quadrant.
    """
    if face_similarity is None or phash_distance is None:
        return RelationAssessment(
            relation=EvidenceRelation.UNDETERMINED,
            face_similarity=face_similarity,
            phash_similarity=(
                1.0 - phash_distance / 64.0 if phash_distance is not None else None
            ),
            phash_distance=phash_distance,
            face_signal_high=None,
            phash_signal_high=None,
        )

    phash_similarity = 1.0 - phash_distance / 64.0
    face_high = face_similarity >= face_high_threshold
    phash_high = phash_distance <= phash_near_duplicate_max_distance

    if face_high and phash_high:
        relation = EvidenceRelation.SAME_PHOTO_REPUBLISHED
    elif face_high and not phash_high:
        relation = EvidenceRelation.SAME_PERSON_DIFFERENT_PHOTO
    elif not face_high and phash_high:
        relation = EvidenceRelation.VISUAL_MATCH_FACE_MISMATCH
    else:
        relation = EvidenceRelation.UNRELATED

    return RelationAssessment(
        relation=relation,
        face_similarity=face_similarity,
        phash_similarity=phash_similarity,
        phash_distance=phash_distance,
        face_signal_high=face_high,
        phash_signal_high=phash_high,
    )
