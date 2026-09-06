"""Structured verification results.

INVARIANT: every candidate that enters produces exactly one VerificationResult.
There is no path where a failure disappears. `StageHistory` records what
happened at each stage, so a rejected candidate can be interrogated as fully as
a surviving one -- which is the point of the phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from tracelock.acquisition.fetcher import AcquisitionResult
from tracelock.acquisition.provenance import SourceProvenance
from tracelock.acquisition.validation import ImageValidation
from tracelock.verification.policy import SimilarityBand
from tracelock.verification.relation import RelationAssessment
from tracelock.core.reasons import (
    RejectionReason,
    Stage,
    StageOutcome,
    VerificationStatus,
)

SCHEMA_VERSION = "verification-result/1"


@dataclass(frozen=True, slots=True)
class StageRecord:
    """What happened at one stage."""

    stage: Stage
    outcome: StageOutcome
    detail: str = ""
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "outcome": self.outcome.value,
            "detail": self.detail,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


@dataclass(slots=True)
class StageHistory:
    """Ordered record of a candidate's progress. Mutable during the run only."""

    records: list[StageRecord] = field(default_factory=list)

    def record(
        self,
        stage: Stage,
        outcome: StageOutcome,
        detail: str = "",
        elapsed: float = 0.0,
    ) -> None:
        self.records.append(StageRecord(stage, outcome, detail, elapsed))

    def mark_skipped(self, from_stage: Stage) -> None:
        """Mark every later stage as never-run.

        Explicit SKIPPED entries mean the absence of a measurement is itself
        recorded, rather than being inferred from a missing field.
        """
        for stage in Stage:
            if stage.order >= from_stage.order and not self.has(stage):
                self.records.append(StageRecord(stage, StageOutcome.SKIPPED))

    def has(self, stage: Stage) -> bool:
        return any(r.stage is stage for r in self.records)

    @property
    def furthest_stage(self) -> Stage:
        reached = [r.stage for r in self.records if r.outcome is StageOutcome.OK]
        return max(reached, key=lambda s: s.order) if reached else Stage.DISCOVERED

    @property
    def failed_stage(self) -> Stage | None:
        failed = [r.stage for r in self.records if r.outcome is StageOutcome.FAILED]
        return failed[0] if failed else None

    def to_list(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.records]


@dataclass(frozen=True, slots=True)
class FaceEvidence:
    """Face measurements for a candidate. No raw embedding.

    The 512-d vector is biometric data and is deliberately absent from this
    structure so it cannot reach a serialized artifact by accident. Phase 4
    commits the quantized digest, never the floats.
    """

    faces_detected: int
    det_score: float
    bbox: dict[str, float]
    quality_aggregate: float
    quality_band: str
    embedding_dimension: int
    embedding_model_id: str
    embedding_quantized_sha256: str
    selection_ambiguous: bool
    selection_margin: float
    warnings: tuple[str, ...] = ()
    pose_deviation_deg: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "faces_detected": self.faces_detected,
            "det_score": round(self.det_score, 4),
            "bbox": self.bbox,
            "quality_aggregate": round(self.quality_aggregate, 4),
            "quality_band": self.quality_band,
            "embedding_dimension": self.embedding_dimension,
            "embedding_model_id": self.embedding_model_id,
            # A commitment to the embedding, not the embedding.
            "embedding_quantized_sha256": self.embedding_quantized_sha256,
            "selection_ambiguous": self.selection_ambiguous,
            "selection_margin": round(self.selection_margin, 4),
            "pose_deviation_deg": (
                round(self.pose_deviation_deg, 3)
                if self.pose_deviation_deg is not None
                else None
            ),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class DuplicateInfo:
    """Exact and near-duplicate signals, kept separate.

    Two files can have DIFFERENT bytes and be VISUALLY IDENTICAL (recompression,
    resizing). Collapsing these would lose the distinction between a byte-exact
    republication and a re-encoded one.
    """

    is_exact_duplicate: bool
    duplicate_of_candidate_id: str | None = None
    near_duplicate_of_candidate_ids: tuple[str, ...] = ()
    cas_reference_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_exact_duplicate": self.is_exact_duplicate,
            "duplicate_of_candidate_id": self.duplicate_of_candidate_id,
            "near_duplicate_of_candidate_ids": list(self.near_duplicate_of_candidate_ids),
            "cas_reference_count": self.cas_reference_count,
        }


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Everything measured about one candidate."""

    schema_version: str
    candidate_id: str
    provider: str
    rank: int | None
    source_url: str
    media_url: str | None

    status: VerificationStatus
    rejection_reasons: tuple[RejectionReason, ...]
    stage_history: list[dict[str, Any]]
    furthest_stage: Stage
    failed_stage: Stage | None

    acquisition: dict[str, Any] | None
    validation: dict[str, Any] | None
    provenance: dict[str, Any] | None
    face: FaceEvidence | None
    relation: RelationAssessment | None
    duplicate: DuplicateInfo | None

    content_sha256: str | None
    cas_path: str | None
    phash: str | None

    face_similarity: float | None
    similarity_band: SimilarityBand | None
    # Calibrated P(same identity | similarity). None when no calibration model
    # is loaded -- without one there IS no probability, and inventing a number
    # would be exactly the failure this project guards against.
    identity_probability: float | None

    warnings: tuple[str, ...]
    verified_at: datetime
    elapsed_seconds: float

    # The title/author/snippet/publish-date text the DISCOVERY provider
    # (SerpAPI) returned alongside this candidate's URL. Fetched at discovery
    # time regardless of whether anything downstream reads it; carried here
    # unmodified so `social_profile` can mine it without a second network
    # call. Never influences status, similarity, or any threshold -- it is a
    # passenger, not an input to verification.
    discovery_metadata: dict[str, Any] | None = None

    @property
    def is_verified(self) -> bool:
        return self.status is VerificationStatus.VERIFIED_CANDIDATE

    @property
    def is_rejected(self) -> bool:
        return self.status is VerificationStatus.REJECTED

    @property
    def primary_reason(self) -> RejectionReason | None:
        return self.rejection_reasons[0] if self.rejection_reasons else None

    def explain(self) -> str:
        """One-sentence human explanation. Drives the demo table."""
        if self.status is VerificationStatus.VERIFIED_CANDIDATE:
            relation = self.relation.relation.value if self.relation else "UNDETERMINED"
            return "survived verification as {0} (face similarity {1:.4f}, " \
                   "PROVISIONAL)".format(relation, self.face_similarity or 0.0)

        if self.status is VerificationStatus.INCONCLUSIVE:
            return (
                "measured but inconclusive: face similarity {0:.4f} falls in the "
                "uncalibrated indeterminate band".format(self.face_similarity or 0.0)
            )

        reason = self.primary_reason
        if reason is None:
            return "rejected without a recorded reason (this is a bug)"
        return "rejected at stage {0}: {1}".format(reason.stage.value, reason.explanation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "provider": self.provider,
            "rank": self.rank,
            "source_url": self.source_url,
            "media_url": self.media_url,
            "status": self.status.value,
            "explanation": self.explain(),
            "rejection_reasons": [
                {
                    "reason": r.value,
                    "stage": r.stage.value,
                    "explanation": r.explanation,
                }
                for r in self.rejection_reasons
            ],
            "furthest_stage": self.furthest_stage.value,
            "failed_stage": self.failed_stage.value if self.failed_stage else None,
            "stage_history": self.stage_history,
            "content_sha256": self.content_sha256,
            "cas_path": self.cas_path,
            "phash": self.phash,
            "face_similarity": (
                round(self.face_similarity, 6)
                if self.face_similarity is not None
                else None
            ),
            "similarity_band": (
                self.similarity_band.value if self.similarity_band else None
            ),
            "identity_probability": (
                round(self.identity_probability, 6)
                if self.identity_probability is not None
                else None
            ),
            "acquisition": self.acquisition,
            "validation": self.validation,
            "provenance": self.provenance,
            "face": self.face.to_dict() if self.face else None,
            "relation": self.relation.to_dict() if self.relation else None,
            "duplicate": self.duplicate.to_dict() if self.duplicate else None,
            "warnings": list(self.warnings),
            "verified_at": self.verified_at.isoformat(),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "discovery_metadata": self.discovery_metadata,
        }


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def build_provenance_dict(provenance: SourceProvenance | None) -> dict[str, Any] | None:
    return provenance.to_dict() if provenance else None


def build_acquisition_dict(
    acquisition: AcquisitionResult | None,
) -> dict[str, Any] | None:
    return acquisition.to_dict() if acquisition else None


def build_validation_dict(
    validation: ImageValidation | None,
) -> dict[str, Any] | None:
    return validation.to_dict() if validation else None
