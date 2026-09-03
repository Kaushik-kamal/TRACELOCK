"""The verification firewall.

DISCOVERY MAKES CLAIMS. TRACELOCK RE-MEASURES EVIDENCE.

Nothing here converts a similarity into an identity probability. Thresholds are
provisional operational filters, stamped uncalibrated, and recorded as such.
"""

from tracelock.verification.models import (
    DuplicateInfo,
    FaceEvidence,
    StageHistory,
    StageRecord,
    VerificationResult,
)
from tracelock.verification.phash import (
    PerceptualHash,
    compute_phash,
    hamming_distance,
    is_near_duplicate,
    phash_from_bytes,
)
from tracelock.verification.policy import (
    DEFAULT_POLICY,
    SimilarityBand,
    VerificationPolicy,
)
from tracelock.core.reasons import (
    RejectionReason,
    Stage,
    StageOutcome,
    VerificationStatus,
    reasons_for_stage,
)
from tracelock.verification.relation import (
    EvidenceRelation,
    RelationAssessment,
    assess_relation,
)
from tracelock.verification.verifier import CandidateVerifier

__all__ = [
    "CandidateVerifier", "VerificationResult", "FaceEvidence", "DuplicateInfo",
    "StageHistory", "StageRecord",
    "Stage", "StageOutcome", "VerificationStatus", "RejectionReason",
    "reasons_for_stage",
    "VerificationPolicy", "DEFAULT_POLICY", "SimilarityBand",
    "PerceptualHash", "compute_phash", "phash_from_bytes", "hamming_distance",
    "is_near_duplicate",
    "EvidenceRelation", "RelationAssessment", "assess_relation",
]
