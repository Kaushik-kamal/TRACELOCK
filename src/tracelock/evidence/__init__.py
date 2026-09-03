"""Phase 3 -- evidence aggregation and explainable trust scoring.

Duplicates count once. One publisher, one vote. No score without calibration.
"""

from tracelock.evidence.aggregate import (
    CORROBORATION_LAMBDA,
    AggregatedEvidence,
    EvidenceItem,
    FunnelCounts,
    aggregate,
)
from tracelock.evidence.trust import (
    EvidenceQuality,
    QualityComponent,
    TrustBand,
    TrustScore,
    UncalibratedScoreRefused,
    assess_quality,
    measure_acquisition_integrity,
    measure_metadata_completeness,
    score_evidence,
)

__all__ = [
    "aggregate", "AggregatedEvidence", "EvidenceItem", "FunnelCounts",
    "CORROBORATION_LAMBDA",
    "score_evidence", "TrustScore", "TrustBand", "EvidenceQuality",
    "QualityComponent", "assess_quality", "UncalibratedScoreRefused",
    "measure_metadata_completeness", "measure_acquisition_integrity",
]
