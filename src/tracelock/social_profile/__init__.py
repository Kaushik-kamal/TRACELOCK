"""Tier-1 public profile evidence: additive, unanchored, verification-gated.

See `models.py` for the tier semantics and `relate.py` for the single gate
that keeps rejected candidates out of this package entirely.
"""

from tracelock.social_profile.hints import extract_hints
from tracelock.social_profile.models import (
    EvidenceChainStep,
    ExtractionMethod,
    ProfileEvidenceSummary,
    ProfileHint,
    ProfileRelationship,
    ProfileTier,
)
from tracelock.social_profile.relate import relate

__all__ = [
    "extract_hints",
    "relate",
    "EvidenceChainStep",
    "ExtractionMethod",
    "ProfileEvidenceSummary",
    "ProfileHint",
    "ProfileRelationship",
    "ProfileTier",
]
