"""Tier classification: turn extracted hints into an honest verdict.

THE GATE, IN ONE PLACE
-----------------------
`relate()` is the ONLY function in this package that reads a candidate's
`status`. Every other function in `hints.py` and `models.py` is a pure
transform that does not know or care whether a candidate was verified,
rejected, or never analysed. Putting the gate here, once, means it cannot be
duplicated-and-drifted across multiple call sites, and a single test
(`test_rejected_never_reaches_relate`) can assert the whole package's most
important property in one place.

Rejected and inconclusive candidates are filtered out BEFORE hint extraction
even runs -- not merely excluded from the tier decision afterward. A hint
extracted from a rejected candidate's metadata is never computed, let alone
shown.

WHY A PROFILE-SHAPED URL ALONE IS NOT ENOUGH
----------------------------------------------
Reaching VERIFIED_HIGH_CONFIDENCE requires TWO things to both be true of the
SAME candidate: its `status` is VERIFIED_CANDIDATE (the calibrated face
verification already cleared it), AND its url is profile-shaped. A
profile-shaped URL on a REJECTED or INCONCLUSIVE candidate produces nothing
here at all -- the loop below never even looks at its URL, because it never
reaches the hint-extraction step.
"""

from __future__ import annotations

from typing import Any

from tracelock.core.reasons import VerificationStatus
from tracelock.social_profile.hints import extract_hints
from tracelock.social_profile.models import (
    EvidenceChainStep,
    ExtractionMethod,
    ProfileHint,
    ProfileRelationship,
    ProfileEvidenceSummary,
    ProfileTier,
)


def _normalized_handle(handle: str) -> str:
    return handle.strip().lstrip("@").lower()


def _agreeing_methods(hints: list[ProfileHint]) -> tuple[ProfileHint, ProfileHint] | None:
    """Two hints, from DIFFERENT extraction methods, naming the same handle.

    Independent agreement is the only thing that promotes a candidate past
    DISCOVERED_LINK without its url itself being the profile page -- the same
    "independent sources" principle `evidence.aggregate` already applies to
    photo corroboration, applied here to a handle instead of a publisher
    domain.
    """
    by_method: dict[ExtractionMethod, list[ProfileHint]] = {}
    for hint in hints:
        if hint.handle:
            by_method.setdefault(hint.method, []).append(hint)

    methods = list(by_method)
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            for hint_a in by_method[methods[i]]:
                for hint_b in by_method[methods[j]]:
                    if _normalized_handle(hint_a.handle) == _normalized_handle(hint_b.handle):
                        return hint_a, hint_b
    return None


def _verification_step(result: dict[str, Any]) -> EvidenceChainStep:
    similarity = result.get("face_similarity")
    sim_text = "{0:.4f}".format(similarity) if similarity is not None else "unavailable"
    return EvidenceChainStep(
        claim=(
            "This candidate independently cleared calibrated face "
            "verification (similarity {0}).".format(sim_text)
        ),
        source="verification.status = VERIFIED_CANDIDATE (candidate {0})".format(
            result.get("candidate_id", "?")
        ),
    )


def _hint_step(hint: ProfileHint) -> EvidenceChainStep:
    return EvidenceChainStep(
        claim=hint.method.explanation,
        source=hint.source_field,
        quoted_text=hint.quoted_text,
    )


def _classify(result: dict[str, Any], hints: list[ProfileHint]) -> ProfileRelationship | None:
    profile_shaped = next(
        (h for h in hints if h.method is ExtractionMethod.URL_IS_PROFILE_SHAPED), None
    )
    if profile_shaped is not None:
        return ProfileRelationship(
            tier=ProfileTier.VERIFIED_HIGH_CONFIDENCE,
            platform=profile_shaped.platform,
            handle=profile_shaped.handle,
            profile_url=profile_shaped.profile_url,
            source_candidate_id=result.get("candidate_id", ""),
            source_url=result.get("source_url", ""),
            face_similarity=result.get("face_similarity"),
            evidence_chain=(
                _verification_step(result),
                _hint_step(profile_shaped),
            ),
        )

    agreement = _agreeing_methods(hints)
    if agreement is not None:
        hint_a, hint_b = agreement
        chosen = hint_a if hint_a.profile_url else hint_b
        return ProfileRelationship(
            tier=ProfileTier.UNVERIFIED_POSSIBLE_MATCH,
            platform=chosen.platform,
            handle=chosen.handle,
            profile_url=chosen.profile_url,
            source_candidate_id=result.get("candidate_id", ""),
            source_url=result.get("source_url", ""),
            face_similarity=result.get("face_similarity"),
            evidence_chain=(
                _verification_step(result),
                _hint_step(hint_a),
                _hint_step(hint_b),
                EvidenceChainStep(
                    claim=(
                        "Two independent extraction methods ({0} and {1}) "
                        "agree on the same handle.".format(
                            hint_a.method.value, hint_b.method.value
                        )
                    ),
                    source="social_profile.relate agreement rule",
                ),
            ),
        )

    if hints:
        sole = hints[0]
        return ProfileRelationship(
            tier=ProfileTier.DISCOVERED_LINK,
            platform=sole.platform,
            handle=sole.handle,
            profile_url=sole.profile_url,
            source_candidate_id=result.get("candidate_id", ""),
            source_url=result.get("source_url", ""),
            face_similarity=result.get("face_similarity"),
            evidence_chain=(
                _verification_step(result),
                _hint_step(sole),
            ),
        )

    return None


def relate(results: list[dict[str, Any]]) -> ProfileEvidenceSummary:
    """Tier-1 public profile evidence for a finished run. No network call.

    Takes the SAME `results` list `ingest.social.summarise` already takes --
    the fully finished, already-verified-or-rejected candidate list -- and is
    called from the same place in the runner, after verification, never
    before it.
    """
    relationships: list[ProfileRelationship] = []
    considered = 0

    for result in results:
        if result.get("status") != VerificationStatus.VERIFIED_CANDIDATE.value:
            continue
        considered += 1

        hints = extract_hints(result)
        if not hints:
            continue

        relationship = _classify(result, hints)
        if relationship is not None:
            relationships.append(relationship)

    if relationships:
        by_tier: dict[str, int] = {}
        for rel in relationships:
            by_tier[rel.tier.display] = by_tier.get(rel.tier.display, 0) + 1
        counts = ", ".join(
            "{0} {1}".format(n, tier) for tier, n in by_tier.items()
        )
        statement = (
            "{0} public profile relationship(s) found across {1} verified "
            "candidate(s): {2}.".format(len(relationships), considered, counts)
        )
    else:
        statement = "No verified public profile relationship found."

    return ProfileEvidenceSummary(
        relationships=tuple(relationships),
        candidates_considered=considered,
        statement=statement,
    )
