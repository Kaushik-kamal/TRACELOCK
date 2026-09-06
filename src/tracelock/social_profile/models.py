"""Data shapes for public profile evidence -- Tier 1.

TIER 1, PRECISELY
-----------------
This module and its siblings (`hints.py`, `relate.py`) answer one narrow
question: of the candidates TRACELOCK already downloaded and face-verified,
does the metadata the DISCOVERY PROVIDER already returned (a page title, an
author field, a URL's own path structure) suggest a linkable public profile?

Nothing here makes a network call. Nothing here re-ranks, re-scores, or
reclassifies a candidate. A REJECTED or INCONCLUSIVE candidate cannot produce
a profile relationship at all -- `relate.py` filters to VERIFIED_CANDIDATE
before this package's logic ever runs, so "the face didn't match" and "no
public profile relationship found" can never be confused with each other.

THREE TIERS, NEVER COLLAPSED
-----------------------------
The same discipline `verification.relation.EvidenceRelation` already applies
to the photo-identity question applies here to the profile-identity question:
report the raw signal and its provenance, and let the reader see exactly why
a tier was assigned rather than trusting a single number.

    DISCOVERED_LINK
        A plausible profile URL or handle surfaced. Nothing corroborates it
        beyond the one signal that produced it.

    UNVERIFIED_POSSIBLE_MATCH
        Two INDEPENDENT extraction methods (say, a URL-derived handle and a
        title-text handle) agree on the same platform and handle. Agreement
        between independent signals is real corroboration -- the same
        principle `evidence.aggregate` uses for independent publishers -- but
        it is still short of certainty.

    VERIFIED_HIGH_CONFIDENCE
        The face-verified photo lives ON the profile page itself: the
        candidate's OWN url is profile-shaped (not a post/status permalink),
        and that candidate already cleared calibrated face verification. This
        is the strongest claim Tier 1 can support, and it is still a claim
        about a PAGE, not a claim about a person's real-world identity --
        the evidence chain says exactly that.

A profile-shaped URL alone is never sufficient for the top tier: it must
belong to a candidate whose `status` is VERIFIED_CANDIDATE. `relate.py`
enforces this as a structural filter, not a threshold that could be tuned
away by accident.

WHAT THIS IS NOT
----------------
Not a claim that a real person "owns" an account. Not identity resolution.
Not a database of who someone is. It is a description of what the discovery
provider's own metadata says about pages TRACELOCK already proved contain the
subject's face -- with the exact reasoning attached, so a reader can disagree
with it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ProfileTier(str, Enum):
    DISCOVERED_LINK = "DISCOVERED_LINK"
    UNVERIFIED_POSSIBLE_MATCH = "UNVERIFIED_POSSIBLE_MATCH"
    VERIFIED_HIGH_CONFIDENCE = "VERIFIED_HIGH_CONFIDENCE"

    @property
    def display(self) -> str:
        return {
            ProfileTier.DISCOVERED_LINK: "Discovered link only",
            ProfileTier.UNVERIFIED_POSSIBLE_MATCH: "Unverified possible match",
            ProfileTier.VERIFIED_HIGH_CONFIDENCE: "Verified high-confidence",
        }[self]

    @property
    def claim(self) -> str:
        """The precise, non-overreaching sentence this tier is allowed to make."""
        return {
            ProfileTier.DISCOVERED_LINK: (
                "A public URL was found. Nothing beyond that URL's own "
                "existence has been checked."
            ),
            ProfileTier.UNVERIFIED_POSSIBLE_MATCH: (
                "Two independent pieces of metadata agree on this platform "
                "and handle, but the face-verified photo was not found on "
                "this exact page."
            ),
            ProfileTier.VERIFIED_HIGH_CONFIDENCE: (
                "The face-verified photo was found on this exact page, and "
                "the page's own URL is profile-shaped for this platform."
            ),
        }[self]


class ExtractionMethod(str, Enum):
    """HOW a hint was pulled from metadata TRACELOCK already had. No method
    here performs a network call; each is a pure transform of a string
    already sitting on the verified candidate."""

    URL_IS_PROFILE_SHAPED = "url_is_profile_shaped"
    HANDLE_IN_POST_URL = "handle_embedded_in_post_url"
    TITLE_TEXT_PATTERN = "title_text_pattern"
    AUTHOR_FIELD = "author_field"

    @property
    def explanation(self) -> str:
        return {
            ExtractionMethod.URL_IS_PROFILE_SHAPED: (
                "the verified candidate's own URL matches this platform's "
                "profile-page path shape, not a post/status shape"
            ),
            ExtractionMethod.HANDLE_IN_POST_URL: (
                "this platform embeds the author's handle directly in its "
                "post URLs, and the handle was read from that same URL"
            ),
            ExtractionMethod.TITLE_TEXT_PATTERN: (
                "the discovery provider's page-title text matched a common "
                "platform naming convention (e.g. '<name> (@handle)')"
            ),
            ExtractionMethod.AUTHOR_FIELD: (
                "the discovery provider populated an author/channel/uploader "
                "field directly for this result"
            ),
        }[self]


@dataclass(frozen=True, slots=True)
class EvidenceChainStep:
    """One link in the chain, always attributable to something concrete.

    `claim` is what is being asserted. `source` says exactly where that came
    from -- a field name, a candidate id, a policy rule -- never "the
    system decided". `quoted_text`, when present, is the literal string the
    claim was read from, so a reader can check the extraction themselves
    instead of trusting a summary of it.
    """

    claim: str
    source: str
    quoted_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"claim": self.claim, "source": self.source}
        if self.quoted_text is not None:
            payload["quoted_text"] = self.quoted_text
        return payload


@dataclass(frozen=True, slots=True)
class ProfileHint:
    """One extracted signal pointing at a platform + handle/URL.

    A hint is not a relationship. `relate.py` combines one or more hints
    (plus the source candidate's verification status) into a
    `ProfileRelationship` with a tier attached. A hint on its own never
    reaches the UI.
    """

    candidate_id: str
    platform: str
    method: ExtractionMethod
    handle: str | None
    profile_url: str | None
    quoted_text: str
    source_field: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "platform": self.platform,
            "method": self.method.value,
            "handle": self.handle,
            "profile_url": self.profile_url,
            "quoted_text": self.quoted_text,
            "source_field": self.source_field,
        }


@dataclass(frozen=True, slots=True)
class ProfileRelationship:
    """One (platform, handle-or-url) conclusion, with its full reasoning attached.

    `evidence_chain` is never empty -- constructing one with no steps is a
    programming error in `relate.py`, not a valid state, and a test asserts
    exactly that for every VERIFIED_HIGH_CONFIDENCE instance.
    """

    tier: ProfileTier
    platform: str
    handle: str | None
    profile_url: str | None
    source_candidate_id: str
    source_url: str
    face_similarity: float | None
    evidence_chain: tuple[EvidenceChainStep, ...]

    def __post_init__(self) -> None:
        if not self.evidence_chain:
            raise ValueError(
                "ProfileRelationship must carry at least one evidence-chain "
                "step -- a tier with no stated reasoning is exactly the "
                "unsupported claim this feature exists to prevent."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier.value,
            "tier_display": self.tier.display,
            "tier_claim": self.tier.claim,
            "platform": self.platform,
            "handle": self.handle,
            "profile_url": self.profile_url,
            "source_candidate_id": self.source_candidate_id,
            "source_url": self.source_url,
            "face_similarity": (
                round(self.face_similarity, 6)
                if self.face_similarity is not None
                else None
            ),
            "evidence_chain": [step.to_dict() for step in self.evidence_chain],
        }


@dataclass(frozen=True, slots=True)
class ProfileEvidenceSummary:
    """The whole-run rollup, mirroring `ingest.social.summarise`'s shape."""

    relationships: tuple[ProfileRelationship, ...]
    candidates_considered: int
    statement: str

    def to_dict(self) -> dict[str, Any]:
        by_tier: dict[str, int] = {}
        for rel in self.relationships:
            by_tier[rel.tier.value] = by_tier.get(rel.tier.value, 0) + 1
        return {
            "relationships": [r.to_dict() for r in self.relationships],
            "by_tier": by_tier,
            "candidates_considered": self.candidates_considered,
            "statement": self.statement,
        }
