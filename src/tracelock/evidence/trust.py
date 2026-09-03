"""Phase 3 -- the explainable trust score.

    T = 100 * P_id * (0.7 + 0.3*Q) * (1 + 0.15*C)

MULTIPLICATIVE, NOT ADDITIVE -- and that is the whole design
------------------------------------------------------------
A weighted sum lets a candidate with a terrible face match but excellent
sourcing score respectably. That is exactly backwards: if it is not the same
person, nothing else can rescue it. Identity must GATE, not merely contribute,
so it multiplies. P_id = 0 makes T = 0 regardless of everything else.

THE THREE TERMS

  P_id   Calibrated P(same identity), from tracelock.calibration. NOT a raw
         cosine. If no calibration model is loaded this is None and the scorer
         REFUSES to produce a score rather than substituting a similarity
         wearing a percentage sign.

         Aggregated as the MAXIMUM over unique verified images, not the mean
         and not a Bayesian combination. Reasons:
           - a mean punishes finding extra weaker evidence, which is perverse
           - naive-Bayes accumulation assumes independent observations, and
             these share a probe and a model, so their errors are correlated;
             multiplying them would manufacture certainty
         Multiple sources are rewarded through C instead, where the
         independence assumption is actually defensible.

  Q      Evidence quality in [0,1] -- see `EvidenceQuality`. Enters as
         (0.7 + 0.3*Q), a DAMPENER with a floor of 0.7: poor sourcing discounts
         a match by at most 30% and can never eliminate it, because a real face
         on a badly-documented page is still a real face.

  C      Corroboration in [0,1), saturating in the number of INDEPENDENT
         publishers. Enters as (1 + 0.15*C), a bonus capped at +15%: independent
         confirmation should help meaningfully but must not manufacture trust
         from nothing, and independence is rarely as clean as it looks.

Every constant above is defensible in one sentence, and `explain()` prints the
arithmetic so a reader can recompute the number by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from tracelock.evidence.aggregate import AggregatedEvidence, EvidenceItem

# Quality dampener: floor 0.70, so Q can discount by at most 30%.
QUALITY_FLOOR = 0.70
QUALITY_RANGE = 0.30

# Corroboration bonus cap: at most +15%.
CORROBORATION_WEIGHT = 0.15

# Weights within Q. Sum to 1.0.
QUALITY_WEIGHTS = {
    "face_quality": 0.50,       # measured by Phase 1, the best-grounded term
    "metadata_completeness": 0.30,
    "acquisition_integrity": 0.20,
}


class TrustBand(str, Enum):
    """Coarse verdicts over T. Cutoffs are working defaults, stated as such."""

    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    INSUFFICIENT = "INSUFFICIENT"

    @classmethod
    def from_score(cls, score: float) -> "TrustBand":
        if score >= 70.0:
            return cls.STRONG
        if score >= 45.0:
            return cls.MODERATE
        if score >= 20.0:
            return cls.WEAK
        return cls.INSUFFICIENT


@dataclass(frozen=True, slots=True)
class QualityComponent:
    name: str
    value: float
    weight: float
    available: bool
    note: str = ""

    @property
    def contribution(self) -> float:
        return self.value * self.weight

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": round(self.value, 4),
            "weight": self.weight,
            "contribution": round(self.contribution, 4),
            "available": self.available,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class EvidenceQuality:
    components: tuple[QualityComponent, ...]
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "components": [c.to_dict() for c in self.components],
        }


@dataclass(frozen=True, slots=True)
class TrustScore:
    """A fully explainable trust score."""

    score: float
    band: TrustBand

    identity_probability: float
    quality: EvidenceQuality
    corroboration: float
    independent_publishers: int

    quality_multiplier: float
    corroboration_multiplier: float

    calibrated: bool
    calibration_note: str
    strongest_item_sha256: str | None
    limitations: tuple[str, ...]

    def explain(self) -> list[str]:
        """The arithmetic, line by line, so it can be checked by hand."""
        return [
            "T = 100 x P_id x (0.7 + 0.3*Q) x (1 + 0.15*C)",
            "",
            "  P_id  = {0:.4f}   calibrated P(same identity), max over unique images".format(
                self.identity_probability
            ),
            "  Q     = {0:.4f}   evidence quality".format(self.quality.score),
            "  C     = {0:.4f}   corroboration across {1} independent publisher(s)".format(
                self.corroboration, self.independent_publishers
            ),
            "",
            "  quality multiplier       = 0.7 + 0.3 x {0:.4f} = {1:.4f}".format(
                self.quality.score, self.quality_multiplier
            ),
            "  corroboration multiplier = 1 + 0.15 x {0:.4f} = {1:.4f}".format(
                self.corroboration, self.corroboration_multiplier
            ),
            "",
            "  T = 100 x {0:.4f} x {1:.4f} x {2:.4f} = {3:.2f}".format(
                self.identity_probability,
                self.quality_multiplier,
                self.corroboration_multiplier,
                self.score,
            ),
            "  band = {0}".format(self.band.value),
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 2),
            "band": self.band.value,
            "formula": "T = 100 * P_id * (0.7 + 0.3*Q) * (1 + 0.15*C)",
            "terms": {
                "P_id": round(self.identity_probability, 6),
                "Q": round(self.quality.score, 6),
                "C": round(self.corroboration, 6),
            },
            "multipliers": {
                "quality": round(self.quality_multiplier, 6),
                "corroboration": round(self.corroboration_multiplier, 6),
            },
            "independent_publishers": self.independent_publishers,
            "quality_breakdown": self.quality.to_dict(),
            "calibrated": self.calibrated,
            "calibration_note": self.calibration_note,
            "strongest_item_sha256": self.strongest_item_sha256,
            "limitations": list(self.limitations),
            "explanation": self.explain(),
        }


class UncalibratedScoreRefused(Exception):
    """Raised when asked to score without a calibration model.

    Deliberately an exception rather than a degraded number. Without
    calibration there is no P_id, and a "trust score" built on a raw cosine
    would be a fabricated statistic wearing a percentage sign.
    """


class NoVerifiedEvidence(UncalibratedScoreRefused):
    """Every candidate was examined and none met the same-person threshold.

    A SUBCLASS so that existing `except UncalibratedScoreRefused` handlers keep
    working unchanged -- but a distinct type, because the two situations are
    not the same thing and were previously indistinguishable:

        UncalibratedScoreRefused  we cannot score (no calibration model)
        NoVerifiedEvidence        we scored nothing because nothing qualified

    The second is a COMPLETED forensic result. Reporting it as a pipeline
    failure -- which is what happened -- tells an operator the tool broke when
    in fact it did its job and declined to claim a match.
    """


def assess_quality(
    item: EvidenceItem, *, metadata_completeness: float, acquisition_integrity: float
) -> EvidenceQuality:
    """Weighted quality for the strongest evidence item.

    Built only from signals we actually measure. Source reputation is a term
    the architecture anticipated but no reputation table exists yet, so it is
    ABSENT rather than stubbed with a fabricated constant.
    """
    components = (
        QualityComponent(
            "face_quality",
            item.face_quality,
            QUALITY_WEIGHTS["face_quality"],
            available=True,
            note="Phase 1 aggregate: pixel size, sharpness, detection "
            "confidence, pose, exposure, frame containment",
        ),
        QualityComponent(
            "metadata_completeness",
            metadata_completeness,
            QUALITY_WEIGHTS["metadata_completeness"],
            available=True,
            note="presence of title, text, author and timestamp on the source "
            "record; measures completeness, NOT truthfulness",
        ),
        QualityComponent(
            "acquisition_integrity",
            acquisition_integrity,
            QUALITY_WEIGHTS["acquisition_integrity"],
            available=True,
            note="HTTP 200, magic bytes matched the declared Content-Type, no "
            "unexpected redirect",
        ),
    )
    score = sum(c.contribution for c in components)
    return EvidenceQuality(components=components, score=min(1.0, max(0.0, score)))


def score_evidence(
    evidence: AggregatedEvidence,
    *,
    metadata_completeness: float,
    acquisition_integrity: float,
    calibration_note: str = "",
    calibrated: bool = True,
) -> TrustScore:
    """Compute the trust score. Refuses rather than guessing."""
    if not evidence.has_evidence:
        raise NoVerifiedEvidence(
            "no verified evidence to score: every candidate was rejected or "
            "inconclusive"
        )

    strongest = evidence.strongest()
    if strongest is None or strongest.identity_probability is None:
        raise UncalibratedScoreRefused(
            "the strongest evidence item carries no calibrated identity "
            "probability. Fit a calibration model (scripts/calibrate.py) "
            "before scoring -- a trust score built on a raw cosine similarity "
            "would be a fabricated statistic."
        )

    quality = assess_quality(
        strongest,
        metadata_completeness=metadata_completeness,
        acquisition_integrity=acquisition_integrity,
    )
    corroboration = evidence.corroboration_factor()

    quality_multiplier = QUALITY_FLOOR + QUALITY_RANGE * quality.score
    corroboration_multiplier = 1.0 + CORROBORATION_WEIGHT * corroboration

    raw = 100.0 * strongest.identity_probability * quality_multiplier * corroboration_multiplier
    score = max(0.0, min(100.0, raw))

    limitations = _limitations(evidence, strongest, calibration_note)

    return TrustScore(
        score=score,
        band=TrustBand.from_score(score),
        identity_probability=strongest.identity_probability,
        quality=quality,
        corroboration=corroboration,
        independent_publishers=evidence.independent_publisher_count,
        quality_multiplier=quality_multiplier,
        corroboration_multiplier=corroboration_multiplier,
        calibrated=calibrated,
        calibration_note=calibration_note,
        strongest_item_sha256=strongest.content_sha256,
        limitations=limitations,
    )


def _limitations(
    evidence: AggregatedEvidence, strongest: EvidenceItem, calibration_note: str
) -> tuple[str, ...]:
    """State what this score does NOT establish. Always non-empty."""
    limits = [
        "This score reflects what the system OBSERVED at run time. It does not "
        "establish that any source's claims are true -- a blockchain anchor "
        "proves the observation was not altered, not that it was correct.",
        "Source reputation is not scored: no reputation table exists, so the "
        "quality term omits it rather than substituting a fabricated value.",
    ]

    if calibration_note:
        limits.append("Calibration: " + calibration_note)

    if evidence.independent_publisher_count <= 1:
        limits.append(
            "Only one independent publisher: corroboration contributes nothing. "
            "A single source is an assertion, not consensus."
        )

    non_independent = [i for i in evidence.items if not i.is_independent_corroboration]
    if non_independent:
        limits.append(
            "{0} unique image(s) were republications of the probe rather than "
            "independent photographs; they contribute provenance but no new "
            "identity evidence.".format(len(non_independent))
        )

    if evidence.funnel.verified < evidence.funnel.discovered / 2:
        limits.append(
            "Fewer than half of discovered candidates survived verification "
            "({0}/{1}); the discovery layer's precision is low.".format(
                evidence.funnel.verified, evidence.funnel.discovered
            )
        )

    return tuple(limits)


def measure_metadata_completeness(results: list[dict[str, Any]]) -> float:
    """Fraction of expected metadata fields present across verified candidates.

    Measures PRESENCE, not truthfulness. A post can carry a perfect timestamp
    and still be lying about it.
    """
    verified = [r for r in results if r["status"] == "VERIFIED_CANDIDATE"]
    if not verified:
        return 0.0

    total = 0.0
    for result in verified:
        present = 0
        if result.get("source_url"):
            present += 1
        if result.get("media_url"):
            present += 1
        if (result.get("provenance") or {}).get("registrable_domain"):
            present += 1
        validation = result.get("validation") or {}
        if validation.get("detected_format"):
            present += 1
        total += present / 4.0

    return total / len(verified)


def measure_acquisition_integrity(results: list[dict[str, Any]]) -> float:
    """Fraction of verified acquisitions that were clean.

    Clean means HTTP 200, no unexpected redirect, and the server's declared
    Content-Type agreed with the actual magic bytes.
    """
    verified = [r for r in results if r["status"] == "VERIFIED_CANDIDATE"]
    if not verified:
        return 0.0

    clean = 0
    for result in verified:
        acquisition = result.get("acquisition") or {}
        validation = result.get("validation") or {}
        if (
            acquisition.get("status_code") == 200
            and not acquisition.get("url_changed")
            and validation.get("content_type_was_honest")
        ):
            clean += 1

    return clean / len(verified)
