"""Build a labelled calibration set from completed verification runs.

WHY THIS SOURCE
---------------
Every Phase 2 run already measured a real cosine similarity between a probe and
independently downloaded candidate bytes, using the production face engine. The
measurements exist; what is missing is LABELS.

Reusing them costs no API credits, requires no new downloads, and calibrates on
exactly the distribution the system encounters in production -- which is a
better fit than a curated benchmark would be.

THE LABELLING PROBLEM, AND HOW IT IS HANDLED HONESTLY
-----------------------------------------------------
Labels cannot be invented. This module PROPOSES labels together with the
evidence for each proposal, and records that evidence in the dataset. It never
silently asserts ground truth.

Two proposal rules, each auditable:

  GENUINE   the probe is a photograph of a public figure, and the candidate was
            published on the subject's own official domain or by an established
            news outlet that captioned it as depicting them. This is source
            attribution -- the same method used to label LFW, and it carries
            the same weakness: a mis-captioned photograph becomes a mislabel.

  IMPOSTOR  the candidate is a profile photograph published under a DIFFERENT
            named party (another person's social profile, another account's
            avatar, stock photography). Different named party, different
            identity.

Every pair carries its `labeling_basis` string into the dataset, and the
operator reviews the generated manifest before it is used. Proposals are not
conclusions.

CONSENT
-------
These images were collected by the pipeline from public pages; no subject
consented to being a calibration pair. `DatasetProvenance` therefore carries no
consent record, which makes the resulting report `citable=False` and prints a
non-citable notice. That is the correct outcome, not a workaround: the numbers
are usable for tuning our own thresholds and are not publishable as a study.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from tracelock.calibration.contract import (
    CalibrationPair,
    DatasetKind,
    DatasetProvenance,
    ObservationSet,
    PairLabel,
    SimilarityObservation,
)

# Domains that publish photographs OF a named subject rather than BY arbitrary
# users. Presence here is evidence for a GENUINE proposal, never proof.
# Deliberately a data file, not a heuristic: a reviewer can disagree with a row.
DEFAULT_ATTRIBUTION_HINTS: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LabelProposal:
    """A proposed label plus the evidence for it. Reviewable, not authoritative."""

    content_sha256: str
    probe_sha256: str
    similarity: float
    domain: str
    source_url: str
    label: PairLabel
    basis: str
    confidence: str  # "attribution" | "distinct-party" | "operator"

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_sha256": self.content_sha256,
            "probe_sha256": self.probe_sha256,
            "similarity": round(self.similarity, 6),
            "domain": self.domain,
            "source_url": self.source_url,
            "label": self.label.value,
            "basis": self.basis,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class ProbeLabelRule:
    """How to label candidates measured against one probe.

    An explicit operator declaration. Identity is never inferred automatically.

    `domains` optionally restricts the rule to specific registrable domains.
    This is what keeps the labelling NON-CIRCULAR: a GENUINE label must rest on
    evidence independent of the similarity being calibrated. Source attribution
    -- "the subject's official domain published this" -- is independent.
    Similarity is not, and labelling by it would calibrate the measurement
    against itself and produce a meaningless model.
    """

    probe_sha256_prefix: str
    label: PairLabel
    basis: str
    confidence: str
    subject_ref: str = ""
    domains: tuple[str, ...] = ()

    def matches(self, probe_sha256: str, domain: str = "") -> bool:
        if not probe_sha256.startswith(self.probe_sha256_prefix):
            return False
        if self.domains and domain not in self.domains:
            return False
        return True


def load_measurements(run_paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Extract every completed similarity measurement from verification runs.

    Only candidates that reached the COMPARED stage are usable: a rejection at
    acquisition or validation produced no similarity to calibrate on.
    """
    measurements: list[dict[str, Any]] = []

    for path in run_paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        probe_sha = (payload.get("probe") or {}).get("sha256", "")

        for result in payload.get("results", []):
            similarity = result.get("face_similarity")
            content_sha = result.get("content_sha256")
            if similarity is None or not content_sha:
                continue

            measurements.append(
                {
                    "probe_sha256": probe_sha,
                    "content_sha256": content_sha,
                    "similarity": float(similarity),
                    "domain": (result.get("provenance") or {}).get(
                        "registrable_domain", ""
                    ),
                    "source_url": result.get("source_url") or result.get("media_url") or "",
                    "run": str(path),
                }
            )

    return measurements


def deduplicate(measurements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse repeat measurements of the same (probe, content) pair.

    Seven identical Phase 2 runs contributed the same blobs seven times. Left
    uncollapsed they would inflate n sevenfold and make the confidence interval
    a lie -- the single most damaging error available in a calibration.
    """
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for row in measurements:
        key = (row["probe_sha256"], row["content_sha256"])
        seen.setdefault(key, row)
    return list(seen.values())


def propose_labels(
    measurements: list[dict[str, Any]], rules: list[ProbeLabelRule]
) -> tuple[list[LabelProposal], list[dict[str, Any]]]:
    """Apply operator-declared rules. Returns (proposals, unlabelled)."""
    proposals: list[LabelProposal] = []
    unlabelled: list[dict[str, Any]] = []

    for row in measurements:
        rule = next(
            (r for r in rules if r.matches(row["probe_sha256"], row.get("domain", ""))),
            None,
        )
        if rule is None:
            unlabelled.append(row)
            continue

        proposals.append(
            LabelProposal(
                content_sha256=row["content_sha256"],
                probe_sha256=row["probe_sha256"],
                similarity=row["similarity"],
                domain=row["domain"],
                source_url=row["source_url"],
                label=rule.label,
                basis=rule.basis,
                confidence=rule.confidence,
            )
        )

    return proposals, unlabelled


def build_observation_set(
    proposals: list[LabelProposal],
    *,
    name: str,
    description: str,
    model_id: str,
) -> ObservationSet:
    """Turn reviewed proposals into an ObservationSet ready for evaluation.

    Provenance is DatasetKind.REAL with NO consent records, so every derived
    report is correctly marked non-citable. The fixture stamp is untouched:
    this is real measured data, not synthetic.
    """
    provenance = DatasetProvenance(
        kind=DatasetKind.REAL,
        name=name,
        description=description,
        consent=(),  # none obtained -> report renders as non-citable
    )

    observations = tuple(
        SimilarityObservation(
            pair=CalibrationPair(
                pair_id="{0}:{1}".format(
                    proposal.probe_sha256[:8], proposal.content_sha256[:8]
                ),
                image_a="cas://{0}".format(proposal.probe_sha256),
                image_b="cas://{0}".format(proposal.content_sha256),
                label=proposal.label,
                # Subject refs are pseudonymous and consistent within a label,
                # satisfying the CalibrationPair consistency invariant without
                # recording anyone's name.
                subject_a="probe-{0}".format(proposal.probe_sha256[:8]),
                subject_b=(
                    "probe-{0}".format(proposal.probe_sha256[:8])
                    if proposal.label is PairLabel.GENUINE
                    else "other-{0}".format(proposal.content_sha256[:8])
                ),
            ),
            similarity=proposal.similarity,
            model_id=model_id,
        )
        for proposal in proposals
    )

    return ObservationSet(
        provenance=provenance, observations=observations, model_id=model_id
    )


def write_manifest(
    proposals: list[LabelProposal],
    unlabelled: list[dict[str, Any]],
    path: str | Path,
    *,
    rules: list[ProbeLabelRule],
) -> Path:
    """Write the reviewable draft manifest.

    The operator reads this BEFORE the labels are used. It records what was
    proposed, on what basis, and what could not be labelled at all.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": "calibration-manifest/1",
        "review_required": True,
        "notice": (
            "PROPOSED labels, not confirmed ground truth. Each row records the "
            "basis for its proposal. Review before fitting; edit or delete any "
            "row you disagree with."
        ),
        "rules": [
            {
                "probe_sha256_prefix": rule.probe_sha256_prefix,
                "label": rule.label.value,
                "basis": rule.basis,
                "confidence": rule.confidence,
                "subject_ref": rule.subject_ref,
                "domains": list(rule.domains),
            }
            for rule in rules
        ],
        "counts": {
            "proposed": len(proposals),
            "genuine": sum(1 for p in proposals if p.label is PairLabel.GENUINE),
            "impostor": sum(1 for p in proposals if p.label is PairLabel.IMPOSTOR),
            "unlabelled": len(unlabelled),
        },
        "proposals": [p.to_dict() for p in proposals],
        "unlabelled": unlabelled,
    }

    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return target


def load_manifest(path: str | Path) -> list[LabelProposal]:
    """Read a reviewed manifest back into proposals."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        LabelProposal(
            content_sha256=row["content_sha256"],
            probe_sha256=row["probe_sha256"],
            similarity=float(row["similarity"]),
            domain=row.get("domain", ""),
            source_url=row.get("source_url", ""),
            label=PairLabel(row["label"]),
            basis=row.get("basis", ""),
            confidence=row.get("confidence", "operator"),
        )
        for row in payload.get("proposals", [])
    ]
