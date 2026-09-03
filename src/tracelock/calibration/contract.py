"""Calibration data contract.

THE PROBLEM THIS SOLVES
-----------------------
A similarity score is not an identity probability. Turning one into the other
requires fitting a curve to labelled pairs of KNOWN identity. Without that data
the mapping is invented.

The temptation in a competition is to generate plausible-looking pairs, fit a
ROC to them, and present the result as evidence. That would be fabricated
empirical evidence -- the most serious integrity failure available to us, and
far worse than shipping with no calibration at all.

THE MECHANISM
-------------
Every dataset declares its `kind`:

    DatasetKind.REAL     consented images of known identity
    DatasetKind.FIXTURE  synthetic vectors for exercising the maths

Anything derived from a FIXTURE dataset is stamped `is_fixture=True` all the
way through to the rendered report, which prints a refusal banner instead of
results. `tests/test_calibration.py` asserts a fixture-derived report can never
present itself as empirical. The stamp is not a convention -- it propagates
structurally and cannot be dropped by forgetting to set a flag.

Phase 1 ships the CONTRACT, the LOADERS and the METRICS. It ships no real
calibration data, and claims none.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterator


class PairLabel(str, Enum):
    """Ground-truth relationship between two images."""

    GENUINE = "GENUINE"    # same identity
    IMPOSTOR = "IMPOSTOR"  # different identities


class DatasetKind(str, Enum):
    """Provenance class. Governs whether results may be cited as evidence."""

    REAL = "REAL"
    FIXTURE = "FIXTURE"


@dataclass(frozen=True, slots=True)
class ConsentRecord:
    """Who agreed to what.

    Required on REAL datasets. A face dataset without a consent record is not
    a dataset we are willing to use, regardless of where it came from.
    """

    subject_ref: str          # pseudonymous handle, never a legal name
    consent_obtained: bool
    consent_date: str | None = None
    source: str = ""          # "self", "written consent", "public dataset: LFW"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_ref": self.subject_ref,
            "consent_obtained": self.consent_obtained,
            "consent_date": self.consent_date,
            "source": self.source,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class CalibrationPair:
    """One labelled pair of images."""

    image_a: str
    image_b: str
    label: PairLabel
    pair_id: str = ""
    subject_a: str = ""
    subject_b: str = ""

    def __post_init__(self) -> None:
        if self.label is PairLabel.GENUINE and self.subject_a and self.subject_b:
            if self.subject_a != self.subject_b:
                raise ValueError(
                    "pair {0!r} is labelled GENUINE but names two different "
                    "subjects ({1!r} vs {2!r})".format(
                        self.pair_id, self.subject_a, self.subject_b
                    )
                )
        if self.label is PairLabel.IMPOSTOR and self.subject_a and self.subject_b:
            if self.subject_a == self.subject_b:
                raise ValueError(
                    "pair {0!r} is labelled IMPOSTOR but names the same "
                    "subject {1!r}".format(self.pair_id, self.subject_a)
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "image_a": self.image_a,
            "image_b": self.image_b,
            "label": self.label.value,
            "subject_a": self.subject_a,
            "subject_b": self.subject_b,
        }


@dataclass(frozen=True, slots=True)
class DatasetProvenance:
    """Where a dataset came from and whether it may be cited."""

    kind: DatasetKind
    name: str
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    consent: tuple[ConsentRecord, ...] = ()

    @property
    def is_fixture(self) -> bool:
        return self.kind is DatasetKind.FIXTURE

    @property
    def citable(self) -> bool:
        """True only for REAL data with at least one consent record."""
        return self.kind is DatasetKind.REAL and bool(self.consent)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "is_fixture": self.is_fixture,
            "citable": self.citable,
            "consent": [c.to_dict() for c in self.consent],
        }


@dataclass(frozen=True, slots=True)
class CalibrationDataset:
    """A labelled pair set plus its provenance."""

    provenance: DatasetProvenance
    pairs: tuple[CalibrationPair, ...]

    def __iter__(self) -> Iterator[CalibrationPair]:
        return iter(self.pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    @property
    def is_fixture(self) -> bool:
        return self.provenance.is_fixture

    @property
    def genuine_pairs(self) -> tuple[CalibrationPair, ...]:
        return tuple(p for p in self.pairs if p.label is PairLabel.GENUINE)

    @property
    def impostor_pairs(self) -> tuple[CalibrationPair, ...]:
        return tuple(p for p in self.pairs if p.label is PairLabel.IMPOSTOR)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.provenance.name,
            "kind": self.provenance.kind.value,
            "is_fixture": self.is_fixture,
            "citable": self.provenance.citable,
            "total_pairs": len(self.pairs),
            "genuine_pairs": len(self.genuine_pairs),
            "impostor_pairs": len(self.impostor_pairs),
        }


@dataclass(frozen=True, slots=True)
class SimilarityObservation:
    """A pair with its measured similarity. The input to every metric."""

    pair: CalibrationPair
    similarity: float
    model_id: str

    @property
    def label(self) -> PairLabel:
        return self.pair.label

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair": self.pair.to_dict(),
            "similarity": round(self.similarity, 6),
            "model_id": self.model_id,
        }


@dataclass(frozen=True, slots=True)
class ObservationSet:
    """Similarity observations plus the provenance they inherit.

    `is_fixture` propagates from the dataset. This is the link that makes the
    anti-fabrication guarantee structural rather than a matter of remembering
    to set a flag.
    """

    provenance: DatasetProvenance
    observations: tuple[SimilarityObservation, ...]
    model_id: str

    @property
    def is_fixture(self) -> bool:
        return self.provenance.is_fixture

    @property
    def genuine_scores(self) -> tuple[float, ...]:
        return tuple(
            o.similarity for o in self.observations if o.label is PairLabel.GENUINE
        )

    @property
    def impostor_scores(self) -> tuple[float, ...]:
        return tuple(
            o.similarity for o in self.observations if o.label is PairLabel.IMPOSTOR
        )

    def __len__(self) -> int:
        return len(self.observations)
