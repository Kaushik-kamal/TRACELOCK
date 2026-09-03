"""Loading calibration datasets and turning them into similarity observations.

Two loaders:

  ManifestLoader   reads a JSON manifest of real, consented image pairs.
                   Produces DatasetKind.REAL. Refuses a manifest with no
                   consent records.

  FixtureLoader    generates synthetic embedding pairs for testing the metric
                   code. Produces DatasetKind.FIXTURE, which propagates into
                   every downstream report and blocks it from being cited.

There is deliberately no loader that scrapes faces from the web. Calibration
data must be consented, and a loader that made it easy to skip that step would
undermine the safeguard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from tracelock.calibration.contract import (
    CalibrationDataset,
    CalibrationPair,
    ConsentRecord,
    DatasetKind,
    DatasetProvenance,
    ObservationSet,
    PairLabel,
    SimilarityObservation,
)
from tracelock.face.similarity import cosine_similarity


class CalibrationLoader(Protocol):
    """Anything that can produce a CalibrationDataset."""

    def load(self) -> CalibrationDataset: ...


class ManifestLoader:
    """Load real, consented pairs from a JSON manifest.

    Expected shape:

        {
          "name": "tracelock-consented-v1",
          "description": "...",
          "consent": [
            {"subject_ref": "subject-01", "consent_obtained": true,
             "consent_date": "2026-09-01", "source": "self"}
          ],
          "pairs": [
            {"pair_id": "p001", "image_a": "a.jpg", "image_b": "b.jpg",
             "label": "GENUINE", "subject_a": "subject-01",
             "subject_b": "subject-01"}
          ]
        }
    """

    def __init__(self, manifest_path: str | Path, *, require_consent: bool = True):
        self.manifest_path = Path(manifest_path)
        self.require_consent = require_consent

    def load(self) -> CalibrationDataset:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                "calibration manifest not found: {0}".format(self.manifest_path)
            )

        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))

        consent = tuple(
            ConsentRecord(
                subject_ref=entry["subject_ref"],
                consent_obtained=bool(entry.get("consent_obtained", False)),
                consent_date=entry.get("consent_date"),
                source=entry.get("source", ""),
                notes=entry.get("notes", ""),
            )
            for entry in payload.get("consent", [])
        )

        if self.require_consent and not any(c.consent_obtained for c in consent):
            raise ValueError(
                "manifest {0} declares no obtained consent. A face calibration "
                "set without consent is not usable here. Pass "
                "require_consent=False only for a public research dataset whose "
                "licence you have verified.".format(self.manifest_path)
            )

        base = self.manifest_path.parent
        pairs = tuple(
            CalibrationPair(
                pair_id=entry.get("pair_id", ""),
                image_a=str(base / entry["image_a"]),
                image_b=str(base / entry["image_b"]),
                label=PairLabel(entry["label"]),
                subject_a=entry.get("subject_a", ""),
                subject_b=entry.get("subject_b", ""),
            )
            for entry in payload.get("pairs", [])
        )

        return CalibrationDataset(
            provenance=DatasetProvenance(
                kind=DatasetKind.REAL,
                name=payload.get("name", self.manifest_path.stem),
                description=payload.get("description", ""),
                consent=consent,
            ),
            pairs=pairs,
        )


class FixtureLoader:
    """Generate synthetic embedding pairs. FOR TESTING THE MATHS ONLY.

    Produces unit vectors with a controllable angular separation between the
    genuine and impostor populations. This verifies that ROC/EER/FAR code is
    correct. It says nothing whatsoever about face recognition performance,
    and DatasetKind.FIXTURE makes that structural rather than advisory.
    """

    def __init__(
        self,
        *,
        n_genuine: int = 30,
        n_impostor: int = 200,
        dimension: int = 512,
        genuine_similarity: float = 0.70,
        impostor_similarity: float = 0.05,
        noise: float = 0.08,
        seed: int = 20260901,
    ):
        self.n_genuine = n_genuine
        self.n_impostor = n_impostor
        self.dimension = dimension
        self.genuine_similarity = genuine_similarity
        self.impostor_similarity = impostor_similarity
        self.noise = noise
        self.seed = seed

    def load(self) -> CalibrationDataset:
        pairs = tuple(
            CalibrationPair(
                pair_id="fixture-genuine-{0:04d}".format(i),
                image_a="<synthetic>",
                image_b="<synthetic>",
                label=PairLabel.GENUINE,
                subject_a="synthetic-{0}".format(i),
                subject_b="synthetic-{0}".format(i),
            )
            for i in range(self.n_genuine)
        ) + tuple(
            CalibrationPair(
                pair_id="fixture-impostor-{0:04d}".format(i),
                image_a="<synthetic>",
                image_b="<synthetic>",
                label=PairLabel.IMPOSTOR,
                subject_a="synthetic-a-{0}".format(i),
                subject_b="synthetic-b-{0}".format(i),
            )
            for i in range(self.n_impostor)
        )

        return CalibrationDataset(
            provenance=DatasetProvenance(
                kind=DatasetKind.FIXTURE,
                name="synthetic-fixture",
                description=(
                    "Generated unit vectors. Verifies metric correctness only. "
                    "Not evidence of face recognition performance."
                ),
            ),
            pairs=pairs,
        )

    def observe(self) -> ObservationSet:
        """Generate the dataset AND its similarity scores in one step."""
        dataset = self.load()
        rng = np.random.default_rng(self.seed)

        observations = []
        for pair in dataset:
            target = (
                self.genuine_similarity
                if pair.label is PairLabel.GENUINE
                else self.impostor_similarity
            )
            a = _unit(rng.normal(size=self.dimension))
            b = _unit(_rotate_toward(a, target, rng, self.noise))
            observations.append(
                SimilarityObservation(
                    pair=pair,
                    similarity=cosine_similarity(a, b),
                    model_id="synthetic-fixture",
                )
            )

        return ObservationSet(
            provenance=dataset.provenance,
            observations=tuple(observations),
            model_id="synthetic-fixture",
        )


def observe_dataset(
    dataset: CalibrationDataset, engine, *, skip_failures: bool = True
) -> ObservationSet:
    """Run a real FaceEngine over a dataset to produce similarity observations.

    Provenance is carried through from the dataset unchanged, so a REAL
    dataset yields a citable report and a FIXTURE never can.
    """
    from tracelock.face.errors import FaceEngineError

    cache: dict[str, object] = {}
    observations: list[SimilarityObservation] = []

    def embed(path: str):
        if path not in cache:
            cache[path] = engine.analyze(path).embedding
        return cache[path]

    for pair in dataset:
        try:
            similarity = cosine_similarity(embed(pair.image_a), embed(pair.image_b))
        except (FaceEngineError, FileNotFoundError):
            if skip_failures:
                continue
            raise
        observations.append(
            SimilarityObservation(
                pair=pair, similarity=similarity, model_id=engine.model_id
            )
        )

    return ObservationSet(
        provenance=dataset.provenance,
        observations=tuple(observations),
        model_id=engine.model_id,
    )


# --------------------------------------------------------------------------


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def _rotate_toward(
    anchor: np.ndarray, target_cos: float, rng: np.random.Generator, noise: float
) -> np.ndarray:
    """Build a vector at approximately `target_cos` from `anchor`."""
    target = float(np.clip(target_cos + rng.normal(0.0, noise), -0.99, 0.99))

    orthogonal = rng.normal(size=anchor.shape[0])
    orthogonal -= np.dot(orthogonal, anchor) * anchor
    orthogonal = _unit(orthogonal)

    return target * anchor + np.sqrt(max(0.0, 1.0 - target**2)) * orthogonal
