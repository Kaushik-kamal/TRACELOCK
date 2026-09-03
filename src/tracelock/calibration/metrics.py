"""Biometric evaluation metrics: ROC, FAR, FRR, EER, AUC.

Implemented in numpy rather than pulled from scikit-learn. Two reasons, both
substantive: it avoids a heavy dependency for ~60 lines of arithmetic, and in a
forensic system the metric definitions should be readable in the repository
rather than delegated to a library the reader has to go and check.

DEFINITIONS (biometric convention, which differs from generic ML)
-----------------------------------------------------------------
  FAR  False Accept Rate  = impostor pairs scoring >= threshold / all impostors
                            "how often we wrongly declare a match"
  FRR  False Reject Rate  = genuine pairs scoring < threshold / all genuines
                            "how often we miss a true match"
  EER  Equal Error Rate   = the rate where FAR == FRR

FAR is the number that matters for TRACELOCK. A false accept means asserting
that an innocent person appears in discovered content. A false reject only
means a missed candidate. The costs are not symmetric, so EER is reported for
comparability but `threshold_at_far` is the operating point to actually use.

STATISTICAL HONESTY
-------------------
Every report carries `n_genuine` / `n_impostor` and a Wilson confidence
interval on the error rates. With a few dozen pairs the interval is wide, and
saying so is the difference between a measurement and a decoration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from tracelock.calibration.contract import DatasetProvenance, ObservationSet

FIXTURE_BANNER = (
    "=" * 70 + "\n"
    "  SYNTHETIC FIXTURE -- NOT EMPIRICAL EVIDENCE\n"
    "  These numbers come from generated vectors, not from images of real\n"
    "  people. They verify that the metric code is correct. They say NOTHING\n"
    "  about how this system performs on faces. DO NOT CITE.\n"
    + "=" * 70
)


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    """One threshold and the error rates it produces."""

    threshold: float
    far: float
    frr: float

    @property
    def tar(self) -> float:
        """True Accept Rate = 1 - FRR."""
        return 1.0 - self.frr

    def to_dict(self) -> dict[str, float]:
        return {
            "threshold": round(self.threshold, 6),
            "far": round(self.far, 6),
            "frr": round(self.frr, 6),
            "tar": round(self.tar, 6),
        }


@dataclass(frozen=True, slots=True)
class RocCurve:
    thresholds: np.ndarray
    far: np.ndarray
    frr: np.ndarray

    @property
    def tar(self) -> np.ndarray:
        return 1.0 - self.frr

    def auc(self) -> float:
        """Area under the FAR-vs-TAR curve, via the trapezoid rule."""
        order = np.argsort(self.far)
        return float(np.trapezoid(self.tar[order], self.far[order]))


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Evaluation result. Inherits `is_fixture` from the observation set.

    `render()` refuses to print metrics for fixture data. That refusal is the
    whole point of this class.
    """

    provenance: DatasetProvenance
    model_id: str
    n_genuine: int
    n_impostor: int
    eer: float
    eer_threshold: float
    auc: float
    operating_points: tuple[OperatingPoint, ...]
    genuine_mean: float
    genuine_std: float
    impostor_mean: float
    impostor_std: float

    @property
    def is_fixture(self) -> bool:
        return self.provenance.is_fixture

    @property
    def citable(self) -> bool:
        return self.provenance.citable

    def confidence_interval_note(self) -> str:
        """Wilson interval width on the EER, given the sample size."""
        n = min(self.n_genuine, self.n_impostor)
        if n == 0:
            return "no data"
        low, high = wilson_interval(self.eer, n)
        return (
            "EER {0:.4f}, 95% CI [{1:.4f}, {2:.4f}] on n={3} "
            "(the smaller of the two class counts)".format(self.eer, low, high, n)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance.to_dict(),
            "model_id": self.model_id,
            "is_fixture": self.is_fixture,
            "citable": self.citable,
            "n_genuine": self.n_genuine,
            "n_impostor": self.n_impostor,
            "eer": round(self.eer, 6),
            "eer_threshold": round(self.eer_threshold, 6),
            "auc": round(self.auc, 6),
            "genuine_mean": round(self.genuine_mean, 6),
            "genuine_std": round(self.genuine_std, 6),
            "impostor_mean": round(self.impostor_mean, 6),
            "impostor_std": round(self.impostor_std, 6),
            "operating_points": [p.to_dict() for p in self.operating_points],
            "confidence_note": self.confidence_interval_note(),
        }

    def render(self) -> str:
        """Human-readable report. Refuses to present fixture data as results."""
        if self.is_fixture:
            return (
                FIXTURE_BANNER
                + "\n\nDataset : {0}\nPairs   : {1} genuine / {2} impostor\n"
                "Metrics computed successfully; withheld because this dataset "
                "is a fixture.".format(
                    self.provenance.name, self.n_genuine, self.n_impostor
                )
            )

        lines = [
            "CALIBRATION REPORT",
            "=" * 70,
            "dataset      : {0}".format(self.provenance.name),
            "model        : {0}".format(self.model_id),
            "genuine pairs: {0}".format(self.n_genuine),
            "impostor     : {0}".format(self.n_impostor),
            "",
            "genuine  similarity: mean {0:.4f}  sd {1:.4f}".format(
                self.genuine_mean, self.genuine_std
            ),
            "impostor similarity: mean {0:.4f}  sd {1:.4f}".format(
                self.impostor_mean, self.impostor_std
            ),
            "",
            "AUC          : {0:.4f}".format(self.auc),
            self.confidence_interval_note(),
            "",
            "operating points:",
            "  {0:>10}  {1:>10}  {2:>10}".format("threshold", "FAR", "FRR"),
        ]
        for point in self.operating_points:
            lines.append(
                "  {0:>10.4f}  {1:>10.4f}  {2:>10.4f}".format(
                    point.threshold, point.far, point.frr
                )
            )

        if not self.citable:
            lines += [
                "",
                "NOTE: this dataset carries no consent record and is marked "
                "non-citable.",
            ]
        return "\n".join(lines)


# ==========================================================================
# Core computations
# ==========================================================================


def roc_curve(
    genuine: Sequence[float], impostor: Sequence[float], *, steps: int = 512
) -> RocCurve:
    """Sweep thresholds across the observed score range."""
    g = np.asarray(genuine, dtype=np.float64)
    i = np.asarray(impostor, dtype=np.float64)

    if g.size == 0 or i.size == 0:
        raise ValueError(
            "ROC needs both genuine and impostor scores; got {0} and {1}".format(
                g.size, i.size
            )
        )

    low = float(min(g.min(), i.min()))
    high = float(max(g.max(), i.max()))
    if math.isclose(low, high):
        low, high = low - 1e-6, high + 1e-6

    thresholds = np.linspace(low, high, steps)

    # FAR: impostors at or above threshold. FRR: genuines below it.
    far = np.array([float(np.mean(i >= t)) for t in thresholds])
    frr = np.array([float(np.mean(g < t)) for t in thresholds])

    return RocCurve(thresholds=thresholds, far=far, frr=frr)


def equal_error_rate(curve: RocCurve) -> tuple[float, float]:
    """EER and the threshold achieving it.

    Returns the point minimising |FAR - FRR|; with discrete samples the two
    rarely coincide exactly, so the EER is reported as their mean there.
    """
    index = int(np.argmin(np.abs(curve.far - curve.frr)))
    eer = float((curve.far[index] + curve.frr[index]) / 2.0)
    return eer, float(curve.thresholds[index])


def threshold_at_far(curve: RocCurve, target_far: float) -> OperatingPoint:
    """Lowest threshold whose FAR does not exceed `target_far`.

    THE operating point for TRACELOCK: it fixes the rate of wrongly asserting
    a match, which is the error that actually harms someone.
    """
    if not 0.0 <= target_far <= 1.0:
        raise ValueError("target_far must be in [0, 1], got {0}".format(target_far))

    admissible = np.where(curve.far <= target_far)[0]
    index = int(admissible[0]) if admissible.size else int(np.argmin(curve.far))

    return OperatingPoint(
        threshold=float(curve.thresholds[index]),
        far=float(curve.far[index]),
        frr=float(curve.frr[index]),
    )


def wilson_interval(rate: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Wilson rather than the normal approximation because error rates near 0
    with small n are exactly where the naive interval breaks -- and small n
    near 0 is precisely our situation.
    """
    if n <= 0:
        return 0.0, 1.0

    denominator = 1.0 + z * z / n
    centre = (rate + z * z / (2 * n)) / denominator
    spread = (
        z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n))
    ) / denominator

    return max(0.0, centre - spread), min(1.0, centre + spread)


def evaluate(
    observations: ObservationSet, *, far_targets: Sequence[float] = (0.01, 0.001)
) -> CalibrationReport:
    """Full evaluation of an observation set.

    The returned report inherits `is_fixture` from the observations, so a
    fixture can never be laundered into an empirical-looking result.
    """
    genuine = observations.genuine_scores
    impostor = observations.impostor_scores

    if not genuine or not impostor:
        raise ValueError(
            "evaluation needs both classes; got {0} genuine and {1} impostor "
            "observations".format(len(genuine), len(impostor))
        )

    curve = roc_curve(genuine, impostor)
    eer, eer_threshold = equal_error_rate(curve)

    points = [threshold_at_far(curve, target) for target in far_targets]
    points.append(OperatingPoint(eer_threshold, eer, eer))

    return CalibrationReport(
        provenance=observations.provenance,
        model_id=observations.model_id,
        n_genuine=len(genuine),
        n_impostor=len(impostor),
        eer=eer,
        eer_threshold=eer_threshold,
        auc=curve.auc(),
        operating_points=tuple(points),
        genuine_mean=float(np.mean(genuine)),
        genuine_std=float(np.std(genuine)),
        impostor_mean=float(np.mean(impostor)),
        impostor_std=float(np.std(impostor)),
    )
