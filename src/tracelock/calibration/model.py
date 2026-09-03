"""The fitted calibration model.

Turns a cosine similarity into a probability of identity match, using
coefficients fitted to LABELLED PAIRS. Until such a model exists, nothing in
TRACELOCK is permitted to express an identity probability -- that rule is
enforced structurally by `VerificationPolicy`, which derives `calibrated` from
the presence of a model rather than from a settable flag.

PLATT SCALING
-------------
    P(match | s) = sigmoid(a * s + b)

One feature, two parameters, fitted by Newton-Raphson on the log-likelihood.
Implemented here rather than imported from scikit-learn for the same reason the
ROC code is: in a forensic system the reader should be able to audit the
estimator without leaving the repository. It is about thirty lines.

WHAT A MODEL DOES NOT DO
------------------------
It does not make a small dataset large. `n_genuine` and `n_impostor` travel
with the model, every report prints the Wilson interval, and a model fitted on
a few dozen pairs says so loudly. Calibrated does not mean certain.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tracelock.calibration.contract import DatasetProvenance
from tracelock.calibration.metrics import (
    equal_error_rate,
    roc_curve,
    threshold_at_far,
    wilson_interval,
)

MODEL_SCHEMA_VERSION = "calibration-model/1"

# Operating points fitted for and recorded on every model.
# FAR is the error that matters: a false accept asserts that an innocent person
# appears in discovered content. A false reject only loses a candidate.
DEFAULT_FAR_TARGETS: tuple[float, ...] = (0.01, 0.001)


def sigmoid(x: float | np.ndarray):
    return 1.0 / (1.0 + np.exp(-x))


def fit_platt(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    max_iterations: int = 100,
    tolerance: float = 1e-9,
) -> tuple[float, float]:
    """Fit P(match|s) = sigmoid(a*s + b) by Newton-Raphson.

    `labels` is 1 for genuine, 0 for impostor.

    Uses Platt's own target smoothing rather than raw 0/1 labels. With few
    positives, unsmoothed targets drive the coefficients toward a hard step at
    the boundary -- a model that reports 0.999 on the strength of a handful of
    pairs. Smoothing keeps the fitted probabilities honest about sample size.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)

    if s.size != y.size or s.size == 0:
        raise ValueError("scores and labels must be non-empty and the same length")

    n_pos = float(np.sum(y == 1))
    n_neg = float(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            "both classes are required to fit a calibration: got {0} genuine "
            "and {1} impostor".format(int(n_pos), int(n_neg))
        )

    hi = 1.0 / (n_pos + 2.0)
    lo = 1.0 / (n_neg + 2.0)
    target = np.where(y == 1, 1.0 - hi, lo)

    a, b = 0.0, 0.0
    for _ in range(max_iterations):
        p = sigmoid(a * s + b)
        w = np.clip(p * (1.0 - p), 1e-12, None)
        residual = p - target

        grad_a = float(np.sum(residual * s))
        grad_b = float(np.sum(residual))

        h_aa = float(np.sum(w * s * s)) + 1e-12
        h_ab = float(np.sum(w * s))
        h_bb = float(np.sum(w)) + 1e-12

        determinant = h_aa * h_bb - h_ab * h_ab
        if abs(determinant) < 1e-18:
            break

        step_a = (h_bb * grad_a - h_ab * grad_b) / determinant
        step_b = (h_aa * grad_b - h_ab * grad_a) / determinant

        a -= step_a
        b -= step_b

        if max(abs(step_a), abs(step_b)) < tolerance:
            break

    return float(a), float(b)


@dataclass(frozen=True, slots=True)
class CalibrationModel:
    """A fitted mapping from cosine similarity to identity probability."""

    schema_version: str
    platt_a: float
    platt_b: float

    n_genuine: int
    n_impostor: int
    eer: float
    eer_threshold: float
    auc: float

    # threshold -> (far, frr) at each fitted operating point
    operating_points: dict[str, dict[str, float]]

    genuine_mean: float
    genuine_min: float
    impostor_mean: float
    impostor_max: float

    provenance: dict[str, Any]
    labeling_basis: str
    fitted_at: str
    model_id: str

    # ------------------------------------------------------------------

    def probability(self, cosine_similarity: float) -> float:
        """Calibrated P(same identity | similarity). This is the ONLY place in
        TRACELOCK permitted to produce an identity probability."""
        return float(sigmoid(self.platt_a * cosine_similarity + self.platt_b))

    def threshold_for(self, far_target: float) -> float:
        key = "far_{0:g}".format(far_target)
        if key not in self.operating_points:
            raise KeyError(
                "no operating point fitted for FAR={0}; available: {1}".format(
                    far_target, sorted(self.operating_points)
                )
            )
        return self.operating_points[key]["threshold"]

    @property
    def separation(self) -> float:
        """Gap between the impostor maximum and the genuine minimum.

        Positive means the two populations do not overlap in this sample. It is
        a description of THIS dataset, not a generalisation guarantee.
        """
        return self.genuine_min - self.impostor_max

    @property
    def is_separable(self) -> bool:
        return self.separation > 0

    def recommended_boundaries(self) -> tuple[float, float]:
        """(reject_below, verify_at_or_above) in cosine space.

        WHY NOT THE FITTED FAR THRESHOLD ALONE
        --------------------------------------
        With cleanly separated populations every FAR target collapses onto the
        same threshold, sitting flush against the impostor maximum. Zero
        observed false accepts in 35 impostors does NOT mean FAR is zero: by
        the rule of three the 95% upper bound is 3/35, about 9%. Placing the
        VERIFIED boundary there would treat a small sample as certainty.

        So two boundaries, each principled rather than tuned:

          reject_below      the fitted FAR operating point. Below it the model
                            says impostor and the data agrees.

          verify_at_or_above the MAXIMUM-MARGIN midpoint between the observed
                            impostor maximum and genuine minimum. This is the
                            SVM decision rule: the point furthest from both
                            populations. It is derived from the data without
                            being chosen to produce a desired verdict.

        Between them sits the honest indeterminate band.
        """
        reject_below = self.operating_points.get("far_0.01", {}).get(
            "threshold", self.eer_threshold
        )

        if self.is_separable:
            verify_at = (self.impostor_max + self.genuine_min) / 2.0
        else:
            # Overlapping populations: no margin exists. Fall back to the
            # genuine mean, and `is_separable` warns the caller.
            verify_at = self.genuine_mean

        return float(reject_below), float(max(verify_at, reject_below))

    def confidence_note(self) -> str:
        n = min(self.n_genuine, self.n_impostor)
        low, high = wilson_interval(self.eer, n)
        return (
            "EER {0:.4f}, 95% CI [{1:.4f}, {2:.4f}] on n={3} (the smaller class). "
            "With a sample this size the interval is wide; treat the operating "
            "points as indicative, not definitive.".format(self.eer, low, high, n)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "fitted_at": self.fitted_at,
            "platt_a": round(self.platt_a, 8),
            "platt_b": round(self.platt_b, 8),
            "n_genuine": self.n_genuine,
            "n_impostor": self.n_impostor,
            "eer": round(self.eer, 6),
            "eer_threshold": round(self.eer_threshold, 6),
            "auc": round(self.auc, 6),
            "operating_points": self.operating_points,
            "distribution": {
                "genuine_mean": round(self.genuine_mean, 6),
                "genuine_min": round(self.genuine_min, 6),
                "impostor_mean": round(self.impostor_mean, 6),
                "impostor_max": round(self.impostor_max, 6),
                "separation": round(self.separation, 6),
                "is_separable": self.is_separable,
            },
            "labeling_basis": self.labeling_basis,
            "provenance": self.provenance,
            "confidence_note": self.confidence_note(),
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        distribution = payload.get("distribution", {})
        return cls(
            schema_version=payload["schema_version"],
            platt_a=payload["platt_a"],
            platt_b=payload["platt_b"],
            n_genuine=payload["n_genuine"],
            n_impostor=payload["n_impostor"],
            eer=payload["eer"],
            eer_threshold=payload["eer_threshold"],
            auc=payload["auc"],
            operating_points=payload["operating_points"],
            genuine_mean=distribution.get("genuine_mean", 0.0),
            genuine_min=distribution.get("genuine_min", 0.0),
            impostor_mean=distribution.get("impostor_mean", 0.0),
            impostor_max=distribution.get("impostor_max", 0.0),
            provenance=payload.get("provenance", {}),
            labeling_basis=payload.get("labeling_basis", ""),
            fitted_at=payload["fitted_at"],
            model_id=payload["model_id"],
        )

    def render(self) -> str:
        lines = [
            "CALIBRATION MODEL  {0}".format(self.model_id),
            "=" * 70,
            "fitted           : {0}".format(self.fitted_at),
            "pairs            : {0} genuine / {1} impostor".format(
                self.n_genuine, self.n_impostor
            ),
            "",
            "similarity distributions",
            "  genuine        : mean {0:.4f}   min {1:.4f}".format(
                self.genuine_mean, self.genuine_min
            ),
            "  impostor       : mean {0:.4f}   max {1:.4f}".format(
                self.impostor_mean, self.impostor_max
            ),
            "  separation     : {0:+.4f}  ({1})".format(
                self.separation,
                "populations do not overlap in this sample"
                if self.is_separable
                else "POPULATIONS OVERLAP -- thresholds cannot separate cleanly",
            ),
            "",
            "P(match | s)     = sigmoid({0:.4f} * s + {1:.4f})".format(
                self.platt_a, self.platt_b
            ),
            "AUC              : {0:.4f}".format(self.auc),
            "",
            "operating points",
            "  {0:<14} {1:>10} {2:>10} {3:>10}".format("target", "threshold", "FAR", "FRR"),
        ]
        for name, point in sorted(self.operating_points.items()):
            lines.append(
                "  {0:<14} {1:>10.4f} {2:>10.4f} {3:>10.4f}".format(
                    name, point["threshold"], point["far"], point["frr"]
                )
            )
        lines += ["", self.confidence_note()]
        return "\n".join(lines)


def fit_calibration_model(
    genuine: Sequence[float],
    impostor: Sequence[float],
    *,
    provenance: DatasetProvenance,
    labeling_basis: str,
    far_targets: Sequence[float] = DEFAULT_FAR_TARGETS,
    model_id: str = "tracelock-calibration-v1",
) -> CalibrationModel:
    """Fit a calibration model from labelled similarity scores."""
    if provenance.is_fixture:
        raise ValueError(
            "refusing to fit a calibration model from FIXTURE data. Synthetic "
            "scores verify the metric code; they cannot calibrate a face model."
        )

    g = list(genuine)
    i = list(impostor)
    if not g or not i:
        raise ValueError(
            "both classes required: got {0} genuine, {1} impostor".format(len(g), len(i))
        )

    scores = g + i
    labels = [1] * len(g) + [0] * len(i)
    platt_a, platt_b = fit_platt(scores, labels)

    curve = roc_curve(g, i)
    eer, eer_threshold = equal_error_rate(curve)

    points: dict[str, dict[str, float]] = {}
    for target in far_targets:
        point = threshold_at_far(curve, target)
        points["far_{0:g}".format(target)] = {
            "threshold": round(point.threshold, 6),
            "far": round(point.far, 6),
            "frr": round(point.frr, 6),
        }
    points["eer"] = {
        "threshold": round(eer_threshold, 6),
        "far": round(eer, 6),
        "frr": round(eer, 6),
    }

    return CalibrationModel(
        schema_version=MODEL_SCHEMA_VERSION,
        platt_a=platt_a,
        platt_b=platt_b,
        n_genuine=len(g),
        n_impostor=len(i),
        eer=eer,
        eer_threshold=eer_threshold,
        auc=curve.auc(),
        operating_points=points,
        genuine_mean=float(np.mean(g)),
        genuine_min=float(np.min(g)),
        impostor_mean=float(np.mean(i)),
        impostor_max=float(np.max(i)),
        provenance=provenance.to_dict(),
        labeling_basis=labeling_basis,
        fitted_at=datetime.now(timezone.utc).isoformat(),
        model_id=model_id,
    )
