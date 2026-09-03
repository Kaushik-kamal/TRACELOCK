"""Classification policy.

THE HONEST POSITION ON THRESHOLDS
---------------------------------
We have no calibration data. Phase 1 shipped the calibration architecture and
deliberately shipped no results.

The wrong response is to pick a number that looks plausible and quietly treat
it as a decision boundary. The right response is a WIDE INCONCLUSIVE BAND:

    similarity < floor            REJECTED       clearly not supported
    floor <= similarity < ceiling INCONCLUSIVE   measured, cannot conclude
    similarity >= ceiling         VERIFIED       supported, still provisional

The gap between floor and ceiling is not sloppiness -- it is the width of our
actual uncertainty, made visible. Narrowing it requires calibration data, and
narrowing it without that data would be manufacturing confidence.

WHAT THIS MODULE MUST NEVER DO
------------------------------
Emit a probability. `0.62 cosine` is a geometric measurement. "62% same
person" is a calibrated statistical claim, and we have not earned it. Every
threshold below is stamped `calibrated=False` and travels into the artifact
with that stamp attached, so no downstream reader can mistake one for the other.

WHY THESE PARTICULAR NUMBERS
----------------------------
They are conservative round figures, not findings:

  0.35 floor    Well below any commonly cited ArcFace operating point. Chosen
                so REJECTED means "not close", not "just missed a cutoff".
  0.55 ceiling  Comfortably above typical impostor scores while remaining
                below where a confident claim would sit. Deliberately leaves a
                0.20-wide band of honest uncertainty.

Both are configuration, overridable per run, and recorded in the artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from tracelock.verification.phash import NEAR_DUPLICATE_MAX_DISTANCE


class SimilarityBand(str, Enum):
    """Coarse band for a similarity. NOT a probability, and never rendered as one."""

    LOW = "LOW"
    INDETERMINATE = "INDETERMINATE"
    HIGH = "HIGH"


@dataclass(frozen=True, slots=True)
class VerificationPolicy:
    """Operational filtering parameters. All provisional, all recorded."""

    # --- face similarity (PROVISIONAL, UNCALIBRATED) --------------------
    similarity_floor: float = 0.35
    similarity_ceiling: float = 0.55

    # --- quality gating -------------------------------------------------
    # A degraded face yields an unreliable embedding. Blur in particular pulls
    # embeddings toward the population mean and INFLATES similarity, so a
    # quality floor is a guard against false ACCEPTS, not just weak evidence.
    min_face_quality: float = 0.20
    reject_on_low_quality: bool = False  # warn by default; opt in to reject

    # --- multi-face handling --------------------------------------------
    reject_ambiguous_faces: bool = False  # warn by default

    # --- perceptual hashing ---------------------------------------------
    phash_near_duplicate_max_distance: int = NEAR_DUPLICATE_MAX_DISTANCE

    # --- duplicates ------------------------------------------------------
    reject_exact_duplicates: bool = True

    # A fitted CalibrationModel, or None. `calibrated` is DERIVED from this,
    # never set directly -- claiming calibration requires producing the model.
    calibration: Any = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.similarity_floor <= 1.0:
            raise ValueError("similarity_floor must be in [0, 1]")
        if not 0.0 <= self.similarity_ceiling <= 1.0:
            raise ValueError("similarity_ceiling must be in [0, 1]")
        if self.similarity_floor > self.similarity_ceiling:
            raise ValueError(
                "similarity_floor ({0}) cannot exceed similarity_ceiling "
                "({1})".format(self.similarity_floor, self.similarity_ceiling)
            )
        if self.calibration is not None:
            for attribute in ("platt_a", "platt_b", "n_genuine", "n_impostor"):
                if not hasattr(self.calibration, attribute):
                    raise ValueError(
                        "calibration must be a fitted CalibrationModel, got "
                        "{0!r}".format(type(self.calibration).__name__)
                    )
            if self.calibration.n_genuine < 1 or self.calibration.n_impostor < 1:
                raise ValueError(
                    "a calibration model must be fitted on both classes"
                )

    @property
    def calibrated(self) -> bool:
        """True only when backed by a fitted model. Not settable."""
        return self.calibration is not None

    @classmethod
    def from_calibration(cls, model, **overrides) -> "VerificationPolicy":
        """Build a policy whose boundaries come from a fitted model."""
        reject_below, verify_at = model.recommended_boundaries()
        params = dict(
            similarity_floor=round(reject_below, 4),
            similarity_ceiling=round(verify_at, 4),
            calibration=model,
        )
        params.update(overrides)
        return cls(**params)

    def identity_probability(self, similarity: float) -> float | None:
        """Calibrated P(same identity | similarity), or None when uncalibrated.

        Returning None rather than a number is deliberate: without a model
        there IS no probability, and inventing one is the failure this project
        has guarded against throughout.
        """
        if self.calibration is None:
            return None
        return self.calibration.probability(similarity)

    @property
    def inconclusive_band_width(self) -> float:
        """How wide our admitted uncertainty is. Reported, not hidden."""
        return self.similarity_ceiling - self.similarity_floor

    def band_for(self, similarity: float) -> SimilarityBand:
        if similarity < self.similarity_floor:
            return SimilarityBand.LOW
        if similarity < self.similarity_ceiling:
            return SimilarityBand.INDETERMINATE
        return SimilarityBand.HIGH

    def to_dict(self) -> dict[str, Any]:
        return {
            "similarity_floor": self.similarity_floor,
            "similarity_ceiling": self.similarity_ceiling,
            "inconclusive_band_width": round(self.inconclusive_band_width, 4),
            "min_face_quality": self.min_face_quality,
            "reject_on_low_quality": self.reject_on_low_quality,
            "reject_ambiguous_faces": self.reject_ambiguous_faces,
            "phash_near_duplicate_max_distance": self.phash_near_duplicate_max_distance,
            "reject_exact_duplicates": self.reject_exact_duplicates,
            "calibrated": self.calibrated,
            "thresholds_are_provisional": not self.calibrated,
            "calibration_model": (
                {
                    "model_id": self.calibration.model_id,
                    "fitted_at": self.calibration.fitted_at,
                    "n_genuine": self.calibration.n_genuine,
                    "n_impostor": self.calibration.n_impostor,
                    "auc": self.calibration.auc,
                    "separation": round(self.calibration.separation, 6),
                    "confidence_note": self.calibration.confidence_note(),
                }
                if self.calibration
                else None
            ),
            "disclaimer": (
                (
                    "Thresholds are derived from a FITTED calibration model. "
                    "Identity probabilities are calibrated but the sample is "
                    "small -- see calibration_model.confidence_note for the "
                    "interval. Calibrated does not mean certain."
                )
                if self.calibrated
                else (
                    "Similarity thresholds are PROVISIONAL and UNCALIBRATED. "
                    "They are operational filters, not identity probabilities. "
                    "No probability of identity is claimed anywhere."
                )
            ),
        }


DEFAULT_POLICY = VerificationPolicy()
