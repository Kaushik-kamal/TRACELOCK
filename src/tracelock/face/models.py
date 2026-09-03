"""Structured, immutable results from the Face Intelligence Engine.

WHY FROZEN DATACLASSES AND NOT PYDANTIC
---------------------------------------
Phase 0 used Pydantic for `Candidate` because that data arrives as untrusted
JSON from third-party APIs and needs coercion and validation at the boundary.

Face results are the opposite problem: they originate in-process from numpy,
need read-only enforcement on a 512-float array, and are never parsed from
external input. Frozen dataclasses give immutability without fighting Pydantic
over numpy types. Serialization is explicit via `to_dict()`.

EMBEDDINGS AND THE BLOCKCHAIN
-----------------------------
`Embedding.vector` is a read-only numpy array and MUST NOT be serialized into
any evidence structure. A face embedding is biometric data under GDPR Art. 9;
on a public immutable ledger it can never be withdrawn. `to_dict()` therefore
omits the vector by default and emits only its dimension and norm.

Phase 4 will commit `quantize()` output instead -- the hook exists here so that
landing it requires no public API change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

import numpy as np

from tracelock.face.errors import InvalidEmbeddingError

SCHEMA_VERSION = "face-result/1"


# ==========================================================================
# Geometry
# ==========================================================================


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned face box in pixel coordinates (x1, y1) top-left."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def min_side(self) -> float:
        """Shorter side. The binding constraint on how much face detail exists."""
        return min(self.width, self.height)

    def to_dict(self) -> dict[str, float]:
        return {
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
            "width": round(self.width, 2),
            "height": round(self.height, 2),
        }


@dataclass(frozen=True, slots=True)
class Pose:
    """Head pose in DEGREES, from buffalo_l's 1k3d68 landmark model.

    Units confirmed empirically: rotating the input image by +20 degrees moved
    `roll` to -19.76 while yaw and pitch stayed flat.

    Roll is deliberately excluded from the frontality metric -- it is in-plane
    rotation, which the 5-point similarity alignment corrects before the
    recognition model ever sees the crop. Penalising it would double-count a
    problem that has already been solved.
    """

    pitch: float
    yaw: float
    roll: float

    @property
    def frontal_deviation_deg(self) -> float:
        """Out-of-plane deviation from frontal. Roll excluded, by design."""
        return math.hypot(self.pitch, self.yaw)

    def to_dict(self) -> dict[str, float]:
        return {
            "pitch_deg": round(self.pitch, 3),
            "yaw_deg": round(self.yaw, 3),
            "roll_deg": round(self.roll, 3),
            "frontal_deviation_deg": round(self.frontal_deviation_deg, 3),
        }


# ==========================================================================
# Embedding
# ==========================================================================


@dataclass(frozen=True, slots=True)
class QuantizationSpec:
    """How an embedding is reduced to deterministic bytes for future hashing.

    PHASE 1 DOES NOT HASH ANYTHING. This exists so Phase 4 can commit an
    embedding without changing any public signature.

    `int8_scaled` at scale=127 measured on the live probe: round-trip cosine
    against the original is 0.998635, occupying only [-24, 15] of the int8
    range. That range under-use is deliberate and harmless: the output is a
    commitment input, never a reconstruction target, and coarser quantization
    is MORE robust to cross-machine float drift, which is the actual threat.
    """

    version: str = "int8-v1"
    method: Literal["int8_scaled", "round_decimals"] = "int8_scaled"
    scale: int = 127
    decimals: int = 6

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "method": self.method,
            "scale": self.scale,
            "decimals": self.decimals,
        }


DEFAULT_QUANTIZATION = QuantizationSpec()


@dataclass(frozen=True, slots=True)
class Embedding:
    """An L2-normalized face embedding. The vector is read-only."""

    vector: np.ndarray
    model_id: str

    def __post_init__(self) -> None:
        vector = self.vector
        if vector.ndim != 1:
            raise InvalidEmbeddingError(
                "embedding must be 1-D, got shape {0}".format(vector.shape)
            )
        if not np.isfinite(vector).all():
            raise InvalidEmbeddingError("embedding contains NaN or Inf")
        # Enforce immutability. Bypass frozen dataclass to store the guarded array.
        guarded = np.array(vector, dtype=np.float32, copy=True)
        guarded.setflags(write=False)
        object.__setattr__(self, "vector", guarded)

    @property
    def dimension(self) -> int:
        return int(self.vector.shape[0])

    @property
    def l2_norm(self) -> float:
        return float(np.linalg.norm(self.vector))

    @property
    def is_normalized(self) -> bool:
        return math.isclose(self.l2_norm, 1.0, abs_tol=1e-5)

    def quantize(self, spec: QuantizationSpec = DEFAULT_QUANTIZATION) -> bytes:
        """Deterministic byte encoding. Phase 4 hashes THIS, never the floats."""
        if spec.method == "int8_scaled":
            return np.round(self.vector * spec.scale).astype(np.int8).tobytes()
        if spec.method == "round_decimals":
            return np.round(self.vector, spec.decimals).astype(np.float32).tobytes()
        raise ValueError("unknown quantization method: {0}".format(spec.method))

    def to_dict(self, *, include_vector: bool = False) -> dict[str, Any]:
        """Serialize. The vector is EXCLUDED by default -- see module docstring."""
        payload: dict[str, Any] = {
            "model_id": self.model_id,
            "dimension": self.dimension,
            "l2_norm": round(self.l2_norm, 8),
            "is_normalized": self.is_normalized,
        }
        if include_vector:
            # Callers must opt in explicitly, and must never route this into
            # an evidence bundle.
            payload["vector"] = [float(v) for v in self.vector]
        return payload


# ==========================================================================
# Quality
# ==========================================================================


class QualityBand(str, Enum):
    """Coarse buckets over the aggregate score.

    Cutoffs are WORKING DEFAULTS, not calibrated findings. They are documented
    as such and must not be cited as empirical thresholds.
    """

    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    MARGINAL = "MARGINAL"
    POOR = "POOR"

    @classmethod
    def from_score(cls, score: float) -> "QualityBand":
        if score >= 0.80:
            return cls.EXCELLENT
        if score >= 0.60:
            return cls.GOOD
        if score >= 0.35:
            return cls.MARGINAL
        return cls.POOR


@dataclass(frozen=True, slots=True)
class QualityMetric:
    """One interpretable quality signal.

    Carries both the RAW measurement and the normalized [0,1] score, because a
    normalized score alone hides the evidence. `uncalibrated` marks metrics
    whose normalization constant is a working default rather than a measured
    one -- honesty about which numbers are earned.
    """

    name: str
    raw_value: float
    score: float
    unit: str
    uncalibrated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "raw_value": round(self.raw_value, 4),
            "score": round(self.score, 4),
            "unit": self.unit,
            "uncalibrated": self.uncalibrated,
        }


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Per-metric scores plus an explainable aggregate.

    AGGREGATION: weighted GEOMETRIC mean, not arithmetic.

    A geometric mean cannot be rescued by a strong term: a 20px face is
    unusable no matter how sharp or well-lit it is, and an arithmetic mean
    would let sharpness mask that. Same argument as the Phase 3 trust score --
    a necessary condition must gate, not merely contribute.
    """

    metrics: tuple[QualityMetric, ...]
    weights: dict[str, float]
    aggregate: float
    band: QualityBand

    def get(self, name: str) -> QualityMetric | None:
        return next((m for m in self.metrics if m.name == name), None)

    def explain(self) -> list[str]:
        """Human-readable contribution breakdown, ordered by weight."""
        lines = []
        for metric in sorted(
            self.metrics, key=lambda m: self.weights.get(m.name, 0.0), reverse=True
        ):
            weight = self.weights.get(metric.name, 0.0)
            lines.append(
                "{0:<22} raw={1:>10.3f} {2:<6} score={3:.3f}  weight={4:.2f}{5}".format(
                    metric.name,
                    metric.raw_value,
                    metric.unit,
                    metric.score,
                    weight,
                    "  [uncalibrated]" if metric.uncalibrated else "",
                )
            )
        return lines

    def to_dict(self) -> dict[str, Any]:
        return {
            "aggregate": round(self.aggregate, 4),
            "band": self.band.value,
            "aggregation": "weighted_geometric_mean",
            "weights": self.weights,
            "metrics": [m.to_dict() for m in self.metrics],
        }


# ==========================================================================
# Warnings
# ==========================================================================


class WarningCode(str, Enum):
    """Non-fatal observations that later phases may act on."""

    MULTIPLE_FACES = "MULTIPLE_FACES"
    AMBIGUOUS_PRIMARY_FACE = "AMBIGUOUS_PRIMARY_FACE"
    LOW_DETECTION_CONFIDENCE = "LOW_DETECTION_CONFIDENCE"
    SMALL_FACE = "SMALL_FACE"
    BLURRY_FACE = "BLURRY_FACE"
    EXTREME_POSE = "EXTREME_POSE"
    TRUNCATED_FACE = "TRUNCATED_FACE"
    EXPOSURE_CLIPPING = "EXPOSURE_CLIPPING"
    LOW_AGGREGATE_QUALITY = "LOW_AGGREGATE_QUALITY"
    EMBEDDING_NOT_NORMALIZED = "EMBEDDING_NOT_NORMALIZED"


@dataclass(frozen=True, slots=True)
class AnalysisWarning:
    code: WarningCode
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code.value, "message": self.message, "detail": self.detail}


# ==========================================================================
# Provenance
# ==========================================================================


@dataclass(frozen=True, slots=True)
class ModelProvenance:
    """Everything needed to answer: why did this image produce this embedding?

    `model_file_sha256` is the real version pin. A pack NAME can be re-released;
    the bytes cannot. Without per-file digests, "buffalo_l" is not a reproducible
    statement.
    """

    pack_name: str
    model_dir: str
    recognition_model: str
    detection_model: str
    embedding_dimension: int
    det_size: tuple[int, int]
    ctx_id: int
    providers: tuple[str, ...]
    model_file_sha256: dict[str, str]
    package_versions: dict[str, str]

    @property
    def model_id(self) -> str:
        """Short stable identifier for this exact model configuration."""
        return "{0}:{1}".format(self.pack_name, self.recognition_model)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pack_name": self.pack_name,
            "model_id": self.model_id,
            "model_dir": self.model_dir,
            "recognition_model": self.recognition_model,
            "detection_model": self.detection_model,
            "embedding_dimension": self.embedding_dimension,
            "det_size": list(self.det_size),
            "ctx_id": self.ctx_id,
            "providers": list(self.providers),
            "model_file_sha256": self.model_file_sha256,
            "package_versions": self.package_versions,
        }


@dataclass(frozen=True, slots=True)
class ImageRef:
    """The analyzed image, identified by content rather than by path."""

    path: str
    sha256: str
    width: int
    height: int
    channels: int

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
        }


# ==========================================================================
# Faces
# ==========================================================================


@dataclass(frozen=True, slots=True)
class FaceSummary:
    """A detected face that was NOT selected as primary.

    Retained deliberately. Discarding non-primary faces would erase the reason
    the selection was or was not ambiguous, and multi-face context feeds the
    Phase 2 multiple-comparisons penalty.
    """

    index: int
    bbox: BoundingBox
    det_score: float
    selection_score: float
    area_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "bbox": self.bbox.to_dict(),
            "det_score": round(self.det_score, 4),
            "selection_score": round(self.selection_score, 4),
            "area_ratio": round(self.area_ratio, 5),
        }


@dataclass(frozen=True, slots=True)
class DetectedFace:
    """The primary face, fully analyzed."""

    index: int
    bbox: BoundingBox
    det_score: float
    landmarks_5: np.ndarray
    pose: Pose | None
    embedding: Embedding
    quality: QualityReport
    age_estimate: int | None = None
    sex_estimate: str | None = None

    def __post_init__(self) -> None:
        guarded = np.array(self.landmarks_5, dtype=np.float32, copy=True)
        guarded.setflags(write=False)
        object.__setattr__(self, "landmarks_5", guarded)

    def to_dict(self, *, include_vector: bool = False) -> dict[str, Any]:
        return {
            "index": self.index,
            "bbox": self.bbox.to_dict(),
            "det_score": round(self.det_score, 4),
            "landmarks_5": [[round(float(x), 2), round(float(y), 2)] for x, y in self.landmarks_5],
            "pose": self.pose.to_dict() if self.pose else None,
            "embedding": self.embedding.to_dict(include_vector=include_vector),
            "quality": self.quality.to_dict(),
            # Model ESTIMATES, not facts. Present for the Phase 5 minors
            # safeguard; never to be treated as ground truth about a person.
            "age_estimate": self.age_estimate,
            "sex_estimate": self.sex_estimate,
        }


@dataclass(frozen=True, slots=True)
class SelectionReport:
    """Why this face was chosen. Makes the policy auditable after the fact."""

    policy: str
    weights: dict[str, float]
    chosen_index: int
    margin: float
    ambiguous: bool
    ambiguity_threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "weights": self.weights,
            "chosen_index": self.chosen_index,
            "margin": round(self.margin, 4),
            "ambiguous": self.ambiguous,
            "ambiguity_threshold": self.ambiguity_threshold,
        }


@dataclass(frozen=True, slots=True)
class FaceAnalysisResult:
    """Complete structured output of one analysis."""

    schema_version: str
    image: ImageRef
    model: ModelProvenance
    faces_detected: int
    primary: DetectedFace
    others: tuple[FaceSummary, ...]
    selection: SelectionReport
    warnings: tuple[AnalysisWarning, ...]
    analyzed_at: datetime
    elapsed_seconds: float

    @property
    def embedding(self) -> Embedding:
        """Convenience accessor for the primary face's embedding."""
        return self.primary.embedding

    @property
    def has_warning(self) -> bool:
        return bool(self.warnings)

    def warning_codes(self) -> tuple[str, ...]:
        return tuple(w.code.value for w in self.warnings)

    def to_dict(self, *, include_vector: bool = False) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "analyzed_at": self.analyzed_at.isoformat(),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "image": self.image.to_dict(),
            "model": self.model.to_dict(),
            "faces_detected": self.faces_detected,
            "primary": self.primary.to_dict(include_vector=include_vector),
            "others": [f.to_dict() for f in self.others],
            "selection": self.selection.to_dict(),
            "warnings": [w.to_dict() for w in self.warnings],
        }


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
