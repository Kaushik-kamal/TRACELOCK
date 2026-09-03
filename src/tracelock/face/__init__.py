"""Face Intelligence Engine.

Produces measurements, never identity verdicts. Converting a similarity into
an identity probability requires calibration -- see `tracelock.calibration`.
"""

from tracelock.face.engine import FaceEngine
from tracelock.face.errors import (
    AmbiguousFaceError,
    DetectionError,
    EmbeddingDimensionMismatch,
    EmbeddingError,
    FaceEngineError,
    ImageError,
    ImageNotFoundError,
    InvalidEmbeddingError,
    ModelInitializationError,
    NoFaceDetectedError,
    UnsupportedImageError,
    ZeroNormEmbeddingError,
)
from tracelock.face.models import (
    AnalysisWarning,
    BoundingBox,
    DetectedFace,
    Embedding,
    FaceAnalysisResult,
    FaceSummary,
    ImageRef,
    ModelProvenance,
    Pose,
    QualityBand,
    QualityMetric,
    QualityReport,
    QuantizationSpec,
    SelectionReport,
    WarningCode,
)
from tracelock.face.similarity import (
    angular_distance,
    cosine_similarity,
    similarity_matrix,
    validate_similarity,
)

__all__ = [
    "FaceEngine",
    "FaceAnalysisResult",
    "DetectedFace",
    "FaceSummary",
    "Embedding",
    "QuantizationSpec",
    "BoundingBox",
    "Pose",
    "ImageRef",
    "ModelProvenance",
    "SelectionReport",
    "QualityReport",
    "QualityMetric",
    "QualityBand",
    "AnalysisWarning",
    "WarningCode",
    "cosine_similarity",
    "angular_distance",
    "similarity_matrix",
    "validate_similarity",
    "FaceEngineError",
    "ModelInitializationError",
    "ImageError",
    "ImageNotFoundError",
    "UnsupportedImageError",
    "DetectionError",
    "NoFaceDetectedError",
    "AmbiguousFaceError",
    "EmbeddingError",
    "InvalidEmbeddingError",
    "EmbeddingDimensionMismatch",
    "ZeroNormEmbeddingError",
]
