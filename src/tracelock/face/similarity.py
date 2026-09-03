"""Similarity utilities for face embeddings.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
-----------------------------------------
It does not decide whether two faces are the same person.

There is no threshold constant here, and no function returns a boolean verdict.
A cosine similarity is a geometric quantity; an identity claim is a statistical
one. Converting between them requires a calibrated mapping fitted to labelled
pairs -- see `tracelock.calibration`. Writing `if sim > 0.5` anywhere in this
codebase would be inventing empirical evidence that does not exist.

WHY COSINE
----------
ArcFace optimises an additive ANGULAR margin, so angular separation is the
quantity the model was actually trained to make meaningful. Cosine similarity
on L2-normalized embeddings is exactly that, which is why the threshold is
comparatively stable across datasets -- unlike softmax-trained embeddings where
Euclidean distance drifts.
"""

from __future__ import annotations

import math

import numpy as np

from tracelock.face.errors import (
    EmbeddingDimensionMismatch,
    InvalidEmbeddingError,
    ZeroNormEmbeddingError,
)
from tracelock.face.models import Embedding

# Below this L2 norm an embedding has no meaningful direction.
MIN_NORM = 1e-8


def _as_vector(value: Embedding | np.ndarray, *, label: str) -> np.ndarray:
    """Accept either an Embedding or a raw array; validate structure."""
    vector = value.vector if isinstance(value, Embedding) else np.asarray(value)

    if vector.ndim != 1:
        raise InvalidEmbeddingError(
            "{0} must be 1-D, got shape {1}".format(label, vector.shape)
        )
    if vector.size == 0:
        raise InvalidEmbeddingError("{0} is empty".format(label))
    if not np.isfinite(vector).all():
        raise InvalidEmbeddingError("{0} contains NaN or Inf".format(label))

    return vector.astype(np.float64, copy=False)


def cosine_similarity(
    left: Embedding | np.ndarray, right: Embedding | np.ndarray
) -> float:
    """Cosine similarity in [-1, 1].

    Raises
        EmbeddingDimensionMismatch  differing dimensions -- almost always
                                    embeddings from different model packs,
                                    which must never be compared silently.
        ZeroNormEmbeddingError      either vector has no direction. Returning
                                    0.0 here would be a silent lie: 0.0 is a
                                    real, meaningful value (orthogonal).
        InvalidEmbeddingError       NaN, Inf, empty, or wrong rank.
    """
    a = _as_vector(left, label="left embedding")
    b = _as_vector(right, label="right embedding")

    if a.shape[0] != b.shape[0]:
        raise EmbeddingDimensionMismatch(a.shape[0], b.shape[0])

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))

    if norm_a < MIN_NORM:
        raise ZeroNormEmbeddingError("left embedding", norm_a)
    if norm_b < MIN_NORM:
        raise ZeroNormEmbeddingError("right embedding", norm_b)

    similarity = float(np.dot(a, b) / (norm_a * norm_b))

    # Floating point can nudge a unit-vector dot product a few ULPs outside
    # [-1, 1]; clamp so downstream arccos never sees a domain error.
    return max(-1.0, min(1.0, similarity))


def angular_distance(
    left: Embedding | np.ndarray, right: Embedding | np.ndarray
) -> float:
    """Normalized angular distance in [0, 1]: arccos(cos) / pi.

    Provided because ArcFace's training objective is angular. Angular distance
    is linear in the quantity the model optimises, whereas cosine compresses
    differences near +/-1 -- which matters when fitting a calibration curve.
    """
    return math.acos(cosine_similarity(left, right)) / math.pi


def similarity_matrix(
    left: list[Embedding | np.ndarray], right: list[Embedding | np.ndarray]
) -> np.ndarray:
    """Pairwise cosine similarities, shape (len(left), len(right)).

    Exists for calibration, which computes thousands of pairs. Validates and
    normalizes once per vector instead of once per pair.
    """
    if not left or not right:
        return np.zeros((len(left), len(right)), dtype=np.float64)

    def stack(values, label):
        vectors = [_as_vector(v, label=label) for v in values]
        dimension = vectors[0].shape[0]
        for vector in vectors[1:]:
            if vector.shape[0] != dimension:
                raise EmbeddingDimensionMismatch(dimension, vector.shape[0])
        matrix = np.vstack(vectors)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if float(norms.min()) < MIN_NORM:
            raise ZeroNormEmbeddingError(label, float(norms.min()))
        return matrix / norms

    a = stack(left, "left embedding")
    b = stack(right, "right embedding")

    if a.shape[1] != b.shape[1]:
        raise EmbeddingDimensionMismatch(a.shape[1], b.shape[1])

    return np.clip(a @ b.T, -1.0, 1.0)


def validate_similarity(value: float) -> float:
    """Assert a similarity is a finite number in [-1, 1]. Returns it unchanged."""
    if not math.isfinite(value):
        raise InvalidEmbeddingError("similarity is not finite: {0}".format(value))
    if not -1.0 <= value <= 1.0:
        raise InvalidEmbeddingError(
            "similarity {0} outside [-1, 1] -- indicates unnormalized input "
            "or a numerical fault".format(value)
        )
    return value
