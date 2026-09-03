"""Similarity utilities.

Includes a test asserting that NO identity threshold constant exists anywhere
in the face package. That is a real invariant of this project, not a style
preference: a hardcoded `sim > 0.5` would be inventing empirical evidence.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from tracelock.face.errors import (
    EmbeddingDimensionMismatch,
    InvalidEmbeddingError,
    ZeroNormEmbeddingError,
)
from tracelock.face.models import Embedding
from tracelock.face.similarity import (
    angular_distance,
    cosine_similarity,
    similarity_matrix,
    validate_similarity,
)


def unit(values) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        vector = unit(np.arange(1, 513))
        assert cosine_similarity(vector, vector) == pytest.approx(1.0, abs=1e-6)

    def test_opposite_vectors_score_minus_one(self):
        vector = unit(np.arange(1, 513))
        assert cosine_similarity(vector, -vector) == pytest.approx(-1.0, abs=1e-6)

    def test_orthogonal_vectors_score_zero(self):
        a = np.zeros(512, dtype=np.float32); a[0] = 1.0
        b = np.zeros(512, dtype=np.float32); b[1] = 1.0
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_accepts_embedding_objects(self):
        vector = unit(np.random.default_rng(1).normal(size=512))
        embedding = Embedding(vector=vector, model_id="test")
        assert cosine_similarity(embedding, embedding) == pytest.approx(1.0, abs=1e-6)

    def test_mixes_embedding_and_array(self):
        vector = unit(np.random.default_rng(2).normal(size=512))
        embedding = Embedding(vector=vector, model_id="test")
        assert cosine_similarity(embedding, vector) == pytest.approx(1.0, abs=1e-6)

    def test_is_symmetric(self):
        rng = np.random.default_rng(3)
        a, b = unit(rng.normal(size=512)), unit(rng.normal(size=512))
        assert cosine_similarity(a, b) == pytest.approx(cosine_similarity(b, a))

    def test_magnitude_does_not_matter(self):
        rng = np.random.default_rng(4)
        a, b = rng.normal(size=512), rng.normal(size=512)
        assert cosine_similarity(a, b) == pytest.approx(
            cosine_similarity(a * 1000.0, b * 0.001), abs=1e-9
        )

    def test_output_always_within_range(self):
        rng = np.random.default_rng(5)
        for _ in range(50):
            value = cosine_similarity(rng.normal(size=128), rng.normal(size=128))
            assert -1.0 <= value <= 1.0

    def test_identical_vectors_never_exceed_one(self):
        # Float error can push a unit dot product a few ULPs past 1.0, which
        # would make arccos() raise. The clamp must hold.
        for dim in (2, 64, 512, 4096):
            vector = unit(np.random.default_rng(dim).normal(size=dim))
            assert cosine_similarity(vector, vector) <= 1.0


class TestErrorHandling:
    def test_dimension_mismatch_raises(self):
        with pytest.raises(EmbeddingDimensionMismatch) as caught:
            cosine_similarity(np.ones(512), np.ones(128))
        assert caught.value.left == 512
        assert caught.value.right == 128

    def test_mismatch_message_warns_about_model_packs(self):
        with pytest.raises(EmbeddingDimensionMismatch, match="different model packs"):
            cosine_similarity(np.ones(512), np.ones(256))

    def test_zero_vector_raises_rather_than_returning_zero(self):
        # 0.0 is a legitimate similarity (orthogonal). Returning it for an
        # undefined case would be a silent lie.
        with pytest.raises(ZeroNormEmbeddingError):
            cosine_similarity(np.zeros(512), unit(np.ones(512)))

    def test_zero_on_the_right_also_raises(self):
        with pytest.raises(ZeroNormEmbeddingError):
            cosine_similarity(unit(np.ones(512)), np.zeros(512))

    def test_both_zero_raises(self):
        with pytest.raises(ZeroNormEmbeddingError):
            cosine_similarity(np.zeros(512), np.zeros(512))

    def test_nan_raises(self):
        bad = np.ones(512); bad[0] = np.nan
        with pytest.raises(InvalidEmbeddingError, match="NaN or Inf"):
            cosine_similarity(bad, np.ones(512))

    def test_inf_raises(self):
        bad = np.ones(512); bad[5] = np.inf
        with pytest.raises(InvalidEmbeddingError, match="NaN or Inf"):
            cosine_similarity(bad, np.ones(512))

    def test_two_dimensional_input_raises(self):
        with pytest.raises(InvalidEmbeddingError, match="1-D"):
            cosine_similarity(np.ones((2, 512)), np.ones(512))

    def test_empty_input_raises(self):
        with pytest.raises(InvalidEmbeddingError, match="empty"):
            cosine_similarity(np.array([]), np.array([]))


class TestAngularDistance:
    def test_identical_is_zero(self):
        vector = unit(np.arange(1, 513))
        assert angular_distance(vector, vector) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_is_one(self):
        vector = unit(np.arange(1, 513))
        assert angular_distance(vector, -vector) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_is_half(self):
        a = np.zeros(64); a[0] = 1.0
        b = np.zeros(64); b[1] = 1.0
        assert angular_distance(a, b) == pytest.approx(0.5, abs=1e-6)

    def test_monotonically_decreasing_in_similarity(self):
        rng = np.random.default_rng(7)
        anchor = unit(rng.normal(size=256))
        pairs = []
        for _ in range(20):
            other = unit(rng.normal(size=256))
            pairs.append((cosine_similarity(anchor, other), angular_distance(anchor, other)))
        pairs.sort()
        distances = [d for _, d in pairs]
        assert distances == sorted(distances, reverse=True)


class TestSimilarityMatrix:
    def test_shape(self):
        rng = np.random.default_rng(8)
        left = [unit(rng.normal(size=128)) for _ in range(3)]
        right = [unit(rng.normal(size=128)) for _ in range(5)]
        assert similarity_matrix(left, right).shape == (3, 5)

    def test_agrees_with_pairwise(self):
        rng = np.random.default_rng(9)
        left = [unit(rng.normal(size=128)) for _ in range(3)]
        right = [unit(rng.normal(size=128)) for _ in range(4)]
        matrix = similarity_matrix(left, right)
        for i, a in enumerate(left):
            for j, b in enumerate(right):
                assert matrix[i, j] == pytest.approx(cosine_similarity(a, b), abs=1e-9)

    def test_self_matrix_has_unit_diagonal(self):
        rng = np.random.default_rng(10)
        vectors = [unit(rng.normal(size=64)) for _ in range(4)]
        matrix = similarity_matrix(vectors, vectors)
        assert np.allclose(np.diag(matrix), 1.0, atol=1e-6)

    def test_empty_input(self):
        assert similarity_matrix([], []).shape == (0, 0)

    def test_dimension_mismatch_raises(self):
        with pytest.raises(EmbeddingDimensionMismatch):
            similarity_matrix([np.ones(128)], [np.ones(512)])

    def test_ragged_left_raises(self):
        with pytest.raises(EmbeddingDimensionMismatch):
            similarity_matrix([np.ones(128), np.ones(64)], [np.ones(128)])


class TestValidateSimilarity:
    @pytest.mark.parametrize("value", [-1.0, -0.5, 0.0, 0.5, 1.0])
    def test_accepts_valid(self, value):
        assert validate_similarity(value) == value

    @pytest.mark.parametrize("value", [1.5, -1.5, 100.0])
    def test_rejects_out_of_range(self, value):
        with pytest.raises(InvalidEmbeddingError, match="outside"):
            validate_similarity(value)

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_rejects_non_finite(self, value):
        with pytest.raises(InvalidEmbeddingError, match="not finite"):
            validate_similarity(value)


class TestNoIdentityThresholds:
    """A project invariant, enforced by test.

    Phase 1 measures. It does not decide identity. A magic constant compared
    against a similarity would be fabricating a calibrated claim.
    """

    def test_face_package_declares_no_identity_threshold(self):
        face_dir = Path(__file__).resolve().parents[1] / "src" / "tracelock" / "face"
        banned = ("IDENTITY_THRESHOLD", "SAME_PERSON", "MATCH_THRESHOLD", "is_same_person")

        offenders = []
        for source in face_dir.glob("*.py"):
            text = source.read_text(encoding="utf-8")
            for token in banned:
                if token in text:
                    offenders.append("{0}: {1}".format(source.name, token))

        assert not offenders, (
            "identity-decision constants found in the face package: {0}. "
            "Similarity is not identity -- that mapping belongs in "
            "tracelock.calibration, fitted to labelled data.".format(offenders)
        )

    def test_similarity_module_returns_no_booleans(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "src" / "tracelock" / "face" / "similarity.py"
        ).read_text(encoding="utf-8")
        assert "-> bool" not in source, (
            "similarity.py must not return verdicts, only measurements"
        )
