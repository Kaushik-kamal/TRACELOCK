"""Face engine: models, and live inference against the real probe.

TEST FIXTURE POLICY
-------------------
No biometric data is committed to this repository. Tests that need a real face
use `data/probes/me.jpg`, which is gitignored and supplied by the operator
under the consent policy in the README. Those tests are marked `model` and SKIP
when the probe or the model pack is absent, so a fresh clone still runs green.

Everything that can be tested without a face -- dataclass invariants,
immutability, quantization, error handling on synthetic images -- runs always.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from tracelock.face.errors import (
    ImageNotFoundError,
    InvalidEmbeddingError,
    NoFaceDetectedError,
    UnsupportedImageError,
)
from tracelock.face.models import (
    SCHEMA_VERSION,
    BoundingBox,
    Embedding,
    Pose,
    QuantizationSpec,
)
from tracelock.face.similarity import cosine_similarity

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE = REPO_ROOT / "data" / "probes" / "me.jpg"

cv2 = pytest.importorskip("cv2")


def has_model() -> bool:
    try:
        import insightface  # noqa: F401
    except ImportError:
        return False
    return (Path.home() / ".insightface" / "models" / "buffalo_l").is_dir()


needs_model = pytest.mark.skipif(
    not has_model(), reason="buffalo_l model pack not available"
)
needs_probe = pytest.mark.skipif(
    not PROBE.is_file(), reason="data/probes/me.jpg not present (gitignored)"
)


@pytest.fixture(scope="module")
def engine():
    from tracelock.face import FaceEngine

    return FaceEngine()


@pytest.fixture(scope="module")
def result(engine):
    return engine.analyze(PROBE)


# ==========================================================================
# Model-free tests
# ==========================================================================


class TestBoundingBox:
    def test_derived_geometry(self):
        box = BoundingBox(10, 20, 110, 170)
        assert box.width == 100
        assert box.height == 150
        assert box.area == 15000
        assert box.center == (60, 95)
        assert box.min_side == 100

    def test_inverted_box_has_no_negative_size(self):
        box = BoundingBox(100, 100, 50, 50)
        assert box.width == 0 and box.height == 0 and box.area == 0


class TestPose:
    def test_frontal_deviation_excludes_roll(self):
        assert Pose(0, 0, 90).frontal_deviation_deg == pytest.approx(0.0)

    def test_deviation_combines_pitch_and_yaw(self):
        assert Pose(3, 4, 0).frontal_deviation_deg == pytest.approx(5.0)


class TestEmbedding:
    def _unit(self, dim=512, seed=1):
        vector = np.random.default_rng(seed).normal(size=dim).astype(np.float32)
        return vector / np.linalg.norm(vector)

    def test_reports_dimension_and_norm(self):
        embedding = Embedding(self._unit(), "test-model")
        assert embedding.dimension == 512
        assert embedding.l2_norm == pytest.approx(1.0, abs=1e-6)
        assert embedding.is_normalized

    def test_vector_is_read_only(self):
        embedding = Embedding(self._unit(), "test-model")
        with pytest.raises(ValueError):
            embedding.vector[0] = 99.0

    def test_construction_copies_the_input(self):
        source = self._unit()
        embedding = Embedding(source, "test-model")
        source[0] = 12345.0
        assert embedding.vector[0] != 12345.0

    def test_unnormalized_input_is_flagged_not_silently_fixed(self):
        embedding = Embedding(self._unit() * 5.0, "test-model")
        assert not embedding.is_normalized

    def test_rejects_nan(self):
        bad = self._unit(); bad[0] = np.nan
        with pytest.raises(InvalidEmbeddingError, match="NaN or Inf"):
            Embedding(bad, "test-model")

    def test_rejects_two_dimensional(self):
        with pytest.raises(InvalidEmbeddingError, match="1-D"):
            Embedding(np.ones((2, 512), dtype=np.float32), "test-model")


class TestQuantization:
    def _unit(self, seed=2):
        vector = np.random.default_rng(seed).normal(size=512).astype(np.float32)
        return vector / np.linalg.norm(vector)

    def test_is_deterministic(self):
        embedding = Embedding(self._unit(), "m")
        assert embedding.quantize() == embedding.quantize()

    def test_equal_embeddings_quantize_equally(self):
        vector = self._unit()
        assert Embedding(vector, "m").quantize() == Embedding(vector.copy(), "m").quantize()

    def test_int8_emits_one_byte_per_dimension(self):
        assert len(Embedding(self._unit(), "m").quantize()) == 512

    def test_survives_tiny_float_perturbation(self):
        # The point of quantizing: cross-machine float drift must not change
        # the committed bytes.
        vector = self._unit()
        nudged = vector + np.float32(1e-7)
        assert Embedding(vector, "m").quantize() == Embedding(nudged, "m").quantize()

    def test_different_embeddings_quantize_differently(self):
        assert Embedding(self._unit(2), "m").quantize() != Embedding(self._unit(3), "m").quantize()

    def test_round_decimals_method(self):
        embedding = Embedding(self._unit(), "m")
        spec = QuantizationSpec(version="dec-v1", method="round_decimals", decimals=4)
        assert len(embedding.quantize(spec)) == 512 * 4

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="unknown quantization method"):
            Embedding(self._unit(), "m").quantize(
                QuantizationSpec(version="x", method="nope")  # type: ignore[arg-type]
            )


class TestEmbeddingSerialization:
    def _embedding(self):
        vector = np.random.default_rng(4).normal(size=512).astype(np.float32)
        return Embedding(vector / np.linalg.norm(vector), "m")

    def test_vector_is_excluded_by_default(self):
        # Biometric data must not leak into a serialized structure by accident.
        payload = self._embedding().to_dict()
        assert "vector" not in payload
        assert payload["dimension"] == 512

    def test_vector_requires_explicit_opt_in(self):
        assert "vector" in self._embedding().to_dict(include_vector=True)


class TestImageErrors:
    @needs_model
    def test_missing_file_raises(self, engine, tmp_path):
        with pytest.raises(ImageNotFoundError):
            engine.analyze(tmp_path / "nope.jpg")

    @needs_model
    def test_non_image_raises(self, engine, tmp_path):
        path = tmp_path / "fake.jpg"
        path.write_text("this is not an image", encoding="utf-8")
        with pytest.raises(UnsupportedImageError, match="could not decode"):
            engine.analyze(path)

    @needs_model
    def test_empty_file_raises(self, engine, tmp_path):
        path = tmp_path / "empty.jpg"
        path.write_bytes(b"")
        with pytest.raises(UnsupportedImageError, match="empty"):
            engine.analyze(path)

    @needs_model
    def test_image_with_no_face_raises(self, engine, tmp_path):
        path = tmp_path / "noface.jpg"
        rng = np.random.default_rng(7)
        cv2.imwrite(
            str(path), rng.integers(0, 255, (400, 400, 3), dtype=np.uint8)
        )
        with pytest.raises(NoFaceDetectedError) as caught:
            engine.analyze(path)
        # The rejection carries the image identity so it can be recorded.
        assert caught.value.image_sha256


# ==========================================================================
# Live model tests
# ==========================================================================


@needs_model
@needs_probe
class TestLiveAnalysis:
    def test_probe_is_analyzed(self, result):
        assert result.schema_version == SCHEMA_VERSION
        assert result.faces_detected >= 1

    def test_exactly_one_primary_face(self, result):
        assert result.primary is not None
        assert len(result.others) == result.faces_detected - 1

    def test_embedding_dimension_is_512(self, result):
        assert result.embedding.dimension == 512

    def test_embedding_is_l2_normalized(self, result):
        assert result.embedding.l2_norm == pytest.approx(1.0, abs=1e-5)
        assert result.embedding.is_normalized

    def test_embedding_is_finite(self, result):
        assert np.isfinite(result.embedding.vector).all()

    def test_detection_confidence_is_a_probability(self, result):
        assert 0.0 <= result.primary.det_score <= 1.0

    def test_bbox_lies_within_the_image(self, result):
        box = result.primary.bbox
        assert box.x1 < box.x2 and box.y1 < box.y2
        assert box.x2 <= result.image.width + 1
        assert box.y2 <= result.image.height + 1

    def test_five_landmarks_present(self, result):
        assert result.primary.landmarks_5.shape == (5, 2)

    def test_landmarks_are_read_only(self, result):
        with pytest.raises(ValueError):
            result.primary.landmarks_5[0, 0] = 0.0

    def test_pose_is_populated(self, result):
        assert result.primary.pose is not None
        assert math.isfinite(result.primary.pose.yaw)

    def test_quality_metrics_within_range(self, result):
        for metric in result.primary.quality.metrics:
            assert 0.0 <= metric.score <= 1.0, metric.name
        assert 0.0 <= result.primary.quality.aggregate <= 1.0

    def test_image_sha256_matches_the_file(self, result):
        import hashlib

        assert result.image.sha256 == hashlib.sha256(PROBE.read_bytes()).hexdigest()

    def test_selection_report_is_populated(self, result):
        assert result.selection.policy
        assert result.selection.chosen_index == result.primary.index
        assert sum(result.selection.weights.values()) == pytest.approx(1.0)

    def test_serializes_without_the_embedding(self, result):
        import json

        payload = result.to_dict()
        assert json.dumps(payload)
        assert "vector" not in payload["primary"]["embedding"]


@needs_model
@needs_probe
class TestProvenance:
    def test_records_the_recognition_model_filename(self, engine):
        assert engine.provenance.recognition_model.endswith(".onnx")

    def test_hashes_every_model_file(self, engine):
        hashes = engine.provenance.model_file_sha256
        assert hashes
        for name, digest in hashes.items():
            assert name.endswith(".onnx")
            assert len(digest) == 64

    def test_records_package_versions(self, engine):
        versions = engine.provenance.package_versions
        for key in ("python", "insightface", "onnxruntime", "numpy"):
            assert key in versions

    def test_model_id_is_stable(self, engine):
        assert engine.model_id == engine.provenance.model_id
        assert engine.provenance.pack_name in engine.model_id


@needs_model
@needs_probe
class TestDeterminism:
    def test_repeat_analysis_is_bit_identical(self, engine):
        report = engine.verify_determinism(PROBE)
        assert report["bit_identical"] is True
        assert report["max_abs_diff"] == 0.0
        assert report["quantized_identical"] is True

    def test_self_similarity_is_one(self, engine, result):
        second = engine.analyze(PROBE)
        assert cosine_similarity(result.embedding, second.embedding) == pytest.approx(
            1.0, abs=1e-6
        )

    def test_image_hash_is_stable(self, engine, result):
        assert engine.analyze(PROBE).image.sha256 == result.image.sha256


@needs_model
@needs_probe
class TestRobustness:
    """Behaviour under transformations, which is where a naive engine breaks."""

    def test_recompression_barely_moves_the_embedding(self, engine, tmp_path):
        image = cv2.imread(str(PROBE))
        path = tmp_path / "recompressed.jpg"
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 70])

        similarity = cosine_similarity(
            engine.analyze(PROBE).embedding, engine.analyze(path).embedding
        )
        # Same pixels, different bytes: identity must survive, and the image
        # hash must NOT -- that distinction is the whole basis of Phase 2's
        # "same photo republished" vs "different photo, same person" split.
        assert similarity > 0.9

    def test_recompression_changes_the_image_hash(self, engine, tmp_path):
        image = cv2.imread(str(PROBE))
        path = tmp_path / "recompressed.jpg"
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 70])
        assert engine.analyze(path).image.sha256 != engine.analyze(PROBE).image.sha256

    def test_downscaling_lowers_the_quality_score(self, engine, tmp_path):
        image = cv2.imread(str(PROBE))
        small = cv2.resize(image, (image.shape[1] // 6, image.shape[0] // 6))
        path = tmp_path / "small.jpg"
        cv2.imwrite(str(path), small)

        try:
            degraded = engine.analyze(path)
        except NoFaceDetectedError:
            pytest.skip("face no longer detectable at this scale, which is also correct")

        assert degraded.primary.quality.aggregate < engine.analyze(PROBE).primary.quality.aggregate

    def test_heavy_blur_lowers_sharpness(self, engine, tmp_path):
        image = cv2.imread(str(PROBE))
        path = tmp_path / "blurred.jpg"
        cv2.imwrite(str(path), cv2.GaussianBlur(image, (15, 15), 0))

        try:
            blurred = engine.analyze(path)
        except NoFaceDetectedError:
            pytest.skip("face no longer detectable when blurred, which is also correct")

        assert (
            blurred.primary.quality.get("sharpness").raw_value
            < engine.analyze(PROBE).primary.quality.get("sharpness").raw_value
        )
