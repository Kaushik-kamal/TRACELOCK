"""Quality metrics and aggregation.

Uses synthetic image patches (gradients, noise, flat fields) rather than faces.
These exercise the MEASUREMENT code -- whether a Laplacian variance falls when
an image is blurred, whether clipping is counted correctly. No face model
needed, and no biometric fixture committed to the repository.
"""

from __future__ import annotations

import numpy as np
import pytest

from tracelock.face.models import BoundingBox, Pose, QualityBand, QualityMetric
from tracelock.face.quality import (
    POSE_DEGRADATION_DEG,
    QUALITY_WEIGHTS,
    RECOGNITION_INPUT_PX,
    aggregate_quality,
    assess,
    crop_face,
    measure_detection_confidence,
    measure_exposure_integrity,
    measure_face_pixel_size,
    measure_frame_containment,
    measure_pose_frontality,
    measure_sharpness,
    observe_mean_luminance,
)

cv2 = pytest.importorskip("cv2")


def textured_patch(size: int = 200, seed: int = 42) -> np.ndarray:
    """Sharp, high-frequency, mid-exposure patch."""
    rng = np.random.default_rng(seed)
    return rng.integers(60, 200, size=(size, size, 3), dtype=np.uint8)


def blurred(patch: np.ndarray, kernel: int = 21) -> np.ndarray:
    return cv2.GaussianBlur(patch, (kernel, kernel), 0)


class TestWeights:
    def test_weights_sum_to_one(self):
        assert sum(QUALITY_WEIGHTS.values()) == pytest.approx(1.0)

    def test_pixel_size_carries_the_most_weight(self):
        assert QUALITY_WEIGHTS["face_pixel_size"] == max(QUALITY_WEIGHTS.values())


class TestSharpness:
    def test_blurring_lowers_the_score(self):
        sharp = measure_sharpness(textured_patch())
        soft = measure_sharpness(blurred(textured_patch()))
        assert soft.score < sharp.score
        assert soft.raw_value < sharp.raw_value

    def test_flat_field_scores_near_zero(self):
        flat = np.full((100, 100, 3), 128, dtype=np.uint8)
        assert measure_sharpness(flat).score < 0.01

    def test_empty_crop_is_zero(self):
        assert measure_sharpness(np.empty((0, 0), dtype=np.uint8)).score == 0.0

    def test_flagged_uncalibrated(self):
        # The normalization constant is a working default, and the output
        # must say so rather than implying a measured threshold.
        assert measure_sharpness(textured_patch()).uncalibrated is True

    def test_score_bounded(self):
        # A pathologically high-contrast patch must still clamp to 1.0.
        checker = np.indices((100, 100)).sum(axis=0) % 2 * 255
        patch = np.stack([checker] * 3, axis=-1).astype(np.uint8)
        assert 0.0 <= measure_sharpness(patch).score <= 1.0


class TestFacePixelSize:
    def test_large_face_scores_one(self):
        assert measure_face_pixel_size(BoundingBox(0, 0, 300, 300)).score == 1.0

    def test_at_model_input_size_scores_one(self):
        box = BoundingBox(0, 0, RECOGNITION_INPUT_PX, RECOGNITION_INPUT_PX)
        assert measure_face_pixel_size(box).score == pytest.approx(1.0)

    def test_tiny_face_scores_zero(self):
        assert measure_face_pixel_size(BoundingBox(0, 0, 20, 20)).score == 0.0

    def test_intermediate_is_between(self):
        assert 0.0 < measure_face_pixel_size(BoundingBox(0, 0, 70, 70)).score < 1.0

    def test_uses_the_shorter_side(self):
        # A wide, short box is limited by its height: 400px of width buys
        # nothing when there are only 40 rows of face.
        wide = measure_face_pixel_size(BoundingBox(0, 0, 400, 40))
        assert wide.raw_value == 40
        # 40px sits just above the 32px floor, so it scores low but non-zero.
        assert 0.0 < wide.score < 0.2
        # And it must match a square box of the same shorter side.
        assert wide.score == measure_face_pixel_size(BoundingBox(0, 0, 40, 40)).score

    def test_below_the_usable_floor_scores_zero(self):
        assert measure_face_pixel_size(BoundingBox(0, 0, 400, 32)).score == 0.0
        assert measure_face_pixel_size(BoundingBox(0, 0, 400, 10)).score == 0.0

    def test_score_increases_monotonically_with_size(self):
        sizes = [32, 50, 70, 90, 112, 200]
        scores = [
            measure_face_pixel_size(BoundingBox(0, 0, s, s)).score for s in sizes
        ]
        assert scores == sorted(scores)

    def test_is_calibrated_not_a_working_default(self):
        # Anchored to the model's real 112px input, so it is not flagged.
        assert measure_face_pixel_size(BoundingBox(0, 0, 200, 200)).uncalibrated is False


class TestDetectionConfidence:
    @pytest.mark.parametrize("value", [0.0, 0.25, 0.5, 0.9, 1.0])
    def test_passes_through(self, value):
        assert measure_detection_confidence(value).score == pytest.approx(value)

    def test_clamps_out_of_range(self):
        assert measure_detection_confidence(1.7).score == 1.0
        assert measure_detection_confidence(-0.4).score == 0.0


class TestPoseFrontality:
    def test_frontal_scores_one(self):
        assert measure_pose_frontality(Pose(0.0, 0.0, 0.0)).score == pytest.approx(1.0)

    def test_roll_is_ignored(self):
        # In-plane rotation is corrected by the 5-point alignment, so it must
        # not be penalised. This is the key design decision in this metric.
        frontal = measure_pose_frontality(Pose(0.0, 0.0, 0.0))
        rolled = measure_pose_frontality(Pose(0.0, 0.0, 45.0))
        assert rolled.score == pytest.approx(frontal.score)

    def test_yaw_lowers_the_score(self):
        assert measure_pose_frontality(Pose(0.0, 40.0, 0.0)).score < 0.5

    def test_pitch_lowers_the_score(self):
        assert measure_pose_frontality(Pose(40.0, 0.0, 0.0)).score < 0.5

    def test_extreme_pose_floors_at_zero(self):
        assert measure_pose_frontality(Pose(0.0, POSE_DEGRADATION_DEG + 30, 0.0)).score == 0.0

    def test_combined_axes_compound(self):
        single = measure_pose_frontality(Pose(0.0, 30.0, 0.0)).score
        both = measure_pose_frontality(Pose(30.0, 30.0, 0.0)).score
        assert both < single

    def test_missing_pose_scores_neutral_not_zero(self):
        # Absence of a pose model is not evidence of bad pose.
        metric = measure_pose_frontality(None)
        assert metric.score == 0.5
        assert metric.uncalibrated is True


class TestExposureIntegrity:
    def test_well_exposed_scores_high(self):
        assert measure_exposure_integrity(textured_patch()).score > 0.9

    def test_blown_out_scores_low(self):
        white = np.full((100, 100, 3), 255, dtype=np.uint8)
        assert measure_exposure_integrity(white).score < 0.1

    def test_crushed_black_scores_low(self):
        black = np.zeros((100, 100, 3), dtype=np.uint8)
        assert measure_exposure_integrity(black).score < 0.1

    def test_dark_but_unclipped_is_not_penalised(self):
        # THE FAIRNESS TEST. A uniformly dark-but-intact patch retains all its
        # information. Scoring it down is how mean-brightness metrics encode
        # skin-tone bias -- this metric must not do that.
        rng = np.random.default_rng(11)
        dark = rng.integers(20, 70, size=(100, 100, 3), dtype=np.uint8)
        assert measure_exposure_integrity(dark).score > 0.9

    def test_bright_but_unclipped_is_not_penalised(self):
        rng = np.random.default_rng(12)
        bright = rng.integers(190, 250, size=(100, 100, 3), dtype=np.uint8)
        assert measure_exposure_integrity(bright).score > 0.9

    def test_mean_luminance_is_observed_but_not_scored(self):
        rng = np.random.default_rng(13)
        dark = rng.integers(20, 70, size=(100, 100, 3), dtype=np.uint8)
        assert observe_mean_luminance(dark) < 80
        assert measure_exposure_integrity(dark).score > 0.9


class TestFrameContainment:
    def test_fully_inside_scores_one(self):
        assert measure_frame_containment(BoundingBox(100, 100, 200, 200), 500, 500).score == 1.0

    def test_half_outside_scores_half(self):
        metric = measure_frame_containment(BoundingBox(-50, 0, 50, 100), 500, 500)
        assert metric.score == pytest.approx(0.5, abs=0.01)

    def test_entirely_outside_scores_zero(self):
        assert measure_frame_containment(BoundingBox(600, 600, 700, 700), 500, 500).score == 0.0

    def test_corner_truncation(self):
        metric = measure_frame_containment(BoundingBox(-25, -25, 75, 75), 500, 500)
        assert metric.score == pytest.approx(0.5625, abs=0.01)


class TestAggregation:
    def _metrics(self, **overrides) -> tuple[QualityMetric, ...]:
        base = {name: 1.0 for name in QUALITY_WEIGHTS}
        base.update(overrides)
        return tuple(
            QualityMetric(name, score, score, "x") for name, score in base.items()
        )

    def test_all_perfect_scores_one(self):
        assert aggregate_quality(self._metrics()).aggregate == pytest.approx(1.0, abs=1e-6)

    def test_geometric_mean_is_dragged_down_by_one_bad_metric(self):
        # The whole reason for geometric over arithmetic: an unusable face
        # cannot be rescued by everything else being perfect.
        report = aggregate_quality(self._metrics(face_pixel_size=0.01))
        assert report.aggregate < 0.4

    def test_geometric_beats_arithmetic_at_catching_failure(self):
        metrics = self._metrics(face_pixel_size=0.01)
        geometric = aggregate_quality(metrics).aggregate
        arithmetic = sum(
            QUALITY_WEIGHTS[m.name] * m.score for m in metrics
        )
        assert geometric < arithmetic

    def test_result_bounded(self):
        rng = np.random.default_rng(14)
        for _ in range(30):
            scores = {name: float(rng.random()) for name in QUALITY_WEIGHTS}
            assert 0.0 <= aggregate_quality(self._metrics(**scores)).aggregate <= 1.0

    def test_zero_metric_does_not_raise(self):
        # log(0) would be -inf; the epsilon floor must handle it.
        report = aggregate_quality(self._metrics(sharpness=0.0))
        assert 0.0 <= report.aggregate <= 1.0

    def test_explain_covers_every_metric(self):
        report = aggregate_quality(self._metrics())
        assert len(report.explain()) == len(QUALITY_WEIGHTS)

    def test_get_finds_a_metric_by_name(self):
        report = aggregate_quality(self._metrics())
        assert report.get("sharpness") is not None
        assert report.get("nonexistent") is None


class TestQualityBand:
    @pytest.mark.parametrize(
        "score,band",
        [
            (0.95, QualityBand.EXCELLENT),
            (0.80, QualityBand.EXCELLENT),
            (0.70, QualityBand.GOOD),
            (0.60, QualityBand.GOOD),
            (0.45, QualityBand.MARGINAL),
            (0.35, QualityBand.MARGINAL),
            (0.20, QualityBand.POOR),
            (0.00, QualityBand.POOR),
        ],
    )
    def test_banding(self, score, band):
        assert QualityBand.from_score(score) is band


class TestCropFace:
    def test_crops_to_the_box(self):
        image = textured_patch(300)
        crop = crop_face(image, BoundingBox(50, 60, 150, 200))
        assert crop.shape[:2] == (140, 100)

    def test_clips_to_image_bounds(self):
        image = textured_patch(100)
        crop = crop_face(image, BoundingBox(-20, -20, 120, 120))
        assert crop.shape[:2] == (100, 100)

    def test_degenerate_box_returns_empty(self):
        assert crop_face(textured_patch(100), BoundingBox(500, 500, 600, 600)).size == 0


class TestAssessEndToEnd:
    def test_produces_a_full_report(self):
        image = textured_patch(400)
        bbox = BoundingBox(100, 100, 250, 250)
        report = assess(
            face_crop=crop_face(image, bbox),
            bbox=bbox,
            det_score=0.95,
            pose=Pose(1.0, 2.0, 3.0),
            image_width=400,
            image_height=400,
        )
        assert len(report.metrics) == len(QUALITY_WEIGHTS)
        assert 0.0 <= report.aggregate <= 1.0
        assert report.band in set(QualityBand)

    def test_all_metric_scores_within_range(self):
        image = textured_patch(400)
        bbox = BoundingBox(50, 50, 350, 350)
        report = assess(crop_face(image, bbox), bbox, 0.8, Pose(5, 5, 5), 400, 400)
        for metric in report.metrics:
            assert 0.0 <= metric.score <= 1.0, metric.name

    def test_bad_input_scores_worse_than_good(self):
        good_box = BoundingBox(50, 50, 350, 350)
        good_image = textured_patch(400)
        good = assess(crop_face(good_image, good_box), good_box, 0.95, Pose(0, 0, 0), 400, 400)

        bad_box = BoundingBox(0, 0, 40, 40)
        bad_image = blurred(textured_patch(400), 31)
        bad = assess(crop_face(bad_image, bad_box), bad_box, 0.35, Pose(45, 45, 0), 400, 400)

        assert bad.aggregate < good.aggregate
