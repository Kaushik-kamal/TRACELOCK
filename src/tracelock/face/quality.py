"""Face quality metrics.

Quality is not cosmetic here. In Phase 3 it feeds the evidence-quality term Q
of the trust score, and in Phase 2 it can justify rejecting a candidate before
identity is even considered. So every metric below states three things:

    WHAT it measures, WHY it matters, and HOW it fails.

A metric whose failure mode you cannot state is a metric you cannot defend.

AGGREGATION: weighted geometric mean. See QualityReport for the argument --
briefly, a 20px face is unusable no matter how sharp it is, and an arithmetic
mean would let sharpness paper over that.

WHAT IS AND IS NOT CALIBRATED
-----------------------------
Two metrics have PRINCIPLED normalization constants:
  - face_pixel_size  -> anchored to 112px, the recognition model's actual input
  - pose_frontality  -> anchored to 60 degrees of out-of-plane deviation
Three are WORKING DEFAULTS, flagged `uncalibrated=True` in the output:
  - sharpness, exposure_integrity, frame_containment
They are honest measurements with un-tuned scaling. Phase 3 calibration can
fit them; until then the report says so rather than implying rigour.
"""

from __future__ import annotations

import numpy as np

from tracelock.face.models import (
    BoundingBox,
    Pose,
    QualityBand,
    QualityMetric,
    QualityReport,
)

# --------------------------------------------------------------------------
# Aggregation weights. Sum to 1.0.
# --------------------------------------------------------------------------
QUALITY_WEIGHTS: dict[str, float] = {
    "face_pixel_size": 0.30,      # no pixels, no signal -- most fundamental
    "sharpness": 0.25,            # blur destroys the texture ArcFace reads
    "detection_confidence": 0.20, # the detector's own doubt
    "pose_frontality": 0.15,      # out-of-plane rotation degrades matching
    "exposure_integrity": 0.05,   # clipped pixels are unrecoverable
    "frame_containment": 0.05,    # a truncated face is a partial face
}

# ArcFace (w600k_r50) consumes 112x112 aligned crops. A face already smaller
# than that is being upscaled, inventing no new detail. Principled anchor.
RECOGNITION_INPUT_PX = 112
MIN_USABLE_FACE_PX = 32

# Beyond ~60 degrees out-of-plane, ArcFace degrades sharply. Principled anchor.
POSE_DEGRADATION_DEG = 60.0

# Laplacian variance is unbounded and scales with resolution and content, so
# this is a scaling constant rather than a threshold.
#
# ANCHORED TO A MEASUREMENT, n=1 (see docs/PHASE1_FACE_ENGINE.md):
# A progressive Gaussian-blur sweep on the live probe gave:
#     kernel  1 -> lap_var 104.2, cos-to-original 0.9998
#     kernel  9 -> lap_var  13.2, cos 0.9911
#     kernel 21 -> lap_var   4.4, cos 0.9472
#     kernel 35 -> lap_var   2.8, cos 0.8200
# So a clearly usable face crop sits near 100, and self-similarity only breaks
# down below ~4. An earlier guess of 500.0 scored that same usable face at
# 0.21 and raised a false BLURRY_FACE warning.
#
# STILL FLAGGED UNCALIBRATED, deliberately. One subject, one image, synthetic
# Gaussian blur. And critically, a self-similarity sweep CANNOT detect the
# failure mode this metric exists to catch: blur pulling embeddings toward the
# population mean and INFLATING impostor similarity. Measuring that needs
# labelled impostor pairs, which Phase 1 does not have.
SHARPNESS_REFERENCE = 100.0

# Numerical floor so a single zero metric does not make log() undefined.
# A floor, not a policy: a genuinely unusable face still lands near zero.
_EPS = 1e-6

# Warning thresholds (working defaults).
WARN_SMALL_FACE_PX = 60
WARN_LOW_DET_SCORE = 0.60
WARN_BLUR_SCORE = 0.30
WARN_POSE_DEG = 35.0
WARN_CLIPPING_RATIO = 0.10
WARN_LOW_AGGREGATE = 0.35


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


# ==========================================================================
# Individual metrics
# ==========================================================================


def measure_sharpness(face_crop: np.ndarray) -> QualityMetric:
    """Variance of the Laplacian over the face crop.

    WHAT   High-frequency energy. A sharp face has strong edges; blur removes
           them and the variance collapses.
    WHY    ArcFace reads fine texture. A blurred crop yields an embedding that
           sits closer to the population mean, inflating similarity to
           everyone -- blur causes FALSE MATCHES, not just weak ones.
    FAILS  (a) Scales with resolution and JPEG quality: a sharp but heavily
           compressed face scores low. (b) Busy background inside the bbox
           inflates it, which is why we crop to the box and not the frame.
           (c) The reference constant is uncalibrated.
    """
    if face_crop.size == 0:
        return QualityMetric("sharpness", 0.0, 0.0, "lap_var", uncalibrated=True)

    grey = _to_grey(face_crop)
    # cv2 is imported lazily so this module is importable without opencv.
    import cv2

    variance = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    return QualityMetric(
        name="sharpness",
        raw_value=variance,
        score=_clamp01(variance / SHARPNESS_REFERENCE),
        unit="lap_var",
        uncalibrated=True,
    )


def measure_face_pixel_size(bbox: BoundingBox) -> QualityMetric:
    """Shorter side of the face box, in pixels.

    WHAT   How much real detail the face occupies.
    WHY    The recognition model resizes to 112x112. Below that it is
           upscaling; below ~32px there is essentially nothing to upscale.
    FAILS  Says nothing about whether those pixels are sharp or well exposed --
           a large blurry face still scores 1.0 here. Only meaningful combined
           with sharpness, which is why neither dominates the aggregate.
    """
    min_side = bbox.min_side
    if min_side <= MIN_USABLE_FACE_PX:
        score = 0.0
    elif min_side >= RECOGNITION_INPUT_PX:
        score = 1.0
    else:
        score = (min_side - MIN_USABLE_FACE_PX) / (
            RECOGNITION_INPUT_PX - MIN_USABLE_FACE_PX
        )
    return QualityMetric("face_pixel_size", min_side, _clamp01(score), "px")


def measure_detection_confidence(det_score: float) -> QualityMetric:
    """The detector's own confidence, passed through unmodified.

    WHAT   SCRFD's score for this box.
    WHY    Low confidence usually means occlusion, extreme pose, motion blur,
           or a false positive. It is the detector telling you it is unsure.
    FAILS  It is a DETECTION confidence, not a recognition-quality measure.
           SCRFD occasionally reports high confidence on face-like patterns
           (posters, statues, printed photos) that are not live subjects.
    """
    return QualityMetric("detection_confidence", det_score, _clamp01(det_score), "prob")


def measure_pose_frontality(pose: Pose | None) -> QualityMetric:
    """How close to frontal the head is. ROLL IS EXCLUDED.

    WHAT   Out-of-plane deviation, sqrt(pitch^2 + yaw^2), in degrees.
    WHY    ArcFace is trained mostly on near-frontal faces and degrades as
           yaw grows. Roll is deliberately ignored: in-plane rotation is
           removed by the 5-point similarity alignment before the recognition
           model sees the crop, so penalising it would double-count a solved
           problem. (Confirmed empirically: rotating the input image moved
           roll one-for-one while yaw and pitch stayed flat.)
    FAILS  The pose estimate comes from the 3D landmark model, which is itself
           least reliable at extreme pose -- so this metric degrades exactly
           where it matters most. Treat large values as "probably bad" rather
           than as a precise angle.
    """
    if pose is None:
        # No pose model: score neutrally rather than punishing an absence.
        return QualityMetric("pose_frontality", float("nan"), 0.5, "deg", uncalibrated=True)

    deviation = pose.frontal_deviation_deg
    score = _clamp01(1.0 - deviation / POSE_DEGRADATION_DEG)
    return QualityMetric("pose_frontality", deviation, score, "deg")


def measure_exposure_integrity(face_crop: np.ndarray) -> QualityMetric:
    """Fraction of pixels crushed to black or blown to white.

    WHAT   Proportion of clipped samples (<=2 or >=253) in the face crop.

    WHY    Clipped pixels carry NO recoverable information. This is genuine
           data loss, unlike merely dark or bright exposure.

    FAIRNESS -- the reason this metric is clipping and not mean brightness:
           Thresholding on mean luminance systematically penalises darker skin
           tones and produces exactly the kind of demographic bias face systems
           are rightly criticised for. Clipping is tone-neutral: a saturated
           pixel is destroyed information regardless of the subject. Mean
           luminance IS still reported below as a diagnostic observation, but
           it is deliberately NOT scored.

    FAILS  Small specular highlights (glasses, jewellery) clip harmlessly and
           are counted the same as a blown-out face. The scaling is a working
           default.
    """
    if face_crop.size == 0:
        return QualityMetric("exposure_integrity", 1.0, 0.0, "ratio", uncalibrated=True)

    grey = _to_grey(face_crop)
    total = grey.size
    clipped = int(np.count_nonzero((grey <= 2) | (grey >= 253)))
    ratio = clipped / total if total else 1.0

    # Full marks up to 2% clipping; zero at 40%. Working default.
    score = _clamp01(1.0 - max(0.0, ratio - 0.02) / 0.38)
    return QualityMetric("exposure_integrity", ratio, score, "ratio", uncalibrated=True)


def measure_frame_containment(
    bbox: BoundingBox, image_width: int, image_height: int
) -> QualityMetric:
    """Fraction of the face box that actually lies inside the image.

    WHAT   intersection(bbox, image) / area(bbox).
    WHY    Detectors extrapolate boxes past the frame edge. A face cut off by
           the frame is missing landmarks the aligner needs, so the crop that
           reaches ArcFace is geometrically wrong, not merely incomplete.
    FAILS  Only detects TRUNCATION BY THE FRAME. A face half-hidden behind a
           hand, a mask, or another person is fully contained and scores 1.0.
           Occlusion detection is a separate problem this does not solve.
    """
    if image_width <= 0 or image_height <= 0 or bbox.area <= 0:
        return QualityMetric("frame_containment", 0.0, 0.0, "ratio", uncalibrated=True)

    inner_w = max(0.0, min(bbox.x2, image_width) - max(bbox.x1, 0.0))
    inner_h = max(0.0, min(bbox.y2, image_height) - max(bbox.y1, 0.0))
    ratio = (inner_w * inner_h) / bbox.area

    return QualityMetric(
        "frame_containment", ratio, _clamp01(ratio), "ratio", uncalibrated=True
    )


def observe_mean_luminance(face_crop: np.ndarray) -> float:
    """Diagnostic only -- deliberately NOT scored. See measure_exposure_integrity."""
    if face_crop.size == 0:
        return float("nan")
    return float(np.mean(_to_grey(face_crop)))


# ==========================================================================
# Aggregation
# ==========================================================================


def aggregate_quality(metrics: tuple[QualityMetric, ...]) -> QualityReport:
    """Weighted geometric mean over the normalized metric scores.

        Q = exp( sum_i w_i * ln(max(score_i, eps)) )

    Geometric, not arithmetic, so that a near-zero on any single dimension
    drags the aggregate down instead of being averaged away by strong terms.
    """
    weights = {m.name: QUALITY_WEIGHTS.get(m.name, 0.0) for m in metrics}
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return QualityReport(metrics, weights, 0.0, QualityBand.POOR)

    log_sum = sum(
        weights[m.name] * np.log(max(m.score, _EPS))
        for m in metrics
        if weights[m.name] > 0
    )
    aggregate = float(np.exp(log_sum / total_weight))
    aggregate = _clamp01(aggregate)

    return QualityReport(
        metrics=metrics,
        weights=weights,
        aggregate=aggregate,
        band=QualityBand.from_score(aggregate),
    )


def assess(
    face_crop: np.ndarray,
    bbox: BoundingBox,
    det_score: float,
    pose: Pose | None,
    image_width: int,
    image_height: int,
) -> QualityReport:
    """Run every metric and aggregate. The module's single entry point."""
    metrics = (
        measure_face_pixel_size(bbox),
        measure_sharpness(face_crop),
        measure_detection_confidence(det_score),
        measure_pose_frontality(pose),
        measure_exposure_integrity(face_crop),
        measure_frame_containment(bbox, image_width, image_height),
    )
    return aggregate_quality(metrics)


# ==========================================================================
# Helpers
# ==========================================================================


def _to_grey(image: np.ndarray) -> np.ndarray:
    """Greyscale view of a BGR or already-grey array."""
    if image.ndim == 2:
        return image
    import cv2

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def crop_face(image: np.ndarray, bbox: BoundingBox, *, margin: float = 0.0) -> np.ndarray:
    """Crop the face box, clipped to the image. Empty array if degenerate.

    Margin 0.0 by default: quality is measured on the FACE, not its
    surroundings. Background texture would inflate the sharpness metric.
    """
    height, width = image.shape[:2]

    pad_x = bbox.width * margin
    pad_y = bbox.height * margin

    x1 = max(0, int(round(bbox.x1 - pad_x)))
    y1 = max(0, int(round(bbox.y1 - pad_y)))
    x2 = min(width, int(round(bbox.x2 + pad_x)))
    y2 = min(height, int(round(bbox.y2 + pad_y)))

    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0), dtype=image.dtype)

    return image[y1:y2, x1:x2]
