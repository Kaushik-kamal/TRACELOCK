"""Phase 1 live validation.

Runs the Face Intelligence Engine against a real image and prints a compact
diagnostic report.

The 512-dimensional embedding is NEVER printed. It is biometric data; the
report shows its dimension, L2 norm, and a truncated digest of the quantized
form so runs can be compared without exposing the vector.

    python scripts\\face_engine_report.py --image data\\probes\\me.jpg
    python scripts\\face_engine_report.py --image data\\probes\\me.jpg --compare other.jpg
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tracelock.face import (  # noqa: E402
    FaceEngine,
    FaceEngineError,
    NoFaceDetectedError,
    cosine_similarity,
)

RULE = "=" * 72
THIN = "-" * 72


def emit(text: str = "") -> None:
    print(text, flush=True)


def header(text: str) -> None:
    emit()
    emit(RULE)
    emit(text)
    emit(RULE)


def section(text: str) -> None:
    emit()
    emit(text)
    emit(THIN)


def report(result, *, engine: FaceEngine) -> None:
    section("IMAGE")
    emit("  path            : {0}".format(result.image.path))
    emit("  sha256          : {0}".format(result.image.sha256))
    emit(
        "  dimensions      : {0}x{1}x{2}".format(
            result.image.width, result.image.height, result.image.channels
        )
    )

    section("MODEL PROVENANCE")
    model = result.model
    emit("  pack            : {0}".format(model.pack_name))
    emit("  detection       : {0}".format(model.detection_model))
    emit("  recognition     : {0}".format(model.recognition_model))
    emit("  det_size        : {0}".format(list(model.det_size)))
    emit("  providers       : {0}".format(", ".join(model.providers)))
    emit("  versions        : {0}".format(
        "  ".join("{0}={1}".format(k, v) for k, v in model.package_versions.items())
    ))
    emit("  model files     :")
    for name, digest in sorted(model.model_file_sha256.items()):
        emit("      {0:<18} sha256:{1}...".format(name, digest[:24]))

    section("DETECTION")
    emit("  faces detected  : {0}".format(result.faces_detected))
    primary = result.primary
    box = primary.bbox
    emit("  primary index   : {0}".format(primary.index))
    emit(
        "  bounding box    : x1={0:.1f} y1={1:.1f} x2={2:.1f} y2={3:.1f}  "
        "({4:.0f}x{5:.0f} px)".format(
            box.x1, box.y1, box.x2, box.y2, box.width, box.height
        )
    )
    emit("  det confidence  : {0:.4f}".format(primary.det_score))
    if primary.pose:
        pose = primary.pose
        emit(
            "  pose (deg)      : pitch={0:+.2f}  yaw={1:+.2f}  roll={2:+.2f}".format(
                pose.pitch, pose.yaw, pose.roll
            )
        )
        emit(
            "  out-of-plane    : {0:.2f} deg  (roll excluded -- alignment "
            "corrects it)".format(pose.frontal_deviation_deg)
        )
    emit("  landmarks       : {0} points".format(len(primary.landmarks_5)))
    emit(
        "  attribute est.  : age~{0}  sex~{1}   (model estimates, not facts)".format(
            primary.age_estimate, primary.sex_estimate
        )
    )

    if result.others:
        emit()
        emit("  other faces (retained, not discarded):")
        for other in result.others:
            emit(
                "      #{0}  det={1:.3f}  sel_score={2:.4f}  "
                "{3:.0f}x{4:.0f}px".format(
                    other.index,
                    other.det_score,
                    other.selection_score,
                    other.bbox.width,
                    other.bbox.height,
                )
            )

    section("PRIMARY FACE SELECTION")
    selection = result.selection
    emit("  policy          : {0}".format(selection.policy))
    emit("  weights         : {0}".format(
        "  ".join("{0}={1:.2f}".format(k, v) for k, v in selection.weights.items())
    ))
    emit("  chosen index    : {0}".format(selection.chosen_index))
    emit(
        "  margin over #2  : {0:.4f}   (threshold {1:.2f})".format(
            selection.margin, selection.ambiguity_threshold
        )
    )
    emit("  ambiguous       : {0}".format("YES" if selection.ambiguous else "no"))

    section("EMBEDDING")
    embedding = result.embedding
    emit("  model id        : {0}".format(embedding.model_id))
    emit("  dimension       : {0}".format(embedding.dimension))
    emit("  L2 norm         : {0:.10f}".format(embedding.l2_norm))
    emit("  normalized      : {0}".format("yes" if embedding.is_normalized else "NO"))
    # The vector itself is deliberately not printed -- biometric data.
    quantized = embedding.quantize()
    emit(
        "  quantized digest: sha256:{0}...  ({1} bytes, int8-v1)".format(
            hashlib.sha256(quantized).hexdigest()[:24], len(quantized)
        )
    )
    emit("  NOTE            : the 512-d vector is withheld from this report by design")

    section("QUALITY")
    quality = primary.quality
    for line in quality.explain():
        emit("  " + line)
    emit()
    emit(
        "  aggregate       : {0:.4f}  [{1}]   (weighted geometric mean)".format(
            quality.aggregate, quality.band.value
        )
    )

    section("WARNINGS")
    if result.warnings:
        for warning in result.warnings:
            emit("  [{0}]".format(warning.code.value))
            emit("      {0}".format(warning.message))
    else:
        emit("  none")

    section("TIMING")
    emit("  analysis        : {0:.3f}s".format(result.elapsed_seconds))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="face_engine_report",
        description="Phase 1 live validation of the Face Intelligence Engine.",
    )
    parser.add_argument("--image", type=Path, default=Path("data/probes/me.jpg"))
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="Second image; prints cosine similarity. NO identity verdict is "
        "given -- that requires calibration (Phase 3).",
    )
    parser.add_argument(
        "--verify-determinism",
        action="store_true",
        help="Analyze twice and confirm the embeddings are bit-identical.",
    )
    parser.add_argument("--json", type=Path, default=None, help="Write the result as JSON.")
    args = parser.parse_args(argv)

    header("TRACELOCK - PHASE 1 - FACE INTELLIGENCE ENGINE")

    try:
        emit("Loading buffalo_l ...")
        engine = FaceEngine()
        emit("Model ready: {0}".format(engine.model_id))
    except FaceEngineError as exc:
        emit("MODEL INITIALIZATION FAILED: {0}".format(exc))
        return 2

    try:
        result = engine.analyze(args.image)
    except NoFaceDetectedError as exc:
        emit()
        emit("NO FACE DETECTED: {0}".format(exc))
        emit("For a probe image this is a user error. For a candidate it would be")
        emit("a legitimate rejection reason.")
        return 1
    except FaceEngineError as exc:
        emit()
        emit("ANALYSIS FAILED: {0}".format(exc))
        return 1

    report(result, engine=engine)

    if args.verify_determinism:
        section("DETERMINISM CHECK")
        check = engine.verify_determinism(args.image)
        emit("  bit identical   : {0}".format(check["bit_identical"]))
        emit("  max abs diff    : {0:.3e}".format(check["max_abs_diff"]))
        emit("  quantized equal : {0}".format(check["quantized_identical"]))
        emit("  det score delta : {0:.3e}".format(check["det_score_delta"]))

    if args.compare:
        section("COMPARISON")
        try:
            other = engine.analyze(args.compare)
        except FaceEngineError as exc:
            emit("  could not analyze {0}: {1}".format(args.compare, exc))
            return 1

        similarity = cosine_similarity(result.embedding, other.embedding)
        emit("  image A         : {0}".format(result.image.path))
        emit("  image B         : {0}".format(other.image.path))
        emit("  cosine similarity: {0:.6f}".format(similarity))
        emit()
        emit("  NO IDENTITY VERDICT IS GIVEN.")
        emit("  A similarity is a geometric quantity. Converting it into an")
        emit("  identity probability requires a calibrated mapping fitted to")
        emit("  labelled pairs -- see tracelock.calibration. Phase 1 measures;")
        emit("  Phase 3 decides.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(result.to_dict(), indent=2), encoding="utf-8"
        )
        section("ARTIFACT")
        emit("  {0}".format(args.json))
        emit("  (embedding vector excluded by design)")

    header("PHASE 1 VALIDATION COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
