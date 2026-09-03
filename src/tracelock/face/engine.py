"""The Face Intelligence Engine.

Loads buffalo_l once, then turns an image path into a FaceAnalysisResult.

WHAT THIS ENGINE DOES NOT DO
----------------------------
It never decides that two faces are the same person. It produces measurements.
Phase 3 turns measurements into calibrated probabilities. There is no identity
threshold anywhere in this file.

REPRODUCIBILITY
---------------
Every result carries a ModelProvenance recording the SHA-256 of each .onnx
file, the ONNX Runtime version and selected providers, the detector input size,
and package versions -- enough to answer "why did this image produce this
embedding" without guessing. A pack NAME is not a version; the bytes are.

Determinism was measured on this build rather than assumed: embeddings are
bit-identical across repeated calls, across separate processes, and across
thread counts. `verify_determinism()` re-checks that at runtime instead of
trusting the measurement to hold forever.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from tracelock.face import quality as quality_module
from tracelock.face.errors import (
    AmbiguousFaceError,
    ImageNotFoundError,
    ModelInitializationError,
    NoFaceDetectedError,
    UnsupportedImageError,
)
from tracelock.face.models import (
    SCHEMA_VERSION,
    AnalysisWarning,
    BoundingBox,
    DetectedFace,
    Embedding,
    FaceAnalysisResult,
    FaceSummary,
    ImageRef,
    ModelProvenance,
    Pose,
    SelectionReport,
    WarningCode,
    utc_now,
)
from tracelock.face.selection import (
    DEFAULT_AMBIGUITY_THRESHOLD,
    POLICY_NAME,
    WEIGHTS,
    FaceCandidate,
    select_primary,
)

DEFAULT_PACK = "buffalo_l"
DEFAULT_DET_SIZE = (640, 640)
EXPECTED_EMBEDDING_DIM = 512

# buffalo_l ships five sub-models. Three of them are load-bearing here:
#
#   detection        bbox, det_score, kps
#   recognition      the 512-d embedding
#   landmark_3d_68   pose, which feeds pose-frontality in the quality score
#
# The other two are not. `landmark_2d_106` is never read, and `genderage`
# infers attributes this system has no business deriving -- dropping it is a
# privacy improvement as well as a speed one.
#
# An earlier attempt dropped landmark_3d_68 too. Embeddings stayed bit-identical
# but QUALITY changed on all 12 test images, because pose comes from that model
# and quality feeds the trust score. Speed is not worth silently moving a
# score, so it stays.
REQUIRED_MODULES = ("detection", "recognition", "landmark_3d_68")

# ONNX Runtime defaults intra_op_num_threads to the core count. With N worker
# threads each running a session that wants all 16 cores, the machine ends up
# with 16xN threads fighting over 16 cores.
#
# Measured on this 16-core machine, total seconds for 12 images:
#
#     intra_op   1 worker   2 workers   4 workers   8 workers
#            1      12.74        4.80        2.50        2.25
#            4       3.46        2.89        2.28        1.94
#           16      18.20       13.39        6.17        4.19   <- the default
#
# The earlier finding that "2 workers beat 4" was real, but it was a symptom of
# intra_op=16, not a property of the workload. Constrained to 4 intra-op
# threads, more workers win.
DEFAULT_INTRA_OP_THREADS = 4
DEFAULT_INTER_OP_THREADS = 1


class FaceEngine:
    """Wraps InsightFace FaceAnalysis with structure, provenance and quality.

    Construction loads the model (seconds). Reuse one instance across images.
    """

    def __init__(
        self,
        *,
        pack_name: str = DEFAULT_PACK,
        det_size: tuple[int, int] = DEFAULT_DET_SIZE,
        ctx_id: int = -1,
        providers: tuple[str, ...] = ("CPUExecutionProvider",),
        ambiguity_threshold: float = DEFAULT_AMBIGUITY_THRESHOLD,
        strict_ambiguity: bool = False,
        hash_model_files: bool = True,
        intra_op_threads: int = DEFAULT_INTRA_OP_THREADS,
        modules: tuple[str, ...] = REQUIRED_MODULES,
    ) -> None:
        self.pack_name = pack_name
        self.det_size = det_size
        self.ctx_id = ctx_id
        self.ambiguity_threshold = ambiguity_threshold
        self.strict_ambiguity = strict_ambiguity
        self.intra_op_threads = intra_op_threads
        self.modules = tuple(modules)

        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ModelInitializationError(
                "insightface is not installed. Run: pip install insightface onnxruntime"
            ) from exc

        try:
            self._app = FaceAnalysis(
                name=pack_name,
                providers=list(providers),
                allowed_modules=list(self.modules),
            )
            self._app.prepare(ctx_id=ctx_id, det_size=det_size)
            self._apply_thread_limits(providers)
        except Exception as exc:
            raise ModelInitializationError(
                "failed to initialise model pack {0!r}: {1}".format(pack_name, exc)
            ) from exc

        self._provenance = self._build_provenance(providers, hash_model_files)

    def _apply_thread_limits(self, providers: tuple[str, ...]) -> None:
        """Rebuild each session with a bounded intra-op thread pool.

        InsightFace does not expose SessionOptions, so the sessions are
        replaced after `prepare`. The model FILE is unchanged, so input and
        output names cached during init stay correct -- only the threading
        changes, and threading cannot alter a deterministic model's output.

        Best-effort: if the InsightFace internals differ from what this
        expects, the original sessions are left in place. A slower engine is a
        far better outcome than an engine that fails to start.
        """
        if not self.intra_op_threads:
            return

        try:
            import onnxruntime as ort
        except ImportError:  # pragma: no cover - insightface implies onnxruntime
            return

        for model in getattr(self._app, "models", {}).values():
            model_file = getattr(model, "model_file", None)
            if not model_file or getattr(model, "session", None) is None:
                continue
            try:
                options = ort.SessionOptions()
                options.intra_op_num_threads = self.intra_op_threads
                options.inter_op_num_threads = DEFAULT_INTER_OP_THREADS
                options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                model.session = ort.InferenceSession(
                    model_file, sess_options=options, providers=list(providers)
                )
            except Exception:
                continue

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    def _build_provenance(
        self, providers: tuple[str, ...], hash_files: bool
    ) -> ModelProvenance:
        model_dir = self._resolve_model_dir()

        file_hashes: dict[str, str] = {}
        if hash_files and model_dir and Path(model_dir).is_dir():
            for onnx in sorted(Path(model_dir).glob("*.onnx")):
                file_hashes[onnx.name] = _sha256_file(onnx)

        recognition = self._app.models.get("recognition")
        detection = self._app.models.get("detection")

        return ModelProvenance(
            pack_name=self.pack_name,
            model_dir=str(model_dir) if model_dir else "",
            recognition_model=_model_filename(recognition),
            detection_model=_model_filename(detection),
            embedding_dimension=EXPECTED_EMBEDDING_DIM,
            det_size=self.det_size,
            ctx_id=self.ctx_id,
            providers=tuple(providers),
            model_file_sha256=file_hashes,
            package_versions=_package_versions(),
        )

    def _resolve_model_dir(self) -> str | None:
        for model in self._app.models.values():
            path = getattr(model, "model_file", None)
            if path:
                return str(Path(path).parent)
        return None

    @property
    def provenance(self) -> ModelProvenance:
        return self._provenance

    @property
    def model_id(self) -> str:
        return self._provenance.model_id

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze(self, image_path: str | Path) -> FaceAnalysisResult:
        """Analyze one image. Raises rather than returning None on failure."""
        started = time.perf_counter()
        path = Path(image_path)

        image, image_ref = self._load_image(path)
        height, width = image.shape[:2]

        faces = self._app.get(image)
        if not faces:
            raise NoFaceDetectedError(
                "no face detected in {0}".format(path.name),
                image_sha256=image_ref.sha256,
            )

        candidates = [
            FaceCandidate(
                index=i,
                bbox=BoundingBox(*(float(v) for v in face.bbox[:4])),
                det_score=float(face.det_score),
            )
            for i, face in enumerate(faces)
        ]

        outcome = select_primary(
            candidates, width, height, ambiguity_threshold=self.ambiguity_threshold
        )

        if outcome.ambiguous and self.strict_ambiguity:
            raise AmbiguousFaceError(
                "primary face selection is ambiguous (margin {0:.4f} < {1:.4f})".format(
                    outcome.margin, self.ambiguity_threshold
                ),
                margin=outcome.margin,
                candidates=len(candidates),
            )

        chosen_index = outcome.winner.candidate.index
        raw_face = faces[chosen_index]
        bbox = outcome.winner.candidate.bbox

        pose = _extract_pose(raw_face)
        crop = quality_module.crop_face(image, bbox)
        quality_report = quality_module.assess(
            face_crop=crop,
            bbox=bbox,
            det_score=float(raw_face.det_score),
            pose=pose,
            image_width=width,
            image_height=height,
        )

        embedding = Embedding(
            vector=np.asarray(raw_face.normed_embedding, dtype=np.float32),
            model_id=self._provenance.model_id,
        )
        self._validate_embedding(embedding)

        primary = DetectedFace(
            index=chosen_index,
            bbox=bbox,
            det_score=float(raw_face.det_score),
            landmarks_5=np.asarray(raw_face.kps, dtype=np.float32),
            pose=pose,
            embedding=embedding,
            quality=quality_report,
            age_estimate=_safe_int(getattr(raw_face, "age", None)),
            sex_estimate=getattr(raw_face, "sex", None),
        )

        others = tuple(
            FaceSummary(
                index=scored.candidate.index,
                bbox=scored.candidate.bbox,
                det_score=scored.candidate.det_score,
                selection_score=scored.selection_score,
                area_ratio=scored.candidate.bbox.area / max(1.0, float(width * height)),
            )
            for scored in outcome.runners_up
        )

        selection = SelectionReport(
            policy=POLICY_NAME,
            weights=dict(WEIGHTS),
            chosen_index=chosen_index,
            margin=outcome.margin,
            ambiguous=outcome.ambiguous,
            ambiguity_threshold=self.ambiguity_threshold,
        )

        warnings = _collect_warnings(
            faces_detected=len(faces),
            outcome_ambiguous=outcome.ambiguous,
            margin=outcome.margin,
            primary=primary,
            quality_report=quality_report,
        )

        return FaceAnalysisResult(
            schema_version=SCHEMA_VERSION,
            image=image_ref,
            model=self._provenance,
            faces_detected=len(faces),
            primary=primary,
            others=others,
            selection=selection,
            warnings=warnings,
            analyzed_at=utc_now(),
            elapsed_seconds=time.perf_counter() - started,
        )

    def verify_determinism(self, image_path: str | Path) -> dict[str, Any]:
        """Analyze twice and report whether the embeddings are bit-identical.

        A runtime check, not an assumption. Measured bit-identical on this
        build across processes and thread counts, but a different machine,
        ONNX Runtime version, or execution provider could break it -- and the
        Phase 4 evidence commitment depends on knowing.
        """
        first = self.analyze(image_path)
        second = self.analyze(image_path)

        a = first.embedding.vector
        b = second.embedding.vector

        return {
            "bit_identical": bool(np.array_equal(a, b)),
            "max_abs_diff": float(np.max(np.abs(a - b))),
            "quantized_identical": first.embedding.quantize() == second.embedding.quantize(),
            "det_score_delta": abs(first.primary.det_score - second.primary.det_score),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(path: Path) -> tuple[np.ndarray, ImageRef]:
        if not path.is_file():
            raise ImageNotFoundError("no such image: {0}".format(path))

        raw = path.read_bytes()
        if not raw:
            raise UnsupportedImageError("image file is empty: {0}".format(path))

        import cv2

        # cv2.imread() fails on non-ASCII paths on Windows. Decoding from a
        # byte buffer avoids that entirely, and we need the bytes for the
        # SHA-256 anyway.
        buffer = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise UnsupportedImageError(
                "could not decode as an image: {0}".format(path)
            )

        height, width = image.shape[:2]
        return image, ImageRef(
            path=str(path),
            sha256=hashlib.sha256(raw).hexdigest(),
            width=int(width),
            height=int(height),
            channels=int(image.shape[2]) if image.ndim == 3 else 1,
        )

    def _validate_embedding(self, embedding: Embedding) -> None:
        from tracelock.face.errors import InvalidEmbeddingError

        if embedding.dimension != EXPECTED_EMBEDDING_DIM:
            raise InvalidEmbeddingError(
                "expected {0}-d embedding from {1}, got {2}".format(
                    EXPECTED_EMBEDDING_DIM, self.pack_name, embedding.dimension
                )
            )


# ==========================================================================
# Helpers
# ==========================================================================


def _collect_warnings(
    *,
    faces_detected: int,
    outcome_ambiguous: bool,
    margin: float,
    primary: DetectedFace,
    quality_report,
) -> tuple[AnalysisWarning, ...]:
    """Non-fatal observations later phases can act on."""
    found: list[AnalysisWarning] = []

    if faces_detected > 1:
        found.append(
            AnalysisWarning(
                WarningCode.MULTIPLE_FACES,
                "{0} faces detected; one selected as primary".format(faces_detected),
                {"faces_detected": faces_detected},
            )
        )

    if outcome_ambiguous:
        found.append(
            AnalysisWarning(
                WarningCode.AMBIGUOUS_PRIMARY_FACE,
                "primary selection margin {0:.4f} is below the ambiguity "
                "threshold".format(margin),
                {"margin": round(margin, 4)},
            )
        )

    if primary.det_score < quality_module.WARN_LOW_DET_SCORE:
        found.append(
            AnalysisWarning(
                WarningCode.LOW_DETECTION_CONFIDENCE,
                "detection confidence {0:.3f} is low".format(primary.det_score),
                {"det_score": round(primary.det_score, 4)},
            )
        )

    if primary.bbox.min_side < quality_module.WARN_SMALL_FACE_PX:
        found.append(
            AnalysisWarning(
                WarningCode.SMALL_FACE,
                "face is {0:.0f}px on its shorter side; below {1}px the "
                "embedding is unreliable".format(
                    primary.bbox.min_side, quality_module.WARN_SMALL_FACE_PX
                ),
                {"min_side_px": round(primary.bbox.min_side, 1)},
            )
        )

    sharpness = quality_report.get("sharpness")
    if sharpness and sharpness.score < quality_module.WARN_BLUR_SCORE:
        found.append(
            AnalysisWarning(
                WarningCode.BLURRY_FACE,
                "face appears blurred (laplacian variance {0:.1f}); blur pulls "
                "embeddings toward the population mean and can INFLATE "
                "similarity".format(sharpness.raw_value),
                {"laplacian_variance": round(sharpness.raw_value, 2)},
            )
        )

    if primary.pose and primary.pose.frontal_deviation_deg > quality_module.WARN_POSE_DEG:
        found.append(
            AnalysisWarning(
                WarningCode.EXTREME_POSE,
                "out-of-plane deviation {0:.1f} degrees".format(
                    primary.pose.frontal_deviation_deg
                ),
                {"deviation_deg": round(primary.pose.frontal_deviation_deg, 2)},
            )
        )

    exposure = quality_report.get("exposure_integrity")
    if exposure and exposure.raw_value > quality_module.WARN_CLIPPING_RATIO:
        found.append(
            AnalysisWarning(
                WarningCode.EXPOSURE_CLIPPING,
                "{0:.1%} of face pixels are clipped".format(exposure.raw_value),
                {"clipping_ratio": round(exposure.raw_value, 4)},
            )
        )

    containment = quality_report.get("frame_containment")
    if containment and containment.raw_value < 0.98:
        found.append(
            AnalysisWarning(
                WarningCode.TRUNCATED_FACE,
                "{0:.1%} of the face box lies inside the frame".format(
                    containment.raw_value
                ),
                {"containment_ratio": round(containment.raw_value, 4)},
            )
        )

    if quality_report.aggregate < quality_module.WARN_LOW_AGGREGATE:
        found.append(
            AnalysisWarning(
                WarningCode.LOW_AGGREGATE_QUALITY,
                "aggregate quality {0:.3f} ({1})".format(
                    quality_report.aggregate, quality_report.band.value
                ),
                {"aggregate": round(quality_report.aggregate, 4)},
            )
        )

    if not primary.embedding.is_normalized:
        found.append(
            AnalysisWarning(
                WarningCode.EMBEDDING_NOT_NORMALIZED,
                "embedding L2 norm is {0:.6f}, expected 1.0".format(
                    primary.embedding.l2_norm
                ),
                {"l2_norm": round(primary.embedding.l2_norm, 8)},
            )
        )

    return tuple(found)


def _extract_pose(raw_face) -> Pose | None:
    """Pose in degrees from the 1k3d68 landmark model, if the pack provides it."""
    pose = getattr(raw_face, "pose", None)
    if pose is None or len(pose) < 3:
        return None
    return Pose(pitch=float(pose[0]), yaw=float(pose[1]), roll=float(pose[2]))


def _model_filename(model) -> str:
    if model is None:
        return "unknown"
    path = getattr(model, "model_file", None)
    return Path(path).name if path else type(model).__name__


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_int(value) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _package_versions() -> dict[str, str]:
    """Versions that materially affect the numbers this engine produces."""
    versions: dict[str, str] = {"python": _python_version()}
    for name, module_path in (
        ("insightface", "insightface"),
        ("onnxruntime", "onnxruntime"),
        ("numpy", "numpy"),
        ("opencv", "cv2"),
    ):
        try:
            module = __import__(module_path)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[name] = "not-installed"

    # Thread count changes reduction order in some kernels, so it is recorded
    # even though it was measured NOT to affect results on this build.
    versions["omp_num_threads"] = os.environ.get("OMP_NUM_THREADS", "default")
    return versions


def _python_version() -> str:
    import sys

    return "{0}.{1}.{2}".format(*sys.version_info[:3])
