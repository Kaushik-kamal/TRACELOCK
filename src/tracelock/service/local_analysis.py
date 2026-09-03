"""MODE B -- local-only analysis.

WHY THIS MODE EXISTS
--------------------
Reverse-image search fetches a URL; it cannot receive a file. So a webcam
frame or a local upload has no discovery path unless the image is published
somewhere reachable.

The tempting shortcut is to quietly upload it to a third-party host and carry
on. TRACELOCK will not: a face is biometric data, and silently transmitting it
to make a demo feel complete is exactly the behaviour this project argues
against. The honest alternative is to do everything that CAN be done offline,
and say plainly what was not done.

WHAT THIS MODE PRODUCES, AND WHAT IT STRUCTURALLY CANNOT
--------------------------------------------------------
It produces measurements: image properties, face detection, quality, and a
cryptographic fingerprint.

It has no field for a trust score, no field for candidates, and no field for
evidence. Not empty ones -- ABSENT ones. A caller cannot render a score from
this report because there is nowhere for one to live, and
`assert_no_fabricated_findings` fails loudly if any ever appear.

That matters because a trust score without discovery would be meaningless:
T = 100 * P_id * (...) requires a candidate to have been compared against.
With no candidates there is no P_id, and inventing one would be fabrication.

NETWORK POSTURE
---------------
This module makes ZERO outbound requests. Everything runs against bytes
already on disk. `tests/test_local_analysis.py` proves it by making any HTTP
call raise.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "local-analysis/1"

# Keys that must never appear in a local-analysis payload. Their presence
# would mean the report is claiming something it did not measure.
FORBIDDEN_KEYS: frozenset[str] = frozenset(
    {
        "trust_score",
        "candidates",
        "evidence",
        "verified_candidates",
        "identity_probability",
        "discovery",
    }
)


class FabricatedFindingError(Exception):
    """A local-only report tried to carry a discovery or scoring finding."""


def assert_no_fabricated_findings(payload: Any, path: str = "$") -> None:
    """Walk a local-analysis payload and refuse fabricated findings.

    A hard failure rather than a silent strip: if one of these keys appears,
    something upstream is inventing results and must be fixed, not papered over.
    """
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in FORBIDDEN_KEYS:
                raise FabricatedFindingError(
                    "local analysis payload carries {0!r} at {1}. Local mode "
                    "performs no discovery and no scoring; a value here would "
                    "be fabricated.".format(key, path)
                )
            assert_no_fabricated_findings(value, "{0}.{1}".format(path, key))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            assert_no_fabricated_findings(value, "{0}[{1}]".format(path, index))


@dataclass(frozen=True, slots=True)
class LocalAnalysisReport:
    """Everything measurable without touching the network.

    Deliberately has no `trust_score`, `candidates` or `evidence` attribute.
    """

    schema_version: str
    mode: str  # always LOCAL_ANALYSIS
    created_at: str

    # Image
    sha256: str
    image_format: str
    width: int
    height: int
    byte_size: int
    mime_type: str

    # Face
    face_detected: bool
    faces_detected: int
    det_score: float | None
    quality_aggregate: float | None
    quality_band: str | None
    face_width: float | None
    face_height: float | None
    pose_deviation_deg: float | None
    quality_metrics: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]

    # Fingerprint -- commitments only, never the embedding
    embedding_dimension: int | None
    embedding_quantized_sha256: str | None
    model_id: str | None
    phash: str | None

    # Provenance
    source_type: str
    source_label: str
    input_kind: str
    filename: str

    reason_no_discovery: str = (
        "No publicly reachable URL was provided for this image. Reverse-image "
        "search fetches a URL and cannot receive a file, so public discovery "
        "was not possible."
    )
    privacy_note: str = (
        "TRACELOCK does not silently upload biometric images to third-party "
        "services. This image never left your machine."
    )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "created_at": self.created_at,
            "image": {
                "sha256": self.sha256,
                "sha256_short": self.sha256[:16] + "…",
                "format": self.image_format,
                "width": self.width,
                "height": self.height,
                "dimensions": "{0} x {1}".format(self.width, self.height),
                "byte_size": self.byte_size,
                "mime_type": self.mime_type,
            },
            "face": {
                "detected": self.face_detected,
                "count": self.faces_detected,
                "det_score": (
                    round(self.det_score, 4) if self.det_score is not None else None
                ),
                "quality": (
                    round(self.quality_aggregate, 4)
                    if self.quality_aggregate is not None
                    else None
                ),
                "quality_band": self.quality_band,
                "face_size": (
                    "{0:.0f} x {1:.0f} px".format(self.face_width, self.face_height)
                    if self.face_width is not None
                    else None
                ),
                "pose_deviation_deg": (
                    round(self.pose_deviation_deg, 2)
                    if self.pose_deviation_deg is not None
                    else None
                ),
                "quality_metrics": list(self.quality_metrics),
                "warnings": list(self.warnings),
            },
            "fingerprint": {
                "image_sha256": self.sha256,
                "perceptual_hash": self.phash,
                "embedding_dimension": self.embedding_dimension,
                # A commitment to the embedding. The vector itself never
                # leaves the face engine.
                "embedding_quantized_sha256": self.embedding_quantized_sha256,
                "model_id": self.model_id,
                "note": (
                    "Commitments only. The raw face embedding is never stored, "
                    "transmitted, or written to any artifact."
                ),
            },
            "provenance": {
                "source_type": self.source_type,
                "source_label": self.source_label,
                "input_kind": self.input_kind,
                "filename": self.filename,
            },
            "privacy": {
                "status": "LOCAL ONLY",
                "network_requests_made": 0,
                "image_transmitted": False,
                "note": self.privacy_note,
            },
            "public_discovery": {
                "performed": False,
                "status": "NOT PERFORMED",
                "reason": self.reason_no_discovery,
                "how_to_enable": (
                    "Provide a publicly reachable URL for this image, then run "
                    "public web discovery."
                ),
            },
        }
        # Structural guarantee, enforced on every serialization.
        assert_no_fabricated_findings(payload)
        return payload


def analyse_locally(trace_input, face_engine) -> LocalAnalysisReport:
    """Run every offline measurement. Makes no network request.

    `face_engine` is injected so this is testable without loading buffalo_l,
    and so the caller controls the singleton.
    """
    from tracelock.face.errors import NoFaceDetectedError

    path = Path(trace_input.local_path)
    data = path.read_bytes()

    # Image properties, straight from the bytes on disk.
    from tracelock.acquisition.validation import validate_image_bytes

    validation = validate_image_bytes(data)

    phash_hex = None
    try:
        from tracelock.verification.phash import phash_from_bytes

        phash_hex = phash_from_bytes(data).hex_digest
    except (ValueError, ImportError):
        phash_hex = None

    face_detected = True
    analysis = None
    try:
        analysis = face_engine.analyze(path)
    except NoFaceDetectedError:
        face_detected = False

    if analysis is None:
        return LocalAnalysisReport(
            schema_version=SCHEMA_VERSION,
            mode="LOCAL_ANALYSIS",
            created_at=datetime.now(timezone.utc).isoformat(),
            sha256=trace_input.sha256,
            image_format=validation.detected_format or "unknown",
            width=validation.width or 0,
            height=validation.height or 0,
            byte_size=len(data),
            mime_type=trace_input.mime_type,
            face_detected=False,
            faces_detected=0,
            det_score=None,
            quality_aggregate=None,
            quality_band=None,
            face_width=None,
            face_height=None,
            pose_deviation_deg=None,
            quality_metrics=(),
            warnings=("No face was detected in this image.",),
            embedding_dimension=None,
            embedding_quantized_sha256=None,
            model_id=None,
            phash=phash_hex,
            source_type=trace_input.source_type.value,
            source_label=trace_input.source_type.label,
            input_kind=trace_input.kind.value,
            filename=trace_input.filename,
        )

    primary = analysis.primary
    quality = primary.quality

    return LocalAnalysisReport(
        schema_version=SCHEMA_VERSION,
        mode="LOCAL_ANALYSIS",
        created_at=datetime.now(timezone.utc).isoformat(),
        sha256=trace_input.sha256,
        image_format=validation.detected_format or "unknown",
        width=validation.width or analysis.image.width,
        height=validation.height or analysis.image.height,
        byte_size=len(data),
        mime_type=trace_input.mime_type,
        face_detected=face_detected,
        faces_detected=analysis.faces_detected,
        det_score=float(primary.det_score),
        quality_aggregate=quality.aggregate,
        quality_band=quality.band.value,
        face_width=primary.bbox.width,
        face_height=primary.bbox.height,
        pose_deviation_deg=(
            primary.pose.frontal_deviation_deg if primary.pose else None
        ),
        quality_metrics=tuple(m.to_dict() for m in quality.metrics),
        warnings=tuple(analysis.warning_codes()),
        embedding_dimension=analysis.embedding.dimension,
        embedding_quantized_sha256=hashlib.sha256(
            analysis.embedding.quantize()
        ).hexdigest(),
        model_id=analysis.embedding.model_id,
        phash=phash_hex,
        source_type=trace_input.source_type.value,
        source_label=trace_input.source_type.label,
        input_kind=trace_input.kind.value,
        filename=trace_input.filename,
    )
