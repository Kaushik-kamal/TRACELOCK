"""Everything measured about the INPUT image, computed exactly once.

The verifier used to recompute the probe's perceptual hash inside the
per-candidate loop -- `_safe_phash(probe.image.path)` on every iteration, 28.8ms
each, for a file that had not changed. Over 25 candidates that is 0.72s spent
re-deriving a constant.

This object is the fix, and the rule it encodes is: anything derived only from
the input belongs here, is computed on construction, and is read from here
afterwards. If a future change needs another input-derived value, it goes in
this class rather than into a loop.

Nothing here is cached across investigations. These are cheap derivations of
one file; the sharing that matters happens at the CAS and face-cache level,
which are keyed by content hash.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class InputAnalysis:
    """The probe, measured once, in the form every later stage wants it."""

    face: Any                       # FaceAnalysisResult
    phash: Any | None               # PerceptualHash | None
    sha256: str
    path: str
    width: int
    height: int

    @property
    def embedding(self):
        return self.face.embedding

    @property
    def quality(self):
        return self.face.primary.quality

    @property
    def quality_aggregate(self) -> float:
        return self.face.primary.quality.aggregate

    @property
    def bounding_box(self):
        return self.face.primary.bbox

    @property
    def aspect_ratio(self) -> float:
        return (self.width / self.height) if self.height else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
            "aspect_ratio": round(self.aspect_ratio, 4),
            "phash": self.phash.hex_digest if self.phash else None,
            "quality": round(self.quality_aggregate, 4),
            "quality_band": self.quality.band.value,
            "faces_detected": self.face.faces_detected,
            "embedding_dimension": self.face.embedding.dimension,
            "model_id": self.face.model.model_id,
        }


def analyze_input(engine, path: str) -> InputAnalysis:
    """Measure the input image once. Raises exactly what `engine.analyze` raises.

    Face analysis is done first and deliberately not guarded: no face means no
    investigation, and the caller already handles that. The perceptual hash is
    best-effort -- a decode failure there costs us near-duplicate detection, not
    the run, so it degrades to None rather than aborting.
    """
    face = engine.analyze(path)

    phash = _safe_phash(path)

    try:
        width, height = _dimensions(path)
    except Exception:
        width = height = 0

    return InputAnalysis(
        face=face,
        phash=phash,
        sha256=face.image.sha256,
        path=path,
        width=width,
        height=height,
    )


def _safe_phash(path: str):
    """Perceptual hash of a file, decoded exactly as the verifier decodes
    candidates -- same library, same flags -- so probe and candidate hashes are
    comparable. Returns None rather than failing the run.
    """
    from tracelock.verification.phash import compute_phash

    try:
        import cv2
        import numpy as np

        data = Path(path).read_bytes()
        array = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        return compute_phash(array) if array is not None else None
    except (OSError, ValueError):
        return None


def _dimensions(path: str) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.width, image.height
