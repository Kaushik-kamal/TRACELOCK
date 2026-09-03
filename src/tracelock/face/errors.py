"""Error taxonomy for the Face Intelligence Engine.

Errors are pipeline control flow, not just diagnostics. Later phases branch on
these types to decide whether a candidate is REJECTED (a real finding worth
recording) or ERRORED (an operational problem that says nothing about identity).
Returning None everywhere would collapse that distinction.

    FaceEngineError
    |
    +-- ModelInitializationError     engine unusable; abort the run
    +-- ImageError                   the input, not the face
    |   +-- ImageNotFoundError
    |   +-- UnsupportedImageError
    +-- DetectionError               ran fine, found nothing usable
    |   +-- NoFaceDetectedError      <- a legitimate candidate REJECTION
    |   +-- AmbiguousFaceError       only raised in strict mode
    +-- EmbeddingError
        +-- EmbeddingDimensionMismatch
        +-- ZeroNormEmbeddingError
        +-- InvalidEmbeddingError
"""

from __future__ import annotations


class FaceEngineError(Exception):
    """Base for every face-engine failure."""


# --------------------------------------------------------------------------
# Model lifecycle
# --------------------------------------------------------------------------


class ModelInitializationError(FaceEngineError):
    """The model pack could not be loaded.

    Operational, never a verdict about an image. Callers should abort rather
    than record a rejection.
    """


# --------------------------------------------------------------------------
# Input problems
# --------------------------------------------------------------------------


class ImageError(FaceEngineError):
    """Base for input-image problems."""


class ImageNotFoundError(ImageError):
    """Path does not exist or is not a file."""


class UnsupportedImageError(ImageError):
    """File exists but could not be decoded into pixels."""


# --------------------------------------------------------------------------
# Detection problems
# --------------------------------------------------------------------------


class DetectionError(FaceEngineError):
    """Base for detection-stage problems."""


class NoFaceDetectedError(DetectionError):
    """No face found above the detector threshold.

    For a CANDIDATE this is a legitimate rejection reason and should be
    recorded, not swallowed. For a PROBE it is a user error.
    """

    def __init__(self, message: str = "no face detected", *, image_sha256: str | None = None):
        super().__init__(message)
        self.image_sha256 = image_sha256


class AmbiguousFaceError(DetectionError):
    """Primary-face selection was too close to call.

    Raised ONLY when the engine runs in strict mode. The default behaviour is
    to select the top-scoring face and attach an AMBIGUOUS_PRIMARY_FACE
    warning -- pretending certainty is worse than flagging doubt, but so is
    refusing to produce any result at all.
    """

    def __init__(self, message: str, *, margin: float, candidates: int):
        super().__init__(message)
        self.margin = margin
        self.candidates = candidates


# --------------------------------------------------------------------------
# Embedding problems
# --------------------------------------------------------------------------


class EmbeddingError(FaceEngineError):
    """Base for embedding problems."""


class InvalidEmbeddingError(EmbeddingError):
    """Embedding contains NaN/Inf, or failed a structural invariant."""


class EmbeddingDimensionMismatch(EmbeddingError):
    """Two embeddings have different dimensions.

    Almost always means embeddings from different model packs are being
    compared, which would silently produce meaningless similarities.
    """

    def __init__(self, left: int, right: int):
        super().__init__(
            "embedding dimension mismatch: {0} vs {1} -- these are likely from "
            "different model packs and must not be compared".format(left, right)
        )
        self.left = left
        self.right = right


class ZeroNormEmbeddingError(EmbeddingError):
    """An embedding has (near) zero L2 norm, so its direction is undefined.

    Cosine similarity is a ratio against the norms; with a zero norm there is
    no meaningful answer and returning 0.0 would be a silent lie.
    """

    def __init__(self, which: str = "embedding", norm: float = 0.0):
        super().__init__(
            "{0} has near-zero L2 norm ({1:.3e}); cosine similarity is "
            "undefined".format(which, norm)
        )
        self.norm = norm
