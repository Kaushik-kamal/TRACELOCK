"""Primary-face selection policy.

Split out of `engine.py` on purpose: this is a documented decision procedure
that must be testable WITHOUT loading a 275 MB model. Every test in
`tests/test_selection.py` runs on plain numbers in milliseconds.

THE POLICY
----------
For a probe image, the subject is normally the prominent face. Three signals,
combined as a weighted sum of values that are each already in [0, 1]:

    score = 0.50 * area_rel + 0.30 * det_score + 0.20 * centrality

  area_rel    face area / largest face area in THIS image.  RELATIVE, not
              absolute -- selection is a comparison among the faces present,
              and absolute area ratios (a face is often 5-15% of a frame)
              compress into a range too narrow to discriminate.

  det_score   detector confidence, already [0, 1]. Stops a spurious
              low-confidence blob from winning on size alone.

  centrality  1 - (distance from face centre to image centre / max distance).
              Weakest signal and weighted lowest: framing convention is real
              but subjects are not reliably centred.

WEIGHT RATIONALE
  Area dominates because in a portrait the subject IS the big face.
  Confidence is the guard rail against detector noise.
  Centrality only breaks near-ties between similarly sized faces.

AMBIGUITY
  The margin between the top two scores is reported, always. When it falls
  below `ambiguity_threshold` the engine attaches an AMBIGUOUS_PRIMARY_FACE
  warning rather than silently pretending certainty. Strict mode raises
  instead. Returning nothing at all would be worse than returning a flagged
  best guess -- the caller can then decide.

DETERMINISM
  Ties are broken by (-selection_score, -area, x1, y1), which is a total order
  over distinct boxes. Two faces can only remain tied if they occupy the
  identical rectangle, which the detector does not emit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from tracelock.face.models import BoundingBox

POLICY_NAME = "area_confidence_centrality/1"

WEIGHT_AREA = 0.50
WEIGHT_DETECTION = 0.30
WEIGHT_CENTRALITY = 0.20

WEIGHTS: dict[str, float] = {
    "area_rel": WEIGHT_AREA,
    "det_score": WEIGHT_DETECTION,
    "centrality": WEIGHT_CENTRALITY,
}

# Below this top-1 vs top-2 gap, the choice is flagged as ambiguous.
# A WORKING DEFAULT: 0.10 on a [0,1] score is roughly "the runner-up is within
# a fifth of the area weight". Not calibrated against labelled data.
DEFAULT_AMBIGUITY_THRESHOLD = 0.10


@dataclass(frozen=True, slots=True)
class FaceCandidate:
    """Minimal input to the policy. Deliberately not tied to insightface."""

    index: int
    bbox: BoundingBox
    det_score: float


@dataclass(frozen=True, slots=True)
class ScoredFace:
    candidate: FaceCandidate
    area_rel: float
    centrality: float
    selection_score: float


@dataclass(frozen=True, slots=True)
class SelectionOutcome:
    winner: ScoredFace
    ranked: tuple[ScoredFace, ...]
    margin: float
    ambiguous: bool
    threshold: float

    @property
    def runners_up(self) -> tuple[ScoredFace, ...]:
        return self.ranked[1:]


def compute_centrality(bbox: BoundingBox, image_width: int, image_height: int) -> float:
    """1.0 when the face centre sits on the image centre, 0.0 at a corner.

    Normalized by the half-diagonal, so the value is resolution-independent
    and comparable across images of different aspect ratios.
    """
    if image_width <= 0 or image_height <= 0:
        return 0.0

    face_cx, face_cy = bbox.center
    image_cx, image_cy = image_width / 2.0, image_height / 2.0

    offset = math.hypot(face_cx - image_cx, face_cy - image_cy)
    half_diagonal = math.hypot(image_cx, image_cy)
    if half_diagonal <= 0:
        return 0.0

    return max(0.0, min(1.0, 1.0 - offset / half_diagonal))


def score_faces(
    candidates: list[FaceCandidate], image_width: int, image_height: int
) -> list[ScoredFace]:
    """Score every candidate. Pure function of its inputs."""
    if not candidates:
        return []

    max_area = max(c.bbox.area for c in candidates)

    scored: list[ScoredFace] = []
    for candidate in candidates:
        area_rel = (candidate.bbox.area / max_area) if max_area > 0 else 0.0
        centrality = compute_centrality(candidate.bbox, image_width, image_height)
        det = max(0.0, min(1.0, candidate.det_score))

        selection_score = (
            WEIGHT_AREA * area_rel
            + WEIGHT_DETECTION * det
            + WEIGHT_CENTRALITY * centrality
        )
        scored.append(
            ScoredFace(
                candidate=candidate,
                area_rel=area_rel,
                centrality=centrality,
                selection_score=selection_score,
            )
        )
    return scored


def select_primary(
    candidates: list[FaceCandidate],
    image_width: int,
    image_height: int,
    *,
    ambiguity_threshold: float = DEFAULT_AMBIGUITY_THRESHOLD,
) -> SelectionOutcome:
    """Choose the primary face. Deterministic; never raises on ambiguity.

    Raises ValueError only when handed an empty list -- callers must deal with
    "no faces" before reaching the policy, because that is a detection outcome,
    not a selection one.
    """
    if not candidates:
        raise ValueError("select_primary requires at least one candidate")

    scored = score_faces(candidates, image_width, image_height)

    # Total order: score desc, then area desc, then position. Fully deterministic.
    ranked = tuple(
        sorted(
            scored,
            key=lambda s: (
                -s.selection_score,
                -s.candidate.bbox.area,
                s.candidate.bbox.x1,
                s.candidate.bbox.y1,
            ),
        )
    )

    winner = ranked[0]
    # A lone face is never ambiguous: margin is the full score, by convention.
    margin = (
        winner.selection_score - ranked[1].selection_score
        if len(ranked) > 1
        else winner.selection_score
    )
    ambiguous = len(ranked) > 1 and margin < ambiguity_threshold

    return SelectionOutcome(
        winner=winner,
        ranked=ranked,
        margin=margin,
        ambiguous=ambiguous,
        threshold=ambiguity_threshold,
    )
