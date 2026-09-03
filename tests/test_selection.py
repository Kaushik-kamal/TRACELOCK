"""Primary-face selection policy.

Runs on plain numbers -- no model, no images, milliseconds. That is why the
policy lives in its own module.
"""

from __future__ import annotations

import pytest

from tracelock.face.models import BoundingBox
from tracelock.face.selection import (
    WEIGHTS,
    FaceCandidate,
    compute_centrality,
    score_faces,
    select_primary,
)

IMAGE_W, IMAGE_H = 1000, 1000


def face(index: int, x1: float, y1: float, size: float, det: float = 0.9):
    return FaceCandidate(
        index=index,
        bbox=BoundingBox(x1, y1, x1 + size, y1 + size),
        det_score=det,
    )


class TestCentrality:
    def test_dead_centre_scores_one(self):
        bbox = BoundingBox(450, 450, 550, 550)
        assert compute_centrality(bbox, IMAGE_W, IMAGE_H) == pytest.approx(1.0)

    def test_corner_scores_near_zero(self):
        bbox = BoundingBox(0, 0, 20, 20)
        assert compute_centrality(bbox, IMAGE_W, IMAGE_H) < 0.05

    def test_bounded_in_unit_range(self):
        for x in (0, 200, 500, 800, 980):
            value = compute_centrality(BoundingBox(x, x, x + 20, x + 20), IMAGE_W, IMAGE_H)
            assert 0.0 <= value <= 1.0

    def test_resolution_independent(self):
        # Same relative position in differently sized images -> same score.
        small = compute_centrality(BoundingBox(90, 90, 110, 110), 200, 200)
        large = compute_centrality(BoundingBox(900, 900, 1100, 1100), 2000, 2000)
        assert small == pytest.approx(large, abs=1e-9)

    def test_degenerate_image_is_zero(self):
        assert compute_centrality(BoundingBox(0, 0, 10, 10), 0, 0) == 0.0


class TestWeights:
    def test_weights_sum_to_one(self):
        assert sum(WEIGHTS.values()) == pytest.approx(1.0)

    def test_area_is_the_dominant_signal(self):
        assert WEIGHTS["area_rel"] > WEIGHTS["det_score"] > WEIGHTS["centrality"]


class TestScoring:
    def test_area_is_relative_to_largest_in_image(self):
        scored = score_faces([face(0, 0, 0, 200), face(1, 500, 500, 100)], IMAGE_W, IMAGE_H)
        by_index = {s.candidate.index: s for s in scored}
        assert by_index[0].area_rel == pytest.approx(1.0)
        # 100x100 vs 200x200 -> quarter the area.
        assert by_index[1].area_rel == pytest.approx(0.25)

    def test_score_stays_in_unit_range(self):
        for scored in score_faces(
            [face(0, 0, 0, 500, det=1.0), face(1, 900, 900, 20, det=0.1)],
            IMAGE_W,
            IMAGE_H,
        ):
            assert 0.0 <= scored.selection_score <= 1.0

    def test_empty_input_gives_empty_output(self):
        assert score_faces([], IMAGE_W, IMAGE_H) == []


class TestSelection:
    def test_single_face_wins(self):
        outcome = select_primary([face(0, 400, 400, 200)], IMAGE_W, IMAGE_H)
        assert outcome.winner.candidate.index == 0
        assert not outcome.ambiguous

    def test_single_face_is_never_ambiguous(self):
        # Even a tiny, off-centre, low-confidence lone face is unambiguous:
        # there is nothing to confuse it with.
        outcome = select_primary([face(0, 0, 0, 15, det=0.3)], IMAGE_W, IMAGE_H)
        assert not outcome.ambiguous

    def test_larger_face_beats_smaller(self):
        outcome = select_primary(
            [face(0, 100, 100, 80), face(1, 400, 400, 300)], IMAGE_W, IMAGE_H
        )
        assert outcome.winner.candidate.index == 1

    def test_confidence_breaks_equal_size_and_position(self):
        outcome = select_primary(
            [face(0, 100, 450, 100, det=0.30), face(1, 800, 450, 100, det=0.99)],
            IMAGE_W,
            IMAGE_H,
        )
        assert outcome.winner.candidate.index == 1

    def test_centrality_breaks_otherwise_identical_faces(self):
        outcome = select_primary(
            [face(0, 10, 10, 100, det=0.9), face(1, 450, 450, 100, det=0.9)],
            IMAGE_W,
            IMAGE_H,
        )
        assert outcome.winner.candidate.index == 1

    def test_ranked_is_ordered_by_score_descending(self):
        outcome = select_primary(
            [face(0, 0, 0, 50), face(1, 450, 450, 300), face(2, 200, 200, 150)],
            IMAGE_W,
            IMAGE_H,
        )
        scores = [s.selection_score for s in outcome.ranked]
        assert scores == sorted(scores, reverse=True)

    def test_all_faces_are_retained_in_ranking(self):
        outcome = select_primary(
            [face(i, i * 100, 100, 60) for i in range(5)], IMAGE_W, IMAGE_H
        )
        assert len(outcome.ranked) == 5
        assert len(outcome.runners_up) == 4

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="at least one candidate"):
            select_primary([], IMAGE_W, IMAGE_H)


class TestAmbiguity:
    def test_two_near_identical_faces_are_ambiguous(self):
        # Mirrored about the centre: same size, same confidence, same centrality.
        outcome = select_primary(
            [face(0, 300, 450, 100, det=0.9), face(1, 600, 450, 100, det=0.9)],
            IMAGE_W,
            IMAGE_H,
        )
        assert outcome.ambiguous
        assert outcome.margin < 0.10

    def test_clearly_dominant_face_is_not_ambiguous(self):
        outcome = select_primary(
            [face(0, 400, 400, 400, det=0.99), face(1, 20, 20, 40, det=0.5)],
            IMAGE_W,
            IMAGE_H,
        )
        assert not outcome.ambiguous
        assert outcome.margin > 0.10

    def test_threshold_is_configurable(self):
        candidates = [face(0, 300, 450, 100), face(1, 600, 450, 100)]
        assert not select_primary(
            candidates, IMAGE_W, IMAGE_H, ambiguity_threshold=0.0
        ).ambiguous
        assert select_primary(
            candidates, IMAGE_W, IMAGE_H, ambiguity_threshold=0.99
        ).ambiguous

    def test_margin_is_always_reported(self):
        outcome = select_primary(
            [face(0, 400, 400, 200), face(1, 100, 100, 100)], IMAGE_W, IMAGE_H
        )
        assert outcome.margin > 0


class TestDeterminism:
    def test_repeated_calls_agree(self):
        candidates = [face(i, i * 90, 300, 120, det=0.8 + i * 0.01) for i in range(6)]
        results = {
            select_primary(candidates, IMAGE_W, IMAGE_H).winner.candidate.index
            for _ in range(25)
        }
        assert len(results) == 1

    def test_input_order_does_not_change_the_winner(self):
        candidates = [face(0, 400, 400, 200), face(1, 100, 100, 150), face(2, 700, 700, 180)]
        forward = select_primary(candidates, IMAGE_W, IMAGE_H)
        backward = select_primary(list(reversed(candidates)), IMAGE_W, IMAGE_H)
        assert forward.winner.candidate.index == backward.winner.candidate.index

    def test_exact_ties_resolve_deterministically_by_position(self):
        # Two boxes identical in every scored dimension, differing only in x.
        # The total-order tie-break must still produce a stable winner.
        a = FaceCandidate(0, BoundingBox(100, 450, 200, 550), 0.9)
        b = FaceCandidate(1, BoundingBox(800, 450, 900, 550), 0.9)
        winners = {
            select_primary([a, b], IMAGE_W, IMAGE_H).winner.candidate.index
            for _ in range(10)
        } | {
            select_primary([b, a], IMAGE_W, IMAGE_H).winner.candidate.index
            for _ in range(10)
        }
        assert len(winners) == 1
