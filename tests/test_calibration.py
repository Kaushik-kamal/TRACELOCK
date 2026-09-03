"""Calibration contract and metrics.

The most important tests here are in TestAntiFabrication. They assert that a
report derived from synthetic data can never present itself as empirical
evidence. That is the integrity guarantee of this subsystem.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tracelock.calibration.contract import (
    CalibrationDataset,
    CalibrationPair,
    ConsentRecord,
    DatasetKind,
    DatasetProvenance,
    ObservationSet,
    PairLabel,
    SimilarityObservation,
)
from tracelock.calibration.loaders import FixtureLoader, ManifestLoader
from tracelock.calibration.metrics import (
    FIXTURE_BANNER,
    equal_error_rate,
    evaluate,
    roc_curve,
    threshold_at_far,
    wilson_interval,
)


def pair(label: PairLabel, index: int = 0) -> CalibrationPair:
    same = label is PairLabel.GENUINE
    return CalibrationPair(
        pair_id="p{0}".format(index),
        image_a="a.jpg",
        image_b="b.jpg",
        label=label,
        subject_a="s{0}".format(index),
        subject_b="s{0}".format(index if same else index + 1000),
    )


def observations(genuine: list[float], impostor: list[float], kind=DatasetKind.REAL):
    provenance = DatasetProvenance(
        kind=kind,
        name="test",
        consent=(
            (ConsentRecord("subject-01", True, source="self"),)
            if kind is DatasetKind.REAL
            else ()
        ),
    )
    items = [
        SimilarityObservation(pair(PairLabel.GENUINE, i), score, "test-model")
        for i, score in enumerate(genuine)
    ] + [
        SimilarityObservation(pair(PairLabel.IMPOSTOR, i), score, "test-model")
        for i, score in enumerate(impostor)
    ]
    return ObservationSet(provenance, tuple(items), "test-model")


# ==========================================================================


class TestPairContract:
    def test_genuine_pair_with_different_subjects_is_rejected(self):
        with pytest.raises(ValueError, match="GENUINE but names two different"):
            CalibrationPair("a.jpg", "b.jpg", PairLabel.GENUINE, "p1", "alice", "bob")

    def test_impostor_pair_with_same_subject_is_rejected(self):
        with pytest.raises(ValueError, match="IMPOSTOR but names the same"):
            CalibrationPair("a.jpg", "b.jpg", PairLabel.IMPOSTOR, "p1", "alice", "alice")

    def test_consistent_pairs_are_accepted(self):
        CalibrationPair("a.jpg", "b.jpg", PairLabel.GENUINE, "p1", "alice", "alice")
        CalibrationPair("a.jpg", "b.jpg", PairLabel.IMPOSTOR, "p2", "alice", "bob")

    def test_subjects_are_optional(self):
        CalibrationPair("a.jpg", "b.jpg", PairLabel.GENUINE)


class TestProvenance:
    def test_real_with_consent_is_citable(self):
        provenance = DatasetProvenance(
            DatasetKind.REAL, "real", consent=(ConsentRecord("s1", True),)
        )
        assert provenance.citable
        assert not provenance.is_fixture

    def test_real_without_consent_is_not_citable(self):
        assert not DatasetProvenance(DatasetKind.REAL, "real").citable

    def test_fixture_is_never_citable(self):
        provenance = DatasetProvenance(
            DatasetKind.FIXTURE, "fixture", consent=(ConsentRecord("s1", True),)
        )
        assert not provenance.citable
        assert provenance.is_fixture


class TestDataset:
    def test_splits_by_label(self):
        dataset = CalibrationDataset(
            DatasetProvenance(DatasetKind.REAL, "d"),
            tuple(pair(PairLabel.GENUINE, i) for i in range(3))
            + tuple(pair(PairLabel.IMPOSTOR, i) for i in range(7)),
        )
        assert len(dataset) == 10
        assert len(dataset.genuine_pairs) == 3
        assert len(dataset.impostor_pairs) == 7

    def test_summary_reports_counts_and_kind(self):
        dataset = CalibrationDataset(
            DatasetProvenance(DatasetKind.FIXTURE, "f"), (pair(PairLabel.GENUINE),)
        )
        summary = dataset.summary()
        assert summary["is_fixture"] is True
        assert summary["total_pairs"] == 1


class TestRocCurve:
    def test_perfectly_separated_gives_auc_one(self):
        curve = roc_curve([0.9, 0.92, 0.95], [0.0, 0.05, 0.1])
        assert curve.auc() == pytest.approx(1.0, abs=0.02)

    def test_eer_is_zero_when_separated(self):
        eer, _ = equal_error_rate(roc_curve([0.9, 0.92, 0.95], [0.0, 0.05, 0.1]))
        assert eer == pytest.approx(0.0, abs=0.02)

    def test_identical_distributions_give_eer_near_half(self):
        rng = np.random.default_rng(3)
        scores = rng.normal(0.5, 0.1, 300)
        eer, _ = equal_error_rate(roc_curve(scores, rng.normal(0.5, 0.1, 300)))
        assert 0.35 < eer < 0.65

    def test_far_decreases_as_threshold_rises(self):
        curve = roc_curve([0.8, 0.85, 0.9], [0.1, 0.2, 0.3])
        assert curve.far[0] >= curve.far[-1]

    def test_frr_increases_as_threshold_rises(self):
        curve = roc_curve([0.8, 0.85, 0.9], [0.1, 0.2, 0.3])
        assert curve.frr[0] <= curve.frr[-1]

    def test_rates_bounded(self):
        curve = roc_curve([0.7, 0.8], [0.1, 0.2, 0.75])
        assert np.all((curve.far >= 0) & (curve.far <= 1))
        assert np.all((curve.frr >= 0) & (curve.frr <= 1))

    def test_missing_a_class_raises(self):
        with pytest.raises(ValueError, match="both genuine and impostor"):
            roc_curve([0.9], [])
        with pytest.raises(ValueError, match="both genuine and impostor"):
            roc_curve([], [0.1])


class TestThresholdAtFar:
    def test_respects_the_target(self):
        rng = np.random.default_rng(4)
        curve = roc_curve(rng.normal(0.7, 0.08, 400), rng.normal(0.15, 0.08, 4000))
        point = threshold_at_far(curve, 0.01)
        assert point.far <= 0.01 + 1e-9

    def test_stricter_target_needs_a_higher_threshold(self):
        rng = np.random.default_rng(5)
        curve = roc_curve(rng.normal(0.7, 0.08, 400), rng.normal(0.15, 0.08, 4000))
        assert threshold_at_far(curve, 0.001).threshold >= threshold_at_far(curve, 0.05).threshold

    def test_stricter_target_costs_more_false_rejects(self):
        rng = np.random.default_rng(6)
        curve = roc_curve(rng.normal(0.6, 0.12, 400), rng.normal(0.2, 0.12, 4000))
        assert threshold_at_far(curve, 0.001).frr >= threshold_at_far(curve, 0.05).frr

    def test_tar_is_the_complement_of_frr(self):
        point = threshold_at_far(roc_curve([0.8, 0.9], [0.1, 0.2]), 0.01)
        assert point.tar == pytest.approx(1.0 - point.frr)

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_invalid_target_raises(self, bad):
        with pytest.raises(ValueError, match="target_far"):
            threshold_at_far(roc_curve([0.8], [0.1]), bad)


class TestWilsonInterval:
    def test_contains_the_estimate(self):
        low, high = wilson_interval(0.05, 100)
        assert low <= 0.05 <= high

    def test_narrows_as_n_grows(self):
        small = wilson_interval(0.05, 20)
        large = wilson_interval(0.05, 2000)
        assert (large[1] - large[0]) < (small[1] - small[0])

    def test_bounded_in_unit_range(self):
        for rate in (0.0, 0.01, 0.5, 0.99, 1.0):
            low, high = wilson_interval(rate, 50)
            assert 0.0 <= low <= high <= 1.0

    def test_zero_n_is_maximally_uncertain(self):
        assert wilson_interval(0.5, 0) == (0.0, 1.0)


class TestEvaluate:
    def test_produces_a_report(self):
        report = evaluate(observations([0.8, 0.85, 0.9], [0.1, 0.15, 0.2]))
        assert report.n_genuine == 3
        assert report.n_impostor == 3
        assert report.auc > 0.9

    def test_reports_both_class_counts(self):
        report = evaluate(observations([0.8] * 5, [0.1] * 40))
        assert report.n_genuine == 5
        assert report.n_impostor == 40

    def test_confidence_note_uses_the_smaller_class(self):
        assert "n=5" in evaluate(observations([0.8] * 5, [0.1] * 40)).confidence_interval_note()

    def test_missing_a_class_raises(self):
        with pytest.raises(ValueError, match="both classes"):
            evaluate(observations([0.8, 0.9], []))

    def test_serializes(self):
        payload = evaluate(observations([0.8, 0.9], [0.1, 0.2])).to_dict()
        assert json.dumps(payload)
        assert "eer" in payload and "confidence_note" in payload


# ==========================================================================
# THE INTEGRITY TESTS
# ==========================================================================


class TestAntiFabrication:
    """A fixture-derived result must never look like empirical evidence."""

    def test_fixture_flag_propagates_dataset_to_report(self):
        report = evaluate(observations([0.8, 0.9], [0.1, 0.2], kind=DatasetKind.FIXTURE))
        assert report.is_fixture is True
        assert report.citable is False

    def test_fixture_report_refuses_to_render_metrics(self):
        rendered = evaluate(
            observations([0.8, 0.9], [0.1, 0.2], kind=DatasetKind.FIXTURE)
        ).render()
        assert "NOT EMPIRICAL EVIDENCE" in rendered
        assert "DO NOT CITE" in rendered
        # The actual numbers must be withheld, not merely captioned.
        assert "AUC" not in rendered

    def test_real_report_does_render_metrics(self):
        rendered = evaluate(observations([0.8, 0.9], [0.1, 0.2])).render()
        assert "AUC" in rendered
        assert "NOT EMPIRICAL EVIDENCE" not in rendered

    def test_fixture_flag_survives_serialization(self):
        payload = evaluate(
            observations([0.8, 0.9], [0.1, 0.2], kind=DatasetKind.FIXTURE)
        ).to_dict()
        assert payload["is_fixture"] is True
        assert payload["citable"] is False

    def test_fixture_loader_is_always_marked_fixture(self):
        assert FixtureLoader(n_genuine=5, n_impostor=5).load().is_fixture is True

    def test_fixture_observations_stay_marked(self):
        assert FixtureLoader(n_genuine=5, n_impostor=10).observe().is_fixture is True

    def test_consent_alone_cannot_launder_a_fixture(self):
        # Even a fully consented fixture is still a fixture.
        provenance = DatasetProvenance(
            DatasetKind.FIXTURE, "sneaky", consent=(ConsentRecord("s1", True),)
        )
        assert not provenance.citable

    def test_uncalibrated_real_data_is_flagged_non_citable(self):
        report = evaluate(
            ObservationSet(
                DatasetProvenance(DatasetKind.REAL, "no-consent"),
                observations([0.8], [0.1]).observations,
                "m",
            )
        )
        assert not report.citable
        assert "no consent record" in report.render()


class TestFixtureLoader:
    def test_generates_the_requested_counts(self):
        result = FixtureLoader(n_genuine=12, n_impostor=30).observe()
        assert len(result.genuine_scores) == 12
        assert len(result.impostor_scores) == 30

    def test_separates_the_populations(self):
        result = FixtureLoader(
            n_genuine=50, n_impostor=200, genuine_similarity=0.75, impostor_similarity=0.05
        ).observe()
        assert np.mean(result.genuine_scores) > np.mean(result.impostor_scores) + 0.4

    def test_is_reproducible_from_the_seed(self):
        a = FixtureLoader(n_genuine=10, n_impostor=10, seed=99).observe()
        b = FixtureLoader(n_genuine=10, n_impostor=10, seed=99).observe()
        assert a.genuine_scores == b.genuine_scores

    def test_scores_are_valid_similarities(self):
        result = FixtureLoader(n_genuine=20, n_impostor=20).observe()
        for score in result.genuine_scores + result.impostor_scores:
            assert -1.0 <= score <= 1.0

    def test_metrics_run_end_to_end_on_a_fixture(self):
        # Proves the maths works, which is the fixture's only legitimate use.
        report = evaluate(FixtureLoader(n_genuine=40, n_impostor=300).observe())
        assert report.auc > 0.9
        assert report.is_fixture


class TestManifestLoader:
    def _write(self, tmp_path, payload):
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_loads_a_consented_manifest(self, tmp_path):
        path = self._write(
            tmp_path,
            {
                "name": "consented-v1",
                "consent": [
                    {"subject_ref": "s1", "consent_obtained": True, "source": "self"}
                ],
                "pairs": [
                    {
                        "pair_id": "p1",
                        "image_a": "a.jpg",
                        "image_b": "b.jpg",
                        "label": "GENUINE",
                        "subject_a": "s1",
                        "subject_b": "s1",
                    }
                ],
            },
        )
        dataset = ManifestLoader(path).load()
        assert dataset.provenance.kind is DatasetKind.REAL
        assert dataset.provenance.citable
        assert len(dataset) == 1

    def test_rejects_a_manifest_without_consent(self, tmp_path):
        path = self._write(tmp_path, {"name": "no-consent", "consent": [], "pairs": []})
        with pytest.raises(ValueError, match="declares no obtained consent"):
            ManifestLoader(path).load()

    def test_rejects_consent_that_was_not_obtained(self, tmp_path):
        path = self._write(
            tmp_path,
            {
                "name": "refused",
                "consent": [{"subject_ref": "s1", "consent_obtained": False}],
                "pairs": [],
            },
        )
        with pytest.raises(ValueError, match="declares no obtained consent"):
            ManifestLoader(path).load()

    def test_consent_requirement_can_be_waived_explicitly(self, tmp_path):
        path = self._write(tmp_path, {"name": "public-dataset", "consent": [], "pairs": []})
        assert ManifestLoader(path, require_consent=False).load() is not None

    def test_missing_manifest_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ManifestLoader(tmp_path / "nope.json").load()

    def test_image_paths_resolve_relative_to_the_manifest(self, tmp_path):
        path = self._write(
            tmp_path,
            {
                "name": "rel",
                "consent": [{"subject_ref": "s1", "consent_obtained": True}],
                "pairs": [
                    {"image_a": "x.jpg", "image_b": "y.jpg", "label": "IMPOSTOR"}
                ],
            },
        )
        loaded = ManifestLoader(path).load().pairs[0]
        assert str(tmp_path) in loaded.image_a
