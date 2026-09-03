"""Phase 3 -- evidence aggregation and trust scoring.

The tests that matter here enforce the design principles: duplicates count
once, one publisher gets one vote, and no score is emitted without calibration.
"""

from __future__ import annotations

import json

import pytest

from tracelock.evidence.aggregate import CORROBORATION_LAMBDA, aggregate
from tracelock.evidence.trust import (
    QUALITY_FLOOR,
    TrustBand,
    UncalibratedScoreRefused,
    measure_acquisition_integrity,
    measure_metadata_completeness,
    score_evidence,
)


def result(
    *,
    candidate_id="c1",
    status="VERIFIED_CANDIDATE",
    sha="a" * 64,
    domain="example.com",
    probability=0.8,
    similarity=0.45,
    quality=0.8,
    relation="SAME_PERSON_DIFFERENT_PHOTO",
    phash=30,
    reasons=(),
):
    return {
        "candidate_id": candidate_id,
        "status": status,
        "content_sha256": sha,
        "source_url": "https://{0}/post".format(domain),
        "media_url": "https://cdn.{0}/i.jpg".format(domain),
        "cas_path": "data/cas/blobs/aa/{0}".format(sha),
        "face_similarity": similarity,
        "identity_probability": probability,
        "provenance": {"registrable_domain": domain},
        "acquisition": {"ok": True, "status_code": 200, "url_changed": False},
        "validation": {"ok": True, "detected_format": "JPEG",
                       "content_type_was_honest": True},
        "face": {"quality_aggregate": quality},
        "relation": {"relation": relation, "phash_distance": phash},
        "rejection_reasons": [
            {"reason": r, "stage": "VALIDATED", "explanation": "x"} for r in reasons
        ],
    }


def rejected(candidate_id, reason, **kwargs):
    return result(candidate_id=candidate_id, status="REJECTED",
                  reasons=(reason,), **kwargs)


class TestFunnelAccounting:
    def test_every_candidate_lands_in_a_terminal_bucket(self):
        results = [
            result(candidate_id="c1"),
            result(candidate_id="c2", status="INCONCLUSIVE", sha="b" * 64),
            rejected("c3", "HTTP_ERROR", sha="c" * 64),
        ]
        funnel = aggregate(results).funnel
        assert funnel.discovered == 3
        assert funnel.verified + funnel.inconclusive + funnel.rejected == 3

    def test_rejection_breakdown_is_counted(self):
        results = [
            rejected("c1", "HTTP_ERROR", sha="a" * 64),
            rejected("c2", "HTTP_ERROR", sha="b" * 64),
            rejected("c3", "NOT_AN_IMAGE", sha="c" * 64),
        ]
        breakdown = aggregate(results).rejection_breakdown
        assert breakdown["HTTP_ERROR"] == 2
        assert breakdown["NOT_AN_IMAGE"] == 1


class TestDuplicatesCountOnce:
    """PRINCIPLE 1: never count duplicate images as independent evidence."""

    def test_same_bytes_collapse_to_one_item(self):
        results = [
            result(candidate_id="c1", sha="a" * 64, domain="one.com"),
            result(candidate_id="c2", sha="a" * 64, domain="two.com"),
            result(candidate_id="c3", sha="a" * 64, domain="three.com"),
        ]
        evidence = aggregate(results)
        assert evidence.funnel.unique_images == 1
        assert len(evidence.items) == 1

    def test_republication_still_records_every_publisher(self):
        # One image, three places it was published. Both facts are preserved.
        results = [
            result(candidate_id="c1", sha="a" * 64, domain="one.com"),
            result(candidate_id="c2", sha="a" * 64, domain="two.com"),
        ]
        item = aggregate(results).items[0]
        assert set(item.domains) == {"one.com", "two.com"}
        assert item.publisher_count == 2

    def test_duplicates_rejected_at_validation_still_contribute_publisher(self):
        results = [
            result(candidate_id="c1", sha="a" * 64, domain="one.com"),
            rejected("c2", "DUPLICATE_CONTENT", sha="a" * 64, domain="two.com"),
        ]
        evidence = aggregate(results)
        assert evidence.funnel.unique_images == 1
        assert "two.com" in evidence.items[0].domains

    def test_distinct_bytes_stay_distinct(self):
        results = [
            result(candidate_id="c1", sha="a" * 64),
            result(candidate_id="c2", sha="b" * 64, domain="other.com"),
        ]
        assert aggregate(results).funnel.unique_images == 2

    def test_duplicate_groups_are_recorded(self):
        results = [
            result(candidate_id="c1", sha="a" * 64, domain="one.com"),
            result(candidate_id="c2", sha="a" * 64, domain="two.com"),
        ]
        groups = aggregate(results).duplicate_groups
        assert "a" * 64 in groups
        assert len(groups["a" * 64]) == 2


class TestOnePublisherOneVote:
    """PRINCIPLE 2: multiple URLs from one domain are not corroboration."""

    def test_same_domain_counts_once(self):
        results = [
            result(candidate_id="c{0}".format(i), sha=chr(97 + i) * 64,
                   domain="same.com")
            for i in range(4)
        ]
        evidence = aggregate(results)
        assert evidence.funnel.unique_images == 4      # four distinct images
        assert evidence.independent_publisher_count == 1  # one publisher

    def test_one_publisher_yields_zero_corroboration(self):
        results = [result(candidate_id="c1", domain="only.com")]
        assert aggregate(results).corroboration_factor() == 0.0

    def test_subdomains_collapse_to_the_registrable_domain(self):
        # Provenance already supplies eTLD+1; aggregation must not re-split it.
        results = [
            result(candidate_id="c1", sha="a" * 64, domain="news.com"),
            result(candidate_id="c2", sha="b" * 64, domain="news.com"),
        ]
        assert aggregate(results).independent_publisher_count == 1

    def test_corroboration_grows_and_saturates(self):
        def factor(n):
            return aggregate([
                result(candidate_id="c{0}".format(i), sha=chr(97 + i) * 64,
                       domain="d{0}.com".format(i))
                for i in range(n)
            ]).corroboration_factor()

        assert factor(1) == 0.0
        assert factor(2) == pytest.approx(1 - pow(2.718281828, -CORROBORATION_LAMBDA), abs=0.01)
        assert factor(2) < factor(3) < factor(5) < factor(10)
        assert factor(10) < 1.0
        # Diminishing returns: the 1->2 jump exceeds the 9->10 jump.
        assert (factor(2) - factor(1)) > (factor(10) - factor(9))


class TestIndependentCorroboration:
    """Only a DIFFERENT photograph is independent identity evidence."""

    def test_republished_photo_is_not_corroboration(self):
        results = [
            result(candidate_id="c1", sha="a" * 64, domain="one.com",
                   relation="SAME_PHOTO_REPUBLISHED", phash=2),
            result(candidate_id="c2", sha="b" * 64, domain="two.com",
                   relation="SAME_PHOTO_REPUBLISHED", phash=3),
        ]
        evidence = aggregate(results)
        assert evidence.independent_publisher_count == 0
        assert evidence.corroboration_factor() == 0.0

    def test_different_photo_is_corroboration(self):
        results = [
            result(candidate_id="c1", sha="a" * 64, domain="one.com"),
            result(candidate_id="c2", sha="b" * 64, domain="two.com"),
        ]
        assert aggregate(results).independent_publisher_count == 2

    def test_mixed_relations_count_only_the_independent_ones(self):
        results = [
            result(candidate_id="c1", sha="a" * 64, domain="one.com"),
            result(candidate_id="c2", sha="b" * 64, domain="two.com",
                   relation="SAME_PHOTO_REPUBLISHED", phash=2),
        ]
        evidence = aggregate(results)
        assert evidence.independent_publisher_count == 1
        assert "one.com" in evidence.independent_domains
        assert "two.com" not in evidence.independent_domains


class TestTrustScore:
    def _score(self, results, **kwargs):
        params = dict(metadata_completeness=1.0, acquisition_integrity=1.0)
        params.update(kwargs)
        return score_evidence(aggregate(results), **params)

    def test_score_is_bounded(self):
        results = [
            result(candidate_id="c{0}".format(i), sha=chr(97 + i) * 64,
                   domain="d{0}.com".format(i), probability=0.99, quality=1.0)
            for i in range(12)
        ]
        assert 0.0 <= self._score(results).score <= 100.0

    def test_identity_gates_multiplicatively(self):
        # PRINCIPLE: identity is necessary, not merely contributory. A zero
        # P_id must zero the score no matter how good everything else is.
        results = [
            result(candidate_id="c{0}".format(i), sha=chr(97 + i) * 64,
                   domain="d{0}.com".format(i), probability=0.0, quality=1.0)
            for i in range(8)
        ]
        assert self._score(results).score == pytest.approx(0.0)

    def test_quality_dampens_but_never_eliminates(self):
        results = [result(candidate_id="c1", probability=0.9, quality=0.0)]
        score = self._score(results, metadata_completeness=0.0,
                            acquisition_integrity=0.0)
        # Zero quality still retains the 0.7 floor: a real face on a badly
        # documented page is still a real face.
        assert score.quality_multiplier == pytest.approx(QUALITY_FLOOR)
        assert score.score > 0.0

    def test_corroboration_bonus_is_capped(self):
        many = [
            result(candidate_id="c{0}".format(i), sha=chr(97 + i) * 64,
                   domain="d{0}.com".format(i), probability=0.8)
            for i in range(20)
        ]
        assert self._score(many).corroboration_multiplier <= 1.15

    def test_more_publishers_raises_the_score(self):
        one = [result(candidate_id="c1", domain="a.com", probability=0.8)]
        five = [
            result(candidate_id="c{0}".format(i), sha=chr(97 + i) * 64,
                   domain="d{0}.com".format(i), probability=0.8)
            for i in range(5)
        ]
        assert self._score(five).score > self._score(one).score

    def test_uses_the_strongest_not_the_mean(self):
        # Finding extra weaker evidence must not punish the score.
        strong_only = [result(candidate_id="c1", probability=0.9)]
        strong_plus_weak = strong_only + [
            result(candidate_id="c2", sha="b" * 64, domain="two.com", probability=0.5)
        ]
        assert (
            self._score(strong_plus_weak).identity_probability
            == pytest.approx(self._score(strong_only).identity_probability)
        )

    def test_bands(self):
        assert TrustBand.from_score(95.0) is TrustBand.STRONG
        assert TrustBand.from_score(50.0) is TrustBand.MODERATE
        assert TrustBand.from_score(30.0) is TrustBand.WEAK
        assert TrustBand.from_score(5.0) is TrustBand.INSUFFICIENT

    def test_explanation_recomputes_by_hand(self):
        score = self._score([result(candidate_id="c1", probability=0.8, quality=0.9)])
        expected = (
            100.0
            * score.identity_probability
            * score.quality_multiplier
            * score.corroboration_multiplier
        )
        assert score.score == pytest.approx(expected)
        assert any("T = 100 x" in line for line in score.explain())

    def test_serializes(self):
        payload = self._score([result(candidate_id="c1")]).to_dict()
        assert json.dumps(payload)
        assert payload["formula"]
        assert payload["terms"]["P_id"] is not None


class TestRefusalWithoutCalibration:
    """PRINCIPLE: no score without a calibrated identity probability."""

    def test_refuses_when_probability_is_absent(self):
        results = [result(candidate_id="c1", probability=None)]
        with pytest.raises(UncalibratedScoreRefused, match="calibration model"):
            score_evidence(aggregate(results), metadata_completeness=1.0,
                           acquisition_integrity=1.0)

    def test_refuses_when_there_is_no_verified_evidence(self):
        results = [rejected("c1", "HTTP_ERROR")]
        with pytest.raises(UncalibratedScoreRefused, match="no verified evidence"):
            score_evidence(aggregate(results), metadata_completeness=1.0,
                           acquisition_integrity=1.0)

    def test_refusal_is_an_exception_not_a_zero(self):
        # A zero would be indistinguishable from a real measurement of zero.
        results = [result(candidate_id="c1", probability=None)]
        evidence = aggregate(results)
        assert evidence.has_evidence  # there IS evidence...
        with pytest.raises(UncalibratedScoreRefused):  # ...but no score
            score_evidence(evidence, metadata_completeness=1.0,
                           acquisition_integrity=1.0)


class TestLimitations:
    def test_limitations_are_always_present(self):
        score = score_evidence(
            aggregate([result(candidate_id="c1")]),
            metadata_completeness=1.0, acquisition_integrity=1.0,
        )
        assert score.limitations
        assert any("does not establish" in x.lower() for x in score.limitations)

    def test_single_publisher_is_flagged(self):
        score = score_evidence(
            aggregate([result(candidate_id="c1", domain="only.com")]),
            metadata_completeness=1.0, acquisition_integrity=1.0,
        )
        assert any("one independent publisher" in x.lower() for x in score.limitations)

    def test_absent_source_reputation_is_disclosed(self):
        score = score_evidence(
            aggregate([result(candidate_id="c1")]),
            metadata_completeness=1.0, acquisition_integrity=1.0,
        )
        assert any("reputation" in x.lower() for x in score.limitations)


class TestQualityMeasures:
    def test_metadata_completeness_is_a_ratio(self):
        assert 0.0 <= measure_metadata_completeness([result()]) <= 1.0

    def test_metadata_completeness_is_zero_without_verified(self):
        assert measure_metadata_completeness([rejected("c1", "HTTP_ERROR")]) == 0.0

    def test_acquisition_integrity_penalises_a_redirect(self):
        dirty = result()
        dirty["acquisition"]["url_changed"] = True
        assert measure_acquisition_integrity([dirty]) == 0.0
        assert measure_acquisition_integrity([result()]) == 1.0

    def test_acquisition_integrity_penalises_dishonest_content_type(self):
        dirty = result()
        dirty["validation"]["content_type_was_honest"] = False
        assert measure_acquisition_integrity([dirty]) == 0.0
