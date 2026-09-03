"""A completed search that verified nothing is not a failure.

THE DEFECT THIS FIXES
---------------------
`score_evidence` raised `UncalibratedScoreRefused` for BOTH "there is no
calibration model" and "nothing met the threshold". The runner routed that
into `_fail("scoring", ...)`, so a run that worked perfectly -- 44 candidates
discovered, 23 downloaded, 23 faces detected, 23 embeddings compared -- was
reported to the operator as `status: failed` with every later stage SKIPPED.

Measured on the run that prompted this: highest similarity 0.3240 against a
required 0.3528, with the control probe scoring 0.8394-0.9609 through the
identical pipeline. Nothing was broken. The tool declined to claim a match,
which is the single most important thing it does, and then described itself as
having failed.

WHAT IS AND IS NOT CHANGED
--------------------------
Only the REPORTING of a zero. No threshold moves, no candidate is
reclassified, no score is invented. The tests below assert exactly that.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tracelock.evidence.trust import (
    NoVerifiedEvidence,
    UncalibratedScoreRefused,
)

JS = Path("web/app.js").read_text(encoding="utf-8")


# ==========================================================================
# 1. The two refusals are now distinguishable
# ==========================================================================


class TestRefusalsAreDistinct:
    def test_no_evidence_has_its_own_type(self):
        assert issubclass(NoVerifiedEvidence, UncalibratedScoreRefused)
        assert NoVerifiedEvidence is not UncalibratedScoreRefused

    def test_existing_handlers_still_catch_it(self):
        """Subclassing keeps every prior `except UncalibratedScoreRefused`."""
        with pytest.raises(UncalibratedScoreRefused):
            raise NoVerifiedEvidence("nothing qualified")

    def _empty_bundle(self):
        from tracelock.evidence.aggregate import AggregatedEvidence, FunnelCounts

        return AggregatedEvidence(
            items=(),
            funnel=FunnelCounts(
                discovered=23, downloaded=23, validated=23, analysed=23,
                verified=0, inconclusive=1, rejected=22, duplicates=0,
                unique_images=0, independent_publishers=0,
            ),
            rejection_breakdown={"LOW_FACE_SIMILARITY": 22},
            independent_domains=(),
            all_verified_domains=(),
        )

    def test_zero_verified_raises_the_no_evidence_type(self):
        from tracelock.evidence.trust import score_evidence

        with pytest.raises(NoVerifiedEvidence):
            score_evidence(
                self._empty_bundle(), metadata_completeness=1.0,
                acquisition_integrity=1.0, calibrated=True,
            )

    def test_the_message_says_what_actually_happened(self):
        from tracelock.evidence.trust import score_evidence

        with pytest.raises(NoVerifiedEvidence) as caught:
            score_evidence(
                self._empty_bundle(), metadata_completeness=1.0,
                acquisition_integrity=1.0, calibrated=True,
            )
        assert "every candidate was rejected or inconclusive" in str(caught.value)


# ==========================================================================
# 2. The runner reaches a distinct terminal state
# ==========================================================================


class TestRunnerStateSeparation:
    def _runner_source(self) -> str:
        return Path("src/tracelock/service/runner.py").read_text(encoding="utf-8")

    def test_no_match_is_not_routed_through_fail(self):
        source = self._runner_source()
        block = source[source.index("def _complete_no_match"):]
        block = block[: block.index("def _fail_soft")]

        assert 'self.run.status = "completed_no_match"' in block
        assert "self._fail(" not in block, "a no-match must never call _fail"

    def test_the_scoring_stage_is_marked_done_not_failed(self):
        source = self._runner_source()
        block = source[source.index("def _complete_no_match"):]
        block = block[: block.index("def _fail_soft")]
        assert 'self._done(\n            "scoring"' in block or '_done(\n            "scoring"' in block

    def test_a_real_scoring_failure_still_fails(self):
        """No calibration model is still a failure, and must stay one."""
        source = self._runner_source()
        assert 'self._fail("scoring", "No trust score could be produced."' in source

    def test_no_trust_score_is_fabricated(self):
        source = self._runner_source()
        block = source[source.index("def _complete_no_match"):]
        block = block[: block.index("def _fail_soft")]

        # No score key, and nothing that could render as one.
        assert '"trust_score"' not in block
        assert "score=" not in block

    def test_nothing_is_anchored_without_verified_evidence(self):
        source = self._runner_source()
        block = source[source.index("def _complete_no_match"):]
        block = block[: block.index("def _fail_soft")]
        assert '"anchor": None' in block
        assert 'self._skip("anchor"' in block

    def test_the_summary_reports_only_measured_values(self):
        source = self._runner_source()
        block = source[source.index("def _complete_no_match"):]
        block = block[: block.index("def _fail_soft")]

        for field in (
            "engines_answered", "candidates_discovered", "candidates_examined",
            "candidates_face_analysed", "highest_similarity", "threshold_required",
        ):
            assert field in block, field
        # The highest similarity is taken from results, never invented.
        assert "max(similarities)" in block


# ==========================================================================
# 3. No matching logic was touched
# ==========================================================================


class TestMatchingLogicUnchanged:
    def test_the_calibrated_boundary_is_untouched(self):
        from tracelock.calibration.model import CalibrationModel
        from tracelock.verification.policy import VerificationPolicy

        policy = VerificationPolicy.from_calibration(
            CalibrationModel.load("data/calibration/model.json")
        )
        # The exact values the diagnostics measured against.
        assert round(policy.similarity_floor, 4) == 0.2938
        assert round(policy.similarity_ceiling, 4) == 0.3528

    def test_the_no_match_path_never_reclassifies_a_candidate(self):
        source = Path("src/tracelock/service/runner.py").read_text(encoding="utf-8")
        block = source[source.index("def _complete_no_match"):]
        block = block[: block.index("def _fail_soft")]

        # "NO_VERIFIED_MATCH" contains the word VERIFIED, so check for the
        # patterns that would actually reclassify something.
        for forbidden in (
            'VerificationStatus.VERIFIED', '"status": "VERIFIED',
            "similarity_ceiling =", "similarity_floor =",
            'r["status"] =', "result['status'] =",
        ):
            assert forbidden not in block, forbidden
        assert '"verified": 0' in block, "the count must be reported as zero"

    def test_inconclusive_candidates_are_not_promoted(self):
        """0.3240 was INCONCLUSIVE and must stay that way."""
        from tracelock.calibration.model import CalibrationModel
        from tracelock.verification.policy import VerificationPolicy

        policy = VerificationPolicy.from_calibration(
            CalibrationModel.load("data/calibration/model.json")
        )
        observed_best = 0.3240
        assert observed_best < policy.similarity_ceiling
        assert observed_best >= policy.similarity_floor
        assert policy.band_for(observed_best).value != "HIGH"


# ==========================================================================
# 4. The UI presents it as completed, not broken
# ==========================================================================


class TestNoMatchUi:
    def test_the_state_routes_to_results_not_the_failbox(self):
        assert 'snapshot.status === "completed_no_match"' in JS
        block = JS[JS.index('snapshot.status === "completed_no_match"'):]
        block = block[: block.index('snapshot.status === "complete"')]
        assert "renderNoMatch" in block
        assert 'show("screen-results")' in block
        assert "failbox" not in block

    def test_it_leads_with_a_completed_status(self):
        block = JS[JS.index("function renderNoMatch"):]
        block = block[: block.index("function renderResults")]
        assert "LIVE SEARCH COMPLETED" in block
        assert "status ok" in block

    def test_it_states_the_outcome_plainly(self):
        block = JS[JS.index("function renderNoMatch"):]
        block = block[: block.index("function renderResults")]
        assert "No verified same-person match found" in block

    def test_it_shows_the_numbers_a_judge_needs(self):
        block = JS[JS.index("function renderNoMatch"):]
        block = block[: block.index("function renderResults")]
        for label in ("Engines answered", "Discovered", "Examined",
                      "Face-analysed", "Highest similarity", "Threshold required"):
            assert label in block, label

    def test_it_explains_similarity_is_not_identity(self):
        """The sentence is built at runtime, so assert on the VALUE.

        Source-grepping would miss it: the string is written as two adjacent
        literals and only becomes one phrase when Python joins them.
        """
        runner = Path("src/tracelock/service/runner.py").read_text(encoding="utf-8")
        collapsed = " ".join(runner.split()).replace('" "', "")
        assert "not automatically treated as identity matches" in collapsed

        block = JS[JS.index("function renderNoMatch"):]
        block = block[: block.index("function renderResults")]
        assert "esc(m.note" in block, "the view must render the backend note"
        assert "look</em> alike" in block

    def test_it_never_shows_a_trust_score_or_anchor(self):
        block = JS[JS.index("function renderNoMatch"):]
        block = block[: block.index("function renderResults")]
        assert "trust_score" not in block
        assert "both require" in block  # the explicit denial

    def test_the_word_failed_never_appears_in_this_view(self):
        block = JS[JS.index("function renderNoMatch"):]
        block = block[: block.index("function renderResults")]
        assert not re.search(r"\bfailed\b", block, re.IGNORECASE)

    def test_it_offers_a_way_forward(self):
        block = JS[JS.index("function renderNoMatch"):]
        block = block[: block.index("function renderResults")]
        assert "Investigate another image" in block

    def test_the_verified_match_view_is_unchanged(self):
        """renderResults must still handle the success path exactly as before."""
        block = JS[JS.index("function renderResults"):]
        assert "What was investigated?" in block
        assert "What was verified?" in block
        assert "liveSearchProof" in block


# ==========================================================================
# 5. Anchoring a no-match needs no schema change (audited, not implemented)
# ==========================================================================


class TestFingerprintToleratesAnEmptyBundle:
    """Recorded because it settles whether anchoring a no-match is risky.

    It is not: every leaf already degrades safely. Anchoring is withheld here
    by CHOICE, not by a schema limit -- there is no verified evidence, so there
    is nothing whose integrity is worth committing.
    """

    def test_every_leaf_extractor_tolerates_missing_evidence(self):
        source = Path("src/tracelock/chain/fingerprint.py").read_text(encoding="utf-8")

        assert 'artifact.get("trust_score") or {}' in source
        assert 'evidence.get("items", [])' in source
        assert 'evidence.get("independent_domains", [])' in source

    def test_a_zero_score_stays_inside_the_contract_range(self):
        source = Path("src/tracelock/chain/fingerprint.py").read_text(encoding="utf-8")
        assert "max(0, min(10000," in source

        contract = Path("contracts/EvidenceNotary.sol").read_text(encoding="utf-8")
        assert "trustScoreBp > 10000" in contract  # 0 is valid
