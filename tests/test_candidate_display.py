"""The candidate evidence gallery -- and the boundary that keeps it safe.

WHY THIS EXISTS
---------------
"No match found" is a weak thing to show a judge. "We found 25 visually
similar faces, downloaded 23, measured every one, rejected 22 and refused to
call the 23rd a match" is the actual product. All of that was already in the
API response and simply was not rendered.

THE TWO LINES THIS FILE DEFENDS
-------------------------------
1. TRUTH. Displaying a rejected candidate must never edge it toward being a
   match. The verdict shown is the pipeline's own `status` and
   `rejection_reasons[0]`, the similarity is the measured one, the threshold
   is quoted from the run's own policy, and no threshold moves.

2. SCOPE. Candidate media is served per-investigation. The CAS is shared by
   every run the process has ever performed, so an unscoped route would let
   anyone holding a hash read another run's evidence. Path safety alone does
   not prevent that -- the digest must also belong to the run being asked.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

JS = Path("web/app.js").read_text(encoding="utf-8")
CSS = Path("web/styles.css").read_text(encoding="utf-8")
API = Path("src/tracelock/api/app.py").read_text(encoding="utf-8")

ORDER = ["analysedCandidates", "candidateKind", "candidateVerdict",
         "candidateCard", "candidateGroup", "candidateVerificationSection",
         "renderNoMatch"]


def block(name: str) -> str:
    """Source of one function, up to whichever function follows it."""
    start = JS.index("function {0}(".format(name))
    nxt = ORDER[ORDER.index(name) + 1]
    return JS[start : JS.index("function {0}(".format(nxt), start)]


@pytest.fixture
def client():
    from tracelock.api import create_app

    return TestClient(create_app())


@pytest.fixture
def stored_blob():
    blobs = [b for b in Path("data/cas/blobs").rglob("*") if b.is_file()]
    if not blobs:
        pytest.skip("no CAS blobs on this machine")
    return blobs[0].name


@pytest.fixture
def run_with(stored_blob):
    """Register a fake run owning exactly the digests given."""
    from tracelock.service.runner import STORE

    created = []

    def make(run_id, digests):
        result = {
            "verification": {"results": [{"content_sha256": d} for d in digests]}
        }
        # snapshot() matters: "/candidate/.." normalises to the run's own
        # status URL, so the fake must answer there like a real run does.
        run = SimpleNamespace(
            result=result,
            snapshot=lambda: {"run_id": run_id, "status": "complete",
                              "result": result},
        )
        STORE._runs[run_id] = run
        created.append(run_id)
        return run_id

    yield make
    for run_id in created:
        STORE._runs.pop(run_id, None)


# ==========================================================================
# 1. Only genuinely analysed candidates become cards
# ==========================================================================


class TestOnlyAnalysedCandidatesAppear:
    def test_selection_requires_a_measured_similarity(self):
        source = block("analysedCandidates")
        assert "face_similarity !== null" in source
        assert "face_similarity !== undefined" in source

    def test_a_download_failure_is_counted_never_carded(self):
        """HTTP_ERROR means no comparison happened -- a card would imply one."""
        source = block("candidateVerificationSection")
        assert "notRetrievable = all.length - analysed.length" in source
        assert "no comparison was performed and none is implied" in source

    def test_the_rejected_list_is_bounded(self):
        source = block("candidateVerificationSection")
        assert "rejected.slice(0, 10)" in source

    def test_every_group_is_ranked_by_similarity(self):
        source = block("candidateVerificationSection")
        assert "b.face_similarity - a.face_similarity" in source
        assert source.count(".sort(byScore)") == 3

    def test_nothing_renders_when_nothing_was_analysed(self):
        source = block("candidateVerificationSection")
        assert 'if (!analysed.length) return "";' in source


# ==========================================================================
# 2. A rejection stays a rejection
# ==========================================================================


class TestRejectionsStayRejections:
    def test_the_group_is_chosen_by_pipeline_status(self):
        source = block("candidateKind")
        assert 'r.status === "VERIFIED_CANDIDATE"' in source
        assert 'r.status === "INCONCLUSIVE"' in source

    def test_the_verdict_comes_from_the_pipeline_not_the_view(self):
        source = block("candidateVerdict")
        assert "rejection_reasons" in source
        assert "reason.explanation" in source

    def test_only_the_verified_branch_says_same_person(self):
        source = block("candidateVerdict")
        assert source.count("VERIFIED SAME PERSON") == 1
        verified_branch = source[: source.index('kind === "inconclusive"')]
        assert "VERIFIED SAME PERSON" in verified_branch

    def test_inconclusive_is_never_called_a_match(self):
        source = block("candidateVerdict")
        assert "INCONCLUSIVE" in source
        assert "insufficient evidence " in source

    def test_a_rejection_names_the_pipeline_reason_code(self):
        source = block("candidateVerdict")
        assert '"REJECTED — " + code.replace(/_/g, " ")' in source

    def test_the_similarity_shown_is_the_measured_one(self):
        for name in ("candidateVerdict", "candidateCard"):
            source = block(name)
            for forbidden in ("* 100", "Math.max", "Math.round(", "|| 0.5"):
                assert forbidden not in source, "{0}: {1}".format(name, forbidden)
        assert "r.face_similarity.toFixed(4)" in block("candidateCard")

    def test_the_threshold_is_quoted_from_the_run_not_hardcoded(self):
        source = block("candidateVerificationSection")
        assert "policy.similarity_ceiling" in source
        for name in ("candidateVerdict", "candidateCard",
                     "candidateVerificationSection"):
            assert "0.3528" not in block(name), name
            assert "0.2938" not in block(name), name

    def test_the_rejected_note_states_the_measurement_happened(self):
        source = " ".join(block("candidateVerificationSection").split())
        assert "downloaded, and " in source
        assert "measured against your face" in source
        assert "none met the calibrated same-person " in source


# ==========================================================================
# 3. Matching logic is untouched
# ==========================================================================


class TestMatchingUnchanged:
    def test_the_calibrated_boundary_is_unchanged(self):
        from tracelock.calibration.model import CalibrationModel
        from tracelock.verification.policy import VerificationPolicy

        policy = VerificationPolicy.from_calibration(
            CalibrationModel.load("data/calibration/model.json")
        )
        assert round(policy.similarity_floor, 4) == 0.2938
        assert round(policy.similarity_ceiling, 4) == 0.3528

    def test_the_view_never_assigns_a_verdict_field(self):
        """Assignment only -- `status === "X"` is a comparison and is fine.

        A naive "status =" substring matches `status ===`, which is exactly
        the read the view is SUPPOSED to do. The control below keeps the
        pattern honest: a regex typo would make this pass vacuously.
        """
        fields = ("status", "face_similarity", "identity_probability",
                  "similarity_band", "similarity_ceiling", "similarity_floor")

        def offenders(text):
            hits = []
            for field in fields:
                pattern = r"\b{0}\s*=(?![=>])".format(field)
                hits += re.findall(r".{0,40}" + pattern + r".{0,20}", text)
            return hits

        control = ('rejected.forEach((r) => { r.status = "VERIFIED_CANDIDATE"; '
                   'r.face_similarity = 0.99; });')
        assert offenders(control), "the detector itself is broken"
        assert not offenders('r.status === "VERIFIED_CANDIDATE"')

        for name in ("candidateKind", "candidateVerdict", "candidateCard",
                     "candidateGroup", "candidateVerificationSection"):
            found = offenders(block(name))
            assert not found, "{0} mutates a verdict: {1}".format(name, found)

    def test_the_snapshot_view_does_not_reclassify(self):
        source = API[API.index("def _public_snapshot"):]
        source = source[: source.index('@api.get("/investigation/{run_id}")')]
        assert 'k != "cas_path"' in source
        assert 'clean["platform"]' in source
        for forbidden in ('clean["status"]', 'clean["face_similarity"]',
                          'clean["identity_probability"]'):
            assert forbidden not in source, forbidden


# ==========================================================================
# 4. Candidate media is scoped to its own investigation
# ==========================================================================


class TestCandidateMediaIsRunScoped:
    def test_the_unscoped_route_is_gone(self, client):
        paths = client.app.openapi()["paths"]
        assert "/api/candidate/{content_sha256}" not in paths
        assert "/api/investigation/{run_id}/candidate/{content_sha256}" in paths

    def test_a_run_can_read_its_own_candidate(self, client, run_with, stored_blob):
        run_with("runowner", [stored_blob])
        response = client.get("/api/investigation/runowner/candidate/" + stored_blob)
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert len(response.content) > 0

    def test_another_run_cannot_read_it(self, client, run_with, stored_blob):
        """The point: the bytes exist and are readable by their owner."""
        run_with("runowner", [stored_blob])
        run_with("runother", [])
        assert client.get(
            "/api/investigation/runowner/candidate/" + stored_blob
        ).status_code == 200
        assert client.get(
            "/api/investigation/runother/candidate/" + stored_blob
        ).status_code == 404

    def test_an_unknown_run_is_refused(self, client, stored_blob):
        assert client.get(
            "/api/investigation/nosuchrun/candidate/" + stored_blob
        ).status_code == 404

    def test_a_malformed_digest_is_refused(self, client, run_with):
        run_with("runowner", [])
        assert client.get(
            "/api/investigation/runowner/candidate/abc"
        ).status_code == 400

    def test_traversal_cannot_escape_the_store(self, client, run_with):
        """The property is "no arbitrary file comes back", not "4xx".

        A bare ".." normalises away to /api/investigation/runowner -- the
        run's OWN status endpoint, which legitimately answers 200 with JSON.
        That is URL normalisation landing on another route, not a traversal
        reaching the filesystem, so the assertion is on what comes back.
        """
        run_with("runowner", [])
        for probe in ("..%2F..%2Fetc%2Fpasswd", "..", "%2e%2e%2f" * 4,
                      "0" * 64 + ".json", "....//....//etc/passwd"):
            response = client.get(
                "/api/investigation/runowner/candidate/" + probe
            )
            assert not response.headers["content-type"].startswith("image/"), probe
            if response.status_code == 200:
                # Only the run's own JSON status may legitimately answer here.
                assert response.json().get("run_id") == "runowner", probe

    def test_scope_is_checked_before_the_filesystem_answers(self):
        """Otherwise the 404-vs-200 split leaks what other runs hold."""
        source = API[API.index("async def candidate_image"):]
        cut = source.index("\ndef ") if "\ndef " in source else len(source)
        source = source[:cut]
        assert source.index("_run_candidate_digests") < source.index("path.is_file()")


# ==========================================================================
# 5. No filesystem paths on the wire
# ==========================================================================


class TestNoPathLeaks:
    def test_cas_path_is_stripped_from_the_snapshot(self):
        source = API[API.index("def _public_snapshot"):]
        assert 'k != "cas_path"' in source

    def test_the_stored_result_is_not_mutated(self):
        """Scrubbing in place would corrupt the run and the artifact."""
        source = API[API.index("def _public_snapshot"):]
        source = source[: source.index('@api.get("/investigation/{run_id}")')]
        assert "clean = {k: v for k, v in entry.items()" in source
        assert '"verification": {**verification, "results": public_results}' in source
        for forbidden in ("entry.pop(", "del entry[", "results[i] ="):
            assert forbidden not in source, forbidden

    def test_both_delivery_paths_use_the_scrubbed_view(self):
        """A raw snapshot on the socket would bypass the scrub entirely."""
        assert "send_json(run.snapshot())" not in API
        assert "return run.snapshot()" not in API
        assert "_public_snapshot(run)" in API

    def test_the_evidence_bundle_keeps_its_cas_path_deliberately(self):
        """A documented exception, not an oversight.

        `evidence.items[].cas_path` stays on the wire. It is a RELATIVE store
        path ("data/cas/blobs/.."), never an absolute one, and the eleven-leaf
        fingerprint hashes `evidence["items"]` WHOLESALE as `tl:evidence_items`
        -- so removing the field from the served bundle would make that bundle
        no longer reproduce the anchored Merkle root. The tamper-evidence
        claim is worth more than hiding a relative path the UI never reads.

        This test proves the coupling, so the trade-off stays visible.
        """
        import copy
        import glob
        import json

        from tracelock.chain.fingerprint import fingerprint_evidence

        for path in sorted(glob.glob("data/runs/*.json")):
            if "anchor" in path:
                continue
            try:
                artifact = json.load(open(path, encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            items = (artifact.get("evidence") or {}).get("items") or []
            if items and any("cas_path" in i for i in items):
                break
        else:
            pytest.skip("no artifact with cas_path items on this machine")

        stripped = copy.deepcopy(artifact)
        for item in stripped["evidence"]["items"]:
            item.pop("cas_path", None)

        assert (fingerprint_evidence(artifact).merkle_root
                != fingerprint_evidence(stripped).merkle_root)

    def test_only_relative_store_paths_are_ever_served(self):
        """Absolute paths are the thing actually forbidden, and never appear."""
        from tracelock.acquisition.cas import ContentAddressedStore

        path = ContentAddressedStore("data/cas").blob_path("a" * 64)
        assert not path.is_absolute()

    def test_the_ui_never_references_a_filesystem_path(self):
        for name in ("candidateCard", "candidateVerificationSection"):
            source = block(name)
            assert "cas_path" not in source, name
            assert "data/cas" not in source, name


# ==========================================================================
# 6. Provenance is shown, and never invented
# ==========================================================================


class TestProvenanceIsReal:
    def test_the_card_shows_the_recorded_provenance(self):
        source = block("candidateCard")
        for field in ("provenance", "registrable_domain", "source_url",
                      "content_sha256", "identity_probability",
                      "similarity_band", "r.platform"):
            assert field in source, field

    def test_platform_reuses_the_existing_classifier(self):
        source = API[API.index("def _public_snapshot"):]
        source = source[: source.index('@api.get("/investigation/{run_id}")')]
        assert "classify_source" in source
        assert "SourceCategory.SOCIAL" in source

    def test_platform_is_none_when_not_social(self):
        from tracelock.ingest.social import SourceCategory, classify_source

        assert classify_source("https://github.com/x").category is not SourceCategory.SOCIAL
        assert classify_source("https://in.linkedin.com/in/a").platform == "LinkedIn"
        assert classify_source("https://reddit.com/r/a").platform == "Reddit"

    def test_a_missing_source_is_stated_not_faked(self):
        assert "no source page recorded" in block("candidateCard")

    def test_a_missing_thumbnail_degrades_rather_than_faking_one(self):
        source = block("candidateCard")
        assert "onerror" in source
        assert "nothumb" in source
        assert "image<br>unavailable" in source
        assert ".cand-card.nothumb .cc-fallback  { display: block; }" in CSS

    def test_a_candidate_without_stored_bytes_still_renders(self):
        source = block("candidateCard")
        assert "r.content_sha256 && runId" in source
        assert 'thumb ? "" : " nothumb"' in source


# ==========================================================================
# 7. Both result screens, and the funnel
# ==========================================================================


class TestBothScreensShowIt:
    def test_the_summary_is_derived_from_the_results_themselves(self):
        """Deriving it means the counts can never contradict the cards."""
        source = block("candidateVerificationSection")
        for label in ("discovered", "face-analysed", "verified",
                      "inconclusive", "rejected", "not retrievable"):
            assert '"{0}"'.format(label) in source, label

    def test_the_no_match_screen_explains_why(self):
        source = JS[JS.index("function renderNoMatch"):]
        source = source[: source.index("function renderResults")]
        assert "candidateVerificationSection" in source
        assert "Why these results were rejected" in source

    def test_the_verified_screen_shows_it_too(self):
        source = JS[JS.index("function renderResults"):]
        assert "candidateVerificationSection(result, {})" in source

    def test_the_no_match_screen_keeps_its_own_summary(self):
        source = JS[JS.index("function renderNoMatch"):]
        source = source[: source.index("function renderResults")]
        assert "LIVE SEARCH COMPLETED" in source
        assert "Highest similarity" in source
        assert "Threshold required" in source

    def test_the_social_evidence_is_not_displaced(self):
        """Requirement 2's proof must survive alongside the new section."""
        source = JS[JS.index("function renderResults"):]
        assert "liveSearchProof" in source
        assert "What was verified?" in source

    def test_the_run_id_is_tracked_for_thumbnail_urls(self):
        """activeRunId is already null by render time; this is the fix."""
        assert "let resultRunId = null;" in JS
        assert JS.count("resultRunId = snapshot.run_id;") == 3
        assert "const runId = resultRunId;" in block("candidateVerificationSection")

    def test_the_three_states_are_visually_distinct(self):
        for kind in ("verified", "inconclusive", "rejected"):
            assert ".cand-card.{0}".format(kind) in CSS, kind
            assert ".cc-verdict.{0}".format(kind) in CSS, kind
