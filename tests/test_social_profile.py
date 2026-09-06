"""Tier-1 public profile evidence: the ten properties this feature must hold.

WHY THIS FILE EXISTS
---------------------
`social_profile` answers a genuinely new and riskier question than anything
else in this codebase: not "does this photo match", but "does this page
belong to an account". Every property below defends against one specific way
that question could be answered dishonestly -- a rejected candidate leaking
through, a bare URL masquerading as a verified claim, an empty chain behind a
confident-sounding tier, or a UI showing the wrong badge for the wrong tier.

None of these are hypothetical. `LOW_FACE_SIMILARITY` matching `"VERIFIED"`
as a substring, and `"verified_candidate"` (lowercase) silently failing to
match `VerificationStatus.VERIFIED_CANDIDATE.value` (uppercase), are both
real defects this project has already shipped and fixed. The tests here are
written to fail loudly if this feature repeats either mistake.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tracelock.core.reasons import VerificationStatus
from tracelock.social_profile import ProfileTier, extract_hints, relate
from tracelock.social_profile.models import EvidenceChainStep, ProfileRelationship
from tracelock.social_profile.relate import _classify

JS = Path("web/app.js").read_text(encoding="utf-8")
FINGERPRINT_SRC = Path("src/tracelock/chain/fingerprint.py").read_text(encoding="utf-8")
TRUST_SRC = Path("src/tracelock/evidence/trust.py").read_text(encoding="utf-8")
AGGREGATE_SRC = Path("src/tracelock/evidence/aggregate.py").read_text(encoding="utf-8")
RELATE_SRC = Path("src/tracelock/social_profile/relate.py").read_text(encoding="utf-8")
HINTS_SRC = Path("src/tracelock/social_profile/hints.py").read_text(encoding="utf-8")
MODELS_SRC = Path("src/tracelock/social_profile/models.py").read_text(encoding="utf-8")


def _result(
    candidate_id="cand-001",
    status="VERIFIED_CANDIDATE",
    source_url="https://x.com/janedoe",
    face_similarity=0.91,
    identity_probability=0.97,
    discovery_metadata=None,
    content_sha256="a" * 64,
    registrable_domain=None,
):
    """A minimal, valid-shaped VerificationResult.to_dict() fixture."""
    from urllib.parse import urlsplit

    domain = registrable_domain or urlsplit(source_url).hostname or ""
    return {
        "candidate_id": candidate_id,
        "provider": "serpapi",
        "rank": 1,
        "source_url": source_url,
        "media_url": source_url,
        "status": status,
        "rejection_reasons": [],
        "content_sha256": content_sha256,
        "cas_path": "data/cas/blobs/{0}/{1}".format(content_sha256[:2], content_sha256),
        "phash": None,
        "face_similarity": face_similarity,
        "similarity_band": "HIGH" if face_similarity and face_similarity >= 0.35 else "LOW",
        "identity_probability": identity_probability,
        "provenance": {"registrable_domain": domain, "host": domain, "url": source_url},
        "face": {"quality_aggregate": 0.9},
        "relation": {"relation": "SAME_PERSON_DIFFERENT_PHOTO"},
        "acquisition": {"ok": True},
        "validation": {"ok": True},
        "discovery_metadata": discovery_metadata,
    }


# ==========================================================================
# 1. Rejected candidates can NEVER generate profile relationships
# ==========================================================================


class TestRejectedNeverGeneratesRelationships:
    def test_a_rejected_candidate_with_a_perfect_profile_url_produces_nothing(self):
        rejected = _result(
            status="REJECTED",
            source_url="https://x.com/janedoe",
            discovery_metadata={"title": "Jane Doe (@janedoe) / X"},
        )
        summary = relate([rejected])
        assert summary.relationships == ()
        assert summary.statement == "No verified public profile relationship found."

    def test_an_inconclusive_candidate_produces_nothing(self):
        inconclusive = _result(
            status="INCONCLUSIVE",
            source_url="https://www.linkedin.com/in/johndoe/",
        )
        assert relate([inconclusive]).relationships == ()

    def test_rejected_and_verified_together_only_the_verified_one_counts(self):
        rejected = _result(candidate_id="c1", status="REJECTED",
                            source_url="https://x.com/lookalike")
        verified = _result(candidate_id="c2", status="VERIFIED_CANDIDATE",
                            source_url="https://x.com/realmatch")
        summary = relate([rejected, verified])
        assert len(summary.relationships) == 1
        assert summary.relationships[0].source_candidate_id == "c2"
        assert summary.candidates_considered == 1  # rejected is never "considered"

    def test_the_gate_runs_before_hint_extraction_not_after(self):
        """Structural: the status check must precede extract_hints in source
        order, so a rejected candidate's metadata is never even inspected,
        not merely discarded afterward."""
        body = RELATE_SRC[RELATE_SRC.index("def relate("):]
        gate_pos = body.index("!= VerificationStatus.VERIFIED_CANDIDATE.value")
        extract_pos = body.index("extract_hints(result)")
        assert gate_pos < extract_pos

    def test_extract_hints_itself_has_no_status_gate_by_design(self):
        """extract_hints is a pure extractor with no opinion on verification
        status -- the SOLE gate lives in relate(), in one place. This test
        documents that split so a future edit cannot add a second,
        potentially inconsistent gate inside hints.py.

        Checks for an actual GATING comparison, not the bare word "status" --
        which legitimately appears in this function's own docstring prose.
        """
        body = HINTS_SRC[HINTS_SRC.index("def extract_hints"):]
        for forbidden in ('result["status"]', 'result.get("status")',
                           '== "REJECTED"', '!= "VERIFIED_CANDIDATE"',
                           "VerificationStatus"):
            assert forbidden not in body, forbidden


# ==========================================================================
# 2. A profile URL alone cannot automatically become VERIFIED_HIGH_CONFIDENCE
# ==========================================================================


class TestProfileUrlAloneIsNotEnough:
    def test_profile_shaped_url_on_unverified_candidate_yields_nothing(self):
        for status in ("REJECTED", "INCONCLUSIVE"):
            result = _result(status=status, source_url="https://x.com/janedoe")
            assert relate([result]).relationships == (), status

    def test_the_top_tier_requires_verified_status_as_well_as_url_shape(self):
        verified = _result(status="VERIFIED_CANDIDATE", source_url="https://x.com/janedoe")
        rel = relate([verified]).relationships[0]
        assert rel.tier == ProfileTier.VERIFIED_HIGH_CONFIDENCE
        # Same URL, same hint-extraction outcome, different status -> nothing.
        unverified = _result(status="INCONCLUSIVE", source_url="https://x.com/janedoe")
        assert relate([unverified]).relationships == ()

    def test_classify_is_not_publicly_exported(self):
        """The tier-decision function is private; only relate() -- which
        carries the status gate -- is part of the public surface."""
        import tracelock.social_profile as pkg

        assert "_classify" not in pkg.__all__
        assert not hasattr(pkg, "_classify")

    def test_two_disagreeing_hints_do_not_promote_to_a_higher_tier(self):
        """A URL-derived handle and a title-derived handle that DISAGREE must
        not be averaged, guessed, or upgraded -- only exact agreement between
        independent methods promotes a tier."""
        result = _result(
            status="VERIFIED_CANDIDATE",
            source_url="https://x.com/realhandle/status/123",
            discovery_metadata={"title": "Someone Else (@totallydifferent) / X"},
        )
        rel = relate([result]).relationships[0]
        assert rel.tier == ProfileTier.DISCOVERED_LINK


# ==========================================================================
# 3. Every VERIFIED_HIGH_CONFIDENCE result has a non-empty evidence chain
# ==========================================================================


class TestVerifiedHighConfidenceAlwaysHasAChain:
    def test_constructing_one_with_an_empty_chain_raises(self):
        with pytest.raises(ValueError):
            ProfileRelationship(
                tier=ProfileTier.VERIFIED_HIGH_CONFIDENCE,
                platform="X", handle="janedoe", profile_url="https://x.com/janedoe",
                source_candidate_id="c1", source_url="https://x.com/janedoe",
                face_similarity=0.9, evidence_chain=(),
            )

    def test_a_real_verified_high_confidence_result_has_a_populated_chain(self):
        result = _result(status="VERIFIED_CANDIDATE", source_url="https://x.com/janedoe")
        rel = relate([result]).relationships[0]
        assert rel.tier == ProfileTier.VERIFIED_HIGH_CONFIDENCE
        assert len(rel.evidence_chain) >= 1
        assert all(isinstance(s, EvidenceChainStep) for s in rel.evidence_chain)

    def test_every_tier_in_practice_carries_a_chain_not_just_the_top_one(self):
        """The __post_init__ guard applies to ALL tiers, not only the top
        one -- assert this holds for a real DISCOVERED_LINK result too."""
        result = _result(
            status="VERIFIED_CANDIDATE",
            source_url="https://www.tiktok.com/@someone/video/998877",
        )
        rel = relate([result]).relationships[0]
        assert rel.tier == ProfileTier.DISCOVERED_LINK
        assert len(rel.evidence_chain) >= 1


# ==========================================================================
# 4. Empty/ambiguous metadata -> UNVERIFIED or nothing, never a false VERIFIED
# ==========================================================================


class TestAmbiguousMetadataNeverFalselyVerifies:
    def test_no_metadata_and_a_non_shaped_url_produces_nothing(self):
        result = _result(
            status="VERIFIED_CANDIDATE",
            source_url="https://www.instagram.com/p/Cxyz123abc/",
            discovery_metadata=None,
        )
        assert relate([result]).relationships == ()

    def test_garbage_title_text_produces_no_hint(self):
        result = _result(
            status="VERIFIED_CANDIDATE",
            source_url="https://www.instagram.com/p/Cxyz123abc/",
            discovery_metadata={"title": "completely unrelated caption text"},
        )
        assert relate([result]).relationships == ()

    def test_a_non_social_url_is_never_a_source_of_any_relationship(self):
        result = _result(
            status="VERIFIED_CANDIDATE",
            source_url="https://randomnewsblog.example/article/42",
            discovery_metadata={"title": "Some News Article | RandomNewsBlog"},
        )
        assert extract_hints(result) == []
        assert relate([result]).relationships == ()

    def test_disagreeing_signals_land_at_discovered_not_unverified_or_verified(self):
        result = _result(
            status="VERIFIED_CANDIDATE",
            source_url="https://www.linkedin.com/posts/johndoe_announcement-activity-1/",
            discovery_metadata={"title": "John Doe | LinkedIn"},
        )
        rel = relate([result]).relationships[0]
        # "johndoe" (url slug) vs "John Doe" (display name) do not string-match
        # -- conservative by design, never fuzzy-matched into agreement.
        assert rel.tier == ProfileTier.DISCOVERED_LINK


# ==========================================================================
# 5. Zero additional network calls
# ==========================================================================


class TestZeroNetworkCalls:
    def test_no_networking_module_is_imported_anywhere_in_the_package(self):
        forbidden = ("httpx", "requests", "socket", "urllib.request", "aiohttp")
        for path, src in (
            ("hints.py", HINTS_SRC), ("relate.py", RELATE_SRC), ("models.py", MODELS_SRC),
        ):
            for name in forbidden:
                assert not re.search(r"^\s*(import|from)\s+{0}\b".format(re.escape(name)),
                                      src, re.MULTILINE), (path, name)

    def test_socket_creation_is_never_attempted_during_a_full_run(self, monkeypatch):
        import socket as socket_module

        def _refuse(*a, **k):
            raise AssertionError("social_profile attempted to open a socket")

        monkeypatch.setattr(socket_module, "socket", _refuse)

        results = [
            _result(candidate_id="c1", source_url="https://x.com/janedoe",
                    discovery_metadata={"title": "Jane Doe (@janedoe) / X"}),
            _result(candidate_id="c2", status="REJECTED",
                    source_url="https://instagram.com/p/abc123/"),
            _result(candidate_id="c3", status="INCONCLUSIVE",
                    source_url="https://www.linkedin.com/in/someone/"),
            _result(candidate_id="c4", source_url="https://www.tiktok.com/@x/video/1"),
        ]
        summary = relate(results)  # must not raise
        assert summary.candidates_considered == 2  # c1 and c4 are VERIFIED; c2/c3 are not


# ==========================================================================
# 7. discovered_profiles does not affect the trust score
# ==========================================================================


class TestTrustScoreUnaffected:
    def test_neither_file_references_the_new_feature(self):
        assert "discovered_profiles" not in TRUST_SRC
        assert "social_profile" not in TRUST_SRC
        assert "discovered_profiles" not in AGGREGATE_SRC
        assert "social_profile" not in AGGREGATE_SRC

    def test_the_same_evidence_scores_identically_with_and_without_profile_data(self):
        from tracelock.evidence.aggregate import aggregate
        from tracelock.evidence.trust import score_evidence

        with_profile = _result(
            source_url="https://x.com/janedoe",  # profile-shaped
            discovery_metadata={"title": "Jane Doe (@janedoe) / X"},
            registrable_domain="x.com",
        )
        without_profile = _result(
            source_url="https://x.com/janedoe/status/999",  # post-shaped
            discovery_metadata=None,
            registrable_domain="x.com",
        )

        score_a = score_evidence(
            aggregate([with_profile]),
            metadata_completeness=1.0, acquisition_integrity=1.0, calibrated=True,
        )
        score_b = score_evidence(
            aggregate([without_profile]),
            metadata_completeness=1.0, acquisition_integrity=1.0, calibrated=True,
        )
        assert score_a.to_dict() == score_b.to_dict()


# ==========================================================================
# 8. discovered_profiles does not modify blockchain fingerprints / leaves
# ==========================================================================


class TestFingerprintUnaffected:
    def test_fingerprint_source_never_references_the_new_feature(self):
        assert "discovered_profiles" not in FINGERPRINT_SRC
        assert "social_profile" not in FINGERPRINT_SRC
        assert '"verification"' not in FINGERPRINT_SRC

    def test_leaf_tags_are_exactly_the_original_eleven(self):
        from tracelock.chain.fingerprint import LEAF_TAGS

        assert LEAF_TAGS == (
            "tl:schema", "tl:run_id", "tl:probe_commitment", "tl:probe_model",
            "tl:evidence_items", "tl:evidence_funnel", "tl:independent_domains",
            "tl:trust_score", "tl:verification_policy", "tl:source_artifacts",
            "tl:pipeline",
        )
        assert len(LEAF_TAGS) == 11
        assert not any("profile" in tag for tag in LEAF_TAGS)

    def test_the_merkle_root_is_identical_whether_or_not_the_key_is_present(self):
        from tracelock.chain.fingerprint import fingerprint_evidence

        base_artifact = {
            "schema_version": "verification-run/1",
            "run_id": "run-abc123",
            "probe": {"sha256": "f" * 64, "embedding_quantized_sha256": "e" * 64,
                      "model_id": "buffalo_l", "embedding_dimension": 512},
            "evidence": {"items": [], "funnel": {}, "independent_domains": []},
            "trust_score": {"score": 0, "band": "NONE"},
            "verification_policy": {"calibrated": True},
            "source_artifacts": {},
        }
        without_key = dict(base_artifact)
        with_key = dict(base_artifact)
        with_key["verification"] = {
            "discovered_profiles": {
                "relationships": [{"tier": "VERIFIED_HIGH_CONFIDENCE",
                                   "profile_url": "https://x.com/janedoe"}],
                "statement": "1 public profile relationship(s) found.",
            }
        }

        root_without = fingerprint_evidence(without_key).merkle_root
        root_with = fingerprint_evidence(with_key).merkle_root
        assert root_without == root_with


# ==========================================================================
# 9 & 10. The UI distinguishes all three categories, and cannot mislabel one
# ==========================================================================


class TestUiCategoriesAreDistinctAndCannotBeMislabelled:
    def test_all_three_backend_tier_values_have_a_ui_entry(self):
        """Ties the two sides together deliberately -- this project has
        already shipped one enum-casing mismatch between backend and UI
        (StopTracker vs VerificationStatus.value); this test exists so that
        class of bug cannot recur here silently."""
        block = JS[JS.index("const PROFILE_TIER_META"):JS.index("function profileTierMeta")]
        for tier in ProfileTier:
            # Object keys here are bare JS identifiers (DISCOVERED_LINK: {...}),
            # not quoted strings -- matched as a key, not a string literal.
            assert re.search(r"\b{0}\s*:".format(re.escape(tier.value)), block), tier.value

    def test_the_three_ui_labels_and_icons_are_distinct(self):
        block = JS[JS.index("const PROFILE_TIER_META"):JS.index("function profileTierMeta")]
        for icon in ("✓", "?", "○"):
            assert icon in block
        assert block.count("label:") == 3
        labels = re.findall(r'label:\s*"([^"]+)"', block)
        assert len(set(labels)) == 3

    def test_the_verified_label_appears_only_under_the_verified_key(self):
        block = JS[JS.index("const PROFILE_TIER_META"):JS.index("function profileTierMeta")]
        verified_entry = block[block.index("VERIFIED_HIGH_CONFIDENCE"):
                                block.index("UNVERIFIED_POSSIBLE_MATCH")]
        assert "Verified high-confidence" in verified_entry
        rest = block[block.index("UNVERIFIED_POSSIBLE_MATCH"):]
        assert "Verified high-confidence" not in rest

    def test_the_tier_badge_is_derived_solely_from_rel_tier(self):
        """profileTierMeta must be a pure lookup on the tier string -- no
        other field (profile_url, handle, similarity) may influence which
        badge renders. Structural: the function body is exactly the lookup,
        nothing else."""
        start = JS.index("function profileTierMeta")
        end = JS.index("function profileEvidenceChain")
        body = JS[start:end]
        assert "PROFILE_TIER_META[tier]" in body
        for forbidden in ("profile_url", "handle", "face_similarity", ".url"):
            assert forbidden not in body

    def test_an_unrecognised_tier_falls_back_to_the_weakest_not_the_strongest(self):
        start = JS.index("function profileTierMeta")
        end = JS.index("function profileEvidenceChain")
        body = JS[start:end]
        fallback = body[body.index("||"):]
        assert '"discovered"' in fallback
        assert '"verified"' not in fallback

    def test_the_section_is_placed_after_verified_evidence_not_inside_it(self):
        results_fn = JS[JS.index("function renderResults"):]
        cand_pos = results_fn.index("candidateVerificationSection(result, {})")
        profile_pos = results_fn.index("publicProfileEvidenceSection(result)")
        tamper_pos = results_fn.index("Can this result be tampered with?")
        assert cand_pos < profile_pos < tamper_pos

    def test_the_empty_state_uses_the_exact_required_sentence(self):
        assert "No verified public profile relationship found." in JS

    def test_no_page_is_auto_embedded_only_plain_links(self):
        section = JS[JS.index("function profileCard"):JS.index("function publicProfileEvidenceSection")]
        for forbidden in ("<iframe", "<embed", "<object"):
            assert forbidden not in section
        assert 'target="_blank"' in section
