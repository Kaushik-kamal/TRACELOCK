"""Verification: pHash, the 2x2 relation, policy, and the full pipeline.

Pipeline tests use a FAKE face engine so the whole DISCOVERED -> CLASSIFIED
path can be exercised offline in milliseconds, including every rejection
branch. Live behaviour is covered by scripts/verify_candidates.py.
"""

from __future__ import annotations

import io
import json

import httpx
import numpy as np
import pytest
import respx

from tracelock.acquisition.cas import ContentAddressedStore
from tracelock.acquisition.fetcher import FetchPolicy
from tracelock.core.models import Candidate
from tracelock.core.reasons import (
    RejectionReason,
    Stage,
    StageOutcome,
    VerificationStatus,
    reasons_for_stage,
)
from tracelock.face.errors import NoFaceDetectedError
from tracelock.verification.phash import (
    HASH_BITS,
    compute_phash,
    is_near_duplicate,
    phash_from_bytes,
)
from tracelock.verification.policy import SimilarityBand, VerificationPolicy
from tracelock.verification.relation import EvidenceRelation, assess_relation
from tracelock.verification.verifier import CandidateVerifier

cv2 = pytest.importorskip("cv2")

URL = "https://example.test/photo.jpg"


# ==========================================================================
# Fixtures: synthetic images and a fake face engine
# ==========================================================================


def make_image(seed: int = 1, size: int = 240) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 210, size=(size // 8, size // 8, 3), dtype=np.uint8)
    return cv2.resize(base, (size, size), interpolation=cv2.INTER_LINEAR)


def encode(image: np.ndarray, quality: int = 92) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return buffer.tobytes()


def unit_vector(seed: int, dim: int = 512) -> np.ndarray:
    vector = np.random.default_rng(seed).normal(size=dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


def FakeEmbedding(vector, model_id="fake:v1"):
    """Build a REAL Embedding.

    Deliberately not a duck-typed stand-in: `cosine_similarity` accepts only an
    Embedding or a raw array, and that strictness is intentional -- it is what
    stops two incompatible vector types being compared silently. The fake
    engine therefore returns the genuine type.
    """
    from tracelock.face.models import Embedding

    return Embedding(vector=np.asarray(vector, dtype=np.float32), model_id=model_id)


class FakeAnalysis:
    """Minimal stand-in for FaceAnalysisResult."""

    def __init__(
        self,
        vector,
        *,
        path="fake.jpg",
        faces=1,
        det=0.9,
        quality=0.8,
        ambiguous=False,
        model_id="fake:v1",
    ):
        from tracelock.face.models import BoundingBox

        self.embedding = FakeEmbedding(vector, model_id)
        self.faces_detected = faces
        self.image = type("Img", (), {"path": path, "sha256": "0" * 64})()
        self.primary = type(
            "P", (), {
                "det_score": det,
                "bbox": BoundingBox(10, 10, 110, 110),
                "quality": type("Q", (), {
                    "aggregate": quality,
                    "band": type("B", (), {"value": "GOOD"})(),
                })(),
                "pose": None,
            }
        )()
        self.selection = type("S", (), {"ambiguous": ambiguous, "margin": 0.5})()

    def warning_codes(self):
        return ()


class FakeEngine:
    """Face engine stub. `behaviour` maps a CAS path substring to an outcome."""

    def __init__(self, default_vector, *, raise_no_face=False, quality=0.8,
                 ambiguous=False, model_id="fake:v1", dim=512):
        self.default_vector = default_vector
        self.raise_no_face = raise_no_face
        self.quality = quality
        self.ambiguous = ambiguous
        self.model_id = model_id
        self.dim = dim
        self.calls = 0

    def analyze(self, path):
        self.calls += 1
        if self.raise_no_face:
            raise NoFaceDetectedError("no face", image_sha256="a" * 64)
        return FakeAnalysis(
            self.default_vector, path=str(path), quality=self.quality,
            ambiguous=self.ambiguous, model_id=self.model_id,
        )


def make_candidate(url=URL, index=0) -> Candidate:
    return Candidate(
        provider="serpapi",
        post_url="https://site.test/post/{0}".format(index),
        image_url=url,
        title="candidate {0}".format(index),
        rank=index + 1,
    )


@pytest.fixture
def probe_analysis():
    return FakeAnalysis(unit_vector(1), path="probe.jpg")


def open_policy(**kwargs) -> VerificationPolicy:
    defaults = dict(block_private_targets=False)
    defaults.update(kwargs)
    return defaults


# ==========================================================================
# Perceptual hashing
# ==========================================================================


class TestPerceptualHash:
    def test_identical_images_hash_identically(self):
        image = make_image(1)
        assert compute_phash(image).bits == compute_phash(image.copy()).bits

    def test_distance_to_self_is_zero(self):
        h = compute_phash(make_image(1))
        assert h.distance(h) == 0
        assert h.similarity(h) == 1.0

    def test_hex_digest_is_16_characters(self):
        assert len(compute_phash(make_image(1)).hex_digest) == 16

    def test_recompression_barely_changes_the_hash(self):
        # SAME PHOTO, DIFFERENT BYTES -- the case that separates exact from
        # near duplication.
        image = make_image(2)
        original = phash_from_bytes(encode(image, quality=95))
        recompressed = phash_from_bytes(encode(image, quality=40))
        assert original.distance(recompressed) <= 10
        assert is_near_duplicate(original, recompressed)

    def test_resizing_barely_changes_the_hash(self):
        image = make_image(3)
        small = cv2.resize(image, (120, 120))
        assert compute_phash(image).distance(compute_phash(small)) <= 10

    def test_unrelated_images_hash_far_apart(self):
        a = compute_phash(make_image(10))
        b = compute_phash(make_image(999))
        assert a.distance(b) > 10
        assert not is_near_duplicate(a, b)

    def test_distance_is_bounded_by_the_bit_width(self):
        a = compute_phash(make_image(4))
        b = compute_phash(make_image(5))
        assert 0 <= a.distance(b) <= HASH_BITS

    def test_similarity_is_in_unit_range(self):
        a = compute_phash(make_image(6))
        b = compute_phash(make_image(7))
        assert 0.0 <= a.similarity(b) <= 1.0

    def test_distance_is_symmetric(self):
        a, b = compute_phash(make_image(8)), compute_phash(make_image(9))
        assert a.distance(b) == b.distance(a)

    def test_brightness_shift_is_tolerated(self):
        # The DC term is excluded from the median precisely so overall
        # brightness does not dominate the hash.
        image = make_image(11)
        brighter = np.clip(image.astype(np.int16) + 28, 0, 255).astype(np.uint8)
        assert compute_phash(image).distance(compute_phash(brighter)) <= 12

    def test_greyscale_input_is_accepted(self):
        grey = cv2.cvtColor(make_image(12), cv2.COLOR_BGR2GRAY)
        assert len(compute_phash(grey).hex_digest) == 16

    def test_empty_image_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_phash(np.empty((0, 0), dtype=np.uint8))

    def test_undecodable_bytes_raise(self):
        with pytest.raises(ValueError, match="could not be decoded"):
            phash_from_bytes(b"not an image")

    def test_algorithm_is_pinned_in_the_output(self):
        # Changing the definition invalidates stored hashes, so the version
        # travels with the value.
        assert compute_phash(make_image(1)).to_dict()["algorithm"] == "phash-dct-8x8/1"


# ==========================================================================
# The 2x2 relation
# ==========================================================================


class TestEvidenceRelation:
    def _assess(self, face_sim, phash_dist):
        return assess_relation(
            face_similarity=face_sim,
            phash_distance=phash_dist,
            face_high_threshold=0.55,
            phash_near_duplicate_max_distance=10,
        )

    def test_high_face_high_phash_is_same_photo(self):
        result = self._assess(0.95, 2)
        assert result.relation is EvidenceRelation.SAME_PHOTO_REPUBLISHED

    def test_high_face_low_phash_is_different_photo_same_person(self):
        result = self._assess(0.80, 40)
        assert result.relation is EvidenceRelation.SAME_PERSON_DIFFERENT_PHOTO

    def test_low_face_high_phash_is_the_anomaly_quadrant(self):
        result = self._assess(0.10, 3)
        assert result.relation is EvidenceRelation.VISUAL_MATCH_FACE_MISMATCH

    def test_low_face_low_phash_is_unrelated(self):
        assert self._assess(0.05, 44).relation is EvidenceRelation.UNRELATED

    def test_only_a_different_photo_counts_as_corroboration(self):
        # Republication is provenance, not independent identity evidence.
        # Phase 3 must not double-count it.
        assert self._assess(0.80, 40).relation.is_independent_corroboration
        assert not self._assess(0.95, 2).relation.is_independent_corroboration
        assert not self._assess(0.05, 44).relation.is_independent_corroboration

    def test_signals_are_never_collapsed(self):
        result = self._assess(0.80, 40)
        payload = result.to_dict()
        # Both raw signals must survive independently so a reader can
        # re-derive the quadrant rather than trust it.
        assert payload["face_similarity"] == pytest.approx(0.80)
        assert payload["phash_similarity"] is not None
        assert payload["phash_distance"] == 40
        assert payload["face_signal_high"] is True
        assert payload["phash_signal_high"] is False

    def test_missing_face_signal_is_undetermined(self):
        assert self._assess(None, 5).relation is EvidenceRelation.UNDETERMINED

    def test_missing_phash_signal_is_undetermined(self):
        assert self._assess(0.9, None).relation is EvidenceRelation.UNDETERMINED

    def test_every_relation_has_an_explanation(self):
        for relation in EvidenceRelation:
            assert len(relation.explanation) > 20

    def test_marked_provisional(self):
        assert self._assess(0.9, 2).provisional is True


# ==========================================================================
# Policy
# ==========================================================================


class TestVerificationPolicy:
    def test_default_bands(self):
        policy = VerificationPolicy()
        assert policy.band_for(0.10) is SimilarityBand.LOW
        assert policy.band_for(0.45) is SimilarityBand.INDETERMINATE
        assert policy.band_for(0.80) is SimilarityBand.HIGH

    def test_boundaries_are_inclusive_at_the_floor(self):
        policy = VerificationPolicy(similarity_floor=0.35, similarity_ceiling=0.55)
        assert policy.band_for(0.35) is SimilarityBand.INDETERMINATE
        assert policy.band_for(0.55) is SimilarityBand.HIGH

    def test_inconclusive_band_is_deliberately_wide(self):
        # The gap IS our admitted uncertainty. Narrowing it needs data.
        assert VerificationPolicy().inconclusive_band_width == pytest.approx(0.20)

    def test_calibrated_is_derived_and_has_no_setter(self):
        # Stronger than the old guard: `calibrated` is not a field at all, so
        # it cannot be set to True by any means. It is derived from whether a
        # fitted model is present.
        with pytest.raises(TypeError):
            VerificationPolicy(calibrated=True)

    def test_uncalibrated_by_default(self):
        policy = VerificationPolicy()
        assert policy.calibrated is False
        assert policy.to_dict()["thresholds_are_provisional"] is True
        assert policy.to_dict()["calibration_model"] is None

    def test_uncalibrated_policy_returns_no_probability(self):
        # Without a model there IS no probability. Returning a number would be
        # inventing one.
        assert VerificationPolicy().identity_probability(0.9) is None

    def test_non_model_calibration_is_rejected(self):
        with pytest.raises(ValueError, match="must be a fitted CalibrationModel"):
            VerificationPolicy(calibration="pretending to be calibrated")

    def test_calibrated_policy_reports_probabilities(self):
        from tracelock.calibration.contract import DatasetKind, DatasetProvenance
        from tracelock.calibration.model import fit_calibration_model

        model = fit_calibration_model(
            genuine=[0.41, 0.43, 0.45, 0.42, 0.44],
            impostor=[0.05, 0.10, 0.15, 0.20, 0.29, 0.01, 0.12],
            provenance=DatasetProvenance(DatasetKind.REAL, "unit-test"),
            labeling_basis="synthetic scores exercising the fitting path",
        )
        policy = VerificationPolicy.from_calibration(model)

        assert policy.calibrated is True
        assert policy.to_dict()["thresholds_are_provisional"] is False
        # Monotonic and bounded.
        low = policy.identity_probability(0.05)
        high = policy.identity_probability(0.45)
        assert 0.0 <= low < high <= 1.0

    def test_calibration_cannot_be_fitted_from_a_fixture(self):
        from tracelock.calibration.contract import DatasetKind, DatasetProvenance
        from tracelock.calibration.model import fit_calibration_model

        with pytest.raises(ValueError, match="FIXTURE"):
            fit_calibration_model(
                genuine=[0.4], impostor=[0.1],
                provenance=DatasetProvenance(DatasetKind.FIXTURE, "synthetic"),
                labeling_basis="should never be permitted",
            )

    def test_inverted_bounds_rejected(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            VerificationPolicy(similarity_floor=0.8, similarity_ceiling=0.2)

    def test_out_of_range_bounds_rejected(self):
        with pytest.raises(ValueError):
            VerificationPolicy(similarity_floor=-0.5)
        with pytest.raises(ValueError):
            VerificationPolicy(similarity_ceiling=1.5)

    def test_serialized_policy_carries_the_disclaimer(self):
        payload = VerificationPolicy().to_dict()
        assert payload["calibrated"] is False
        assert payload["thresholds_are_provisional"] is True
        assert "not identity probabilities" in payload["disclaimer"]


# ==========================================================================
# Rejection taxonomy
# ==========================================================================


class TestRejectionTaxonomy:
    def test_every_reason_maps_to_a_stage(self):
        for reason in RejectionReason:
            assert isinstance(reason.stage, Stage)

    def test_every_reason_has_a_real_explanation(self):
        for reason in RejectionReason:
            assert len(reason.explanation) > 30, reason.value

    def test_every_stage_except_classified_can_reject(self):
        for stage in (Stage.ACQUIRED, Stage.VALIDATED, Stage.ANALYZED, Stage.COMPARED):
            assert reasons_for_stage(stage), stage.value

    def test_stage_order_is_monotonic(self):
        orders = [s.order for s in Stage]
        assert orders == sorted(orders)


# ==========================================================================
# The pipeline
# ==========================================================================


class TestVerifierPipeline:
    def _verifier(self, tmp_path, engine, **policy_kwargs):
        return CandidateVerifier(
            engine,
            store=ContentAddressedStore(tmp_path / "cas"),
            policy=VerificationPolicy(**policy_kwargs),
            fetch_policy=FetchPolicy(block_private_targets=False),
        )

    @respx.mock
    def test_matching_candidate_is_verified(self, tmp_path, probe_analysis):
        respx.get(URL).mock(
            return_value=httpx.Response(200, content=encode(make_image(1)),
                                        headers={"content-type": "image/jpeg"})
        )
        # Same vector as the probe -> similarity 1.0
        verifier = self._verifier(tmp_path, FakeEngine(unit_vector(1)))
        result = verifier.verify(probe_analysis, make_candidate())

        assert result.status is VerificationStatus.VERIFIED_CANDIDATE
        assert result.face_similarity == pytest.approx(1.0, abs=1e-6)
        assert result.similarity_band is SimilarityBand.HIGH
        assert result.content_sha256
        assert result.phash

    @respx.mock
    def test_dissimilar_candidate_is_rejected(self, tmp_path, probe_analysis):
        respx.get(URL).mock(return_value=httpx.Response(200, content=encode(make_image(2))))
        verifier = self._verifier(tmp_path, FakeEngine(unit_vector(777)))
        result = verifier.verify(probe_analysis, make_candidate())

        assert result.status is VerificationStatus.REJECTED
        assert RejectionReason.LOW_FACE_SIMILARITY in result.rejection_reasons
        assert result.failed_stage is Stage.COMPARED

    @respx.mock
    def test_http_error_is_rejected_at_the_acquired_stage(self, tmp_path, probe_analysis):
        respx.get(URL).mock(return_value=httpx.Response(404))
        verifier = self._verifier(tmp_path, FakeEngine(unit_vector(1)))
        result = verifier.verify(probe_analysis, make_candidate())

        assert result.status is VerificationStatus.REJECTED
        assert result.primary_reason is RejectionReason.HTTP_ERROR
        assert result.failed_stage is Stage.ACQUIRED
        assert result.face is None

    @respx.mock
    def test_html_masquerading_as_image_is_rejected_with_a_precise_reason(
        self, tmp_path, probe_analysis
    ):
        # The exact scenario named in the phase brief.
        respx.get(URL).mock(
            return_value=httpx.Response(
                200,
                content=b"<!DOCTYPE html><html><body>Sign in</body></html>",
                headers={"content-type": "image/jpeg"},
            )
        )
        verifier = self._verifier(tmp_path, FakeEngine(unit_vector(1)))
        result = verifier.verify(probe_analysis, make_candidate())

        assert result.primary_reason is RejectionReason.NOT_AN_IMAGE
        assert result.failed_stage is Stage.VALIDATED
        assert "HTML" in result.explain()

    @respx.mock
    def test_no_face_is_rejected_at_the_analyzed_stage(self, tmp_path, probe_analysis):
        respx.get(URL).mock(return_value=httpx.Response(200, content=encode(make_image(1))))
        verifier = self._verifier(tmp_path, FakeEngine(unit_vector(1), raise_no_face=True))
        result = verifier.verify(probe_analysis, make_candidate())

        assert result.primary_reason is RejectionReason.NO_FACE_DETECTED
        assert result.failed_stage is Stage.ANALYZED
        # Bytes were still acquired and stored -- the failure is downstream.
        assert result.content_sha256 is not None

    @respx.mock
    def test_inconclusive_band_produces_inconclusive(self, tmp_path, probe_analysis):
        respx.get(URL).mock(return_value=httpx.Response(200, content=encode(make_image(1))))

        # Construct a vector at a known middling angle to the probe.
        probe_vec = probe_analysis.embedding.vector
        orthogonal = unit_vector(42)
        orthogonal = orthogonal - np.dot(orthogonal, probe_vec) * probe_vec
        orthogonal = orthogonal / np.linalg.norm(orthogonal)
        target = 0.45
        mid = target * probe_vec + np.sqrt(1 - target**2) * orthogonal

        verifier = self._verifier(tmp_path, FakeEngine(mid.astype(np.float32)))
        result = verifier.verify(probe_analysis, make_candidate())

        assert result.status is VerificationStatus.INCONCLUSIVE
        assert result.similarity_band is SimilarityBand.INDETERMINATE
        assert not result.rejection_reasons

    @respx.mock
    def test_exact_duplicate_is_detected_and_both_refs_kept(self, tmp_path, probe_analysis):
        payload = encode(make_image(1))
        url_a, url_b = "https://a.test/1.jpg", "https://b.test/2.jpg"
        respx.get(url_a).mock(return_value=httpx.Response(200, content=payload))
        respx.get(url_b).mock(return_value=httpx.Response(200, content=payload))

        verifier = self._verifier(tmp_path, FakeEngine(unit_vector(1)))
        first = verifier.verify(probe_analysis, make_candidate(url_a, 0), 0)
        second = verifier.verify(probe_analysis, make_candidate(url_b, 1), 1)

        assert first.status is VerificationStatus.VERIFIED_CANDIDATE
        assert second.status is VerificationStatus.REJECTED
        assert second.primary_reason is RejectionReason.DUPLICATE_CONTENT
        assert second.duplicate.is_exact_duplicate
        assert second.duplicate.duplicate_of_candidate_id == first.candidate_id
        # One blob, two source references.
        assert second.duplicate.cas_reference_count == 2

    @respx.mock
    def test_near_duplicate_has_different_bytes_but_is_detected(
        self, tmp_path, probe_analysis
    ):
        image = make_image(1)
        url_a, url_b = "https://a.test/1.jpg", "https://b.test/2.jpg"
        respx.get(url_a).mock(return_value=httpx.Response(200, content=encode(image, 95)))
        respx.get(url_b).mock(return_value=httpx.Response(200, content=encode(image, 35)))

        verifier = self._verifier(tmp_path, FakeEngine(unit_vector(1)))
        first = verifier.verify(probe_analysis, make_candidate(url_a, 0), 0)
        second = verifier.verify(probe_analysis, make_candidate(url_b, 1), 1)

        # DIFFERENT bytes -- so NOT an exact duplicate...
        assert first.content_sha256 != second.content_sha256
        assert not second.duplicate.is_exact_duplicate
        # ...but visually the same image.
        assert first.candidate_id in second.duplicate.near_duplicate_of_candidate_ids

    @respx.mock
    def test_ambiguous_faces_warn_by_default_and_reject_when_configured(
        self, tmp_path, probe_analysis
    ):
        respx.get(URL).mock(return_value=httpx.Response(200, content=encode(make_image(1))))
        engine = FakeEngine(unit_vector(1), ambiguous=True)

        lenient = self._verifier(tmp_path, engine)
        assert lenient.verify(probe_analysis, make_candidate()).status is (
            VerificationStatus.VERIFIED_CANDIDATE
        )

        strict = self._verifier(tmp_path / "s", engine, reject_ambiguous_faces=True)
        strict_result = strict.verify(probe_analysis, make_candidate())
        assert strict_result.primary_reason is RejectionReason.AMBIGUOUS_MULTIPLE_FACES

    @respx.mock
    def test_low_quality_warns_by_default_and_rejects_when_configured(
        self, tmp_path, probe_analysis
    ):
        respx.get(URL).mock(return_value=httpx.Response(200, content=encode(make_image(1))))
        engine = FakeEngine(unit_vector(1), quality=0.05)

        lenient = self._verifier(tmp_path, engine)
        lenient_result = lenient.verify(probe_analysis, make_candidate())
        assert lenient_result.status is VerificationStatus.VERIFIED_CANDIDATE
        assert any("quality" in w for w in lenient_result.warnings)

        strict = self._verifier(tmp_path / "s", engine, reject_on_low_quality=True)
        assert strict.verify(probe_analysis, make_candidate()).primary_reason is (
            RejectionReason.LOW_FACE_QUALITY
        )

    def test_candidate_with_no_media_url_is_rejected(self, tmp_path, probe_analysis):
        verifier = self._verifier(tmp_path, FakeEngine(unit_vector(1)))
        candidate = Candidate(provider="serpapi", post_url="https://site.test/p")
        result = verifier.verify(probe_analysis, candidate)

        assert result.primary_reason is RejectionReason.NO_MEDIA_URL
        assert result.failed_stage is Stage.DISCOVERED

    @respx.mock
    def test_ssrf_target_is_rejected(self, tmp_path, probe_analysis):
        verifier = CandidateVerifier(
            FakeEngine(unit_vector(1)),
            store=ContentAddressedStore(tmp_path / "cas"),
            fetch_policy=FetchPolicy(block_private_targets=True),
        )
        result = verifier.verify(
            probe_analysis, make_candidate("http://169.254.169.254/latest/meta-data/")
        )
        assert result.primary_reason is RejectionReason.BLOCKED_URL_TARGET


class TestPipelineInvariants:
    """The guarantees that make rejection transparency real."""

    @respx.mock
    def test_every_candidate_produces_exactly_one_result(self, tmp_path, probe_analysis):
        urls = ["https://x.test/{0}.jpg".format(i) for i in range(6)]
        respx.get(urls[0]).mock(return_value=httpx.Response(200, content=encode(make_image(1))))
        respx.get(urls[1]).mock(return_value=httpx.Response(404))
        respx.get(urls[2]).mock(return_value=httpx.Response(200, content=b"<html>x</html>"))
        respx.get(urls[3]).mock(side_effect=httpx.ReadTimeout("slow"))
        respx.get(urls[4]).mock(return_value=httpx.Response(200, content=b""))
        respx.get(urls[5]).mock(return_value=httpx.Response(200, content=encode(make_image(9))))

        verifier = CandidateVerifier(
            FakeEngine(unit_vector(1)),
            store=ContentAddressedStore(tmp_path / "cas"),
            fetch_policy=FetchPolicy(block_private_targets=False),
        )
        candidates = [make_candidate(u, i) for i, u in enumerate(urls)]
        results = verifier.verify_all(probe_analysis, candidates)

        # THE core invariant: nothing disappears.
        assert len(results) == len(candidates)

    @respx.mock
    def test_every_rejection_names_a_reason_and_a_stage(self, tmp_path, probe_analysis):
        urls = ["https://x.test/{0}.jpg".format(i) for i in range(4)]
        respx.get(urls[0]).mock(return_value=httpx.Response(404))
        respx.get(urls[1]).mock(return_value=httpx.Response(200, content=b"<html>x</html>"))
        respx.get(urls[2]).mock(side_effect=httpx.ReadTimeout("slow"))
        respx.get(urls[3]).mock(return_value=httpx.Response(200, content=b""))

        verifier = CandidateVerifier(
            FakeEngine(unit_vector(1)),
            store=ContentAddressedStore(tmp_path / "cas"),
            fetch_policy=FetchPolicy(block_private_targets=False),
        )
        results = verifier.verify_all(
            probe_analysis, [make_candidate(u, i) for i, u in enumerate(urls)]
        )

        for result in results:
            assert result.status is VerificationStatus.REJECTED
            assert result.rejection_reasons
            assert result.failed_stage is not None
            assert result.primary_reason.stage is result.failed_stage
            assert len(result.explain()) > 20

    @respx.mock
    def test_stage_history_records_progress_and_skips(self, tmp_path, probe_analysis):
        respx.get(URL).mock(return_value=httpx.Response(404))
        verifier = CandidateVerifier(
            FakeEngine(unit_vector(1)),
            store=ContentAddressedStore(tmp_path / "cas"),
            fetch_policy=FetchPolicy(block_private_targets=False),
        )
        result = verifier.verify(probe_analysis, make_candidate())

        stages = {r["stage"]: r["outcome"] for r in result.stage_history}
        assert stages["DISCOVERED"] == StageOutcome.OK.value
        assert stages["ACQUIRED"] == StageOutcome.FAILED.value
        # Later stages are explicitly recorded as never-run, not merely absent.
        assert stages["ANALYZED"] == StageOutcome.SKIPPED.value
        assert stages["CLASSIFIED"] == StageOutcome.SKIPPED.value

    @respx.mock
    def test_result_serializes_to_json(self, tmp_path, probe_analysis):
        respx.get(URL).mock(return_value=httpx.Response(200, content=encode(make_image(1))))
        verifier = CandidateVerifier(
            FakeEngine(unit_vector(1)),
            store=ContentAddressedStore(tmp_path / "cas"),
            fetch_policy=FetchPolicy(block_private_targets=False),
        )
        payload = verifier.verify(probe_analysis, make_candidate()).to_dict()

        assert json.dumps(payload)
        assert payload["schema_version"] == "verification-result/1"
        assert payload["face_similarity"] is not None
        assert payload["relation"] is not None

    @respx.mock
    def test_raw_embedding_never_reaches_the_artifact(self, tmp_path, probe_analysis):
        respx.get(URL).mock(return_value=httpx.Response(200, content=encode(make_image(1))))
        verifier = CandidateVerifier(
            FakeEngine(unit_vector(1)),
            store=ContentAddressedStore(tmp_path / "cas"),
            fetch_policy=FetchPolicy(block_private_targets=False),
        )
        payload = verifier.verify(probe_analysis, make_candidate()).to_dict()
        serialized = json.dumps(payload)

        # Biometric data must not leak into a persisted artifact.
        assert "vector" not in serialized
        assert payload["face"]["embedding_quantized_sha256"]
        assert "embedding" not in payload["face"] or isinstance(
            payload["face"].get("embedding_dimension"), int
        )

    @respx.mock
    def test_no_probability_language_in_the_artifact(self, tmp_path, probe_analysis):
        respx.get(URL).mock(return_value=httpx.Response(200, content=encode(make_image(1))))
        verifier = CandidateVerifier(
            FakeEngine(unit_vector(1)),
            store=ContentAddressedStore(tmp_path / "cas"),
            fetch_policy=FetchPolicy(block_private_targets=False),
        )
        result = verifier.verify(probe_analysis, make_candidate())
        payload = result.to_dict()

        # A cosine is a geometric measurement. "92% same person" is a
        # calibrated statistical claim we have not earned. Assert on the
        # SEMANTICS rather than scanning for substrings -- filesystem paths in
        # the artifact make a raw substring scan meaningless.
        similarity = payload["face_similarity"]
        assert isinstance(similarity, float)
        assert -1.0 <= similarity <= 1.0

        # The band is a coarse label, never a percentage.
        assert payload["similarity_band"] in {"LOW", "INDETERMINATE", "HIGH"}

        # The relation carries both raw signals and is marked provisional.
        assert payload["relation"]["provisional"] is True

        # And no field anywhere renders a similarity as a percent string.
        def walk(node):
            if isinstance(node, dict):
                for value in node.values():
                    yield from walk(value)
            elif isinstance(node, list):
                for value in node:
                    yield from walk(value)
            elif isinstance(node, str):
                yield node

        for text in walk(payload):
            assert "% same" not in text.lower()
            assert "% match" not in text.lower()

    def test_policy_disclaimer_travels_with_the_run(self):
        # The run artifact serializes the policy; that is where the explicit
        # "not identity probabilities" statement must appear.
        payload = VerificationPolicy().to_dict()
        assert payload["calibrated"] is False
        assert "not identity probabilities" in payload["disclaimer"]

    @respx.mock
    def test_similarity_rejection_still_names_its_stage(self, tmp_path, probe_analysis):
        # Regression: a LOW_FACE_SIMILARITY rejection reaches the terminal
        # return through the SUCCESS path, where no stage recorded FAILED.
        # `failed_stage` must still be populated from the reason.
        respx.get(URL).mock(return_value=httpx.Response(200, content=encode(make_image(3))))
        verifier = CandidateVerifier(
            FakeEngine(unit_vector(4242)),
            store=ContentAddressedStore(tmp_path / "cas"),
            fetch_policy=FetchPolicy(block_private_targets=False),
        )
        result = verifier.verify(probe_analysis, make_candidate())

        assert result.status is VerificationStatus.REJECTED
        assert result.failed_stage is Stage.COMPARED
        assert result.primary_reason.stage is result.failed_stage
