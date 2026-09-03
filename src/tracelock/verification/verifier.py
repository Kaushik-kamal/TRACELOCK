"""The verification firewall.

    DISCOVERY MAKES CLAIMS.  TRACELOCK RE-MEASURES EVIDENCE.

A provider handed us a URL and asserted a relationship. This module tests that
assertion against bytes it downloads itself, and is built to REJECT.

    DISCOVERED -> ACQUIRED -> VALIDATED -> ANALYZED -> COMPARED -> CLASSIFIED

WHAT IS RE-DERIVED, NOT TRUSTED
-------------------------------
  the bytes          re-downloaded; the provider's thumbnail is not evidence
  the content type   magic bytes decide, headers are only recorded
  the face           detection re-run on our bytes, not assumed from the match
  the similarity     computed from our embeddings, not the provider's ranking
  the identity       NOT decided -- measured, banded, and left provisional

INVARIANT
---------
N candidates in, N results out. Always. A failure is a finding.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Iterable

import httpx

from tracelock.acquisition.cas import ContentAddressedStore, SourceReference
from tracelock.acquisition.fetcher import AcquisitionResult, FetchPolicy, fetch_media
from tracelock.acquisition.provenance import describe_source
from tracelock.acquisition.validation import ImageValidation, validate_image_bytes
from tracelock.core.models import Candidate
from tracelock.face.errors import (
    EmbeddingDimensionMismatch,
    FaceEngineError,
    NoFaceDetectedError,
)
from tracelock.face.models import FaceAnalysisResult
from tracelock.face.similarity import cosine_similarity
from tracelock.verification.models import (
    SCHEMA_VERSION,
    DuplicateInfo,
    FaceEvidence,
    StageHistory,
    VerificationResult,
    build_acquisition_dict,
    build_provenance_dict,
    build_validation_dict,
    utc_now,
)
from tracelock.verification.phash import PerceptualHash, compute_phash
from tracelock.verification.policy import DEFAULT_POLICY, VerificationPolicy
from tracelock.core.reasons import (
    RejectionReason,
    Stage,
    StageOutcome,
    VerificationStatus,
)
from tracelock.verification.relation import assess_relation


class CandidateVerifier:
    """Verifies discovered candidates against a probe analysis."""

    def __init__(
        self,
        face_engine,
        *,
        store: ContentAddressedStore | None = None,
        policy: VerificationPolicy | None = None,
        fetch_policy: FetchPolicy | None = None,
        client: httpx.Client | None = None,
        fetch=None,
        analyze=None,
        probe_phash: PerceptualHash | None = None,
        timer=None,
    ) -> None:
        self.engine = face_engine
        self.store = store or ContentAddressedStore()
        self.policy = policy or DEFAULT_POLICY
        self.fetch_policy = fetch_policy or FetchPolicy()
        self._client = client

        # Optional seams for caching and prefetching. Both default to the
        # direct call, so behaviour is identical unless a caller supplies one.
        # They exist so the runner can share a warmed download cache and a
        # sha256-keyed face cache WITHOUT this class knowing either exists --
        # verification logic stays a pure function of the bytes it is given.
        self._fetch = fetch
        self._analyze = analyze

        # The probe's perceptual hash is a constant for the whole run. It used
        # to be recomputed inside this loop -- 28.8ms per candidate to
        # re-derive the same value from the same unchanged file.
        self._probe_phash = probe_phash
        self._timer = timer

        # Cross-candidate state for duplicate detection within a run.
        self._seen_hashes: dict[str, str] = {}          # sha256 -> candidate_id
        self._seen_phashes: list[tuple[str, PerceptualHash]] = []

    # ------------------------------------------------------------------

    def verify_all(
        self, probe: FaceAnalysisResult, candidates: Iterable[Candidate]
    ) -> list[VerificationResult]:
        """Verify every candidate. Returns exactly one result per input."""
        return [
            self.verify(probe, candidate, index)
            for index, candidate in enumerate(candidates)
        ]

    def verify(
        self, probe: FaceAnalysisResult, candidate: Candidate, index: int = 0
    ) -> VerificationResult:
        """Run one candidate through the pipeline. Never raises."""
        started = time.perf_counter()
        history = StageHistory()
        warnings: list[str] = []

        candidate_id = _candidate_id(candidate, index)
        media_url = candidate.image_url or candidate.thumbnail_url
        provenance = describe_source(candidate.post_url or media_url or "")

        history.record(Stage.DISCOVERED, StageOutcome.OK, "provider={0}".format(candidate.provider))

        # -- no media to fetch ------------------------------------------
        if not media_url:
            return self._reject(
                candidate_id, candidate, media_url, provenance, history,
                RejectionReason.NO_MEDIA_URL, started, warnings,
            )

        if candidate.image_url is None and candidate.thumbnail_url:
            warnings.append(
                "no full-resolution image_url; falling back to the thumbnail, "
                "which is lower quality evidence"
            )

        # -- STAGE: ACQUIRED --------------------------------------------
        acquisition = (
            self._fetch(media_url)
            if self._fetch is not None
            else fetch_media(media_url, policy=self.fetch_policy, client=self._client)
        )
        if not acquisition.ok:
            history.record(
                Stage.ACQUIRED, StageOutcome.FAILED, acquisition.detail,
                acquisition.elapsed_seconds,
            )
            return self._reject(
                candidate_id, candidate, media_url, provenance, history,
                acquisition.reason or RejectionReason.DOWNLOAD_FAILED,
                started, warnings, acquisition=acquisition,
            )

        history.record(
            Stage.ACQUIRED, StageOutcome.OK,
            "{0} bytes, HTTP {1}".format(acquisition.byte_size, acquisition.status_code),
            acquisition.elapsed_seconds,
        )
        if acquisition.url_changed:
            warnings.append(
                "redirected to a different final URL: {0}".format(acquisition.final_url)
            )

        # -- STAGE: VALIDATED -------------------------------------------
        validation = validate_image_bytes(
            acquisition.content or b"",
            declared_content_type=acquisition.declared_content_type,
        )
        if not validation.ok:
            history.record(Stage.VALIDATED, StageOutcome.FAILED, validation.detail)
            return self._reject(
                candidate_id, candidate, media_url, provenance, history,
                validation.reason or RejectionReason.NOT_AN_IMAGE,
                started, warnings, acquisition=acquisition, validation=validation,
            )

        warnings.extend(validation.warnings)
        history.record(
            Stage.VALIDATED, StageOutcome.OK,
            "{0} {1}x{2}".format(validation.detected_format, validation.width, validation.height),
        )

        # -- CAS: store original bytes, detect exact duplicates ----------
        entry = self.store.put(
            acquisition.content or b"",
            SourceReference(
                requested_url=media_url,
                final_url=acquisition.final_url,
                candidate_id=candidate_id,
                provider=candidate.provider,
            ),
        )

        duplicate_of = self._seen_hashes.get(entry.sha256)
        is_exact_duplicate = duplicate_of is not None
        if not is_exact_duplicate:
            self._seen_hashes[entry.sha256] = candidate_id

        if is_exact_duplicate and self.policy.reject_exact_duplicates:
            duplicate = DuplicateInfo(
                is_exact_duplicate=True,
                duplicate_of_candidate_id=duplicate_of,
                cas_reference_count=entry.reference_count,
            )
            history.record(
                Stage.VALIDATED, StageOutcome.FAILED,
                "byte-identical to {0}".format(duplicate_of),
            )
            return self._reject(
                candidate_id, candidate, media_url, provenance, history,
                RejectionReason.DUPLICATE_CONTENT, started, warnings,
                acquisition=acquisition, validation=validation,
                content_sha256=entry.sha256, cas_path=str(entry.path),
                duplicate=duplicate,
            )

        # -- CHEAP PREFILTER: near-duplicate before any embedding --------
        # A perceptual hash costs ~29ms; a face embedding costs ~1.73s. Doing
        # this after the embedding, as it used to be, meant every resized or
        # re-encoded copy of an image we had already seen paid the full 1.73s
        # to be told it was a copy.
        candidate_phash = self._safe_phash(entry.path)
        near_duplicates = self._find_near_duplicates(candidate_id, candidate_phash)
        if candidate_phash:
            self._seen_phashes.append((candidate_id, candidate_phash))

        phash_distance = None
        if candidate_phash and self._probe_phash:
            phash_distance = self._probe_phash.distance(candidate_phash)

        if near_duplicates and self.policy.reject_exact_duplicates:
            # A near-identical copy carries no new identity information, but it
            # IS a real separate publication, so it is recorded as a duplicate
            # of a known image rather than discarded. Provenance survives; the
            # embedding is simply not recomputed.
            history.record(
                Stage.VALIDATED, StageOutcome.FAILED,
                "perceptually identical to {0}".format(near_duplicates[0]),
            )
            return self._reject(
                candidate_id, candidate, media_url, provenance, history,
                RejectionReason.DUPLICATE_CONTENT, started, warnings,
                acquisition=acquisition, validation=validation,
                content_sha256=entry.sha256, cas_path=str(entry.path),
                duplicate=DuplicateInfo(
                    is_exact_duplicate=False,
                    duplicate_of_candidate_id=near_duplicates[0],
                    cas_reference_count=entry.reference_count,
                    near_duplicate_of_candidate_ids=tuple(near_duplicates),
                ),
                phash=candidate_phash,
            )

        # -- STAGE: ANALYZED --------------------------------------------
        # The engine reads the CAS blob: the ORIGINAL bytes. Any resizing it
        # performs internally is a derived model input, never the evidence.
        analysis: FaceAnalysisResult | None = None
        analysis_started = time.perf_counter()
        try:
            analysis = (
                self._analyze(entry.sha256, entry.path)
                if self._analyze is not None
                else self.engine.analyze(entry.path)
            )
        except NoFaceDetectedError:
            history.record(
                Stage.ANALYZED, StageOutcome.FAILED,
                "no face in the bytes we downloaded",
                time.perf_counter() - analysis_started,
            )
            return self._reject(
                candidate_id, candidate, media_url, provenance, history,
                RejectionReason.NO_FACE_DETECTED, started, warnings,
                acquisition=acquisition, validation=validation,
                content_sha256=entry.sha256, cas_path=str(entry.path),
                phash=self._safe_phash(entry.path),
            )
        except FaceEngineError as exc:
            history.record(
                Stage.ANALYZED, StageOutcome.FAILED, str(exc),
                time.perf_counter() - analysis_started,
            )
            return self._reject(
                candidate_id, candidate, media_url, provenance, history,
                RejectionReason.FACE_ANALYSIS_FAILED, started, warnings,
                acquisition=acquisition, validation=validation,
                content_sha256=entry.sha256, cas_path=str(entry.path),
            )

        face_evidence = _build_face_evidence(analysis)
        if self._timer is not None:
            self._timer.add("face_analysis_cpu", time.perf_counter() - analysis_started)

        history.record(
            Stage.ANALYZED, StageOutcome.OK,
            "{0} face(s), det={1:.3f}, quality={2:.3f}".format(
                analysis.faces_detected,
                analysis.primary.det_score,
                analysis.primary.quality.aggregate,
            ),
            time.perf_counter() - analysis_started,
        )
        warnings.extend(analysis.warning_codes())

        # Ambiguity and quality: warn by default, reject only if configured.
        if analysis.selection.ambiguous and self.policy.reject_ambiguous_faces:
            return self._reject(
                candidate_id, candidate, media_url, provenance, history,
                RejectionReason.AMBIGUOUS_MULTIPLE_FACES, started, warnings,
                acquisition=acquisition, validation=validation,
                content_sha256=entry.sha256, cas_path=str(entry.path),
                face=face_evidence, phash=self._safe_phash(entry.path),
            )

        quality = analysis.primary.quality.aggregate
        if quality < self.policy.min_face_quality:
            if self.policy.reject_on_low_quality:
                return self._reject(
                    candidate_id, candidate, media_url, provenance, history,
                    RejectionReason.LOW_FACE_QUALITY, started, warnings,
                    acquisition=acquisition, validation=validation,
                    content_sha256=entry.sha256, cas_path=str(entry.path),
                    face=face_evidence, phash=self._safe_phash(entry.path),
                )
            warnings.append(
                "face quality {0:.3f} is below the {1:.2f} floor; the "
                "embedding may be unreliable".format(quality, self.policy.min_face_quality)
            )

        # -- STAGE: COMPARED --------------------------------------------
        try:
            similarity = cosine_similarity(probe.embedding, analysis.embedding)
        except EmbeddingDimensionMismatch as exc:
            history.record(Stage.COMPARED, StageOutcome.FAILED, str(exc))
            return self._reject(
                candidate_id, candidate, media_url, provenance, history,
                RejectionReason.EMBEDDING_INCOMPATIBLE, started, warnings,
                acquisition=acquisition, validation=validation,
                content_sha256=entry.sha256, cas_path=str(entry.path),
                face=face_evidence,
            )

        if probe.embedding.model_id != analysis.embedding.model_id:
            warnings.append(
                "probe and candidate embeddings came from different model ids "
                "({0} vs {1})".format(
                    probe.embedding.model_id, analysis.embedding.model_id
                )
            )

        history.record(
            Stage.COMPARED, StageOutcome.OK,
            "face_sim={0:.4f} phash_distance={1}".format(similarity, phash_distance),
        )

        # -- STAGE: CLASSIFIED ------------------------------------------
        relation = assess_relation(
            face_similarity=similarity,
            phash_distance=phash_distance,
            face_high_threshold=self.policy.similarity_ceiling,
            phash_near_duplicate_max_distance=self.policy.phash_near_duplicate_max_distance,
        )

        band = self.policy.band_for(similarity)
        identity_probability = self.policy.identity_probability(similarity)
        reasons: tuple[RejectionReason, ...] = ()

        if band.value == "LOW":
            status = VerificationStatus.REJECTED
            reasons = (RejectionReason.LOW_FACE_SIMILARITY,)
        elif band.value == "INDETERMINATE":
            status = VerificationStatus.INCONCLUSIVE
        else:
            status = VerificationStatus.VERIFIED_CANDIDATE

        history.record(
            Stage.CLASSIFIED, StageOutcome.OK,
            "{0} (band {1}, PROVISIONAL)".format(status.value, band.value),
        )

        # A similarity rejection is still a rejection and must name its stage.
        # `history.failed_stage` is empty here because every stage RAN fine --
        # the candidate failed on the measurement, not on a pipeline error. So
        # the stage comes from the reason itself, keeping the invariant
        # `primary_reason.stage == failed_stage` true on every rejection path.
        failed_stage = reasons[0].stage if reasons else history.failed_stage

        duplicate = DuplicateInfo(
            is_exact_duplicate=is_exact_duplicate,
            duplicate_of_candidate_id=duplicate_of,
            near_duplicate_of_candidate_ids=near_duplicates,
            cas_reference_count=entry.reference_count,
        )

        return VerificationResult(
            schema_version=SCHEMA_VERSION,
            candidate_id=candidate_id,
            provider=candidate.provider,
            rank=candidate.rank,
            source_url=candidate.post_url or "",
            media_url=media_url,
            status=status,
            rejection_reasons=reasons,
            stage_history=history.to_list(),
            furthest_stage=history.furthest_stage,
            failed_stage=failed_stage,
            acquisition=build_acquisition_dict(acquisition),
            validation=build_validation_dict(validation),
            provenance=build_provenance_dict(provenance),
            face=face_evidence,
            relation=relation,
            duplicate=duplicate,
            content_sha256=entry.sha256,
            cas_path=str(entry.path),
            phash=candidate_phash.hex_digest if candidate_phash else None,
            face_similarity=similarity,
            similarity_band=band,
            identity_probability=identity_probability,
            warnings=tuple(dict.fromkeys(warnings)),
            verified_at=utc_now(),
            elapsed_seconds=time.perf_counter() - started,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reject(
        self,
        candidate_id: str,
        candidate: Candidate,
        media_url: str | None,
        provenance,
        history: StageHistory,
        reason: RejectionReason,
        started: float,
        warnings: list[str],
        *,
        acquisition: AcquisitionResult | None = None,
        validation: ImageValidation | None = None,
        content_sha256: str | None = None,
        cas_path: str | None = None,
        face: FaceEvidence | None = None,
        phash: PerceptualHash | None = None,
        duplicate: DuplicateInfo | None = None,
    ) -> VerificationResult:
        """Build a rejection result. Every failure path routes through here."""
        history.mark_skipped(reason.stage)

        return VerificationResult(
            schema_version=SCHEMA_VERSION,
            candidate_id=candidate_id,
            provider=candidate.provider,
            rank=candidate.rank,
            source_url=candidate.post_url or "",
            media_url=media_url,
            status=VerificationStatus.REJECTED,
            rejection_reasons=(reason,),
            stage_history=history.to_list(),
            furthest_stage=history.furthest_stage,
            failed_stage=reason.stage,
            acquisition=build_acquisition_dict(acquisition),
            validation=build_validation_dict(validation),
            provenance=build_provenance_dict(provenance),
            face=face,
            relation=None,
            duplicate=duplicate,
            content_sha256=content_sha256,
            cas_path=cas_path,
            phash=phash.hex_digest if phash else None,
            face_similarity=None,
            similarity_band=None,
            identity_probability=None,
            warnings=tuple(dict.fromkeys(warnings)),
            verified_at=utc_now(),
            elapsed_seconds=time.perf_counter() - started,
        )

    def _safe_phash(self, path: Path) -> PerceptualHash | None:
        """pHash a stored blob. Returns None rather than failing the pipeline."""
        try:
            import cv2
            import numpy as np

            data = Path(path).read_bytes()
            array = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            return compute_phash(array) if array is not None else None
        except (OSError, ValueError):
            return None

    def _find_near_duplicates(
        self, candidate_id: str, phash: PerceptualHash | None
    ) -> tuple[str, ...]:
        """Visually near-identical candidates seen earlier in this run.

        Distinct from exact duplication: these have DIFFERENT bytes (so
        different SHA-256) but look the same -- recompression or resizing.
        """
        if phash is None:
            return ()
        limit = self.policy.phash_near_duplicate_max_distance
        return tuple(
            other_id
            for other_id, other_hash in self._seen_phashes
            if other_id != candidate_id and phash.distance(other_hash) <= limit
        )


# --------------------------------------------------------------------------


def _candidate_id(candidate: Candidate, index: int) -> str:
    """Stable id derived from the candidate's own content, not its position."""
    basis = (candidate.image_url or candidate.post_url or str(index)).encode("utf-8")
    return "cand-{0:03d}-{1}".format(index, hashlib.sha256(basis).hexdigest()[:8])


def _build_face_evidence(analysis: FaceAnalysisResult) -> FaceEvidence:
    primary = analysis.primary
    return FaceEvidence(
        faces_detected=analysis.faces_detected,
        det_score=primary.det_score,
        bbox=primary.bbox.to_dict(),
        quality_aggregate=primary.quality.aggregate,
        quality_band=primary.quality.band.value,
        embedding_dimension=analysis.embedding.dimension,
        embedding_model_id=analysis.embedding.model_id,
        # A commitment to the embedding. The vector itself never leaves here.
        embedding_quantized_sha256=hashlib.sha256(
            analysis.embedding.quantize()
        ).hexdigest(),
        selection_ambiguous=analysis.selection.ambiguous,
        selection_margin=analysis.selection.margin,
        warnings=analysis.warning_codes(),
        pose_deviation_deg=(
            primary.pose.frontal_deviation_deg if primary.pose else None
        ),
    )
