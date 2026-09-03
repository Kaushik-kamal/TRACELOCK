"""Phase 2 live integration.

    SEARCH ENGINE:  "I found 20 possible candidates."
    TRACELOCK:      "I independently tested them."

Loads candidates from a REAL Phase 0 discovery artifact -- no URL is hardcoded
here -- re-downloads each one, and re-measures everything from the bytes it
fetched itself.

Rejected candidates are shown, not hidden. A verifier that accepts everything
has tested nothing.

    python scripts\\verify_candidates.py --run data\\runs\\search_gate_...json --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tracelock.acquisition.cas import ContentAddressedStore  # noqa: E402
from tracelock.acquisition.fetcher import FetchPolicy  # noqa: E402
from tracelock.core.models import Candidate  # noqa: E402
from tracelock.core.reasons import RejectionReason, VerificationStatus  # noqa: E402
from tracelock.core.runs import make_run_id, run_artifact_path  # noqa: E402
from tracelock.face import FaceEngine, FaceEngineError  # noqa: E402
from tracelock.verification.policy import VerificationPolicy  # noqa: E402
from tracelock.verification.verifier import CandidateVerifier  # noqa: E402

# Windows consoles default to cp1252, which cannot encode characters that
# routinely appear in real URLs and titles (mathematical alphanumerics, emoji,
# CJK). A UnicodeEncodeError while RENDERING would abort the run and lose the
# artifact -- turning a display problem into destroyed evidence. Force UTF-8
# and degrade unencodable characters instead of dying on them.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RULE = "=" * 78
THIN = "-" * 78

STATUS_MARK = {
    VerificationStatus.VERIFIED_CANDIDATE: "PASS",
    VerificationStatus.INCONCLUSIVE: "????",
    VerificationStatus.REJECTED: "FAIL",
}


def emit(text: str = "") -> None:
    """Print, degrading gracefully rather than aborting the run.

    A rendering failure must never destroy a completed verification. The
    fallback below is the last line of defence if reconfigure() was
    unavailable (a redirected or wrapped stream).
    """
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", "ascii") or "ascii"
        print(text.encode(encoding, errors="replace").decode(encoding), flush=True)


def header(text: str) -> None:
    emit()
    emit(RULE)
    emit(text)
    emit(RULE)


def load_discovery_artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("discovery artifact not found: {0}".format(path))
    return json.loads(path.read_text(encoding="utf-8"))


def candidates_from_artifact(payload: dict[str, Any]) -> list[Candidate]:
    """Rehydrate Phase 0 candidates. Nothing is filtered or reordered here."""
    return [Candidate(**entry) for entry in payload.get("candidates", [])]


def render_candidate(index: int, result) -> None:
    emit()
    emit("#{0:<3} {1}".format(index + 1, result.candidate_id))
    emit("     source   : {0}".format((result.source_url or "-")[:66]))
    emit("     media    : {0}".format((result.media_url or "-")[:66]))

    acquired = result.acquisition or {}
    if acquired.get("ok"):
        emit(
            "     ACQUIRED  OK    {0} bytes, HTTP {1}{2}".format(
                acquired.get("byte_size"),
                acquired.get("status_code"),
                "  (redirected)" if acquired.get("url_changed") else "",
            )
        )
    else:
        emit("     ACQUIRED  FAIL  {0}".format(acquired.get("detail", "not attempted")))

    validated = result.validation or {}
    if validated.get("ok"):
        emit(
            "     IMAGE     OK    {0} {1}x{2}{3}".format(
                validated.get("detected_format"),
                validated.get("width"),
                validated.get("height"),
                ""
                if validated.get("content_type_was_honest")
                else "   [Content-Type disagreed with the bytes]",
            )
        )
    elif validated:
        emit("     IMAGE     FAIL  {0}".format(validated.get("detail")))

    if result.face:
        face = result.face
        emit(
            "     FACE      OK    {0} detected, det={1:.3f}, quality={2:.3f} [{3}]".format(
                face.faces_detected,
                face.det_score,
                face.quality_aggregate,
                face.quality_band,
            )
        )
    elif result.content_sha256:
        emit("     FACE      FAIL  no face in the bytes we downloaded")

    if result.face_similarity is not None:
        relation = result.relation
        if result.identity_probability is not None:
            emit(
                "     FACE SIM  {0:.4f}   band={1}   P(identity)={2:.3f}  "
                "[CALIBRATED]".format(
                    result.face_similarity, result.similarity_band.value,
                    result.identity_probability,
                )
            )
        else:
            emit(
                "     FACE SIM  {0:.4f}   band={1}  (PROVISIONAL, uncalibrated)".format(
                    result.face_similarity, result.similarity_band.value
                )
            )
        if relation and relation.phash_distance is not None:
            emit(
                "     pHASH     distance={0}/64  similarity={1:.4f}".format(
                    relation.phash_distance, relation.phash_similarity
                )
            )
            emit("     RELATION  {0}".format(relation.relation.value))

    if result.duplicate and result.duplicate.is_exact_duplicate:
        emit(
            "     DUPLICATE byte-identical to {0}".format(
                result.duplicate.duplicate_of_candidate_id
            )
        )
    if result.duplicate and result.duplicate.near_duplicate_of_candidate_ids:
        emit(
            "     NEAR-DUP  visually matches {0} (different bytes)".format(
                ", ".join(result.duplicate.near_duplicate_of_candidate_ids)
            )
        )

    emit("     STATUS    {0}  {1}".format(STATUS_MARK[result.status], result.status.value))
    emit("     REASON    {0}".format(result.explain()))

    for warning in result.warnings[:3]:
        emit("     warn      {0}".format(warning[:70]))


def render_table(results: list) -> None:
    header("VERIFICATION TABLE")
    emit(
        "{0:<4} {1:<22} {2:<9} {3:<6} {4:<9} {5:<7} {6}".format(
            "#", "SOURCE", "DOWNLOAD", "FACE", "FACE SIM", "pHASH", "DECISION"
        )
    )
    emit(THIN)

    for index, result in enumerate(results):
        acquired = (result.acquisition or {}).get("ok")
        source = (result.provenance or {}).get("registrable_domain") or "-"

        download = "ok" if acquired else "FAILED"
        if result.validation and not result.validation.get("ok"):
            download = "not-image"

        # Distinguish "we looked and found no face" from "we never looked".
        # A duplicate is rejected at VALIDATED, before face analysis runs;
        # rendering both as "none" would misattribute the rejection.
        face = "-"
        if result.face:
            face = "{0} found".format(result.face.faces_detected)
        elif result.primary_reason is RejectionReason.DUPLICATE_CONTENT:
            face = "n/a dup"
        elif result.primary_reason is RejectionReason.NO_FACE_DETECTED:
            face = "none"
        elif result.content_sha256:
            face = "not run"

        similarity = (
            "{0:.4f}".format(result.face_similarity)
            if result.face_similarity is not None
            else "-"
        )
        probability = (
            "{0:.3f}".format(result.identity_probability)
            if result.identity_probability is not None
            else "-"
        )
        phash = (
            "{0}/64".format(result.relation.phash_distance)
            if result.relation and result.relation.phash_distance is not None
            else "-"
        )

        emit(
            "{0:<4} {1:<22} {2:<9} {3:<8} {4:<9} {5:<7} {6:<7} {7}".format(
                index + 1, source[:22], download, face, similarity, probability,
                phash, result.status.value,
            )
        )

    emit(THIN)
    emit("Rejected candidates are shown deliberately. They are the evidence that")
    emit("verification is real: a system that accepts everything has tested nothing.")


def build_artifact(
    *,
    run_id: str,
    discovery_path: Path,
    discovery_payload: dict[str, Any],
    probe,
    results: list,
    policy: VerificationPolicy,
    fetch_policy: FetchPolicy,
    store: ContentAddressedStore,
    limit: int | None,
) -> dict[str, Any]:
    """Structured evidence artifact, shaped for Phase 4's notary to consume."""
    counts = {
        "discovered": len(discovery_payload.get("candidates", [])),
        "examined": len(results),
        "verified": sum(1 for r in results if r.status is VerificationStatus.VERIFIED_CANDIDATE),
        "inconclusive": sum(1 for r in results if r.status is VerificationStatus.INCONCLUSIVE),
        "rejected": sum(1 for r in results if r.status is VerificationStatus.REJECTED),
    }

    rejection_breakdown: dict[str, int] = {}
    for result in results:
        for reason in result.rejection_reasons:
            rejection_breakdown[reason.value] = rejection_breakdown.get(reason.value, 0) + 1

    return {
        "schema_version": "verification-run/1",
        "run_id": run_id,
        "phase": "2-acquisition-verification",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_artifact": {
            "path": str(discovery_path),
            "run_id": discovery_payload.get("run_id"),
            "provider": discovery_payload.get("provider"),
            "candidate_count": len(discovery_payload.get("candidates", [])),
        },
        "probe": {
            # The probe image hash ties this run to Phase 0's artifact.
            "sha256": probe.image.sha256,
            "path": probe.image.path,
            "model_id": probe.model.model_id,
            "embedding_dimension": probe.embedding.dimension,
            # A commitment to the embedding, never the embedding itself.
            "embedding_quantized_sha256": __import__("hashlib")
            .sha256(probe.embedding.quantize())
            .hexdigest(),
        },
        "configuration": {
            "limit": limit,
            "verification_policy": policy.to_dict(),
            "fetch_policy": fetch_policy.to_dict(),
        },
        "counts": counts,
        "rejection_breakdown": rejection_breakdown,
        "cas": store.stats(),
        "results": [r.to_dict() for r in results],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="verify_candidates",
        description="Phase 2: independently re-acquire and verify discovered candidates.",
    )
    parser.add_argument(
        "--run", type=Path, required=True,
        help="Phase 0 discovery artifact (data/runs/search_gate_*.json)",
    )
    parser.add_argument(
        "--probe",
        type=Path,
        default=None,
        help="Probe image. Defaults to the one recorded in the discovery "
        "artifact -- verifying against a different face is almost always a "
        "mistake.",
    )
    parser.add_argument(
        "--allow-probe-mismatch",
        action="store_true",
        help="Permit verifying against a probe that is NOT the image the "
        "discovery run searched. Results become meaningless unless you know "
        "exactly why you want this.",
    )
    parser.add_argument(
        "--limit", type=int, default=5,
        help="How many candidates to verify. Keeps network cost controlled.",
    )
    parser.add_argument("--cas", type=Path, default=Path("data/cas"))
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("data/calibration/model.json"),
        help="Fitted calibration model. When present, thresholds and identity "
        "probabilities come from it instead of provisional constants.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--similarity-floor", type=float, default=None)
    parser.add_argument("--similarity-ceiling", type=float, default=None)
    parser.add_argument(
        "--reject-low-quality", action="store_true",
        help="Reject rather than warn when face quality is below the floor.",
    )
    args = parser.parse_args(argv)

    header("TRACELOCK - PHASE 2 - ACQUISITION + VERIFICATION FIREWALL")
    emit("DISCOVERY MAKES CLAIMS.  TRACELOCK RE-MEASURES EVIDENCE.")

    # --- load the real discovery artifact ------------------------------
    try:
        discovery = load_discovery_artifact(args.run)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        emit("\nFAILED to load discovery artifact: {0}".format(exc))
        return 2

    all_candidates = candidates_from_artifact(discovery)
    if not all_candidates:
        emit("\nThe discovery artifact contains no candidates.")
        return 2

    candidates = all_candidates[: args.limit] if args.limit else all_candidates

    emit()
    emit("SEARCH ENGINE ({0}/{1}):".format(
        (discovery.get("provider") or {}).get("name", "?"),
        (discovery.get("provider") or {}).get("engine", "?"),
    ))
    emit('  "I found {0} possible candidates."'.format(len(all_candidates)))
    emit()
    emit("TRACELOCK:")
    emit('  "I will independently re-download and re-measure {0} of them."'.format(
        len(candidates)
    ))

    # --- probe ----------------------------------------------------------
    # The probe MUST be the image discovery was seeded with. Comparing
    # candidates against a different face silently produces impostor-level
    # similarities that look like a real negative result -- the same class of
    # bug Stage 2.5 catches one layer up.
    recorded = discovery.get("probe") or {}
    probe_path = args.probe or Path(recorded.get("local_path") or "data/probes/me.jpg")

    emit()
    emit("Loading face engine ...")
    try:
        engine = FaceEngine()
        probe = engine.analyze(probe_path)
    except FaceEngineError as exc:
        emit("FAILED to analyze the probe: {0}".format(exc))
        return 2

    emit("  probe    : {0}".format(args.probe))
    emit("  sha256   : {0}".format(probe.image.sha256))
    emit("  model    : {0}".format(probe.model.model_id))
    emit("  face     : det={0:.4f}  quality={1:.4f}".format(
        probe.primary.det_score, probe.primary.quality.aggregate
    ))

    recorded_sha = recorded.get("sha256")
    if recorded_sha and probe.image.sha256 != recorded_sha:
        if not args.allow_probe_mismatch:
            header("PROBE MISMATCH - REFUSING TO RUN")
            emit("The discovery run searched a DIFFERENT image than this probe.")
            emit()
            emit("  discovery searched : {0}".format(recorded_sha))
            emit("  this probe         : {0}".format(probe.image.sha256))
            emit("  probe path         : {0}".format(probe_path))
            emit()
            emit("Verifying candidates against a face that was never searched")
            emit("produces impostor-level similarities that LOOK like a genuine")
            emit("negative result. That is worse than no answer.")
            emit()
            emit("Pass --probe pointing at the searched image, or")
            emit("--allow-probe-mismatch if the mismatch is deliberate.")
            emit()
            return 3

        emit()
        emit("  WARNING: probe differs from the discovery image "
             "(--allow-probe-mismatch). Similarities below are NOT meaningful "
             "as identity evidence.")

    # --- verify ---------------------------------------------------------
    policy_kwargs: dict[str, Any] = {"reject_on_low_quality": args.reject_low_quality}
    if args.similarity_floor is not None:
        policy_kwargs["similarity_floor"] = args.similarity_floor
    if args.similarity_ceiling is not None:
        policy_kwargs["similarity_ceiling"] = args.similarity_ceiling

    # Load the fitted calibration if one exists. Its presence is what makes
    # `policy.calibrated` true -- there is no flag to set.
    calibration = None
    if args.calibration and Path(args.calibration).is_file():
        from tracelock.calibration.model import CalibrationModel

        calibration = CalibrationModel.load(args.calibration)

    if calibration is not None:
        policy = VerificationPolicy.from_calibration(calibration, **policy_kwargs)
    else:
        policy = VerificationPolicy(**policy_kwargs)
    fetch_policy = FetchPolicy()
    store = ContentAddressedStore(args.cas)

    header("INDEPENDENT VERIFICATION")
    if policy.calibrated:
        emit("thresholds: floor={0:.4f} ceiling={1:.4f}   CALIBRATED".format(
            policy.similarity_floor, policy.similarity_ceiling))
        emit("model     : {0}  ({1} genuine / {2} impostor, AUC {3:.4f})".format(
            calibration.model_id, calibration.n_genuine,
            calibration.n_impostor, calibration.auc))
        emit(calibration.confidence_note())
    else:
        emit("thresholds: floor={0} ceiling={1}  PROVISIONAL, UNCALIBRATED".format(
            policy.similarity_floor, policy.similarity_ceiling))
        emit("The {0:.2f}-wide indeterminate band is admitted uncertainty, not "
             "sloppiness.".format(policy.inconclusive_band_width))

    verifier = CandidateVerifier(
        engine, store=store, policy=policy, fetch_policy=fetch_policy
    )

    results = []
    for index, candidate in enumerate(candidates):
        result = verifier.verify(probe, candidate, index)
        results.append(result)
        render_candidate(index, result)

    # --- summary --------------------------------------------------------
    render_table(results)

    verified = [r for r in results if r.status is VerificationStatus.VERIFIED_CANDIDATE]
    inconclusive = [r for r in results if r.status is VerificationStatus.INCONCLUSIVE]
    rejected = [r for r in results if r.status is VerificationStatus.REJECTED]

    header("SUMMARY")
    emit("  discovered by search engine : {0}".format(len(all_candidates)))
    emit("  independently examined      : {0}".format(len(results)))
    emit("  ---")
    emit("  VERIFIED                    : {0}".format(len(verified)))
    emit("  INCONCLUSIVE                : {0}".format(len(inconclusive)))
    emit("  REJECTED                    : {0}".format(len(rejected)))

    if rejected:
        emit()
        emit("  rejection reasons:")
        breakdown: dict[str, int] = {}
        for result in rejected:
            for reason in result.rejection_reasons:
                key = "{0} (at {1})".format(reason.value, reason.stage.value)
                breakdown[key] = breakdown.get(key, 0) + 1
        for reason, count in sorted(breakdown.items(), key=lambda kv: -kv[1]):
            emit("    {0:>2}x  {1}".format(count, reason))

    emit()
    emit("  CAS: {0} blobs, {1} bytes".format(
        store.stats()["blob_count"], store.stats()["total_bytes"]
    ))

    # --- artifact -------------------------------------------------------
    run_id = make_run_id("verify", probe.image.sha256)
    out_path = args.out or run_artifact_path(Path("data/runs"), run_id)
    artifact = build_artifact(
        run_id=run_id,
        discovery_path=args.run,
        discovery_payload=discovery,
        probe=probe,
        results=results,
        policy=policy,
        fetch_policy=fetch_policy,
        store=store,
        limit=args.limit,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    header("EVIDENCE ARTIFACT")
    emit("  {0}".format(out_path))
    emit("  {0} bytes".format(out_path.stat().st_size))
    emit("  (no API keys, no raw embeddings -- quantized digests only)")
    emit("  Shaped for the Phase 4 notary to consume.")
    emit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
