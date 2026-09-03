"""Phase 3 -- aggregate verified evidence and produce the final report.

Runs entirely on a persisted Phase 2 artifact. No network, no API credits, no
re-download: aggregation and scoring are reproducible from the JSON alone.

    python scripts\\build_evidence.py --verify data\\runs\\verify_....json
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tracelock.core.runs import make_run_id, run_artifact_path  # noqa: E402
from tracelock.evidence.aggregate import aggregate  # noqa: E402
from tracelock.evidence.trust import (  # noqa: E402
    UncalibratedScoreRefused,
    measure_acquisition_integrity,
    measure_metadata_completeness,
    score_evidence,
)

RULE = "=" * 78
THIN = "-" * 78


def emit(text: str = "") -> None:
    print(text, flush=True)


def header(text: str) -> None:
    emit()
    emit(RULE)
    emit(text)
    emit(RULE)


def render_funnel(funnel) -> None:
    header("PIPELINE FUNNEL")
    rows = [
        ("discovered by search engine", funnel.discovered, ""),
        ("downloaded by TRACELOCK", funnel.downloaded, "independently re-fetched"),
        ("validated as real images", funnel.validated, "magic bytes, not headers"),
        ("analysed for faces", funnel.analysed, "detection re-run on our bytes"),
        ("VERIFIED", funnel.verified, ""),
        ("inconclusive", funnel.inconclusive, ""),
        ("rejected", funnel.rejected, ""),
        ("  of which duplicates", funnel.duplicates, "byte-identical"),
    ]
    for label, value, note in rows:
        emit("  {0:<32} {1:>4}   {2}".format(label, value, note))

    emit()
    emit("  AFTER DE-DUPLICATION")
    emit("  {0:<32} {1:>4}   {2}".format(
        "unique images", funnel.unique_images,
        "the same photo republished counts ONCE"))
    emit("  {0:<32} {1:>4}   {2}".format(
        "independent publishers", funnel.independent_publishers,
        "distinct eTLD+1, one vote each"))


def render_evidence(evidence, show: int) -> None:
    header("VERIFIED EVIDENCE  (unique images, de-duplicated)")
    if not evidence.has_evidence:
        emit("  none")
        return

    emit("{0:<4} {1:<24} {2:<8} {3:<8} {4:<7} {5}".format(
        "#", "PUBLISHER(S)", "P(ID)", "SIM", "pHASH", "RELATION"))
    emit(THIN)

    for index, item in enumerate(evidence.items[:show], 1):
        publishers = ", ".join(sorted(set(item.domains))) or "-"
        emit("{0:<4} {1:<24} {2:<8} {3:<8} {4:<7} {5}".format(
            index,
            publishers[:24],
            "{0:.3f}".format(item.identity_probability)
            if item.identity_probability is not None else "-",
            "{0:.4f}".format(item.face_similarity),
            "{0}/64".format(item.phash_distance) if item.phash_distance is not None else "-",
            item.relation,
        ))

    if len(evidence.items) > show:
        emit("  ... {0} more in the artifact".format(len(evidence.items) - show))

    if evidence.duplicate_groups:
        emit()
        emit("  DUPLICATE GROUPS  (same bytes, multiple candidates -> counted once)")
        for digest, ids in list(evidence.duplicate_groups.items())[:5]:
            emit("    {0}...  {1} candidates".format(digest[:16], len(ids)))


def render_score(score) -> None:
    header("TRUST SCORE")
    for line in score.explain():
        emit("  " + line)

    emit()
    emit("  QUALITY BREAKDOWN")
    for component in score.quality.components:
        emit("    {0:<24} value={1:.4f}  weight={2:.2f}  ->  {3:.4f}".format(
            component.name, component.value, component.weight, component.contribution))
        emit("      {0}".format(component.note))

    emit()
    emit("  CALIBRATION: {0}".format("YES" if score.calibrated else "NO"))
    if score.calibration_note:
        emit("    {0}".format(score.calibration_note))

    header("LIMITATIONS  (what this score does NOT establish)")
    for limitation in score.limitations:
        emit("  - {0}".format(limitation))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build_evidence",
        description="Phase 3: aggregate verified evidence and score it.",
    )
    parser.add_argument("--verify", type=Path, required=True,
                        help="Phase 2 verification artifact.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--show", type=int, default=15)
    args = parser.parse_args(argv)

    if not args.verify.is_file():
        emit("No verification artifact at {0}".format(args.verify))
        return 2

    payload = json.loads(args.verify.read_text(encoding="utf-8"))
    results = payload.get("results", [])

    header("TRACELOCK - PHASE 3 - EVIDENCE AGGREGATION & TRUST SCORING")
    emit("Duplicates counted once. One publisher, one vote.")
    emit()
    emit("  source artifact : {0}".format(args.verify))
    emit("  probe sha256    : {0}".format((payload.get("probe") or {}).get("sha256", "?")))
    emit("  model           : {0}".format((payload.get("probe") or {}).get("model_id", "?")))

    evidence = aggregate(results)
    render_funnel(evidence.funnel)
    render_evidence(evidence, args.show)

    if evidence.independent_domains:
        emit()
        emit("  INDEPENDENT PUBLISHERS ({0}):".format(len(evidence.independent_domains)))
        for domain in evidence.independent_domains:
            emit("    - {0}".format(domain))

    policy = (payload.get("configuration") or {}).get("verification_policy") or {}
    model_info = policy.get("calibration_model") or {}
    calibration_note = model_info.get("confidence_note", "")
    calibrated = bool(policy.get("calibrated"))

    score = None
    try:
        score = score_evidence(
            evidence,
            metadata_completeness=measure_metadata_completeness(results),
            acquisition_integrity=measure_acquisition_integrity(results),
            calibration_note=calibration_note,
            calibrated=calibrated,
        )
        render_score(score)
    except UncalibratedScoreRefused as exc:
        header("NO TRUST SCORE PRODUCED")
        emit("  {0}".format(exc))
        emit()
        emit("  Refusing to emit a number is the correct outcome here. A score")
        emit("  built on uncalibrated similarity would be a fabricated statistic.")

    # --- artifact ---------------------------------------------------------
    probe_sha = (payload.get("probe") or {}).get("sha256", "0" * 64)
    run_id = make_run_id("evidence", probe_sha)
    out_path = args.out or run_artifact_path(Path("data/runs"), run_id)

    artifact: dict[str, Any] = {
        "schema_version": "evidence-report/1",
        "run_id": run_id,
        "phase": "3-evidence-aggregation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_artifacts": {
            "verification": str(args.verify),
            "verification_run_id": payload.get("run_id"),
            "discovery": (payload.get("source_artifact") or {}).get("path"),
        },
        "probe": payload.get("probe"),
        "verification_policy": policy,
        "evidence": evidence.to_dict(),
        "trust_score": score.to_dict() if score else None,
        "scored": score is not None,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")

    header("EVIDENCE ARTIFACT")
    emit("  {0}".format(out_path))
    emit("  {0} bytes".format(out_path.stat().st_size))
    emit("  Ready for the Phase 4 notary to anchor.")
    emit()

    return 0 if score else 1


if __name__ == "__main__":
    raise SystemExit(main())
