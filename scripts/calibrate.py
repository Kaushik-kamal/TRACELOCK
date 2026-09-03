"""Fit TRACELOCK's calibration model from completed verification runs.

No API credits. No new downloads. Every similarity was already measured by the
production face engine during Phase 2; this only supplies labels and fits.

    # 1. propose labels and write a reviewable manifest
    python scripts\\calibrate.py propose

    # 2. read data\\calibration\\manifest.json, edit if you disagree

    # 3. fit
    python scripts\\calibrate.py fit

LABELLING IS DECLARED, NOT INFERRED
-----------------------------------
Rules live in `--rules` (default data/calibration/rules.json). Each names a
probe, the label to apply to everything measured against it, and the BASIS for
that claim. Nothing here guesses identity.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tracelock.calibration.contract import PairLabel  # noqa: E402
from tracelock.calibration.from_runs import (  # noqa: E402
    ProbeLabelRule,
    build_observation_set,
    deduplicate,
    load_manifest,
    load_measurements,
    propose_labels,
    write_manifest,
)
from tracelock.calibration.metrics import evaluate  # noqa: E402
from tracelock.calibration.model import fit_calibration_model  # noqa: E402

RULE = "=" * 74
DEFAULT_RULES = Path("data/calibration/rules.json")
DEFAULT_MANIFEST = Path("data/calibration/manifest.json")
DEFAULT_MODEL = Path("data/calibration/model.json")


def emit(text: str = "") -> None:
    print(text, flush=True)


def header(text: str) -> None:
    emit()
    emit(RULE)
    emit(text)
    emit(RULE)


def load_rules(path: Path) -> list[ProbeLabelRule]:
    if not path.is_file():
        raise FileNotFoundError(
            "no labelling rules at {0}.\n"
            "Rules must be declared explicitly -- identity is never inferred "
            "automatically. See docs/PHASE3_CALIBRATION.md for the format.".format(path)
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        ProbeLabelRule(
            probe_sha256_prefix=row["probe_sha256_prefix"],
            label=PairLabel(row["label"]),
            basis=row["basis"],
            confidence=row.get("confidence", "operator"),
            subject_ref=row.get("subject_ref", ""),
            domains=tuple(row.get("domains", ())),
        )
        for row in payload.get("rules", [])
    ]


def gather(runs_glob: str) -> list[dict]:
    paths = sorted(glob.glob(runs_glob))
    if not paths:
        raise FileNotFoundError("no verification runs matched {0}".format(runs_glob))

    raw = load_measurements(paths)
    unique = deduplicate(raw)

    emit("  runs scanned        : {0}".format(len(paths)))
    emit("  measurements found  : {0}".format(len(raw)))
    emit(
        "  unique (probe,blob) : {0}   [{1} repeat measurements collapsed]".format(
            len(unique), len(raw) - len(unique)
        )
    )
    if len(raw) != len(unique):
        emit(
            "  NOTE: repeat runs of the same candidates would otherwise inflate n\n"
            "        and make the confidence interval dishonest."
        )
    return unique


def command_propose(args) -> int:
    header("CALIBRATION -- PROPOSE LABELS")
    emit("Proposals are reviewable evidence, not ground truth.")
    emit()

    rules = load_rules(args.rules)
    emit("  labelling rules     : {0}".format(len(rules)))
    for rule in rules:
        scope = ", ".join(rule.domains) if rule.domains else "all domains"
        emit(
            "    probe {0}... -> {1:<8} [{2}]  scope: {3}".format(
                rule.probe_sha256_prefix, rule.label.value, rule.confidence, scope
            )
        )
    emit()

    measurements = gather(args.runs)
    proposals, unlabelled = propose_labels(measurements, rules)

    genuine = [p for p in proposals if p.label is PairLabel.GENUINE]
    impostor = [p for p in proposals if p.label is PairLabel.IMPOSTOR]

    emit()
    emit("  proposed GENUINE    : {0}".format(len(genuine)))
    emit("  proposed IMPOSTOR   : {0}".format(len(impostor)))
    emit("  unlabelled          : {0}  (no rule matched their probe)".format(len(unlabelled)))

    if genuine:
        values = sorted(p.similarity for p in genuine)
        emit()
        emit("  genuine similarity  : {0:.4f} .. {1:.4f}".format(values[0], values[-1]))
        emit("    domains: {0}".format(
            ", ".join(sorted({p.domain for p in genuine if p.domain})[:8])
        ))
    if impostor:
        values = sorted(p.similarity for p in impostor)
        emit("  impostor similarity : {0:.4f} .. {1:.4f}".format(values[0], values[-1]))

    if genuine and impostor:
        gap = min(p.similarity for p in genuine) - max(p.similarity for p in impostor)
        emit()
        emit("  SEPARATION          : {0:+.4f}".format(gap))
        emit(
            "    {0}".format(
                "populations do not overlap in this sample"
                if gap > 0
                else "POPULATIONS OVERLAP -- no threshold separates them cleanly"
            )
        )

    path = write_manifest(proposals, unlabelled, args.manifest, rules=rules)
    header("MANIFEST WRITTEN")
    emit("  {0}".format(path))
    emit()
    emit("  REVIEW IT before fitting. Every row records the basis for its")
    emit("  proposed label. Delete or edit any row you disagree with, then:")
    emit()
    emit("      python scripts\\calibrate.py fit")
    emit()
    return 0


def command_fit(args) -> int:
    header("CALIBRATION -- FIT")

    if not Path(args.manifest).is_file():
        emit("No manifest at {0}. Run `propose` first.".format(args.manifest))
        return 2

    proposals = load_manifest(args.manifest)
    genuine = [p.similarity for p in proposals if p.label is PairLabel.GENUINE]
    impostor = [p.similarity for p in proposals if p.label is PairLabel.IMPOSTOR]

    emit("  manifest            : {0}".format(args.manifest))
    emit("  genuine pairs       : {0}".format(len(genuine)))
    emit("  impostor pairs      : {0}".format(len(impostor)))

    if not genuine or not impostor:
        emit()
        emit("Both classes are required. Cannot fit.")
        return 2

    bases = sorted({p.basis for p in proposals})
    labeling_basis = " | ".join(bases)

    observations = build_observation_set(
        proposals,
        name=args.name,
        description=(
            "Similarity measurements harvested from completed TRACELOCK "
            "verification runs. Labels proposed by declared rules and reviewed "
            "by the operator."
        ),
        model_id=args.model_id,
    )

    report = evaluate(observations)
    model = fit_calibration_model(
        genuine,
        impostor,
        provenance=observations.provenance,
        labeling_basis=labeling_basis,
        model_id=args.model_id,
    )

    header("EVALUATION")
    emit(report.render())

    header("FITTED MODEL")
    emit(model.render())

    emit()
    emit("  labelling basis:")
    for basis in bases:
        emit("    - {0}".format(basis))

    saved = model.save(args.out)
    header("MODEL SAVED")
    emit("  {0}".format(saved))
    emit()
    emit("  Phase 2 and Phase 3 read this file. VerificationPolicy derives")
    emit("  `calibrated` from its presence -- there is no flag to set.")
    emit()

    if not model.is_separable:
        emit("  WARNING: the two populations OVERLAP. Any threshold will make")
        emit("  both kinds of error. Treat every downstream verdict accordingly.")
        emit()

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="calibrate",
        description="Fit TRACELOCK's identity-probability calibration.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = {
        "--runs": ("data/runs/verify_*.json", "Glob of verification run artifacts."),
        "--rules": (str(DEFAULT_RULES), "Declared labelling rules."),
        "--manifest": (str(DEFAULT_MANIFEST), "Reviewable label manifest."),
    }

    propose = sub.add_parser("propose", help="Propose labels and write a manifest.")
    for flag, (default, help_text) in common.items():
        propose.add_argument(flag, type=Path if flag != "--runs" else str,
                             default=default, help=help_text)

    fit = sub.add_parser("fit", help="Fit the model from a reviewed manifest.")
    fit.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    fit.add_argument("--out", type=Path, default=DEFAULT_MODEL)
    fit.add_argument("--name", default="tracelock-runs-v1")
    fit.add_argument("--model-id", default="buffalo_l:w600k_r50.onnx")

    args = parser.parse_args(argv)

    try:
        if args.command == "propose":
            return command_propose(args)
        return command_fit(args)
    except (FileNotFoundError, ValueError) as exc:
        emit()
        emit("FAILED: {0}".format(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
