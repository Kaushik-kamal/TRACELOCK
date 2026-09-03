"""Phase 4 -- re-verify an anchored evidence artifact.

    python scripts\\verify_anchor.py --evidence data\\runs\\evidence_....json

Recomputes every leaf from the artifact on disk, rebuilds the Merkle root, and
compares it against the record on chain. On a mismatch it names WHICH leaf
changed -- tamper localisation, which is the whole reason for the Merkle tree.

    --tamper-demo <field>   deliberately corrupt one field in a COPY of the
                            artifact and show the detection. Nothing on disk or
                            on chain is modified.

Three verdicts, and the difference between the last two matters:
    INTACT        the artifact is unchanged since it was anchored
    TAMPERED      it has been modified
    NOT_ANCHORED  the chain has no record of this root
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tracelock.chain.compiler import load_or_compile  # noqa: E402
from tracelock.chain.config import (  # noqa: E402
    build_adapter_with_contract,
    load_chain_config,
)
from tracelock.chain.errors import ChainError  # noqa: E402
from tracelock.chain.notary import (  # noqa: E402
    AnchorRecord,
    EvidenceNotary,
    VerificationVerdict,
)

RULE = "=" * 74

EXIT_INTACT = 0
EXIT_TAMPERED = 1
EXIT_FAILED = 2
EXIT_NOT_ANCHORED = 5

# Fields the tamper demo can corrupt, and how.
TAMPER_TARGETS = {
    "trust_score": ("trust_score", "score", 99.99),
    "publishers": ("evidence", "independent_domains", ["fabricated.example"]),
    "run_id": (None, "run_id", "evidence_TAMPERED"),
}


def emit(text: str = "") -> None:
    print(text, flush=True)


def header(text: str) -> None:
    emit()
    emit(RULE)
    emit(text)
    emit(RULE)


def apply_tamper(artifact: dict, target: str) -> tuple[dict, str]:
    """Corrupt one field in a COPY. The original is never touched."""
    if target not in TAMPER_TARGETS:
        raise ValueError(
            "unknown tamper target {0!r}; choose from {1}".format(
                target, sorted(TAMPER_TARGETS)
            )
        )

    mutated = copy.deepcopy(artifact)
    section, field, value = TAMPER_TARGETS[target]

    if section is None:
        before = mutated.get(field)
        mutated[field] = value
    else:
        before = mutated.setdefault(section, {}).get(field)
        mutated[section][field] = value

    return mutated, "{0} : {1!r} -> {2!r}".format(field, before, value)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="verify_anchor")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument(
        "--anchor",
        type=Path,
        default=None,
        help="Anchor record. Defaults to <evidence>.anchor.json",
    )
    parser.add_argument("--network", default=None)
    parser.add_argument(
        "--tamper-demo",
        default=None,
        choices=sorted(TAMPER_TARGETS),
        help="Corrupt one field in a copy and show the detection.",
    )
    args = parser.parse_args(argv)

    header("TRACELOCK - PHASE 4 - RE-VERIFICATION")

    if not args.evidence.is_file():
        emit("No evidence artifact at {0}".format(args.evidence))
        return EXIT_FAILED

    anchor_path = args.anchor or args.evidence.with_suffix(".anchor.json")
    if not anchor_path.is_file():
        emit("No anchor record at {0}".format(anchor_path))
        emit("Anchor the artifact first: scripts/anchor_evidence.py")
        return EXIT_FAILED

    artifact = json.loads(args.evidence.read_text(encoding="utf-8"))
    record = AnchorRecord.from_dict(
        json.loads(anchor_path.read_text(encoding="utf-8"))
    )

    emit("  artifact     : {0}".format(args.evidence))
    emit("  anchor record: {0}".format(anchor_path))
    emit()
    emit("  ANCHORED AT")
    emit("    tx         : {0}".format(record.tx_hash))
    emit("    chain      : {0} ({1})".format(record.chain_id, record.network))
    emit("    contract   : {0}".format(record.contract_address))
    emit("    block      : {0}  {1}".format(record.block_number, record.block_time_utc))
    emit("    submitter  : {0}".format(record.submitter))
    if record.explorer_url:
        emit("    explorer   : {0}".format(record.explorer_url))

    tamper_note = ""
    if args.tamper_demo:
        artifact, tamper_note = apply_tamper(artifact, args.tamper_demo)
        header("TAMPER DEMONSTRATION")
        emit("  Corrupting a COPY of the artifact in memory.")
        emit("  Neither the file on disk nor the chain is modified.")
        emit()
        emit("    {0}".format(tamper_note))

    # --- re-verify --------------------------------------------------------
    network = args.network or record.network or None
    config = load_chain_config(
        network=network, contract_address=record.contract_address
    )

    emit()
    emit("Recomputing the artifact and querying the chain ...")
    try:
        compiled = load_or_compile()
        adapter, address, auto_deployed = build_adapter_with_contract(
            config, compiled
        )
        if auto_deployed:
            emit()
            emit("  NOTE: the local chain is ephemeral -- the original anchor")
            emit("  no longer exists, so this will report NOT_ANCHORED. Use a")
            emit("  public testnet for an anchor that persists across runs.")
            record = AnchorRecord.from_dict(
                {**record.to_dict(),
                 "on_chain": {**record.to_dict()["on_chain"],
                              "contract_address": address}}
            )
        notary = EvidenceNotary(adapter, record.contract_address)
        result = notary.reverify(artifact, record)
    except ChainError as exc:
        header("RE-VERIFICATION COULD NOT COMPLETE")
        emit("  {0}".format(exc))
        emit()
        emit("  This is NOT a tamper finding -- the chain could not be reached,")
        emit("  so the artifact state is unknown.")
        return EXIT_FAILED

    # --- verdict ----------------------------------------------------------
    header("VERDICT: {0}".format(result.verdict.value))
    emit("  {0}".format(result.verdict.explanation))
    emit()
    emit("  anchored root   : {0}".format(result.expected_root))
    emit("  recomputed root : {0}".format(result.recomputed_root))
    emit("  match           : {0}".format(result.expected_root == result.recomputed_root))

    if result.on_chain and result.on_chain.exists:
        emit()
        emit("  ON-CHAIN RECORD")
        emit("    trust score   : {0}".format(result.on_chain.trust_score))
        emit("    unique images : {0}".format(result.on_chain.evidence_count))
        emit("    submitter     : {0}".format(result.on_chain.submitter))
        emit("    block time    : {0}".format(result.on_chain.timestamp))

    if result.changed_leaves:
        emit()
        emit("  TAMPER LOCALISED TO {0} LEAF/LEAVES".format(len(result.changed_leaves)))
        emit("  {0:<26} {1:<20} {2}".format("leaf", "anchored", "recomputed"))
        emit("  " + "-" * 68)
        for tag, before, after in result.changed_leaves:
            emit(
                "  {0:<26} {1:<20} {2}".format(
                    tag, (before or "")[:18], (after or "")[:18]
                )
            )
        emit()
        emit("  Every other leaf is unchanged. This is what the Merkle")
        emit("  structure buys: not just that something changed, but what.")

    if result.verdict is VerificationVerdict.INTACT:
        emit()
        emit("  The artifact is byte-for-byte the evidence that was notarised.")
        emit("  This proves it was not ALTERED. It does not prove it was CORRECT.")
        return EXIT_INTACT

    if result.verdict is VerificationVerdict.NOT_ANCHORED:
        return EXIT_NOT_ANCHORED

    return EXIT_TAMPERED


if __name__ == "__main__":
    raise SystemExit(main())
