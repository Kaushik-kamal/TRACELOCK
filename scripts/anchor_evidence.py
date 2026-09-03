"""Phase 4 -- anchor a Phase 3 evidence artifact on chain.

    python scripts\\anchor_evidence.py --evidence data\\runs\\evidence_....json

Only a 32-byte Merkle root and packed metadata go on chain. No images, no
embeddings, no personal data -- enforced by assert_no_biometric_leak before
anything is hashed.

Nothing is reported as anchored without a CONFIRMED receipt.
"""

from __future__ import annotations

import argparse
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
from tracelock.chain.errors import (  # noqa: E402
    AlreadyAnchoredError,
    AnchorNotConfirmedError,
    ChainError,
)
from tracelock.chain.fingerprint import (  # noqa: E402
    BiometricLeakError,
    fingerprint_evidence,
)
from tracelock.chain.notary import EvidenceNotary  # noqa: E402

RULE = "=" * 74

EXIT_OK = 0
EXIT_FAILED = 2
EXIT_ALREADY_ANCHORED = 3
EXIT_UNKNOWN_STATE = 4


def emit(text: str = "") -> None:
    print(text, flush=True)


def header(text: str) -> None:
    emit()
    emit(RULE)
    emit(text)
    emit(RULE)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="anchor_evidence")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--network", default=None)
    parser.add_argument("--contract", default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the commitment and stop. No transaction is sent.",
    )
    args = parser.parse_args(argv)

    header("TRACELOCK - PHASE 4 - ANCHOR EVIDENCE")

    if not args.evidence.is_file():
        emit("No evidence artifact at {0}".format(args.evidence))
        return EXIT_FAILED

    artifact = json.loads(args.evidence.read_text(encoding="utf-8"))
    trust = artifact.get("trust_score") or {}

    emit("  artifact    : {0}".format(args.evidence))
    emit("  run id      : {0}".format(artifact.get("run_id")))
    emit("  trust score : {0} [{1}]".format(trust.get("score"), trust.get("band")))

    if not artifact.get("scored"):
        emit()
        emit("  REFUSING: this artifact carries no trust score. Anchoring")
        emit("  unscored evidence would commit to an incomplete finding.")
        return EXIT_FAILED

    # --- commitment -------------------------------------------------------
    emit()
    emit("Computing commitment ...")
    try:
        fingerprint = fingerprint_evidence(artifact)
    except BiometricLeakError as exc:
        header("REFUSING TO ANCHOR - BIOMETRIC DATA PRESENT")
        emit("  {0}".format(exc))
        return EXIT_FAILED

    emit("  merkle root : {0}".format(fingerprint.merkle_root_hex))
    emit(
        "  leaves      : {0} domain-separated commitments".format(
            len(fingerprint.leaves)
        )
    )
    emit(
        "  probe       : {0}...  (commitment, NOT the image)".format(
            ("0x" + fingerprint.probe_commitment.hex())[:26]
        )
    )
    emit(
        "  on-chain    : trust {0}bp, {1} unique images, {2} publishers".format(
            fingerprint.trust_score_bp,
            fingerprint.evidence_count,
            fingerprint.independent_publishers,
        )
    )
    emit()
    emit("  NOTHING ELSE IS TRANSMITTED. No image bytes, no face embedding,")
    emit("  no URLs -- only the hashes above.")

    if args.dry_run:
        header("DRY RUN - NO TRANSACTION SENT")
        emit("  The commitment above is what WOULD be anchored.")
        return EXIT_OK

    # --- anchor -----------------------------------------------------------
    config = load_chain_config(network=args.network, contract_address=args.contract)
    emit()
    emit(
        "  network     : {0} (chain {1})".format(
            config.network_key, config.profile.chain_id
        )
    )
    emit("  contract    : {0}".format(config.contract_address or "(not set)"))
    emit("  account     : {0}".format(config.account_address() or "(no key set)"))

    if not config.contract_address and not config.is_local:
        emit()
        emit("  No contract address. Deploy first:")
        emit(
            "    python scripts\\deploy_contract.py --network {0}".format(
                config.network_key
            )
        )
        return EXIT_FAILED

    try:
        compiled = load_or_compile()
        adapter, address, auto_deployed = build_adapter_with_contract(
            config, compiled
        )
        if auto_deployed:
            emit()
            emit("  The local chain is in-process and ephemeral, so the notary")
            emit("  was deployed fresh for this run at")
            emit("    {0}".format(address))
        notary = EvidenceNotary(adapter, address)

        emit()
        emit("Submitting transaction and waiting for confirmation ...")
        record, _receipt = notary.anchor(
            artifact, confirmations=config.confirmations
        )

    except AlreadyAnchoredError as exc:
        header("ALREADY ANCHORED")
        emit("  {0}".format(exc))
        emit()
        emit("  The contract working as intended, not a fault.")
        return EXIT_ALREADY_ANCHORED
    except AnchorNotConfirmedError as exc:
        header("ANCHOR STATE UNKNOWN")
        emit("  {0}".format(exc))
        emit()
        emit("  NOT reporting this as anchored. Check the explorer, then")
        emit("  re-run verification once the transaction settles.")
        return EXIT_UNKNOWN_STATE
    except ChainError as exc:
        header("ANCHORING FAILED")
        emit("  {0}".format(exc))
        emit()
        emit("  Nothing was anchored.")
        return EXIT_FAILED

    header("ANCHORED")
    emit("  tx hash     : {0}".format(record.tx_hash))
    emit("  chain id    : {0}  ({1})".format(record.chain_id, record.network))
    emit("  contract    : {0}".format(record.contract_address))
    emit("  block       : {0}".format(record.block_number))
    emit("  block time  : {0}".format(record.block_time_utc))
    emit("  gas used    : {0}".format(record.gas_used))
    emit("  submitter   : {0}".format(record.submitter))
    if record.explorer_url:
        emit()
        emit("  VERIFY INDEPENDENTLY: {0}".format(record.explorer_url))

    out = args.out or args.evidence.with_suffix(".anchor.json")
    out.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")

    emit()
    emit("  anchor record: {0}".format(out))
    emit()
    emit(
        "  This proves the bundle existed unchanged at block {0}.".format(
            record.block_number
        )
    )
    emit("  It does NOT prove the evidence is correct.")
    emit()
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
