"""Phase 4 -- deploy the EvidenceNotary contract.

    python scripts\\deploy_contract.py --network local   # in-process, for testing
    python scripts\\deploy_contract.py --network amoy    # public testnet

Records the address in data/chain/deployment.json so anchor and verify find it
without further configuration.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tracelock.chain.compiler import load_or_compile  # noqa: E402
from tracelock.chain.config import (  # noqa: E402
    build_adapter,
    load_chain_config,
    save_deployment,
)
from tracelock.chain.errors import ChainError  # noqa: E402

RULE = "=" * 74


def emit(text: str = "") -> None:
    print(text, flush=True)


def header(text: str) -> None:
    emit()
    emit(RULE)
    emit(text)
    emit(RULE)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="deploy_contract")
    parser.add_argument(
        "--network", default=None, help="amoy | base_sepolia | sepolia | local"
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    header("TRACELOCK - PHASE 4 - DEPLOY EVIDENCE NOTARY")

    config = load_chain_config(network=args.network)
    for key, value in config.describe().items():
        emit("  {0:<16}: {1}".format(key, value))

    if config.is_local:
        emit()
        emit("  NOTE: the local chain has no explorer. Use it for testing; a")
        emit("        reviewer can only independently verify a PUBLIC testnet.")

    emit()
    emit("Compiling ...")
    try:
        compiled = load_or_compile()
        emit(
            "  solc {0}, {1} bytes of bytecode".format(
                compiled.solc_version, len(compiled.bytecode) // 2
            )
        )
        emit("  source sha256   : {0}".format(compiled.source_sha256))
    except ChainError as exc:
        emit("COMPILATION FAILED: {0}".format(exc))
        return 2

    emit()
    emit("Deploying ...")
    try:
        adapter = build_adapter(config, compiled)
        address, receipt = adapter.deploy(compiled)
    except ChainError as exc:
        header("DEPLOYMENT FAILED")
        emit("  {0}".format(exc))
        emit()
        emit("  Nothing was deployed. No address is being reported.")
        return 2

    header("DEPLOYED")
    emit("  contract   : {0}".format(address))
    emit("  tx         : {0}".format(receipt.tx_hash))
    emit("  block      : {0}".format(receipt.block_number))
    emit("  gas used   : {0}".format(receipt.gas_used))
    emit("  chain id   : {0}".format(receipt.chain_id))
    explorer = adapter.network.address_url(address)
    if explorer:
        emit("  explorer   : {0}".format(explorer))

    payload = {
        "contract_address": address,
        "chain_id": receipt.chain_id,
        "network": receipt.network,
        "tx_hash": receipt.tx_hash,
        "block_number": receipt.block_number,
        "deployer": receipt.submitter,
        "solc_version": compiled.solc_version,
        "source_sha256": compiled.source_sha256,
        "abi": compiled.abi,
        "deployed_at": datetime.now(timezone.utc).isoformat(),
    }
    path = save_deployment(payload, args.out) if args.out else save_deployment(payload)

    emit()
    emit("  recorded   : {0}".format(path))
    emit("  anchor and verify will find this address automatically.")
    emit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
