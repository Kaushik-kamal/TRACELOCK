"""Phase 4 -- the evidence notary.

Consumes a Phase 3 evidence artifact and anchors a commitment to it. Re-derives
NO verification logic: Phase 2 decided what was verified, Phase 3 scored it,
and this layer only commits to the result and later proves it unchanged.

WHAT AN ANCHOR PROVES, PRECISELY
--------------------------------
    "This exact evidence bundle existed, in this exact form, at or before
     block N, and was submitted by this address."

It does NOT prove the evidence is correct. A blockchain cannot make a lying
source honest. Every report states this, because a notary that overclaims is
worse than none.

RE-VERIFICATION
---------------
Given an artifact and its anchor record, `reverify` recomputes every leaf from
the artifact on disk, rebuilds the root, and compares it against the chain. On
a mismatch it reports WHICH leaf changed -- tamper localisation, which is the
whole reason for the Merkle structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from tracelock.chain.adapter import ChainAdapter, OnChainAnchor, TxReceipt
from tracelock.chain.fingerprint import (
    EvidenceFingerprint,
    diff_leaves,
    fingerprint_evidence,
)

ANCHOR_SCHEMA_VERSION = "evidence-anchor/1"


class VerificationVerdict(str, Enum):
    """Outcome of re-verifying an artifact against its anchor."""

    INTACT = "INTACT"
    TAMPERED = "TAMPERED"
    NOT_ANCHORED = "NOT_ANCHORED"

    @property
    def explanation(self) -> str:
        return {
            VerificationVerdict.INTACT: (
                "the artifact on disk recomputes to the root recorded on chain: "
                "it has not been altered since it was anchored"
            ),
            VerificationVerdict.TAMPERED: (
                "the artifact recomputes to a DIFFERENT root than the one "
                "anchored. It has been modified since notarisation"
            ),
            VerificationVerdict.NOT_ANCHORED: (
                "no anchor for this root exists on chain. The artifact may "
                "never have been anchored, or was anchored on another network"
            ),
        }[self]


@dataclass(frozen=True, slots=True)
class AnchorRecord:
    """The provenance record written back beside the evidence artifact."""

    schema_version: str
    merkle_root: str
    run_id: str
    case_id: str
    probe_commitment: str
    pipeline_hash: str
    trust_score_bp: int
    evidence_count: int
    independent_publishers: int
    leaves: dict[str, str]

    tx_hash: str
    chain_id: int
    network: str
    contract_address: str
    block_number: int
    block_timestamp: int
    gas_used: int
    submitter: str
    explorer_url: str
    confirmations: int

    anchored_at: str
    fingerprint_version: str
    # Identity of the chain that ACTUALLY executed this anchor, copied from
    # the adapter's profile. Interfaces render this and never re-derive a
    # label from configuration.
    network_display_name: str = ""
    ephemeral: bool = False

    @property
    def block_time_utc(self) -> str:
        return datetime.fromtimestamp(
            self.block_timestamp, tz=timezone.utc
        ).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fingerprint_version": self.fingerprint_version,
            "merkle_root": self.merkle_root,
            "run_id": self.run_id,
            "case_id": self.case_id,
            "commitments": {
                "probe": self.probe_commitment,
                "pipeline": self.pipeline_hash,
                "leaves": self.leaves,
            },
            "on_chain": {
                "tx_hash": self.tx_hash,
                "chain_id": self.chain_id,
                "network": self.network,
                "network_display_name": self.network_display_name,
                "ephemeral": self.ephemeral,
                "publicly_verifiable": bool(self.explorer_url),
                "contract_address": self.contract_address,
                "block_number": self.block_number,
                "block_timestamp": self.block_timestamp,
                "block_time_utc": self.block_time_utc,
                "gas_used": self.gas_used,
                "submitter": self.submitter,
                "confirmations": self.confirmations,
                "explorer_url": self.explorer_url,
            },
            "anchored_values": {
                "trust_score_bp": self.trust_score_bp,
                "trust_score": self.trust_score_bp / 100.0,
                "evidence_count": self.evidence_count,
                "independent_publishers": self.independent_publishers,
            },
            "anchored_at": self.anchored_at,
            "proves": (
                "This exact evidence bundle existed in this form at or before "
                "block {0} on {1} and was submitted by {2}. It does NOT "
                "establish that the evidence is correct -- a chain cannot make "
                "a lying source honest.{3}".format(
                    self.block_number,
                    self.network_display_name or self.network,
                    self.submitter,
                    (
                        " This anchor is on an EPHEMERAL LOCAL CHAIN: it is "
                        "lost when the process restarts and CANNOT be verified "
                        "by anyone else."
                        if self.ephemeral
                        else ""
                    ),
                )
            ),
            "contains_no_personal_data": True,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AnchorRecord":
        chain = payload["on_chain"]
        values = payload.get("anchored_values", {})
        commitments = payload.get("commitments", {})
        return cls(
            schema_version=payload["schema_version"],
            fingerprint_version=payload.get("fingerprint_version", ""),
            merkle_root=payload["merkle_root"],
            run_id=payload.get("run_id", ""),
            case_id=payload.get("case_id", ""),
            probe_commitment=commitments.get("probe", ""),
            pipeline_hash=commitments.get("pipeline", ""),
            leaves=commitments.get("leaves", {}),
            trust_score_bp=int(values.get("trust_score_bp", 0)),
            evidence_count=int(values.get("evidence_count", 0)),
            independent_publishers=int(values.get("independent_publishers", 0)),
            tx_hash=chain["tx_hash"],
            chain_id=int(chain["chain_id"]),
            network=chain.get("network", ""),
            contract_address=chain["contract_address"],
            block_number=int(chain["block_number"]),
            block_timestamp=int(chain["block_timestamp"]),
            gas_used=int(chain.get("gas_used", 0)),
            submitter=chain.get("submitter", ""),
            explorer_url=chain.get("explorer_url", ""),
            confirmations=int(chain.get("confirmations", 0)),
            network_display_name=chain.get("network_display_name", ""),
            ephemeral=bool(chain.get("ephemeral", False)),
            anchored_at=payload.get("anchored_at", ""),
        )


@dataclass(frozen=True, slots=True)
class ReverificationResult:
    """Outcome of checking an artifact against its on-chain anchor."""

    verdict: VerificationVerdict
    expected_root: str
    recomputed_root: str
    changed_leaves: tuple[tuple[str, str, str], ...]
    on_chain: OnChainAnchor | None
    checked_at: str

    @property
    def is_intact(self) -> bool:
        return self.verdict is VerificationVerdict.INTACT

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "explanation": self.verdict.explanation,
            "expected_root": self.expected_root,
            "recomputed_root": self.recomputed_root,
            "roots_match": self.expected_root == self.recomputed_root,
            "changed_leaves": [
                {"leaf": tag, "anchored": before, "recomputed": after}
                for tag, before, after in self.changed_leaves
            ],
            "on_chain": self.on_chain.to_dict() if self.on_chain else None,
            "checked_at": self.checked_at,
        }


class EvidenceNotary:
    """Anchors evidence artifacts and re-verifies them."""

    def __init__(self, adapter: ChainAdapter, contract_address: str) -> None:
        self.adapter = adapter
        self.contract_address = contract_address

    # ------------------------------------------------------------------

    def fingerprint(self, artifact: dict[str, Any]) -> EvidenceFingerprint:
        """Derive the commitment set. Pure; no chain access."""
        return fingerprint_evidence(artifact)

    def anchor(
        self, artifact: dict[str, Any], *, confirmations: int = 1
    ) -> tuple[AnchorRecord, TxReceipt]:
        """Anchor an artifact. Raises rather than returning on any failure.

        There is deliberately no code path that returns an AnchorRecord for a
        transaction that was not confirmed.
        """
        fingerprint = self.fingerprint(artifact)
        receipt = self.adapter.anchor(
            self.contract_address, fingerprint, confirmations=confirmations
        )

        record = AnchorRecord(
            schema_version=ANCHOR_SCHEMA_VERSION,
            fingerprint_version=fingerprint.version,
            merkle_root=fingerprint.merkle_root_hex,
            run_id=fingerprint.run_id,
            case_id="0x" + fingerprint.case_id.hex(),
            probe_commitment="0x" + fingerprint.probe_commitment.hex(),
            pipeline_hash="0x" + fingerprint.pipeline_hash.hex(),
            leaves=fingerprint.leaf_hex(),
            trust_score_bp=fingerprint.trust_score_bp,
            evidence_count=fingerprint.evidence_count,
            independent_publishers=fingerprint.independent_publishers,
            tx_hash=receipt.tx_hash,
            chain_id=receipt.chain_id,
            network=receipt.network,
            contract_address=receipt.contract_address or self.contract_address,
            block_number=receipt.block_number,
            block_timestamp=receipt.block_timestamp,
            gas_used=receipt.gas_used,
            submitter=receipt.submitter,
            explorer_url=receipt.explorer_url,
            confirmations=receipt.confirmations,
            network_display_name=receipt.network_display_name,
            ephemeral=receipt.ephemeral,
            anchored_at=datetime.now(timezone.utc).isoformat(),
        )
        return record, receipt

    def reverify(
        self, artifact: dict[str, Any], record: AnchorRecord
    ) -> ReverificationResult:
        """Recompute the artifact and compare it against the chain.

        Three outcomes, and the difference between the last two matters:
          INTACT        recomputed root matches, and the chain has it
          TAMPERED      the artifact changed since anchoring
          NOT_ANCHORED  the chain has no record of this root
        """
        recomputed = self.fingerprint(artifact)
        expected_root = record.merkle_root
        recomputed_root = recomputed.merkle_root_hex

        on_chain = self.adapter.lookup(
            record.contract_address or self.contract_address,
            bytes.fromhex(expected_root[2:]),
        )

        checked_at = datetime.now(timezone.utc).isoformat()

        if not on_chain.exists:
            return ReverificationResult(
                verdict=VerificationVerdict.NOT_ANCHORED,
                expected_root=expected_root,
                recomputed_root=recomputed_root,
                changed_leaves=(),
                on_chain=on_chain,
                checked_at=checked_at,
            )

        if recomputed_root == expected_root:
            return ReverificationResult(
                verdict=VerificationVerdict.INTACT,
                expected_root=expected_root,
                recomputed_root=recomputed_root,
                changed_leaves=(),
                on_chain=on_chain,
                checked_at=checked_at,
            )

        # Localise the tamper: which leaf moved?
        changed = tuple(diff_leaves(record.leaves, recomputed.leaf_hex()))
        return ReverificationResult(
            verdict=VerificationVerdict.TAMPERED,
            expected_root=expected_root,
            recomputed_root=recomputed_root,
            changed_leaves=changed,
            on_chain=on_chain,
            checked_at=checked_at,
        )
