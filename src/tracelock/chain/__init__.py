"""Phase 4 -- blockchain anchoring and re-verification.

Only cryptographic commitments reach the chain: a 32-byte Merkle root plus
packed metadata. No images, no embeddings, no personal data.

An anchor proves an evidence bundle existed unchanged at a point in time. It
does not prove the evidence is correct.
"""

from tracelock.chain.adapter import (
    NETWORKS,
    ChainAdapter,
    NetworkProfile,
    OnChainAnchor,
    TxReceipt,
    Web3Adapter,
    make_local_adapter,
    make_rpc_adapter,
)
from tracelock.chain.compiler import (
    CompiledContract,
    compile_contract,
    load_or_compile,
)
from tracelock.chain.errors import (
    AlreadyAnchoredError,
    AnchorNotConfirmedError,
    AnchorRejectedError,
    ChainConfigError,
    ChainConnectionError,
    ChainError,
    ChainMismatchError,
    CompilationError,
    ContractNotDeployedError,
    InsufficientFundsError,
)
from tracelock.chain.fingerprint import (
    FINGERPRINT_VERSION,
    LEAF_TAGS,
    BiometricLeakError,
    EvidenceFingerprint,
    assert_no_biometric_leak,
    canonical_bytes,
    diff_leaves,
    fingerprint_evidence,
    leaf_hash,
    merkle_proof,
    merkle_root,
)
from tracelock.chain.notary import (
    ANCHOR_SCHEMA_VERSION,
    AnchorRecord,
    EvidenceNotary,
    ReverificationResult,
    VerificationVerdict,
)

__all__ = [
    "fingerprint_evidence", "EvidenceFingerprint", "merkle_root", "merkle_proof",
    "leaf_hash", "canonical_bytes", "diff_leaves", "assert_no_biometric_leak",
    "BiometricLeakError", "LEAF_TAGS", "FINGERPRINT_VERSION",
    "EvidenceNotary", "AnchorRecord", "ReverificationResult",
    "VerificationVerdict", "ANCHOR_SCHEMA_VERSION",
    "ChainAdapter", "Web3Adapter", "NetworkProfile", "NETWORKS",
    "TxReceipt", "OnChainAnchor", "make_local_adapter", "make_rpc_adapter",
    "CompiledContract", "compile_contract", "load_or_compile",
    "ChainError", "ChainConfigError", "ChainConnectionError",
    "ChainMismatchError", "ContractNotDeployedError", "InsufficientFundsError",
    "AnchorRejectedError", "AlreadyAnchoredError", "AnchorNotConfirmedError",
    "CompilationError",
]
