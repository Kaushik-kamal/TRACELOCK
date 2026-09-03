"""Phase 4 error taxonomy.

The governing rule for this phase:

    NEVER CLAIM SOMETHING WAS ANCHORED WHEN IT WAS NOT.

A notary that reports success on a failed write is worse than no notary: it
manufactures false confidence in evidence. So every failure mode below is an
explicit exception with its own type, and the anchoring path has no branch that
returns a success-shaped object without a confirmed receipt.

    ChainError
    |
    +-- ChainConfigError        missing RPC, key, or address -- setup, not chain
    +-- ChainConnectionError    cannot reach the node
    +-- ChainMismatchError      connected to the wrong network
    +-- ContractNotDeployedError no code at the configured address
    +-- InsufficientFundsError  wallet cannot pay gas
    +-- AnchorRejectedError     the contract reverted
    |   +-- AlreadyAnchoredError
    +-- AnchorNotConfirmedError submitted but never confirmed -- state UNKNOWN
    +-- CompilationError        solc failed
"""

from __future__ import annotations


class ChainError(Exception):
    """Base for every blockchain-layer failure."""


class ChainConfigError(ChainError):
    """Missing or invalid configuration. A setup problem, not a chain problem.

    Kept distinct so an unset RPC URL is never reported as an anchoring
    failure -- the same discipline Phase 0 applies to API keys.
    """


class ChainConnectionError(ChainError):
    """The RPC endpoint is unreachable."""


class ChainMismatchError(ChainError):
    """Connected to a different chain than configured.

    Anchoring to the wrong network would produce a transaction hash that looks
    valid and resolves nowhere, which is exactly the kind of false evidence
    this phase exists to prevent.
    """

    def __init__(self, expected: int, actual: int):
        super().__init__(
            "chain id mismatch: configured for {0} but the node reports {1}. "
            "Refusing to anchor -- a transaction on the wrong network would "
            "look valid and prove nothing.".format(expected, actual)
        )
        self.expected = expected
        self.actual = actual


class ContractNotDeployedError(ChainError):
    """No contract code at the configured address on this chain."""

    def __init__(self, address: str, chain_id: int):
        super().__init__(
            "no contract code at {0} on chain {1}. Deploy first "
            "(scripts/deploy_contract.py), or check TL_CONTRACT_ADDRESS.".format(
                address, chain_id
            )
        )
        self.address = address
        self.chain_id = chain_id


class InsufficientFundsError(ChainError):
    """The submitting wallet cannot pay for gas."""

    def __init__(self, address: str, balance_wei: int, chain_name: str):
        super().__init__(
            "wallet {0} holds {1} wei on {2} -- not enough for gas. Fund it "
            "from the faucet before anchoring.".format(
                address, balance_wei, chain_name
            )
        )
        self.address = address
        self.balance_wei = balance_wei


class AnchorRejectedError(ChainError):
    """The contract reverted. Nothing was written."""


class AlreadyAnchoredError(AnchorRejectedError):
    """This exact root is already on chain.

    Not a fault. The contract enforces it so an identical bundle cannot be
    anchored twice and later presented as two independent observations.
    """

    def __init__(self, merkle_root: str):
        super().__init__(
            "evidence root {0} is already anchored. Re-anchoring identical "
            "evidence would let one observation masquerade as two.".format(
                merkle_root
            )
        )
        self.merkle_root = merkle_root


class AnchorNotConfirmedError(ChainError):
    """Submitted, but no receipt within the timeout. State is UNKNOWN.

    Deliberately NOT treated as either success or failure. The transaction may
    still confirm. The caller is given the hash and told to check, because
    guessing in either direction would be a lie.
    """

    def __init__(self, tx_hash: str, timeout: float, explorer_url: str = ""):
        message = (
            "transaction {0} was submitted but not confirmed within {1}s. "
            "Its state is UNKNOWN -- it may still confirm. This is NOT being "
            "reported as anchored.".format(tx_hash, timeout)
        )
        if explorer_url:
            message += " Check: {0}".format(explorer_url)
        super().__init__(message)
        self.tx_hash = tx_hash
        self.explorer_url = explorer_url


class CompilationError(ChainError):
    """solc could not compile the contract."""
