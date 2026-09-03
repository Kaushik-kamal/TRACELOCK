"""Chain adapters.

The demo target is a PUBLIC TESTNET, because that is the only kind of anchor a
reviewer can independently verify. A local chain proves the code works; it
proves nothing to someone watching a recording, since there is no explorer URL
to open.

Networks are configuration, not code. Switching from Amoy to Base Sepolia is
one environment variable, so a faucet outage on demo day is a one-line change
rather than a rewrite -- the failure mode the original architecture flagged as
most likely.

`LocalTesterAdapter` runs an in-process EVM (eth-tester) so the whole test
suite exercises real contract behaviour, including reverts and event decoding,
with no node, no network, and no funded wallet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from tracelock.chain.errors import (
    AlreadyAnchoredError,
    AnchorNotConfirmedError,
    AnchorRejectedError,
    ChainConfigError,
    ChainConnectionError,
    ChainMismatchError,
    ContractNotDeployedError,
    InsufficientFundsError,
)


@dataclass(frozen=True, slots=True)
class NetworkProfile:
    """A supported chain. Adding one is a data change, not a code change.

    CHAIN IDENTITY IS NOT A UI DECISION
    -----------------------------------
    `display_name` is the ONLY string any interface may show for a chain, and
    it comes from the profile that actually executed the transaction -- never
    from configuration intent, an environment default, or a UI constant.

    The bug this prevents was real: the status endpoint read TL_CHAIN (default
    "amoy") while the browser hardcoded "local", so the interface advertised
    Polygon Amoy while every anchor lived in an ephemeral in-process EVM.
    Claiming a public chain for a local-only anchor is the most damaging lie
    this system could tell, because the whole point of anchoring is that a
    third party can check it.
    """

    key: str
    name: str
    chain_id: int
    explorer_tx: str
    explorer_address: str
    faucet: str
    is_testnet: bool = True
    # True when the chain exists only inside this process and its state is
    # lost on restart. An ephemeral chain has no explorer and no third-party
    # verifiability, and must never be presented as if it did.
    ephemeral: bool = False

    @property
    def display_name(self) -> str:
        """The only chain label an interface may render."""
        return self.name

    @property
    def persistence_note(self) -> str:
        return (
            "Ephemeral - resets on restart. Not publicly verifiable."
            if self.ephemeral
            else "Public testnet. Anchors are independently verifiable."
        )

    @property
    def publicly_verifiable(self) -> bool:
        return not self.ephemeral and bool(self.explorer_tx)

    def tx_url(self, tx_hash: str) -> str:
        """Explorer URL, or "" when none can honestly be produced.

        An ephemeral chain returns "" unconditionally: there is no explorer,
        and emitting a plausible-looking link would be fabricating evidence.
        """
        if self.ephemeral or not self.explorer_tx:
            return ""
        return self.explorer_tx.format(tx_hash)

    def address_url(self, address: str) -> str:
        if self.ephemeral or not self.explorer_address:
            return ""
        return self.explorer_address.format(address)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "display_name": self.display_name,
            "chain_id": self.chain_id,
            "ephemeral": self.ephemeral,
            "publicly_verifiable": self.publicly_verifiable,
            "persistence_note": self.persistence_note,
            "faucet": self.faucet,
        }


# Split so a dead endpoint fails fast while a slow-but-live chain still gets
# time to mine. Connect is the phase that hangs on an unreachable host.
CONNECT_TIMEOUT = 8.0
READ_TIMEOUT = 60.0

NETWORKS: dict[str, NetworkProfile] = {
    # Primary demo target: ~2s blocks feel immediate on camera, gas is
    # negligible, and Polygonscan decodes event logs clearly.
    "amoy": NetworkProfile(
        key="amoy",
        name="Polygon Amoy Testnet",
        chain_id=80002,
        explorer_tx="https://amoy.polygonscan.com/tx/{0}",
        explorer_address="https://amoy.polygonscan.com/address/{0}",
        faucet="https://faucet.polygon.technology/",
    ),
    # Fallback: the most reliable faucet of the three -- Coinbase's does not
    # require a mainnet balance, unlike Sepolia's.
    "base_sepolia": NetworkProfile(
        key="base_sepolia",
        name="Base Sepolia Testnet",
        chain_id=84532,
        explorer_tx="https://sepolia.basescan.org/tx/{0}",
        explorer_address="https://sepolia.basescan.org/address/{0}",
        faucet="https://portal.cdp.coinbase.com/products/faucet",
    ),
    "sepolia": NetworkProfile(
        key="sepolia",
        name="Ethereum Sepolia Testnet",
        chain_id=11155111,
        explorer_tx="https://sepolia.etherscan.io/tx/{0}",
        explorer_address="https://sepolia.etherscan.io/address/{0}",
        faucet="https://cloud.google.com/application/web3/faucet/ethereum/sepolia",
    ),
    # Tests and CI. NEVER the demo target: no explorer, so nothing a reviewer
    # can check independently.
    "local": NetworkProfile(
        key="local",
        name="LOCAL DEMO CHAIN",
        chain_id=131277322940537,
        explorer_tx="",
        explorer_address="",
        faucet="",
        ephemeral=True,
    ),
}


@dataclass(frozen=True, slots=True)
class TxReceipt:
    """A CONFIRMED transaction. Never constructed for an unconfirmed one."""

    tx_hash: str
    block_number: int
    block_timestamp: int
    chain_id: int
    network: str
    contract_address: str
    gas_used: int
    submitter: str
    explorer_url: str
    confirmations: int
    # Copied from the profile that ACTUALLY executed this transaction, so the
    # label travels with the receipt and cannot be re-derived from config.
    network_display_name: str = ""
    ephemeral: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "tx_hash": self.tx_hash,
            "block_number": self.block_number,
            "block_timestamp": self.block_timestamp,
            "chain_id": self.chain_id,
            "network": self.network,
            "network_display_name": self.network_display_name,
            "ephemeral": self.ephemeral,
            "contract_address": self.contract_address,
            "gas_used": self.gas_used,
            "submitter": self.submitter,
            "explorer_url": self.explorer_url,
            "confirmations": self.confirmations,
        }


@dataclass(frozen=True, slots=True)
class OnChainAnchor:
    """An anchor record read back from the chain."""

    exists: bool
    merkle_root: str
    timestamp: int
    submitter: str
    trust_score_bp: int
    evidence_count: int

    @property
    def trust_score(self) -> float:
        return self.trust_score_bp / 100.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "exists": self.exists,
            "merkle_root": self.merkle_root,
            "timestamp": self.timestamp,
            "submitter": self.submitter,
            "trust_score_bp": self.trust_score_bp,
            "trust_score": self.trust_score,
            "evidence_count": self.evidence_count,
        }


class ChainAdapter(Protocol):
    """What the notary needs from a chain. Implementations must never fake."""

    network: NetworkProfile

    def deploy(self, compiled) -> tuple[str, TxReceipt]: ...
    def anchor(self, contract_address: str, fingerprint, *, confirmations: int) -> TxReceipt: ...
    def lookup(self, contract_address: str, merkle_root: bytes) -> OnChainAnchor: ...
    def account_address(self) -> str: ...


class Web3Adapter:
    """web3.py adapter shared by real networks and the in-process tester."""

    def __init__(self, w3, account, network: NetworkProfile, compiled) -> None:
        self.w3 = w3
        self.account = account
        self.network = network
        self.compiled = compiled

    # ------------------------------------------------------------------

    def account_address(self) -> str:
        return self.account.address if hasattr(self.account, "address") else str(self.account)

    def _contract(self, address: str):
        from web3 import Web3

        checksum = Web3.to_checksum_address(address)
        if self.w3.eth.get_code(checksum) in (b"", b"0x", "0x"):
            raise ContractNotDeployedError(address, self.network.chain_id)
        return self.w3.eth.contract(address=checksum, abi=self.compiled.abi)

    def _assert_chain(self) -> None:
        try:
            actual = self.w3.eth.chain_id
        except Exception as exc:
            raise ChainConnectionError(
                "cannot reach {0}: {1}".format(self.network.name, exc)
            ) from exc
        if actual != self.network.chain_id:
            raise ChainMismatchError(self.network.chain_id, actual)

    def _assert_funded(self) -> None:
        address = self.account_address()
        balance = self.w3.eth.get_balance(address)
        if balance == 0:
            raise InsufficientFundsError(address, balance, self.network.name)

    # ------------------------------------------------------------------

    def deploy(self, compiled) -> tuple[str, TxReceipt]:
        self._assert_chain()
        self._assert_funded()

        contract = self.w3.eth.contract(abi=compiled.abi, bytecode=compiled.bytecode)
        receipt = self._send(contract.constructor(), to_address=None)

        address = receipt.contract_address
        if not address:
            raise AnchorRejectedError("deployment produced no contract address")
        return address, receipt

    def anchor(self, contract_address: str, fingerprint, *, confirmations: int = 1) -> TxReceipt:
        self._assert_chain()
        self._assert_funded()

        contract = self._contract(contract_address)

        # Refuse before spending gas on a guaranteed revert.
        existing = self.lookup(contract_address, fingerprint.merkle_root)
        if existing.exists:
            raise AlreadyAnchoredError(fingerprint.merkle_root_hex)

        call = contract.functions.anchor(
            fingerprint.merkle_root,
            fingerprint.case_id,
            fingerprint.probe_commitment,
            fingerprint.pipeline_hash,
            fingerprint.trust_score_bp,
            fingerprint.evidence_count,
            fingerprint.independent_publishers,
        )
        return self._send(call, to_address=contract_address, confirmations=confirmations)

    def lookup(self, contract_address: str, merkle_root: bytes) -> OnChainAnchor:
        contract = self._contract(contract_address)
        exists, record = contract.functions.verify(merkle_root).call()
        timestamp, submitter, trust_bp, count = record

        return OnChainAnchor(
            exists=bool(exists),
            merkle_root="0x" + merkle_root.hex(),
            timestamp=int(timestamp),
            submitter=str(submitter),
            trust_score_bp=int(trust_bp),
            evidence_count=int(count),
        )

    # ------------------------------------------------------------------

    def _send(self, call, *, to_address: str | None, confirmations: int = 1) -> TxReceipt:
        """Build, sign, send, and WAIT. Returns only on a confirmed receipt."""
        address = self.account_address()

        try:
            gas = call.estimate_gas({"from": address})
        except Exception as exc:
            # A failed estimate almost always means the call would revert.
            message = str(exc)
            if "AlreadyAnchored" in message:
                raise AlreadyAnchoredError("(root already present)") from exc
            raise AnchorRejectedError(
                "the contract would revert: {0}".format(message)
            ) from exc

        tx = call.build_transaction(
            {
                "from": address,
                "nonce": self.w3.eth.get_transaction_count(address),
                "gas": int(gas * 1.25),
                "chainId": self.network.chain_id,
            }
        )
        # Fee fields are left to web3's build_transaction, which picks
        # EIP-1559 or legacy per what the node supports. Forcing `gasPrice`
        # here conflicts with maxFeePerGas on 1559 chains and is rejected
        # outright -- only fill it in if nothing was set at all.
        if "gasPrice" not in tx and "maxFeePerGas" not in tx:
            try:
                tx["gasPrice"] = self.w3.eth.gas_price
            except Exception:
                pass

        try:
            signed = self.w3.eth.account.sign_transaction(tx, self.account.key)
            raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
            tx_hash = self.w3.eth.send_raw_transaction(raw)
        except Exception as exc:
            raise AnchorRejectedError("submission failed: {0}".format(exc)) from exc

        hex_hash = tx_hash.hex()
        if not hex_hash.startswith("0x"):
            hex_hash = "0x" + hex_hash

        try:
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
        except Exception as exc:
            # Submitted but unconfirmed. State UNKNOWN -- never reported as
            # anchored, and never reported as failed either.
            raise AnchorNotConfirmedError(
                hex_hash, 180.0, self.network.tx_url(hex_hash)
            ) from exc

        if receipt.get("status") != 1:
            raise AnchorRejectedError(
                "transaction {0} reverted on chain".format(hex_hash)
            )

        block = self.w3.eth.get_block(receipt["blockNumber"])
        head = self.w3.eth.block_number

        return TxReceipt(
            tx_hash=hex_hash,
            block_number=int(receipt["blockNumber"]),
            block_timestamp=int(block["timestamp"]),
            chain_id=self.network.chain_id,
            network=self.network.key,
            contract_address=str(receipt.get("contractAddress") or to_address or ""),
            gas_used=int(receipt["gasUsed"]),
            submitter=address,
            explorer_url=self.network.tx_url(hex_hash),
            confirmations=max(1, head - int(receipt["blockNumber"]) + 1),
            network_display_name=self.network.display_name,
            ephemeral=self.network.ephemeral,
        )


def make_local_adapter(compiled) -> Web3Adapter:
    """In-process EVM. For tests and CI only -- never the demo target."""
    from eth_account import Account
    from web3 import EthereumTesterProvider, Web3

    provider = EthereumTesterProvider()
    w3 = Web3(provider)

    funded = w3.eth.accounts[0]
    account = Account.create()
    w3.eth.send_transaction(
        {"from": funded, "to": account.address, "value": w3.to_wei(10, "ether")}
    )

    network = NETWORKS["local"]
    actual = w3.eth.chain_id
    if actual != network.chain_id:
        # eth-tester's chain id varies by version; adopt what it reports so the
        # mismatch guard stays meaningful for real networks.
        network = NetworkProfile(
            key="local", name="LOCAL DEMO CHAIN", chain_id=actual,
            explorer_tx="", explorer_address="", faucet="", ephemeral=True,
        )

    return Web3Adapter(w3, account, network, compiled)


def make_rpc_adapter(
    network_key: str, rpc_url: str, private_key: str, compiled
) -> Web3Adapter:
    """Adapter for a real network. Requires an RPC URL and a funded key."""
    if network_key not in NETWORKS:
        raise ChainConfigError(
            "unknown network {0!r}; known: {1}".format(
                network_key, sorted(NETWORKS)
            )
        )
    if not rpc_url:
        raise ChainConfigError(
            "no RPC URL. Set TL_RPC_URL for {0}.".format(network_key)
        )
    if not private_key:
        raise ChainConfigError(
            "no private key. Set TL_PRIVATE_KEY (a TESTNET key only -- never "
            "a key holding real funds), and keep it out of version control."
        )

    from eth_account import Account
    from web3 import Web3

    # A 60s read timeout with no CONNECT timeout meant an unroutable RPC took
    # 22.9s to fail -- measured against 192.0.2.1 (RFC 5737, guaranteed
    # non-routable). The connect phase is bounded separately and tightly: an
    # endpoint that has not accepted a socket in 8s is not going to.
    w3 = Web3(Web3.HTTPProvider(
        rpc_url,
        request_kwargs={"timeout": (CONNECT_TIMEOUT, READ_TIMEOUT)},
    ))
    if not w3.is_connected():
        raise ChainConnectionError("cannot connect to {0}".format(rpc_url))

    key = private_key if private_key.startswith("0x") else "0x" + private_key
    return Web3Adapter(w3, Account.from_key(key), NETWORKS[network_key], compiled)
