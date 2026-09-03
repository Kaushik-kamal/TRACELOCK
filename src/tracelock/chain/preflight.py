"""Public-chain configuration checks that cost nothing at startup.

The split this module enforces:

    STRUCTURAL   is the configuration well-formed?   local, instant, no network
    OPERATIONAL  does the chain actually answer?     network, on demand only

Startup and `/api/chain/status` run the structural half only. A malformed
private key or a nonsense RPC URL is caught there, before any connection is
attempted, because a 64-hex-character check does not need a socket.

The operational half runs when the operator actually asks to anchor publicly.
Requiring a live RPC round-trip at import time would make `python run.py` slow
and would make the server's health depend on a third party being up.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
It does not validate that a key controls a funded account, that the contract
exists, or that gas will be sufficient. Those are answerable only by the chain,
and guessing at them locally would produce a confident "ready" that the next
transaction contradicts. `READY` here means "nothing is obviously wrong",
which is the strongest honest claim available without a network call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import urlsplit


class PublicConfigStatus(str, Enum):
    """How far the public configuration gets before something stops it."""

    NOT_CONFIGURED = "not_configured"      # local mode; nothing was asked for
    INCOMPLETE = "incomplete"              # public asked for, credentials missing
    INVALID_CREDENTIALS = "invalid_credentials"  # present but malformed
    READY = "ready"                        # structurally sound, untested
    UNREACHABLE = "unreachable"            # RPC did not answer (runtime only)
    FAILED = "failed"                      # chain rejected the transaction

    @property
    def usable(self) -> bool:
        """Is it worth ATTEMPTING a public anchor in this state?"""
        return self is PublicConfigStatus.READY

    @property
    def headline(self) -> str:
        return {
            PublicConfigStatus.NOT_CONFIGURED: "Local demo chain",
            PublicConfigStatus.INCOMPLETE: "Public chain not fully configured",
            PublicConfigStatus.INVALID_CREDENTIALS: "Public chain configuration is invalid",
            PublicConfigStatus.READY: "Public chain configured",
            PublicConfigStatus.UNREACHABLE: "Public blockchain unavailable",
            PublicConfigStatus.FAILED: "Public anchor unavailable",
        }[self]


# A raw secp256k1 key is 32 bytes: 64 hex characters, optionally 0x-prefixed.
_PRIVATE_KEY = re.compile(r"^(0x)?[0-9a-fA-F]{64}$")


@dataclass(frozen=True, slots=True)
class Preflight:
    """The structural verdict. Contains no secret and is safe to serialise."""

    status: PublicConfigStatus
    network: str
    problems: tuple[str, ...] = field(default_factory=tuple)
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status.usable

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status.value,
            "headline": self.status.headline,
            "network": self.network,
            "checked": "structure_only",
            "note": (
                "Configuration structure only. Connectivity, funding and "
                "contract state are checked when an anchor is actually "
                "attempted."
            ),
        }
        if self.problems:
            payload["problems"] = list(self.problems)
        if self.detail:
            payload["detail"] = self.detail
        return payload


def check_public_config(env: dict[str, str] | None = None) -> Preflight:
    """Structural check only. No sockets, no DNS, no chain calls.

    Reports WHICH variable is wrong, never its value -- the problem strings are
    surfaced in the UI and written to logs.
    """
    import os

    from tracelock.chain.adapter import NETWORKS
    from tracelock.chain.config import DEFAULT_NETWORK, _load_dotenv

    if env is None:
        _load_dotenv()
        env = dict(os.environ)

    network = (env.get("TL_CHAIN") or DEFAULT_NETWORK).strip()

    if network not in NETWORKS:
        return Preflight(
            PublicConfigStatus.INVALID_CREDENTIALS, network,
            ("TL_CHAIN is not a known network",),
            "Known networks: {0}.".format(", ".join(sorted(NETWORKS))),
        )

    if not NETWORKS[network].publicly_verifiable:
        return Preflight(PublicConfigStatus.NOT_CONFIGURED, network)

    rpc = (env.get("TL_RPC_URL") or "").strip()
    key = (env.get("TL_PRIVATE_KEY") or "").strip()

    missing = []
    if not rpc:
        missing.append("TL_RPC_URL is not set")
    if not key:
        missing.append("TL_PRIVATE_KEY is not set")
    if missing:
        return Preflight(
            PublicConfigStatus.INCOMPLETE, network, tuple(missing),
            "Falling back to the local demo chain.",
        )

    # Both present -- are they even the right SHAPE? Catching this here saves a
    # doomed connection and gives a far better message than a signing error.
    malformed = []

    scheme = urlsplit(rpc).scheme.lower()
    if scheme not in ("http", "https", "ws", "wss"):
        malformed.append("TL_RPC_URL is not an http(s) or ws(s) URL")
    elif not urlsplit(rpc).hostname:
        malformed.append("TL_RPC_URL has no host")

    if not _PRIVATE_KEY.match(key):
        malformed.append(
            "TL_PRIVATE_KEY is not a 32-byte hex key (expected 64 hex "
            "characters, optionally 0x-prefixed)"
        )

    if malformed:
        return Preflight(
            PublicConfigStatus.INVALID_CREDENTIALS, network, tuple(malformed),
            "Public anchoring will not be attempted with invalid configuration.",
        )

    return Preflight(PublicConfigStatus.READY, network)


def classify_anchor_failure(exc: Exception) -> tuple[PublicConfigStatus, str, str]:
    """Turn a chain exception into (status, user message, log detail).

    The user message never contains an endpoint, an address, a key, or a
    stack frame. The log detail keeps the original text so a developer can
    diagnose it server-side.
    """
    raw = str(exc)
    lowered = raw.lower()

    connectivity = (
        "connection", "cannot connect", "could not connect", "connect to",
        "timeout", "timed out", "max retries", "getaddrinfo",
        "name or service not known", "connectionpool", "unreachable",
        "ssl", "certificate", "refused", "network is unreachable",
    )
    if any(marker in lowered for marker in connectivity):
        return (
            PublicConfigStatus.UNREACHABLE,
            "We couldn't reach the public blockchain. No public verification "
            "record was created.",
            raw,
        )

    signing = ("invalid private key", "signature", "sign", "account", "nonce too low")
    if any(marker in lowered for marker in signing):
        return (
            PublicConfigStatus.INVALID_CREDENTIALS,
            "The configured blockchain account could not sign this "
            "transaction. No public verification record was created.",
            raw,
        )

    funding = ("insufficient funds", "not enough for gas", "gas required", "underpriced")
    if any(marker in lowered for marker in funding):
        return (
            PublicConfigStatus.FAILED,
            "The blockchain account does not have enough testnet funds to pay "
            "for this transaction. No public verification record was created.",
            raw,
        )

    contract = ("revert", "execution reverted", "no contract code", "out of gas")
    if any(marker in lowered for marker in contract):
        return (
            PublicConfigStatus.FAILED,
            "The blockchain rejected this transaction. No public verification "
            "record was created.",
            raw,
        )

    return (
        PublicConfigStatus.FAILED,
        "The public anchor could not be completed. No public verification "
        "record was created.",
        raw,
    )
