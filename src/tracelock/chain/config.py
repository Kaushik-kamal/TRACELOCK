"""Chain configuration, resolved from the environment.

SECRETS NEVER TOUCH THE REPOSITORY
----------------------------------
The private key is read from TL_PRIVATE_KEY and is never written to an
artifact, a log line, or an error message. `describe()` exists so the CLI can
show the operator what is configured without printing the key: it reports only
whether a key is present and which address it derives to, which is public
information anyway.

Use a throwaway TESTNET key. Nothing in this project should ever be pointed at
a wallet holding real funds.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tracelock.chain.adapter import NETWORKS, NetworkProfile
from tracelock.chain.errors import ChainConfigError

# Default to the LOCAL chain, not a public one. An unconfigured install then
# reports "LOCAL DEMO CHAIN" truthfully instead of advertising Polygon Amoy
# while every anchor lives in an in-process EVM. Set TL_CHAIN=amoy once real
# credentials exist.
DEFAULT_NETWORK = "local"

logger = logging.getLogger(__name__)


class ChainMode(str, Enum):
    """How an anchor can actually be verified.

    Derived from what the process CAN do, never from what configuration asked
    for. A public network with missing credentials is not "public mode with a
    problem" -- it is local mode, because local is where the transaction will
    actually execute, and the interface must describe reality.
    """

    LOCAL_DEMO = "local_demo"
    PUBLIC = "public"

    @property
    def badge(self) -> str:
        return {
            ChainMode.LOCAL_DEMO: "LOCAL DEMO CHAIN",
            ChainMode.PUBLIC: "PUBLICLY VERIFIABLE",
        }[self]

    @property
    def explanation(self) -> str:
        return {
            ChainMode.LOCAL_DEMO: (
                "This anchor is functional for this running demo but resets "
                "when the server restarts."
            ),
            ChainMode.PUBLIC: (
                "Anyone can independently verify this evidence anchor."
            ),
        }[self]


@dataclass(frozen=True, slots=True)
class ModeResolution:
    """The mode that will actually be used, and why.

    `fell_back` is the honest part: when public mode was requested but could
    not be honoured, that is stated rather than quietly becoming local.
    """

    mode: ChainMode
    network_key: str
    requested_network: str
    fell_back: bool = False
    reason: str = ""
    missing: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mode": self.mode.value,
            "badge": self.mode.badge,
            "explanation": self.mode.explanation,
            "network": self.network_key,
        }
        if self.fell_back:
            payload["requested_network"] = self.requested_network
            payload["fell_back"] = True
            payload["reason"] = self.reason
            payload["missing"] = list(self.missing)
        return payload


PUBLIC_REQUIREMENTS = ("TL_RPC_URL", "TL_PRIVATE_KEY")


def resolve_chain_mode(
    *, network: str | None = None, env: dict[str, str] | None = None
) -> ModeResolution:
    """Decide between local demo and public, from the environment alone.

    Rules, in order:

      * an unknown or absent network is local -- the project must run with no
        configuration at all
      * "local" is local
      * a public network with every credential present is public
      * a public network with anything missing FALLS BACK to local, and says
        so; it never half-starts and never crashes at anchor time

    Pure and side-effect free apart from one log line, so it can be called
    from a health endpoint without touching the chain.
    """
    _load_dotenv()
    source = env if env is not None else os.environ

    requested = (network or source.get("TL_CHAIN") or DEFAULT_NETWORK).strip()

    if requested not in NETWORKS:
        return ModeResolution(
            ChainMode.LOCAL_DEMO, DEFAULT_NETWORK, requested,
            fell_back=True,
            reason="Unknown network {0!r}; known networks are {1}.".format(
                requested, ", ".join(sorted(NETWORKS))
            ),
        )

    profile = NETWORKS[requested]
    if not profile.publicly_verifiable:
        return ModeResolution(ChainMode.LOCAL_DEMO, requested, requested)

    missing = tuple(name for name in PUBLIC_REQUIREMENTS if not source.get(name, "").strip())
    if missing:
        reason = (
            "Public anchoring on {0} needs {1}. Falling back to the local demo "
            "chain -- anchoring and tamper detection still work, but the "
            "anchor will not be publicly verifiable.".format(
                profile.display_name, " and ".join(missing)
            )
        )
        logger.warning("chain: %s", reason)
        return ModeResolution(
            ChainMode.LOCAL_DEMO, DEFAULT_NETWORK, requested,
            fell_back=True, reason=reason, missing=missing,
        )

    return ModeResolution(ChainMode.PUBLIC, requested, requested)
DEFAULT_DEPLOYMENT = Path("data/chain/deployment.json")


@dataclass(frozen=True, slots=True)
class ChainConfig:
    network_key: str
    rpc_url: str
    private_key: str
    contract_address: str
    confirmations: int
    resolution: "ModeResolution | None" = None

    @property
    def mode(self) -> "ChainMode":
        return self.resolution.mode if self.resolution else ChainMode.LOCAL_DEMO

    @property
    def profile(self) -> NetworkProfile:
        if self.network_key not in NETWORKS:
            raise ChainConfigError(
                "unknown network {0!r}; known: {1}".format(
                    self.network_key, sorted(NETWORKS)
                )
            )
        return NETWORKS[self.network_key]

    @property
    def is_local(self) -> bool:
        return self.network_key == "local"

    @property
    def has_key(self) -> bool:
        return bool(self.private_key)

    def account_address(self) -> str:
        """Public address for the configured key, or "" when unset."""
        if not self.private_key:
            return ""
        from eth_account import Account

        key = self.private_key if self.private_key.startswith("0x") else "0x" + self.private_key
        try:
            return Account.from_key(key).address
        except Exception:
            return "(invalid key)"

    def describe(self) -> dict[str, Any]:
        """Safe to print and safe to log. Contains no secret."""
        profile = self.profile if self.network_key in NETWORKS else None
        return {
            "network": self.network_key,
            "network_display_name": profile.display_name if profile else "UNKNOWN CHAIN",
            "ephemeral": profile.ephemeral if profile else True,
            "publicly_verifiable": profile.publicly_verifiable if profile else False,
            "persistence_note": profile.persistence_note if profile else "",
            "chain_id": profile.chain_id if profile else None,
            "rpc_url_set": bool(self.rpc_url),
            "private_key_set": self.has_key,
            "account": self.account_address(),
            "contract_address": self.contract_address or "(not set)",
            "confirmations": self.confirmations,
            "mode": self.mode.value,
            "mode_badge": self.mode.badge,
            "mode_explanation": self.mode.explanation,
            **(
                {"fallback": self.resolution.to_dict()}
                if self.resolution and self.resolution.fell_back
                else {}
            ),
        }

    def require_for_write(self) -> None:
        """Fail loudly and early, before anything is attempted."""
        if self.is_local:
            return
        missing = []
        if not self.rpc_url:
            missing.append("TL_RPC_URL")
        if not self.private_key:
            missing.append("TL_PRIVATE_KEY")
        if missing:
            raise ChainConfigError(
                "missing {0}. Set them in .env (which is gitignored). Use a "
                "TESTNET key only.\n  faucet: {1}".format(
                    " and ".join(missing), self.profile.faucet
                )
            )


def load_chain_config(
    *,
    network: str | None = None,
    contract_address: str | None = None,
    confirmations: int | None = None,
) -> ChainConfig:
    """Read configuration from the environment, with CLI overrides.

    The network is whatever `resolve_chain_mode` says is actually usable, not
    whatever was requested. That is what makes an incomplete public setup a
    graceful fallback instead of a crash at anchor time.
    """
    _load_dotenv()
    resolution = resolve_chain_mode(network=network)

    return ChainConfig(
        network_key=resolution.network_key,
        rpc_url=os.environ.get("TL_RPC_URL", ""),
        private_key=os.environ.get("TL_PRIVATE_KEY", ""),
        contract_address=(
            contract_address
            or os.environ.get("TL_CONTRACT_ADDRESS", "")
            or _address_from_deployment()
        ),
        confirmations=confirmations
        if confirmations is not None
        else int(os.environ.get("TL_CONFIRMATIONS", "1")),
        resolution=resolution,
    )


def _load_dotenv() -> None:
    """Populate os.environ from .env without overriding real env vars."""
    path = Path(".env")
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _address_from_deployment(path: str | Path = DEFAULT_DEPLOYMENT) -> str:
    """Fall back to the address recorded by the last deployment."""
    target = Path(path)
    if not target.is_file():
        return ""
    try:
        import json

        return json.loads(target.read_text(encoding="utf-8")).get("contract_address", "")
    except Exception:
        return ""


def save_deployment(payload: dict[str, Any], path: str | Path = DEFAULT_DEPLOYMENT) -> Path:
    """Record a deployment so later commands find the address automatically."""
    import json

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def build_adapter(config: ChainConfig, compiled):
    """Construct the right adapter for the configured network."""
    from tracelock.chain.adapter import make_local_adapter, make_rpc_adapter

    if config.is_local:
        return make_local_adapter(compiled)

    config.require_for_write()
    return make_rpc_adapter(
        config.network_key, config.rpc_url, config.private_key, compiled
    )


# One in-process EVM per server process, for the local demo chain only.
#
# Without this, every call built a NEW eth-tester chain and redeployed the
# contract onto it -- so an anchor written by one request lived on a different
# chain than the request that tried to verify it, and VERIFY and TAMPER TEST
# could never succeed locally. They failed with "no contract code at 0x...",
# which reads as a broken feature rather than as the ephemerality it actually
# was.
#
# Caching makes the local chain behave like a chain for the life of the
# process: anchor, verify, and tamper-test all see the same state. It does NOT
# make it persistent, and it does not touch public networks -- the chain still
# dies with the process, which is exactly what the UI says it does.
_LOCAL_CHAIN: dict[str, Any] = {}


def reset_local_chain() -> None:
    """Drop the cached local EVM. Tests use this for isolation."""
    _LOCAL_CHAIN.clear()


def build_adapter_with_contract(config: ChainConfig, compiled):
    """Adapter plus a usable contract address.

    The local chain is IN-PROCESS and therefore ephemeral: nothing survives a
    restart. Within one process, though, the same chain and contract are
    reused, so an anchor can actually be re-verified.

    This is a genuine deployment on a real EVM, not a stub. For public
    networks nothing is auto-deployed and nothing is cached -- a missing
    address there is an operator error and must surface as one.

    Returns (adapter, contract_address, was_auto_deployed).
    """
    if not config.is_local:
        return build_adapter(config, compiled), config.contract_address, False

    cached = _LOCAL_CHAIN.get("entry")
    if cached is not None:
        adapter, address = cached
        return adapter, address, True

    adapter = build_adapter(config, compiled)
    address, _receipt = adapter.deploy(compiled)
    _LOCAL_CHAIN["entry"] = (adapter, address)
    return adapter, address, True
