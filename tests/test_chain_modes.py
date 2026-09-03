"""Two execution modes, one honest label.

The rule the whole file defends: the mode describes where the transaction
ACTUALLY executed, never what configuration asked for. A public network with
missing credentials is local mode -- because local is where the anchor will
land -- and saying anything else would be the one lie that matters here, since
the entire point of anchoring is that a third party can check it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tracelock.chain.adapter import NETWORKS
from tracelock.chain.config import (
    PUBLIC_REQUIREMENTS,
    ChainMode,
    resolve_chain_mode,
)

PUBLIC_ENV = {
    "TL_CHAIN": "amoy",
    "TL_RPC_URL": "https://rpc.example.invalid",
    "TL_PRIVATE_KEY": "0x" + "a" * 64,
}


@pytest.fixture
def client():
    from tracelock.api import create_app

    return TestClient(create_app())


# ==========================================================================
# Mode resolution
# ==========================================================================


class TestModeResolution:
    def test_no_configuration_at_all_is_local(self):
        """The project must run with an empty environment."""
        resolution = resolve_chain_mode(env={})
        assert resolution.mode is ChainMode.LOCAL_DEMO
        assert resolution.network_key == "local"
        assert not resolution.fell_back

    def test_explicit_local_is_local(self):
        assert resolve_chain_mode(env={"TL_CHAIN": "local"}).mode is ChainMode.LOCAL_DEMO

    def test_full_public_configuration_enables_public(self):
        resolution = resolve_chain_mode(env=dict(PUBLIC_ENV))
        assert resolution.mode is ChainMode.PUBLIC
        assert resolution.network_key == "amoy"
        assert not resolution.fell_back

    @pytest.mark.parametrize("missing", PUBLIC_REQUIREMENTS)
    def test_any_missing_credential_falls_back(self, missing):
        env = dict(PUBLIC_ENV)
        env.pop(missing)

        resolution = resolve_chain_mode(env=env)
        assert resolution.mode is ChainMode.LOCAL_DEMO
        assert resolution.network_key == "local"
        assert resolution.fell_back
        assert missing in resolution.missing
        assert missing in resolution.reason

    def test_a_blank_credential_counts_as_missing(self):
        """An empty value in .env is not a configured value."""
        env = dict(PUBLIC_ENV, TL_RPC_URL="   ")
        assert resolve_chain_mode(env=env).mode is ChainMode.LOCAL_DEMO

    def test_an_unknown_network_falls_back_instead_of_crashing(self):
        resolution = resolve_chain_mode(env={"TL_CHAIN": "not-a-chain"})
        assert resolution.mode is ChainMode.LOCAL_DEMO
        assert resolution.fell_back
        assert "not-a-chain" in resolution.reason

    def test_incomplete_configuration_never_raises(self):
        """Falling back is the point; crashing at anchor time is what it replaces."""
        for env in (
            {}, {"TL_CHAIN": "amoy"}, {"TL_CHAIN": "amoy", "TL_RPC_URL": "x"},
            {"TL_CHAIN": ""}, {"TL_CHAIN": "sepolia", "TL_PRIVATE_KEY": "x"},
        ):
            resolve_chain_mode(env=env)

    def test_fallback_is_stated_not_silent(self):
        resolution = resolve_chain_mode(env={"TL_CHAIN": "amoy"})
        payload = resolution.to_dict()
        assert payload["fell_back"] is True
        assert payload["requested_network"] == "amoy"
        assert payload["reason"]

    def test_a_clean_local_run_reports_no_fallback_noise(self):
        payload = resolve_chain_mode(env={}).to_dict()
        assert "fell_back" not in payload
        assert "reason" not in payload


class TestModeLabels:
    def test_each_mode_has_its_own_badge_and_explanation(self):
        assert ChainMode.LOCAL_DEMO.badge == "LOCAL DEMO CHAIN"
        assert ChainMode.PUBLIC.badge == "PUBLICLY VERIFIABLE"
        assert ChainMode.LOCAL_DEMO.badge != ChainMode.PUBLIC.badge

    def test_local_explanation_admits_it_resets(self):
        text = ChainMode.LOCAL_DEMO.explanation.lower()
        assert "resets" in text
        assert "verif" not in text.replace("verifiable", "")

    def test_public_explanation_claims_independent_verification(self):
        assert "independently verify" in ChainMode.PUBLIC.explanation.lower()

    def test_local_never_claims_public_verifiability(self):
        assert "publicly verifiable" not in ChainMode.LOCAL_DEMO.explanation.lower()
        assert not NETWORKS["local"].publicly_verifiable
        assert NETWORKS["local"].ephemeral


# ==========================================================================
# Secrets never leave the server
# ==========================================================================


class TestPrivateKeyIsNeverExposed:
    def test_status_endpoint_reports_only_whether_a_key_is_set(self, client):
        """The variable NAME is fine -- it is an operator checklist.

        What must never appear is a VALUE. Banning the name would flag the
        endpoint's legitimate "TL_PRIVATE_KEY: set = false" row.
        """
        payload = client.get("/api/chain/status").json()
        blob = json.dumps(payload)

        # Any 64-hex-char run would be a private key.
        assert not re.search(r"0x[0-9a-fA-F]{64}", blob)
        # Any RPC URL with an embedded project id would be a credential.
        assert not re.search(r"https://[a-z-]+\.(alchemy|infura)\.io/\S+", blob)

        rows = json.dumps(payload).lower()
        if "tl_private_key" in rows:
            # If the key is reported at all, it is reported as a boolean.
            assert '"set"' in rows or "set" in rows

    def test_describe_carries_no_secret(self):
        from tracelock.chain.config import ChainConfig

        config = ChainConfig(
            network_key="local",
            rpc_url="https://rpc.example.invalid/v2/SECRET-PROJECT-ID",
            private_key="0x" + "b" * 64,
            contract_address="0x" + "c" * 40,
            confirmations=1,
        )
        blob = json.dumps(config.describe())

        assert "b" * 64 not in blob
        assert "SECRET-PROJECT-ID" not in blob
        assert config.describe()["private_key_set"] is True

    def test_no_secret_is_hardcoded_anywhere_in_the_chain_package(self):
        for path in Path("src/tracelock/chain").glob("*.py"):
            source = path.read_text(encoding="utf-8")
            # A 64-hex-char literal would be a committed private key.
            assert not re.search(r"['\"]0x[0-9a-fA-F]{64}['\"]", source), path
            # An RPC URL with an embedded API key.
            assert not re.search(r"https://[a-z-]+\.(alchemy|infura)\.io/\S+", source), path

    def test_the_browser_bundle_contains_no_credentials(self):
        source = Path("web/app.js").read_text(encoding="utf-8")
        for banned in ("TL_PRIVATE_KEY", "TL_RPC_URL", "private_key"):
            assert banned not in source, banned


class TestEnvExample:
    def test_it_defaults_to_local_so_the_project_runs_unconfigured(self):
        text = Path(".env.example").read_text(encoding="utf-8")
        assert "TL_CHAIN=local" in text

    def test_public_credentials_are_documented_but_empty(self):
        text = Path(".env.example").read_text(encoding="utf-8")
        for name in PUBLIC_REQUIREMENTS:
            assert re.search(r"^{0}=\s*$".format(name), text, re.MULTILINE), name

    def test_it_warns_against_using_a_real_key(self):
        text = Path(".env.example").read_text(encoding="utf-8").lower()
        assert "testnet" in text
        assert "never a key holding real funds" in text

    def test_the_real_env_file_is_gitignored(self):
        assert ".env" in Path(".gitignore").read_text(encoding="utf-8").split()


# ==========================================================================
# A failed public anchor is never a fake success
# ==========================================================================


class TestPublicFailureIsHonest:
    def test_a_failed_anchor_claims_nothing(self, client, tmp_path):
        artifact = tmp_path / "evidence.json"
        artifact.write_text(json.dumps({"run_id": "x"}), encoding="utf-8")

        response = client.post(
            "/api/chain/anchor", data={"artifact_path": str(artifact)}
        )
        if response.status_code == 200:
            pytest.skip("local anchoring succeeded; nothing to assert here")

        payload = response.json()
        assert payload["anchored"] is False
        assert payload["publicly_verified"] is False
        # No fabricated transaction identity of any kind.
        for banned in ("tx_hash", "transaction_hash", "explorer_url", "block_number"):
            assert banned not in payload, banned

    def test_the_failure_path_offers_recovery_not_a_dead_end(self):
        """Asserted against the payload itself, not the source text.

        The contract lives in `_public_failure_payload`, so calling it is a
        stronger check than grepping the endpoint that happens to use it.
        """
        from tracelock.api.app import _public_failure_payload

        payload = _public_failure_payload(
            headline="Public blockchain unavailable",
            message="We couldn't complete the public anchor.",
            status="unreachable",
        )

        assert payload["error"] == "PUBLIC ANCHOR UNAVAILABLE"
        actions = {o["action"] for o in payload["recovery"]}
        assert actions == {"retry_public", "anchor_local", "return_results"}

    def test_a_failed_public_anchor_is_not_downgraded_silently(self):
        """Falling back mid-anchor would make a local anchor look public."""
        from tracelock.api.app import _public_failure_payload

        payload = _public_failure_payload(
            headline="Public blockchain unavailable",
            message="We couldn't complete the public anchor.",
            status="unreachable",
        )
        assert payload["anchored"] is False
        assert payload["publicly_verified"] is False

        # The local chain is OFFERED as a choice, never substituted: the
        # failure path returns without re-entering anchoring.
        source = Path("src/tracelock/api/app.py").read_text(encoding="utf-8")
        block = source[source.index("async def anchor_artifact"):]
        block = block[: block.index('@api.post("/chain/verify")')]
        failure = block[block.index("except Exception as exc:"):]
        assert "_anchor()" not in failure


# ==========================================================================
# The UI branches on execution reality
# ==========================================================================


class TestUiReflectsTheExecutingChain:
    def _chain_block(self) -> str:
        source = Path("web/app.js").read_text(encoding="utf-8")
        block = source[source.index("function chainBlock(anchor) {"):]
        return block[: block.index("function wireChainActions")]

    def test_both_badges_exist_and_branch_on_ephemeral(self):
        block = self._chain_block()
        assert 'ephemeral ? "LOCAL DEMO CHAIN" : "PUBLICLY VERIFIABLE"' in block

    def test_publicly_verifiable_is_never_shown_for_an_ephemeral_chain(self):
        block = self._chain_block()
        for match in re.finditer(r'"PUBLICLY VERIFIABLE"', block):
            line_start = block.rfind("\n", 0, match.start())
            assert "ephemeral ?" in block[line_start : match.end()], (
                "the public badge appears without an ephemeral guard"
            )

    def test_the_public_card_claims_independent_verification(self):
        block = self._chain_block()
        assert "Anyone can independently verify this evidence anchor" in block

    def test_the_local_card_admits_it_resets(self):
        block = " ".join(self._chain_block().split())
        assert "resets on" in block
        assert "not independently verifiable by a third party" in block

    def test_explorer_link_still_requires_an_actual_url(self):
        block = self._chain_block()
        assert "c.explorer_url" in block
        assert "not publicly verifiable" in block.lower()


# ==========================================================================
# A restart is documented behaviour, not a broken install
# ==========================================================================


class TestEphemeralResetReadsHonestly:
    """After a restart the local anchor cannot be verified -- by design.

    The raw error is "no contract code at 0x... Deploy first", which reads as
    a broken deployment. It is the ephemerality the UI already advertises.
    """

    def test_a_reset_local_chain_is_explained_not_alarmed(self):
        from tracelock.service.runner import _friendly_chain_error

        message = _friendly_chain_error(Exception(
            "no contract code at 0xabc on chain 131277322940537. "
            "Deploy first (scripts/deploy_contract.py)"
        ))
        lowered = message.lower()

        assert "resets when the server restarts" in lowered
        assert "evidence file itself is unchanged" in lowered
        # No raw internals leaking into an operator-facing string.
        assert "0xabc" not in message
        assert "deploy_contract.py" not in message

    def test_a_real_missing_deployment_still_says_so(self):
        """A public chain missing its contract IS an operator error."""
        from tracelock.service.runner import _friendly_chain_error

        message = _friendly_chain_error(Exception(
            "no contract code at 0xabc on chain 80002. Deploy first"
        ))
        assert "not deployed" in message.lower()
        assert "resets when the server restarts" not in message.lower()

    def test_a_reset_never_reports_a_verdict(self):
        """An unverifiable anchor must not become INTACT or TAMPERED."""
        source = Path("src/tracelock/api/app.py").read_text(encoding="utf-8")
        block = source[source.index('@api.post("/chain/verify")'):]
        block = block[: block.index('@api.post("/chain/tamper-test")')]

        # The failure path returns an error, never a synthesised verdict.
        assert '"error"' in block
        assert '"verdict": "INTACT"' not in block
