"""Chain identity truthfulness.

THE BUG THIS EXISTS TO PREVENT
------------------------------
The interface once advertised "chain: amoy" while every anchor was written to
an ephemeral in-process EVM. Two independent sources of truth had drifted
apart: the status endpoint read TL_CHAIN (default "amoy") while the browser
hardcoded "local".

Claiming a public chain for a local-only anchor is the most damaging lie this
system can tell, because the entire value of anchoring is that a third party
can independently check it. A local anchor is unverifiable by anyone.

THE RULE
--------
The chain label must come from the NetworkProfile that actually executed the
transaction. Not from configuration intent, not from an environment default,
not from a UI constant.
"""

from __future__ import annotations

import json

import pytest

from tracelock.chain.adapter import NETWORKS, NetworkProfile

web3 = pytest.importorskip("web3")
pytest.importorskip("eth_tester")

PUBLIC_NAMES = ("amoy", "polygon", "sepolia", "base", "mainnet", "ethereum")


def make_artifact():
    return {
        "schema_version": "evidence-report/1",
        "run_id": "evidence_chain_identity_test",
        "phase": "3-evidence-aggregation",
        "source_artifacts": {},
        "probe": {
            "sha256": "a" * 64,
            "model_id": "buffalo_l:w600k_r50.onnx",
            "embedding_dimension": 512,
            "embedding_quantized_sha256": "b" * 64,
        },
        "verification_policy": {"calibrated": True},
        "evidence": {
            "funnel": {"unique_images": 2, "independent_publishers": 2},
            "independent_domains": ["a.com", "b.com"],
            "items": [],
        },
        "trust_score": {"score": 88.0, "band": "STRONG"},
        "scored": True,
    }


@pytest.fixture(scope="module")
def compiled():
    from tracelock.chain.compiler import load_or_compile

    return load_or_compile()


# ==========================================================================
# Profile-level identity
# ==========================================================================


class TestNetworkProfileIdentity:
    def test_local_is_marked_ephemeral(self):
        assert NETWORKS["local"].ephemeral is True

    def test_public_networks_are_not_ephemeral(self):
        for key, profile in NETWORKS.items():
            if key != "local":
                assert profile.ephemeral is False, key

    def test_local_display_name_names_no_public_chain(self):
        name = NETWORKS["local"].display_name.lower()
        for token in PUBLIC_NAMES:
            assert token not in name, "local profile must not name {0}".format(token)

    def test_local_display_name_says_local_and_demo(self):
        assert NETWORKS["local"].display_name == "LOCAL DEMO CHAIN"

    def test_amoy_display_name_is_exact(self):
        assert NETWORKS["amoy"].display_name == "Polygon Amoy Testnet"

    def test_local_never_produces_an_explorer_link(self):
        profile = NETWORKS["local"]
        assert profile.tx_url("0x" + "a" * 64) == ""
        assert profile.address_url("0x" + "b" * 40) == ""

    def test_local_is_not_publicly_verifiable(self):
        assert NETWORKS["local"].publicly_verifiable is False

    def test_public_networks_are_publicly_verifiable(self):
        for key, profile in NETWORKS.items():
            if key != "local":
                assert profile.publicly_verifiable is True, key
                assert profile.tx_url("0xabc").startswith("http")

    def test_ephemeral_profile_cannot_emit_a_link_even_with_a_template(self):
        # Belt and braces: if someone later gives a local-style profile an
        # explorer template, `ephemeral` still suppresses the link.
        rogue = NetworkProfile(
            key="rogue", name="ROGUE", chain_id=1,
            explorer_tx="https://example.com/tx/{0}",
            explorer_address="https://example.com/a/{0}",
            faucet="", ephemeral=True,
        )
        assert rogue.tx_url("0xabc") == ""
        assert rogue.address_url("0xabc") == ""
        assert rogue.publicly_verifiable is False

    def test_persistence_note_distinguishes_the_two(self):
        assert "Ephemeral" in NETWORKS["local"].persistence_note
        assert "resets on restart" in NETWORKS["local"].persistence_note
        assert "verifiable" in NETWORKS["amoy"].persistence_note


# ==========================================================================
# Receipts and anchor records carry the EXECUTING chain
# ==========================================================================


class TestAnchorRecordIdentity:
    def test_local_anchor_is_labelled_local_and_ephemeral(self, compiled):
        from tracelock.chain.adapter import make_local_adapter
        from tracelock.chain.notary import EvidenceNotary

        adapter = make_local_adapter(compiled)
        address, _ = adapter.deploy(compiled)
        record, receipt = EvidenceNotary(adapter, address).anchor(make_artifact())

        assert receipt.ephemeral is True
        assert receipt.network_display_name == "LOCAL DEMO CHAIN"
        assert record.ephemeral is True
        assert record.explorer_url == ""

    def test_local_anchor_record_cannot_render_amoy(self, compiled):
        """THE REQUIRED TEST: local adapter => UI cannot render 'amoy'."""
        from tracelock.chain.adapter import make_local_adapter
        from tracelock.chain.notary import EvidenceNotary

        adapter = make_local_adapter(compiled)
        address, _ = adapter.deploy(compiled)
        record, _ = EvidenceNotary(adapter, address).anchor(make_artifact())

        # Everything the UI is given, serialized exactly as it is sent.
        rendered = json.dumps(record.to_dict()).lower()
        for token in PUBLIC_NAMES:
            assert token not in rendered, (
                "a LOCAL anchor leaked the public-chain token {0!r} into the "
                "payload the UI renders".format(token)
            )

    def test_local_anchor_states_it_is_not_verifiable(self, compiled):
        from tracelock.chain.adapter import make_local_adapter
        from tracelock.chain.notary import EvidenceNotary

        adapter = make_local_adapter(compiled)
        address, _ = adapter.deploy(compiled)
        record, _ = EvidenceNotary(adapter, address).anchor(make_artifact())

        payload = record.to_dict()
        assert payload["on_chain"]["ephemeral"] is True
        assert payload["on_chain"]["publicly_verifiable"] is False
        proves = payload["proves"].lower()
        assert "ephemeral local chain" in proves
        assert "cannot be verified by anyone else" in proves

    def test_amoy_record_renders_the_public_name_and_explorer(self):
        """THE REQUIRED TEST: amoy adapter => UI renders 'Polygon Amoy Testnet'.

        Built from an Amoy profile rather than a live transaction: this asserts
        the LABELLING contract, and a real Amoy anchor would need a funded
        wallet and network access, which the suite must never require.
        """
        from tracelock.chain.notary import AnchorRecord

        amoy = NETWORKS["amoy"]
        tx_hash = "0x" + "ab" * 32

        record = AnchorRecord(
            schema_version="evidence-anchor/1",
            fingerprint_version="tracelock-fingerprint/1",
            merkle_root="0x" + "cd" * 32,
            run_id="evidence_test", case_id="0x" + "ef" * 32,
            probe_commitment="0x" + "11" * 32, pipeline_hash="0x" + "22" * 32,
            leaves={}, trust_score_bp=8800, evidence_count=2,
            independent_publishers=2,
            tx_hash=tx_hash, chain_id=amoy.chain_id, network=amoy.key,
            contract_address="0x" + "33" * 20, block_number=12345,
            block_timestamp=1788000000, gas_used=73617,
            submitter="0x" + "44" * 20,
            explorer_url=amoy.tx_url(tx_hash), confirmations=2,
            network_display_name=amoy.display_name, ephemeral=amoy.ephemeral,
            anchored_at="2026-09-01T00:00:00+00:00",
        )

        chain = record.to_dict()["on_chain"]
        assert chain["network_display_name"] == "Polygon Amoy Testnet"
        assert chain["ephemeral"] is False
        assert chain["publicly_verifiable"] is True
        assert chain["explorer_url"] == "https://amoy.polygonscan.com/tx/{0}".format(tx_hash)
        assert chain["chain_id"] == 80002

    def test_record_round_trip_preserves_identity(self, compiled):
        from tracelock.chain.adapter import make_local_adapter
        from tracelock.chain.notary import AnchorRecord, EvidenceNotary

        adapter = make_local_adapter(compiled)
        address, _ = adapter.deploy(compiled)
        record, _ = EvidenceNotary(adapter, address).anchor(make_artifact())

        restored = AnchorRecord.from_dict(json.loads(json.dumps(record.to_dict())))
        assert restored.ephemeral is True
        assert restored.network_display_name == "LOCAL DEMO CHAIN"


# ==========================================================================
# One source of truth: the server, not the browser
# ==========================================================================


class TestSingleSourceOfTruth:
    def test_status_reports_the_chain_that_will_execute(self, monkeypatch):
        from fastapi.testclient import TestClient

        from tracelock.api import create_app

        monkeypatch.setenv("TL_CHAIN", "local")
        payload = TestClient(create_app()).get("/api/chain/status").json()

        assert payload["network"] == "local"
        assert payload["network_display_name"] == "LOCAL DEMO CHAIN"
        assert payload["ephemeral"] is True
        assert payload["publicly_verifiable"] is False
        assert payload["explorer"] == ""

    def test_local_status_payload_names_no_public_chain(self, monkeypatch):
        from fastapi.testclient import TestClient

        from tracelock.api import create_app

        monkeypatch.setenv("TL_CHAIN", "local")
        rendered = json.dumps(
            TestClient(create_app()).get("/api/chain/status").json()
        ).lower()
        for token in PUBLIC_NAMES:
            assert token not in rendered

    def test_amoy_status_reports_the_public_identity(self, monkeypatch):
        """A FULLY configured public chain reports its public identity.

        Credentials are supplied because that is what makes the claim true.
        Requesting amoy without an RPC URL or a key now resolves to local --
        see `test_requesting_a_public_chain_without_credentials_reports_local`
        -- since no anchor could ever reach Amoy in that state, and advertising
        it would be a claim the system cannot honour.
        """
        from fastapi.testclient import TestClient

        from tracelock.api import create_app

        monkeypatch.setenv("TL_CHAIN", "amoy")
        monkeypatch.setenv("TL_RPC_URL", "https://rpc.example.invalid")
        monkeypatch.setenv("TL_PRIVATE_KEY", "0x" + "a" * 64)

        payload = TestClient(create_app()).get("/api/chain/status").json()

        assert payload["network_display_name"] == "Polygon Amoy Testnet"
        assert payload["ephemeral"] is False
        assert payload["publicly_verifiable"] is True
        assert payload["chain_id"] == 80002

    def test_requesting_a_public_chain_without_credentials_reports_local(
        self, monkeypatch
    ):
        """The status must describe where an anchor would ACTUALLY land.

        Before the two-mode work, TL_CHAIN=amoy with no credentials made the
        status advertise Polygon Amoy while every anchor attempt failed. The
        honest answer is the chain that can actually execute.
        """
        from fastapi.testclient import TestClient

        from tracelock.api import create_app

        monkeypatch.setenv("TL_CHAIN", "amoy")
        monkeypatch.delenv("TL_RPC_URL", raising=False)
        monkeypatch.delenv("TL_PRIVATE_KEY", raising=False)

        payload = TestClient(create_app()).get("/api/chain/status").json()

        assert payload["network_display_name"] == "LOCAL DEMO CHAIN"
        assert payload["ephemeral"] is True
        assert payload["publicly_verifiable"] is False

    def test_default_network_is_the_honest_one(self):
        # An unconfigured install must report LOCAL, not advertise a public
        # chain it cannot actually reach.
        from tracelock.chain.config import DEFAULT_NETWORK

        assert DEFAULT_NETWORK == "local"

    def test_browser_cannot_choose_the_network(self):
        """The network is server configuration, not a client parameter.

        Accepting it from the request is exactly how the label and the
        transaction diverged in the first place.
        """
        import importlib
        import inspect

        # `tracelock.api.app` resolves to the FastAPI instance re-exported in
        # __init__, so the module has to be imported explicitly.
        app_module = importlib.import_module("tracelock.api.app")

        for name in ("investigate", "anchor_artifact", "verify_anchor"):
            function = getattr(app_module, name)
            parameters = inspect.signature(function).parameters
            assert "network" not in parameters, (
                "{0}() accepts a client-supplied network; the server must own "
                "this choice".format(name)
            )


class TestUiRendersOnlyTheExecutingIdentity:
    """Source-level guards on the browser bundle."""

    def _app_js(self) -> str:
        from pathlib import Path

        return (
            Path(__file__).resolve().parents[1] / "web" / "app.js"
        ).read_text(encoding="utf-8")

    def test_ui_hardcodes_no_network(self):
        source = self._app_js()
        for token in ('network: "local"', "network:'local'", '"amoy"', "'amoy'"):
            assert token not in source, (
                "web/app.js hardcodes {0!r}; the chain label must come from the "
                "server".format(token)
            )

    def test_ui_renders_the_display_name_field(self):
        assert "network_display_name" in self._app_js()

    def test_ui_branches_on_ephemeral(self):
        source = self._app_js()
        assert "ephemeral" in source
        assert "resets on restart" in source.lower()

    def test_ui_only_links_an_explorer_when_one_exists(self):
        # The explorer link must be conditional on explorer_url being present,
        # which an ephemeral chain never provides.
        source = self._app_js()
        assert "c.explorer_url" in source
        assert "not publicly verifiable" in source.lower()
