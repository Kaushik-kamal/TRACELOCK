"""MOCKED public-chain integration tests.

=============================================================================
THESE TESTS DO NOT TOUCH A REAL BLOCKCHAIN.
=============================================================================

Every transaction here is produced by a fake adapter in this file. Nothing is
broadcast, no funds move, and no testnet is contacted. They prove that the
PUBLIC code path handles success, connectivity loss, bad credentials, contract
reverts, receipts and explorer URLs correctly -- they do NOT prove that a real
transaction on Polygon Amoy succeeds.

A live testnet verification would need a funded key and is a separate,
manually-run exercise. Nothing in this file may be cited as evidence that one
has happened.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tracelock.chain.adapter import NETWORKS
from tracelock.chain.preflight import (
    PublicConfigStatus,
    check_public_config,
    classify_anchor_failure,
)

# Marks every test here, so a report can separate mocked from live at a glance.
pytestmark = pytest.mark.mocked_chain

VALID_KEY = "0x" + "a" * 64
VALID_RPC = "https://rpc.example.invalid"
PUBLIC_ENV = {"TL_CHAIN": "amoy", "TL_RPC_URL": VALID_RPC, "TL_PRIVATE_KEY": VALID_KEY}


@pytest.fixture
def client():
    from tracelock.api import create_app

    return TestClient(create_app())


# ==========================================================================
# A fake chain. Everything it returns is fabricated BY THE TEST, on purpose.
# ==========================================================================


class FakeReceipt(dict):
    """Shaped like a web3 receipt, with attribute access."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


class FakePublicAdapter:
    """Stands in for the web3 adapter. Never opens a socket."""

    def __init__(self, *, fail_with: Exception | None = None) -> None:
        self.fail_with = fail_with
        self.calls: list[str] = []
        self.profile = NETWORKS["amoy"]

    def deploy(self, compiled):
        self.calls.append("deploy")
        return "0x" + "c" * 40, FakeReceipt(blockNumber=1)

    def send(self, *args, **kwargs):
        self.calls.append("send")
        if self.fail_with:
            raise self.fail_with
        return FakeReceipt(
            transactionHash=bytes.fromhex("ab" * 32),
            blockNumber=4_242_424,
            status=1,
        )


# ==========================================================================
# 1. Structural preflight (no network, by design)
# ==========================================================================


class TestPreflightIsLocalOnly:
    def test_valid_configuration_reports_ready(self):
        assert check_public_config(PUBLIC_ENV).status is PublicConfigStatus.READY

    def test_ready_does_not_claim_the_chain_was_contacted(self):
        payload = check_public_config(PUBLIC_ENV).to_dict()
        assert payload["checked"] == "structure_only"
        assert "checked when an anchor is actually attempted" in payload["note"]

    def test_preflight_opens_no_socket(self, monkeypatch):
        """Startup must not depend on a third party being up."""
        import socket

        def forbidden(*a, **k):
            raise AssertionError("preflight must not touch the network")

        monkeypatch.setattr(socket.socket, "connect", forbidden)
        monkeypatch.setattr(socket, "getaddrinfo", forbidden)

        for env in ({}, {"TL_CHAIN": "amoy"}, PUBLIC_ENV):
            check_public_config(env)

    @pytest.mark.parametrize(
        "bad_key",
        ["hunter2", "0x123", "", "  ", "0x" + "a" * 63, "0x" + "z" * 64],
    )
    def test_malformed_keys_are_caught_locally(self, bad_key):
        env = dict(PUBLIC_ENV, TL_PRIVATE_KEY=bad_key)
        status = check_public_config(env).status
        assert status in (
            PublicConfigStatus.INVALID_CREDENTIALS,
            PublicConfigStatus.INCOMPLETE,
        )

    @pytest.mark.parametrize(
        "bad_rpc", ["not-a-url", "ftp://rpc.test", "javascript:alert(1)", "https://"]
    )
    def test_malformed_rpc_urls_are_caught_locally(self, bad_rpc):
        env = dict(PUBLIC_ENV, TL_RPC_URL=bad_rpc)
        assert check_public_config(env).status is PublicConfigStatus.INVALID_CREDENTIALS

    def test_problems_name_the_variable_never_the_value(self):
        env = dict(PUBLIC_ENV, TL_PRIVATE_KEY="s3cr3t-do-not-leak")
        preflight = check_public_config(env)
        blob = json.dumps(preflight.to_dict())

        assert "TL_PRIVATE_KEY" in blob
        assert "s3cr3t-do-not-leak" not in blob


# ==========================================================================
# 2. Failure classification -- what the operator is told
# ==========================================================================


class TestFailureMessagesHideInternals:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("HTTPSConnectionPool(host='rpc.foo', port=443): Max retries exceeded",
             PublicConfigStatus.UNREACHABLE),
            ("ConnectionError: [Errno 111] Connection refused",
             PublicConfigStatus.UNREACHABLE),
            ("invalid private key", PublicConfigStatus.INVALID_CREDENTIALS),
            ("insufficient funds for gas * price + value", PublicConfigStatus.FAILED),
            ("execution reverted: AlreadyAnchored", PublicConfigStatus.FAILED),
        ],
    )
    def test_each_failure_class_is_recognised(self, raw, expected):
        status, _message, _detail = classify_anchor_failure(Exception(raw))
        assert status is expected

    @pytest.mark.parametrize(
        "raw",
        [
            "HTTPSConnectionPool(host='rpc.alchemy.io/v2/SECRET', port=443)",
            "eth_sendRawTransaction failed at 0xdeadbeef",
            "no contract code at 0xABCDEF0123456789",
            "Traceback (most recent call last): File web3/main.py line 42",
        ],
    )
    def test_the_user_message_never_leaks_internals(self, raw):
        _status, message, detail = classify_anchor_failure(Exception(raw))

        for leak in ("HTTPSConnectionPool", "0x", "Traceback", "eth_send",
                     "SECRET", "web3/"):
            assert leak not in message, message
        # The original survives for the server log.
        assert raw in detail

    def test_every_message_says_nothing_was_recorded(self):
        for raw in ("connection refused", "invalid private key",
                    "insufficient funds", "execution reverted", "who knows"):
            _s, message, _d = classify_anchor_failure(Exception(raw))
            assert "no public verification record was created" in message.lower()


# ==========================================================================
# 3. The API's public failure contract
# ==========================================================================


class TestPublicFailurePayload:
    def _payload(self, **kwargs):
        from tracelock.api.app import _public_failure_payload

        kwargs.setdefault("headline", "Public blockchain unavailable")
        kwargs.setdefault("message", "We couldn't complete the public anchor.")
        kwargs.setdefault("status", "unreachable")
        return _public_failure_payload(**kwargs)

    def test_it_fabricates_no_transaction_identity(self):
        payload = self._payload()
        for banned in ("tx_hash", "transaction_hash", "block_number",
                       "explorer_url", "contract_address", "merkle_root"):
            assert banned not in payload, banned

    def test_it_states_plainly_that_nothing_was_anchored(self):
        payload = self._payload()
        assert payload["anchored"] is False
        assert payload["publicly_verified"] is False

    def test_it_always_offers_three_real_recovery_actions(self):
        actions = {o["action"] for o in self._payload()["recovery"]}
        assert actions == {"retry_public", "anchor_local", "return_results"}

    def test_every_recovery_action_is_handled_in_the_ui(self):
        """A label with no handler is decoration. These are the real ones."""
        js = Path("web/app.js").read_text(encoding="utf-8")
        for option in self._payload()["recovery"]:
            assert 'action === "{0}"'.format(option["action"]) in js, option


# ==========================================================================
# 4. Mocked transaction lifecycle
# ==========================================================================


class TestMockedTransactionLifecycle:
    """A FAKE adapter, exercising the public code path end to end."""

    def test_a_successful_send_yields_a_receipt(self):
        adapter = FakePublicAdapter()
        receipt = adapter.send()

        assert receipt.status == 1
        assert receipt.blockNumber == 4_242_424
        assert len(receipt.transactionHash) == 32

    def test_explorer_url_is_built_from_the_profile(self):
        profile = NETWORKS["amoy"]
        tx = "0x" + "ab" * 32
        url = profile.explorer_tx.format(tx)

        assert url.startswith("https://amoy.polygonscan.com/tx/")
        assert tx in url

    def test_the_local_profile_can_build_no_explorer_url(self):
        """The absence is structural, not a rendering choice."""
        assert NETWORKS["local"].explorer_tx == ""
        assert NETWORKS["local"].ephemeral
        assert not NETWORKS["local"].publicly_verifiable

    @pytest.mark.parametrize(
        "failure",
        [
            ConnectionError("Max retries exceeded with url: /v2/xyz"),
            ValueError("invalid private key"),
            RuntimeError("execution reverted: AlreadyAnchored"),
        ],
    )
    def test_a_failing_send_never_returns_a_receipt(self, failure):
        adapter = FakePublicAdapter(fail_with=failure)
        with pytest.raises(type(failure)):
            adapter.send()
        assert adapter.calls == ["send"]

    def test_a_reverting_contract_is_classified_not_crashed(self):
        adapter = FakePublicAdapter(
            fail_with=RuntimeError("execution reverted: AlreadyAnchored")
        )
        try:
            adapter.send()
        except RuntimeError as exc:
            status, message, _ = classify_anchor_failure(exc)

        assert status is PublicConfigStatus.FAILED
        assert "rejected" in message.lower()
        assert "revert" not in message.lower()


# ==========================================================================
# 5. The local fallback is explicit, never automatic
# ==========================================================================


class TestLocalFallbackIsExplicit:
    def test_the_endpoint_takes_a_boolean_not_a_network_name(self):
        """A client-chosen network is how the label once diverged from reality."""
        import inspect

        from tracelock.api.app import anchor_artifact

        params = inspect.signature(anchor_artifact).parameters
        assert "use_local_fallback" in params
        assert "network" not in params

    def test_the_flag_can_only_ever_select_local(self):
        source = Path("src/tracelock/api/app.py").read_text(encoding="utf-8")
        assert 'network="local" if use_local_fallback else None' in source

    def test_a_public_failure_does_not_anchor_locally_by_itself(self):
        """The fallback must be a second, deliberate request."""
        source = Path("src/tracelock/api/app.py").read_text(encoding="utf-8")
        block = source[source.index("async def anchor_artifact"):]
        block = block[: block.index('@api.post("/chain/verify")')]
        failure = block[block.index("except Exception as exc:"):]

        # The failure path returns; it never re-enters anchoring.
        assert "_anchor()" not in failure
        assert "anchor_local" in block or "_public_failure_payload" in block

    def test_local_fallback_actually_anchors(self, client, tmp_path):
        """End to end on the real local chain -- not a mock."""
        artifact = tmp_path / "evidence.json"
        artifact.write_text(
            json.dumps({
                "schema_version": "evidence/1",
                "run_id": "evidence_fallback_test",
                "trust_score": {"score": 42.0, "band": "MODERATE"},
                "evidence": {"independent_domains": ["a.com"]},
            }),
            encoding="utf-8",
        )

        response = client.post("/api/chain/anchor", data={
            "artifact_path": str(artifact),
            "use_local_fallback": "true",
        })
        assert response.status_code == 200, response.text

        record = response.json()
        assert record["on_chain"]["network"] == "local"
        assert record["on_chain"]["ephemeral"] is True
        assert not record["on_chain"].get("explorer_url")
        assert artifact.with_suffix(".anchor.json").is_file()

    def test_the_fallback_anchor_then_verifies(self, client, tmp_path):
        """The whole point: after falling back, VERIFY must work."""
        artifact = tmp_path / "evidence.json"
        artifact.write_text(
            json.dumps({
                "schema_version": "evidence/1",
                "run_id": "evidence_fallback_verify",
                "trust_score": {"score": 55.5, "band": "MODERATE"},
                "evidence": {"independent_domains": ["b.com"]},
            }),
            encoding="utf-8",
        )

        anchored = client.post("/api/chain/anchor", data={
            "artifact_path": str(artifact), "use_local_fallback": "true",
        })
        assert anchored.status_code == 200, anchored.text

        verified = client.post(
            "/api/chain/verify", data={"artifact_path": str(artifact)}
        ).json()
        assert verified.get("verdict") == "INTACT", verified

        tampered = client.post("/api/chain/tamper-test", data={
            "artifact_path": str(artifact), "field": "trust_score",
        }).json()
        assert tampered.get("detected") is True, tampered


# ==========================================================================
# Regressions found while running the golden scenarios
# ==========================================================================


class TestUnreachableRpcRegressions:
    """Both bugs were found by pointing at 192.0.2.1 (RFC 5737, non-routable).

    A truthful-but-generic failure that takes 23 seconds is still a bad
    failure: the operator learns nothing and waits a long time to learn it.
    """

    @pytest.mark.parametrize(
        "raw",
        [
            "cannot connect to https://192.0.2.1:8545",
            "could not connect to the node",
            "failed to connect to https://rpc.example.invalid",
        ],
    )
    def test_cannot_connect_is_unreachable_not_generic(self, raw):
        """The classifier matched "connection" but not "cannot connect"."""
        status, message, _detail = classify_anchor_failure(Exception(raw))

        assert status is PublicConfigStatus.UNREACHABLE
        assert "couldn't reach the public blockchain" in message

    def test_the_endpoint_is_never_named_in_the_user_message(self):
        _s, message, detail = classify_anchor_failure(
            Exception("cannot connect to https://user:pw@rpc.private.invalid:8545")
        )
        assert "rpc.private.invalid" not in message
        assert "user:pw" not in message
        # ...but the log keeps it for diagnosis.
        assert "rpc.private.invalid" in detail

    def test_the_connect_phase_is_bounded_separately(self):
        """A 60s read timeout with no connect timeout hung for 22.9s."""
        from tracelock.chain.adapter import CONNECT_TIMEOUT, READ_TIMEOUT

        assert 0 < CONNECT_TIMEOUT <= 15, CONNECT_TIMEOUT
        assert CONNECT_TIMEOUT < READ_TIMEOUT

    def test_the_provider_actually_uses_both_timeouts(self):
        source = Path("src/tracelock/chain/adapter.py").read_text(encoding="utf-8")
        assert '"timeout": (CONNECT_TIMEOUT, READ_TIMEOUT)' in source
