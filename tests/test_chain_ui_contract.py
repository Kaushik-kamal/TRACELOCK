"""The anchor section's contract: honest labels, and a REAL tamper test.

The UI change this covers is presentational, but two things underneath it are
not, and both are load-bearing:

  * the tamper test must actually corrupt something and actually re-verify --
    the previous button printed a CLI command and claimed nothing
  * the evidence file must survive that test untouched

And the honesty the redesign must not lose: a local ephemeral chain stays
labelled as one, and never acquires an explorer URL or a public network name.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

WEB = Path("web")


@pytest.fixture
def client():
    from tracelock.api import create_app

    return TestClient(create_app())


def read(name: str) -> str:
    return (WEB / name).read_text(encoding="utf-8")


# ==========================================================================
# The tamper test is real
# ==========================================================================


class TestTamperTestIsReal:
    def test_the_endpoint_exists_and_is_not_a_canned_message(self):
        """The old button printed a CLI command and detected nothing."""
        app_source = Path(
            "src/tracelock/api/app.py"
        ).read_text(encoding="utf-8")

        assert "/chain/tamper-test" in app_source
        # It must call the SAME verification the normal verify button calls.
        block = app_source[app_source.index("async def tamper_test"):]
        block = block[: block.index("# =====")]
        # Count CALL SITES, not prose -- the docstring mentions the name too.
        calls = block.count("notary.reverify(")
        assert calls == 2, (
            "must verify BOTH the untouched and the mutated bundle -- one "
            "verdict alone proves nothing (found {0} calls)".format(calls)
        )

    def test_ui_no_longer_tells_the_user_to_run_a_cli_command(self):
        source = read("app.js")
        assert "verify_anchor.py --evidence" not in source
        assert "/api/chain/tamper-test" in source

    def test_missing_anchor_is_refused_not_faked(self, client, tmp_path):
        artifact = tmp_path / "evidence.json"
        artifact.write_text(json.dumps({"run_id": "x"}), encoding="utf-8")

        response = client.post(
            "/api/chain/tamper-test", data={"artifact_path": str(artifact)}
        )
        assert response.status_code == 400
        payload = response.json()
        assert "not been anchored" in payload["error"]
        assert "detected" not in payload

    def test_missing_artifact_is_a_404(self, client, tmp_path):
        response = client.post(
            "/api/chain/tamper-test",
            data={"artifact_path": str(tmp_path / "nope.json")},
        )
        assert response.status_code == 404

    def test_tamper_targets_match_the_cli(self):
        """The UI demo and the CLI demo must corrupt the same fields.

        If they drift, the button demonstrates something the documented
        script does not.
        """
        from tracelock.api.app import TAMPER_TARGETS

        script = Path("scripts/verify_anchor.py").read_text(encoding="utf-8")
        block = script[script.index("TAMPER_TARGETS = {"):]
        block = block[: block.index("}")]

        for key in TAMPER_TARGETS:
            assert '"{0}"'.format(key) in block, key


# ==========================================================================
# The redesign did not cost any honesty
# ==========================================================================


class TestLocalChainStaysHonest:
    def test_the_environment_label_is_still_present(self):
        source = read("app.js")
        assert "LOCAL DEMO CHAIN" in source
        assert "ephemeral" in source

    def test_a_local_chain_is_never_labelled_as_a_public_network(self):
        """The badge for an ephemeral chain is fixed text, not a network name.

        Reading `network_display_name` in the ephemeral branch is exactly how a
        local anchor could come to display "Amoy".
        """
        source = read("app.js")
        block = source[source.index("function chainBlock(anchor) {"):]
        block = block[: block.index("function wireChainActions")]

        badge = re.search(r"env-badge[\s\S]*?</span>", block)
        assert badge, "environment badge not found"
        assert 'ephemeral ? "LOCAL DEMO CHAIN"' in badge.group(0)

    def test_no_public_network_name_is_hardcoded_anywhere_in_the_ui(self):
        source = read("app.js").lower()
        for network in ("amoy", "polygon", "sepolia", "mainnet", "ethereum"):
            assert network not in source, network

    def test_the_limitation_text_survives_the_redesign(self):
        """Softening the visual treatment must not soften the disclosure."""
        # Collapsed so a line wrap in the template cannot hide a phrase.
        source = " ".join(read("app.js").split())

        for phrase in (
            "resets on restart",
            "not independently verifiable by a third party",
            "Integrity and tamper detection are fully functional",
            "not publicly verifiable",
        ):
            assert phrase in source, phrase

    def test_the_integrity_not_truth_caveat_survives(self):
        source = read("app.js")
        assert "integrity</strong>, not truth" in source
        assert "cannot prove the original source" in source


class TestSuccessIsThePrimaryMessage:
    def test_a_successful_anchor_does_not_lead_with_a_warning(self):
        source = read("app.js")
        block = source[source.index("function chainBlock(anchor) {"):]
        block = block[: block.index("function wireChainActions")]
        anchored = block[block.index('<div class="chain anchored">'):]
        headline = anchored[: anchored.index("env-badge")]

        assert "EVIDENCE INTEGRITY ANCHORED" in headline
        assert "⚠" not in headline, "a warning glyph leads the success state"

    def test_an_ephemeral_anchor_no_longer_uses_the_not_anchored_style(self):
        """`chain none` is the NOT-ANCHORED style. A successful ephemeral
        anchor sharing it is what made a working feature look broken."""
        source = read("app.js")
        block = source[source.index("function chainBlock(anchor) {"):]
        block = block[: block.index("function wireChainActions")]

        assert 'ephemeral ? "none" : "anchored"' not in block
        assert '<div class="chain anchored">' in block

    def test_the_environment_card_is_informational_not_an_error(self):
        css = read("styles.css")
        env = css[css.index(".env-card {"):]
        env = env[: env.index("}")]
        assert "--info" in env, "the environment card should read as information"
        assert "--fail" not in env


class TestVerifyButtonMatchesExecutionReality:
    def test_the_label_depends_on_whether_the_chain_is_public(self):
        source = read("app.js")
        assert 'ephemeral ? "VERIFY INTEGRITY" : "VERIFY ON BLOCKCHAIN"' in source

    def test_a_local_chain_never_offers_to_verify_on_a_blockchain(self):
        """Wording must not imply public verification of a local anchor."""
        source = read("app.js")
        occurrences = source.count('"VERIFY ON BLOCKCHAIN"')
        # Only inside the ternary, twice (render + restore after click).
        assert occurrences == 2
        for match in re.finditer(r'"VERIFY ON BLOCKCHAIN"', source):
            line_start = source.rfind("\n", 0, match.start())
            line = source[line_start : match.end() + 40]
            assert "ephemeral ?" in line, (
                "the public wording appears without an ephemeral guard"
            )


class TestHashReadability:
    def test_hashes_are_truncated_for_display(self):
        source = read("app.js")
        assert "function shortHash(" in source
        assert "…" in source

    def test_the_full_value_is_still_reachable(self):
        """Truncation must not remove access to the real hash."""
        source = read("app.js")
        assert 'data-copy="${esc(value)}"' in source
        assert 'title="${esc(value)}"' in source
        assert "clipboard.writeText" in source


# ==========================================================================
# The local chain must survive long enough to be verified
# ==========================================================================


class TestLocalChainRoundTrip:
    """Anchor then verify must work within one process.

    Every call used to build a NEW in-memory EVM and redeploy the contract, so
    an anchor written by one request lived on a different chain than the
    request that tried to verify it. VERIFY and TAMPER TEST failed with "no
    contract code at 0x...", which reads as a broken feature rather than as
    the ephemerality it actually was.
    """

    def test_the_same_local_chain_is_reused_within_a_process(self):
        from tracelock.chain.compiler import load_or_compile
        from tracelock.chain.config import (
            build_adapter_with_contract,
            load_chain_config,
            reset_local_chain,
        )

        reset_local_chain()
        try:
            config = load_chain_config(network="local")
            compiled = load_or_compile()

            adapter_a, address_a, auto_a = build_adapter_with_contract(config, compiled)
            adapter_b, address_b, auto_b = build_adapter_with_contract(config, compiled)

            assert auto_a and auto_b, "local chain should self-deploy"
            assert address_a == address_b, "contract redeployed on a fresh chain"
            assert adapter_a is adapter_b, "a second EVM was created"
        finally:
            reset_local_chain()

    def test_resetting_gives_a_genuinely_new_chain(self):
        """Still ephemeral: the cache is per-process, not persistence."""
        from tracelock.chain.compiler import load_or_compile
        from tracelock.chain.config import (
            build_adapter_with_contract,
            load_chain_config,
            reset_local_chain,
        )

        reset_local_chain()
        try:
            config = load_chain_config(network="local")
            compiled = load_or_compile()
            first, _, _ = build_adapter_with_contract(config, compiled)
            reset_local_chain()
            second, _, _ = build_adapter_with_contract(config, compiled)
            assert first is not second
        finally:
            reset_local_chain()

    def test_public_networks_are_never_cached(self):
        """Caching a public adapter could serve a stale contract address."""
        source = Path(
            "src/tracelock/chain/config.py"
        ).read_text(encoding="utf-8")
        block = source[source.index("def build_adapter_with_contract"):]

        early_return = block.index("if not config.is_local:")
        cache_read = block.index("_LOCAL_CHAIN.get(")
        assert early_return < cache_read, (
            "public networks must return before touching the local cache"
        )
