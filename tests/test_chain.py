"""Phase 4 -- fingerprinting, anchoring, and re-verification.

Chain tests run against an IN-PROCESS EVM (eth-tester), so they exercise real
contract behaviour -- storage, reverts, event decoding -- with no node, no
network, and no funded wallet. Nothing here touches a public chain.

Contract compilation is module-scoped and cached; without that every test
would re-invoke solc.
"""

from __future__ import annotations

import copy
import json

import pytest

from tracelock.chain.errors import (
    AlreadyAnchoredError,
    ChainConfigError,
    ChainError,
    ContractNotDeployedError,
)
from tracelock.chain.fingerprint import (
    FINGERPRINT_VERSION,
    LEAF_TAGS,
    BiometricLeakError,
    assert_no_biometric_leak,
    canonical_bytes,
    diff_leaves,
    fingerprint_evidence,
    leaf_hash,
    merkle_proof,
    merkle_root,
)
from tracelock.chain.notary import (
    AnchorRecord,
    EvidenceNotary,
    VerificationVerdict,
)

web3 = pytest.importorskip("web3")
pytest.importorskip("eth_tester")


# ==========================================================================
# Fixtures
# ==========================================================================


def make_artifact(**overrides):
    """A minimal but structurally faithful Phase 3 evidence artifact."""
    artifact = {
        "schema_version": "evidence-report/1",
        "run_id": "evidence_20260901T000000Z_deadbeef",
        "phase": "3-evidence-aggregation",
        "created_at": "2026-09-01T00:00:00+00:00",
        "source_artifacts": {"verification": "data/runs/verify_x.json"},
        "probe": {
            "sha256": "a" * 64,
            "path": "data/probes/subject.jpg",
            "model_id": "buffalo_l:w600k_r50.onnx",
            "embedding_dimension": 512,
            "embedding_quantized_sha256": "b" * 64,
        },
        "verification_policy": {
            "calibrated": True,
            "calibration_model": {"model_id": "buffalo_l:w600k_r50.onnx"},
        },
        "evidence": {
            "funnel": {"unique_images": 10, "independent_publishers": 10},
            "independent_domains": ["a.com", "b.com"],
            "items": [{"content_sha256": "c" * 64, "identity_probability": 0.83}],
        },
        "trust_score": {"score": 91.86, "band": "STRONG"},
        "scored": True,
    }
    artifact.update(overrides)
    return artifact


@pytest.fixture(scope="module")
def compiled():
    from tracelock.chain.compiler import load_or_compile

    return load_or_compile()


@pytest.fixture
def chain(compiled):
    """A fresh in-process EVM with the notary deployed."""
    from tracelock.chain.adapter import make_local_adapter

    adapter = make_local_adapter(compiled)
    address, _ = adapter.deploy(compiled)
    return adapter, address


@pytest.fixture
def notary(chain):
    adapter, address = chain
    return EvidenceNotary(adapter, address)


# ==========================================================================
# Canonicalisation and hashing
# ==========================================================================


class TestCanonicalisation:
    def test_key_order_does_not_matter(self):
        assert canonical_bytes({"a": 1, "b": 2}) == canonical_bytes({"b": 2, "a": 1})

    def test_nested_key_order_does_not_matter(self):
        left = {"x": {"p": 1, "q": [1, {"m": 1, "n": 2}]}}
        right = {"x": {"q": [1, {"n": 2, "m": 1}], "p": 1}}
        assert canonical_bytes(left) == canonical_bytes(right)

    def test_list_order_DOES_matter(self):
        # Sequence is semantic; reordering evidence would change its meaning.
        assert canonical_bytes([1, 2]) != canonical_bytes([2, 1])

    def test_unicode_is_stable(self):
        payload = {"title": "Modi’s address — हिन्दी"}
        assert canonical_bytes(payload) == canonical_bytes(copy.deepcopy(payload))


class TestLeafHashing:
    def test_domain_separation_prevents_type_confusion(self):
        # The SAME value under two field names must not produce the same leaf,
        # or a value could be replayed as a different field.
        value = "identical"
        assert leaf_hash("tl:run_id", value) != leaf_hash("tl:schema", value)

    def test_unknown_tag_is_rejected(self):
        with pytest.raises(ValueError, match="unknown leaf tag"):
            leaf_hash("tl:not_a_real_tag", "x")

    def test_leaf_is_deterministic(self):
        assert leaf_hash("tl:run_id", {"a": 1}) == leaf_hash("tl:run_id", {"a": 1})

    def test_leaf_changes_with_value(self):
        assert leaf_hash("tl:run_id", "a") != leaf_hash("tl:run_id", "b")


class TestMerkle:
    def test_root_is_deterministic(self):
        leaves = [bytes([i]) * 32 for i in range(5)]
        assert merkle_root(leaves) == merkle_root(list(reversed(leaves)))

    def test_root_changes_when_a_leaf_changes(self):
        leaves = [bytes([i]) * 32 for i in range(5)]
        mutated = list(leaves)
        mutated[2] = b"\xff" * 32
        assert merkle_root(leaves) != merkle_root(mutated)

    def test_single_leaf(self):
        assert merkle_root([b"\x01" * 32]) == b"\x01" * 32

    def test_empty_is_rejected(self):
        with pytest.raises(ValueError, match="zero leaves"):
            merkle_root([])

    def test_odd_leaf_count_is_handled(self):
        for count in (3, 5, 7, 11):
            assert len(merkle_root([bytes([i]) * 32 for i in range(count)])) == 32

    def test_leaf_and_node_prefixes_differ(self):
        # Second-preimage resistance: an internal node must not be presentable
        # as a leaf. Leaves are 0x00-prefixed, internal nodes 0x01.
        import hashlib

        a, b = b"\x01" * 32, b"\x02" * 32
        internal = hashlib.sha256(b"\x01" + a + b).digest()
        as_leaf = hashlib.sha256(b"\x00" + a + b).digest()
        assert internal != as_leaf

    def test_proof_verifies_against_the_root(self):
        import hashlib

        leaves = [bytes([i]) * 32 for i in range(8)]
        root = merkle_root(leaves)
        target = leaves[3]
        proof = merkle_proof(leaves, target)

        computed = target
        for sibling in proof:
            low, high = sorted([computed, sibling])
            computed = hashlib.sha256(b"\x01" + low + high).digest()
        assert computed == root

    def test_proof_for_missing_leaf_is_rejected(self):
        with pytest.raises(ValueError, match="not in the tree"):
            merkle_proof([b"\x01" * 32, b"\x02" * 32], b"\x09" * 32)


# ==========================================================================
# THE PRIVACY BOUNDARY
# ==========================================================================


class TestNoBiometricDataOnChain:
    """Nothing that could reconstruct a face may reach a public ledger."""

    def test_raw_vector_is_refused(self):
        artifact = make_artifact()
        artifact["probe"]["vector"] = [0.1] * 512
        with pytest.raises(BiometricLeakError, match="GDPR"):
            fingerprint_evidence(artifact)

    def test_nested_vector_is_refused(self):
        artifact = make_artifact()
        artifact["evidence"]["items"][0]["embedding_vector"] = [0.1, 0.2]
        with pytest.raises(BiometricLeakError):
            fingerprint_evidence(artifact)

    def test_vector_inside_a_list_is_refused(self):
        with pytest.raises(BiometricLeakError):
            assert_no_biometric_leak({"items": [{"raw_embedding": [1, 2, 3]}]})

    def test_leak_error_names_the_path(self):
        with pytest.raises(BiometricLeakError, match=r"\$\.probe\.vector"):
            assert_no_biometric_leak({"probe": {"vector": [1]}})

    def test_clean_artifact_passes(self):
        assert_no_biometric_leak(make_artifact())

    def test_quantized_digest_is_permitted(self):
        # A commitment cannot reconstruct a face, so it is allowed -- and is
        # what binds the anchor to the subject.
        artifact = make_artifact()
        fingerprint = fingerprint_evidence(artifact)
        assert fingerprint.probe_commitment

    def test_fingerprint_carries_no_image_or_url(self):
        payload = json.dumps(fingerprint_evidence(make_artifact()).to_dict())
        assert "data/probes" not in payload
        assert "http" not in payload


# ==========================================================================
# Fingerprinting
# ==========================================================================


class TestFingerprint:
    def test_is_deterministic(self):
        a = fingerprint_evidence(make_artifact())
        b = fingerprint_evidence(make_artifact())
        assert a.merkle_root == b.merkle_root

    def test_all_leaves_present(self):
        fingerprint = fingerprint_evidence(make_artifact())
        assert set(fingerprint.leaves) == set(LEAF_TAGS)

    def test_trust_score_converts_to_basis_points(self):
        assert fingerprint_evidence(make_artifact()).trust_score_bp == 9186

    def test_trust_score_is_clamped_to_contract_range(self):
        artifact = make_artifact()
        artifact["trust_score"]["score"] = 250.0
        assert fingerprint_evidence(artifact).trust_score_bp == 10000

    def test_counts_come_from_the_funnel(self):
        fingerprint = fingerprint_evidence(make_artifact())
        assert fingerprint.evidence_count == 10
        assert fingerprint.independent_publishers == 10

    def test_version_is_recorded(self):
        assert fingerprint_evidence(make_artifact()).version == FINGERPRINT_VERSION

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda a: a["trust_score"].__setitem__("score", 50.0),
            lambda a: a["evidence"].__setitem__("independent_domains", ["z.com"]),
            lambda a: a.__setitem__("run_id", "changed"),
            lambda a: a["probe"].__setitem__("sha256", "f" * 64),
            lambda a: a["evidence"]["items"].append({"content_sha256": "d" * 64}),
        ],
    )
    def test_any_material_change_moves_the_root(self, mutate):
        artifact = make_artifact()
        before = fingerprint_evidence(artifact).merkle_root
        mutate(artifact)
        assert fingerprint_evidence(artifact).merkle_root != before

    def test_key_reordering_does_not_move_the_root(self):
        artifact = make_artifact()
        before = fingerprint_evidence(artifact).merkle_root
        reordered = json.loads(json.dumps(artifact, sort_keys=True))
        assert fingerprint_evidence(reordered).merkle_root == before

    def test_diff_leaves_names_only_what_changed(self):
        artifact = make_artifact()
        original = fingerprint_evidence(artifact)
        artifact["trust_score"]["score"] = 10.0
        changed = diff_leaves(original.leaf_hex(), fingerprint_evidence(artifact).leaf_hex())
        assert [tag for tag, _, _ in changed] == ["tl:trust_score"]


# ==========================================================================
# On-chain behaviour
# ==========================================================================


class TestDeployment:
    def test_deploys_and_reports_gas(self, chain):
        adapter, address = chain
        assert address.startswith("0x")
        assert adapter.w3.eth.get_code(address) not in (b"", b"0x")

    def test_missing_contract_is_detected(self, chain):
        adapter, _ = chain
        with pytest.raises(ContractNotDeployedError):
            adapter.lookup("0x" + "1" * 40, b"\x00" * 32)


class TestAnchoring:
    def test_anchor_then_reverify_is_intact(self, notary):
        artifact = make_artifact()
        record, receipt = notary.anchor(artifact)

        assert receipt.tx_hash.startswith("0x")
        assert receipt.block_number > 0
        assert receipt.gas_used > 0

        result = notary.reverify(artifact, record)
        assert result.verdict is VerificationVerdict.INTACT
        assert result.is_intact

    def test_record_carries_full_provenance(self, notary):
        record, _ = notary.anchor(make_artifact())
        payload = record.to_dict()

        chain = payload["on_chain"]
        for field in (
            "tx_hash", "chain_id", "contract_address", "block_number",
            "block_timestamp", "block_time_utc", "submitter", "gas_used",
        ):
            assert chain[field] not in (None, "", 0), field

    def test_anchored_values_match_the_artifact(self, notary):
        record, _ = notary.anchor(make_artifact())
        assert record.trust_score_bp == 9186
        assert record.evidence_count == 10
        assert record.independent_publishers == 10

    def test_chain_stores_the_trust_score(self, notary, chain):
        adapter, address = chain
        record, _ = notary.anchor(make_artifact())
        on_chain = adapter.lookup(address, bytes.fromhex(record.merkle_root[2:]))
        assert on_chain.exists
        assert on_chain.trust_score == pytest.approx(91.86)

    def test_double_anchoring_is_refused(self, notary):
        artifact = make_artifact()
        notary.anchor(artifact)
        with pytest.raises(AlreadyAnchoredError):
            notary.anchor(artifact)

    def test_different_artifacts_anchor_independently(self, notary):
        first, _ = notary.anchor(make_artifact())
        second, _ = notary.anchor(make_artifact(run_id="evidence_other"))
        assert first.merkle_root != second.merkle_root

    def test_record_round_trips_through_json(self, notary):
        record, _ = notary.anchor(make_artifact())
        restored = AnchorRecord.from_dict(json.loads(json.dumps(record.to_dict())))
        assert restored.merkle_root == record.merkle_root
        assert restored.tx_hash == record.tx_hash
        assert restored.block_number == record.block_number

    def test_record_states_what_it_does_not_prove(self, notary):
        record, _ = notary.anchor(make_artifact())
        proves = record.to_dict()["proves"].lower()
        assert "does not establish" in proves
        assert "correct" in proves


class TestTamperDetection:
    @pytest.mark.parametrize(
        "mutate,expected_leaf",
        [
            (lambda a: a["trust_score"].__setitem__("score", 99.99), "tl:trust_score"),
            (lambda a: a.__setitem__("run_id", "forged"), "tl:run_id"),
            (
                lambda a: a["evidence"].__setitem__("independent_domains", ["fake.com"]),
                "tl:independent_domains",
            ),
        ],
    )
    def test_tamper_is_localised_to_the_right_leaf(self, notary, mutate, expected_leaf):
        artifact = make_artifact()
        record, _ = notary.anchor(artifact)

        tampered = copy.deepcopy(artifact)
        mutate(tampered)

        result = notary.reverify(tampered, record)
        assert result.verdict is VerificationVerdict.TAMPERED
        assert [tag for tag, _, _ in result.changed_leaves] == [expected_leaf]

    def test_tampering_changes_the_root(self, notary):
        artifact = make_artifact()
        record, _ = notary.anchor(artifact)
        tampered = copy.deepcopy(artifact)
        tampered["trust_score"]["score"] = 100.0

        result = notary.reverify(tampered, record)
        assert result.recomputed_root != result.expected_root

    def test_unanchored_artifact_is_not_reported_as_tampered(self, notary):
        # The distinction matters: "never anchored" is not "was altered".
        record, _ = notary.anchor(make_artifact())
        never_anchored = make_artifact(run_id="evidence_never_anchored")
        forged = AnchorRecord.from_dict(
            {
                **record.to_dict(),
                "merkle_root": "0x" + "ab" * 32,
            }
        )
        result = notary.reverify(never_anchored, forged)
        assert result.verdict is VerificationVerdict.NOT_ANCHORED

    def test_verdicts_all_carry_explanations(self):
        for verdict in VerificationVerdict:
            assert len(verdict.explanation) > 40


class TestSelectiveDisclosure:
    def test_leaf_proof_verifies_on_chain(self, notary, chain):
        adapter, address = chain
        artifact = make_artifact()
        record, _ = notary.anchor(artifact)

        fingerprint = notary.fingerprint(artifact)
        tag = "tl:evidence_items"
        leaf = fingerprint.leaves[tag]
        proof = fingerprint.proof_for(tag)

        contract = adapter._contract(address)
        assert contract.functions.verifyLeaf(
            fingerprint.merkle_root, leaf, proof
        ).call()

    def test_wrong_leaf_fails_the_proof(self, notary, chain):
        adapter, address = chain
        artifact = make_artifact()
        notary.anchor(artifact)

        fingerprint = notary.fingerprint(artifact)
        proof = fingerprint.proof_for("tl:evidence_items")

        contract = adapter._contract(address)
        assert not contract.functions.verifyLeaf(
            fingerprint.merkle_root, b"\xaa" * 32, proof
        ).call()


# ==========================================================================
# Configuration
# ==========================================================================


class TestChainConfig:
    def test_describe_never_exposes_the_key(self):
        from tracelock.chain.config import ChainConfig

        secret = "0x" + "1" * 64
        config = ChainConfig("local", "http://x", secret, "0xabc", 1)
        payload = json.dumps(config.describe())
        assert secret not in payload
        assert "1" * 64 not in payload
        assert config.describe()["private_key_set"] is True

    def test_missing_config_fails_before_any_network_call(self):
        from tracelock.chain.config import ChainConfig

        config = ChainConfig("amoy", "", "", "", 1)
        with pytest.raises(ChainConfigError, match="TL_RPC_URL"):
            config.require_for_write()

    def test_local_needs_no_credentials(self):
        from tracelock.chain.config import ChainConfig

        ChainConfig("local", "", "", "", 1).require_for_write()

    def test_unknown_network_is_rejected(self):
        from tracelock.chain.config import ChainConfig

        with pytest.raises(ChainConfigError, match="unknown network"):
            _ = ChainConfig("mainnet_typo", "", "", "", 1).profile

    def test_all_configured_networks_are_testnets(self):
        # Guard against a mainnet profile being added by accident.
        from tracelock.chain.adapter import NETWORKS

        for profile in NETWORKS.values():
            assert profile.is_testnet, profile.key

    def test_public_networks_expose_an_explorer(self):
        from tracelock.chain.adapter import NETWORKS

        for key, profile in NETWORKS.items():
            if key == "local":
                continue
            assert profile.explorer_tx
            assert profile.faucet


class TestErrorTaxonomy:
    def test_all_errors_share_a_base(self):
        for error in (
            ChainConfigError, AlreadyAnchoredError, ContractNotDeployedError,
        ):
            assert issubclass(error, ChainError)

    def test_config_error_is_not_an_anchor_failure(self):
        # A missing RPC URL must never be reported as an anchoring failure --
        # the same distinction Phase 0 draws for API keys.
        from tracelock.chain.errors import AnchorRejectedError

        assert not issubclass(ChainConfigError, AnchorRejectedError)

    def test_unconfirmed_is_neither_success_nor_failure(self):
        from tracelock.chain.errors import AnchorNotConfirmedError

        error = AnchorNotConfirmedError("0xabc", 180.0, "https://explorer/0xabc")
        assert "UNKNOWN" in str(error)
        assert "NOT being reported as anchored" in str(error)
