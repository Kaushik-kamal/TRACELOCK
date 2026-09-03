"""Phase 4 -- turning an evidence artifact into a bytes32 commitment.

    canonical JSON (RFC 8785)  ->  domain-separated leaves  ->  Merkle root

WHY MERKLE AND NOT A SINGLE HASH
--------------------------------
A single digest over the whole bundle would work, and would be simpler. It buys
nothing, and costs two capabilities that matter:

  TAMPER LOCALISATION   recompute every leaf and identify WHICH one broke.
                        "the trust score was altered but the evidence items are
                        intact" is actionable; "verification failed" is not.

  SELECTIVE DISCLOSURE  prove one leaf belonged to the anchored bundle by
                        revealing that leaf plus a short branch, without
                        exposing the probe or the other evidence. For a system
                        handling biometrics that is a real privacy feature.

WHY RFC 8785 AND NOT json.dumps(sort_keys=True)
-----------------------------------------------
The stdlib form is ALMOST canonical and diverges on float representation and
unicode escaping. That divergence surfaces as an intermittent tamper alert on
an untouched file -- the worst possible failure for a notary, because it
destroys trust in true positives.

WHY LEAVES ARE DOMAIN-SEPARATED
-------------------------------
Each leaf is prefixed with its field name before hashing. Without it a value
from one field can be replayed as another field -- a genuine type-confusion
attack on the tree, not a theoretical one.

WHAT IS NEVER COMMITTED
-----------------------
No image bytes, no face embedding, no URL of a private individual. The probe
appears only as the quantized-embedding commitment Phase 1 already produces.
`assert_no_biometric_leak` enforces this before anything is hashed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import jcs

# Bumping this changes every root. It is part of the pipeline fingerprint so an
# old anchor is never silently re-interpreted under new rules.
FINGERPRINT_VERSION = "tracelock-fingerprint/1"

# Domain-separation tags. Order is fixed; leaves are sorted by tag before
# pairing, so the tree is deterministic regardless of dict iteration order.
LEAF_TAGS: tuple[str, ...] = (
    "tl:schema",
    "tl:run_id",
    "tl:probe_commitment",
    "tl:probe_model",
    "tl:evidence_items",
    "tl:evidence_funnel",
    "tl:independent_domains",
    "tl:trust_score",
    "tl:verification_policy",
    "tl:source_artifacts",
    "tl:pipeline",
)

# Keys that must never appear in anything we hash or transmit.
FORBIDDEN_KEYS: frozenset[str] = frozenset({"vector", "embedding_vector", "raw_embedding"})


class BiometricLeakError(Exception):
    """Raised when a raw biometric vector is found in a bundle bound for chain.

    A hard failure by design: silently stripping the field would hide a bug
    that must be fixed upstream.
    """


def assert_no_biometric_leak(payload: Any, path: str = "$") -> None:
    """Walk the artifact and refuse if a raw embedding is present."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in FORBIDDEN_KEYS:
                raise BiometricLeakError(
                    "raw biometric data at {0}.{1}: a face embedding must never "
                    "be committed to a public ledger (GDPR Art. 9 -- it could "
                    "never be withdrawn). Commit the quantized digest "
                    "instead.".format(path, key)
                )
            assert_no_biometric_leak(value, "{0}.{1}".format(path, key))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            assert_no_biometric_leak(value, "{0}[{1}]".format(path, index))


def canonical_bytes(value: Any) -> bytes:
    """RFC 8785 JSON Canonicalization Scheme."""
    return jcs.canonicalize(value)


def leaf_hash(tag: str, value: Any) -> bytes:
    """One domain-separated leaf: SHA-256(tag || 0x00 || canonical(value))."""
    if tag not in LEAF_TAGS:
        raise ValueError("unknown leaf tag {0!r}".format(tag))
    digest = hashlib.sha256()
    digest.update(b"\x00")  # leaf prefix -- see merkle_root
    digest.update(tag.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(canonical_bytes(value))
    return digest.digest()


def merkle_root(leaves: list[bytes]) -> bytes:
    """Sorted-pair Merkle root over pre-hashed leaves.

    Leaf and internal nodes carry different prefixes (0x00 / 0x01). Omitting
    that distinction is the classic second-preimage weakness: without it an
    internal node can be presented as a leaf.

    Sorted pairs make inclusion proofs order-independent, which is what the
    contract's `verifyLeaf` expects.
    """
    if not leaves:
        raise ValueError("cannot build a Merkle root over zero leaves")

    level = sorted(leaves)
    while len(level) > 1:
        nxt: list[bytes] = []
        for index in range(0, len(level), 2):
            if index + 1 == len(level):
                nxt.append(level[index])  # odd node promotes unchanged
                continue
            left, right = level[index], level[index + 1]
            low, high = (left, right) if left <= right else (right, left)
            nxt.append(hashlib.sha256(b"\x01" + low + high).digest())
        level = nxt
    return level[0]


def merkle_proof(leaves: list[bytes], target: bytes) -> list[bytes]:
    """Inclusion branch for `target`. Empty when the tree has a single leaf."""
    level = sorted(leaves)
    if target not in level:
        raise ValueError("target leaf is not in the tree")

    proof: list[bytes] = []
    current = target
    while len(level) > 1:
        nxt: list[bytes] = []
        for index in range(0, len(level), 2):
            if index + 1 == len(level):
                nxt.append(level[index])
                if level[index] == current:
                    pass  # promoted, no sibling this level
                continue
            left, right = level[index], level[index + 1]
            low, high = (left, right) if left <= right else (right, left)
            parent = hashlib.sha256(b"\x01" + low + high).digest()
            if current == left:
                proof.append(right)
                current = parent
            elif current == right:
                proof.append(left)
                current = parent
            nxt.append(parent)
        level = nxt
    return proof


@dataclass(frozen=True, slots=True)
class EvidenceFingerprint:
    """The commitment set derived from one Phase 3 evidence artifact."""

    version: str
    merkle_root: bytes
    leaves: dict[str, bytes]
    run_id: str
    case_id: bytes
    probe_commitment: bytes
    pipeline_hash: bytes
    trust_score_bp: int
    evidence_count: int
    independent_publishers: int

    @property
    def merkle_root_hex(self) -> str:
        return "0x" + self.merkle_root.hex()

    def leaf_hex(self) -> dict[str, str]:
        return {tag: "0x" + value.hex() for tag, value in self.leaves.items()}

    def proof_for(self, tag: str) -> list[bytes]:
        return merkle_proof(list(self.leaves.values()), self.leaves[tag])

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "merkle_root": self.merkle_root_hex,
            "run_id": self.run_id,
            "case_id": "0x" + self.case_id.hex(),
            "probe_commitment": "0x" + self.probe_commitment.hex(),
            "pipeline_hash": "0x" + self.pipeline_hash.hex(),
            "trust_score_bp": self.trust_score_bp,
            "evidence_count": self.evidence_count,
            "independent_publishers": self.independent_publishers,
            "leaves": self.leaf_hex(),
        }


def _sha256_json(value: Any) -> bytes:
    return hashlib.sha256(canonical_bytes(value)).digest()


def fingerprint_evidence(artifact: dict[str, Any]) -> EvidenceFingerprint:
    """Derive the on-chain commitment set from a Phase 3 evidence artifact.

    Pure and deterministic: the same artifact always produces the same root, on
    any machine, which is what makes re-verification meaningful.
    """
    assert_no_biometric_leak(artifact)

    evidence = artifact.get("evidence") or {}
    trust = artifact.get("trust_score") or {}
    probe = artifact.get("probe") or {}
    policy = artifact.get("verification_policy") or {}

    # The probe is committed by its quantized-embedding digest and image hash.
    # Neither can reconstruct a face; both bind the anchor to this subject.
    probe_commitment_source = {
        "image_sha256": probe.get("sha256"),
        "embedding_quantized_sha256": probe.get("embedding_quantized_sha256"),
        "model_id": probe.get("model_id"),
        "embedding_dimension": probe.get("embedding_dimension"),
    }

    pipeline_source = {
        "fingerprint_version": FINGERPRINT_VERSION,
        "artifact_schema": artifact.get("schema_version"),
        "phase": artifact.get("phase"),
        "model_id": probe.get("model_id"),
        "calibration": (policy.get("calibration_model") or {}).get("model_id"),
        "calibrated": policy.get("calibrated"),
    }

    values: dict[str, Any] = {
        "tl:schema": artifact.get("schema_version"),
        "tl:run_id": artifact.get("run_id"),
        "tl:probe_commitment": probe_commitment_source,
        "tl:probe_model": probe.get("model_id"),
        "tl:evidence_items": evidence.get("items", []),
        "tl:evidence_funnel": evidence.get("funnel", {}),
        "tl:independent_domains": evidence.get("independent_domains", []),
        "tl:trust_score": trust,
        "tl:verification_policy": policy,
        "tl:source_artifacts": artifact.get("source_artifacts", {}),
        "tl:pipeline": pipeline_source,
    }

    leaves = {tag: leaf_hash(tag, values[tag]) for tag in LEAF_TAGS}
    root = merkle_root(list(leaves.values()))

    run_id = artifact.get("run_id") or ""
    score = float(trust.get("score") or 0.0)

    return EvidenceFingerprint(
        version=FINGERPRINT_VERSION,
        merkle_root=root,
        leaves=leaves,
        run_id=run_id,
        case_id=hashlib.sha256(run_id.encode("utf-8")).digest(),
        probe_commitment=_sha256_json(probe_commitment_source),
        pipeline_hash=_sha256_json(pipeline_source),
        # Basis points: 91.86 -> 9186. Clamped so the contract's range check
        # can never be tripped by a malformed artifact.
        trust_score_bp=max(0, min(10000, int(round(score * 100)))),
        evidence_count=int((evidence.get("funnel") or {}).get("unique_images") or 0),
        independent_publishers=int(
            (evidence.get("funnel") or {}).get("independent_publishers") or 0
        ),
    )


def diff_leaves(
    stored: dict[str, str], recomputed: dict[str, str]
) -> list[tuple[str, str, str]]:
    """Which leaves changed. This is what makes tamper LOCALISATION possible."""
    changed = []
    for tag in LEAF_TAGS:
        before = stored.get(tag)
        after = recomputed.get(tag)
        if before != after:
            changed.append((tag, before or "(absent)", after or "(absent)"))
    return changed
