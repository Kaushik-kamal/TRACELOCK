// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title EvidenceNotary
/// @notice Tamper-evident anchoring of TRACELOCK evidence bundles.
///
/// WHAT GOES ON CHAIN
/// ------------------
/// A 32-byte Merkle root and a little packed metadata. Nothing else. No
/// images, no face embeddings, no URLs, no personal data of any kind.
///
/// This is not a cost optimisation, it is a hard requirement. A face embedding
/// is biometric data under GDPR Art. 9; on a public immutable ledger it could
/// never be withdrawn. The client commits to a quantized digest instead, and
/// the raw vector never leaves the operator's machine.
///
/// WHAT AN ANCHOR PROVES
/// ---------------------
/// That this exact evidence bundle existed, unchanged, at or before the block
/// timestamp, and was submitted by this address. It proves nothing about
/// whether the evidence is CORRECT -- a chain cannot make a lying source
/// honest. That boundary is stated in the client and in the report.
///
/// NO OWNER, NO UPGRADE, NO PAUSE
/// ------------------------------
/// Deliberate. An immutable notary with no admin is the correct trust model:
/// access control would let someone with a key rewrite history, which is
/// precisely what a notary must not permit.
contract EvidenceNotary {
    /// @dev Packed into ONE 256-bit storage slot:
    ///      40 + 160 + 16 + 32 = 248 bits.
    ///      uint40 for the timestamp is what makes it fit; it overflows in
    ///      the year 36812, which is an acceptable horizon.
    struct Anchor {
        uint40  timestamp;      // block time when anchored
        address submitter;      // 160 bits
        uint16  trustScoreBp;   // trust score in basis points, 0..10000
        uint32  evidenceCount;  // unique verified images in the bundle
    }

    /// @notice merkleRoot => Anchor. Presence of a non-zero timestamp means anchored.
    mapping(bytes32 => Anchor) public anchors;

    /// @notice Total anchors recorded. Cheap read for demos and health checks.
    uint256 public anchorCount;

    /// @dev Wide fields live in the event, not storage: logs cost roughly an
    ///      order of magnitude less gas and indexers read them for free.
    event EvidenceAnchored(
        bytes32 indexed merkleRoot,
        bytes32 indexed caseId,
        address indexed submitter,
        bytes32 probeCommitment,
        bytes32 pipelineHash,
        uint16  trustScoreBp,
        uint32  evidenceCount,
        uint32  independentPublishers,
        uint40  timestamp
    );

    error AlreadyAnchored(bytes32 merkleRoot);
    error EmptyRoot();
    error TrustScoreOutOfRange(uint16 trustScoreBp);

    /// @notice Anchor an evidence bundle.
    /// @param merkleRoot Root over the canonical evidence bundle.
    /// @param caseId Opaque case identifier (keccak of the run id).
    /// @param probeCommitment Commitment to the probe image, NOT the image.
    /// @param pipelineHash Fingerprint of code + model versions that produced it.
    /// @param trustScoreBp Trust score in basis points (9186 == 91.86).
    /// @param evidenceCount Unique verified images after de-duplication.
    /// @param independentPublishers Distinct eTLD+1 publishers corroborating.
    function anchor(
        bytes32 merkleRoot,
        bytes32 caseId,
        bytes32 probeCommitment,
        bytes32 pipelineHash,
        uint16  trustScoreBp,
        uint32  evidenceCount,
        uint32  independentPublishers
    ) external {
        if (merkleRoot == bytes32(0)) revert EmptyRoot();
        if (anchors[merkleRoot].timestamp != 0) revert AlreadyAnchored(merkleRoot);
        if (trustScoreBp > 10000) revert TrustScoreOutOfRange(trustScoreBp);

        uint40 nowTs = uint40(block.timestamp);

        anchors[merkleRoot] = Anchor({
            timestamp:     nowTs,
            submitter:     msg.sender,
            trustScoreBp:  trustScoreBp,
            evidenceCount: evidenceCount
        });

        unchecked { ++anchorCount; }

        emit EvidenceAnchored(
            merkleRoot,
            caseId,
            msg.sender,
            probeCommitment,
            pipelineHash,
            trustScoreBp,
            evidenceCount,
            independentPublishers,
            nowTs
        );
    }

    /// @notice Look up an anchor.
    /// @return exists True when this root has been anchored.
    /// @return record The stored anchor (zeroed when `exists` is false).
    function verify(bytes32 merkleRoot)
        external
        view
        returns (bool exists, Anchor memory record)
    {
        record = anchors[merkleRoot];
        exists = record.timestamp != 0;
    }

    /// @notice Prove one leaf belongs to an anchored bundle, without revealing
    ///         the rest of it. This is what makes selective disclosure possible:
    ///         show that a specific candidate image was part of the notarised
    ///         evidence while withholding the probe and everything else.
    /// @dev Sorted-pair construction, so proofs are order independent.
    ///
    ///      SHA-256, NOT keccak256. The client builds the tree with SHA-256 to
    ///      stay consistent with the rest of TRACELOCK -- the content-addressed
    ///      store, the probe commitment and every leaf use it. Using keccak
    ///      here (the Solidity default, and the obvious thing to reach for)
    ///      would mean no client-generated proof could ever verify on chain.
    ///
    ///      Internal nodes are 0x01-prefixed to match the client and to keep
    ///      leaves and internal nodes in separate domains, which is what
    ///      defeats second-preimage attacks on the tree.
    function verifyLeaf(
        bytes32 merkleRoot,
        bytes32 leaf,
        bytes32[] calldata proof
    ) external view returns (bool) {
        if (anchors[merkleRoot].timestamp == 0) return false;

        bytes32 computed = leaf;
        for (uint256 i = 0; i < proof.length; ++i) {
            bytes32 sibling = proof[i];
            (bytes32 low, bytes32 high) = computed <= sibling
                ? (computed, sibling)
                : (sibling, computed);
            computed = sha256(abi.encodePacked(bytes1(0x01), low, high));
        }
        return computed == merkleRoot;
    }

    /// @notice Convenience read used by the CLI's re-verification path.
    function isAnchored(bytes32 merkleRoot) external view returns (bool) {
        return anchors[merkleRoot].timestamp != 0;
    }
}
