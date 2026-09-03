"""What the product CLAIMS must match what it can actually do.

The distinction this file exists to defend:

    BLOCKCHAIN INTEGRITY  !=  IMAGE TRUTH

An anchor commits to a bundle of measurements at a point in time. It proves
that bundle has not changed since. It says nothing whatsoever about whether the
photograph depicts what someone claims it depicts, whether the person in it is
who they say, or whether the sources that published it were honest.

Blurring those two would be the most damaging thing this product could do,
because "on the blockchain" reads to a non-specialist as "therefore true".
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# Claims no evidence-integrity system can support. Each is a regex because the
# wording varies; the underlying overreach does not.
FORBIDDEN_CLAIMS = [
    (r"blockchain\s+proves?\s+(?:this\s+)?(?:image|photo|identity|person)",
     "claims the blockchain proves something about the image"),
    (r"prov(?:es|en)\s+authentic", "claims proven authenticity"),
    (r"image\s+is\s+authentic", "claims the image is authentic"),
    (r"cannot\s+be\s+faked", "claims something cannot be faked"),
    (r"100%\s+(?:accurate|certain|proof)", "claims absolute certainty"),
    (r"immutable\s+proof\s+of\s+truth", "conflates immutability with truth"),
    (r"guarantees?\s+(?:the\s+)?(?:image|photo)\s+is", "guarantees the image"),
    (r"confirms?\s+(?:the\s+)?(?:person|identity)\s+is",
     "claims confirmed identity"),
]


def sources() -> list[Path]:
    paths = sorted(Path("src").rglob("*.py"))
    paths += [Path("web/app.js"), Path("web/index.html")]
    return [p for p in paths if p.is_file()]


class TestNoOverstatedClaims:
    def test_no_overstated_claim_anywhere_in_the_product(self):
        offences = []
        for path in sources():
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern, label in FORBIDDEN_CLAIMS:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    line = text[: match.start()].count("\n") + 1
                    offences.append("{0}:{1} -- {2}".format(path, line, label))
        assert not offences, offences

    def test_the_integrity_not_truth_caveat_is_visible(self):
        flat = " ".join(Path("web/app.js").read_text(encoding="utf-8").split())
        assert "integrity</strong>, not truth" in flat
        assert "cannot prove the original source" in flat

    def test_the_caveat_says_what_an_anchor_CAN_do_too(self):
        """A caveat that only denies is less useful than one that also scopes."""
        flat = " ".join(Path("web/app.js").read_text(encoding="utf-8").split())
        assert "can prove that evidence changed" in flat.lower()


class TestStrongClaimsAreGuarded:
    def test_publicly_verifiable_only_renders_for_a_non_ephemeral_chain(self):
        js = Path("web/app.js").read_text(encoding="utf-8")
        found = list(re.finditer(r'"PUBLICLY VERIFIABLE"', js))
        assert found, "the public badge is missing entirely"

        for match in found:
            line_start = js.rfind("\n", 0, match.start())
            assert "ephemeral ?" in js[line_start : match.end()], (
                "PUBLICLY VERIFIABLE rendered without an ephemeral guard"
            )

    def test_the_ui_never_calls_anything_immutable(self):
        """Scoped to what a PERSON reads, not to internal prose.

        `face/models.py` legitimately uses the word in a docstring explaining
        why embeddings must never be anchored -- "on a public immutable ledger
        it can never be withdrawn" is a true statement about ledgers and the
        reason for a safety constraint. What must not happen is telling a USER
        their evidence is immutable, which overstates what an anchor does.
        """
        for path in (Path("web/app.js"), Path("web/index.html")):
            text = path.read_text(encoding="utf-8").lower()
            assert "immutable" not in text, path

    def test_verified_always_refers_to_a_candidate_never_to_truth(self):
        """`VERIFIED_CANDIDATE` is a measurement outcome, not a truth claim."""
        from tracelock.core.reasons import VerificationStatus

        assert VerificationStatus.VERIFIED_CANDIDATE.value == "VERIFIED_CANDIDATE"
        # There is deliberately no status meaning "true" or "authentic".
        values = {s.value.lower() for s in VerificationStatus}
        for banned in ("authentic", "true", "genuine", "proven"):
            assert not any(banned in v for v in values), banned


class TestFailureWordingIsHonest:
    def test_a_platform_refusal_never_becomes_nonexistence(self):
        from tracelock.service import failures

        message = failures.platform_blocked("Instagram", "requires sign-in").message
        lowered = message.lower()
        assert "could not retrieve" in lowered
        for forbidden in ("does not exist", "no image exists", "there is no"):
            assert forbidden not in lowered

    def test_an_empty_discovery_result_is_not_absence_of_evidence(self):
        """'Found nothing' and 'nothing exists' are different claims."""
        source = Path("src/tracelock/service/runner.py").read_text(encoding="utf-8")
        assert "does NOT prove the subject has no web presence" in source

    @pytest.mark.parametrize(
        "raw",
        [
            "connection refused",
            "invalid private key",
            "insufficient funds",
            "execution reverted",
        ],
    )
    def test_every_public_failure_denies_making_a_record(self, raw):
        from tracelock.chain.preflight import classify_anchor_failure

        _status, message, _detail = classify_anchor_failure(Exception(raw))
        assert "no public verification record was created" in message.lower()
