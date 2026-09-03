"""Run identity.

Run ids must be sortable, unique, and traceable to the probe. Once these feed
the evidence bundle, a collision or a mis-attribution becomes a chain-of-
custody defect rather than a cosmetic bug.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest

from tracelock.core.runs import (
    make_run_id,
    run_artifact_path,
    sha256_file,
    utc_stamp,
)

FIXED = datetime(2026, 8, 31, 18, 45, 1, tzinfo=timezone.utc)
DIGEST = "a3f91c02" + "0" * 56


class TestUtcStamp:
    def test_format(self):
        assert utc_stamp(FIXED) == "20260831T184501Z"

    def test_converts_other_zones_to_utc(self):
        from datetime import timedelta

        plus_two = FIXED.astimezone(timezone(timedelta(hours=2)))
        assert utc_stamp(plus_two) == "20260831T184501Z"

    def test_rejects_naive_datetime(self):
        # A naive datetime would silently record local time as UTC.
        with pytest.raises(ValueError, match="naive"):
            utc_stamp(datetime(2026, 8, 31, 18, 45, 1))


class TestMakeRunId:
    def test_expected_shape(self):
        assert make_run_id("search_gate", DIGEST, FIXED) == (
            "search_gate_20260831T184501Z_a3f91c02"
        )

    def test_deterministic(self):
        assert make_run_id("search_gate", DIGEST, FIXED) == make_run_id(
            "search_gate", DIGEST, FIXED
        )

    def test_lexicographic_order_matches_chronological(self):
        from datetime import timedelta

        earlier = make_run_id("g", DIGEST, FIXED)
        later = make_run_id("g", DIGEST, FIXED + timedelta(seconds=1))
        assert earlier < later

    def test_different_probes_differ(self):
        other = "ffffffff" + "0" * 56
        assert make_run_id("g", DIGEST, FIXED) != make_run_id("g", other, FIXED)

    def test_uses_only_filename_safe_characters(self):
        run_id = make_run_id("search_gate", DIGEST, FIXED)
        assert all(char.isalnum() or char == "_" for char in run_id)

    @pytest.mark.parametrize("bad", ["Search Gate", "search-gate", "search/gate", ""])
    def test_rejects_unsafe_prefix(self, bad):
        with pytest.raises(ValueError, match="prefix"):
            make_run_id(bad, DIGEST, FIXED)

    @pytest.mark.parametrize("bad", ["short", "NOTLOWER" + "0" * 56, "zz" * 32, ""])
    def test_rejects_bad_digest(self, bad):
        with pytest.raises(ValueError, match="probe_sha256"):
            make_run_id("g", bad, FIXED)


class TestSha256File:
    def test_matches_hashlib(self, tmp_path):
        target = tmp_path / "probe.jpg"
        payload = b"not really a jpeg"
        target.write_bytes(payload)
        assert sha256_file(target) == hashlib.sha256(payload).hexdigest()

    def test_chunking_does_not_change_digest(self, tmp_path):
        target = tmp_path / "big.bin"
        payload = b"x" * 100_000
        target.write_bytes(payload)
        assert sha256_file(target, chunk_size=1024) == hashlib.sha256(payload).hexdigest()

    def test_empty_file(self, tmp_path):
        target = tmp_path / "empty.bin"
        target.write_bytes(b"")
        assert sha256_file(target) == hashlib.sha256(b"").hexdigest()


class TestRunArtifactPath:
    def test_creates_missing_parent(self, tmp_path):
        runs_dir = tmp_path / "data" / "runs"
        path = run_artifact_path(runs_dir, "search_gate_20260831T184501Z_a3f91c02")
        assert runs_dir.is_dir()
        assert path.name == "search_gate_20260831T184501Z_a3f91c02.json"

    def test_idempotent_on_existing_dir(self, tmp_path):
        runs_dir = tmp_path / "runs"
        run_artifact_path(runs_dir, "a1b2c3d4")
        assert run_artifact_path(runs_dir, "a1b2c3d4").parent == runs_dir

    def test_custom_suffix(self, tmp_path):
        path = run_artifact_path(tmp_path, "run_1", suffix=".raw.json")
        assert path.name == "run_1.raw.json"
