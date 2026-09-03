"""Run identity and artifact paths.

A run id is sortable, unique, and traceable back to the probe image:

    search_gate_20260831T184501Z_a3f91c02.json
                |                |
                UTC timestamp    first 8 hex of probe SHA-256

Sortable-by-name means `ls` gives chronological order.  Binding the probe
digest into the filename means a run artifact can never be silently
attributed to the wrong input -- which matters once these feed the evidence
bundle in a later phase.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

_TIMESTAMP_FMT = "%Y%m%dT%H%M%SZ"
_SAFE_PREFIX = re.compile(r"^[a-z0-9_]+$")

# SHA-256 of an empty input, used when no probe is available.
EMPTY_DIGEST = hashlib.sha256(b"").hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Stream a file through SHA-256. Chunked so large media stays out of RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_stamp(when: datetime | None = None) -> str:
    """UTC timestamp in the run-id format. `when` is injectable for tests."""
    moment = when or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        raise ValueError("refusing naive datetime: run ids must be UTC-explicit")
    return moment.astimezone(timezone.utc).strftime(_TIMESTAMP_FMT)


def make_run_id(prefix: str, probe_sha256: str, when: datetime | None = None) -> str:
    """Build a run id. Deterministic given (prefix, digest, when)."""
    if not _SAFE_PREFIX.match(prefix):
        raise ValueError(f"prefix must match [a-z0-9_]+, got {prefix!r}")
    if len(probe_sha256) < 8 or not re.fullmatch(r"[0-9a-f]+", probe_sha256):
        raise ValueError("probe_sha256 must be lowercase hex, at least 8 chars")
    return f"{prefix}_{utc_stamp(when)}_{probe_sha256[:8]}"


def run_artifact_path(runs_dir: Path, run_id: str, suffix: str = ".json") -> Path:
    """Path for a run artifact. Creates the parent directory."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir / f"{run_id}{suffix}"
