"""Content-addressed storage.

Blobs are keyed by the SHA-256 of the bytes actually downloaded, never by
filename or URL. Three properties follow, and all three matter downstream:

  DEDUPLICATION IS EXPLICIT
      Two URLs resolving to identical bytes store one blob and BOTH source
      references. The duplicate is reported, not silently merged.

  EVIDENCE SURVIVES LINK ROT
      The post is deleted; the bytes remain. Phase 4 anchors the hash, so the
      chain proves the stored bytes are unchanged since notarisation. This is
      the strongest real-world argument for the whole system.

  RUNS ARE REPRODUCIBLE
      A second run hits cache and produces identical hashes, which is what
      makes deterministic evidence bundles possible at all.

NO DATABASE
-----------
Filesystem plus one JSON sidecar per blob. A database would add an operational
dependency, a schema migration story, and a failure mode, in exchange for
nothing at our scale. Layout:

    data/cas/
        blobs/<aa>/<sha256>          the bytes, verbatim
        refs/<aa>/<sha256>.json      where they came from

The two-character shard keeps directory sizes sane on Windows.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SHARD_LENGTH = 2


@dataclass(frozen=True, slots=True)
class SourceReference:
    """One place a blob was obtained from."""

    requested_url: str
    final_url: str | None = None
    candidate_id: str = ""
    provider: str = ""
    acquired_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_url": self.requested_url,
            "final_url": self.final_url,
            "candidate_id": self.candidate_id,
            "provider": self.provider,
            "acquired_at": self.acquired_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SourceReference":
        return cls(
            requested_url=payload.get("requested_url", ""),
            final_url=payload.get("final_url"),
            candidate_id=payload.get("candidate_id", ""),
            provider=payload.get("provider", ""),
            acquired_at=payload.get("acquired_at", ""),
        )


@dataclass(frozen=True, slots=True)
class CasEntry:
    """Result of storing bytes."""

    sha256: str
    path: Path
    byte_size: int
    was_duplicate: bool
    references: tuple[SourceReference, ...]

    @property
    def reference_count(self) -> int:
        return len(self.references)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "path": str(self.path),
            "byte_size": self.byte_size,
            "was_duplicate": self.was_duplicate,
            "reference_count": self.reference_count,
            "references": [r.to_dict() for r in self.references],
        }


class ContentAddressedStore:
    """Filesystem CAS. Safe to construct repeatedly against the same root."""

    def __init__(self, root: str | Path = "data/cas") -> None:
        self.root = Path(root)
        self.blobs_dir = self.root / "blobs"
        self.refs_dir = self.root / "refs"

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def blob_path(self, sha256: str) -> Path:
        _validate_digest(sha256)
        return self.blobs_dir / sha256[:SHARD_LENGTH] / sha256

    def refs_path(self, sha256: str) -> Path:
        _validate_digest(sha256)
        return self.refs_dir / sha256[:SHARD_LENGTH] / "{0}.json".format(sha256)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def contains(self, sha256: str) -> bool:
        return self.blob_path(sha256).is_file()

    def get(self, sha256: str) -> bytes | None:
        path = self.blob_path(sha256)
        return path.read_bytes() if path.is_file() else None

    def references(self, sha256: str) -> tuple[SourceReference, ...]:
        path = self.refs_path(sha256)
        if not path.is_file():
            return ()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return ()
        return tuple(SourceReference.from_dict(item) for item in payload.get("references", []))

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def put(self, content: bytes, reference: SourceReference | None = None) -> CasEntry:
        """Store bytes, returning whether they were already present.

        `was_duplicate=True` is the exact-duplicate signal the verifier turns
        into a DUPLICATE_CONTENT rejection. Both references are kept either way.
        """
        digest = hashlib.sha256(content).hexdigest()
        blob = self.blob_path(digest)

        was_duplicate = blob.is_file()
        if not was_duplicate:
            blob.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(blob, content)

        existing = list(self.references(digest))
        if reference is not None and not any(
            r.requested_url == reference.requested_url for r in existing
        ):
            existing.append(reference)
            self._write_references(digest, existing)

        return CasEntry(
            sha256=digest,
            path=blob,
            byte_size=len(content),
            was_duplicate=was_duplicate,
            references=tuple(existing),
        )

    def _write_references(self, sha256: str, references: list[SourceReference]) -> None:
        path = self.refs_path(sha256)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sha256": sha256,
            "reference_count": len(references),
            "references": [r.to_dict() for r in references],
        }
        _atomic_write(
            path, json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        blobs = list(self.blobs_dir.rglob("*")) if self.blobs_dir.is_dir() else []
        files = [p for p in blobs if p.is_file()]
        return {
            "root": str(self.root),
            "blob_count": len(files),
            "total_bytes": sum(p.stat().st_size for p in files),
        }

    def verify_integrity(self, sha256: str) -> bool:
        """Re-hash a stored blob and confirm it still matches its address.

        Detects on-disk corruption or tampering. Phase 4's re-verification path
        depends on this holding.
        """
        content = self.get(sha256)
        if content is None:
            return False
        return hashlib.sha256(content).hexdigest() == sha256


# --------------------------------------------------------------------------


def _validate_digest(sha256: str) -> None:
    if len(sha256) != 64 or not all(c in "0123456789abcdef" for c in sha256):
        raise ValueError(
            "not a lowercase hex SHA-256 digest: {0!r}".format(sha256[:80])
        )


def _atomic_write(path: Path, data: bytes) -> None:
    """Write via a temp file and rename, so a crash cannot leave a partial blob.

    A truncated file sitting at a content address would be corruption that
    looks like valid evidence -- the one failure mode a CAS must not have.
    """
    handle, temp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    temp_path = Path(temp_name)
    try:
        with open(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
        temp_path.replace(path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
