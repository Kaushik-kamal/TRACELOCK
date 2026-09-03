"""Compile EvidenceNotary.sol.

Uses py-solc-x, which downloads a pinned solc binary on demand. Deliberately
not Foundry or Hardhat: both need a separate toolchain (Rust or Node) that the
rest of this project does not, and a hackathon build should not require one.

The compiled ABI and bytecode are cached to disk so tests and the CLI do not
re-invoke solc, and so the exact artifact that was deployed can be inspected.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tracelock.chain.errors import CompilationError

SOLC_VERSION = "0.8.24"
CONTRACT_NAME = "EvidenceNotary"

DEFAULT_SOURCE = Path("contracts/EvidenceNotary.sol")
DEFAULT_BUILD = Path("contracts/build/EvidenceNotary.json")


@dataclass(frozen=True, slots=True)
class CompiledContract:
    name: str
    abi: list[dict[str, Any]]
    bytecode: str
    solc_version: str
    source_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "abi": self.abi,
            "bytecode": self.bytecode,
            "solc_version": self.solc_version,
            "source_sha256": self.source_sha256,
        }

    def save(self, path: str | Path = DEFAULT_BUILD) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path = DEFAULT_BUILD) -> "CompiledContract":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=payload["name"],
            abi=payload["abi"],
            bytecode=payload["bytecode"],
            solc_version=payload["solc_version"],
            source_sha256=payload["source_sha256"],
        )


def _source_digest(source: str) -> str:
    import hashlib

    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def compile_contract(
    source_path: str | Path = DEFAULT_SOURCE, *, optimize: bool = True
) -> CompiledContract:
    """Compile the notary contract, installing solc if it is not present."""
    path = Path(source_path)
    if not path.is_file():
        raise CompilationError("contract source not found: {0}".format(path))

    source = path.read_text(encoding="utf-8")

    try:
        import solcx
    except ImportError as exc:
        raise CompilationError(
            "py-solc-x is not installed. Run: pip install py-solc-x"
        ) from exc

    try:
        installed = {str(v) for v in solcx.get_installed_solc_versions()}
        if SOLC_VERSION not in installed:
            solcx.install_solc(SOLC_VERSION)

        compiled = solcx.compile_source(
            source,
            output_values=["abi", "bin"],
            solc_version=SOLC_VERSION,
            optimize=optimize,
            optimize_runs=200,
        )
    except Exception as exc:  # solcx raises a family of its own errors
        raise CompilationError("solc {0} failed: {1}".format(SOLC_VERSION, exc)) from exc

    key = next((k for k in compiled if k.endswith(":" + CONTRACT_NAME)), None)
    if key is None:
        raise CompilationError(
            "{0} not found in compiler output; got {1}".format(
                CONTRACT_NAME, list(compiled)
            )
        )

    entry = compiled[key]
    bytecode = entry["bin"]
    if not bytecode:
        raise CompilationError("compiler produced empty bytecode")

    return CompiledContract(
        name=CONTRACT_NAME,
        abi=entry["abi"],
        bytecode=bytecode if bytecode.startswith("0x") else "0x" + bytecode,
        solc_version=SOLC_VERSION,
        source_sha256=_source_digest(source),
    )


def load_or_compile(
    source_path: str | Path = DEFAULT_SOURCE,
    build_path: str | Path = DEFAULT_BUILD,
) -> CompiledContract:
    """Return the cached build if it matches the current source, else compile.

    The source digest is the cache key. An edited contract always recompiles,
    so a stale ABI can never be deployed against changed source.
    """
    path = Path(source_path)
    build = Path(build_path)

    if build.is_file() and path.is_file():
        try:
            cached = CompiledContract.load(build)
            if cached.source_sha256 == _source_digest(path.read_text(encoding="utf-8")):
                return cached
        except (json.JSONDecodeError, KeyError, OSError):
            pass  # unreadable cache is simply a cache miss

    compiled = compile_contract(path)
    compiled.save(build)
    return compiled
