"""Demo example images.

WHAT AN EXAMPLE IS, AND WHAT IT IS NOT
--------------------------------------
An example is a pre-selected INPUT image. Everything downstream of it is live:
the reverse-image search queries the real internet, the candidates are
genuinely discovered, every one is independently re-downloaded and re-verified,
and the trust score is computed from those measurements.

Nothing about the RESULT is staged. Selecting an example only saves the
operator from finding an image.

Every example therefore carries `InputKind.DEMO_EXAMPLE`, which propagates into
the run and is displayed in the UI. That marking is structural: the runner
copies it into the result, so a staged input can never be presented as an
organic one.

WHY EXAMPLES NEED A PUBLIC URL
------------------------------
Reverse-image search fetches a URL; it cannot receive a local file. An example
without a `public_url` can be previewed and face-checked but cannot be
investigated, and the registry says so rather than failing later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REGISTRY_PATH = Path("data/examples/registry.json")


@dataclass(frozen=True, slots=True)
class ExampleImage:
    """One bundled demo input."""

    example_id: str
    title: str
    description: str
    local_path: str
    public_url: str
    attribution: str = ""
    # Written by scripts/preflight_examples.py. An example is demo-ready only
    # once it has actually been TESTED -- never because it was declared.
    preflight: dict[str, Any] = field(default_factory=dict)
    recommended: bool = False

    @property
    def exists(self) -> bool:
        return Path(self.local_path).is_file()

    @property
    def is_investigable(self) -> bool:
        """Without a public URL there is nothing to hand a search provider."""
        return bool(self.public_url)

    @property
    def verified(self) -> bool:
        """Has this example actually passed preflight?"""
        return bool(self.preflight.get("passed"))

    @property
    def verification_label(self) -> str:
        if not self.public_url:
            return "LOCAL ANALYSIS ONLY"
        if self.verified:
            return "VERIFIED"
        if self.preflight:
            return "PREFLIGHT FAILED"
        return "NOT YET VERIFIED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "title": self.title,
            "description": self.description,
            "attribution": self.attribution,
            "available": self.exists,
            "investigable": self.is_investigable,
            # Served through the API, never as a filesystem path.
            "preview_url": "/api/examples/{0}/preview".format(self.example_id),
            "kind": "DEMO_EXAMPLE",
            "verified": self.verified,
            "verification_label": self.verification_label,
            "recommended": self.recommended,
            "preflight": {
                "passed": self.preflight.get("passed"),
                "checked_at": self.preflight.get("checked_at"),
                "issue": self.preflight.get("issue"),
                "face_quality": self.preflight.get("face_quality"),
                "note": self.preflight.get("note", ""),
            } if self.preflight else None,
        }


def _default_registry() -> list[dict[str, Any]]:
    """Registry seeded from whatever probes the operator already has.

    Deliberately NOT shipped with downloaded stock faces: bundling strangers'
    photographs to make a demo look fuller is exactly the kind of casual
    biometric collection this project argues against.
    """
    return [
        {
            "example_id": "public-figure",
            "title": "Public figure",
            "description": (
                "A head of government. Widely published, so reverse image "
                "search finds genuine matches across many independent news "
                "domains -- the case where corroboration is meaningful."
            ),
            "local_path": "data/probes/modi.jpg",
            "public_url": "https://raw.githubusercontent.com/Kaushik-kamal/jpg/main/modi.jpg.jpeg",
            "attribution": "Official / press photograph of a public figure.",
            "recommended": True,
        },
        {
            "example_id": "operator-self",
            "title": "Low web presence",
            "description": (
                "A private individual with almost no indexed presence. "
                "Discovery returns visually similar strangers and verification "
                "rejects all of them -- the case that proves the firewall works."
            ),
            "local_path": "data/probes/me.jpg",
            "public_url": "",
            "attribution": "Operator's own photograph, used with consent.",
        },
    ]


def load_registry(path: str | Path = REGISTRY_PATH) -> list[ExampleImage]:
    """Load the example registry, writing the default one on first use."""
    target = Path(path)

    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "schema_version": "examples/1",
                    "notice": (
                        "DEMO INPUT images. The investigations they trigger are "
                        "fully live -- only the starting image is pre-selected. "
                        "Add your own entries here; each needs a public_url for "
                        "discovery to be possible."
                    ),
                    "examples": _default_registry(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    return [
        ExampleImage(
            example_id=row["example_id"],
            title=row.get("title", row["example_id"]),
            description=row.get("description", ""),
            local_path=row.get("local_path", ""),
            public_url=row.get("public_url", ""),
            attribution=row.get("attribution", ""),
            preflight=row.get("preflight", {}) or {},
            recommended=bool(row.get("recommended", False)),
        )
        for row in payload.get("examples", [])
    ]


def find(example_id: str, path: str | Path = REGISTRY_PATH) -> ExampleImage | None:
    return next(
        (e for e in load_registry(path) if e.example_id == example_id), None
    )


def to_trace_input(example: ExampleImage, work_dir: Path):
    """Turn an example into a TraceInput -- the same object every source yields."""
    from tracelock.service.inputs import (
        InputError,
        InputKind,
        SourceType,
        from_bytes,
    )

    source = Path(example.local_path)
    if not source.is_file():
        raise InputError(
            "That example image is not available on this machine.",
            hint="Check data/examples/registry.json.",
        )

    return from_bytes(
        source.read_bytes(),
        source_type=SourceType.EXAMPLE,
        filename=source.name,
        work_dir=work_dir,
        kind=InputKind.DEMO_EXAMPLE,
        image_url=example.public_url or None,
        provenance={
            "origin": "bundled demo example",
            "example_id": example.example_id,
            "attribution": example.attribution,
            "note": (
                "Pre-selected INPUT only. Discovery, verification and scoring "
                "are performed live against the real internet."
            ),
        },
    )
