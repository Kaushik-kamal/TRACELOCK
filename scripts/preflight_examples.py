"""Validate every demo example before it is trusted in a demo.

    python scripts\\preflight_examples.py            # check all
    python scripts\\preflight_examples.py --add ID --url URL --title "…"
    python scripts\\preflight_examples.py --stamp    # write results into the registry

WHY THIS EXISTS
---------------
An example that fails live is the worst thing that can happen in a demo, and
the failure modes are invisible until you try: a host that 403s automated
clients, an image with no usable face, a link that has rotted.

So examples are not asserted to work -- they are TESTED, and the result is
stamped into the registry with a timestamp. The UI shows what was verified and
when, and refuses to present an unverified example as demo-ready.

WHAT IS CHECKED, IN ORDER
-------------------------
  1. URL reachable          the same hardened fetcher candidates go through
  2. Image decodable        magic bytes, not the file extension
  3. Face usable            Stage 2.5 -- present AND large/clean enough to seed
                            a search (an icon-scale detection is not usable)

WHAT IS DELIBERATELY NOT CHECKED
--------------------------------
Whether discovery will return good results. That depends on what a third-party
search index contains today, is outside our control, and asserting it would be
a promise this tool cannot keep. Preflight verifies the INPUT is sound; it does
not guarantee the OUTPUT.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tracelock.service.examples import REGISTRY_PATH, load_registry  # noqa: E402

RULE = "=" * 78
THIN = "-" * 78


def emit(text: str = "") -> None:
    print(text, flush=True)


def header(text: str) -> None:
    emit()
    emit(RULE)
    emit(text)
    emit(RULE)


def check_one(example, engine) -> dict:
    """Run the full preflight on one example."""
    from tracelock.discovery.search_image import check_search_image

    result = {
        "example_id": example.example_id,
        "title": example.title,
        "local_file_present": example.exists,
        "has_public_url": bool(example.public_url),
        "url_reachable": None,
        "image_decodable": None,
        "face_usable": None,
        "passed": False,
        "issue": None,
        "detail": "",
        "image": None,
        "face": None,
        "warnings": [],
    }

    if not example.public_url:
        result["issue"] = "NO_PUBLIC_URL"
        result["detail"] = (
            "No public URL. This example can be previewed and analysed locally "
            "but cannot be used for public discovery."
        )
        return result

    check = check_search_image(example.public_url, engine)

    issue = check.issue.value if check.issue else None
    result["issue"] = issue
    result["detail"] = check.detail
    result["url_reachable"] = issue != "URL_UNFETCHABLE"
    result["image_decodable"] = issue not in ("URL_UNFETCHABLE", "NOT_AN_IMAGE")
    result["face_usable"] = check.ok
    result["passed"] = check.ok
    result["warnings"] = [w.value for w in check.warnings]

    if check.image_format:
        result["image"] = {
            "format": check.image_format,
            "width": check.width,
            "height": check.height,
            "byte_size": check.byte_size,
            "sha256": check.content_sha256,
        }
    if check.faces_detected is not None:
        result["face"] = {
            "count": check.faces_detected,
            "det_score": check.det_score,
            "quality": check.quality_aggregate,
            "quality_band": check.quality_band,
            "min_side_px": check.face_min_side_px,
        }
    return result


def render(result: dict) -> None:
    mark = "PASS" if result["passed"] else "FAIL"
    emit()
    emit("  [{0}]  {1}  --  {2}".format(mark, result["example_id"], result["title"]))

    def line(label, state, extra=""):
        icon = {True: "OK  ", False: "FAIL", None: "--  "}[state]
        emit("        {0}  {1:<22} {2}".format(icon, label, extra))

    line("local file present", result["local_file_present"])
    line("public URL declared", result["has_public_url"])
    line("URL reachable", result["url_reachable"])
    line("image decodable", result["image_decodable"])

    face = result["face"] or {}
    line(
        "face usable",
        result["face_usable"],
        "{0} face(s), quality {1:.2f} [{2}], {3:.0f}px".format(
            face.get("count", 0),
            face.get("quality") or 0.0,
            face.get("quality_band") or "-",
            face.get("min_side_px") or 0.0,
        )
        if face
        else "",
    )

    image = result["image"] or {}
    if image:
        emit("        info  {0} {1}x{2}, {3} bytes".format(
            image["format"], image["width"], image["height"], image["byte_size"]))

    if result["warnings"]:
        emit("        warn  {0}".format(", ".join(result["warnings"])))
    if not result["passed"] and result["detail"]:
        emit("        why   {0}".format(result["detail"][:150]))


def stamp_registry(results: list[dict], path: Path) -> None:
    """Write verification status back into the registry.

    The UI reads this: an example that has not passed preflight is shown as
    unverified rather than presented as demo-ready.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_id = {r["example_id"]: r for r in results}
    checked_at = datetime.now(timezone.utc).isoformat()

    for entry in payload.get("examples", []):
        result = by_id.get(entry.get("example_id"))
        if result is None:
            continue
        entry["preflight"] = {
            "passed": result["passed"],
            "checked_at": checked_at,
            "issue": result["issue"],
            "face_quality": (result["face"] or {}).get("quality"),
            "image": result["image"],
            "note": (
                "Input verified: URL reachable, image decodable, face usable. "
                "Discovery RESULTS are not guaranteed -- they depend on a "
                "third-party index."
                if result["passed"]
                else result["detail"][:200]
            ),
        }

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def add_example(args, path: Path) -> int:
    """Append a new example to the registry, then preflight it."""
    payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
        "schema_version": "examples/1", "examples": []
    }

    if any(e.get("example_id") == args.add for e in payload["examples"]):
        emit("An example with id {0!r} already exists.".format(args.add))
        return 2

    payload["examples"].append({
        "example_id": args.add,
        "title": args.title or args.add,
        "description": args.description or "",
        "local_path": args.local_path or "",
        "public_url": args.url or "",
        "attribution": args.attribution or "",
    })
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    emit("Added {0!r} to {1}. Running preflight ...".format(args.add, path))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="preflight_examples")
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--stamp", action="store_true",
                        help="Write results back into the registry.")
    parser.add_argument("--add", metavar="ID", help="Add a new example, then check it.")
    parser.add_argument("--url", help="Public image URL for --add.")
    parser.add_argument("--title", help="Card title for --add.")
    parser.add_argument("--description", help="Card description for --add.")
    parser.add_argument("--local-path", help="Local copy for --add.")
    parser.add_argument("--attribution", help="Provenance note for --add.")
    args = parser.parse_args(argv)

    header("TRACELOCK - DEMO EXAMPLE PREFLIGHT")
    emit("Examples are TESTED, not asserted. Results are stamped into the registry.")

    if args.add:
        if add_example(args, args.registry) != 0:
            return 2

    examples = load_registry(args.registry)
    if not examples:
        emit("\nNo examples registered.")
        return 2

    emit()
    emit("  registry : {0}".format(args.registry))
    emit("  examples : {0}".format(len(examples)))

    emit()
    emit("Loading face engine ...")
    from tracelock.face import FaceEngine

    engine = FaceEngine()

    header("CHECKS")
    results = [check_one(example, engine) for example in examples]
    for result in results:
        render(result)

    passed = [r for r in results if r["passed"]]
    local_only = [r for r in results if not r["has_public_url"]]
    failed = [r for r in results if not r["passed"] and r["has_public_url"]]

    header("SUMMARY")
    emit("  demo-ready (public discovery) : {0}".format(len(passed)))
    emit("  local-analysis only           : {0}".format(len(local_only)))
    emit("  FAILING                       : {0}".format(len(failed)))

    for result in failed:
        emit("      {0}: {1}".format(result["example_id"], result["issue"]))

    if args.stamp or args.add:
        stamp_registry(results, args.registry)
        emit()
        emit("  Registry stamped with verification status and timestamp.")

    emit()
    emit("  NOTE: preflight verifies the INPUT is sound. It does not guarantee")
    emit("  what discovery will return -- that depends on a third-party index")
    emit("  and is outside our control.")
    emit()

    if failed:
        emit("  Fix or remove the failing examples before demonstrating.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
