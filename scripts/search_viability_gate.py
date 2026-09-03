"""DAY ZERO VIABILITY GATE.

Answers exactly one question:

    Can we dynamically discover candidate web content from an input image
    in our current environment?

Nothing else. No face recognition, no blockchain, no scoring. If this fails,
the TRACELOCK architecture needs rethinking before anything else is built.

HONESTY CONSTRAINTS (these are the point of the script)
------------------------------------------------------
  * No hardcoded expected URL. The script has no idea what it should find.
  * No mocked or cached results. `no_cache=true` is sent on every query.
  * No pretending. Zero candidates is reported as FAIL, loudly.
  * The raw provider response is always persisted, pass or fail.

EXIT CODES -- the distinction matters
-------------------------------------
  0  PASS                 dynamic discovery works here
  1  ARCHITECTURE RISK    provider blocked, or no usable candidates
  2  SETUP ERROR          missing key / bad config -- NOT a viability verdict
  3  INPUT ERROR          bad image path or unreadable image

Conflating 1 and 2 would produce a false negative on the single most important
decision in the project, so they are kept strictly separate.

USAGE
-----
    # Preferred: the probe is already public (no new disclosure)
    python scripts/search_viability_gate.py --image data/probes/me.jpg \
        --image-url https://example.com/my-public-photo.jpg

    # Fallback: host it temporarily (explicit opt-in, 1h retention)
    python scripts/search_viability_gate.py --image data/probes/me.jpg \
        --allow-upload
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make `src/` importable when run directly, before the package is installed.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from PIL import Image, UnidentifiedImageError  # noqa: E402

from tracelock.core.config import load_settings  # noqa: E402
from tracelock.core.models import ProbeRef  # noqa: E402
from tracelock.core.runs import (  # noqa: E402
    make_run_id,
    run_artifact_path,
    sha256_file,
)
from tracelock.discovery.base import (  # noqa: E402
    DiscoveryError,
    ProviderBlockedError,
    ProviderConfigError,
    ProviderResult,
    ProviderSchemaError,
    ProviderTransportError,
    check_requirements,
)
from tracelock.discovery.hosting import HostingError, upload_temporary  # noqa: E402
from tracelock.discovery.serpapi_lens import SerpApiProvider  # noqa: E402

# Windows consoles default to cp1252 and cannot encode characters that appear
# in real provider titles and URLs. A UnicodeEncodeError while RENDERING would
# abort the run and discard the artifact -- a display problem destroying
# evidence. Same fix as verify_candidates.py; it belonged in both from the
# start.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EXIT_PASS = 0
EXIT_ARCHITECTURE_RISK = 1
EXIT_SETUP_ERROR = 2
EXIT_INPUT_ERROR = 3

RULE = "=" * 72
THIN = "-" * 72


def emit(text: str = "") -> None:
    """Print, degrading rather than aborting. A rendering failure must never
    destroy a completed run."""
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", "ascii") or "ascii"
        print(text.encode(encoding, errors="replace").decode(encoding), flush=True)


def banner(text: str) -> None:
    emit()
    emit(RULE)
    emit(text)
    emit(RULE)


def stage(number: int, text: str) -> None:
    emit()
    emit("[{0}] {1}".format(number, text))
    emit(THIN)


def fail(exit_code: int, headline: str, detail: str, remedy: str = "") -> int:
    """Print a failure block and return the exit code."""
    banner(headline)
    emit(detail)
    if remedy:
        emit()
        emit("WHAT TO DO:")
        emit("  " + remedy.replace("\n", "\n  "))
    emit()
    return exit_code


def validate_probe(path: Path) -> tuple[str, dict[str, Any]]:
    """Confirm the file exists and decodes. Cheap check before spending a call."""
    if not path.is_file():
        raise FileNotFoundError("no such file: {0}".format(path))

    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            info = {
                "format": img.format,
                "mode": img.mode,
                "width": img.width,
                "height": img.height,
            }
    except UnidentifiedImageError as exc:
        raise ValueError("not a decodable image: {0}".format(path)) from exc

    return sha256_file(path), info


def write_artifact(
    path: Path,
    *,
    run_id: str,
    probe: ProbeRef,
    probe_info: dict[str, Any],
    result: ProviderResult | None,
    verdict: str,
    failure: dict[str, Any] | None = None,
) -> None:
    """Persist the run. Written on failure too -- a failed run is evidence."""
    document: dict[str, Any] = {
        "run_id": run_id,
        "phase": "0-search-viability-gate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "probe": {
            "local_path": probe.local_path,
            "sha256": probe.sha256,
            "public_url": probe.public_url,
            **probe_info,
        },
        "failure": failure,
    }

    if result is not None:
        document["provider"] = {
            "name": result.provider,
            "engine": result.engine,
            "elapsed_seconds": result.elapsed_seconds,
            "query": result.query,
        }
        document["counts"] = {
            "candidates": result.count,
            "with_media": len(result.with_media),
        }
        document["candidates"] = [
            candidate.model_dump(mode="json") for candidate in result.candidates
        ]
        # Verbatim provider payload. The whole point of the gate is that this
        # was not written by us.
        document["raw_response"] = result.raw_response

    path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")


def print_candidates(result: ProviderResult, show: int) -> None:
    for candidate in result.candidates[:show]:
        emit()
        emit("  #{0:<3} {1}".format(candidate.rank or "?", candidate.source or "(no source)"))
        emit("       post : {0}".format(candidate.post_url or "-"))
        emit("       image: {0}".format(candidate.image_url or "-"))
        emit("       thumb: {0}".format(candidate.thumbnail_url or "-"))
        if candidate.title:
            emit("       title: {0}".format(candidate.title[:80]))

    remaining = result.count - show
    if remaining > 0:
        emit()
        emit("  ... {0} more (full list in the run artifact)".format(remaining))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="search_viability_gate",
        description="Day-zero gate: prove dynamic discovery works in this environment.",
    )
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Local probe image. Always required (hashed for run identity).",
    )
    parser.add_argument(
        "--image-url",
        default=None,
        help="PREFERRED. Public URL of this same image. Skips uploading entirely.",
    )
    parser.add_argument(
        "--allow-upload",
        action="store_true",
        help="Permit temporary upload to a third-party host when --image-url is absent.",
    )
    parser.add_argument(
        "--engine",
        default=None,
        choices=("google_lens", "yandex_images"),
        help="Override TL_SERP_ENGINE for this run.",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=10,
        help="How many candidates to print (all are saved regardless).",
    )
    parser.add_argument(
        "--skip-search-image-check",
        action="store_true",
        help="Skip the Stage 2.5 guard. Only useful to save the model load "
        "when you have already validated this exact URL.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    banner("TRACELOCK - PHASE 0 - SEARCH VIABILITY GATE")
    emit("Question: can we dynamically discover web content from an input image?")
    emit("This script has no expected answer. It does not know what it should find.")

    # --- Stage 0: configuration -------------------------------------------
    try:
        settings = load_settings()
    except Exception as exc:  # pragma: no cover - pydantic surfaces its own text
        return fail(EXIT_SETUP_ERROR, "SETUP ERROR", str(exc), "Check your .env file.")

    engine = args.engine or settings.serp_engine

    # --- Stage 1: validate the probe locally ------------------------------
    stage(1, "Validating probe image")
    try:
        probe_sha, probe_info = validate_probe(args.image)
    except (FileNotFoundError, ValueError) as exc:
        return fail(
            EXIT_INPUT_ERROR,
            "INPUT ERROR",
            str(exc),
            "Point --image at a real JPEG or PNG.",
        )

    emit("  path   : {0}".format(args.image))
    emit("  sha256 : {0}".format(probe_sha))
    emit(
        "  image  : {0} {1}x{2} {3}".format(
            probe_info["format"], probe_info["width"], probe_info["height"], probe_info["mode"]
        )
    )

    run_id = make_run_id("search_gate", probe_sha)
    artifact = run_artifact_path(settings.runs_dir, run_id)
    emit("  run id : {0}".format(run_id))

    # --- Stage 2: obtain a publicly reachable URL -------------------------
    stage(2, "Resolving a public image URL")
    public_url = args.image_url

    if public_url:
        emit("  Using the URL you supplied. No new disclosure. This is the safe path.")
    elif args.allow_upload:
        emit("  No --image-url given. Uploading temporarily (you passed --allow-upload).")
        emit("  Host: litterbox.catbox.moe   Retention: {0}".format(settings.upload_retention))
        try:
            public_url = upload_temporary(
                args.image,
                retention=settings.upload_retention,
                timeout=settings.http_timeout * 2,
            )
        except HostingError as exc:
            return fail(
                EXIT_ARCHITECTURE_RISK,
                "ARCHITECTURE RISK DETECTED",
                "Stage 2 (image hosting) failed: {0}\n\n"
                "The search provider was never reached, so this is NOT yet a\n"
                "verdict on search viability -- but it does block the local-file\n"
                "path end to end.".format(exc),
                "Re-run with --image-url pointing at an already-public copy of\n"
                "this image. That path avoids hosting entirely and is preferred.",
            )
        emit("  uploaded: {0}".format(public_url))
    else:
        return fail(
            EXIT_SETUP_ERROR,
            "SETUP ERROR",
            "No public image URL available.\n\n"
            "Reverse-image APIs fetch a URL; none accept a local file upload.\n"
            "So the probe must be reachable on the public internet first.",
            "Preferred: --image-url https://.../your-public-photo.jpg\n"
            "Fallback : add --allow-upload to host it temporarily ({0}).\n"
            "           Note this discloses a face image to a third party.".format(
                settings.upload_retention
            ),
        )

    probe = ProbeRef(
        local_path=str(args.image.resolve()),
        sha256=probe_sha,
        public_url=public_url,
    )
    if probe.public_url is None:
        return fail(
            EXIT_INPUT_ERROR,
            "INPUT ERROR",
            "The image URL could not be normalized: {0!r}".format(public_url),
            "Supply an absolute http(s) URL.",
        )

    # --- Stage 2.5: search-image guard ------------------------------------
    # Validate the URL THE PROVIDER WILL FETCH -- not the local file. A local
    # file with a good face says nothing about what the URL serves, which is
    # exactly how a faceless identicon once passed this gate.
    search_check = None
    if not args.skip_search_image_check:
        emit()
        emit("[2.5] Validating the search image (the URL, not the local file)")
        emit(THIN)

        try:
            from tracelock.discovery.search_image import check_search_image
            from tracelock.face import FaceEngine

            emit("  loading face engine ...")
            guard_engine = FaceEngine()
            probe_analysis = guard_engine.analyze(args.image)

            search_check = check_search_image(
                probe.public_url,
                guard_engine,
                probe_analysis=probe_analysis,
                probe_bytes=args.image.read_bytes(),
            )
        except ImportError as exc:
            emit("  SKIPPED: face engine unavailable ({0})".format(exc))
        except Exception as exc:  # pragma: no cover - defensive
            emit("  SKIPPED: guard could not run ({0})".format(exc))

        if search_check is not None:
            if not search_check.ok:
                write_artifact(
                    artifact,
                    run_id=run_id,
                    probe=probe,
                    probe_info=probe_info,
                    result=None,
                    verdict="FAIL_SEARCH_IMAGE",
                    failure={
                        "stage": "search_image_guard",
                        "type": search_check.issue.value,
                        "message": search_check.detail,
                        "check": search_check.to_dict(),
                    },
                )
                return fail(
                    EXIT_INPUT_ERROR,
                    "INPUT ERROR - SEARCH IMAGE REJECTED",
                    "{0}\n\n{1}\n\n{2}\n\nArtifact: {3}".format(
                        search_check.issue.value,
                        search_check.issue.explanation,
                        search_check.detail,
                        artifact,
                    ),
                    "Point --image-url at a publicly reachable photograph that\n"
                    "actually contains the subject's face. The local --image is\n"
                    "NOT what the provider receives.",
                )

            emit(
                "  image    : {0} {1}x{2}, {3} bytes".format(
                    search_check.image_format,
                    search_check.width,
                    search_check.height,
                    search_check.byte_size,
                )
            )
            emit(
                "  face     : {0} detected, det={1:.3f}, quality={2:.3f} [{3}]".format(
                    search_check.faces_detected,
                    search_check.det_score,
                    search_check.quality_aggregate,
                    search_check.quality_band,
                )
            )

            relationship = search_check.probe_relationship
            if relationship:
                emit("  vs probe : cosine={0}  pHash={1}  same_image={2}".format(
                    "{0:.4f}".format(relationship.cosine_similarity)
                    if relationship.cosine_similarity is not None
                    else "-",
                    "{0}/64".format(relationship.phash_distance)
                    if relationship.phash_distance is not None
                    else "-",
                    relationship.is_same_image,
                ))
                emit("             MEASUREMENTS ONLY - no identity claim is made.")

            for warning, detail in zip(
                search_check.warnings, search_check.warning_details
            ):
                emit("  warn     : [{0}] {1}".format(warning.value, detail))

            if not search_check.warnings:
                emit("  guard    : PASSED, no warnings")

    # --- Stage 3: live search ---------------------------------------------
    stage(3, "Live search")
    try:
        provider = SerpApiProvider(
            settings.serpapi_api_key,
            engine=engine,
            timeout=settings.http_timeout,
        )
        check_requirements(provider, probe)
    except ProviderConfigError as exc:
        return fail(
            EXIT_SETUP_ERROR,
            "SETUP ERROR",
            str(exc),
            "Copy .env.example to .env and set TL_SERPAPI_API_KEY.\n"
            "This is a setup gap, NOT evidence that discovery is impossible.",
        )

    emit("  provider : {0}".format(provider.name))
    emit("  engine   : {0}".format(provider.engine))
    emit("  probe url: {0}".format(probe.public_url))
    emit("  querying live (no_cache=true) ...")

    result: ProviderResult | None = None
    try:
        result = provider.search(probe, limit=settings.max_candidates)
    except ProviderConfigError as exc:
        # e.g. HTTP 401 -- the key is wrong. A credential problem, never a
        # verdict on whether dynamic discovery is possible here.
        return fail(
            EXIT_SETUP_ERROR,
            "SETUP ERROR",
            str(exc),
            "Verify TL_SERPAPI_API_KEY in .env against\n"
            "https://serpapi.com/manage-api-key\n"
            "This is a setup gap, NOT evidence that discovery is impossible.",
        )
    except ProviderBlockedError as exc:
        write_artifact(
            artifact,
            run_id=run_id,
            probe=probe,
            probe_info=probe_info,
            result=None,
            verdict="FAIL_BLOCKED",
            failure={"stage": "search", "type": "blocked", "message": str(exc)},
        )
        return fail(
            EXIT_ARCHITECTURE_RISK,
            "ARCHITECTURE RISK DETECTED",
            "The provider refused the request.\n\n{0}\n\n"
            "Artifact: {1}".format(exc, artifact),
            "Check the key and remaining quota at https://serpapi.com/dashboard\n"
            "If quota is exhausted, this is a billing limit, not a hard block --\n"
            "re-run tomorrow or on a fresh key before concluding the architecture\n"
            "is unviable.",
        )
    except ProviderSchemaError as exc:
        write_artifact(
            artifact,
            run_id=run_id,
            probe=probe,
            probe_info=probe_info,
            result=None,
            verdict="FAIL_SCHEMA",
            failure={
                "stage": "parse",
                "type": "schema",
                "message": str(exc),
                "observed_keys": exc.observed_keys,
            },
        )
        return fail(
            EXIT_ARCHITECTURE_RISK,
            "ARCHITECTURE RISK DETECTED",
            "Reached the provider, but could not read the response.\n\n{0}\n\n"
            "Top-level keys actually returned:\n  {1}\n\n"
            "Artifact: {2}".format(
                exc, ", ".join(exc.observed_keys) or "(none)", artifact
            ),
            "Add the correct key to RESULT_KEYS in\n"
            "src/tracelock/discovery/serpapi_lens.py -- this is usually a\n"
            "two-minute fix, not an architecture problem.",
        )
    except ProviderTransportError as exc:
        return fail(
            EXIT_ARCHITECTURE_RISK,
            "ARCHITECTURE RISK DETECTED",
            "Network failure reaching the provider: {0}".format(exc),
            "Check connectivity and retry. Persistent failure here means the\n"
            "environment cannot reach the provider at all.",
        )
    except DiscoveryError as exc:  # pragma: no cover - defensive catch-all
        return fail(
            EXIT_ARCHITECTURE_RISK,
            "ARCHITECTURE RISK DETECTED",
            "Unclassified discovery failure: {0}".format(exc),
            "Inspect the traceback and classify this error type in base.py.",
        )

    emit("  returned in {0}s".format(result.elapsed_seconds))

    # --- Stage 4: report ---------------------------------------------------
    stage(4, "Results")
    usable = result.with_media
    emit("  provider           : {0} / {1}".format(result.provider, result.engine))
    emit("  candidates found   : {0}".format(result.count))
    emit("  with usable media  : {0}".format(len(usable)))

    if result.count:
        print_candidates(result, args.show)

    verdict = "PASS" if (result.count >= 1 and len(usable) >= 1) else "FAIL_NO_CANDIDATES"

    write_artifact(
        artifact,
        run_id=run_id,
        probe=probe,
        probe_info=probe_info,
        result=result,
        verdict=verdict,
        failure=None
        if verdict == "PASS"
        else {"stage": "results", "type": "no_usable_candidates"},
    )

    stage(5, "Artifact")
    emit("  {0}".format(artifact))
    emit("  ({0} bytes, includes the verbatim provider response)".format(
        artifact.stat().st_size
    ))

    # --- Stage 5: verdict --------------------------------------------------
    if verdict != "PASS":
        return fail(
            EXIT_ARCHITECTURE_RISK,
            "ARCHITECTURE RISK DETECTED",
            "The search completed but produced no usable candidates.\n\n"
            "  candidates found  : {0}\n"
            "  with usable media : {1}\n\n"
            "This is an honest negative. The provider worked; the image simply\n"
            "returned nothing actionable.".format(result.count, len(usable)),
            "Try a probe image that is more widely published, or switch engines\n"
            "(--engine yandex_images is markedly better at face matching).\n"
            "If several probes across both engines return nothing, the discovery\n"
            "strategy needs rethinking BEFORE Phase 1.",
        )

    banner("GATE PASSED")
    emit("Dynamic discovery is viable in this environment.")
    emit()
    emit("  {0} candidates discovered, {1} with usable media".format(
        result.count, len(usable)
    ))
    emit("  provider: {0} / {1}".format(result.provider, result.engine))
    emit("  artifact: {0}".format(artifact))
    emit()
    emit("Reminder: these candidates are UNVERIFIED. A search engine claimed a")
    emit("relationship; nothing has been proven. Phase 2 re-derives identity from")
    emit("the downloaded bytes and will reject most of these.")
    emit()
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
