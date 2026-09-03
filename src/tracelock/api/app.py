"""TRACELOCK HTTP API.

Thin. Every route normalizes an input into a `TraceInput` and hands it to the
service layer; none of them re-implement discovery, verification, scoring or
anchoring. The CLI scripts remain the reproducible reference path and call the
same modules.

WHAT THE BROWSER NEVER SEES
---------------------------
  * server filesystem paths -- `TraceInput.to_dict()` omits `local_path`
  * private keys or RPC URLs -- config is never serialized to a response
  * raw face embeddings -- only quantized digests exist in artifacts at all
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from tracelock.ingest import InputType, classify_url
from tracelock.ingest import resolve as resolve_input
from tracelock.service import examples as examples_module
from tracelock.service.inputs import (
    InputError,
    TraceInput,
    from_google_drive,
    from_upload,
    from_url,
    from_webcam,
)
from tracelock.service.runner import ENGINE, STAGES, STORE, start_investigation

WEB_DIR = Path(__file__).resolve().parents[3] / "web"
WORK_DIR = Path("data/inputs")

logger = logging.getLogger(__name__)

api = APIRouter(prefix="/api")


class _ResolutionRefused(Exception):
    """A public URL could not be resolved. Carries the full resolution so the
    caller can explain WHICH platform refused and what to do instead."""

    def __init__(self, resolution) -> None:
        super().__init__(resolution.reason)
        self.resolution = resolution


def _failure_for(resolution):
    """Turn a resolution refusal into a structured stage failure."""
    from tracelock.ingest.adapters import Support
    from tracelock.service import failures

    reason = (resolution.reason or "").lower()

    if "private or internal network" in reason:
        return failures.blocked_target("url_resolution", resolution.reason)
    if "did not respond in time" in reason or "timed out" in reason:
        return failures.timeout("url_resolution", what="that page")

    if resolution.adapter.support is Support.AUTH_REQUIRED:
        return failures.platform_blocked(
            resolution.classification.platform.display,
            resolution.adapter.note,
            resolution.adapter.guidance,
        )

    return failures.unavailable(
        "url_resolution", resolution.reason, resolution.guidance
    )


def _input_error(exc: InputError, failure=None) -> JSONResponse:
    """A readable failure plus at least one way forward.

    Recovery is always attached. A judge who pastes an Instagram link and is
    told only "that failed" is at a dead end; the same judge told "upload it
    instead, or continue with local analysis" is not.
    """
    from tracelock.service.failures import recovery_options

    payload = exc.to_dict()
    payload["recovery"] = recovery_options(failure)
    if failure is not None:
        payload["failure"] = failure.to_dict()
    return JSONResponse(status_code=400, content=payload)


# ==========================================================================
# Input routes -- five sources, ONE normalized object
# ==========================================================================


@api.post("/input/upload")
async def input_upload(file: UploadFile = File(...)) -> Any:
    try:
        trace = from_upload(await file.read(), file.filename or "upload", WORK_DIR)
    except InputError as exc:
        return _input_error(exc)
    return _describe(trace)


@api.post("/input/webcam")
async def input_webcam(file: UploadFile = File(...)) -> Any:
    try:
        trace = from_webcam(await file.read(), WORK_DIR)
    except InputError as exc:
        return _input_error(exc)
    return _describe(trace)


@api.post("/input/url")
async def input_url(url: str = Form(...), candidate_url: str = Form("")) -> Any:
    """Accept ANY public link, not only a direct image.

    A direct image loads straight through. Anything else -- an article, a
    social post, a Drive share link -- is resolved first, using only the
    metadata the site publishes to anonymous visitors, and the resolved image
    is then loaded through exactly the same path. There is one ingestion
    pipeline; the resolver simply works out what to feed it.

    When the page held several images they are returned ranked, so the UI can
    offer the alternatives and the operator can override the top pick.
    """
    def _load() -> Any:
        classification = classify_url(url)

        # A direct image needs no resolution at all -- do not fetch the page.
        if classification.input_type is InputType.DIRECT_IMAGE_URL and not candidate_url:
            return from_url(url, WORK_DIR), None

        resolution = resolve_input(url)

        if candidate_url:
            # An override must be one of the candidates WE found on the page.
            offered = {c.url for c in resolution.candidates}
            if candidate_url not in offered:
                raise InputError(
                    "That image is not one of the candidates found on the page.",
                    hint="Choose one of the images shown, or paste its address directly.",
                )
            resolution = replace(resolution, image_url=candidate_url)

        if not resolution.ok:
            raise _ResolutionRefused(resolution)

        # The resolved URL is loaded through the SAME loader as a pasted direct
        # image, so it gets the identical SSRF guard, size cap and validation.
        trace = from_url(resolution.image_url, WORK_DIR)
        return trace, resolution

    try:
        trace, resolution = await asyncio.to_thread(_load)
    except _ResolutionRefused as refused:
        return _input_error(
            InputError(refused.resolution.reason, hint=refused.resolution.guidance),
            _failure_for(refused.resolution),
        )
    except InputError as exc:
        return _input_error(exc)

    payload = _describe(trace)
    if resolution is not None:
        payload["resolution"] = resolution.to_dict()
    return payload


@api.post("/input/drive")
async def input_drive(url: str = Form(...)) -> Any:
    try:
        trace = await asyncio.to_thread(from_google_drive, url, WORK_DIR)
    except InputError as exc:
        return _input_error(exc)
    return _describe(trace)


@api.post("/input/example")
async def input_example(example_id: str = Form(...)) -> Any:
    example = examples_module.find(example_id)
    if example is None:
        raise HTTPException(status_code=404, detail="Unknown example.")
    try:
        trace = await asyncio.to_thread(
            examples_module.to_trace_input, example, WORK_DIR
        )
    except InputError as exc:
        return _input_error(exc)
    return _describe(trace)


# ==========================================================================
# Examples
# ==========================================================================


@api.get("/examples")
async def list_examples() -> Any:
    return {
        "notice": (
            "DEMO INPUT images. The investigation each triggers is fully live: "
            "discovery queries the real internet and every candidate is "
            "independently verified. Only the starting image is pre-selected."
        ),
        "examples": [e.to_dict() for e in examples_module.load_registry()],
    }


@api.get("/examples/{example_id}/preview")
async def example_preview(example_id: str):
    example = examples_module.find(example_id)
    if example is None or not example.exists:
        raise HTTPException(status_code=404, detail="Example not available.")
    return FileResponse(example.local_path)


# ==========================================================================
# Preview of a normalized input
# ==========================================================================

_PREVIEWS: dict[str, str] = {}

# Normalized inputs by content hash. Local analysis and URL linking both need
# the full TraceInput, not just a path.
_INPUTS: dict[str, TraceInput] = {}


def _describe(trace: TraceInput) -> dict[str, Any]:
    """Public description of an input, plus a preview handle.

    The server path is kept in a lookup table keyed by content hash so the
    browser can request the image without ever being told where it lives.
    """
    _PREVIEWS[trace.sha256] = trace.local_path
    _INPUTS[trace.sha256] = trace
    payload = trace.to_dict()
    payload["preview_url"] = "/api/preview/{0}".format(trace.sha256)
    return payload


@api.get("/preview/{digest}")
async def preview(digest: str):
    path = _PREVIEWS.get(digest)
    if not path or not Path(path).is_file():
        raise HTTPException(status_code=404, detail="Preview not available.")
    return FileResponse(path)


def _run_candidate_digests(run) -> set[str]:
    """Every content hash this specific run actually downloaded and analysed.

    This set is the authorisation boundary for candidate previews. A digest
    that is not in it does not belong to this run, and is not served -- even
    though the bytes may well sit in the shared CAS from some earlier
    investigation.
    """
    result = getattr(run, "result", None) or {}
    results = (result.get("verification") or {}).get("results") or []
    return {
        r["content_sha256"] for r in results
        if isinstance(r.get("content_sha256"), str)
    }


@api.get("/investigation/{run_id}/candidate/{content_sha256}")
async def candidate_image(run_id: str, content_sha256: str):
    """Serve one candidate image belonging to ONE investigation.

    Served from the CAS rather than proxied from its origin, deliberately: the
    CAS holds the EXACT bytes that produced the similarity score. Re-fetching
    the remote URL could return something different, and a thumbnail that did
    not match what was measured would misrepresent the evidence. It also just
    works -- many candidate hosts (licdn.com, pbs.twimg.com, cloudfront)
    refuse hotlinked requests.

    TWO INDEPENDENT CHECKS, AND BOTH ARE LOAD-BEARING
    -------------------------------------------------
    1. `blob_path` rejects anything that is not a 64-character hex digest, so
       no traversal sequence can survive to touch the filesystem.
    2. The digest must appear in THIS run's own results. The CAS is shared
       across every investigation the process has ever performed, so without
       this check a caller holding any hash could read another run's candidate
       media. Path safety alone would not prevent that; scoping does.
    """
    from tracelock.acquisition.cas import ContentAddressedStore

    run = STORE.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Unknown investigation.")

    try:
        path = ContentAddressedStore("data/cas").blob_path(content_sha256)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Not a valid content hash.")

    # Scope check before any filesystem answer, so a caller cannot use the
    # 404/200 difference to probe what other runs hold.
    if content_sha256 not in _run_candidate_digests(run):
        raise HTTPException(
            status_code=404, detail="No such candidate in this investigation."
        )

    if not path.is_file():
        raise HTTPException(status_code=404, detail="That candidate image is not stored.")

    return FileResponse(path, media_type="image/jpeg")


# ==========================================================================
# Face pre-check -- run before the operator commits to an investigation
# ==========================================================================


@api.post("/precheck")
async def precheck(sha256: str = Form(...)) -> Any:
    """Validate the selected image without starting a full investigation."""
    path = _PREVIEWS.get(sha256)
    if not path:
        raise HTTPException(status_code=404, detail="That image is no longer loaded.")

    def _analyse() -> dict[str, Any]:
        from tracelock.face.errors import NoFaceDetectedError

        checks = [{"label": "Image readable", "ok": True, "detail": ""}]
        try:
            engine = ENGINE.get()
            analysis = engine.analyze(path)
        except NoFaceDetectedError:
            checks.append({
                "label": "Face detected", "ok": False,
                "detail": "No face found. TRACELOCK searches by face.",
            })
            return {"ok": False, "checks": checks}
        except Exception as exc:
            checks.append({
                "label": "Face detected", "ok": False,
                "detail": "The face engine could not run: {0}".format(str(exc)[:120]),
            })
            return {"ok": False, "checks": checks}

        quality = analysis.primary.quality
        usable = quality.aggregate >= 0.15
        checks.append({
            "label": "Face detected", "ok": True,
            "detail": "{0} face(s), confidence {1:.2f}".format(
                analysis.faces_detected, analysis.primary.det_score),
        })
        checks.append({
            "label": "Image quality acceptable", "ok": usable,
            "detail": "quality {0:.2f} [{1}]".format(
                quality.aggregate, quality.band.value),
        })
        return {
            "ok": usable,
            "checks": checks,
            "faces_detected": analysis.faces_detected,
            "quality": round(quality.aggregate, 4),
            "quality_band": quality.band.value,
        }

    return await asyncio.to_thread(_analyse)


# ==========================================================================
# MODE B -- local analysis (no network, no discovery, no score)
# ==========================================================================


@api.post("/analyze/local")
async def analyze_local(sha256: str = Form(...)) -> Any:
    """Everything measurable offline. Makes ZERO outbound requests.

    Structurally cannot return a trust score or candidates: the report type
    has no field for either, and `assert_no_fabricated_findings` raises on
    serialization if one ever appears.
    """
    trace = _INPUTS.get(sha256)
    if trace is None:
        raise HTTPException(status_code=404, detail="That image is no longer loaded.")

    def _analyse() -> dict[str, Any]:
        from tracelock.service.local_analysis import analyse_locally

        return analyse_locally(trace, ENGINE.get()).to_dict()

    try:
        return await asyncio.to_thread(_analyse)
    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Local analysis could not complete.",
                "hint": str(exc)[:200],
            },
        )


def _friendly_search_image_error(result) -> str:
    issue = result.issue.value if result.issue else ""
    return {
        "URL_UNFETCHABLE": "That link could not be downloaded.",
        "NOT_AN_IMAGE": "That link does not return an image.",
        "NO_FACE_DETECTED": "No face was found in the image at that link.",
        "FACE_NOT_USABLE": "The face at that link is too small or degraded to search with.",
        "FACE_ANALYSIS_FAILED": "That image could not be analysed.",
    }.get(issue, "That link cannot be used for public discovery.")


def _resolved_image_error(result, resolution) -> str:
    """Failure message that names WHICH image failed.

    After resolution the image being checked may not be the URL the operator
    pasted -- it is the preview image the page published. Saying "that link
    does not return an image" would then be actively confusing, so the message
    distinguishes the post from the image it pointed at.
    """
    base = _friendly_search_image_error(result)

    # When the pasted link WAS the image, there is no distinction to draw.
    if resolution.method.value in ("direct", "sniffed"):
        return base

    what = resolution.classification.label.replace(" detected", "")
    return "{0} was resolved to a preview image, but {1}".format(
        what, base[0].lower() + base[1:]
    )


@api.post("/input/link-url")
async def link_public_url(
    sha256: str = Form(...),
    url: str = Form(...),
    candidate_url: str = Form(""),
) -> Any:
    """Attach a publicly reachable URL to an already-selected local image.

    This is the ONLY route from a webcam frame or local upload to public
    discovery, and it is entirely operator-driven: the user supplies a URL for
    an image they have already published. Nothing is uploaded on their behalf.

    The URL goes through the Stage 2.5 guard -- the same one the CLI uses --
    and the two images are compared perceptually so the operator can see
    whether the public copy IS the image they selected. That comparison is a
    VISUAL fact about pixels; no identity claim is made from it.
    """
    trace = _INPUTS.get(sha256)
    if trace is None:
        raise HTTPException(status_code=404, detail="That image is no longer loaded.")

    def _check() -> dict[str, Any]:
        from tracelock.discovery.search_image import check_search_image

        # STEP 1 -- resolve whatever the operator pasted into an image URL.
        # A social post or article is turned into the preview image the site
        # already publishes; a direct image passes straight through. A page
        # with several images yields a RANKED list, and the operator may
        # override the top pick via `candidate_url`.
        resolution = resolve_input(url)

        if candidate_url:
            # An override must be one of the candidates WE found on the page.
            # Accepting an arbitrary URL here would let a caller smuggle in a
            # target that never went through page resolution.
            offered = {c.url for c in resolution.candidates}
            if candidate_url not in offered:
                return {
                    "ok": False,
                    "error": "That image is not one of the candidates found on the page.",
                    "issue": "CANDIDATE_NOT_OFFERED",
                    "resolution": resolution.to_dict(),
                }
            resolution = replace(resolution, image_url=candidate_url)

        if not resolution.ok:
            from tracelock.service.failures import recovery_options

            failure = _failure_for(resolution)
            return {
                "ok": False,
                "error": failure.message,
                "hint": failure.detail or resolution.guidance,
                "issue": "RESOLUTION_UNAVAILABLE",
                "failure": failure.to_dict(),
                "recovery": recovery_options(failure),
                "resolution": resolution.to_dict(),
            }

        # STEP 2 -- the resolved URL goes through the EXISTING Stage 2.5 guard.
        # Deliberately not a separate validation path: the resolved URL is
        # re-validated exactly like any other, including the SSRF guard, so a
        # page whose og:image points at a private address is still blocked.
        local_bytes = Path(trace.local_path).read_bytes()
        result = check_search_image(
            resolution.image_url, ENGINE.get(), probe_bytes=local_bytes
        )

        if not result.ok:
            from tracelock.service import failures
            from tracelock.service.failures import recovery_options

            issue = result.issue.value if result.issue else ""
            failure = (
                failures.timeout("image_download", what="that image")
                if "TIMEOUT" in issue
                else failures.unavailable(
                    "image_validation",
                    _resolved_image_error(result, resolution),
                    result.issue.explanation if result.issue else "",
                )
            )
            return {
                "ok": False,
                "error": failure.message,
                "hint": failure.detail,
                "issue": issue or None,
                "failure": failure.to_dict(),
                "recovery": recovery_options(failure),
                "resolution": resolution.to_dict(),
            }

        relationship = result.probe_relationship
        distance = relationship.phash_distance if relationship else None
        same_image = relationship.is_same_image if relationship else None

        return {
            "ok": True,
            "url": result.url,
            "resolution": resolution.to_dict(),
            "image": {
                "format": result.image_format,
                "width": result.width,
                "height": result.height,
                "byte_size": result.byte_size,
            },
            "face": {
                "count": result.faces_detected,
                "det_score": result.det_score,
                "quality": result.quality_aggregate,
                "quality_band": result.quality_band,
            },
            "comparison": {
                "phash_distance": distance,
                "phash_max": 64,
                "is_same_image": same_image,
                "verdict": (
                    "This is the same image you selected."
                    if same_image
                    else "This is a DIFFERENT image from the one you selected."
                ),
                "is_identity_claim": False,
                "note": (
                    "A perceptual-hash comparison of PIXELS. It says whether "
                    "the two files look like the same picture. It makes no "
                    "claim about who is depicted -- that requires calibrated "
                    "face verification, which runs during discovery."
                ),
            },
            "warnings": [
                {"code": w.value, "detail": d}
                for w, d in zip(result.warnings, result.warning_details)
            ],
        }

    try:
        payload = await asyncio.to_thread(_check)
    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={"error": "That link could not be checked.", "hint": str(exc)[:200]},
        )

    if not payload.get("ok"):
        return JSONResponse(status_code=400, content=payload)

    # Re-register the input WITH its public URL so discovery becomes possible.
    linked = TraceInput(
        source_type=trace.source_type,
        kind=trace.kind,
        local_path=trace.local_path,
        filename=trace.filename,
        mime_type=trace.mime_type,
        sha256=trace.sha256,
        byte_size=trace.byte_size,
        image_url=payload["url"],
        provenance={**trace.provenance, "public_url_supplied_by": "operator"},
    )
    payload["input"] = _describe(linked)
    return payload


@api.get("/hosting/info")
async def hosting_info() -> Any:
    """What temporary hosting can and cannot do, for the consent screen.

    Every capability here is read from the provider itself, so the screen
    cannot promise a deletion the code will not perform.
    """
    from tracelock.discovery.hosts import available_providers, resolve_provider

    selected = resolve_provider()
    return {
        "selected": selected.key,
        "display_name": selected.display_name,
        "configured": selected.configured,
        "supports_deletion": selected.supports_deletion,
        "retention_note": selected.retention_note,
        "providers": available_providers(),
        "consent_required": True,
        "what_happens": (
            "Reverse-image search engines fetch a URL; they cannot receive a "
            "file. To search the public web with this image, TRACELOCK must "
            "first make a copy of it publicly reachable."
        ),
        "guarantees": [
            "Local analysis has already run and is unaffected.",
            "Nothing is uploaded until you consent.",
            "Every discovered candidate is independently re-downloaded and "
            "face-verified before it counts as evidence.",
            "TRACELOCK never claims a match that verification did not confirm.",
        ],
    }


@api.post("/input/publish")
async def publish_for_discovery(
    sha256: str = Form(...),
    consent: bool = Form(False),
    retention: str = Form("1h"),
) -> Any:
    """Publish a LOCAL image so reverse-image search can fetch it.

    This is the bridge that was missing: an uploaded photo or a webcam frame
    has no public URL, and every reverse-image API fetches a URL rather than
    accepting a file, so local input could never enter discovery at all.

    Three things make this safe to offer rather than dangerous to automate:

      * `consent` is required and defaults to False. A request without it is
        refused, so no code path can publish a face by accident.
      * The operator has already seen the local analysis, so they know what
        they are publishing before they decide.
      * The provider's real capabilities -- especially whether it can delete --
        are shown first and recorded in the provenance afterwards.
    """
    from tracelock.discovery.hosts import (
        HostingError,
        Retention,
        resolve_provider,
    )
    from tracelock.service.failures import recovery_options

    trace = _INPUTS.get(sha256)
    if trace is None:
        raise HTTPException(status_code=404, detail="That image is no longer loaded.")

    if not consent:
        # Not an error the operator made -- a guard doing its job.
        return JSONResponse(
            status_code=400,
            content={
                "error": "Consent is required before publishing this image.",
                "hint": (
                    "TRACELOCK does not upload biometric images without an "
                    "explicit decision."
                ),
                "published": False,
            },
        )

    if trace.image_url:
        # Already public. Publishing again would create a second copy for
        # nothing.
        return {
            "published": False,
            "already_public": True,
            "image_url": trace.image_url,
            "input": _describe(trace),
        }

    provider = resolve_provider()
    if not provider.configured:
        failure = {
            "status": "not_configured",
            "stage": "temporary_hosting",
            "message": (
                "Public discovery from a local image needs a temporary image "
                "host, and {0} is not configured.".format(provider.display_name)
            ),
        }
        return JSONResponse(
            status_code=400,
            content={
                "error": failure["message"],
                "hint": "Set {0} in .env, or paste a public URL for this image "
                        "instead.".format(", ".join(provider.missing_configuration())),
                "published": False,
                "failure": failure,
                "recovery": recovery_options(None),
            },
        )

    def _publish() -> Any:
        data = Path(trace.local_path).read_bytes()
        try:
            window = Retention(retention)
        except ValueError:
            window = Retention.ONE_HOUR
        return provider.upload(data, trace.filename or "probe.jpg", retention=window)

    try:
        hosted = await asyncio.to_thread(_publish)
    except HostingError as exc:
        logger.warning("temporary hosting failed on %s: %s", provider.key, exc)
        return JSONResponse(
            status_code=400,
            content={
                "error": "That image could not be published for search.",
                "hint": str(exc)[:200],
                "published": False,
                "failure": {
                    "status": "unavailable",
                    "stage": "temporary_hosting",
                    "message": "The temporary image host did not accept the upload.",
                },
                "recovery": recovery_options(None),
            },
        )

    # Re-register the input WITH its public URL, exactly as the operator-supplied
    # link route does. Same normalization point, same downstream pipeline.
    published = TraceInput(
        source_type=trace.source_type,
        kind=trace.kind,
        local_path=trace.local_path,
        filename=trace.filename,
        mime_type=trace.mime_type,
        sha256=trace.sha256,
        byte_size=trace.byte_size,
        image_url=hosted.url,
        provenance={
            **trace.provenance,
            "public_url_supplied_by": "temporary_host",
            "temporary_hosting": hosted.to_dict(),
        },
    )
    _INPUTS[sha256] = published

    return {
        "published": True,
        "image_url": hosted.url,
        "hosting": hosted.to_dict(),
        "input": _describe(published),
        "note": (
            "This copy exists so search engines can fetch it. "
            + hosted.note
        ),
    }


# ==========================================================================
# Investigation
# ==========================================================================


@api.post("/investigate")
async def investigate(
    sha256: str = Form(...),
    source_type: str = Form("upload"),
    image_url: str = Form(""),
    kind: str = Form("ORGANIC"),
    filename: str = Form("image"),
    limit: int = Form(0),
    anchor: bool = Form(False),
    mode: str = Form("fast"),
) -> Any:
    """Start a run. Returns a run id; progress streams over the WebSocket."""
    path = _PREVIEWS.get(sha256)
    if not path:
        raise HTTPException(status_code=404, detail="That image is no longer loaded.")

    from tracelock.service.inputs import InputKind, SourceType

    # Prefer the input the SERVER already holds. Rebuilding it from form fields
    # silently dropped `provenance`, so an image published through the consent
    # flow lost every trace of how its URL was obtained -- the evidence
    # artifact then recorded a public URL with no account of where it came
    # from, and the results screen reported "a public URL you supplied" for a
    # file the operator had just consented to publish.
    #
    # For a provenance engine that is the wrong thing to lose, so the stored
    # object wins and the form fields remain the fallback.
    stored = _INPUTS.get(sha256)
    if stored is not None and stored.local_path == path:
        trace = stored
    else:
        trace = TraceInput(
            source_type=SourceType(source_type),
            kind=InputKind(kind),
            local_path=path,
            filename=filename,
            mime_type="image/jpeg",
            sha256=sha256,
            byte_size=Path(path).stat().st_size,
            image_url=image_url or None,
        )

    # The network is server configuration, NOT a client parameter. Letting the
    # browser choose is exactly how the label and the transaction diverged.
    from tracelock.chain.config import load_chain_config

    from tracelock.service.budget import Mode

    try:
        selected = Mode(mode)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Unknown mode '{0}'. Use 'fast' or 'thorough'.".format(mode),
        )

    run = await start_investigation(
        # limit=0 means "let the mode decide" -- an explicit limit still wins,
        # which keeps existing scripts and tests working unchanged.
        trace, limit=limit or None, anchor=anchor,
        network=load_chain_config().network_key,
        mode=selected.value,
    )
    return {
        "run_id": run.run_id,
        "mode": selected.value,
        "stages": [s.to_dict() for s in run.stages],
    }


@api.post("/investigation/{run_id}/cancel")
async def cancel_investigation(run_id: str) -> Any:
    """Ask a running investigation to stop at its next safe checkpoint.

    Cooperative, not forceful: nothing is killed mid-write, so the CAS never
    ends up holding a partial blob. A cancelled run reports no findings -- the
    candidates it happened to reach are an incomplete sample, not a result.
    """
    run = STORE.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Unknown investigation.")

    runner = getattr(run, "_runner", None)
    if runner is None or run.status not in ("running", "pending"):
        return {"run_id": run_id, "cancelled": False, "status": run.status}

    runner.token.cancel("Cancelled by the operator.")
    return {"run_id": run_id, "cancelled": True, "status": run.status}


def _public_snapshot(run) -> dict[str, Any]:
    """The run snapshot as the browser is allowed to see it.

    Two view-layer adjustments, neither of which touches the pipeline, the
    stored result, or the anchored evidence artifact:

      cas_path IS REMOVED
          It is a real filesystem path into the store ("data/cas/blobs/f3/..").
          It is relative rather than absolute, and it is not one of the eleven
          fingerprint leaves, so dropping it from the wire changes no hash and
          no anchor -- it simply stops publishing internal layout that the UI
          has no use for. `content_sha256` is the identifier the UI needs, and
          it stays.

      platform IS ADDED
          Derived by calling the EXISTING `classify_source`, the same pure
          offline classifier the social summary already uses. Nothing is
          reclassified: this only labels a candidate's host for display, so a
          LinkedIn or Reddit lookalike reads as such in the gallery.

    The copy is deliberate. `run.snapshot()` hands back the live result object;
    mutating it here would corrupt the run's own state and, worse, the artifact
    written from it.
    """
    from tracelock.ingest.social import SourceCategory, classify_source

    snapshot = run.snapshot()
    result = snapshot.get("result")
    if not isinstance(result, dict):
        return snapshot

    verification = result.get("verification")
    if not isinstance(verification, dict):
        return snapshot
    results = verification.get("results")
    if not isinstance(results, list):
        return snapshot

    public_results = []
    for entry in results:
        if not isinstance(entry, dict):
            public_results.append(entry)
            continue
        clean = {k: v for k, v in entry.items() if k != "cas_path"}
        classification = classify_source(entry.get("source_url"))
        clean["platform"] = (
            classification.platform
            if classification.category is SourceCategory.SOCIAL
            else None
        )
        public_results.append(clean)

    snapshot["result"] = {
        **result,
        "verification": {**verification, "results": public_results},
    }
    return snapshot


@api.get("/investigation/{run_id}")
async def investigation_status(run_id: str) -> Any:
    run = STORE.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Unknown investigation.")
    return _public_snapshot(run)


@api.websocket("/ws/investigation/{run_id}")
async def investigation_socket(socket: WebSocket, run_id: str) -> None:
    """Stream real stage transitions as the pipeline reaches them."""
    await socket.accept()
    run = STORE.get(run_id)
    if run is None:
        await socket.send_json({"error": "Unknown investigation."})
        await socket.close()
        return

    event = getattr(run, "_notify_event", None)
    try:
        await socket.send_json(_public_snapshot(run))
        while run.status == "running":
            if event is not None:
                try:
                    await asyncio.wait_for(event.wait(), timeout=2.0)
                    event.clear()
                except asyncio.TimeoutError:
                    pass
            else:
                await asyncio.sleep(0.5)
            await socket.send_json(_public_snapshot(run))
        await socket.send_json(_public_snapshot(run))
    except WebSocketDisconnect:
        return
    finally:
        try:
            await socket.close()
        except RuntimeError:
            pass


# ==========================================================================
# Blockchain
# ==========================================================================


@api.get("/chain/status")
async def chain_status() -> Any:
    """What is configured. Never returns a key or an RPC URL."""
    from tracelock.chain.config import load_chain_config

    config = load_chain_config()
    described = config.describe()
    described.pop("rpc_url_set", None)
    profile = config.profile

    return {
        # Identity comes from the profile that will actually execute, never
        # from a UI default. `address_url` returns "" for an ephemeral chain,
        # so no plausible-looking explorer link can be fabricated.
        "network": profile.key,
        "network_display_name": profile.display_name,
        "chain_id": profile.chain_id,
        "ephemeral": profile.ephemeral,
        "publicly_verifiable": profile.publicly_verifiable,
        "persistence_note": profile.persistence_note,
        "configured": bool(config.rpc_url and config.private_key) or config.is_local,
        "account": described["account"],
        "contract_address": described["contract_address"],
        "explorer": profile.address_url(config.contract_address)
        if config.contract_address
        else "",
        "faucet": profile.faucet,
        # Readiness panel. Reports WHETHER each variable is set, never its
        # value -- a private key must never reach the browser.
        "readiness": {
            "mode": "LOCAL MODE" if profile.ephemeral else "PUBLIC TESTNET READY",
            "summary": (
                "Anchors exist only in this demo process."
                if profile.ephemeral
                else "{0} — anchors are publicly verifiable.".format(profile.display_name)
            ),
            "required_env": [
                {
                    "name": "TL_CHAIN",
                    "set": True,
                    "current": profile.key,
                    # Deliberately does NOT list the other valid networks: a
                    # live status payload must never contain the name of a
                    # chain it is not running on.
                    "note": "the network this instance will actually use",
                },
                {
                    "name": "TL_RPC_URL",
                    "set": bool(config.rpc_url),
                    "current": None,
                    "note": "RPC endpoint for the chosen network",
                },
                {
                    "name": "TL_PRIVATE_KEY",
                    "set": config.has_key,
                    "current": None,
                    "note": "TESTNET key only. Never shown, never logged.",
                },
                {
                    "name": "TL_CONTRACT_ADDRESS",
                    "set": bool(config.contract_address),
                    "current": config.contract_address or None,
                    "note": "set automatically by deploy_contract.py",
                },
            ],
        },
    }


class _PublicConfigInvalid(Exception):
    """Public settings are malformed; no transaction was attempted."""

    def __init__(self, preflight) -> None:
        super().__init__("; ".join(preflight.problems))
        self.preflight = preflight


def _public_failure_payload(
    *, headline: str, message: str, status: str, problems: list[str] | None = None
) -> dict[str, Any]:
    """A public failure that claims nothing and always offers a way forward.

    Deliberately contains no tx_hash, block_number or explorer_url -- there is
    no transaction, and inventing any of those is the exact lie this system
    exists to avoid. Technical detail stays in the server log.
    """
    payload: dict[str, Any] = {
        "error": "PUBLIC ANCHOR UNAVAILABLE",
        "headline": headline,
        "message": message,
        "anchored": False,
        "publicly_verified": False,
        "status": status,
        "recovery": [
            {"action": "retry_public", "label": "Retry public anchor"},
            {"action": "anchor_local", "label": "Continue with Local Demo Chain"},
            {"action": "return_results", "label": "Return to Investigation"},
        ],
    }
    if problems:
        payload["problems"] = problems
    return payload


@api.post("/chain/anchor")
async def anchor_artifact(
    artifact_path: str = Form(...),
    use_local_fallback: bool = Form(False),
) -> Any:
    """Anchor an existing evidence artifact.

    `use_local_fallback` is a BOOLEAN, not a network name. The browser can ask
    to fall back to the one non-public, honestly-labelled demo chain; it cannot
    name a network. Letting a client choose the network is exactly how the
    interface once advertised Polygon Amoy while every anchor executed in an
    in-process EVM, and that must stay impossible.

    The fallback is also never automatic. A failed public anchor returns a
    failure with recovery options; only an explicit second request, with this
    flag set, anchors locally. Silently downgrading would mean an operator who
    asked for a publicly verifiable anchor got a local one without being told.
    """
    path = Path(artifact_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Evidence artifact not found.")

    def _anchor() -> dict[str, Any]:
        from tracelock.chain.compiler import load_or_compile
        from tracelock.chain.config import (
            build_adapter_with_contract,
            load_chain_config,
        )
        from tracelock.chain.notary import EvidenceNotary
        from tracelock.chain.preflight import PublicConfigStatus, check_public_config

        artifact = json.loads(path.read_text(encoding="utf-8"))

        # Same pipeline either way -- only the network differs, and "local" is
        # the single value this flag can produce.
        config = load_chain_config(network="local" if use_local_fallback else None)

        if not use_local_fallback and not config.is_local:
            # Structural check before opening a socket: a malformed key or RPC
            # URL is caught here rather than as a signing error later.
            preflight = check_public_config()
            if preflight.status is PublicConfigStatus.INVALID_CREDENTIALS:
                raise _PublicConfigInvalid(preflight)

        compiled = load_or_compile()
        adapter, address, _ = build_adapter_with_contract(config, compiled)
        notary = EvidenceNotary(adapter, address)
        record, _receipt = notary.anchor(artifact, confirmations=config.confirmations)
        path.with_suffix(".anchor.json").write_text(
            json.dumps(record.to_dict(), indent=2), encoding="utf-8"
        )
        return record.to_dict()

    try:
        return await asyncio.to_thread(_anchor)
    except _PublicConfigInvalid as invalid:
        return JSONResponse(
            status_code=400,
            content=_public_failure_payload(
                headline=invalid.preflight.status.headline,
                message=(
                    "The public blockchain settings are not valid, so no "
                    "transaction was attempted. No public verification record "
                    "was created."
                ),
                problems=list(invalid.preflight.problems),
                status=invalid.preflight.status.value,
            ),
        )
    except Exception as exc:
        from tracelock.chain.config import ChainMode, resolve_chain_mode
        from tracelock.chain.preflight import classify_anchor_failure
        from tracelock.service.runner import _friendly_chain_error

        resolution = resolve_chain_mode()
        public = resolution.mode is ChainMode.PUBLIC and not use_local_fallback

        if not public:
            # Local anchoring failed. There is no public claim to retract and
            # no fallback left to offer, so this is an ordinary failure.
            return JSONResponse(
                status_code=400,
                content={
                    "error": _friendly_chain_error(exc),
                    "anchored": False,
                    "publicly_verified": False,
                    "mode": resolution.mode.value,
                },
            )

        # A failed public anchor is NEVER downgraded into a local success
        # behind the operator's back. The local chain is OFFERED; taking it is
        # a second, explicit request.
        status, message, detail = classify_anchor_failure(exc)
        logger.warning(
            "public anchor failed on %s: %s", resolution.network_key, detail
        )
        return JSONResponse(
            status_code=400,
            content=_public_failure_payload(
                headline=status.headline, message=message, status=status.value
            ),
        )


@api.post("/chain/verify")
async def verify_anchor(artifact_path: str = Form(...)) -> Any:
    """Re-verify an artifact against its on-chain anchor."""
    path = Path(artifact_path)
    anchor_path = path.with_suffix(".anchor.json")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Evidence artifact not found.")
    if not anchor_path.is_file():
        return JSONResponse(
            status_code=400,
            content={
                "error": "This evidence has not been anchored yet.",
                "hint": "Anchor it first, then verify.",
            },
        )

    def _verify() -> dict[str, Any]:
        from tracelock.chain.compiler import load_or_compile
        from tracelock.chain.config import (
            build_adapter_with_contract,
            load_chain_config,
        )
        from tracelock.chain.notary import AnchorRecord, EvidenceNotary

        artifact = json.loads(path.read_text(encoding="utf-8"))
        record = AnchorRecord.from_dict(
            json.loads(anchor_path.read_text(encoding="utf-8"))
        )
        config = load_chain_config(
            network=record.network or None, contract_address=record.contract_address
        )
        compiled = load_or_compile()
        adapter, address, auto = build_adapter_with_contract(config, compiled)
        notary = EvidenceNotary(adapter, address if auto else record.contract_address)
        result = notary.reverify(artifact, record)
        payload = result.to_dict()
        payload["ephemeral_chain"] = auto
        return payload

    try:
        return await asyncio.to_thread(_verify)
    except Exception as exc:
        from tracelock.service.runner import _friendly_chain_error

        return JSONResponse(status_code=400, content={"error": _friendly_chain_error(exc)})


# Fields the tamper test can corrupt. Mirrors scripts/verify_anchor.py so the
# UI demonstration and the CLI demonstration alter exactly the same things.
TAMPER_TARGETS: dict[str, tuple[str | None, str, Any]] = {
    "trust_score": ("trust_score", "score", 99.99),
    "publishers": ("evidence", "independent_domains", ["fabricated.example"]),
    "run_id": (None, "run_id", "evidence_TAMPERED"),
}


@api.post("/chain/tamper-test")
async def tamper_test(artifact_path: str = Form(...), field: str = Form("trust_score")) -> Any:
    """Corrupt one field in a COPY and re-run the real integrity check.

    This is a genuine demonstration, not a canned message. The same
    `notary.reverify` that verifies an untouched artifact is run against a
    mutated copy, and the mismatch it reports is real. The stored evidence is
    never modified -- the copy exists only in memory for the length of this
    request.

    What makes the result interesting is the Merkle structure: the report says
    WHICH leaf diverged, so tampering is not merely detected but localised.
    """
    path = Path(artifact_path)
    anchor_path = path.with_suffix(".anchor.json")

    if not path.is_file():
        raise HTTPException(status_code=404, detail="Evidence artifact not found.")
    if not anchor_path.is_file():
        return JSONResponse(
            status_code=400,
            content={
                "error": "This evidence has not been anchored yet.",
                "hint": "Anchor it first, then run the tamper test.",
            },
        )

    def _run() -> dict[str, Any]:
        import copy as _copy

        from tracelock.chain.compiler import load_or_compile
        from tracelock.chain.config import (
            build_adapter_with_contract,
            load_chain_config,
        )
        from tracelock.chain.notary import AnchorRecord, EvidenceNotary

        original = json.loads(path.read_text(encoding="utf-8"))
        record = AnchorRecord.from_dict(
            json.loads(anchor_path.read_text(encoding="utf-8"))
        )

        # The corruption is applied to a deep copy. The file on disk is opened
        # read-only above and never written back.
        mutated = _copy.deepcopy(original)
        section, key, value = TAMPER_TARGETS.get(
            field, TAMPER_TARGETS["trust_score"]
        )
        if section is None:
            before = mutated.get(key)
            mutated[key] = value
        else:
            before = mutated.setdefault(section, {}).get(key)
            mutated[section][key] = value

        config = load_chain_config(
            network=record.network or None,
            contract_address=record.contract_address,
        )
        compiled = load_or_compile()
        adapter, address, auto = build_adapter_with_contract(config, compiled)
        notary = EvidenceNotary(adapter, address if auto else record.contract_address)

        clean = notary.reverify(original, record).to_dict()
        tampered = notary.reverify(mutated, record).to_dict()

        return {
            "field": key,
            "before": before,
            "after": value,
            "baseline": clean,
            "tampered": tampered,
            # The whole point: the untouched artifact still verifies, the
            # altered one does not. Both halves are needed for the claim to
            # mean anything.
            "detected": (
                clean.get("verdict") == "INTACT"
                and tampered.get("verdict") != "INTACT"
            ),
            "artifact_unchanged": True,
            "ephemeral_chain": auto,
        }

    try:
        return await asyncio.to_thread(_run)
    except Exception as exc:
        from tracelock.service.runner import _friendly_chain_error

        return JSONResponse(
            status_code=400, content={"error": _friendly_chain_error(exc)}
        )


# ==========================================================================
# Health / config
# ==========================================================================


@api.get("/resolver/info")
async def resolver_info() -> Any:
    """What the resolver supports -- including where it does NOT work.

    The matrix reports observed anonymous behaviour, not aspiration. A platform
    rated `auth_required` is still accepted and still attempted; we simply say
    so first, so a refusal is an expected outcome rather than a broken demo.
    """
    from tracelock.ingest import Method, capability_matrix
    from tracelock.service.url_resolver import SUPPORT_STATEMENT

    return {
        "support_statement": SUPPORT_STATEMENT,
        "methods": [
            {"key": m.value, "explanation": m.explanation}
            for m in Method if m is not Method.NONE
        ],
        "platforms": capability_matrix(),
        "note": (
            "Only metadata a site publishes to anonymous visitors is read. "
            "TRACELOCK sends no credentials, solves no CAPTCHA, and treats a "
            "platform refusal as a refusal."
        ),
    }


@api.post("/input/classify")
async def classify_input(url: str = Form(...)) -> Any:
    """Identify a pasted link instantly -- offline, no network request.

    Lets the UI say "Instagram post detected - usually requires sign-in"
    while the operator is still typing, so the expectation is set before any
    work starts rather than after it fails.
    """
    from tracelock.ingest import adapter_for, classify_url

    classification = classify_url(url)
    adapter = adapter_for(classification.platform)
    return {
        "classification": classification.to_dict(),
        "adapter": adapter.to_dict(),
        "needs_resolution": classification.input_type.needs_resolution,
    }


@api.get("/health")
async def health() -> Any:
    from tracelock.core.config import load_settings
    from tracelock.service.budget import Budget, Mode
    from tracelock.service.cache import cache_stats

    settings = load_settings()
    return {
        "status": "ok",
        "face_engine_loaded": ENGINE.ready,
        "face_engine_error": ENGINE.error,
        "search_configured": bool(settings.serpapi_api_key),
        "search_engine": settings.serp_engine,
        "calibrated": Path("data/calibration/model.json").is_file(),
        "stages": [{"key": k, "label": v} for k, v in STAGES],
        "modes": [
            {
                "key": m.value,
                "label": m.label,
                "description": m.description,
                "budget": Budget.for_mode(m).to_dict(),
            }
            for m in Mode
        ],
        "cache": cache_stats(),
    }


# ==========================================================================
# App
# ==========================================================================


def create_app() -> FastAPI:
    app = FastAPI(
        title="TRACELOCK",
        description="Digital Identity Evidence & Provenance Engine",
        version="1.0.0",
    )
    app.include_router(api)

    @app.on_event("startup")
    async def _warm_engine() -> None:
        # Load buffalo_l in the background so the 7s cold start is paid while
        # the operator is still choosing an image, not during their first run.
        from tracelock.service.runner import ENGINE

        ENGINE.warm()

    if WEB_DIR.is_dir():
        app.mount(
            "/static", StaticFiles(directory=str(WEB_DIR)), name="static"
        )

        @app.middleware("http")
        async def _no_stale_assets(request, call_next):
            """Never serve a stale app.js or stylesheet.

            The browser cached the bundle across an update and kept running the
            old code -- verified: the server had the fix, `typeof
            reconnectOrRecover` was still "undefined" in the page. During a
            demo that means a refresh can silently resurrect old behaviour, and
            the mismatch is invisible to whoever is presenting.

            These are a handful of local files on localhost, so there is
            nothing to gain from caching them and a working demo to lose.
            """
            response = await call_next(request)
            if request.url.path.startswith("/static/") or request.url.path == "/":
                response.headers["Cache-Control"] = "no-store, must-revalidate"
                response.headers["Pragma"] = "no-cache"
            return response

        @app.get("/", response_class=HTMLResponse)
        async def index() -> HTMLResponse:
            """Serve the page with cache-busted asset URLs.

            no-store headers alone were not enough: a browser that had already
            cached /static/app.js kept executing the old bundle after an
            update -- verified, the server had the fix while the page still
            reported `typeof reconnectOrRecover === "undefined"`.

            Stamping each asset with its own mtime means a changed file gets a
            changed URL, so a stale bundle is not merely discouraged, it is
            unreachable.
            """
            html = (WEB_DIR / "index.html").read_text(encoding="utf-8")

            for asset in ("app.js", "styles.css"):
                path = WEB_DIR / asset
                if path.is_file():
                    html = html.replace(
                        "/static/{0}".format(asset),
                        "/static/{0}?v={1}".format(asset, int(path.stat().st_mtime)),
                    )

            return HTMLResponse(
                html, headers={"Cache-Control": "no-store, must-revalidate"}
            )

    return app


app = create_app()
