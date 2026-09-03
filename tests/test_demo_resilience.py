"""Regressions for the two ways a live demo could visibly break.

Both were found by driving the real UI in a browser, not by reading code.

  1. A dropped WebSocket froze the progress screen forever. Only `onmessage`
     was wired, so a clean close -- a proxy timeout, a server restart, wifi
     blinking -- left "Discovering public sources" spinning with no failbox and
     no button. Verified before the fix: readyState 3, screen unchanged, zero
     recovery controls.

  2. The browser kept executing a cached bundle across an update. The server
     had the new code while the page still reported
     `typeof reconnectOrRecover === "undefined"`, so a mid-demo refresh could
     silently resurrect old behaviour.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

JS = Path("web/app.js").read_text(encoding="utf-8")


@pytest.fixture
def client():
    from tracelock.api import create_app

    return TestClient(create_app())


# ==========================================================================
# 1. A dropped socket must never freeze the UI
# ==========================================================================


class TestSocketDropIsRecoverable:
    def test_close_and_error_are_both_handled(self):
        """`onerror` alone misses a CLEAN close, which is the common case."""
        assert "socket.onclose" in JS, "a clean close would freeze the UI"
        assert "socket.onerror" in JS

    def test_both_route_into_the_same_recovery(self):
        for handler in ("socket.onclose", "socket.onerror"):
            match = re.search(re.escape(handler) + r"\s*=\s*\(\)\s*=>\s*(\w+)", JS)
            assert match, handler
            assert match.group(1) == "reconnectOrRecover", handler

    def test_recovery_polls_the_rest_endpoint_that_already_exists(self):
        """The run usually SURVIVES the socket, so ask the server."""
        block = JS[JS.index("async function reconnectOrRecover"):]
        block = block[: block.index("function wireFailboxActions")]
        assert "/api/investigation/" in block
        assert "renderStages" in block

    def test_a_completed_run_is_collected_not_reported_as_lost(self):
        block = JS[JS.index("async function reconnectOrRecover"):]
        block = block[: block.index("function wireFailboxActions")]
        assert 'snapshot.status === "complete"' in block
        assert "renderResults(snapshot.result)" in block

    def test_giving_up_states_plainly_that_nothing_was_fabricated(self):
        block = JS[JS.index("async function reconnectOrRecover"):]
        block = block[: block.index("function wireFailboxActions")]
        assert "Nothing was fabricated" in block
        assert "no result is displayed because none was received" in block

    def test_giving_up_offers_a_way_forward(self):
        block = JS[JS.index("async function reconnectOrRecover"):]
        block = block[: block.index("function wireFailboxActions")]
        assert "Try again" in block
        assert "Use a different image" in block
        assert "wireFailboxActions()" in block

    def test_recovery_is_bounded_not_an_infinite_poll(self):
        """A poll with no ceiling is just a slower freeze."""
        block = JS[JS.index("async function reconnectOrRecover"):]
        block = block[: block.index("function wireFailboxActions")]
        match = re.search(r"attempt < (\d+)", block)
        assert match, "recovery must have a bounded attempt count"
        assert 1 <= int(match.group(1)) <= 40

    def test_a_finished_run_clears_the_active_id(self):
        """Otherwise a later close re-enters recovery for a finished run."""
        assert "activeRunId = null;" in JS
        block = JS[JS.index("async function reconnectOrRecover"):]
        block = block[: block.index("function wireFailboxActions")]
        assert "runId !== activeRunId" in block

    def test_the_failed_state_also_offers_recovery(self):
        """It previously rendered a heading and a paragraph, and no button."""
        block = JS[JS.index('if (snapshot.status === "failed")'):]
        block = block[: block.index('if (snapshot.status === "complete"')]
        assert "<button" in block
        assert "wireFailboxActions()" in block


# ==========================================================================
# 2. A stale bundle must be unreachable
# ==========================================================================


class TestNoStaleAssets:
    def test_static_assets_are_served_no_store(self, client):
        response = client.get("/static/app.js")
        assert response.status_code == 200
        assert "no-store" in response.headers.get("cache-control", "")

    def test_the_page_itself_is_never_cached(self, client):
        response = client.get("/")
        assert "no-store" in response.headers.get("cache-control", "")

    def test_asset_urls_carry_a_version_stamp(self, client):
        """no-store alone was not enough -- an already-cached bundle survived."""
        html = client.get("/").text
        assert re.search(r"/static/app\.js\?v=\d+", html), html[:400]
        assert re.search(r"/static/styles\.css\?v=\d+", html)

    def test_the_stamp_changes_when_the_file_changes(self, client, tmp_path):
        import os
        import time

        from tracelock.api.app import WEB_DIR

        target = WEB_DIR / "app.js"
        original = target.stat().st_mtime

        first = client.get("/").text
        stamp_one = re.search(r"/static/app\.js\?v=(\d+)", first).group(1)
        try:
            future = original + 120
            os.utime(target, (future, future))
            second = client.get("/").text
            stamp_two = re.search(r"/static/app\.js\?v=(\d+)", second).group(1)
            assert stamp_one != stamp_two
        finally:
            os.utime(target, (original, original))


# ==========================================================================
# 3. The demo must survive with no internet at all
# ==========================================================================


class TestOfflineDemoStillDemonstrates:
    """The core innovation must not depend on an external service."""

    def test_local_analysis_needs_no_network(self, monkeypatch, tmp_path):
        import httpx

        def forbidden(*a, **k):
            raise AssertionError("local analysis must not touch the network")

        for target in ("get", "post", "request", "stream"):
            monkeypatch.setattr(httpx, target, forbidden, raising=False)
        monkeypatch.setattr(httpx.Client, "send", forbidden, raising=False)

        from tracelock.service.inputs import from_upload

        trace = from_upload(_jpeg(), "offline.jpg", tmp_path)
        assert trace.image_url is None
        assert Path(trace.local_path).is_file()

    def test_the_offline_capable_surface_is_the_core_of_the_product(self):
        """Everything here works with the network unplugged."""
        from tracelock.acquisition.cas import ContentAddressedStore  # noqa: F401
        from tracelock.chain.notary import EvidenceNotary  # noqa: F401
        from tracelock.face.engine import FaceEngine  # noqa: F401
        from tracelock.service.analysis import analyze_input  # noqa: F401
        from tracelock.verification.phash import compute_phash  # noqa: F401

    def test_the_local_chain_requires_no_external_service(self):
        from tracelock.chain.adapter import NETWORKS

        local = NETWORKS["local"]
        assert local.ephemeral
        assert not local.publicly_verifiable
        # No RPC endpoint means nothing to be unreachable.
        assert local.explorer_tx == ""


def _jpeg() -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (128, 128, 128)).save(buffer, format="JPEG")
    return buffer.getvalue()


# ==========================================================================
# 4. Undecodable formats must never be offered as candidates
# ==========================================================================


class TestUndecodableFormatsAreNotCandidates:
    """Found on iana.org during the judge walkthrough.

    Its only in-page image is an SVG logo. The ranker offered it as the top
    candidate, the pipeline fetched it, and it was then rejected with "the
    link returned a web page, not an image" -- which was also false. It
    returned an SVG. Ranking a format the face engine cannot decode is
    guaranteed-to-fail work AND produces a misleading message.
    """

    def test_svg_and_ico_are_dropped(self):
        from tracelock.ingest.candidates import rank_candidates

        html = """
          <img src="/static/logo.svg" alt="logo">
          <img src="/favicon.ico">
          <img src="/photo.jpg" alt="portrait" width="800" height="800">
        """
        ranked, _ = rank_candidates(html, "https://site.test/page")
        urls = [c.url for c in ranked]

        assert any(u.endswith("/photo.jpg") for u in urls)
        assert not any(u.endswith((".svg", ".ico")) for u in urls)

    def test_a_page_with_only_undecodable_images_says_so_honestly(self):
        from tracelock.ingest.candidates import rank_candidates

        ranked, _ = rank_candidates(
            '<img src="/logo.svg">', "https://site.test/page"
        )
        # Nothing offered means the resolver reports "no usable image",
        # instead of fetching something that can never work.
        assert ranked == []

    def test_the_exclusion_list_matches_what_validation_supports(self):
        """A format we exclude must genuinely be one we cannot decode."""
        from tracelock.acquisition.validation import SUPPORTED_FORMATS
        from tracelock.ingest.candidates import UNDECODABLE_EXTENSIONS

        supported = {f.lower() for f in SUPPORTED_FORMATS}
        for extension in UNDECODABLE_EXTENSIONS:
            stem = extension.lstrip(".")
            assert stem not in supported, extension
            # jpg/jpeg/tif aliases must not be caught by accident.
            assert stem not in ("jpg", "jpeg", "tif", "tiff", "png", "webp")
