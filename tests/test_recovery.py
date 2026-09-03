"""Structured failures, recovery options, and millisecond timings.

The rule under test: a failure is a RESULT, not a dead end and not a crash. It
says what could not be done, offers at least one way forward, and never carries
findings that could be mistaken for evidence.
"""

from __future__ import annotations

import sys

import pytest
from fastapi.testclient import TestClient

from tracelock.service import failures
from tracelock.service.failures import (
    DEFAULT_RECOVERY,
    RECOVERY_LABELS,
    STAGE_TIMEOUTS,
    FailureStatus,
    recovery_options,
)


@pytest.fixture
def client():
    from tracelock.api import create_app

    return TestClient(create_app())


def jpeg() -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (128, 128, 128)).save(buffer, format="JPEG")
    return buffer.getvalue()


# ==========================================================================
# Structured failures
# ==========================================================================


class TestStructuredFailures:
    def test_timeout_reports_stage_status_and_budget(self):
        failure = failures.timeout("candidate_fetch")
        payload = failure.to_dict()

        assert payload["status"] == "timeout"
        assert payload["stage"] == "candidate_fetch"
        assert payload["message"]
        assert payload["timeout_seconds"] == STAGE_TIMEOUTS["candidate_fetch"]

    def test_every_stage_has_a_bounded_timeout(self):
        """No stage may hang indefinitely."""
        for stage, seconds in STAGE_TIMEOUTS.items():
            assert 0 < seconds <= 120, stage

    def test_a_failure_never_carries_findings(self):
        """Nothing in a failure may be mistaken for evidence."""
        for failure in (
            failures.timeout("search"),
            failures.platform_blocked("Instagram", "requires sign-in"),
            failures.unavailable("url_resolution", "nothing published"),
            failures.blocked_target("url_resolution"),
        ):
            payload = failure.to_dict()
            for banned in (
                "trust_score", "score", "candidates", "verified",
                "similarity", "identity_probability", "image_url",
            ):
                assert banned not in payload, banned

    def test_blocked_platform_says_retrieval_failed_not_nonexistence(self):
        """'Could not be retrieved' and 'does not exist' are different claims.

        Only the first one is ours to make from a platform refusal.
        """
        message = failures.platform_blocked("Instagram", "requires sign-in").message
        lowered = message.lower()

        assert "could not retrieve" in lowered
        for forbidden in ("does not exist", "no image exists", "not found online"):
            assert forbidden not in lowered

    def test_ssrf_refusal_offers_no_retry(self):
        """A blocked private address should not invite trying again."""
        failure = failures.blocked_target("url_resolution")
        assert failure.status is FailureStatus.REFUSED
        assert "use_direct_url" not in failure.recovery

    def test_recovery_always_offers_at_least_one_way_forward(self):
        assert recovery_options(None)
        for failure in (
            failures.timeout("search"),
            failures.platform_blocked("Facebook", "requires sign-in"),
            failures.blocked_target("url_resolution"),
        ):
            options = recovery_options(failure)
            assert options
            assert all(o["label"] for o in options)

    def test_local_analysis_is_always_an_option_for_a_blocked_platform(self):
        """Leaving this out is what turns a blocked platform into a dead end."""
        options = recovery_options(failures.platform_blocked("Instagram", "x"))
        assert any(o["action"] == "continue_local" for o in options)

    def test_every_recovery_action_has_a_human_label(self):
        for action in DEFAULT_RECOVERY:
            assert RECOVERY_LABELS[action]


# ==========================================================================
# The API surfaces them
# ==========================================================================


class TestApiReturnsRecovery:
    def _patch(self, monkeypatch, *, reason, platform_url):
        app_module = sys.modules["tracelock.api.app"]
        from tracelock.ingest import adapter_for, classify_url
        from tracelock.ingest.resolve import Method, ResolvedInput

        def fake(url, **_):
            classification = classify_url(url)
            return ResolvedInput(
                input_url=url,
                classification=classification,
                adapter=adapter_for(classification.platform),
                method=Method.NONE,
                reason=reason,
            )

        monkeypatch.setattr(app_module, "resolve_input", fake)
        return platform_url

    def test_blocked_instagram_returns_recovery_options(self, client, monkeypatch):
        url = self._patch(
            monkeypatch,
            reason="Instagram requires authentication for most post pages.",
            platform_url="https://www.instagram.com/p/ABC/",
        )
        response = client.post("/api/input/url", data={"url": url})
        payload = response.json()

        assert response.status_code == 400
        assert payload["failure"]["status"] == "blocked"
        assert payload["failure"]["stage"] == "url_resolution"
        assert "Instagram" in payload["failure"]["message"]
        actions = {o["action"] for o in payload["recovery"]}
        assert actions == set(DEFAULT_RECOVERY)

    def test_no_resolved_image_url_is_invented_on_failure(self, client, monkeypatch):
        url = self._patch(
            monkeypatch, reason="nothing public",
            platform_url="https://www.instagram.com/p/ABC/",
        )
        payload = client.post("/api/input/url", data={"url": url}).json()

        assert "image_url" not in payload
        assert "resolved_image_url" not in payload
        assert "sha256" not in payload

    def test_ssrf_refusal_surfaces_as_a_refusal(self, client):
        response = client.post(
            "/api/input/url", data={"url": "http://169.254.169.254/latest/meta-data/"}
        )
        payload = response.json()

        assert response.status_code == 400
        assert payload["failure"]["status"] == "refused"
        assert "image_url" not in payload

    def test_link_url_failure_also_carries_recovery(self, client, monkeypatch):
        url = self._patch(
            monkeypatch, reason="Instagram requires authentication.",
            platform_url="https://www.instagram.com/p/ABC/",
        )
        selected = client.post(
            "/api/input/upload", files={"file": ("a.jpg", jpeg(), "image/jpeg")}
        ).json()
        payload = client.post("/api/input/link-url", data={
            "sha256": selected["sha256"], "url": url,
        }).json()

        assert payload["ok"] is False
        assert payload["failure"]["status"] == "blocked"
        assert payload["recovery"]
        assert "image_url" not in payload["resolution"]


# ==========================================================================
# Millisecond timings
# ==========================================================================


class TestMillisecondTimings:
    def test_timings_are_flat_integers_in_milliseconds(self):
        import time

        from tracelock.service.timing import PhaseTimer

        timer = PhaseTimer()
        with timer.measure("input_validation"):
            time.sleep(0.02)
        timer.finish()

        timings = timer.to_timings_ms()
        assert isinstance(timings["input_validation_ms"], int)
        assert timings["input_validation_ms"] >= 15
        assert "total_ms" in timings
        assert all(k.endswith("_ms") for k in timings)

    def test_a_phase_that_never_ran_is_absent_not_zero(self):
        """A zero would read as 'instant' for something that never happened."""
        from tracelock.service.timing import PhaseTimer

        timer = PhaseTimer()
        timer.start("never_finished")
        timer.finish()

        timings = timer.to_timings_ms()
        assert "never_finished_ms" not in timings
        assert "search_ms" not in timings


# ==========================================================================
# A slow server must not hang the pipeline
# ==========================================================================


class TestSlowServerCannotHang:
    """A server that trickles bytes forever must still be bounded.

    httpx's timeout is per socket operation, and `iter_bytes(chunk_size)`
    BLOCKS until a full chunk is buffered -- so a server sending 16 bytes a
    second satisfies every read timeout and never fills a chunk. Measured
    before the fix: still downloading after 120 seconds, size cap nowhere near
    reached. A wall-clock watchdog closes the response instead.
    """

    def _slow_server(self, port):
        import http.server
        import socketserver
        import threading
        import time as _time

        class Slow(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                for _ in range(60):
                    try:
                        self.wfile.write(bytes(16))
                        self.wfile.flush()
                    except Exception:
                        return
                    _time.sleep(0.5)

            def log_message(self, *a):
                pass

        server = socketserver.TCPServer(("127.0.0.1", port), Slow)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    def test_a_trickling_server_is_aborted_at_the_ceiling(self, monkeypatch):
        import time as _time

        import tracelock.acquisition.fetcher as fetcher
        from tracelock.acquisition.fetcher import FetchPolicy, fetch_media

        server = self._slow_server(8974)
        try:
            # The SSRF guard blocks loopback by design, so it is bypassed HERE
            # ONLY, to exercise the timeout path specifically.
            monkeypatch.setattr(fetcher, "is_blocked_target", lambda host: (False, ""))

            started = _time.perf_counter()
            result = fetch_media(
                "http://127.0.0.1:8974/hang.jpg",
                policy=FetchPolicy(timeout=3.0, max_duration=4.0),
            )
            elapsed = _time.perf_counter() - started
        finally:
            server.shutdown()

        assert elapsed < 15, "download was not bounded: {0:.1f}s".format(elapsed)
        assert not result.ok
        assert result.reason.value == "DOWNLOAD_TIMEOUT"
        # Partial bytes must never be treated as content.
        assert not result.content

    def test_the_policy_exposes_a_wall_clock_ceiling(self):
        from tracelock.acquisition.fetcher import FetchPolicy

        policy = FetchPolicy()
        assert policy.max_duration > 0
        assert "max_duration" in policy.to_dict()
