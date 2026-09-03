"""Concurrent multi-source discovery: speed, isolation, and honest merging.

The guarantees under test are as much forensic as they are about speed. Merging
two indexes must never manufacture corroboration: the same image found twice is
one candidate found by two engines, not two pieces of evidence.
"""

from __future__ import annotations

import threading
import time

import pytest

from tracelock.core.models import Candidate, ProbeRef
from tracelock.discovery.multi import (
    DiscoveryOutcome,
    MultiSourceDiscovery,
    SourceOutcome,
    _canonical_url,
    build_providers,
)


class FakeResult:
    def __init__(self, candidates):
        self.candidates = candidates
        self.raw_response = {"fake": True}
        self.query = {"engine": "fake"}


class FakeProvider:
    """A search source with controllable latency and failure behaviour."""

    def __init__(self, name, engine, candidates=(), *, delay=0.0, error=None):
        self.name = name
        self.engine = engine
        self._candidates = list(candidates)
        self._delay = delay
        self._error = error
        self.calls = 0
        self.started_at = None

    def search(self, probe, *, limit=20):
        self.started_at = time.perf_counter()
        self.calls += 1
        if self._delay:
            time.sleep(self._delay)
        if self._error:
            raise self._error
        return FakeResult(self._candidates[:limit])


def candidate(url, *, provider="fake", post=None):
    return Candidate(
        provider=provider, image_url=url,
        post_url=post or "https://{0}/page".format(url.split("/")[2]),
    )


PROBE = ProbeRef(local_path="x.jpg", sha256="a" * 64, public_url="https://t.test/x.jpg")


# ==========================================================================
# Concurrency
# ==========================================================================


class TestConcurrency:
    def test_sources_run_concurrently_not_one_after_another(self):
        """Four 0.3s sources must finish in well under 1.2s."""
        providers = [
            FakeProvider("p{0}".format(i), "engine{0}".format(i),
                         [candidate("https://cdn{0}.test/a.jpg".format(i))],
                         delay=0.3)
            for i in range(4)
        ]
        discovery = MultiSourceDiscovery(providers, max_workers=4)

        started = time.perf_counter()
        outcome = discovery.search(PROBE)
        elapsed = time.perf_counter() - started

        assert len(outcome.candidates) == 4
        assert elapsed < 0.9, "ran sequentially: {0:.2f}s".format(elapsed)

    def test_all_sources_start_before_any_finishes(self):
        """Proves genuine overlap, not merely a fast sequential run."""
        starts = []
        lock = threading.Lock()

        class Recording(FakeProvider):
            def search(self, probe, *, limit=20):
                with lock:
                    starts.append(time.perf_counter())
                time.sleep(0.25)
                return FakeResult([candidate("https://cdn.test/{0}.jpg".format(self.engine))])

        providers = [Recording("p{0}".format(i), "e{0}".format(i)) for i in range(3)]
        MultiSourceDiscovery(providers, max_workers=3).search(PROBE)

        assert len(starts) == 3
        # All three started within a fraction of the 0.25s each one takes.
        assert max(starts) - min(starts) < 0.15


# ==========================================================================
# Isolation, failure resilience, timeouts
# ==========================================================================


class TestSlowSourceIsolation:
    def test_a_slow_source_does_not_hold_back_a_fast_one(self):
        fast = FakeProvider("fast", "fast_engine",
                            [candidate("https://cdn.test/fast.jpg")], delay=0.05)
        slow = FakeProvider("slow", "slow_engine",
                            [candidate("https://cdn.test/slow.jpg")], delay=5.0)

        seen: list[tuple[str, float]] = []
        started = time.perf_counter()

        discovery = MultiSourceDiscovery(
            [slow, fast], source_timeout=0.6, global_timeout=0.8,
        )
        discovery.search(
            PROBE,
            on_candidates=lambda fresh, source: seen.append(
                (source.engine, time.perf_counter() - started)
            ),
        )

        # The fast source streamed its result long before the slow one's budget.
        assert any(engine == "fast_engine" for engine, _ in seen)
        fast_at = next(t for engine, t in seen if engine == "fast_engine")
        assert fast_at < 0.5

    def test_a_hanging_source_is_abandoned_at_the_budget(self):
        hang = FakeProvider("hang", "hanging", [], delay=10.0)
        quick = FakeProvider("quick", "quick", [candidate("https://cdn.test/q.jpg")])

        started = time.perf_counter()
        outcome = MultiSourceDiscovery(
            [hang, quick], source_timeout=0.4, global_timeout=0.6,
        ).search(PROBE)
        elapsed = time.perf_counter() - started

        assert elapsed < 2.0, "did not abandon the hanging source"
        assert outcome.any_source_succeeded
        assert any(s.timed_out for s in outcome.sources)

    def test_one_failing_source_does_not_fail_the_investigation(self):
        broken = FakeProvider("broken", "broken", error=RuntimeError("index down"))
        working = FakeProvider("ok", "ok", [candidate("https://cdn.test/a.jpg")])

        outcome = MultiSourceDiscovery([broken, working]).search(PROBE)

        assert outcome.any_source_succeeded
        assert len(outcome.candidates) == 1
        failed = [s for s in outcome.sources if not s.ok]
        assert len(failed) == 1
        assert "index down" in failed[0].error

    def test_every_source_failing_is_reported_with_reasons(self):
        """An empty result from broken indexes is not 'nothing exists'."""
        providers = [
            FakeProvider("a", "alpha", error=RuntimeError("boom")),
            FakeProvider("b", "beta", error=ValueError("bad schema")),
        ]
        outcome = MultiSourceDiscovery(providers).search(PROBE)

        assert not outcome.any_source_succeeded
        summary = outcome.failure_summary
        assert "alpha" in summary and "beta" in summary

    def test_a_failing_stream_consumer_cannot_kill_a_source(self):
        provider = FakeProvider("p", "e", [candidate("https://cdn.test/a.jpg")])

        def hostile(fresh, source):
            raise RuntimeError("consumer exploded")

        outcome = MultiSourceDiscovery([provider]).search(
            PROBE, on_candidates=hostile
        )
        assert outcome.any_source_succeeded
        assert len(outcome.candidates) == 1

    def test_no_providers_yields_an_empty_outcome_not_a_crash(self):
        outcome = MultiSourceDiscovery([]).search(PROBE)
        assert outcome.candidates == []
        assert not outcome.any_source_succeeded


# ==========================================================================
# Streaming / progressive processing
# ==========================================================================


class TestProgressiveProcessing:
    def test_candidates_stream_before_all_sources_complete(self):
        fast = FakeProvider("fast", "fast", [candidate("https://cdn.test/1.jpg")],
                            delay=0.05)
        slow = FakeProvider("slow", "slow", [candidate("https://cdn.test/2.jpg")],
                            delay=0.6)

        events: list[str] = []
        done = threading.Event()

        def on_candidates(fresh, source):
            events.append(source.engine)
            if source.engine == "fast":
                done.set()

        discovery = MultiSourceDiscovery([slow, fast], global_timeout=3.0)
        thread = threading.Thread(
            target=lambda: discovery.search(PROBE, on_candidates=on_candidates)
        )
        thread.start()

        # The fast source's candidates arrive while the slow one is still going.
        assert done.wait(timeout=0.4), "fast source did not stream early"
        assert events == ["fast"]
        thread.join(timeout=3)

    def test_streamed_batches_contain_only_new_candidates(self):
        shared = "https://cdn.test/same.jpg"
        first = FakeProvider("a", "alpha", [candidate(shared)], delay=0.02)
        second = FakeProvider("b", "beta", [candidate(shared)], delay=0.25)

        batches: list[int] = []
        MultiSourceDiscovery([first, second], global_timeout=3.0).search(
            PROBE, on_candidates=lambda fresh, source: batches.append(len(fresh))
        )

        # The second source's only candidate was already known, so it streams
        # nothing -- it must not be re-offered for processing.
        assert batches == [1]


# ==========================================================================
# URL-level deduplication
# ==========================================================================


class TestUrlDeduplication:
    def test_identical_urls_from_two_sources_are_one_candidate(self):
        url = "https://cdn.test/photo.jpg"
        outcome = MultiSourceDiscovery([
            FakeProvider("a", "alpha", [candidate(url)]),
            FakeProvider("b", "beta", [candidate(url)]),
        ]).search(PROBE)

        assert len(outcome.candidates) == 1

    def test_both_engines_are_retained_for_provenance(self):
        """One candidate, two discoverers -- that is real information."""
        url = "https://cdn.test/photo.jpg"
        discovery = MultiSourceDiscovery([
            FakeProvider("a", "alpha", [candidate(url)]),
            FakeProvider("b", "beta", [candidate(url)]),
        ])
        outcome = discovery.search(PROBE)

        engines = discovery.found_by(outcome.candidates[0])
        assert sorted(engines) == ["alpha", "beta"]

    @pytest.mark.parametrize(
        "a,b",
        [
            ("https://cdn.test/a.jpg", "https://cdn.test/a.jpg?utm_source=x"),
            ("https://cdn.test/a.jpg", "https://www.cdn.test/a.jpg"),
            ("https://cdn.test/a.jpg", "HTTPS://CDN.TEST/a.jpg"),
            ("https://cdn.test/a.jpg", "https://cdn.test/a.jpg#frag"),
            ("https://cdn.test/a.jpg", "https://cdn.test/a.jpg?fbclid=123"),
        ],
    )
    def test_tracking_noise_is_normalised_away(self, a, b):
        assert _canonical_url(a) == _canonical_url(b)

    @pytest.mark.parametrize(
        "a,b",
        [
            # A resizing endpoint: different sizes are DIFFERENT images.
            ("https://cdn.test/i?w=200", "https://cdn.test/i?w=800"),
            ("https://cdn.test/a.jpg", "https://cdn.test/b.jpg"),
            ("https://one.test/a.jpg", "https://two.test/a.jpg"),
        ],
    )
    def test_genuinely_different_urls_stay_separate(self, a, b):
        """Over-normalising would silently discard real evidence."""
        assert _canonical_url(a) != _canonical_url(b)

    def test_a_candidate_with_no_usable_url_is_dropped(self):
        empty = Candidate(provider="p")
        outcome = MultiSourceDiscovery([
            FakeProvider("a", "alpha", [empty])
        ]).search(PROBE)
        assert outcome.candidates == []


# ==========================================================================
# Provider construction
# ==========================================================================


class TestProviderConstruction:
    class _Settings:
        serpapi_api_key = "test-key"
        serp_engine = "google_lens"
        http_timeout = 30.0

    def test_multi_source_builds_every_supported_engine(self):
        from tracelock.discovery.serpapi_lens import SUPPORTED_ENGINES

        providers = build_providers(self._Settings(), multi_source=True)
        assert len(providers) == len(SUPPORTED_ENGINES)
        assert {p.engine for p in providers} == set(SUPPORTED_ENGINES)

    def test_the_configured_engine_is_queried_first(self):
        providers = build_providers(self._Settings(), multi_source=True)
        assert providers[0].engine == "google_lens"

    def test_single_source_preserves_the_old_behaviour_exactly(self):
        providers = build_providers(self._Settings(), multi_source=False)
        assert len(providers) == 1
        assert providers[0].engine == "google_lens"
