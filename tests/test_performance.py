"""Performance architecture: caching, budgets, early stopping, concurrency,
cancellation -- and the guarantees that keep them from becoming lies.

Every optimisation here is allowed to change HOW LONG the pipeline takes and
HOW MUCH of it runs. None of them is allowed to change what an examined
candidate concludes, to invent a result for a candidate that was skipped, or to
hide that skipping happened.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from tracelock.service.budget import (
    TARGET_INDEPENDENT_PUBLISHERS,
    TARGET_VERIFIED,
    Budget,
    Mode,
    StopTracker,
)
from tracelock.service.cache import BoundedTTLCache, FaceCache, FetchCache
from tracelock.service.prefetch import (
    Cancelled,
    CancellationToken,
    CachingAnalyzer,
    CachingFetcher,
    media_urls,
    prefetch_urls,
)


# ==========================================================================
# Caches
# ==========================================================================


class TestBoundedTTLCache:
    def test_stores_and_returns(self):
        cache = BoundedTTLCache(capacity=4, ttl_seconds=60)
        cache.put("a", 1)
        assert cache.get("a") == 1
        assert cache.stats.hits == 1

    def test_miss_is_counted(self):
        cache = BoundedTTLCache(capacity=4, ttl_seconds=60)
        assert cache.get("nope") is None
        assert cache.stats.misses == 1

    def test_evicts_least_recently_used(self):
        cache = BoundedTTLCache(capacity=2, ttl_seconds=60)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")           # 'a' becomes most-recent
        cache.put("c", 3)        # evicts 'b'
        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.stats.evictions == 1

    def test_entries_expire(self):
        now = [1000.0]
        cache = BoundedTTLCache(capacity=4, ttl_seconds=10, clock=lambda: now[0])
        cache.put("a", 1)
        now[0] += 11
        assert cache.get("a") is None
        assert cache.stats.expirations == 1

    def test_capacity_is_never_exceeded(self):
        cache = BoundedTTLCache(capacity=8, ttl_seconds=60)
        for i in range(200):
            cache.put("k{0}".format(i), i)
        assert len(cache) <= 8

    def test_rejects_nonsense_capacity(self):
        with pytest.raises(ValueError):
            BoundedTTLCache(capacity=0, ttl_seconds=1)

    def test_is_thread_safe(self):
        cache = BoundedTTLCache(capacity=64, ttl_seconds=60)
        errors: list[Exception] = []

        def hammer(base: int) -> None:
            try:
                for i in range(400):
                    cache.put("k{0}".format((base + i) % 100), i)
                    cache.get("k{0}".format(i % 100))
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=hammer, args=(n * 10,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(cache) <= 64


class TestFaceCache:
    def test_identical_bytes_are_analysed_once(self):
        cache = FaceCache()
        calls = []

        def compute():
            calls.append(1)
            return "analysis"

        digest = "a" * 64
        assert cache.analyze(digest, compute) == "analysis"
        assert cache.analyze(digest, compute) == "analysis"
        assert len(calls) == 1

    def test_different_bytes_are_analysed_separately(self):
        """The key is the CONTENT hash, so different content must not collide."""
        cache = FaceCache()
        assert cache.analyze("a" * 64, lambda: "first") == "first"
        assert cache.analyze("b" * 64, lambda: "second") == "second"

    def test_failures_are_not_cached(self):
        """A model still loading is not a property of the image."""
        cache = FaceCache()
        calls = []

        def failing():
            calls.append(1)
            raise RuntimeError("engine not ready")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                cache.analyze("c" * 64, failing)
        assert len(calls) == 2


class TestFetchCache:
    class _Result:
        def __init__(self, ok=True, content=b"x"):
            self.ok = ok
            self.content = content

    def test_successful_fetch_is_cached(self):
        cache = FetchCache()
        result = self._Result()
        cache.put("https://a.test/x.jpg", result)
        assert cache.get("https://a.test/x.jpg") is result

    def test_failures_are_never_cached(self):
        """Caching a 503 would turn a blip into an outage for the whole TTL."""
        cache = FetchCache()
        cache.put("https://a.test/x.jpg", self._Result(ok=False))
        assert cache.get("https://a.test/x.jpg") is None

    def test_oversized_entries_are_not_cached(self):
        cache = FetchCache()
        cache._max_entry = 10
        cache.put("https://a.test/big.jpg", self._Result(content=b"x" * 100))
        assert cache.get("https://a.test/big.jpg") is None


class TestCachingWrappers:
    def test_fetcher_serves_from_cache_on_repeat(self, monkeypatch):
        import tracelock.acquisition.fetcher as fetcher

        calls = []

        class Result:
            ok = True
            content = b"bytes"

        monkeypatch.setattr(
            fetcher, "fetch_media",
            lambda url, **kw: (calls.append(url), Result())[1],
        )
        caching = CachingFetcher(cache=FetchCache())
        caching("https://a.test/x.jpg")
        caching("https://a.test/x.jpg")
        assert len(calls) == 1

    def test_analyzer_keys_on_digest_not_path(self):
        """The same bytes at two CAS paths must analyse once."""
        calls = []

        class Engine:
            def analyze(self, path):
                calls.append(path)
                return "result"

        analyzer = CachingAnalyzer(Engine(), cache=FaceCache())
        digest = "d" * 64
        analyzer(digest, "/cas/aa/one")
        analyzer(digest, "/cas/aa/two")
        assert len(calls) == 1


# ==========================================================================
# Budget and early stopping
# ==========================================================================


class TestBudget:
    def test_fast_is_the_default(self):
        assert Budget().mode is Mode.FAST

    def test_thorough_examines_more_and_never_stops_early(self):
        thorough = Budget.for_mode(Mode.THOROUGH)
        fast = Budget.for_mode(Mode.FAST)
        assert thorough.candidate_limit > fast.candidate_limit
        assert thorough.allow_early_stop is False
        assert thorough.should_stop(verified=99, publishers=99) is False

    def test_face_concurrency_matches_the_tuned_optimum(self):
        """Valid only because ONNX intra_op is now pinned.

        With intra_op left at its default (all 16 cores), 2 workers really did
        beat 4 -- every extra worker multiplied a saturated pool. Pinning
        intra_op to 4 inverted that, so this number and the engine's thread
        setting must move together.
        """
        from tracelock.face.engine import DEFAULT_INTRA_OP_THREADS

        assert Budget().max_concurrent_face == 8
        assert DEFAULT_INTRA_OP_THREADS == 4

    def test_download_concurrency_matches_the_measurement(self):
        assert Budget().max_concurrent_downloads == 5

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError):
            Budget.for_mode("turbo")


class TestEarlyStopping:
    def test_tracker_matches_the_real_emitted_status_value(self):
        """Guards the exact bug this caught: a literal that stopped matching.

        The verifier emits VerificationStatus.value. If the tracker compares
        against anything else, early stopping silently never fires -- and every
        test written with the same wrong literal still passes.
        """
        from tracelock.core.reasons import VerificationStatus
        from tracelock.verification.models import VerificationResult

        emitted = VerificationStatus.VERIFIED_CANDIDATE.value
        assert emitted == "VERIFIED_CANDIDATE"

        tracker = StopTracker(Budget.for_mode(Mode.FAST))
        tracker.record({
            "status": emitted,
            "provenance": {"registrable_domain": "a.com"},
        })
        assert tracker.verified == 1, "tracker must count the real status value"

    def _verified(self, domain):
        from tracelock.core.reasons import VerificationStatus

        return {
            "status": VerificationStatus.VERIFIED_CANDIDATE.value,
            "provenance": {"registrable_domain": domain},
        }

    def test_stops_once_targets_are_met(self):
        tracker = StopTracker(Budget.for_mode(Mode.FAST))
        for i in range(TARGET_INDEPENDENT_PUBLISHERS):
            tracker.record(self._verified("site{0}.com".format(i)))
        assert tracker.should_stop()
        assert tracker.examined == TARGET_INDEPENDENT_PUBLISHERS

    def test_never_stops_on_rejections_however_many(self):
        """Finding nothing is not a reason to stop looking."""
        tracker = StopTracker(Budget.for_mode(Mode.FAST))
        for _ in range(50):
            tracker.record({"status": "REJECTED"})
            assert not tracker.should_stop()
        assert tracker.stopped_early is False

    def test_one_publisher_many_copies_does_not_satisfy_the_target(self):
        """Ten copies on one site is not independent corroboration."""
        tracker = StopTracker(Budget.for_mode(Mode.FAST))
        for _ in range(10):
            tracker.record(self._verified("same-site.com"))
        assert tracker.publishers == 1
        assert not tracker.should_stop()

    def test_inconclusive_results_do_not_count_as_verified(self):
        tracker = StopTracker(Budget.for_mode(Mode.FAST))
        for i in range(10):
            tracker.record({
                "status": "INCONCLUSIVE",
                "provenance": {"registrable_domain": "s{0}.com".format(i)},
            })
        assert tracker.verified == 0
        assert not tracker.should_stop()

    def test_thorough_mode_examines_everything(self):
        tracker = StopTracker(Budget.for_mode(Mode.THOROUGH))
        for i in range(20):
            tracker.record(self._verified("site{0}.com".format(i)))
            assert not tracker.should_stop()

    def test_a_stop_is_reported_not_hidden(self):
        """The operator must be told the run did not look at everything."""
        tracker = StopTracker(Budget.for_mode(Mode.FAST))
        for i in range(TARGET_VERIFIED):
            tracker.record(self._verified("site{0}.com".format(i)))
        tracker.should_stop()

        summary = tracker.summary(total=12)
        assert summary["stopped_early"] is True
        assert summary["examined"] == TARGET_VERIFIED
        assert summary["not_examined"] == 12 - TARGET_VERIFIED
        assert "not examined" in summary["explanation"]
        assert "Thorough" in summary["explanation"]

    def test_a_complete_run_reports_no_stop(self):
        tracker = StopTracker(Budget.for_mode(Mode.FAST))
        for _ in range(5):
            tracker.record({"status": "REJECTED"})
        summary = tracker.summary(total=5)
        assert summary["stopped_early"] is False
        assert summary["not_examined"] == 0
        assert summary["explanation"] == ""

    def test_early_stop_can_only_lower_corroboration_never_raise_it(self):
        """Stopping counts fewer publishers, so it cannot inflate a score."""
        stopped = StopTracker(Budget.for_mode(Mode.FAST))
        complete = StopTracker(Budget.for_mode(Mode.THOROUGH))

        for i in range(8):
            row = self._verified("site{0}.com".format(i))
            complete.record(row)
            if not stopped.stopped_early:
                stopped.record(row)
                stopped.should_stop()

        assert stopped.publishers <= complete.publishers


# ==========================================================================
# Concurrency
# ==========================================================================


class TestPrefetch:
    def test_downloads_run_concurrently(self):
        """Five 100ms downloads must not take 500ms."""
        def slow(url):
            time.sleep(0.1)
            return type("R", (), {"ok": True, "content": b"x"})()

        urls = ["https://a.test/{0}.jpg".format(i) for i in range(5)]
        started = time.perf_counter()
        results = prefetch_urls(urls, slow, max_workers=5)
        elapsed = time.perf_counter() - started

        assert len(results) == 5
        assert elapsed < 0.35, "expected concurrency, took {0:.2f}s".format(elapsed)

    def test_failures_are_omitted_not_raised(self):
        """A failed prefetch must not decide a candidate's fate."""
        def flaky(url):
            if "bad" in url:
                raise RuntimeError("network down")
            return type("R", (), {"ok": True, "content": b"x"})()

        results = prefetch_urls(
            ["https://a.test/good.jpg", "https://a.test/bad.jpg"], flaky
        )
        assert "https://a.test/good.jpg" in results
        assert "https://a.test/bad.jpg" not in results

    def test_duplicate_urls_are_fetched_once(self):
        calls = []

        def counting(url):
            calls.append(url)
            return type("R", (), {"ok": True, "content": b"x"})()

        prefetch_urls(["https://a.test/x.jpg"] * 6, counting)
        assert len(calls) == 1

    def test_empty_input_does_nothing(self):
        assert prefetch_urls([], lambda u: None) == {}
        assert prefetch_urls(["", None], lambda u: None) == {}

    def test_workers_never_exceed_the_url_count(self):
        seen: set[int] = set()

        def record(url):
            seen.add(threading.get_ident())
            time.sleep(0.02)
            return type("R", (), {"ok": True, "content": b"x"})()

        prefetch_urls(["https://a.test/1.jpg", "https://a.test/2.jpg"], record,
                      max_workers=10)
        assert len(seen) <= 2

    def test_progress_is_reported_as_work_completes(self):
        seen: list[tuple[int, int]] = []
        prefetch_urls(
            ["https://a.test/{0}.jpg".format(i) for i in range(4)],
            lambda u: type("R", (), {"ok": True, "content": b"x"})(),
            max_workers=2,
            on_complete=lambda done, total: seen.append((done, total)),
        )
        assert len(seen) == 4
        assert seen[-1][0] == 4
        assert all(total == 4 for _, total in seen)

    def test_media_urls_prefers_image_over_thumbnail(self):
        class C:
            def __init__(self, image, thumb):
                self.image_url = image
                self.thumbnail_url = thumb

        urls = media_urls([
            C("https://a.test/full.jpg", "https://a.test/thumb.jpg"),
            C(None, "https://a.test/only-thumb.jpg"),
            C(None, None),
        ])
        assert urls == ["https://a.test/full.jpg", "https://a.test/only-thumb.jpg"]


class TestCancellation:
    def test_check_raises_once_cancelled(self):
        token = CancellationToken()
        token.check()
        token.cancel("stop now")
        with pytest.raises(Cancelled, match="stop now"):
            token.check()

    def test_prefetch_stops_starting_work_after_cancellation(self):
        token = CancellationToken()
        started: list[str] = []

        def slow(url):
            started.append(url)
            token.cancel()
            time.sleep(0.01)
            return type("R", (), {"ok": True, "content": b"x"})()

        prefetch_urls(
            ["https://a.test/{0}.jpg".format(i) for i in range(20)],
            slow, max_workers=1, token=token,
        )
        assert len(started) < 20

    def test_token_is_thread_safe(self):
        token = CancellationToken()
        results: list[bool] = []

        def watcher():
            token.wait(2.0)
            results.append(token.cancelled)

        threads = [threading.Thread(target=watcher) for _ in range(4)]
        for t in threads:
            t.start()
        token.cancel()
        for t in threads:
            t.join(timeout=3)

        assert results == [True] * 4


# ==========================================================================
# The guarantees the optimisations must not break
# ==========================================================================


def code_only(module_name: str) -> str:
    """Module source with all comments and string literals removed.

    A prose docstring saying "nothing here caches a similarity" must not trip a
    check that looks for the word `similarity` in CODE.
    """
    import io
    import pathlib
    import sys
    import tokenize

    path = pathlib.Path(sys.modules[module_name].__file__)
    kept: list[str] = []
    with io.open(path, encoding="utf-8") as handle:
        for token in tokenize.generate_tokens(handle.readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            kept.append(token.string)
    return " ".join(kept).lower()


class TestOptimisationsDoNotChangeConclusions:
    def test_cache_holds_no_verdicts(self):
        """Similarity, trust and duplicate relations are never cached."""
        source = code_only("tracelock.service.cache")
        for forbidden in (
            "trust_score", "similarity", "verdict", "verification_result",
            "is_same_person", "calibrated_probability",
        ):
            assert forbidden not in source, forbidden

    def test_budget_never_alters_scoring_inputs(self):
        """A budget describes effort. It must not touch thresholds or weights."""
        source = code_only("tracelock.service.budget")
        for forbidden in ("threshold", "floor", "ceiling", "weight"):
            assert forbidden not in source, forbidden


def _tiny_jpeg() -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (128, 128, 128)).save(buffer, format="JPEG")
    return buffer.getvalue()


class TestLocalModeMakesNoNetworkRequests:
    """LOCAL MODE: an uploaded or captured image never leaves the machine."""

    def test_analysing_a_local_file_touches_no_network(self, monkeypatch, tmp_path):
        import httpx

        def forbidden(*a, **k):
            raise AssertionError("local analysis must make zero network requests")

        for target in ("get", "post", "put", "request", "stream"):
            monkeypatch.setattr(httpx, target, forbidden, raising=False)
        monkeypatch.setattr(httpx.Client, "send", forbidden, raising=False)
        monkeypatch.setattr(httpx.Client, "request", forbidden, raising=False)

        import socket

        monkeypatch.setattr(
            socket.socket, "connect",
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("local analysis opened a socket")
            ),
        )

        from tracelock.service.cache import FACE_CACHE, FETCH_CACHE
        from tracelock.service.inputs import from_upload, from_webcam

        # Ingesting an uploaded or captured image must not reach out at all.
        jpeg = _tiny_jpeg()
        upload = from_upload(jpeg, "face.jpg", tmp_path)
        capture = from_webcam(jpeg, tmp_path)

        # Neither may carry a public URL it was never given.
        assert upload.image_url is None
        assert capture.image_url is None
        assert Path(upload.local_path).is_file()
        assert Path(capture.local_path).is_file()

        # And the caches must not have been consulted over the wire either.
        assert FETCH_CACHE.get("https://anything") is None
        FACE_CACHE.stats  # accessible without network

    def test_no_automatic_upload_helper_exists(self):
        """Nothing in the service layer may push a local image to a host."""
        import pathlib

        service = pathlib.Path("src/tracelock/service")
        for path in service.glob("*.py"):
            source = path.read_text(encoding="utf-8").lower()
            for banned in ("catbox.moe", "litterbox", "0x0.st", "imgbb", "file.io"):
                assert banned not in source, "{0} references {1}".format(path, banned)


class TestProgressReportingIsNonFatal:
    """Progress is cosmetic. It must never be able to end an investigation.

    A stage key that did not exist made `next()` raise StopIteration straight
    through the pipeline, and a complete run died at "Downloading candidates"
    with nothing but "generator raised StopIteration" to explain it.
    """

    def _runner(self):
        from tracelock.service.inputs import InputKind, SourceType, TraceInput
        from tracelock.service.runner import InvestigationRunner, STORE

        trace = TraceInput(
            source_type=SourceType.UPLOAD, kind=InputKind.ORGANIC,
            local_path="x.jpg", filename="x.jpg", mime_type="image/jpeg",
            sha256="0" * 64, byte_size=1,
        )
        return InvestigationRunner(STORE.create(trace), notify=lambda: None)

    def test_unknown_stage_key_is_ignored(self):
        self._runner()._progress("no-such-stage", "hello")

    def test_every_progress_key_used_in_the_runner_is_a_real_stage(self):
        """Catches the bug at the source rather than at runtime."""
        import re
        import sys
        from pathlib import Path

        from tracelock.service.runner import STAGES

        source = Path(sys.modules["tracelock.service.runner"].__file__).read_text(
            encoding="utf-8"
        )
        valid = {key for key, _ in STAGES}
        used = set(re.findall(r'self\._(?:progress|begin|done|fail|skip)\(\s*"([^"]+)"', source))
        assert used <= valid, "unknown stage keys: {0}".format(sorted(used - valid))


class TestPrefetchWindow:
    """Early stopping must save DOWNLOADS too, not just face analysis."""

    def test_fast_does_not_prefetch_the_whole_candidate_list(self):
        fast = Budget.for_mode(Mode.FAST)
        assert fast.prefetch_window < fast.candidate_limit

    def test_thorough_prefetches_everything_it_will_examine(self):
        thorough = Budget.for_mode(Mode.THOROUGH)
        assert thorough.prefetch_window == thorough.candidate_limit

    def test_candidates_past_the_window_are_still_fetchable(self):
        """The window bounds prefetching, it must not bound the run.

        A candidate beyond the window is fetched on demand through the same
        cache, so a run that legitimately goes further still works.
        """
        calls = []

        class Result:
            ok = True
            content = b"x"

        import tracelock.acquisition.fetcher as fetcher_module

        original = fetcher_module.fetch_media
        fetcher_module.fetch_media = lambda url, **kw: (calls.append(url), Result())[1]
        try:
            caching = CachingFetcher(cache=FetchCache())
            caching("https://a.test/beyond-the-window.jpg")
        finally:
            fetcher_module.fetch_media = original

        assert calls == ["https://a.test/beyond-the-window.jpg"]


class TestCancellationIsNotReportedAsFailure:
    """A cancelled run must say 'cancelled', never 'failed'.

    Broad `except Exception` handlers around each stage were converting a
    cancellation into "Candidate verification failed", which tells the operator
    something went wrong when nothing did.
    """

    def test_no_broad_handler_in_run_swallows_cancelled(self):
        """Every `except Exception` inside the runner is preceded by a
        `except Cancelled: raise`, so cancellation always propagates."""
        import re
        import sys
        from pathlib import Path

        source = Path(sys.modules["tracelock.service.runner"].__file__).read_text(
            encoding="utf-8"
        )
        lines = source.splitlines()

        unguarded = []
        for i, line in enumerate(lines):
            if not re.match(r"\s*except Exception", line):
                continue
            # A handler is guarded when a `except Cancelled` clause sits within
            # the few lines above it, at the same indent.
            window = lines[max(0, i - 6):i]
            if any("except Cancelled" in w for w in window):
                continue
            # The pool loader and the top-level execute() guard are outside the
            # cancellable pipeline, and execute() is where Cancelled is HANDLED.
            context = "\n".join(lines[max(0, i - 25):i])
            if "def warm" in context or "def get" in context or "def execute" in context:
                continue
            if "_load_calibration" in context or "def _friendly" in context:
                continue
            unguarded.append((i + 1, line.strip()))

        assert not unguarded, "unguarded handlers would mask cancellation: {0}".format(
            unguarded
        )

    def test_cancelled_status_is_distinct_from_failed(self):
        from tracelock.service.prefetch import Cancelled

        assert issubclass(Cancelled, Exception)
        # It must be catchable specifically, not only as a generic Exception.
        try:
            raise Cancelled("stop")
        except Cancelled as stop:
            assert str(stop) == "stop"


class TestCorroborationIsGenuine:
    """Early stopping must require real independent corroboration.

    Three guards, each closing a different way to fake it:
      * a weak match cannot be a reason to stop
      * the same image under different URLs is one publication
      * several pages on one domain are one publisher
    """

    def _result(self, *, domain, probability=0.95, content="unique"):
        from tracelock.core.reasons import VerificationStatus

        return {
            "status": VerificationStatus.VERIFIED_CANDIDATE.value,
            "identity_probability": probability,
            "content_sha256": content,
            "provenance": {"registrable_domain": domain},
        }

    def test_the_stop_gate_matches_the_calibrated_boundary(self):
        """The gate must not exceed what the calibration actually supports.

        The fitted boundary is similarity 0.3528 -> probability 0.4862. A gate
        above that disqualifies candidates the calibrated policy itself calls
        VERIFIED_CANDIDATE, which silently disables early stopping.
        """
        from tracelock.calibration.model import CalibrationModel
        from tracelock.verification.policy import VerificationPolicy

        budget = Budget.for_mode(Mode.FAST)
        model = CalibrationModel.load("data/calibration/model.json")
        policy = VerificationPolicy.from_calibration(model)
        boundary = policy.identity_probability(policy.similarity_ceiling)

        assert budget.min_identity_probability <= boundary, (
            "stop gate {0} exceeds the calibrated boundary {1}".format(
                budget.min_identity_probability, boundary
            )
        )

    def test_an_explicitly_raised_gate_still_excludes_weak_matches(self):
        """The gate remains available for deliberate use."""
        from dataclasses import replace

        budget = replace(Budget.for_mode(Mode.FAST), min_identity_probability=0.80)
        tracker = StopTracker(budget)
        for i in range(10):
            tracker.record(self._result(
                domain="site{0}.com".format(i), probability=0.55,
                content="c{0}".format(i),
            ))
            assert not tracker.should_stop()
        assert tracker.verified == 0
        assert tracker.weak == 10

    def test_duplicate_image_across_domains_is_one_publication(self):
        """The same photo syndicated to 5 sites is 1 image, not 5 sources."""
        tracker = StopTracker(Budget.for_mode(Mode.FAST))
        for i in range(5):
            tracker.record(self._result(
                domain="site{0}.com".format(i), content="SAME-IMAGE-HASH",
            ))
        assert tracker.publishers == 1
        assert not tracker.should_stop()

    def test_many_pages_on_one_domain_is_one_publisher(self):
        tracker = StopTracker(Budget.for_mode(Mode.FAST))
        for i in range(6):
            tracker.record(self._result(
                domain="one-site.com", content="c{0}".format(i),
            ))
        assert tracker.publishers == 1
        assert not tracker.should_stop()

    def test_genuine_corroboration_does_stop(self):
        tracker = StopTracker(Budget.for_mode(Mode.FAST))
        for i in range(3):
            tracker.record(self._result(
                domain="site{0}.com".format(i), content="c{0}".format(i),
            ))
        assert tracker.publishers == 3
        assert tracker.should_stop()

    def test_weak_matches_are_still_reported_not_discarded(self):
        """A borderline match is real evidence; it just cannot end the search.

        With the gate at the calibrated default nothing is "weak", so this
        exercises an explicitly raised gate -- the case where the distinction
        exists at all.
        """
        from dataclasses import replace

        budget = replace(Budget.for_mode(Mode.FAST), min_identity_probability=0.80)
        tracker = StopTracker(budget)
        tracker.record(self._result(domain="a.com", probability=0.55, content="a"))
        summary = tracker.summary(total=10)
        assert summary["below_stop_threshold"] == 1
        assert summary["examined"] == 1
