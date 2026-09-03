"""Bounded-concurrency prefetch and cancellation.

Downloads are 30.1% of per-candidate cost and measured 2.86x faster across 5
workers, so they are fetched ahead of the verification loop rather than one at
a time inside it. Verification itself stays strictly sequential, which is the
point: duplicate detection compares each candidate against the ones before it,
so its result must not depend on which thread finished first. A forensic tool
that reorders its own findings between identical runs is not trustworthy, and
2.86x is not worth that.

So the split is:

    parallel    downloading bytes      (no shared state, order irrelevant)
    sequential  verifying candidates   (order-dependent, must be deterministic)

Cancellation is cooperative. `CancellationToken` is checked between candidates
and before each download; nothing is killed mid-write, so the CAS never sees a
partial blob. A cancelled run reports itself as cancelled -- it never reports
partial findings as if they were complete.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, Sequence


class Cancelled(Exception):
    """Raised at a cancellation checkpoint when a run has been cancelled."""


class CancellationToken:
    """A thread-safe 'stop what you are doing' flag."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason = ""

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        return self._reason

    def cancel(self, reason: str = "Cancelled by the operator.") -> None:
        self._reason = reason
        self._event.set()

    def check(self) -> None:
        """Raise if cancelled. Call at safe points, never mid-write."""
        if self._event.is_set():
            raise Cancelled(self._reason or "Cancelled.")

    def wait(self, timeout: float) -> bool:
        return self._event.wait(timeout)


class NullToken(CancellationToken):
    """A token that is never cancelled, for callers that do not need one."""

    def cancel(self, reason: str = "") -> None:  # pragma: no cover - inert
        return


def prefetch_urls(
    urls: Sequence[str],
    fetch: Callable[[str], Any],
    *,
    max_workers: int = 5,
    token: CancellationToken | None = None,
    on_complete: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Download many URLs concurrently. Returns {url: result} for successes.

    Failures are omitted rather than raised: a candidate whose download fails
    is a normal outcome that the verifier will record with a proper rejection
    reason when it reaches that candidate. Prefetching is an optimisation, so
    it must never be the thing that decides a candidate's fate.
    """
    token = token or NullToken()
    unique = list(dict.fromkeys(u for u in urls if u))
    if not unique:
        return {}

    results: dict[str, Any] = {}
    lock = threading.Lock()
    done = 0

    def worker(url: str) -> None:
        nonlocal done
        if token.cancelled:
            return
        try:
            result = fetch(url)
        except Exception:
            result = None
        with lock:
            done += 1
            if result is not None and getattr(result, "ok", False):
                results[url] = result
            if on_complete:
                on_complete(done, len(unique))

    # `max_workers` is capped at the number of URLs so a two-candidate run does
    # not spin up five idle threads.
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(unique)))) as pool:
        list(pool.map(worker, unique))

    return results


class CachingFetcher:
    """A `fetch_media` stand-in backed by the shared download cache.

    Used both by the prefetcher and by the verification loop, so a URL fetched
    ahead of time is a cache hit when the verifier asks for it moments later.
    The verifier does not know it is being fed a cache; it calls one callable
    and receives an `AcquisitionResult` exactly as before.
    """

    def __init__(self, policy=None, *, cache=None, client=None) -> None:
        from tracelock.acquisition.fetcher import FetchPolicy
        from tracelock.service.cache import FETCH_CACHE

        self.policy = policy or FetchPolicy()
        self.cache = cache if cache is not None else FETCH_CACHE
        self.client = client

    def __call__(self, url: str) -> Any:
        from tracelock.acquisition.fetcher import fetch_media

        cached = self.cache.get(url)
        if cached is not None:
            return cached

        result = fetch_media(url, policy=self.policy, client=self.client)
        self.cache.put(url, result)
        return result


class CachingAnalyzer:
    """Face analysis keyed by the sha256 of the bytes, not the path.

    Content addressing is what makes this correct: identical bytes have
    identical analysis, by definition of a deterministic model. Reusing the
    result is not an approximation.
    """

    def __init__(self, engine, *, cache=None) -> None:
        from tracelock.service.cache import FACE_CACHE

        self.engine = engine
        self.cache = cache if cache is not None else FACE_CACHE

    def __call__(self, digest: str, path: str) -> Any:
        return self.cache.analyze(digest, lambda: self.engine.analyze(path))


def media_urls(candidates: Iterable[Any]) -> list[str]:
    """The URL each candidate will actually be fetched from.

    Mirrors the verifier's own choice -- image_url first, thumbnail as
    fallback -- so the prefetch warms exactly the URLs that get requested. If
    these ever diverge the cache simply misses; it cannot cause a wrong fetch.
    """
    urls: list[str] = []
    for candidate in candidates:
        url = getattr(candidate, "image_url", None) or getattr(
            candidate, "thumbnail_url", None
        )
        if url:
            urls.append(url)
    return urls


def prewarm_faces(
    candidates: Sequence[Any],
    fetch: Callable[[str], Any],
    engine,
    *,
    max_workers: int = 2,
    token: CancellationToken | None = None,
    cache=None,
    on_complete: Callable[[int, int], None] | None = None,
) -> int:
    """Populate the face cache concurrently, ahead of the verification loop.

    Face analysis is the single most expensive step (~1.7s per image) and the
    verification loop must stay sequential for deterministic duplicate
    detection. Those two facts look contradictory until you notice that face
    analysis is a PURE function of the image bytes: computing it early, on
    another thread, cannot change what it returns.

    So the analysis is done here in parallel and stored under the sha256 of the
    bytes -- exactly the key the verifier's cached analyzer looks up -- and the
    sequential loop then finds it already there. The loop keeps its ordering
    guarantees; the expensive part stops being serialised.

    `max_workers` defaults to 2 because that is what measurement showed: 2
    workers gave 1.44x, and 4 gave 1.10x -- slower, because ONNX Runtime
    already parallelises internally and more workers just oversubscribe the
    CPU.

    Returns the number of images actually analysed. Failures are silent by
    design: a candidate that cannot be analysed here is simply not cached, and
    the verification loop handles it normally and records the proper rejection
    reason. Pre-warming must never be the thing that decides a verdict.
    """
    import hashlib
    import tempfile
    from pathlib import Path as _Path

    from tracelock.acquisition.validation import validate_image_bytes
    from tracelock.service.cache import FACE_CACHE

    token = token or NullToken()
    cache = cache if cache is not None else FACE_CACHE

    urls = media_urls(candidates)
    if not urls:
        return 0

    analysed = 0
    done = 0
    lock = threading.Lock()

    def warm(url: str) -> None:
        nonlocal analysed, done
        if token.cancelled:
            return
        try:
            result = fetch(url)
            if not getattr(result, "ok", False):
                return

            content = result.content or b""
            if not validate_image_bytes(content).ok:
                return

            digest = hashlib.sha256(content).hexdigest()

            # Already known: another URL served identical bytes, or a previous
            # run analysed them. Asking must not itself write a placeholder,
            # hence `has` rather than `analyze(..., lambda: None)`.
            if cache.has(digest):
                return

            handle = tempfile.NamedTemporaryFile(suffix=".img", delete=False)
            try:
                handle.write(content)
                handle.close()
                cache.analyze(digest, lambda: engine.analyze(handle.name))
                with lock:
                    analysed += 1
            finally:
                try:
                    _Path(handle.name).unlink(missing_ok=True)
                except OSError:
                    pass
        except Exception:
            # Any failure here is recoverable: the loop will do the work.
            return
        finally:
            with lock:
                done += 1
            if on_complete:
                on_complete(done, len(urls))

    with ThreadPoolExecutor(
        max_workers=max(1, min(max_workers, len(urls))),
        thread_name_prefix="face-warm",
    ) as pool:
        list(pool.map(warm, urls))

    return analysed
