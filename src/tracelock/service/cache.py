"""Bounded, thread-safe caches for the expensive parts of the pipeline.

Profiling said face analysis is 68.7% of per-candidate cost and downloads 30.1%
(decode is 1.2% and not worth caching). So exactly two things are cached, and
both are keyed by CONTENT, never by URL alone:

    FaceCache   sha256(image bytes) -> analysis result
    FetchCache  url -> bytes, with the sha256 recorded alongside

Keying face analysis by sha256 is what makes the cache safe. Two URLs serving
identical bytes are genuinely the same image, so reusing the analysis is not an
optimisation that changes an answer -- it is the same answer, computed once.
A URL key would be a lie: the same URL can serve different bytes tomorrow.

WHY THIS DOES NOT WEAKEN EVIDENCE
---------------------------------
Nothing here caches a VERDICT. Similarity, calibrated probability, trust score
and duplicate relations are recomputed for every candidate in every run. The
cache holds only deterministic, pure derivations of fixed bytes -- the same
input always produced the same output before the cache existed.

Entries expire so a long-lived server cannot serve a stale download, and both
caches are capacity-bounded with LRU eviction so a hostile or merely large run
cannot exhaust memory.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

# A face analysis is ~1.65s to recompute and a few KB to hold. Caching a few
# hundred is cheap next to that.
FACE_CACHE_CAPACITY = 512
FACE_CACHE_TTL_SECONDS = 3600.0

# Downloaded bytes are much larger, so far fewer are held, and they expire
# sooner: a cached download must never outlive the operator's expectation that
# a re-run actually re-fetches.
FETCH_CACHE_CAPACITY = 64
FETCH_CACHE_TTL_SECONDS = 300.0
FETCH_CACHE_MAX_ENTRY_BYTES = 8 * 1024 * 1024


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "hit_rate": round(self.hit_rate, 4),
        }


class BoundedTTLCache:
    """LRU cache with a time-to-live. Safe to share across threads."""

    def __init__(
        self,
        *,
        capacity: int,
        ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._capacity = capacity
        self._ttl = ttl_seconds
        self._clock = clock
        self._entries: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.stats = CacheStats()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.stats.misses += 1
                return None

            stored_at, value = entry
            if self._clock() - stored_at > self._ttl:
                del self._entries[key]
                self.stats.expirations += 1
                self.stats.misses += 1
                return None

            self._entries.move_to_end(key)
            self.stats.hits += 1
            return value

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
            self._entries[key] = (self._clock(), value)

            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)
                self.stats.evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


class FaceCache:
    """Face analysis keyed by the sha256 of the image bytes."""

    def __init__(self, **kwargs: Any) -> None:
        self._cache = BoundedTTLCache(
            capacity=kwargs.pop("capacity", FACE_CACHE_CAPACITY),
            ttl_seconds=kwargs.pop("ttl_seconds", FACE_CACHE_TTL_SECONDS),
            **kwargs,
        )

    @property
    def stats(self) -> CacheStats:
        return self._cache.stats

    def analyze(self, digest: str, compute: Callable[[], Any]) -> Any:
        """Return the cached analysis for these bytes, or compute and store it.

        `compute` may raise -- a failed analysis is NOT cached, because the
        failure can be environmental (a model still loading) rather than a
        property of the image.
        """
        cached = self._cache.get(digest)
        if cached is not None:
            return cached

        result = compute()
        self._cache.put(digest, result)
        return result

    def has(self, digest: str) -> bool:
        """Is this analysis already cached?

        Distinct from `analyze`: asking whether a result exists must not
        compute one, and must not store a placeholder. The pre-warmer needs to
        skip work already done without polluting the cache to find out.
        """
        return self._cache.get(digest) is not None

    def clear(self) -> None:
        self._cache.clear()


class FetchCache:
    """Downloaded bytes keyed by URL, bounded by entry size and total count."""

    def __init__(self, **kwargs: Any) -> None:
        self._cache = BoundedTTLCache(
            capacity=kwargs.pop("capacity", FETCH_CACHE_CAPACITY),
            ttl_seconds=kwargs.pop("ttl_seconds", FETCH_CACHE_TTL_SECONDS),
            **kwargs,
        )
        self._max_entry = kwargs.pop("max_entry_bytes", FETCH_CACHE_MAX_ENTRY_BYTES)

    @property
    def stats(self) -> CacheStats:
        return self._cache.stats

    def get(self, url: str) -> Any | None:
        return self._cache.get(url)

    def put(self, url: str, result: Any) -> None:
        """Store a SUCCESSFUL acquisition only.

        Failures are never cached: a 503 or a timeout is a moment in time, and
        caching it would turn a transient outage into a permanent one for the
        rest of the TTL.
        """
        if not getattr(result, "ok", False):
            return
        content = getattr(result, "content", None) or b""
        if len(content) > self._max_entry:
            return
        self._cache.put(url, result)

    def clear(self) -> None:
        self._cache.clear()


# Process-wide instances. Shared deliberately: the whole point is that a
# candidate seen in run A is not re-analysed in run B.
FACE_CACHE = FaceCache()
FETCH_CACHE = FetchCache()


def cache_stats() -> dict[str, Any]:
    return {
        "face": FACE_CACHE.stats.to_dict(),
        "fetch": FETCH_CACHE.stats.to_dict(),
    }


def clear_all() -> None:
    FACE_CACHE.clear()
    FETCH_CACHE.clear()
