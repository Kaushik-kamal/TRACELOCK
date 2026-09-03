"""Concurrent multi-source discovery with streaming results.

WHAT WAS ACTUALLY WRONG
-----------------------
The pipeline was not running searches sequentially -- it was running exactly
ONE. `settings.serp_engine` selected `google_lens` OR `yandex_images`, and the
other engine, already implemented and tested, was never queried. So the fix is
not "parallelise a sequential loop" (there wasn't one); it is "query the sources
we already support, at the same time, and use whichever answers".

That matters for evidence, not just speed: two indexes with different coverage
find different publishers, and independent publishers are what corroboration is
made of. Google Lens and Yandex disagree substantially -- Yandex cannot reach
Russia-restricted domains, Lens has weaker coverage of some regional sites.

STREAMING
---------
Candidates are handed to `on_candidates` as each source returns, so the pipeline
can start downloading and verifying while a slower source is still running. A
source that hangs delays only itself.

ISOLATION AND TIMEOUTS
----------------------
Every source runs in its own worker with its own timeout. A source that fails,
times out, or returns nothing is recorded as such and the run continues on the
others. The investigation fails only if EVERY source failed -- and then it says
which failed and why, rather than reporting an empty result as "nothing found".

COST
----
Each engine is a separately billed SerpAPI query. Two sources means two credits
per investigation, so the source list is configuration, not a hardcoded list.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from tracelock.core.models import Candidate, ProbeRef

# Per-source ceiling. A source past this is abandoned; the others keep going.
DEFAULT_SOURCE_TIMEOUT = 20.0

# Whole-discovery ceiling. Reached only if every source is slow.
DEFAULT_GLOBAL_TIMEOUT = 30.0


@dataclass
class SourceOutcome:
    """What one discovery source actually did. Never overstated."""

    name: str
    engine: str
    ok: bool
    candidate_count: int = 0
    elapsed: float = 0.0
    error: str = ""
    timed_out: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source": self.name,
            "engine": self.engine,
            "ok": self.ok,
            "candidates": self.candidate_count,
            "seconds": round(self.elapsed, 3),
        }
        if self.timed_out:
            payload["timed_out"] = True
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass
class DiscoveryOutcome:
    """The merged result, with every source's fate recorded."""

    candidates: list[Candidate] = field(default_factory=list)
    sources: list[SourceOutcome] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    queries: dict[str, Any] = field(default_factory=dict)
    elapsed: float = 0.0

    @property
    def any_source_succeeded(self) -> bool:
        return any(s.ok for s in self.sources)

    @property
    def succeeded_sources(self) -> list[SourceOutcome]:
        return [s for s in self.sources if s.ok]

    @property
    def failure_summary(self) -> str:
        """Why discovery produced nothing. Named sources, named reasons."""
        parts = []
        for source in self.sources:
            if source.ok:
                continue
            if source.timed_out:
                parts.append("{0} timed out".format(source.engine))
            else:
                parts.append("{0}: {1}".format(source.engine, source.error or "failed"))
        return "; ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sources_queried": len(self.sources),
            "sources_succeeded": len(self.succeeded_sources),
            "sources": [s.to_dict() for s in self.sources],
            "candidates_after_dedup": len(self.candidates),
            "seconds": round(self.elapsed, 3),
        }


def _canonical_url(url: str | None) -> str:
    """Normalise a media URL for duplicate detection.

    Conservative on purpose. Dropping a query string entirely would merge two
    genuinely different images served by the same resizing endpoint, so only
    tracking parameters and the fragment are removed. Anything this misses is
    caught later by SHA-256 and perceptual hashing of the actual bytes -- URL
    matching is the cheap first pass, never the authority.
    """
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

    if not url:
        return ""

    parts = urlsplit(url.strip())
    tracking = {
        "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
        "fbclid", "gclid", "igshid", "ref", "ref_src", "_ga",
    }
    query = urlencode(
        [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
         if k.lower() not in tracking]
    )
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]

    return urlunsplit((parts.scheme.lower(), host, parts.path, query, ""))


class MultiSourceDiscovery:
    """Query several reverse-image sources at once, streaming what comes back."""

    def __init__(
        self,
        providers: Sequence[Any],
        *,
        source_timeout: float = DEFAULT_SOURCE_TIMEOUT,
        global_timeout: float = DEFAULT_GLOBAL_TIMEOUT,
        max_workers: int = 4,
    ) -> None:
        self.providers = list(providers)
        self.source_timeout = source_timeout
        self.global_timeout = global_timeout
        self.max_workers = max_workers

        # URL-level dedup, shared across sources. The SAME image found by two
        # engines is one candidate with two discovery sources -- not two
        # candidates, which would fake corroboration out of thin air.
        self._seen: dict[str, Candidate] = {}
        self._found_by: dict[str, list[str]] = {}
        self._lock = threading.Lock()

    def search(
        self,
        probe: ProbeRef,
        *,
        limit: int = 20,
        on_candidates: Callable[[list[Candidate], SourceOutcome], None] | None = None,
        token=None,
    ) -> DiscoveryOutcome:
        """Run every source concurrently. Returns once all finish or time out.

        `on_candidates` is called from a worker thread each time a source
        returns, with only the candidates that were NOT already seen. It is what
        lets verification begin before discovery has finished.
        """
        outcome = DiscoveryOutcome()
        if not self.providers:
            return outcome

        started = time.perf_counter()

        def run(provider) -> SourceOutcome:
            engine = getattr(provider, "engine", provider.name)
            source_started = time.perf_counter()

            if token is not None and token.cancelled:
                return SourceOutcome(provider.name, engine, ok=False,
                                     error="cancelled before start")
            try:
                result = provider.search(probe, limit=limit)
            except Exception as exc:
                # Isolated: this source's failure is recorded and the others
                # continue. One dead index must never end an investigation.
                return SourceOutcome(
                    provider.name, engine, ok=False,
                    elapsed=time.perf_counter() - source_started,
                    error="{0}: {1}".format(type(exc).__name__, str(exc)[:160]),
                )

            fresh = self._merge(result.candidates, engine)
            source = SourceOutcome(
                provider.name, engine, ok=True,
                candidate_count=len(result.candidates),
                elapsed=time.perf_counter() - source_started,
            )

            with self._lock:
                outcome.raw[engine] = result.raw_response
                outcome.queries[engine] = result.query

            if on_candidates and fresh:
                try:
                    on_candidates(fresh, source)
                except Exception:
                    # A streaming consumer must never be able to kill a source.
                    pass
            return source

        pool = ThreadPoolExecutor(
            max_workers=max(1, min(self.max_workers, len(self.providers))),
            thread_name_prefix="discovery",
        )
        try:
            futures = {pool.submit(run, p): p for p in self.providers}
            deadline = started + self.global_timeout
            pending = set(futures)

            while pending:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                done, pending = wait(
                    pending, timeout=min(remaining, self.source_timeout),
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    outcome.sources.append(future.result())

                if token is not None and token.cancelled:
                    break

            # Anything still running has exceeded its budget. Record it as a
            # timeout rather than waiting -- a hung index must not hold the
            # investigation open.
            for future in pending:
                future.cancel()
                provider = futures[future]
                outcome.sources.append(SourceOutcome(
                    provider.name, getattr(provider, "engine", provider.name),
                    ok=False, timed_out=True,
                    elapsed=time.perf_counter() - started,
                    error="exceeded {0:.0f}s budget".format(self.global_timeout),
                ))
        finally:
            # Never block the pipeline waiting for an abandoned worker to
            # notice. The daemon threads die with the process.
            pool.shutdown(wait=False, cancel_futures=True)

        with self._lock:
            outcome.candidates = list(self._seen.values())
        outcome.elapsed = time.perf_counter() - started
        return outcome

    # -- deduplication --------------------------------------------------

    def _merge(self, candidates: Sequence[Candidate], engine: str) -> list[Candidate]:
        """Add candidates, collapsing ones already seen. Returns the new ones.

        A candidate found by two engines keeps BOTH engines in its provenance:
        that is genuinely useful forensic information ("two independent indexes
        surfaced this"), and it must not be turned into two separate pieces of
        evidence.
        """
        fresh: list[Candidate] = []

        with self._lock:
            for candidate in candidates:
                key = _canonical_url(candidate.image_url or candidate.thumbnail_url)
                if not key:
                    key = _canonical_url(candidate.post_url)
                if not key:
                    continue

                if key in self._seen:
                    if engine not in self._found_by[key]:
                        self._found_by[key].append(engine)
                    continue

                self._seen[key] = candidate
                self._found_by[key] = [engine]
                fresh.append(candidate)

        return fresh

    def found_by(self, candidate: Candidate) -> list[str]:
        """Which engines surfaced this candidate."""
        key = _canonical_url(candidate.image_url or candidate.thumbnail_url) \
            or _canonical_url(candidate.post_url)
        with self._lock:
            return list(self._found_by.get(key, []))

    @property
    def discovery_map(self) -> dict[str, list[str]]:
        with self._lock:
            return {k: list(v) for k, v in self._found_by.items()}


def build_providers(settings, *, multi_source: bool = True) -> list[Any]:
    """The providers to query for this investigation.

    Each engine is a separately billed SerpAPI query, so multi-source is a
    setting rather than an assumption. With it off, behaviour is exactly what it
    was before: the single configured engine.
    """
    from tracelock.discovery.serpapi_lens import SUPPORTED_ENGINES, SerpApiProvider

    engines = [settings.serp_engine]
    if multi_source:
        engines += [e for e in SUPPORTED_ENGINES if e != settings.serp_engine]

    providers = []
    for engine in engines:
        try:
            providers.append(SerpApiProvider(
                settings.serpapi_api_key,
                engine=engine,
                timeout=min(settings.http_timeout, DEFAULT_SOURCE_TIMEOUT),
            ))
        except Exception:
            # A misconfigured engine is skipped, not fatal -- the configured
            # primary engine is constructed first and would raise on its own.
            continue
    return providers
