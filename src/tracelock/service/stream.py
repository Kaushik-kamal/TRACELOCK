"""A candidate stream that lets verification start before discovery finishes.

THE PROBLEM THIS SOLVES
-----------------------
Discovery queries two indexes concurrently, but the pipeline still waited for
BOTH before verifying anything. Measured, Google Lens answered at 3.64s and
Yandex at 5.38s -- so the first candidate sat idle for 1.7s waiting on an index
that had nothing to do with it.

Worse, the whole candidate set was then prioritised, prefetched and pre-warmed
as one batch, so the first piece of evidence waited behind work for candidates
that early stopping would never even reach.

HOW IT WORKS
------------
Discovery pushes each source's results here as that source returns.
Verification pulls WAVES: it takes whatever is available, processes it, checks
whether the evidence is now sufficient, and only asks for more if it is not.

    source A returns ──┐
                       ├──> stream ──> wave 1 ──> enough? ──> stop
    source B returns ──┘                 └─ no ──> wave 2 ...

That makes the candidate budget adaptive rather than fixed: a run with strong
early corroboration examines five candidates, and a run with weak evidence
expands until it runs out of candidates or budget.

ORDERING
--------
Within a wave, candidates are prioritised deterministically. Across waves,
order follows arrival. Duplicate detection still runs in the single sequential
verification loop, so its results do not depend on thread scheduling -- the
stream changes WHEN candidates arrive, never how they are compared.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Sequence


class CandidateStream:
    """Thread-safe hand-off from discovery workers to the verification loop."""

    def __init__(self) -> None:
        self._pending: list[Any] = []
        self._lock = threading.Lock()
        self._arrived = threading.Event()
        self._closed = False
        self._offered = 0
        self._first_at: float | None = None
        self._started = time.perf_counter()

    # -- producer side (discovery worker threads) -----------------------

    def offer(self, candidates: Sequence[Any]) -> None:
        if not candidates:
            return
        with self._lock:
            if self._first_at is None:
                self._first_at = time.perf_counter() - self._started
            self._pending.extend(candidates)
            self._offered += len(candidates)
        self._arrived.set()

    def close(self) -> None:
        """No more candidates will arrive. Wakes any waiting consumer."""
        with self._lock:
            self._closed = True
        self._arrived.set()

    # -- consumer side (the single verification thread) -----------------

    def take(self, *, limit: int, timeout: float) -> list[Any]:
        """Up to `limit` candidates, waiting up to `timeout` for the first.

        Returns an empty list only when the stream is closed and drained, or
        the timeout expired with nothing available. A short wait here is what
        lets verification begin the instant the first source answers instead
        of after the slowest one.
        """
        deadline = time.perf_counter() + timeout

        while True:
            with self._lock:
                if self._pending:
                    wave = self._pending[:limit]
                    del self._pending[:limit]
                    if not self._pending:
                        self._arrived.clear()
                    return wave
                if self._closed:
                    return []

            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return []
            self._arrived.wait(timeout=min(remaining, 0.25))

    def peek(self, limit: int) -> list[Any]:
        """Look at what is queued WITHOUT consuming it.

        Used to warm the next wave's downloads while the current wave is still
        being verified. Non-destructive on purpose: the wave loop remains the
        only consumer, so peeking cannot reorder or drop anything.
        """
        with self._lock:
            return list(self._pending[:limit])

    # -- observation ----------------------------------------------------

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def drained(self) -> bool:
        with self._lock:
            return self._closed and not self._pending

    @property
    def offered(self) -> int:
        with self._lock:
            return self._offered

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._pending)

    @property
    def time_to_first_candidate(self) -> float | None:
        """Seconds from stream creation to the first candidate arriving.

        None when nothing ever arrived -- absent rather than a zero that would
        read as "instant".
        """
        return self._first_at
