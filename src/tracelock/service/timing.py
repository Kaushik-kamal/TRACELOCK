"""Real phase timing for the investigation pipeline.

Every number this module reports is a measured wall-clock interval. Nothing is
estimated, extrapolated, or filled in from a previous run -- a phase that did
not execute has no entry, rather than a plausible-looking zero.

Phases can overlap, because some of them now genuinely run concurrently
(discovery no longer waits for the input embedding). So the phase totals do NOT
sum to the wall-clock total, and pretending otherwise would misreport where the
time went. `overlap_saved` reports the difference explicitly: it is the time
concurrency actually removed, computed from the recorded intervals.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass
class Phase:
    """One measured phase.

    Two kinds, deliberately kept apart:

      interval     one wall-clock span, from start() to stop()
      accumulated  a sum of durations measured elsewhere (worker threads)

    Mixing them on one name produced a nonsense figure once -- start() stored a
    perf_counter origin, add() reset it to zero, and stop() then subtracted
    zero from a perf_counter value, reporting 1,139,129 seconds. So the kind is
    recorded and a name cannot be used both ways.
    """

    name: str
    started_at: float
    ended_at: float | None = None
    count: int = 0
    detail: str = ""
    kind: str = "interval"
    accumulated: float = 0.0

    @property
    def elapsed(self) -> float:
        if self.kind == "accumulated":
            return self.accumulated
        if self.ended_at is None:
            return 0.0
        return self.ended_at - self.started_at

    @property
    def finished(self) -> bool:
        if self.kind == "accumulated":
            return self.accumulated > 0.0
        return self.ended_at is not None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "seconds": round(self.elapsed, 4),
            "kind": self.kind,
        }
        if self.count:
            payload["count"] = self.count
            payload["per_item_seconds"] = round(self.elapsed / self.count, 4)
        if self.detail:
            payload["detail"] = self.detail
        return payload


class PhaseTimer:
    """Records how long each phase of one investigation actually took.

    Safe to use from several threads: the concurrent phases (discovery,
    downloads, face analysis) record from worker threads while the main
    pipeline thread reads. Without the lock the dict could be mutated
    mid-iteration when a snapshot is taken for the progress socket.
    """

    def __init__(self, clock=time.perf_counter) -> None:
        self._clock = clock
        self._phases: dict[str, Phase] = {}
        self._lock = threading.Lock()
        self._started = clock()
        self._ended: float | None = None

    # -- recording ------------------------------------------------------

    def start(self, name: str) -> None:
        with self._lock:
            existing = self._phases.get(name)
            if existing is not None and existing.kind == "accumulated":
                raise ValueError(
                    "phase {0!r} is accumulated; it cannot also be an "
                    "interval. Use a different name.".format(name)
                )
            self._phases[name] = Phase(name=name, started_at=self._clock())

    def stop(self, name: str, *, count: int = 0, detail: str = "") -> float:
        with self._lock:
            phase = self._phases.get(name)
            if phase is None or phase.finished:
                return 0.0
            phase.ended_at = self._clock()
            phase.count = count
            phase.detail = detail
            return phase.elapsed

    def add(self, name: str, seconds: float, *, count: int = 1) -> None:
        """Accumulate time for work measured elsewhere (e.g. inside a worker).

        Used for per-candidate costs that happen on pool threads and would
        otherwise be invisible. Accumulated phases sum CPU spent, so they can
        legitimately exceed wall-clock when work ran in parallel.
        """
        with self._lock:
            phase = self._phases.get(name)
            if phase is not None and phase.kind == "interval":
                raise ValueError(
                    "phase {0!r} is an interval; it cannot also be "
                    "accumulated. Use a different name.".format(name)
                )
            if phase is None:
                phase = Phase(name=name, started_at=0.0, kind="accumulated")
                self._phases[name] = phase
            phase.accumulated += seconds
            phase.count += count

    def measure(self, name: str, *, count: int = 0):
        """Context manager: `with timer.measure("discovery"): ...`"""
        return _Measured(self, name, count)

    def finish(self) -> None:
        with self._lock:
            self._ended = self._clock()

    # -- reading --------------------------------------------------------

    @property
    def total(self) -> float:
        end = self._ended if self._ended is not None else self._clock()
        return end - self._started

    def get(self, name: str) -> float:
        with self._lock:
            phase = self._phases.get(name)
            return phase.elapsed if phase else 0.0

    def mark(self) -> float:
        """Seconds since the investigation began. For 'time to first X'."""
        return self._clock() - self._started

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            finished = [p for p in self._phases.values() if p.finished]
            phases = [p.to_dict() for p in finished]
            # Only wall-clock intervals may enter the overlap figure. An
            # accumulated phase is summed CPU across workers, so including it
            # would report parallelism as if it were time saved twice.
            wall = sum(p.elapsed for p in finished if p.kind == "interval")
            cpu = sum(p.elapsed for p in finished if p.kind == "accumulated")

        total = self.total
        return {
            "phases": phases,
            "total_seconds": round(total, 4),
            "measured_seconds": round(wall, 4),
            "worker_cpu_seconds": round(cpu, 4),
            # Positive when wall-clock phases overlapped -- i.e. what
            # concurrency actually removed. Reported rather than hidden,
            # because the phase list otherwise appears not to add up.
            "overlap_saved_seconds": round(max(0.0, wall - total), 4),
            "note": (
                "Measured wall-clock intervals. Phases can overlap because "
                "discovery, downloads and face analysis run concurrently, so "
                "they do not sum to the total. Phases marked as worker CPU are "
                "summed across threads and are excluded from the overlap "
                "figure."
            ),
        }

    def to_timings_ms(self) -> dict[str, int]:
        """Flat `<phase>_ms` map, for consumers that want plain integers.

        The same measurements as `to_dict`, in milliseconds and one level
        deep. A phase that never finished is ABSENT rather than 0 -- a zero
        would read as "instant" for something that never ran at all.
        """
        with self._lock:
            finished = [p for p in self._phases.values() if p.finished]

        timings = {
            "{0}_ms".format(phase.name): int(round(phase.elapsed * 1000))
            for phase in finished
        }
        timings["total_ms"] = int(round(self.total * 1000))
        return timings

    def summary_lines(self) -> list[str]:
        """Human-readable block, for CLI output and the debug panel."""
        with self._lock:
            phases = [p for p in self._phases.values() if p.finished]

        if not phases:
            return []

        width = max(len(p.name) for p in phases)
        lines = [
            "{0:<{1}}  {2:>7.2f}s{3}".format(
                p.name, width, p.elapsed,
                "  ({0} items, {1:.2f}s each)".format(p.count, p.elapsed / p.count)
                if p.count else "",
            )
            for p in sorted(phases, key=lambda p: -p.elapsed)
        ]
        lines.append("{0:<{1}}  {2:>7.2f}s".format("TOTAL", width, self.total))
        return lines


class _Measured:
    __slots__ = ("_timer", "_name", "_count")

    def __init__(self, timer: PhaseTimer, name: str, count: int) -> None:
        self._timer = timer
        self._name = name
        self._count = count

    def __enter__(self) -> "_Measured":
        self._timer.start(self._name)
        return self

    def __exit__(self, *exc: Any) -> None:
        # Stops even when the body raised: a phase that failed still took time,
        # and hiding that would misreport where a slow failure went.
        self._timer.stop(self._name, count=self._count)

    def set_count(self, count: int) -> None:
        self._count = count


def iter_phase_names() -> Iterator[str]:
    """The phases the pipeline records, in pipeline order."""
    yield from (
        "input_analysis",
        "discovery",
        "candidate_filtering",
        "download",
        "face_detection",
        "face_comparison",
        "corroboration",
        "anchoring",
    )
