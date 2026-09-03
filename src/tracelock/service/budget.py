"""Investigation budget: how much work to do, and when enough is enough.

Profiling showed face analysis is 68.7% of per-candidate cost. The largest
available speed-up is therefore not to parallelise that work but to SKIP it --
stop once the evidence already answers the question, and never extract an
embedding from a candidate that failed to download.

    FAST      12 candidates, stop at 3 verified or 3 independent publishers
    THOROUGH  25 candidates, examine every one

THE TRUTHFULNESS RULE THAT GOVERNS THIS FILE
--------------------------------------------
Stopping early changes how much we LOOKED, never what we FOUND. So:

  * a stop is recorded explicitly -- `stopped_early`, with the count examined
    and the count skipped -- and surfaced in the artifact and the UI
  * corroboration counts only publishers actually verified, so stopping can
    only ever LOWER a trust score, never raise it
  * a stop is never triggered by a rejection: finding nothing is not a reason
    to stop looking, only finding enough is

The asymmetry matters. Early-stopping on success is efficiency. Early-stopping
on failure would be giving up and calling it an answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

# Three independent publishers is the point past which additional corroboration
# barely moves the trust score: the corroboration factor 1-e^(-0.5(n-1)) is
# already at 0.63 by n=3, and each further publisher adds under 0.14. Spending
# ~1.65s of face analysis for that is not a good trade in a live demo.
TARGET_VERIFIED = 3
TARGET_INDEPENDENT_PUBLISHERS = 3


class Mode(str, Enum):
    FAST = "fast"
    THOROUGH = "thorough"

    @property
    def label(self) -> str:
        return {Mode.FAST: "Fast", Mode.THOROUGH: "Thorough"}[self]

    @property
    def description(self) -> str:
        return {
            Mode.FAST: (
                "Examines up to {0} candidates and stops once {1} independent "
                "publishers confirm a match.".format(
                    _LIMITS[Mode.FAST], TARGET_INDEPENDENT_PUBLISHERS
                )
            ),
            Mode.THOROUGH: (
                "Examines every one of up to {0} candidates. Slower, and finds "
                "corroboration a fast run would stop before reaching.".format(
                    _LIMITS[Mode.THOROUGH]
                )
            ),
        }[self]


_LIMITS: dict[Mode, int] = {Mode.FAST: 12, Mode.THOROUGH: 25}


@dataclass(frozen=True, slots=True)
class Budget:
    """Concurrency limits and stopping rules for one investigation.

    Concurrency defaults come from measurement on this machine, not intuition:

        downloads  5 workers -> 2.86x   (network-bound, scales)
        faces      8 workers               (see below)

    The face figure was WRONG in the previous pass, and the reason is worth
    recording. Measurements then showed 2 workers beating 4, and the conclusion
    drawn was "ONNX already uses the cores, so keep worker count low". The
    measurement was real; the conclusion was not. ONNX Runtime was defaulting
    intra_op_num_threads to all 16 cores, so every extra worker multiplied an
    already-saturated thread pool.

    With intra_op pinned to 4 (see face/engine.py), total seconds for 12 images:

        intra_op   1w     2w     4w     6w     8w
               2   5.48   3.66   2.65   2.41   2.32
               4   4.52   3.11   2.50   2.20   2.08   <- best
               8   9.25   6.79   4.64   3.65   3.18

    So the cap was never a property of the workload -- it was a symptom of an
    unconfigured runtime.
    """

    mode: Mode = Mode.FAST
    candidate_limit: int = _LIMITS[Mode.FAST]
    max_concurrent_downloads: int = 5
    max_concurrent_face: int = 8
    target_verified: int = TARGET_VERIFIED
    target_publishers: int = TARGET_INDEPENDENT_PUBLISHERS
    allow_early_stop: bool = True
    # How many candidates to download AHEAD of the verification loop.
    # Prefetching everything would undo early stopping's main saving: a FAST
    # run that stops at candidate 6 would still have paid for 12 downloads.
    # Anything past the window is fetched on demand, through the same cache,
    # so a run that DOES go further is not penalised -- it just pays later.
    prefetch_window: int = 8

    # -- discovery -----------------------------------------------------
    # Querying both engines costs two SerpAPI credits per investigation, so it
    # is a setting rather than an assumption. It buys real source diversity:
    # Lens and Yandex index different publishers, and corroboration counts
    # independent publishers.
    multi_source: bool = True
    source_timeout: float = 20.0
    global_timeout: float = 45.0

    # -- adaptive waves -------------------------------------------------
    # Candidates are processed in waves rather than as one batch. After each
    # wave the evidence is re-checked, and the next wave is requested only if
    # it is still insufficient. That makes the candidate budget adaptive: a run
    # with strong early corroboration examines one wave, a weak one expands.
    #
    # `wave_timeout` is how long a wave waits for candidates to arrive before
    # concluding discovery has nothing more coming. It must exceed the gap
    # between the fastest and slowest source (measured ~1.7s) or the second
    # source's results would be treated as absent.
    wave_size: int = 5
    wave_timeout: float = 8.0

    # The FIRST wave is deliberately small. Time-to-first-evidence is bounded
    # by how much work sits between the first candidate arriving and the first
    # verdict, and a wave of 5 makes the first verdict wait on four candidates
    # it does not need. Later waves are larger because by then throughput
    # matters more than latency.
    first_wave_size: int = 3

    # -- what "enough evidence" means ----------------------------------
    # Early stopping requires CORROBORATION, not a candidate count. A run stops
    # because N independent publishers each carry a match the CALIBRATED policy
    # classified as VERIFIED_CANDIDATE.
    #
    # This was 0.80 in the previous pass and that was a mistake worth
    # recording. The calibrated decision boundary is similarity 0.3528, which
    # maps to an identity probability of 0.4862. Demanding 0.80 therefore
    # required similarity ~0.45 -- far beyond the boundary the calibration
    # actually supports -- so genuinely verified candidates were silently
    # disqualified from counting and early stopping almost never fired.
    #
    # The honest gate is the calibrated one. VERIFIED_CANDIDATE already means
    # "above the fitted decision boundary"; layering an invented constant on
    # top is second-guessing the calibration with a number no data supports.
    # This stays configurable so it can be raised DELIBERATELY, but it defaults
    # to trusting the calibration.
    min_identity_probability: float = 0.0

    @classmethod
    def for_mode(cls, mode: Mode | str) -> "Budget":
        resolved = Mode(mode) if not isinstance(mode, Mode) else mode
        limit = _LIMITS[resolved]
        fast = resolved is Mode.FAST
        return cls(
            mode=resolved,
            candidate_limit=limit,
            allow_early_stop=fast,
            # Thorough examines everything, so there is nothing to save by
            # holding downloads back.
            prefetch_window=8 if fast else limit,
            multi_source=True,
            source_timeout=15.0 if fast else 25.0,
            global_timeout=25.0 if fast else 60.0,
            wave_size=5 if fast else 10,
            first_wave_size=3,
        )

    def should_stop(self, *, verified: int, publishers: int) -> bool:
        """True once the evidence is sufficient. Never true on failure alone."""
        if not self.allow_early_stop:
            return False
        return verified >= self.target_verified and publishers >= self.target_publishers

    def stop_explanation(self, *, examined: int, total: int) -> str:
        skipped = max(0, total - examined)
        if not skipped:
            return ""
        return (
            "Stopped after {0} of {1} candidates: {2} independent publishers had "
            "already confirmed a match. {3} candidate{4} not examined - "
            "run in Thorough mode to check all of them.".format(
                examined, total, self.target_publishers,
                skipped, "" if skipped == 1 else "s",
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "mode_label": self.mode.label,
            "candidate_limit": self.candidate_limit,
            "max_concurrent_downloads": self.max_concurrent_downloads,
            "max_concurrent_face": self.max_concurrent_face,
            "early_stop_enabled": self.allow_early_stop,
            "target_verified": self.target_verified,
            "target_independent_publishers": self.target_publishers,
            "prefetch_window": self.prefetch_window,
            "wave_size": self.wave_size,
            "first_wave_size": self.first_wave_size,
            "multi_source": self.multi_source,
            "source_timeout": self.source_timeout,
            "global_timeout": self.global_timeout,
            "min_identity_probability": self.min_identity_probability,
        }


@dataclass
class StopTracker:
    """Counts confirmed evidence as results arrive, and says when to stop.

    Counts VERIFIED_CANDIDATE results only, and counts each publisher once, so
    ten copies of one photo on one site can never satisfy a stopping rule that
    is meant to measure independent corroboration.
    """

    budget: Budget
    verified: int = 0
    weak: int = 0
    examined: int = 0
    stopped_early: bool = False

    def __post_init__(self) -> None:
        self._publishers: set[str] = set()
        self._content: set[str] = set()

    @property
    def publishers(self) -> int:
        return len(self._publishers)

    def record(self, result: dict[str, Any]) -> None:
        from tracelock.core.reasons import VerificationStatus

        self.examined += 1
        # Compared against the enum, not a hand-typed literal. A literal here
        # silently stopped matching the emitted "VERIFIED_CANDIDATE" and the
        # early-stop path never fired on real data.
        if result.get("status") != VerificationStatus.VERIFIED_CANDIDATE.value:
            return

        # A verified candidate only counts toward STOPPING if the match is
        # strong on its own. A borderline result is still real evidence and is
        # still reported -- it just cannot be the reason we stop looking for
        # more.
        probability = result.get("identity_probability")
        if probability is not None and probability < self.budget.min_identity_probability:
            self.weak += 1
            return

        self.verified += 1

        # Independence is measured at the PUBLISHER, and a duplicate image is
        # not a second publication however many URLs it has. Both guards are
        # needed: without the first, ten pages on one site look like ten
        # sources; without the second, ten copies of one photo do.
        content = result.get("content_sha256") or ""
        if content and content in self._content:
            return
        if content:
            self._content.add(content)

        publisher = (
            (result.get("provenance") or {}).get("registrable_domain")
            or (result.get("provenance") or {}).get("host")
            or ""
        )
        if publisher:
            self._publishers.add(publisher)

    def should_stop(self) -> bool:
        stop = self.budget.should_stop(
            verified=self.verified, publishers=self.publishers
        )
        if stop:
            self.stopped_early = True
        return stop

    def summary(self, total: int) -> dict[str, Any]:
        return {
            "examined": self.examined,
            "available": total,
            "not_examined": max(0, total - self.examined),
            "stopped_early": self.stopped_early,
            "verified": self.verified,
            "below_stop_threshold": self.weak,
            "independent_publishers": self.publishers,
            "explanation": (
                self.budget.stop_explanation(examined=self.examined, total=total)
                if self.stopped_early
                else ""
            ),
        }
