"""Streaming candidates, adaptive waves, and the tuned face runtime.

The theme is that doing less work, sooner, must not change what any candidate
concludes. A candidate verified in wave 1 gets the verdict it would have got in
wave 4, and a candidate never reached is reported as never reached.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

from tracelock.service.stream import CandidateStream


# ==========================================================================
# The candidate stream
# ==========================================================================


class TestCandidateStream:
    def test_take_returns_what_has_arrived(self):
        stream = CandidateStream()
        stream.offer(["a", "b", "c"])
        assert stream.take(limit=2, timeout=0.1) == ["a", "b"]
        assert stream.take(limit=2, timeout=0.1) == ["c"]

    def test_take_waits_for_a_late_producer(self):
        """This wait is the whole point: it lets wave 1 start on the FIRST
        source's results instead of after the slowest one."""
        stream = CandidateStream()

        def produce():
            time.sleep(0.15)
            stream.offer(["late"])

        threading.Thread(target=produce, daemon=True).start()
        started = time.perf_counter()
        wave = stream.take(limit=5, timeout=2.0)
        elapsed = time.perf_counter() - started

        assert wave == ["late"]
        assert 0.1 < elapsed < 1.0

    def test_take_returns_empty_once_closed_and_drained(self):
        stream = CandidateStream()
        stream.offer(["a"])
        stream.close()
        assert stream.take(limit=5, timeout=0.1) == ["a"]
        assert stream.take(limit=5, timeout=0.1) == []
        assert stream.drained

    def test_close_wakes_a_waiting_consumer_immediately(self):
        """A closed stream must not make the consumer serve out its timeout."""
        stream = CandidateStream()

        def closer():
            time.sleep(0.1)
            stream.close()

        threading.Thread(target=closer, daemon=True).start()
        started = time.perf_counter()
        assert stream.take(limit=5, timeout=5.0) == []
        assert time.perf_counter() - started < 1.0

    def test_peek_does_not_consume(self):
        stream = CandidateStream()
        stream.offer(["a", "b", "c"])
        assert stream.peek(2) == ["a", "b"]
        assert stream.pending == 3
        assert stream.take(limit=3, timeout=0.1) == ["a", "b", "c"]

    def test_time_to_first_candidate_is_absent_when_none_arrived(self):
        """Absent, not zero -- zero would read as 'instant'."""
        stream = CandidateStream()
        assert stream.time_to_first_candidate is None
        stream.offer(["a"])
        assert stream.time_to_first_candidate is not None

    def test_concurrent_producers_lose_nothing(self):
        stream = CandidateStream()

        def produce(n):
            for i in range(50):
                stream.offer(["p{0}-{1}".format(n, i)])

        threads = [threading.Thread(target=produce, args=(n,)) for n in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        stream.close()

        drained = []
        while True:
            wave = stream.take(limit=17, timeout=0.1)
            if not wave:
                break
            drained.extend(wave)

        assert len(drained) == 300
        assert len(set(drained)) == 300


# ==========================================================================
# The tuned face runtime
# ==========================================================================


class TestFaceRuntimeTuning:
    def test_unused_submodels_are_not_loaded(self):
        """landmark_2d_106 and genderage are never read by this engine.

        genderage additionally infers attributes this system has no business
        deriving, so not loading it is a privacy improvement too.
        """
        from tracelock.face.engine import REQUIRED_MODULES

        assert "landmark_2d_106" not in REQUIRED_MODULES
        assert "genderage" not in REQUIRED_MODULES

    def test_pose_model_is_retained(self):
        """landmark_3d_68 supplies pose, which feeds the quality aggregate.

        Dropping it left embeddings bit-identical but CHANGED quality on all 12
        test images -- and quality feeds the trust score. Speed is not worth
        silently moving a score.
        """
        from tracelock.face.engine import REQUIRED_MODULES

        assert "landmark_3d_68" in REQUIRED_MODULES
        assert "detection" in REQUIRED_MODULES
        assert "recognition" in REQUIRED_MODULES

    def test_intra_op_threads_are_pinned(self):
        """Unpinned, ONNX takes all cores per session and every worker thread
        multiplies an already-saturated pool."""
        from tracelock.face.engine import DEFAULT_INTRA_OP_THREADS

        assert 1 <= DEFAULT_INTRA_OP_THREADS <= 8

    def test_detector_input_size_is_unchanged(self):
        """det_size must not move without recalibration.

        Measured: 320-512 are 1.26-1.60x faster but shift the embedding
        (cosine 0.93-0.94 against 640). The decision boundary was fitted at
        640, so changing this silently invalidates the calibration.
        """
        from tracelock.face.engine import DEFAULT_DET_SIZE

        assert DEFAULT_DET_SIZE == (640, 640)

    def test_thread_tuning_failure_does_not_break_startup(self):
        """A slower engine beats an engine that will not start."""
        source = Path(
            sys.modules["tracelock.face.engine"].__file__
        ).read_text(encoding="utf-8")
        block = source[source.index("def _apply_thread_limits"):]
        block = block[: block.index("# ----")]
        assert "except Exception" in block
        assert "continue" in block


# ==========================================================================
# Adaptive budget
# ==========================================================================


class TestAdaptiveBudget:
    def test_the_first_wave_is_smaller_than_later_waves(self):
        """Time-to-first-evidence is bounded by the work between the first
        candidate and the first verdict."""
        from tracelock.service.budget import Budget, Mode

        for mode in (Mode.FAST, Mode.THOROUGH):
            budget = Budget.for_mode(mode)
            assert budget.first_wave_size <= budget.wave_size

    def test_thorough_uses_larger_waves_than_fast(self):
        from tracelock.service.budget import Budget, Mode

        assert (
            Budget.for_mode(Mode.THOROUGH).wave_size
            > Budget.for_mode(Mode.FAST).wave_size
        )

    def test_wave_timeout_exceeds_the_gap_between_sources(self):
        """Measured, the two indexes finished ~1.7s apart. A wave timeout
        below that would treat the slower source's results as absent."""
        from tracelock.service.budget import Budget, Mode

        for mode in (Mode.FAST, Mode.THOROUGH):
            assert Budget.for_mode(mode).wave_timeout >= 3.0

    def test_stop_gate_does_not_exceed_the_calibrated_boundary(self):
        """A gate above the fitted boundary silently disables early stopping.

        This was 0.80 against a boundary of 0.4862, which disqualified
        candidates the calibrated policy itself called VERIFIED_CANDIDATE.
        """
        from tracelock.calibration.model import CalibrationModel
        from tracelock.service.budget import Budget, Mode
        from tracelock.verification.policy import VerificationPolicy

        model = CalibrationModel.load("data/calibration/model.json")
        policy = VerificationPolicy.from_calibration(model)
        boundary = policy.identity_probability(policy.similarity_ceiling)

        for mode in (Mode.FAST, Mode.THOROUGH):
            assert Budget.for_mode(mode).min_identity_probability <= boundary


# ==========================================================================
# Speculative work must never sit on the critical path
# ==========================================================================


class TestSpeculativeWorkIsOffTheCriticalPath:
    def test_the_first_wave_does_not_prefetch_ahead(self):
        """Warming the next wave during wave 1 pushed TTFE from 5.1s to 7.4s:
        speculative URLs shared the download pool with the critical path."""
        import tracelock.service.runner as runner_module

        source = Path(runner_module.__file__).read_text(encoding="utf-8")

        assert "if self._waves_prepared else []" in source, (
            "the first wave must not warm the next one"
        )

    def test_no_thread_pool_is_created_inside_a_daemon_thread(self):
        """ThreadPoolExecutor registers an atexit hook that joins its workers,
        so building one in a daemon thread hangs interpreter shutdown.

        Checks CODE only -- comments and docstrings explaining the hazard must
        not themselves trip the check, which is exactly what happened first.
        """
        import io
        import tokenize

        import tracelock.service.runner as runner_module

        code_tokens = []
        with io.open(runner_module.__file__, encoding="utf-8") as handle:
            for token in tokenize.generate_tokens(handle.readline):
                if token.type in (tokenize.COMMENT, tokenize.STRING):
                    continue
                code_tokens.append(token.string)
        code = " ".join(code_tokens)

        assert "ThreadPoolExecutor" not in code
        assert "warm-next-wave" not in code
