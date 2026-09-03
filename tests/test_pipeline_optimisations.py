"""Input reuse, image-level deduplication, prioritisation, and timing.

These cover the optimisations that change HOW MUCH work the pipeline does. The
recurring assertion is that doing less work never changes a conclusion: the
same candidate reaches the same verdict whether it is processed first or last,
cached or fresh, and a skipped candidate is reported as skipped rather than
silently assumed.
"""

from __future__ import annotations

import re
import sys
import threading
import time
from pathlib import Path

import pytest


def tiny_jpeg() -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 130, 140)).save(buffer, format="JPEG")
    return buffer.getvalue()


# -- stubs -----------------------------------------------------------------


class _Band:
    value = "GOOD"


class _FakeQuality:
    aggregate = 0.9
    band = _Band()


class _FakePrimary:
    quality = _FakeQuality()
    bbox = (0, 0, 10, 10)


class _FakeEmbedding:
    dimension = 512


class _FakeModel:
    model_id = "test-model"


class _FakeImage:
    sha256 = "b" * 64


class _FakeFace:
    primary = _FakePrimary()
    embedding = _FakeEmbedding()
    model = _FakeModel()
    image = _FakeImage()
    faces_detected = 1


class _StubEngine:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def analyze(self, path):
        self.calls.append(path)
        return _FakeFace()


# ==========================================================================
# The input image is analysed exactly once
# ==========================================================================


class TestInputAnalysedOnce:
    """The verifier used to recompute the probe's perceptual hash inside the
    per-candidate loop: 28.8ms to re-derive the same value from the same
    unchanged file, 0.72s across 25 candidates."""

    def test_engine_is_called_once(self, tmp_path):
        from tracelock.service.analysis import analyze_input

        image = tmp_path / "probe.jpg"
        image.write_bytes(tiny_jpeg())
        engine = _StubEngine()
        analyze_input(engine, str(image))
        assert len(engine.calls) == 1

    def test_analysis_carries_everything_later_stages_need(self, tmp_path):
        from tracelock.service.analysis import analyze_input

        image = tmp_path / "probe.jpg"
        image.write_bytes(tiny_jpeg())
        analysis = analyze_input(_StubEngine(), str(image))

        assert analysis.sha256
        assert analysis.phash is not None
        assert (analysis.width, analysis.height) == (64, 64)
        assert analysis.embedding is not None
        payload = analysis.to_dict()
        for key in ("sha256", "phash", "width", "height", "quality"):
            assert key in payload

    def test_verifier_no_longer_recomputes_the_probe_hash(self):
        source = Path(
            sys.modules["tracelock.verification.verifier"].__file__
        ).read_text(encoding="utf-8")

        assert not re.search(r"_safe_phash\(\s*Path\(\s*probe", source), (
            "the probe hash is being recomputed inside the candidate loop"
        )
        assert "self._probe_phash" in source

    def test_an_undecodable_probe_degrades_instead_of_failing(self, tmp_path):
        """Losing near-duplicate detection is survivable; losing the run is not."""
        from tracelock.service.analysis import analyze_input

        broken = tmp_path / "broken.jpg"
        broken.write_bytes(b"not an image at all")
        assert analyze_input(_StubEngine(), str(broken)).phash is None


# ==========================================================================
# Image-level deduplication runs BEFORE the expensive embedding
# ==========================================================================


class TestImageLevelDeduplication:
    def test_near_duplicate_check_precedes_face_analysis(self):
        """29ms hash before a 1.73s embedding, not after it."""
        source = Path(
            sys.modules["tracelock.verification.verifier"].__file__
        ).read_text(encoding="utf-8")

        near_dup = source.index("_find_near_duplicates(candidate_id")
        analyzed = source.index("# -- STAGE: ANALYZED")
        assert near_dup < analyzed

    def test_a_duplicate_keeps_a_reference_to_the_original(self):
        """A duplicate is recorded as a duplicate OF something, not dropped."""
        from tracelock.verification.models import DuplicateInfo

        info = DuplicateInfo(
            is_exact_duplicate=False,
            duplicate_of_candidate_id="cand-1",
            near_duplicate_of_candidate_ids=("cand-1",),
        )
        assert info.duplicate_of_candidate_id == "cand-1"
        assert info.near_duplicate_of_candidate_ids == ("cand-1",)

    def test_visually_identical_files_share_a_perceptual_hash(self, tmp_path):
        """Re-encoding changes the bytes but must not change the image identity."""
        import io

        import cv2
        import numpy as np
        from PIL import Image

        from tracelock.verification.phash import compute_phash

        base = Image.new("RGB", (256, 256), (10, 20, 30))
        for x in range(256):
            for y in range(0, 256, 8):
                base.putpixel((x, y), (200, 180, 160))

        high = io.BytesIO()
        low = io.BytesIO()
        base.save(high, format="JPEG", quality=95)
        base.resize((200, 200)).save(low, format="JPEG", quality=60)

        def phash_of(buffer):
            array = cv2.imdecode(
                np.frombuffer(buffer.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            return compute_phash(array)

        # Different bytes, same picture -> small perceptual distance.
        assert phash_of(high).distance(phash_of(low)) <= 8


# ==========================================================================
# Prioritisation
# ==========================================================================


def make_candidate(image_url=None, post_url=None, thumbnail_url=None, provider="p"):
    from tracelock.core.models import Candidate

    return Candidate(
        provider=provider, image_url=image_url,
        post_url=post_url, thumbnail_url=thumbnail_url,
    )


class TestPrioritisation:
    def test_a_new_publisher_outranks_a_repeat(self):
        """Corroboration counts publishers, so unseen domains go first."""
        from tracelock.service.priority import prioritise

        order = prioritise([
            make_candidate("https://alpha.com/1.jpg", "https://alpha.com/p1"),
            make_candidate("https://alpha.com/2.jpg", "https://alpha.com/p2"),
            make_candidate("https://beta.com/1.jpg", "https://beta.com/p1"),
        ])
        hosts = [c.post_url.split("/")[2] for c in order]
        # The second alpha.com entry must fall behind the first beta.com one:
        # a repeat publisher adds nothing to corroboration.
        assert hosts.index("beta.com") < 2
        assert hosts == ["alpha.com", "beta.com", "alpha.com"]

    def test_multi_engine_agreement_raises_priority(self):
        from tracelock.discovery.multi import _canonical_url
        from tracelock.service.priority import score_candidates

        one = make_candidate("https://a.test/1.jpg", "https://a.test/p")
        two = make_candidate("https://b.test/1.jpg", "https://b.test/p")

        scored = {s.candidate.image_url: s for s in score_candidates(
            [one, two],
            found_by={_canonical_url(two.image_url): ["google_lens", "yandex_images"]},
        )}
        assert scored[two.image_url].score > scored[one.image_url].score
        assert any("found by" in r for r in scored[two.image_url].reasons)

    def test_a_full_image_outranks_a_thumbnail(self):
        from tracelock.service.priority import score_candidates

        full = make_candidate("https://a.test/1.jpg", "https://a.test/p")
        thumb = make_candidate(None, "https://b.test/p", "https://b.test/t.jpg")
        assert score_candidates([thumb, full])[0].candidate is full

    def test_declared_dimensions_come_from_the_provider_payload(self):
        from tracelock.core.models import Candidate
        from tracelock.service.priority import score_candidates

        big = Candidate(
            provider="p", image_url="https://a.test/big.jpg",
            post_url="https://a.test/p",
            raw_metadata={"original_width": 1200, "original_height": 900},
        )
        small = Candidate(
            provider="p", image_url="https://b.test/small.jpg",
            post_url="https://b.test/p",
            raw_metadata={"original_width": 90, "original_height": 60},
        )
        assert score_candidates([small, big])[0].candidate is big

    def test_ordering_is_deterministic(self):
        from tracelock.service.priority import prioritise

        candidates = [
            make_candidate(
                "https://s{0}.test/i.jpg".format(i), "https://s{0}.test/p".format(i)
            )
            for i in range(8)
        ]
        first = [c.image_url for c in prioritise(candidates)]
        for _ in range(5):
            assert [c.image_url for c in prioritise(candidates)] == first

    def test_no_invented_source_reliability_table(self):
        """Ranking sites by assumed trustworthiness would bias what is found."""
        source = Path(
            sys.modules["tracelock.service.priority"].__file__
        ).read_text(encoding="utf-8").lower()

        for banned in ("bbc.", "cnn.", "reuters", "trusted_domain", "reliability_score"):
            for line in source.splitlines():
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith('"'):
                    continue
                assert banned not in stripped, line

    def test_reordering_cannot_change_a_verdict(self):
        """Priority selects processing ORDER only -- never a score or a status."""
        source = Path(
            sys.modules["tracelock.service.priority"].__file__
        ).read_text(encoding="utf-8").lower()

        for forbidden in (
            "trust_score", "identity_probability", "similarity_floor",
            "verificationstatus", "verified_candidate",
        ):
            for line in source.splitlines():
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith('"'):
                    continue
                assert forbidden not in stripped, line


# ==========================================================================
# Timing instrumentation
# ==========================================================================


class TestPhaseTiming:
    def test_records_real_elapsed_time(self):
        from tracelock.service.timing import PhaseTimer

        timer = PhaseTimer()
        with timer.measure("work"):
            time.sleep(0.05)
        assert timer.get("work") >= 0.04

    def test_a_phase_that_never_finished_has_no_entry(self):
        """Absent, not a plausible-looking zero."""
        from tracelock.service.timing import PhaseTimer

        timer = PhaseTimer()
        timer.start("unfinished")
        names = {p["name"] for p in timer.to_dict()["phases"]}
        assert "unfinished" not in names
        assert "never_started" not in names

    def test_a_failing_phase_is_still_timed(self):
        from tracelock.service.timing import PhaseTimer

        timer = PhaseTimer()
        with pytest.raises(ValueError):
            with timer.measure("doomed"):
                time.sleep(0.02)
                raise ValueError("boom")
        assert timer.get("doomed") >= 0.01

    def test_overlap_is_reported_not_hidden(self):
        """Concurrent phases do not sum to the total, so say so explicitly."""
        from tracelock.service.timing import PhaseTimer

        timer = PhaseTimer()
        timer.start("a")
        timer.start("b")
        time.sleep(0.06)
        timer.stop("a")
        timer.stop("b")
        timer.finish()

        payload = timer.to_dict()
        assert payload["overlap_saved_seconds"] > 0
        assert "overlap" in payload["note"].lower()

    def test_timer_is_thread_safe(self):
        from tracelock.service.timing import PhaseTimer

        timer = PhaseTimer()
        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(200):
                    timer.add("parallel", 0.001)
                    timer.to_dict()
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors

    def test_no_fabricated_benchmark_values_in_the_module(self):
        """Every number reported must come from a clock, not a constant."""
        source = Path(
            sys.modules["tracelock.service.timing"].__file__
        ).read_text(encoding="utf-8")

        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"'):
                continue
            # No hardcoded plausible-looking durations.
            assert not re.search(r"=\s*[0-9]+\.[0-9]+\s*#?.*second", stripped.lower())


class TestPublisherKeyIsConsistent:
    def test_priority_and_stop_tracker_agree_on_what_a_publisher_is(self):
        """Both must key on registrable_domain with a host fallback.

        If they disagreed, prioritisation would optimise toward a definition of
        "independent publisher" that the stopping rule does not use.
        """
        from tracelock.service.priority import _publisher

        assert _publisher(make_candidate(
            "https://cdn.example.com/a.jpg", "https://news.example.com/story"
        )) == "example.com"

        # Unusual TLD: no registrable domain, so the host is used -- exactly
        # what StopTracker does.
        assert _publisher(make_candidate(
            "https://a.test/1.jpg", "https://a.test/p"
        )) == "a.test"


class TestTimerKindsCannotCollide:
    """One name cannot be both a wall-clock interval and a CPU sum.

    Mixing them silently produced 1,139,129 seconds once: start() stored a
    perf_counter origin, add() reset it to zero, and stop() then subtracted
    zero from a perf_counter value. It now raises instead.
    """

    def test_interval_then_accumulate_raises(self):
        from tracelock.service.timing import PhaseTimer

        timer = PhaseTimer()
        timer.start("phase")
        with pytest.raises(ValueError, match="interval"):
            timer.add("phase", 1.0)

    def test_accumulate_then_interval_raises(self):
        from tracelock.service.timing import PhaseTimer

        timer = PhaseTimer()
        timer.add("phase", 1.0)
        with pytest.raises(ValueError, match="accumulated"):
            timer.start("phase")

    def test_worker_cpu_is_excluded_from_the_overlap_figure(self):
        """CPU summed across threads is not wall-clock time saved."""
        from tracelock.service.timing import PhaseTimer

        timer = PhaseTimer()
        with timer.measure("wall"):
            time.sleep(0.05)
        timer.add("worker_cpu", 100.0)     # absurd on purpose
        timer.finish()

        payload = timer.to_dict()
        assert payload["worker_cpu_seconds"] == pytest.approx(100.0)
        # The 100s of worker CPU must not become 100s of "saved" wall-clock.
        assert payload["overlap_saved_seconds"] < 1.0

    def test_no_phase_reports_an_implausible_duration(self):
        """Guards the exact shape of the bug that was shipped."""
        from tracelock.service.timing import PhaseTimer

        timer = PhaseTimer()
        with timer.measure("a"):
            time.sleep(0.02)
        timer.add("b", 0.01)
        timer.finish()

        for phase in timer.to_dict()["phases"]:
            assert phase["seconds"] < 3600, phase
