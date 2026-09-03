"""Stage 2.5 -- the search-image guard.

Regression cover for a real failure: the gate validated the LOCAL file, found
a face, and passed -- while the URL served a faceless GitHub identicon. The
provider never sees the local file.

The face engine is injected, so every branch runs offline in milliseconds with
no 275 MB model load. HTTP is mocked with respx; nothing here touches the live
internet.
"""

from __future__ import annotations

import io

import httpx
import numpy as np
import pytest
import respx

from tracelock.acquisition.fetcher import FetchPolicy
from tracelock.discovery.search_image import (
    PHASH_SAME_IMAGE_MAX_DISTANCE,
    ProbeRelationship,
    SearchImageIssue,
    SearchImageWarning,
    check_search_image,
)
from tracelock.face.errors import NoFaceDetectedError, UnsupportedImageError

cv2 = pytest.importorskip("cv2")

URL = "https://example.test/avatar.jpg"
OPEN = FetchPolicy(block_private_targets=False)


# ==========================================================================
# Fixtures
# ==========================================================================


def photo_bytes(seed: int = 1, size: int = 240) -> bytes:
    """A textured JPEG. Stands in for a photograph."""
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 210, size=(size // 8, size // 8, 3), dtype=np.uint8)
    image = cv2.resize(base, (size, size), interpolation=cv2.INTER_LINEAR)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    return buf.tobytes()


def identicon_bytes() -> bytes:
    """A flat two-colour PNG -- the real GitHub identicon shape."""
    from PIL import Image

    image = Image.new("RGB", (420, 420), (240, 240, 240))
    for x in range(0, 420, 140):
        for y in range(0, 420, 140):
            if (x + y) % 280 == 0:
                for i in range(x, min(x + 140, 420)):
                    for j in range(y, min(y + 140, 420)):
                        image.putpixel((i, j), (89, 207, 132))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def unit_vector(seed: int, dim: int = 512) -> np.ndarray:
    vector = np.random.default_rng(seed).normal(size=dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


class FakeAnalysis:
    def __init__(self, vector, *, faces=1, det=0.9, quality=0.8, ambiguous=False,
                 margin=0.5, min_side=150.0):
        from tracelock.face.models import BoundingBox, Embedding

        self.embedding = Embedding(vector=np.asarray(vector, np.float32), model_id="fake:v1")
        self.faces_detected = faces
        self.primary = type("P", (), {
            "det_score": det,
            "bbox": BoundingBox(10, 10, 10 + min_side, 10 + min_side),
            "quality": type("Q", (), {
                "aggregate": quality,
                "band": type("B", (), {"value": "GOOD"})(),
            })(),
        })()
        self.selection = type("S", (), {"ambiguous": ambiguous, "margin": margin})()

    def warning_codes(self):
        return ()


class FakeEngine:
    """Injected face engine. `raises` makes it fail like the real one."""

    def __init__(self, vector=None, *, raises=None, **kwargs):
        self.vector = vector if vector is not None else unit_vector(1)
        self.raises = raises
        self.kwargs = kwargs
        self.analyzed_paths = []

    def analyze(self, path):
        self.analyzed_paths.append(str(path))
        if self.raises:
            raise self.raises
        return FakeAnalysis(self.vector, **self.kwargs)


def mock_url(content: bytes, *, status: int = 200, content_type: str = "image/jpeg"):
    respx.get(URL).mock(
        return_value=httpx.Response(
            status, content=content, headers={"content-type": content_type}
        )
    )


# ==========================================================================
# Hard failures
# ==========================================================================


class TestGateFailures:
    @respx.mock
    def test_faceless_image_is_rejected(self):
        """THE regression. A real identicon passed the old gate."""
        mock_url(identicon_bytes(), content_type="image/png")
        engine = FakeEngine(raises=NoFaceDetectedError("no face detected"))

        result = check_search_image(URL, engine, fetch_policy=OPEN)

        assert not result.ok
        assert result.issue is SearchImageIssue.NO_FACE_DETECTED
        assert "faceless" in result.detail.lower()

    @respx.mock
    def test_faceless_rejection_explains_the_consequence(self):
        mock_url(identicon_bytes(), content_type="image/png")
        engine = FakeEngine(raises=NoFaceDetectedError("no face"))
        result = check_search_image(URL, engine, fetch_policy=OPEN)
        # The explanation must say WHY this matters, not just what happened.
        assert "visually similar" in result.issue.explanation.lower()

    @respx.mock
    def test_http_error_is_unfetchable(self):
        mock_url(b"", status=404)
        result = check_search_image(URL, FakeEngine(), fetch_policy=OPEN)
        assert not result.ok
        assert result.issue is SearchImageIssue.URL_UNFETCHABLE
        assert result.http_status == 404

    @respx.mock
    def test_timeout_is_unfetchable(self):
        respx.get(URL).mock(side_effect=httpx.ReadTimeout("slow"))
        result = check_search_image(URL, FakeEngine(), fetch_policy=OPEN)
        assert result.issue is SearchImageIssue.URL_UNFETCHABLE

    @respx.mock
    def test_html_served_as_image_is_rejected(self):
        mock_url(b"<!DOCTYPE html><html>login</html>", content_type="image/jpeg")
        result = check_search_image(URL, FakeEngine(), fetch_policy=OPEN)
        assert not result.ok
        assert result.issue is SearchImageIssue.NOT_AN_IMAGE
        assert "HTML" in result.detail

    @respx.mock
    def test_engine_error_is_reported_distinctly(self):
        mock_url(photo_bytes())
        engine = FakeEngine(raises=UnsupportedImageError("decode blew up"))
        result = check_search_image(URL, engine, fetch_policy=OPEN)
        assert result.issue is SearchImageIssue.FACE_ANALYSIS_FAILED

    @respx.mock
    def test_ssrf_target_is_unfetchable(self):
        result = check_search_image(
            "http://169.254.169.254/latest/meta-data/",
            FakeEngine(),
            fetch_policy=FetchPolicy(block_private_targets=True),
        )
        assert result.issue is SearchImageIssue.URL_UNFETCHABLE

    def test_every_issue_has_a_real_explanation(self):
        for issue in SearchImageIssue:
            assert len(issue.explanation) > 40, issue.value


# ==========================================================================
# The guard validates the URL, not the local file
# ==========================================================================


class TestValidatesTheUrlNotTheLocalFile:
    @respx.mock
    def test_a_good_local_probe_cannot_rescue_a_faceless_url(self):
        """The exact production failure, reproduced.

        Local probe: a real face. URL: an identicon. The old gate passed
        because it only ever looked at the local file.
        """
        mock_url(identicon_bytes(), content_type="image/png")
        engine = FakeEngine(raises=NoFaceDetectedError("no face"))

        result = check_search_image(
            URL,
            engine,
            probe_analysis=FakeAnalysis(unit_vector(1)),  # a perfectly good probe
            probe_bytes=photo_bytes(),
            fetch_policy=OPEN,
        )
        assert not result.ok
        assert result.issue is SearchImageIssue.NO_FACE_DETECTED

    @respx.mock
    def test_the_engine_analyzes_the_downloaded_bytes(self):
        mock_url(photo_bytes())
        engine = FakeEngine()
        check_search_image(URL, engine, fetch_policy=OPEN)
        # One analyze call, on a temp copy of the DOWNLOADED bytes.
        assert len(engine.analyzed_paths) == 1
        assert "tracelock_search_image" in engine.analyzed_paths[0]

    @respx.mock
    def test_scratch_file_is_cleaned_up(self):
        from pathlib import Path

        mock_url(photo_bytes())
        engine = FakeEngine()
        check_search_image(URL, engine, fetch_policy=OPEN)
        assert not Path(engine.analyzed_paths[0]).exists()


# ==========================================================================
# Success + observations
# ==========================================================================


class TestGatePasses:
    @respx.mock
    def test_valid_face_image_passes(self):
        mock_url(photo_bytes())
        result = check_search_image(URL, FakeEngine(), fetch_policy=OPEN)

        assert result.ok
        assert result.issue is None
        assert result.faces_detected == 1
        assert result.image_format == "JPEG"
        assert result.content_sha256

    @respx.mock
    def test_clean_image_produces_no_warnings(self):
        mock_url(photo_bytes())
        result = check_search_image(URL, FakeEngine(), fetch_policy=OPEN)
        assert result.warnings == ()

    @respx.mock
    def test_multiple_faces_warns_but_passes(self):
        mock_url(photo_bytes())
        engine = FakeEngine(faces=4)
        result = check_search_image(URL, engine, fetch_policy=OPEN)
        assert result.ok
        assert SearchImageWarning.MULTIPLE_FACES in result.warnings

    @respx.mock
    def test_low_quality_warns_but_passes(self):
        # Between the GATE floor (0.15) and the WARN threshold (0.35): usable
        # but worth flagging. Below 0.15 it is rejected outright -- see
        # TestInputFitnessGate.
        mock_url(photo_bytes())
        result = check_search_image(URL, FakeEngine(quality=0.25), fetch_policy=OPEN)
        assert result.ok
        assert SearchImageWarning.LOW_FACE_QUALITY in result.warnings

    @respx.mock
    def test_small_face_warns_but_passes(self):
        # Between the GATE floor (50px) and the WARN threshold (80px).
        mock_url(photo_bytes())
        result = check_search_image(URL, FakeEngine(min_side=60.0), fetch_policy=OPEN)
        assert result.ok
        assert SearchImageWarning.SMALL_FACE in result.warnings

    @respx.mock
    def test_ambiguous_selection_warns_but_passes(self):
        mock_url(photo_bytes())
        engine = FakeEngine(faces=2, ambiguous=True, margin=0.02)
        result = check_search_image(URL, engine, fetch_policy=OPEN)
        assert result.ok
        assert SearchImageWarning.AMBIGUOUS_PRIMARY_FACE in result.warnings

    @respx.mock
    def test_every_warning_carries_a_detail_line(self):
        mock_url(photo_bytes())
        # Above both gate floors so warnings fire without rejection.
        engine = FakeEngine(faces=3, quality=0.25, min_side=60.0, ambiguous=True)
        result = check_search_image(URL, engine, fetch_policy=OPEN)
        assert len(result.warnings) == len(result.warning_details)
        assert all(len(d) > 15 for d in result.warning_details)


# ==========================================================================
# THE INTEGRITY BOUNDARY
# ==========================================================================


class TestNoIdentityClaims:
    """The guard measures the probe relationship. It never judges it."""

    @respx.mock
    def test_relationship_is_flagged_as_not_an_identity_claim(self):
        mock_url(photo_bytes())
        result = check_search_image(
            URL,
            FakeEngine(unit_vector(1)),
            probe_analysis=FakeAnalysis(unit_vector(1)),
            probe_bytes=photo_bytes(),
            fetch_policy=OPEN,
        )
        assert result.probe_relationship.is_identity_claim is False
        assert result.probe_relationship.to_dict()["is_identity_claim"] is False

    @respx.mock
    def test_a_dissimilar_probe_does_not_fail_the_gate(self):
        """A LOW similarity must NOT reject. That would be an identity verdict
        rendered by an uncalibrated threshold."""
        mock_url(photo_bytes(1))
        result = check_search_image(
            URL,
            FakeEngine(unit_vector(999)),                 # unrelated face
            probe_analysis=FakeAnalysis(unit_vector(1)),
            probe_bytes=photo_bytes(2),
            fetch_policy=OPEN,
        )
        assert result.ok
        assert result.issue is None
        assert result.probe_relationship.cosine_similarity is not None

    @respx.mock
    def test_a_similar_probe_does_not_pass_any_extra_gate(self):
        mock_url(photo_bytes(1))
        result = check_search_image(
            URL,
            FakeEngine(unit_vector(1)),
            probe_analysis=FakeAnalysis(unit_vector(1)),
            probe_bytes=photo_bytes(1),
            fetch_policy=OPEN,
        )
        assert result.ok
        assert result.probe_relationship.cosine_similarity == pytest.approx(1.0, abs=1e-6)

    @respx.mock
    def test_same_image_is_a_visual_fact_not_an_identity_verdict(self):
        # is_same_image is derived from pHash alone -- pixels, not people.
        data = photo_bytes(5)
        mock_url(data)
        result = check_search_image(
            URL,
            FakeEngine(unit_vector(1)),
            probe_analysis=FakeAnalysis(unit_vector(777)),  # DIFFERENT face
            probe_bytes=data,                               # SAME pixels
            fetch_policy=OPEN,
        )
        assert result.probe_relationship.is_same_image is True
        assert SearchImageWarning.DIFFERS_FROM_PROBE not in result.warnings

    @respx.mock
    def test_different_image_warns_factually(self):
        mock_url(photo_bytes(1))
        result = check_search_image(
            URL,
            FakeEngine(unit_vector(1)),
            probe_analysis=FakeAnalysis(unit_vector(1)),  # SAME face
            probe_bytes=photo_bytes(999),                 # DIFFERENT pixels
            fetch_policy=OPEN,
        )
        assert result.probe_relationship.is_same_image is False
        assert SearchImageWarning.DIFFERS_FROM_PROBE in result.warnings
        # Worded as provenance, not identity.
        detail = " ".join(result.warning_details).lower()
        assert "same image" in detail or "different pictures" in detail

    @respx.mock
    def test_relationship_is_absent_when_no_probe_supplied(self):
        mock_url(photo_bytes())
        result = check_search_image(URL, FakeEngine(), fetch_policy=OPEN)
        assert result.probe_relationship is None

    @respx.mock
    def test_relationship_note_disclaims_identity(self):
        mock_url(photo_bytes())
        result = check_search_image(
            URL, FakeEngine(), probe_analysis=FakeAnalysis(unit_vector(1)),
            fetch_policy=OPEN,
        )
        assert "no identity conclusion" in result.probe_relationship.note.lower()

    def test_module_declares_no_identity_threshold(self):
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1]
            / "src" / "tracelock" / "discovery" / "search_image.py"
        ).read_text(encoding="utf-8")

        for banned in ("IDENTITY_THRESHOLD", "SAME_PERSON", "MATCH_THRESHOLD",
                       "is_same_person", "SIMILARITY_FLOOR"):
            assert banned not in source, (
                "{0} found in the guard: it must not decide identity".format(banned)
            )

    def test_similarity_is_never_compared_to_a_constant(self):
        """`cosine_similarity` may be MEASURED but never thresholded here."""
        from pathlib import Path
        import re

        source = (
            Path(__file__).resolve().parents[1]
            / "src" / "tracelock" / "discovery" / "search_image.py"
        ).read_text(encoding="utf-8")

        # e.g. `similarity > 0.5`, `sim >= 0.35`
        offenders = re.findall(
            r"(?:similarity|sim)\s*[<>]=?\s*[0-9]", source, flags=re.IGNORECASE
        )
        assert not offenders, "identity thresholding found: {0}".format(offenders)


class TestSerialization:
    @respx.mock
    def test_check_serializes(self):
        import json

        mock_url(photo_bytes())
        result = check_search_image(
            URL, FakeEngine(), probe_analysis=FakeAnalysis(unit_vector(1)),
            probe_bytes=photo_bytes(), fetch_policy=OPEN,
        )
        payload = result.to_dict()
        assert json.dumps(payload)
        assert payload["ok"] is True
        assert payload["probe_relationship"]["is_identity_claim"] is False

    @respx.mock
    def test_failure_serializes_with_explanation(self):
        import json

        mock_url(identicon_bytes(), content_type="image/png")
        engine = FakeEngine(raises=NoFaceDetectedError("no face"))
        payload = check_search_image(URL, engine, fetch_policy=OPEN).to_dict()

        assert json.dumps(payload)
        assert payload["issue"] == "NO_FACE_DETECTED"
        assert len(payload["issue_explanation"]) > 40


class TestConstants:
    def test_same_image_threshold_matches_the_phash_convention(self):
        from tracelock.verification.phash import NEAR_DUPLICATE_MAX_DISTANCE

        assert PHASH_SAME_IMAGE_MAX_DISTANCE == NEAR_DUPLICATE_MAX_DISTANCE

    def test_probe_relationship_defaults_to_no_identity_claim(self):
        assert ProbeRelationship(None, None, None).is_identity_claim is False


class TestInputFitnessGate:
    """A detection is not automatically a usable search seed.

    Regression for the real GitHub-logo case: SCRFD fired at det=0.511 on a
    32x32 icon, producing a 10px 'face' at quality 0.002. A presence-only
    guard let it through and the search returned pages about the logo.

    These floors judge IMAGE FITNESS. They make no identity claim and never
    consult the probe.
    """

    @respx.mock
    def test_icon_scale_face_is_rejected(self):
        mock_url(photo_bytes())
        engine = FakeEngine(min_side=10.0, quality=0.002, det=0.511)
        result = check_search_image(URL, engine, fetch_policy=OPEN)

        assert not result.ok
        assert result.issue is SearchImageIssue.FACE_NOT_USABLE
        assert "10px" in result.detail

    @respx.mock
    def test_tiny_face_alone_is_enough_to_reject(self):
        mock_url(photo_bytes())
        result = check_search_image(
            URL, FakeEngine(min_side=20.0, quality=0.9), fetch_policy=OPEN
        )
        assert result.issue is SearchImageIssue.FACE_NOT_USABLE

    @respx.mock
    def test_terrible_quality_alone_is_enough_to_reject(self):
        mock_url(photo_bytes())
        result = check_search_image(
            URL, FakeEngine(min_side=200.0, quality=0.01), fetch_policy=OPEN
        )
        assert result.issue is SearchImageIssue.FACE_NOT_USABLE

    @respx.mock
    def test_a_real_avatar_passes_comfortably(self):
        # Measured on the live GitHub avatar: quality 0.974, det 0.880.
        mock_url(photo_bytes())
        result = check_search_image(
            URL, FakeEngine(min_side=150.0, quality=0.974, det=0.880),
            fetch_policy=OPEN,
        )
        assert result.ok

    @respx.mock
    def test_floors_are_configurable(self):
        mock_url(photo_bytes())
        engine = FakeEngine(min_side=30.0, quality=0.05)

        assert not check_search_image(URL, engine, fetch_policy=OPEN).ok
        assert check_search_image(
            URL, engine, fetch_policy=OPEN, min_face_px=10, min_quality=0.01
        ).ok

    @respx.mock
    def test_rejection_records_the_measurements(self):
        mock_url(photo_bytes())
        result = check_search_image(
            URL, FakeEngine(min_side=10.0, quality=0.002), fetch_policy=OPEN
        )
        assert result.face_min_side_px == 10.0
        assert result.quality_aggregate == pytest.approx(0.002)
        assert result.faces_detected == 1

    @respx.mock
    def test_fitness_gate_ignores_the_probe_entirely(self):
        # An identical probe must not rescue an unusable search image, and a
        # dissimilar one must not condemn a usable one.
        mock_url(photo_bytes())
        vector = unit_vector(1)

        unusable = check_search_image(
            URL, FakeEngine(vector, min_side=10.0, quality=0.002),
            probe_analysis=FakeAnalysis(vector), probe_bytes=photo_bytes(),
            fetch_policy=OPEN,
        )
        assert unusable.issue is SearchImageIssue.FACE_NOT_USABLE

        usable = check_search_image(
            URL, FakeEngine(unit_vector(999), min_side=150.0, quality=0.9),
            probe_analysis=FakeAnalysis(vector), probe_bytes=photo_bytes(2),
            fetch_policy=OPEN,
        )
        assert usable.ok

    @respx.mock
    def test_fitness_rejection_is_not_phrased_as_an_identity_verdict(self):
        mock_url(photo_bytes())
        result = check_search_image(
            URL, FakeEngine(min_side=10.0, quality=0.002), fetch_policy=OPEN
        )
        explanation = result.issue.explanation.lower()
        assert "image fitness" in explanation
        assert "not about who is depicted" in explanation
