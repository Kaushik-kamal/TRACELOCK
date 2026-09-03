"""MODE B -- local analysis, and the privacy boundary it protects.

THE GUARANTEE
-------------
A webcam frame or a local upload NEVER leaves the machine unless the operator
explicitly supplies a public URL themselves. The tempting shortcut -- quietly
uploading to a paste host so discovery "just works" -- would be a silent
biometric disclosure, and this suite exists to make that impossible to
introduce by accident.

`TestNoSilentUpload` proves it by making every HTTP call raise, then running
the whole local path and asserting it completes.
"""

from __future__ import annotations

import io
import json

import pytest

from tracelock.service.inputs import from_upload, from_webcam
from tracelock.service.local_analysis import (
    FORBIDDEN_KEYS,
    FabricatedFindingError,
    LocalAnalysisReport,
    analyse_locally,
    assert_no_fabricated_findings,
)

pytest.importorskip("fastapi")
pytest.importorskip("cv2")
httpx = pytest.importorskip("httpx")
respx = pytest.importorskip("respx")

from fastapi.testclient import TestClient  # noqa: E402


def jpeg(width: int = 400, height: int = 400) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (90, 110, 140)).save(buffer, "JPEG")
    return buffer.getvalue()


class FakeQualityMetric:
    def __init__(self, name, score):
        self.name, self.score = name, score

    def to_dict(self):
        return {"name": self.name, "score": self.score, "raw_value": 1.0,
                "unit": "x", "uncalibrated": False}


class FakeEngine:
    """Face engine stub. Makes no network call, by construction."""

    def __init__(self, *, raise_no_face: bool = False):
        self.raise_no_face = raise_no_face

    def analyze(self, path):
        from tracelock.face.errors import NoFaceDetectedError
        from tracelock.face.models import BoundingBox, Embedding
        import numpy as np

        if self.raise_no_face:
            raise NoFaceDetectedError("no face")

        vector = np.random.default_rng(1).normal(size=512).astype(np.float32)
        vector /= np.linalg.norm(vector)

        quality = type("Q", (), {
            "aggregate": 0.82,
            "band": type("B", (), {"value": "EXCELLENT"})(),
            "metrics": (FakeQualityMetric("sharpness", 0.9),
                        FakeQualityMetric("face_pixel_size", 1.0)),
        })()
        primary = type("P", (), {
            "det_score": 0.91,
            "bbox": BoundingBox(10, 10, 210, 210),
            "quality": quality,
            "pose": None,
        })()

        return type("A", (), {
            "faces_detected": 1,
            "primary": primary,
            "embedding": Embedding(vector=vector, model_id="fake:v1"),
            "image": type("I", (), {"width": 400, "height": 400,
                                    "path": str(path), "sha256": "a" * 64})(),
            "warning_codes": lambda self=None: (),
        })()


@pytest.fixture
def client():
    from tracelock.api import create_app

    return TestClient(create_app())


# ==========================================================================
# 1. Webcam / upload never silently triggers a network upload
# ==========================================================================


class TestNoSilentUpload:
    """The privacy guarantee, proved by making the network unusable."""

    @pytest.fixture
    def no_network(self, monkeypatch):
        """Any outbound HTTP call becomes a hard failure."""

        def forbidden(*args, **kwargs):
            raise AssertionError(
                "an outbound HTTP request was attempted during local-only work"
            )

        for target in ("get", "post", "put", "request", "stream"):
            monkeypatch.setattr(httpx, target, forbidden, raising=False)
        monkeypatch.setattr(httpx.Client, "send", forbidden, raising=False)
        monkeypatch.setattr(httpx.Client, "request", forbidden, raising=False)
        monkeypatch.setattr(httpx.Client, "stream", forbidden, raising=False)

        # The temporary-hosting path must never be reached either.
        import tracelock.discovery.hosting as hosting

        monkeypatch.setattr(
            hosting, "upload_temporary", forbidden, raising=False
        )

    def test_upload_normalization_makes_no_request(self, tmp_path, no_network):
        trace = from_upload(jpeg(), "a.jpg", tmp_path)
        assert trace.local_path
        assert trace.image_url is None

    def test_webcam_normalization_makes_no_request(self, tmp_path, no_network):
        trace = from_webcam(jpeg(), tmp_path)
        assert trace.image_url is None

    def test_local_analysis_makes_no_request(self, tmp_path, no_network):
        trace = from_webcam(jpeg(), tmp_path)
        report = analyse_locally(trace, FakeEngine())
        assert report.mode == "LOCAL_ANALYSIS"
        assert report.face_detected is True

    def test_report_declares_zero_network_requests(self, tmp_path, no_network):
        report = analyse_locally(from_upload(jpeg(), "a.jpg", tmp_path), FakeEngine())
        privacy = report.to_dict()["privacy"]
        assert privacy["network_requests_made"] == 0
        assert privacy["image_transmitted"] is False
        assert privacy["status"] == "LOCAL ONLY"

    def test_privacy_note_states_the_guarantee(self, tmp_path):
        report = analyse_locally(from_upload(jpeg(), "a.jpg", tmp_path), FakeEngine())
        note = report.to_dict()["privacy"]["note"].lower()
        assert "does not silently upload" in note
        assert "never left your machine" in note

    def test_hosting_module_is_not_imported_by_the_local_path(self):
        # The upload path exists for the CLI gate, but the local-analysis
        # module must not reach for it.
        import inspect

        from tracelock.service import local_analysis

        source = inspect.getsource(local_analysis)
        assert "upload_temporary" not in source
        assert "hosting" not in source


# ==========================================================================
# 2 & 3. No fake candidates, no fake trust score
# ==========================================================================


class TestNoFabricatedFindings:
    def test_report_has_no_trust_score_attribute(self, tmp_path):
        report = analyse_locally(from_upload(jpeg(), "a.jpg", tmp_path), FakeEngine())
        # ABSENT, not empty: there is nowhere for a score to live.
        assert not hasattr(report, "trust_score")
        assert not hasattr(report, "candidates")
        assert not hasattr(report, "evidence")

    def test_serialized_report_carries_no_score_or_candidates(self, tmp_path):
        payload = analyse_locally(
            from_upload(jpeg(), "a.jpg", tmp_path), FakeEngine()
        ).to_dict()
        for key in FORBIDDEN_KEYS:
            assert key not in payload, key
        assert key_absent_everywhere(payload, "trust_score")
        assert key_absent_everywhere(payload, "candidates")

    def test_guard_rejects_an_injected_trust_score(self):
        with pytest.raises(FabricatedFindingError, match="trust_score"):
            assert_no_fabricated_findings({"trust_score": {"score": 90}})

    def test_guard_rejects_a_nested_candidate_list(self):
        with pytest.raises(FabricatedFindingError, match="candidates"):
            assert_no_fabricated_findings({"a": {"b": {"candidates": []}}})

    def test_guard_rejects_findings_inside_a_list(self):
        with pytest.raises(FabricatedFindingError):
            assert_no_fabricated_findings({"items": [{"evidence": {}}]})

    def test_guard_names_the_path(self):
        with pytest.raises(FabricatedFindingError, match=r"\$\.a\.b"):
            assert_no_fabricated_findings({"a": {"b": {"trust_score": 1}}})

    def test_discovery_is_explicitly_marked_not_performed(self, tmp_path):
        payload = analyse_locally(
            from_upload(jpeg(), "a.jpg", tmp_path), FakeEngine()
        ).to_dict()
        discovery = payload["public_discovery"]
        assert discovery["performed"] is False
        assert discovery["status"] == "NOT PERFORMED"
        assert "cannot receive a file" in discovery["reason"]

    def test_no_face_still_produces_a_useful_report_not_an_error(self, tmp_path):
        report = analyse_locally(
            from_upload(jpeg(), "a.jpg", tmp_path), FakeEngine(raise_no_face=True)
        )
        payload = report.to_dict()
        assert payload["face"]["detected"] is False
        assert payload["image"]["width"] > 0  # image facts still measured
        assert payload["fingerprint"]["image_sha256"]


def key_absent_everywhere(payload, key) -> bool:
    if isinstance(payload, dict):
        if key in payload:
            return False
        return all(key_absent_everywhere(v, key) for v in payload.values())
    if isinstance(payload, list):
        return all(key_absent_everywhere(v, key) for v in payload)
    return True


# ==========================================================================
# Report content -- must be useful, not an error page
# ==========================================================================


class TestReportIsUseful:
    def test_reports_the_image_facts_the_ui_shows(self, tmp_path):
        payload = analyse_locally(
            from_upload(jpeg(640, 480), "a.jpg", tmp_path), FakeEngine()
        ).to_dict()
        image = payload["image"]
        assert image["width"] == 640 and image["height"] == 480
        assert image["dimensions"] == "640 x 480"
        assert image["format"] == "JPEG"
        assert image["sha256_short"].endswith("…")

    def test_reports_face_measurements(self, tmp_path):
        face = analyse_locally(
            from_upload(jpeg(), "a.jpg", tmp_path), FakeEngine()
        ).to_dict()["face"]
        assert face["detected"] is True
        assert face["count"] == 1
        assert face["quality"] == pytest.approx(0.82)
        assert face["quality_band"] == "EXCELLENT"
        assert face["quality_metrics"]

    def test_fingerprint_commits_but_never_exposes_the_embedding(self, tmp_path):
        payload = analyse_locally(
            from_upload(jpeg(), "a.jpg", tmp_path), FakeEngine()
        ).to_dict()
        fingerprint = payload["fingerprint"]
        assert len(fingerprint["embedding_quantized_sha256"]) == 64
        assert fingerprint["perceptual_hash"]
        assert "vector" not in json.dumps(payload)

    def test_serializes(self, tmp_path):
        payload = analyse_locally(
            from_upload(jpeg(), "a.jpg", tmp_path), FakeEngine()
        ).to_dict()
        assert json.dumps(payload)


# ==========================================================================
# 4 & 5. Public discovery requires an explicit URL, validated by Stage 2.5
# ==========================================================================


class TestPublicDiscoveryRequiresAnExplicitUrl:
    def test_local_input_reports_it_needs_hosting(self, tmp_path):
        assert from_webcam(jpeg(), tmp_path).needs_hosting is True
        assert from_upload(jpeg(), "a.jpg", tmp_path).needs_hosting is True

    def test_api_local_analysis_route_needs_no_url(self, client):
        selected = client.post(
            "/api/input/upload", files={"file": ("a.jpg", jpeg(), "image/jpeg")}
        ).json()
        assert selected["needs_hosting"] is True

        report = client.post(
            "/api/analyze/local", data={"sha256": selected["sha256"]}
        ).json()
        assert report["mode"] == "LOCAL_ANALYSIS"
        assert report["public_discovery"]["performed"] is False

    def test_link_url_route_goes_through_stage_2_5(self, client, monkeypatch):
        """Requirement 5: the URL is validated by the existing guard."""
        calls = {}

        def spy(url, engine, **kwargs):
            calls["url"] = url
            calls["probe_bytes"] = kwargs.get("probe_bytes")
            from tracelock.discovery.search_image import (
                SearchImageCheck,
                SearchImageIssue,
            )

            return SearchImageCheck(
                url=url, ok=False,
                issue=SearchImageIssue.NO_FACE_DETECTED,
                detail="no face",
            )

        import tracelock.discovery.search_image as guard

        monkeypatch.setattr(guard, "check_search_image", spy)

        selected = client.post(
            "/api/input/upload", files={"file": ("a.jpg", jpeg(), "image/jpeg")}
        ).json()
        response = client.post(
            "/api/input/link-url",
            data={"sha256": selected["sha256"], "url": "https://example.com/x.jpg"},
        )

        assert calls["url"] == "https://example.com/x.jpg"
        # The local image is passed so the guard can compare them perceptually.
        assert calls["probe_bytes"] is not None
        assert response.status_code == 400
        assert response.json()["issue"] == "NO_FACE_DETECTED"

    def test_link_url_rejects_an_unusable_link_with_a_readable_message(
        self, client, monkeypatch
    ):
        import tracelock.discovery.search_image as guard
        from tracelock.discovery.search_image import (
            SearchImageCheck,
            SearchImageIssue,
        )

        monkeypatch.setattr(
            guard, "check_search_image",
            lambda url, engine, **kw: SearchImageCheck(
                url=url, ok=False, issue=SearchImageIssue.URL_UNFETCHABLE,
                detail="404",
            ),
        )
        selected = client.post(
            "/api/input/upload", files={"file": ("a.jpg", jpeg(), "image/jpeg")}
        ).json()
        payload = client.post(
            "/api/input/link-url",
            data={"sha256": selected["sha256"], "url": "https://example.com/x.jpg"},
        ).json()

        assert payload["ok"] is False
        assert "could not be downloaded" in payload["error"]

    def test_successful_link_makes_the_input_investigable(self, client, monkeypatch):
        import tracelock.discovery.search_image as guard
        from tracelock.discovery.search_image import (
            ProbeRelationship,
            SearchImageCheck,
        )

        monkeypatch.setattr(
            guard, "check_search_image",
            lambda url, engine, **kw: SearchImageCheck(
                url=url, ok=True, image_format="JPEG", width=400, height=400,
                byte_size=1234, faces_detected=1, det_score=0.9,
                quality_aggregate=0.8, quality_band="EXCELLENT",
                face_min_side_px=200.0,
                probe_relationship=ProbeRelationship(
                    cosine_similarity=0.99, phash_distance=2, is_same_image=True
                ),
            ),
        )
        selected = client.post(
            "/api/input/upload", files={"file": ("a.jpg", jpeg(), "image/jpeg")}
        ).json()
        assert selected["needs_hosting"] is True

        payload = client.post(
            "/api/input/link-url",
            data={"sha256": selected["sha256"], "url": "https://example.com/x.jpg"},
        ).json()

        assert payload["ok"] is True
        # The linked input now HAS a public URL, so Mode A becomes available.
        assert payload["input"]["needs_hosting"] is False
        assert payload["input"]["image_url"] == "https://example.com/x.jpg"
        assert payload["input"]["provenance"]["public_url_supplied_by"] == "operator"

    def test_comparison_is_visual_and_makes_no_identity_claim(
        self, client, monkeypatch
    ):
        import tracelock.discovery.search_image as guard
        from tracelock.discovery.search_image import (
            ProbeRelationship,
            SearchImageCheck,
        )

        monkeypatch.setattr(
            guard, "check_search_image",
            lambda url, engine, **kw: SearchImageCheck(
                url=url, ok=True, image_format="JPEG", width=400, height=400,
                faces_detected=1, det_score=0.9, quality_aggregate=0.8,
                quality_band="GOOD",
                probe_relationship=ProbeRelationship(
                    cosine_similarity=0.42, phash_distance=38, is_same_image=False
                ),
            ),
        )
        selected = client.post(
            "/api/input/upload", files={"file": ("a.jpg", jpeg(), "image/jpeg")}
        ).json()
        comparison = client.post(
            "/api/input/link-url",
            data={"sha256": selected["sha256"], "url": "https://example.com/x.jpg"},
        ).json()["comparison"]

        assert comparison["is_identity_claim"] is False
        assert comparison["phash_distance"] == 38
        assert comparison["is_same_image"] is False
        assert "DIFFERENT image" in comparison["verdict"]
        # It describes PIXELS, not people.
        assert "no claim about who is depicted" in comparison["note"]

    def test_link_url_404s_for_an_unknown_image(self, client):
        response = client.post(
            "/api/input/link-url",
            data={"sha256": "0" * 64, "url": "https://example.com/x.jpg"},
        )
        assert response.status_code == 404


class TestUiOffersBothModes:
    """Source-level guards on the browser bundle."""

    def _read(self, name: str) -> str:
        from pathlib import Path

        return (
            Path(__file__).resolve().parents[1] / "web" / name
        ).read_text(encoding="utf-8")

    def test_markup_offers_every_choice(self):
        """A local image now has THREE routes, not two.

        The public-URL button was relabelled when consented publishing was
        added, so this asserts the intent -- offline analysis, consented
        publishing, and bring-your-own-URL -- rather than one exact string.
        """
        html = self._read("index.html")

        assert "LOCAL ANALYSIS" in html                  # stay offline
        assert "START PUBLIC DISCOVERY" in html          # publish, with consent
        assert "I HAVE A PUBLIC URL" in html             # already public

    def test_public_discovery_is_gated_behind_consent(self):
        """The new route must not bypass the privacy guarantee."""
        html = self._read("index.html")
        assert 'id="mode-consent"' in html
        assert "CONSENT &amp; SEARCH" in html
        assert 'id="consent-cancel"' in html

    def test_markup_states_the_privacy_guarantee(self):
        html = self._read("index.html").lower()
        assert "does not silently upload biometric images" in html
        assert "never leave this machine" in html

    def test_js_renders_a_local_report_not_an_error(self):
        js = self._read("app.js")
        assert "renderLocalAnalysis" in js
        assert "NOT PERFORMED" in js

    def test_js_local_report_shows_no_trust_score(self):
        js = self._read("app.js")
        start = js.index("function renderLocalAnalysis")
        end = js.index("$$(\"[data-goto]\")", start)
        body = js[start:end]
        assert "trust_score" not in body
        assert "No trust score is shown" in body
