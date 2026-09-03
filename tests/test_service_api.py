"""Service layer and HTTP API.

The load-bearing test here is TestSingleNormalizationPoint: every input source
must converge on ONE TraceInput shape, because that is what stops five parallel
pipelines growing out of five input buttons.

No network. No face model. The routes that need the engine are exercised
through the pure normalization path; live behaviour is covered by the CLI.
"""

from __future__ import annotations

import io
import json

import pytest

from tracelock.service.inputs import (
    MAX_UPLOAD_BYTES,
    InputError,
    InputKind,
    SourceType,
    TraceInput,
    drive_direct_url,
    extract_drive_file_id,
    from_bytes,
    from_google_drive,
    from_upload,
    from_url,
    from_webcam,
)

pytest.importorskip("fastapi")
pytest.importorskip("cv2")
httpx = pytest.importorskip("httpx")
respx = pytest.importorskip("respx")

from fastapi.testclient import TestClient  # noqa: E402

from tracelock.acquisition.fetcher import FetchPolicy  # noqa: E402

# Disables ONLY the SSRF host check, so respx-mocked hosts are reachable.
OPEN = FetchPolicy(block_private_targets=False)


def jpeg(width: int = 300, height: int = 300) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (90, 110, 140)).save(buffer, "JPEG")
    return buffer.getvalue()


def png() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (200, 200), (30, 160, 90)).save(buffer, "PNG")
    return buffer.getvalue()


@pytest.fixture
def client():
    from tracelock.api import create_app

    return TestClient(create_app())


# ==========================================================================
# THE ARCHITECTURAL INVARIANT
# ==========================================================================


class TestSingleNormalizationPoint:
    """Five sources in, one object out.

    If these ever diverge, the "one pipeline" guarantee is gone and the input
    buttons have quietly become separate systems.
    """

    def test_every_source_produces_a_trace_input(self, tmp_path):
        made = [
            from_upload(jpeg(), "a.jpg", tmp_path),
            from_webcam(jpeg(), tmp_path),
            from_bytes(
                jpeg(), source_type=SourceType.EXAMPLE, filename="e.jpg",
                work_dir=tmp_path, kind=InputKind.DEMO_EXAMPLE,
            ),
        ]
        for trace in made:
            assert isinstance(trace, TraceInput)

    def test_all_sources_expose_the_same_public_fields(self, tmp_path):
        upload = from_upload(jpeg(), "a.jpg", tmp_path).to_dict()
        webcam = from_webcam(jpeg(), tmp_path).to_dict()
        assert set(upload) == set(webcam)

    def test_pipeline_only_ever_reads_local_path(self, tmp_path):
        # Whatever the source, a real file exists for the engine to open.
        from pathlib import Path

        for trace in (from_upload(jpeg(), "a.jpg", tmp_path),
                      from_webcam(jpeg(), tmp_path)):
            assert Path(trace.local_path).is_file()

    def test_identical_bytes_deduplicate_regardless_of_source(self, tmp_path):
        data = jpeg()
        assert (
            from_upload(data, "a.jpg", tmp_path).sha256
            == from_webcam(data, tmp_path).sha256
        )

    def test_source_type_is_preserved(self, tmp_path):
        assert from_upload(jpeg(), "a.jpg", tmp_path).source_type is SourceType.UPLOAD
        assert from_webcam(jpeg(), tmp_path).source_type is SourceType.WEBCAM


# ==========================================================================
# Validation -- same rules as candidate acquisition
# ==========================================================================


class TestInputValidation:
    def test_accepts_jpeg_and_png(self, tmp_path):
        assert from_upload(jpeg(), "a.jpg", tmp_path).mime_type == "image/jpeg"
        assert from_upload(png(), "a.png", tmp_path).mime_type == "image/png"

    def test_rejects_html_pretending_to_be_an_image(self, tmp_path):
        with pytest.raises(InputError) as caught:
            from_upload(b"<!DOCTYPE html><html>hi</html>", "photo.jpg", tmp_path)
        assert "not an image" in caught.value.message.lower()

    def test_rejects_empty_file(self, tmp_path):
        with pytest.raises(InputError, match="empty"):
            from_upload(b"", "a.jpg", tmp_path)

    def test_rejects_oversized_file(self, tmp_path):
        with pytest.raises(InputError, match="larger than"):
            from_upload(b"\xff\xd8\xff" + b"x" * MAX_UPLOAD_BYTES, "a.jpg", tmp_path)

    def test_rejects_tiny_image(self, tmp_path):
        with pytest.raises(InputError, match="too small"):
            from_upload(jpeg(16, 16), "a.jpg", tmp_path)

    def test_filename_extension_is_not_trusted(self, tmp_path):
        # A PNG named .jpg is still a PNG. Magic bytes decide.
        assert from_upload(png(), "lying.jpg", tmp_path).mime_type == "image/png"

    def test_errors_are_readable_by_a_non_engineer(self, tmp_path):
        with pytest.raises(InputError) as caught:
            from_upload(b"not an image at all", "x.jpg", tmp_path)
        message = caught.value.message
        assert message[0].isupper() and message.endswith(".")
        assert "0x" not in message and "Traceback" not in message


# ==========================================================================
# Demo provenance
# ==========================================================================


class TestDemoProvenance:
    """A staged INPUT must never be readable as staged EVIDENCE."""

    def test_demo_kind_is_carried(self, tmp_path):
        trace = from_bytes(
            jpeg(), source_type=SourceType.EXAMPLE, filename="e.jpg",
            work_dir=tmp_path, kind=InputKind.DEMO_EXAMPLE,
        )
        assert trace.is_demo
        assert trace.to_dict()["kind"] == "DEMO_EXAMPLE"

    def test_demo_notice_says_discovery_is_still_live(self):
        notice = InputKind.DEMO_EXAMPLE.notice.lower()
        assert "live" in notice
        assert "only the starting image" in notice

    def test_organic_input_carries_no_demo_notice(self, tmp_path):
        assert from_upload(jpeg(), "a.jpg", tmp_path).to_dict()["kind_notice"] == ""

    def test_input_kind_is_separate_from_calibration_dataset_kind(self):
        # They answer different questions and must not be merged: one governs
        # how an image reached us, the other whether a calibration is citable.
        from tracelock.calibration.contract import DatasetKind

        assert {k.value for k in InputKind} != {k.value for k in DatasetKind}


# ==========================================================================
# URL and Google Drive
# ==========================================================================


class TestUrlInput:
    """URLs go through the same hardened fetcher as candidate media.

    Test hosts must be REAL resolvable domains: the SSRF guard resolves the
    hostname before any HTTP call, so a fake TLD is rejected before respx can
    intercept it. That ordering is correct in production -- an unresolvable
    host should never be dialled -- so the tests adapt, not the guard.

    `OPEN` disables only the private-target check, so the mocked host is
    reachable while every other control (size cap, redirect limit, magic-byte
    validation) still applies exactly as in production.
    """

    @respx.mock
    def test_loads_a_real_image_url(self, tmp_path):
        url = "https://example.com/photo.jpg"
        respx.get(url).mock(return_value=httpx.Response(200, content=jpeg()))
        trace = from_url(url, tmp_path, fetch_policy=OPEN)
        assert trace.source_type is SourceType.URL
        assert trace.image_url == url
        assert not trace.needs_hosting

    @respx.mock
    def test_html_page_gives_a_useful_message(self, tmp_path):
        url = "https://example.com/page"
        respx.get(url).mock(
            return_value=httpx.Response(
                200, content=b"<!DOCTYPE html><html></html>",
                headers={"content-type": "image/jpeg"},
            )
        )
        with pytest.raises(InputError) as caught:
            from_url(url, tmp_path, fetch_policy=OPEN)
        assert "web page" in caught.value.hint.lower()

    @respx.mock
    def test_403_explains_the_block(self, tmp_path):
        url = "https://example.com/blocked.jpg"
        respx.get(url).mock(return_value=httpx.Response(403))
        with pytest.raises(InputError) as caught:
            from_url(url, tmp_path, fetch_policy=OPEN)
        assert "403" in caught.value.message
        assert "blocking automated access" in caught.value.hint

    @respx.mock
    def test_404_is_explained(self, tmp_path):
        url = "https://example.com/gone.jpg"
        respx.get(url).mock(return_value=httpx.Response(404))
        with pytest.raises(InputError) as caught:
            from_url(url, tmp_path, fetch_policy=OPEN)
        assert "moved or deleted" in caught.value.hint

    def test_ssrf_target_is_refused(self, tmp_path):
        with pytest.raises(InputError) as caught:
            from_url("http://169.254.169.254/latest/meta-data/", tmp_path)
        assert "private network" in caught.value.message

    def test_empty_url_is_refused(self, tmp_path):
        with pytest.raises(InputError, match="No link"):
            from_url("   ", tmp_path)


class TestGoogleDrive:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://drive.google.com/file/d/1AbCdEfGhIjKlMnOp/view?usp=sharing",
             "1AbCdEfGhIjKlMnOp"),
            ("https://drive.google.com/open?id=1AbCdEfGhIjKlMnOp", "1AbCdEfGhIjKlMnOp"),
            ("https://drive.google.com/uc?export=download&id=1AbCdEfGhIjKlMnOp",
             "1AbCdEfGhIjKlMnOp"),
        ],
    )
    def test_extracts_file_id_from_link_shapes(self, url, expected):
        assert extract_drive_file_id(url) == expected

    def test_non_drive_url_yields_no_id(self):
        assert extract_drive_file_id("https://example.com/x.jpg") is None

    def test_direct_url_shape(self):
        assert "uc?export=download&id=ABC" in drive_direct_url("ABC")

    def test_non_drive_link_is_refused_clearly(self, tmp_path):
        with pytest.raises(InputError, match="Google Drive link"):
            from_google_drive("https://example.com/x.jpg", tmp_path)

    @respx.mock
    def test_public_drive_file_imports(self, tmp_path):
        file_id = "1AbCdEfGhIjKlMnOp"
        respx.get(drive_direct_url(file_id)).mock(
            return_value=httpx.Response(200, content=jpeg())
        )
        trace = from_google_drive(
            "https://drive.google.com/file/d/{0}/view".format(file_id),
            tmp_path, fetch_policy=OPEN,
        )
        assert trace.source_type is SourceType.GOOGLE_DRIVE

    @respx.mock
    def test_private_drive_file_explains_sharing(self, tmp_path):
        file_id = "1PrivateFileId12345"
        respx.get(drive_direct_url(file_id)).mock(return_value=httpx.Response(403))
        with pytest.raises(InputError) as caught:
            from_google_drive(
                "https://drive.google.com/file/d/{0}/view".format(file_id),
                tmp_path, fetch_policy=OPEN,
            )
        assert "not shared publicly" in caught.value.message
        assert "Anyone with the link" in caught.value.hint

    @respx.mock
    def test_does_not_pretend_to_sign_in(self, tmp_path):
        # Honest about the Picker/OAuth gap rather than faking it: a file that
        # is not public fails with an explanation, never a fabricated success.
        file_id = "1NotSharedPublicly99"
        respx.get(drive_direct_url(file_id)).mock(return_value=httpx.Response(401))
        with pytest.raises(InputError) as caught:
            from_google_drive(
                "https://drive.google.com/file/d/{0}/view".format(file_id),
                tmp_path, fetch_policy=OPEN,
            )
        assert "does not sign in" in caught.value.hint.lower()


# ==========================================================================
# API surface
# ==========================================================================


class TestApiRoutes:
    def test_index_serves(self, client):
        assert client.get("/").status_code == 200

    def test_health_reports_capability(self, client):
        payload = client.get("/api/health").json()
        for key in ("face_engine_loaded", "search_configured", "calibrated", "stages"):
            assert key in payload

    def test_health_lists_the_real_stages(self, client):
        from tracelock.service.runner import STAGES

        stages = client.get("/api/health").json()["stages"]
        assert [s["key"] for s in stages] == [k for k, _ in STAGES]

    def test_examples_are_marked_demo(self, client):
        payload = client.get("/api/examples").json()
        assert "live" in payload["notice"].lower()
        for example in payload["examples"]:
            assert example["kind"] == "DEMO_EXAMPLE"

    def test_upload_accepts_an_image(self, client):
        response = client.post(
            "/api/input/upload", files={"file": ("a.jpg", jpeg(), "image/jpeg")}
        )
        assert response.status_code == 200
        assert response.json()["source_type"] == "upload"

    def test_upload_rejects_a_non_image_with_a_readable_error(self, client):
        response = client.post(
            "/api/input/upload",
            files={"file": ("a.jpg", b"<html></html>", "image/jpeg")},
        )
        assert response.status_code == 400
        assert "not an image" in response.json()["error"].lower()

    def test_webcam_route_normalizes_the_same_way(self, client):
        response = client.post(
            "/api/input/webcam", files={"file": ("cap.jpg", jpeg(), "image/jpeg")}
        )
        assert response.json()["source_type"] == "webcam"

    def test_unknown_example_is_404(self, client):
        response = client.post("/api/input/example", data={"example_id": "nope"})
        assert response.status_code == 404

    def test_investigation_status_404s_for_unknown_run(self, client):
        assert client.get("/api/investigation/deadbeef").status_code == 404


class TestApiDoesNotLeak:
    def test_no_server_paths_in_input_response(self, client):
        payload = client.post(
            "/api/input/upload", files={"file": ("a.jpg", jpeg(), "image/jpeg")}
        ).json()
        assert "local_path" not in payload
        serialized = json.dumps(payload)
        assert "data\\inputs" not in serialized and "data/inputs" not in serialized

    def test_chain_status_hides_credentials(self, client, monkeypatch):
        """Variable NAMES may be shown; VALUES never may.

        The readiness panel deliberately lists which environment variables are
        required -- that is how an operator knows what to configure. What it
        must never do is echo what they are set to.
        """
        secret_key = "0x" + "d" * 64
        secret_rpc = "https://secret-rpc.example/abcdef123456"
        monkeypatch.setenv("TL_PRIVATE_KEY", secret_key)
        monkeypatch.setenv("TL_RPC_URL", secret_rpc)

        serialized = json.dumps(client.get("/api/chain/status").json())

        # The VALUES must not appear anywhere.
        assert secret_key not in serialized
        assert "d" * 64 not in serialized
        assert secret_rpc not in serialized
        assert "abcdef123456" not in serialized

    def test_chain_status_reports_which_variables_are_set(self, client, monkeypatch):
        monkeypatch.setenv("TL_PRIVATE_KEY", "0x" + "e" * 64)
        readiness = client.get("/api/chain/status").json()["readiness"]
        by_name = {e["name"]: e for e in readiness["required_env"]}

        assert by_name["TL_PRIVATE_KEY"]["set"] is True
        # `set` is reported, the value is not.
        assert by_name["TL_PRIVATE_KEY"]["current"] is None

    def test_preview_is_served_by_hash_not_path(self, client):
        payload = client.post(
            "/api/input/upload", files={"file": ("a.jpg", jpeg(), "image/jpeg")}
        ).json()
        assert payload["preview_url"].startswith("/api/preview/")
        assert client.get(payload["preview_url"]).status_code == 200

    def test_unknown_preview_is_404(self, client):
        assert client.get("/api/preview/" + "0" * 64).status_code == 404


class TestStageContract:
    def test_stages_are_ordered_and_unique(self):
        from tracelock.service.runner import STAGES

        keys = [k for k, _ in STAGES]
        assert len(keys) == len(set(keys))
        assert keys[0] == "received"
        assert keys[-1] == "anchor"

    def test_every_stage_has_a_human_label(self):
        from tracelock.service.runner import STAGES

        for _key, label in STAGES:
            assert label and label[0].isupper()
