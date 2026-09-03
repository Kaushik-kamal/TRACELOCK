"""The bridge from a local image to live reverse-image search.

THE GAP THIS CLOSES
-------------------
`needs_hosting` is `not self.image_url`, and the runner fails at `search_image`
when it is True. So an uploaded photo or a webcam frame could never enter public
discovery: every reverse-image API fetches a URL and none accepts a file.

THE RULE THAT MAKES IT SAFE
---------------------------
Publishing a face is a real biometric disclosure, so it happens only on an
explicit `consent=true`. Every test below that touches the network is mocked;
the one real upload performed during development was separately authorised and
is not repeated here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tracelock.discovery.hosts import (
    PROVIDERS,
    HostingError,
    HostingResult,
    Retention,
    available_providers,
    resolve_provider,
)
from tracelock.discovery.hosts.base import validate_payload
from tracelock.ingest.social import SourceCategory, classify_source, summarise


@pytest.fixture
def client():
    from tracelock.api import create_app

    return TestClient(create_app())


def jpeg(size: int = 600) -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (size, size), (110, 120, 130)).save(buffer, format="JPEG")
    return buffer.getvalue()


class FakeHost:
    """Stands in for a real host. Records what it was asked to do."""

    key = "fake"
    display_name = "Fake Host"

    def __init__(self, *, configured=True, deletes=False, fail=None):
        self._configured = configured
        self._deletes = deletes
        self._fail = fail
        self.uploads: list[bytes] = []
        self.deletions: list[str] = []

    @property
    def configured(self):
        return self._configured

    @property
    def supports_deletion(self):
        return self._deletes

    @property
    def retention_note(self):
        return "Fake host used in tests."

    def missing_configuration(self):
        return () if self._configured else ("TL_FAKE_KEY",)

    def upload(self, data, filename, *, retention):
        if self._fail:
            raise self._fail
        self.uploads.append(data)
        return HostingResult(
            url="https://fake.test/{0}.jpg".format(len(self.uploads)),
            provider=self.key,
            asset_id="asset-1",
            deletion_supported=self._deletes,
        )

    def delete(self, result):
        if not self._deletes:
            return False
        self.deletions.append(result.asset_id)
        return True


def _upload(client) -> str:
    return client.post(
        "/api/input/upload", files={"file": ("j.jpg", jpeg(), "image/jpeg")}
    ).json()["sha256"]


# ==========================================================================
# 1-2. Nothing is uploaded without consent
# ==========================================================================


class TestNothingLeavesWithoutConsent:
    def test_upload_alone_publishes_nothing(self, client, monkeypatch):
        host = FakeHost()
        monkeypatch.setattr(
            "tracelock.discovery.hosts.resolve_provider", lambda *a, **k: host
        )
        payload = client.post(
            "/api/input/upload", files={"file": ("j.jpg", jpeg(), "image/jpeg")}
        ).json()

        assert payload["image_url"] is None
        assert payload["needs_hosting"] is True
        assert host.uploads == []

    def test_webcam_alone_publishes_nothing(self, client, monkeypatch):
        host = FakeHost()
        monkeypatch.setattr(
            "tracelock.discovery.hosts.resolve_provider", lambda *a, **k: host
        )
        payload = client.post(
            "/api/input/webcam", files={"file": ("w.jpg", jpeg(), "image/jpeg")}
        ).json()

        assert payload["image_url"] is None
        assert host.uploads == []

    def test_publish_without_consent_is_refused(self, client, monkeypatch):
        host = FakeHost()
        monkeypatch.setattr(
            "tracelock.discovery.hosts.resolve_provider", lambda *a, **k: host
        )
        sha = _upload(client)

        response = client.post("/api/input/publish", data={"sha256": sha})
        assert response.status_code == 400
        assert response.json()["published"] is False
        assert host.uploads == [], "an image was uploaded without consent"

    def test_consent_false_is_also_refused(self, client, monkeypatch):
        host = FakeHost()
        monkeypatch.setattr(
            "tracelock.discovery.hosts.resolve_provider", lambda *a, **k: host
        )
        sha = _upload(client)
        response = client.post(
            "/api/input/publish", data={"sha256": sha, "consent": "false"}
        )
        assert response.status_code == 400
        assert host.uploads == []


# ==========================================================================
# 3, 5. Explicit consent reaches the host and enters the pipeline
# ==========================================================================


class TestConsentedPublish:
    def _patch(self, monkeypatch, host):
        import tracelock.discovery.hosts as hosts

        monkeypatch.setattr(hosts, "resolve_provider", lambda *a, **k: host)

    def test_consent_calls_the_host(self, client, monkeypatch):
        host = FakeHost()
        self._patch(monkeypatch, host)
        sha = _upload(client)

        payload = client.post(
            "/api/input/publish", data={"sha256": sha, "consent": "true"}
        ).json()

        assert payload["published"] is True
        assert len(host.uploads) == 1
        assert payload["image_url"] == "https://fake.test/1.jpg"

    def test_the_published_url_enters_the_existing_pipeline(self, client, monkeypatch):
        """`needs_hosting` must flip, which is what unblocks discovery."""
        host = FakeHost()
        self._patch(monkeypatch, host)
        sha = _upload(client)

        published = client.post(
            "/api/input/publish", data={"sha256": sha, "consent": "true"}
        ).json()["input"]

        assert published["image_url"] == "https://fake.test/1.jpg"
        assert published["needs_hosting"] is False

    def test_hosting_provenance_is_recorded(self, client, monkeypatch):
        host = FakeHost()
        self._patch(monkeypatch, host)
        sha = _upload(client)

        payload = client.post(
            "/api/input/publish", data={"sha256": sha, "consent": "true"}
        ).json()
        provenance = payload["input"]["provenance"]

        assert provenance["public_url_supplied_by"] == "temporary_host"
        assert provenance["temporary_hosting"]["provider"] == "fake"

    def test_a_delete_token_never_reaches_the_browser(self, client, monkeypatch):
        """It is a credential for destroying the asset."""
        result = HostingResult(
            url="https://fake.test/x.jpg", provider="fake",
            delete_token="SECRET-DELETE-TOKEN",
        )
        assert "delete_token" not in result.to_dict()
        assert "SECRET-DELETE-TOKEN" not in json.dumps(result.to_dict())

    def test_an_already_public_image_is_not_republished(self, client, monkeypatch):
        """A second copy for nothing would be a pointless disclosure."""
        host = FakeHost()
        self._patch(monkeypatch, host)

        loaded = client.post(
            "/api/input/url",
            data={"url": "https://raw.githubusercontent.com/x/y/main/a.jpg"},
        )
        if loaded.status_code != 200:
            pytest.skip("network unavailable for the direct-URL path")

        sha = loaded.json()["sha256"]
        payload = client.post(
            "/api/input/publish", data={"sha256": sha, "consent": "true"}
        ).json()

        assert payload.get("already_public") is True
        assert host.uploads == []


# ==========================================================================
# 4. An unconfigured or failing host fails honestly
# ==========================================================================


class TestHostFailuresAreHonest:
    def test_unconfigured_host_gives_a_recoverable_error(self, client, monkeypatch):
        import tracelock.discovery.hosts as hosts

        host = FakeHost(configured=False)
        monkeypatch.setattr(hosts, "resolve_provider", lambda *a, **k: host)
        sha = _upload(client)

        response = client.post(
            "/api/input/publish", data={"sha256": sha, "consent": "true"}
        )
        payload = response.json()

        assert response.status_code == 400
        assert payload["published"] is False
        assert payload["failure"]["status"] == "not_configured"
        assert payload["recovery"]
        assert "image_url" not in payload

    def test_upload_failure_is_reported_not_faked(self, client, monkeypatch):
        import tracelock.discovery.hosts as hosts

        host = FakeHost(fail=HostingError("host returned HTTP 503"))
        monkeypatch.setattr(hosts, "resolve_provider", lambda *a, **k: host)
        sha = _upload(client)

        response = client.post(
            "/api/input/publish", data={"sha256": sha, "consent": "true"}
        )
        payload = response.json()

        assert response.status_code == 400
        assert payload["published"] is False
        assert "image_url" not in payload
        assert payload["recovery"]

    def test_payload_guards_run_before_any_network_call(self):
        for bad in (b"", b"x" * 10, b"z" * 500):
            with pytest.raises(HostingError):
                validate_payload(bad)
        validate_payload(jpeg())


# ==========================================================================
# 6. Provider capabilities are declared, never assumed
# ==========================================================================


class TestProviderCapabilities:
    def test_catbox_is_the_zero_config_default(self):
        provider = resolve_provider()
        assert provider.configured
        assert provider.key == "catbox"

    def test_no_provider_claims_deletion_it_cannot_perform(self):
        for provider in PROVIDERS.values():
            if not provider.supports_deletion:
                assert provider.delete(
                    HostingResult(url="https://x.test/a.jpg", provider=provider.key)
                ) is False

    def test_only_cloudinary_claims_real_deletion(self):
        """Catbox has no anonymous delete API; ImgBB's is an HTML page."""
        assert PROVIDERS["cloudinary"].supports_deletion is True
        assert PROVIDERS["catbox"].supports_deletion is False
        assert PROVIDERS["imgbb"].supports_deletion is False

    def test_key_based_providers_report_what_they_need(self):
        assert "TL_IMGBB_API_KEY" in PROVIDERS["imgbb"].missing_configuration()
        assert PROVIDERS["catbox"].missing_configuration() == ()

    def test_hosting_info_endpoint_states_capabilities(self, client):
        payload = client.get("/api/hosting/info").json()
        assert payload["consent_required"] is True
        assert payload["guarantees"]
        assert any(p["key"] == "cloudinary" for p in payload["providers"])

    def test_no_credential_VALUES_appear_in_the_capability_matrix(
        self, client, monkeypatch
    ):
        """Variable NAMES are fine -- they tell the operator what to set.

        Banning the name would flag the endpoint's legitimate
        "missing: TL_CLOUDINARY_API_SECRET" row. Only a value is a leak.
        """
        monkeypatch.setenv("TL_CLOUDINARY_CLOUD_NAME", "demo-cloud")
        monkeypatch.setenv("TL_CLOUDINARY_API_KEY", "123456789012345")
        monkeypatch.setenv("TL_CLOUDINARY_API_SECRET", "sUpErSeCrEtValue123")
        monkeypatch.setenv("TL_IMGBB_API_KEY", "imgbb-key-value-xyz")

        blob = json.dumps(client.get("/api/hosting/info").json())

        assert "sUpErSeCrEtValue123" not in blob
        assert "imgbb-key-value-xyz" not in blob
        assert "123456789012345" not in blob


# ==========================================================================
# 8-9. Social classification cannot be faked
# ==========================================================================


class TestSocialClassification:
    @pytest.mark.parametrize(
        "url,platform",
        [
            ("https://www.instagram.com/p/ABC/", "Instagram"),
            ("https://www.facebook.com/x/posts/1", "Facebook"),
            ("https://x.com/a/status/1", "X"),
            ("https://www.tiktok.com/@a/video/1", "TikTok"),
            ("https://www.linkedin.com/in/a/", "LinkedIn"),
            ("https://www.reddit.com/r/a/comments/b/", "Reddit"),
            ("https://www.pinterest.com/pin/1/", "Pinterest"),
            ("https://www.threads.net/@a/post/1", "Threads"),
        ],
    )
    def test_real_social_platforms_are_recognised(self, url, platform):
        result = classify_source(url)
        assert result.is_social
        assert result.platform == platform

    @pytest.mark.parametrize(
        "url,category",
        [
            ("https://www.ndtv.com/india-news/a", SourceCategory.MEDIA),
            ("https://en.wikipedia.org/wiki/A", SourceCategory.ENCYCLOPEDIA),
            ("https://i.imgur.com/a.jpg", SourceCategory.IMAGE_HOST),
            ("https://files.catbox.moe/a.jpg", SourceCategory.IMAGE_HOST),
            ("https://some-company.example/team", SourceCategory.OTHER),
        ],
    )
    def test_non_social_sources_cannot_be_labelled_social(self, url, category):
        result = classify_source(url)
        assert result.category is category
        assert not result.is_social

    def test_a_news_site_is_never_a_social_media_match(self):
        """Real corroboration, but a different claim entirely."""
        assert not classify_source("https://www.hindustantimes.com/x").is_social

    def test_our_own_temporary_host_is_not_a_match(self):
        """The copy WE published must never count as a discovery."""
        assert not classify_source("https://files.catbox.moe/5ioky1.jpg").is_social


# ==========================================================================
# 10, 12. Discovered is not the same as verified
# ==========================================================================


class TestDiscoveredIsNotVerified:
    def _result(self, url, status, similarity=None):
        return {
            "status": status,
            "candidate_id": "c1",
            "face_similarity": similarity,
            "provenance": {"post_url": url, "host": "x"},
        }

    def test_a_discovered_social_url_alone_is_not_a_match(self):
        summary = summarise([
            self._result("https://www.instagram.com/p/A/", "REJECTED"),
        ])
        assert summary["social_discovered"] == 1
        assert summary["social_face_verified"] == 0
        assert summary["requirement_met"] is False
        assert "none could be independently face-verified" in summary["statement"]

    def test_a_verified_social_post_satisfies_the_requirement(self):
        summary = summarise([
            self._result("https://www.facebook.com/a/posts/1",
                         "VERIFIED_CANDIDATE", similarity=0.71),
        ])
        assert summary["social_face_verified"] == 1
        assert summary["requirement_met"] is True
        assert summary["verified_social_posts"][0]["platform"] == "Facebook"

    def test_a_verified_NEWS_result_does_not_satisfy_it(self):
        """Verified, real, and still not a social media post."""
        summary = summarise([
            self._result("https://www.ndtv.com/a", "VERIFIED_CANDIDATE", 0.8),
        ])
        assert summary["social_discovered"] == 0
        assert summary["requirement_met"] is False

    def test_no_results_reports_search_completed_not_failure(self):
        """'Search ran' and 'match found' are different outcomes."""
        summary = summarise([])
        assert summary["requirement_met"] is False
        assert "Live search completed" in summary["statement"]

    def test_the_summary_never_invents_a_similarity(self):
        summary = summarise([
            self._result("https://www.instagram.com/p/A/", "VERIFIED_CANDIDATE"),
        ])
        assert summary["verified_social_posts"][0]["similarity"] is None


# ==========================================================================
# 7. Nothing is pre-picked
# ==========================================================================


class TestNothingIsHardcoded:
    def test_no_candidate_urls_are_baked_into_the_source(self):
        """A pre-picked result would make the whole search theatre."""
        import re

        for path in Path("src/tracelock").rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for line in source.splitlines():
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith('"'):
                    continue
                # A hardcoded social post URL in code would be a planted result.
                assert not re.search(
                    r'["\']https?://(www\.)?(instagram|facebook|twitter|x)\.com/'
                    r'(p|posts|status)/', stripped
                ), "{0}: {1}".format(path, stripped[:80])

    def test_the_search_forces_a_live_query(self):
        source = Path(
            "src/tracelock/discovery/serpapi_lens.py"
        ).read_text(encoding="utf-8")
        assert '"no_cache": "true"' in source

    def test_the_provider_is_given_the_runtime_url(self):
        """The search URL comes from the input, not from a constant."""
        source = Path("src/tracelock/service/runner.py").read_text(encoding="utf-8")
        assert "public_url=trace.image_url" in source

    def test_demo_examples_pre_pick_only_the_input_never_the_results(self):
        """Code only -- the module docstring legitimately says candidates come
        from live search, which is the opposite of shipping them."""
        import io
        import tokenize

        path = Path("src/tracelock/service/examples.py")
        assert "public_url" in path.read_text(encoding="utf-8")

        code = []
        with io.open(path, encoding="utf-8") as handle:
            for token in tokenize.generate_tokens(handle.readline):
                if token.type in (tokenize.COMMENT, tokenize.STRING):
                    continue
                code.append(token.string)
        joined = " ".join(code)

        # No candidate or result list may be shipped alongside an example.
        for banned in ("visual_matches", "image_results", "candidates"):
            assert banned not in joined, banned
