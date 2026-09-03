"""Universal ingestion: classification, ranking, resolution, and the honesty
guarantees that surround them.

The load-bearing tests here are the ones about what the system REFUSES to do:
no fabricated image URL, no claim of platform support that was not observed,
no SSRF bypass via a page's own metadata, and no authentication anywhere.
"""

from __future__ import annotations

import sys

import pytest

from tracelock.ingest import (
    ADAPTERS,
    InputType,
    Method,
    Platform,
    Support,
    adapter_for,
    capability_matrix,
    classify_url,
    rank_candidates,
    resolve,
)
from tracelock.ingest.adapters import _rewrite_drive
from tracelock.ingest.candidates import MIN_USEFUL_DIMENSION


# ==========================================================================
# Classification
# ==========================================================================


class TestClassification:
    @pytest.mark.parametrize(
        "url,expected_type,expected_platform",
        [
            ("https://site.com/a/photo.jpg", InputType.DIRECT_IMAGE_URL, Platform.GENERIC),
            ("https://site.com/a/photo.PNG", InputType.DIRECT_IMAGE_URL, Platform.GENERIC),
            ("https://pbs.twimg.com/media/x.jpg", InputType.DIRECT_IMAGE_URL, Platform.X_TWITTER),
            ("https://www.instagram.com/p/ABC/", InputType.SOCIAL_POST, Platform.INSTAGRAM),
            ("https://instagram.com/reel/XYZ/", InputType.SOCIAL_POST, Platform.INSTAGRAM),
            ("https://www.instagram.com/nasa", InputType.SOCIAL_PROFILE, Platform.INSTAGRAM),
            ("https://x.com/nasa/status/123", InputType.SOCIAL_POST, Platform.X_TWITTER),
            ("https://twitter.com/nasa/status/123", InputType.SOCIAL_POST, Platform.X_TWITTER),
            ("https://x.com/nasa", InputType.SOCIAL_PROFILE, Platform.X_TWITTER),
            ("https://www.linkedin.com/in/someone/", InputType.SOCIAL_PROFILE, Platform.LINKEDIN),
            ("https://www.linkedin.com/posts/abc", InputType.SOCIAL_POST, Platform.LINKEDIN),
            ("https://www.reddit.com/r/pics/comments/a/b/", InputType.SOCIAL_POST, Platform.REDDIT),
            ("https://www.reddit.com/u/someone", InputType.SOCIAL_PROFILE, Platform.REDDIT),
            ("https://youtu.be/dQw4w9WgXcQ", InputType.SOCIAL_POST, Platform.YOUTUBE),
            ("https://www.youtube.com/watch?v=abc", InputType.SOCIAL_POST, Platform.YOUTUBE),
            ("https://www.youtube.com/@channel", InputType.SOCIAL_PROFILE, Platform.YOUTUBE),
            ("https://drive.google.com/file/d/1AbCdEfGhIj/view", InputType.CLOUD_SHARE_LINK, Platform.GOOGLE_DRIVE),
            ("https://en.wikipedia.org/wiki/Cat", InputType.WEBPAGE_URL, Platform.GENERIC),
            ("https://pin.it/abcdef", InputType.SOCIAL_POST, Platform.PINTEREST),
            ("https://www.pinterest.com/pin/12345/", InputType.SOCIAL_POST, Platform.PINTEREST),
        ],
    )
    def test_classifies(self, url, expected_type, expected_platform):
        result = classify_url(url)
        assert result.input_type is expected_type
        assert result.platform is expected_platform

    @pytest.mark.parametrize(
        "url",
        [
            "", "   ", "not a url", "ftp://site.com/x.jpg",
            "file:///etc/passwd", "javascript:alert(1)", "//site.com/x.jpg",
            "data:image/png;base64,AAAA",
        ],
    )
    def test_non_http_is_unknown_not_guessed(self, url):
        """A scheme we do not fetch must never classify as something fetchable."""
        assert classify_url(url).input_type is InputType.UNKNOWN_URL

    def test_image_extension_beats_platform(self):
        """A .jpg on a social CDN is a direct image, not a post to resolve."""
        result = classify_url("https://scontent.cdninstagram.com/v/x.jpg")
        assert result.input_type is InputType.DIRECT_IMAGE_URL
        assert result.platform is Platform.INSTAGRAM

    def test_direct_image_qualifier_is_stable_across_hosts(self):
        """Provenance must not vary by an irrelevant detail."""
        generic = classify_url("https://site.com/x.jpg").qualifier
        platform = classify_url("https://pbs.twimg.com/media/x.jpg").qualifier
        assert generic == platform == "direct_image"

    def test_qualifier_is_platform_specific_for_posts(self):
        assert classify_url("https://instagram.com/p/A/").qualifier == "instagram_post"
        assert classify_url("https://x.com/a/status/1").qualifier == "x_post"
        assert classify_url("https://instagram.com/nasa").qualifier == "instagram_profile"

    def test_local_sources_are_marked_local(self):
        for kind in (InputType.LOCAL_UPLOAD, InputType.WEBCAM_CAPTURE, InputType.DEMO_EXAMPLE):
            assert kind.is_local
            assert not kind.needs_resolution

    def test_classification_makes_no_network_request(self, monkeypatch):
        """Classification must be instant and offline -- the UI calls it per keystroke."""
        import httpx

        def forbidden(*a, **k):
            raise AssertionError("classification must not touch the network")

        for target in ("get", "post", "request", "stream"):
            monkeypatch.setattr(httpx, target, forbidden, raising=False)
        monkeypatch.setattr(httpx.Client, "send", forbidden, raising=False)

        for url in ("https://instagram.com/p/A/", "https://site.com/x.jpg", "nonsense"):
            classify_url(url)


# ==========================================================================
# Capability matrix -- honesty about what does not work
# ==========================================================================


class TestCapabilityMatrix:
    def test_every_platform_has_an_adapter(self):
        for platform in Platform:
            assert adapter_for(platform).platform is platform

    def test_auth_gated_platforms_are_declared_as_such(self):
        """We must not imply Instagram works. It usually does not."""
        for platform in (
            Platform.INSTAGRAM, Platform.FACEBOOK,
            Platform.LINKEDIN, Platform.X_TWITTER,
        ):
            assert ADAPTERS[platform].support is Support.AUTH_REQUIRED
            assert ADAPTERS[platform].guidance, "must tell the operator what to do instead"

    def test_auth_required_expectation_says_we_do_not_authenticate(self):
        text = Support.AUTH_REQUIRED.expectation.lower()
        assert "does not authenticate" in text

    def test_matrix_is_ordered_most_capable_first(self):
        rows = capability_matrix()
        ranking = {"reliable": 0, "best_effort": 1, "auth_required": 2}
        scores = [ranking[r["support"]] for r in rows]
        assert scores == sorted(scores)

    def test_no_adapter_claims_to_authenticate(self):
        """No adapter may describe itself as signing in or using a token."""
        banned = (
            "we log in", "we sign in", "logging in", "signing in",
            "with an access token", "using an access token", "our access token",
            "with a token", "api key", "password", "send a cookie",
            "bypass", "solve the captcha", "evade",
        )
        for adapter in ADAPTERS.values():
            blob = "{0} {1}".format(adapter.note, adapter.guidance).lower()
            for word in banned:
                # "requires sign-in" describes the SITE, not us -- so only the
                # active forms above are banned.
                assert word not in blob, "{0}: {1}".format(adapter.platform, word)


class TestAdapterRewrites:
    def test_drive_share_link_becomes_public_thumbnail(self):
        out = _rewrite_drive("https://drive.google.com/file/d/1AbCdEfGhIjKl/view?usp=sharing")
        assert out == "https://drive.google.com/thumbnail?id=1AbCdEfGhIjKl&sz=w2000"

    def test_drive_open_id_form(self):
        out = _rewrite_drive("https://drive.google.com/open?id=1AbCdEfGhIjKl")
        assert "id=1AbCdEfGhIjKl" in out

    def test_unrecognised_drive_url_is_left_alone(self):
        url = "https://drive.google.com/drive/my-drive"
        assert _rewrite_drive(url) == url

    def test_twitter_is_canonicalised_to_x(self):
        out = adapter_for(Platform.X_TWITTER).rewrite("https://twitter.com/a/status/1")
        assert out == "https://x.com/a/status/1"

    def test_rewrites_stay_on_the_same_public_host(self):
        """A rewrite may not redirect to some other service."""
        for url, host in (
            ("https://drive.google.com/file/d/1AbCdEfGhIjKl/view", "drive.google.com"),
            ("https://twitter.com/a/status/1", "x.com"),
            ("https://www.reddit.com/r/a/comments/b/c/?x=1", "www.reddit.com"),
        ):
            from urllib.parse import urlsplit

            platform = classify_url(url).platform
            assert urlsplit(adapter_for(platform).rewrite(url)).hostname == host


# ==========================================================================
# Multi-image candidate ranking
# ==========================================================================


PAGE = """
<html><head>
  <title>Example Article</title>
  <meta property="og:image" content="/img/hero-photo.jpg">
  <meta property="og:image:width" content="1200">
  <meta property="og:image:height" content="800">
  <meta name="twitter:image" content="https://cdn.test/twitter-card.jpg">
</head><body>
  <img src="/img/site-logo.png" alt="logo" width="120" height="40">
  <img src="/img/tracking-pixel.gif" width="1" height="1">
  <img src="/img/portrait-of-subject.jpg" alt="portrait" width="800" height="1000">
  <img src="data:image/png;base64,AAAA" alt="inline">
  <img srcset="/img/small.jpg 320w, /img/large.jpg 1600w" alt="responsive">
</body></html>
"""


class TestCandidateRanking:
    def test_finds_multiple_candidates(self):
        ranked, title = rank_candidates(PAGE, "https://site.test/article")
        assert title == "Example Article"
        assert len(ranked) >= 4

    def test_og_image_outranks_page_furniture(self):
        ranked, _ = rank_candidates(PAGE, "https://site.test/article")
        assert ranked[0].url.endswith("/img/hero-photo.jpg")
        assert ranked[0].source == "og:image"

    def test_relative_urls_are_absolutised(self):
        ranked, _ = rank_candidates(PAGE, "https://site.test/article")
        assert all(c.url.startswith("http") for c in ranked)

    def test_data_uris_are_excluded(self):
        ranked, _ = rank_candidates(PAGE, "https://site.test/article")
        assert not any(c.url.startswith("data:") for c in ranked)

    def test_tracking_pixel_is_excluded_by_declared_size(self):
        ranked, _ = rank_candidates(PAGE, "https://site.test/article")
        assert not any("tracking-pixel" in c.url for c in ranked)

    def test_logo_ranks_below_a_portrait(self):
        ranked, _ = rank_candidates(PAGE, "https://site.test/article")
        order = [c.url for c in ranked]
        portrait = next(i for i, u in enumerate(order) if "portrait" in u)
        logo = next((i for i, u in enumerate(order) if "site-logo" in u), len(order))
        assert portrait < logo

    def test_srcset_picks_the_widest(self):
        ranked, _ = rank_candidates(PAGE, "https://site.test/article")
        urls = [c.url for c in ranked]
        assert any(u.endswith("/img/large.jpg") for u in urls)
        assert not any(u.endswith("/img/small.jpg") for u in urls)

    def test_ranking_is_deterministic(self):
        """Identical input must produce an identical order, every time."""
        first, _ = rank_candidates(PAGE, "https://site.test/a")
        for _ in range(5):
            again, _ = rank_candidates(PAGE, "https://site.test/a")
            assert [c.url for c in again] == [c.url for c in first]
            assert [c.score for c in again] == [c.score for c in first]

    def test_malformed_html_does_not_raise(self):
        for broken in ("<html><img src=", "<<>>", "", "<img src='/a.jpg'"):
            rank_candidates(broken, "https://site.test/")

    def test_tiny_declared_images_are_dropped(self):
        html = '<img src="/x.jpg" width="{0}" height="{0}">'.format(
            MIN_USEFUL_DIMENSION - 1
        )
        ranked, _ = rank_candidates(html, "https://site.test/")
        assert ranked == []

    def test_non_http_srcs_are_dropped(self):
        html = '<img src="javascript:alert(1)"><img src="file:///etc/passwd">'
        ranked, _ = rank_candidates(html, "https://site.test/")
        assert ranked == []


# ==========================================================================
# Resolution: security and truthfulness
# ==========================================================================


class _Response:
    def __init__(self, *, ok=True, content=b"", status=200, final_url="", reason=None):
        self.ok = ok
        self.content = content
        self.status_code = status
        self.final_url = final_url
        self.reason = reason
        self.detail = ""


class TestResolutionTruthfulness:
    def test_direct_image_needs_no_fetch(self, monkeypatch):
        import tracelock.acquisition.fetcher as fetcher

        monkeypatch.setattr(
            fetcher, "fetch_media",
            lambda *a, **k: pytest.fail("a direct image must not be fetched to resolve"),
        )
        out = resolve("https://site.test/a.jpg")
        assert out.ok
        assert out.method is Method.DIRECT
        assert out.image_url == "https://site.test/a.jpg"

    def test_failure_has_no_image_url_key_at_all(self, monkeypatch):
        """Structurally absent, not None -- nothing can render it by accident."""
        import tracelock.acquisition.fetcher as fetcher

        monkeypatch.setattr(
            fetcher, "fetch_media",
            lambda *a, **k: _Response(ok=False, status=403),
        )
        out = resolve("https://www.instagram.com/p/ABC/")
        assert not out.ok
        assert out.image_url is None
        assert "image_url" not in out.to_dict()
        assert out.to_dict()["reason"]

    def test_auth_gated_failure_names_the_platform_and_the_alternative(self, monkeypatch):
        import tracelock.acquisition.fetcher as fetcher

        monkeypatch.setattr(
            fetcher, "fetch_media",
            lambda *a, **k: _Response(ok=False, status=403),
        )
        out = resolve("https://www.instagram.com/p/ABC/")
        assert "Instagram" in out.reason
        assert "upload" in out.to_dict()["guidance"].lower()

    def test_a_page_with_no_image_is_reported_not_invented(self, monkeypatch):
        import tracelock.acquisition.fetcher as fetcher

        monkeypatch.setattr(
            fetcher, "fetch_media",
            lambda *a, **k: _Response(content=b"<html><body>text only</body></html>"),
        )
        out = resolve("https://site.test/article")
        assert not out.ok
        assert "no usable image" in out.reason

    def test_ranked_alternatives_are_offered(self, monkeypatch):
        import tracelock.acquisition.fetcher as fetcher

        monkeypatch.setattr(
            fetcher, "fetch_media",
            lambda *a, **k: _Response(
                content=PAGE.encode(), final_url="https://site.test/article"
            ),
        )
        out = resolve("https://site.test/article")
        assert out.ok
        assert len(out.candidates) >= 4
        assert out.image_url not in {c.url for c in out.alternatives}
        assert out.to_dict()["candidate_count"] == len(out.candidates)


class TestResolutionSecurity:
    """SSRF protection must survive every path through the resolver."""

    def test_private_targets_are_refused(self):
        for url in (
            "http://127.0.0.1/x.jpg",
            "http://localhost/x.jpg",
            "http://169.254.169.254/latest/meta-data/",
            "http://10.0.0.5/a.jpg",
            "http://192.168.1.1/a.jpg",
            "http://172.16.0.1/a.jpg",
            "http://[::1]/a.jpg",
            "http://0.0.0.0/a.jpg",
        ):
            out = resolve(url)
            assert not out.ok, url
            assert "image_url" not in out.to_dict()

    def test_non_http_schemes_are_refused(self):
        for url in ("file:///etc/passwd", "ftp://site/x.jpg", "gopher://a/1"):
            out = resolve(url)
            assert not out.ok, url

    def test_resolved_image_url_is_re_validated_not_trusted(self, monkeypatch):
        """A page pointing og:image at the metadata endpoint must not win.

        The resolver returns a URL; the acquisition layer fetches it through
        the same guard. This proves the returned URL is never treated as
        pre-approved just because a page published it.
        """
        import tracelock.acquisition.fetcher as fetcher
        from tracelock.acquisition.fetcher import FetchPolicy, fetch_media

        hostile = (
            '<html><head><meta property="og:image" '
            'content="http://169.254.169.254/latest/meta-data/iam/"></head></html>'
        )
        monkeypatch.setattr(
            fetcher, "fetch_media",
            lambda *a, **k: _Response(content=hostile.encode(), final_url="https://site.test/"),
        )
        out = resolve("https://site.test/")

        # The resolver may surface it as a candidate...
        assert out.ok
        assert "169.254.169.254" in out.image_url

        # ...but actually fetching it is refused by the unpatched guard.
        monkeypatch.undo()
        blocked = fetch_media(out.image_url, policy=FetchPolicy())
        assert not blocked.ok
        assert blocked.reason.value == "BLOCKED_URL_TARGET"

    def test_no_authenticated_oembed_endpoint_is_configured(self):
        from tracelock.ingest.resolve import _PUBLIC_OEMBED

        assert Platform.INSTAGRAM not in _PUBLIC_OEMBED
        assert Platform.FACEBOOK not in _PUBLIC_OEMBED

    def test_resolver_source_contains_no_credential_handling(self):
        """Grep the module itself: no cookie, token or auth header anywhere."""
        import pathlib

        module = sys.modules["tracelock.ingest.resolve"]
        source = pathlib.Path(module.__file__).read_text(encoding="utf-8").lower()
        for banned in ("cookie", "authorization", "set-cookie", "bearer", "x-csrf"):
            # Prose in the docstring explains we do NOT do these; the check is
            # that no line ASSIGNS or SENDS one.
            for line in source.splitlines():
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith('"'):
                    continue
                assert banned not in stripped, line


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from tracelock.api import create_app

    return TestClient(create_app())


class TestApiSurface:
    def test_classify_endpoint_is_offline_and_honest(self, client):
        payload = client.post(
            "/api/input/classify", data={"url": "https://www.instagram.com/p/ABC/"}
        ).json()
        assert payload["classification"]["platform"] == "instagram"
        assert payload["classification"]["input_type"] == "social_post"
        assert payload["adapter"]["support"] == "auth_required"
        assert payload["needs_resolution"] is True

    def test_resolver_info_publishes_the_full_matrix(self, client):
        payload = client.get("/api/resolver/info").json()
        platforms = {p["platform"]: p for p in payload["platforms"]}
        assert platforms["instagram"]["support"] == "auth_required"
        assert platforms["generic"]["support"] == "reliable"
        assert all(m["explanation"] for m in payload["methods"])


class TestUniversalUrlIngestion:
    """`/api/input/url` accepts ANY public link, through one pipeline."""

    def _patch_resolver(self, monkeypatch, **kwargs):
        import sys

        from tracelock.ingest import adapter_for, classify_url
        from tracelock.ingest.resolve import Method, ResolvedInput

        app_module = sys.modules["tracelock.api.app"]

        def fake(url, **_):
            classification = classify_url(url)
            return ResolvedInput(
                input_url=url,
                classification=classification,
                adapter=adapter_for(classification.platform),
                method=kwargs.get("method", Method.OPENGRAPH),
                image_url=kwargs.get("image_url"),
                candidates=kwargs.get("candidates", ()),
                reason=kwargs.get("reason", ""),
            )

        monkeypatch.setattr(app_module, "resolve_input", fake)

    def test_direct_image_is_not_resolved_through_a_page(self, client, monkeypatch):
        """A direct image must not cause a page fetch."""
        import sys

        app_module = sys.modules["tracelock.api.app"]
        monkeypatch.setattr(
            app_module, "resolve_input",
            lambda *a, **k: pytest.fail("a direct image must not be resolved"),
        )
        monkeypatch.setattr(
            app_module, "from_url",
            lambda url, work_dir: _fake_trace(url),
        )
        response = client.post(
            "/api/input/url", data={"url": "https://site.test/photo.jpg"}
        )
        assert response.status_code == 200
        assert "resolution" not in response.json()

    def test_webpage_is_resolved_then_loaded(self, client, monkeypatch):
        import sys

        from tracelock.ingest.candidates import ImageCandidate

        app_module = sys.modules["tracelock.api.app"]
        self._patch_resolver(
            monkeypatch,
            image_url="https://cdn.test/hero.jpg",
            candidates=(
                ImageCandidate("https://cdn.test/hero.jpg", 120, "og:image"),
                ImageCandidate("https://cdn.test/other.jpg", 40, "img"),
            ),
        )
        loaded: dict = {}
        monkeypatch.setattr(
            app_module, "from_url",
            lambda url, work_dir: (loaded.setdefault("url", url), _fake_trace(url))[1],
        )

        payload = client.post(
            "/api/input/url", data={"url": "https://news.test/article"}
        ).json()

        # The RESOLVED url was loaded, not the page.
        assert loaded["url"] == "https://cdn.test/hero.jpg"
        assert payload["resolution"]["method"] == "opengraph"
        assert len(payload["resolution"]["alternatives"]) == 1

    def test_unresolvable_link_returns_the_platform_reason(self, client, monkeypatch):
        from tracelock.ingest.resolve import Method

        self._patch_resolver(
            monkeypatch, method=Method.NONE,
            reason="Instagram post detected, but no public image was available.",
        )
        response = client.post(
            "/api/input/url", data={"url": "https://www.instagram.com/p/A/"}
        )
        assert response.status_code == 400
        assert "Instagram" in response.json()["error"]

    def test_candidate_override_must_be_one_we_offered(self, client, monkeypatch):
        """A caller cannot smuggle in a URL that never went through resolution."""
        from tracelock.ingest.candidates import ImageCandidate

        self._patch_resolver(
            monkeypatch,
            image_url="https://cdn.test/hero.jpg",
            candidates=(ImageCandidate("https://cdn.test/hero.jpg", 120, "og:image"),),
        )
        response = client.post("/api/input/url", data={
            "url": "https://news.test/article",
            "candidate_url": "https://evil.test/not-offered.jpg",
        })
        assert response.status_code == 400
        assert "not one of the candidates" in response.json()["error"]


def _fake_trace(url: str):
    from tracelock.service.inputs import InputKind, SourceType, TraceInput

    return TraceInput(
        source_type=SourceType.URL,
        kind=InputKind.ORGANIC,
        local_path="fake.jpg",
        filename="fake.jpg",
        mime_type="image/jpeg",
        sha256="a" * 64,
        byte_size=1234,
        image_url=url,
    )
