"""Public URL resolver.

Turns a pasted social post / article / image link into a publicly published
image URL, then hands it to the EXISTING Stage 2.5 guard.

THE BOUNDARY THIS SUITE DEFENDS
-------------------------------
The resolver reads only metadata a site serves to an anonymous visitor -- the
same tags every chat app consumes to draw a link preview. It sends no
credentials, solves no CAPTCHA, and when a platform declines to serve a public
preview it reports that honestly instead of working around it.

`TestHonestFailure` and `TestNoSeparateValidationPath` are the load-bearing
ones: a refusal must stay a refusal, and a resolved URL must be re-validated
rather than trusted.
"""

from __future__ import annotations

import io
import json

import pytest

from tracelock.service.url_resolver import (
    SUPPORT_STATEMENT,
    Platform,
    ResolutionMethod,
    ResolutionStatus,
    UrlType,
    classify,
    extract_image_from_html,
    resolve_public_url,
)

pytest.importorskip("fastapi")
pytest.importorskip("cv2")
httpx = pytest.importorskip("httpx")
respx = pytest.importorskip("respx")

from fastapi.testclient import TestClient  # noqa: E402

from tracelock.acquisition.fetcher import FetchPolicy  # noqa: E402

# Disables ONLY the SSRF host check so respx-mocked hosts resolve. Every other
# control -- size cap, redirect limit, timeout, magic bytes -- still applies.
OPEN = FetchPolicy(block_private_targets=False)

PAGE = "https://news.example.com/story"


def jpeg(width: int = 400, height: int = 400) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (90, 110, 140)).save(buffer, "JPEG")
    return buffer.getvalue()


def html_page(*, og=None, twitter=None, ld=None, title="Story") -> bytes:
    head = ["<title>{0}</title>".format(title)]
    if og:
        head.append('<meta property="og:image" content="{0}">'.format(og))
    if twitter:
        head.append('<meta name="twitter:image" content="{0}">'.format(twitter))
    if ld:
        head.append(
            '<script type="application/ld+json">{0}</script>'.format(json.dumps(ld))
        )
    return ("<html><head>" + "".join(head) + "</head><body>x</body></html>").encode()


@pytest.fixture
def client():
    from tracelock.api import create_app

    return TestClient(create_app())


# ==========================================================================
# 5. Classification
# ==========================================================================


class TestClassification:
    @pytest.mark.parametrize(
        "url,expected_type,expected_platform",
        [
            ("https://site.com/photo.jpg", UrlType.DIRECT_IMAGE, Platform.GENERIC),
            ("https://site.com/a.PNG", UrlType.DIRECT_IMAGE, Platform.GENERIC),
            ("https://www.instagram.com/p/ABC123/", UrlType.SOCIAL_POST, Platform.INSTAGRAM),
            ("https://instagram.com/reel/XYZ/", UrlType.SOCIAL_POST, Platform.INSTAGRAM),
            ("https://x.com/user/status/123", UrlType.SOCIAL_POST, Platform.X_TWITTER),
            ("https://twitter.com/user/status/123", UrlType.SOCIAL_POST, Platform.X_TWITTER),
            ("https://www.facebook.com/a/posts/1", UrlType.SOCIAL_POST, Platform.FACEBOOK),
            ("https://in.linkedin.com/posts/abc", UrlType.SOCIAL_POST, Platform.LINKEDIN),
            ("https://youtu.be/abc123", UrlType.SOCIAL_POST, Platform.YOUTUBE),
            ("https://news.example.org/article", UrlType.WEBPAGE, Platform.GENERIC),
        ],
    )
    def test_classifies(self, url, expected_type, expected_platform):
        url_type, platform = classify(url)
        assert url_type is expected_type
        assert platform is expected_platform

    def test_instagram_is_identified_as_instagram_post(self):
        """REQUIREMENT 5: classification identifies instagram_post."""
        url_type, platform = classify("https://www.instagram.com/p/POST_ID/")
        assert url_type is UrlType.SOCIAL_POST
        assert platform is Platform.INSTAGRAM
        assert platform.label == "Instagram post"

        # And the provenance record names the platform specifically.
        from tracelock.service.url_resolver import ResolvedUrl

        record = ResolvedUrl(
            input_url="https://www.instagram.com/p/POST_ID/",
            input_type=url_type, platform=platform,
            resolution_method=ResolutionMethod.NONE,
            resolution_status=ResolutionStatus.UNAVAILABLE,
            reason="x",
        )
        assert record.provenance_type == "instagram_post"
        assert record.to_dict()["input_type"] == "instagram_post"

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://x.com/u/status/1", "x_post"),
            ("https://www.facebook.com/p/1", "facebook_post"),
            ("https://www.linkedin.com/posts/a", "linkedin_post"),
            ("https://youtu.be/abc", "youtube_post"),
            ("https://news.example.org/story", "webpage"),
            ("https://cdn.test/a.jpg", "direct_image"),
        ],
    )
    def test_provenance_type_is_platform_qualified(self, url, expected):
        from tracelock.service.url_resolver import ResolvedUrl

        url_type, platform = classify(url)
        record = ResolvedUrl(
            input_url=url, input_type=url_type, platform=platform,
            resolution_method=ResolutionMethod.NONE,
            resolution_status=ResolutionStatus.UNAVAILABLE, reason="x",
        )
        assert record.provenance_type == expected

    def test_image_extension_wins_over_platform_host(self):
        # A .jpg on a platform CDN is a direct image, not a post.
        url_type, platform = classify("https://scontent.instagram.com/x/photo.jpg")
        assert url_type is UrlType.DIRECT_IMAGE
        assert platform is Platform.INSTAGRAM

    def test_classification_makes_no_network_call(self, monkeypatch):
        def forbidden(*a, **k):
            raise AssertionError("classification must not touch the network")

        monkeypatch.setattr(httpx, "get", forbidden, raising=False)
        classify("https://www.instagram.com/p/ABC/")


# ==========================================================================
# Metadata extraction
# ==========================================================================


class TestMetadataExtraction:
    def test_opengraph(self):
        url, method, _ = extract_image_from_html(
            html_page(og="https://cdn.test/a.jpg").decode(), PAGE
        )
        assert url == "https://cdn.test/a.jpg"
        assert method is ResolutionMethod.OPENGRAPH

    def test_twitter_card_when_no_opengraph(self):
        url, method, _ = extract_image_from_html(
            html_page(twitter="https://cdn.test/b.jpg").decode(), PAGE
        )
        assert url == "https://cdn.test/b.jpg"
        assert method is ResolutionMethod.TWITTER_CARD

    def test_opengraph_wins_over_twitter_card(self):
        _url, method, _ = extract_image_from_html(
            html_page(og="https://cdn.test/og.jpg",
                      twitter="https://cdn.test/tw.jpg").decode(), PAGE
        )
        assert method is ResolutionMethod.OPENGRAPH

    def test_schema_org_object(self):
        url, method, _ = extract_image_from_html(
            html_page(ld={"@type": "Article",
                          "image": {"url": "https://cdn.test/c.jpg"}}).decode(), PAGE
        )
        assert url == "https://cdn.test/c.jpg"
        assert method is ResolutionMethod.SCHEMA_ORG

    def test_schema_org_list_and_graph(self):
        url, _m, _ = extract_image_from_html(
            html_page(ld={"@graph": [{"@type": "WebPage",
                                      "image": ["https://cdn.test/d.jpg"]}]}).decode(),
            PAGE,
        )
        assert url == "https://cdn.test/d.jpg"

    def test_relative_urls_are_resolved_against_the_page(self):
        url, _m, _ = extract_image_from_html(
            html_page(og="/img/e.jpg").decode(), "https://news.example.com/a/story"
        )
        assert url == "https://news.example.com/img/e.jpg"

    def test_page_title_is_captured(self):
        _u, _m, title = extract_image_from_html(
            html_page(og="https://cdn.test/a.jpg", title="Headline").decode(), PAGE
        )
        assert title == "Headline"

    def test_no_metadata_returns_none(self):
        url, method, _ = extract_image_from_html("<html><head></head></html>", PAGE)
        assert url is None
        assert method is ResolutionMethod.NONE

    def test_malformed_html_does_not_raise(self):
        broken = '<html><head><meta property="og:image" content="https://cdn.test/f.jpg"'
        url, _m, _ = extract_image_from_html(broken, PAGE)
        assert url in (None, "https://cdn.test/f.jpg")


# ==========================================================================
# 1, 2, 3. Resolution
# ==========================================================================


class TestResolution:
    def test_direct_image_url_passes_through_untouched(self):
        """REQUIREMENT 1: a direct JPG URL still works."""
        result = resolve_public_url("https://cdn.test/photo.jpg", fetch_policy=OPEN)
        assert result.ok
        assert result.input_type is UrlType.DIRECT_IMAGE
        assert result.resolution_method is ResolutionMethod.DIRECT
        assert result.resolved_image_url == "https://cdn.test/photo.jpg"

    def test_direct_image_makes_no_network_call(self, monkeypatch):
        # Nothing to resolve, so nothing should be fetched at this stage.
        import tracelock.acquisition.fetcher as fetcher

        def forbidden(*a, **k):
            raise AssertionError("a direct image needs no page fetch")

        monkeypatch.setattr(fetcher, "fetch_media", forbidden)
        assert resolve_public_url("https://cdn.test/photo.jpg").ok

    @respx.mock
    def test_webpage_with_og_image_resolves(self):
        """REQUIREMENT 2."""
        respx.get(PAGE).mock(return_value=httpx.Response(
            200, content=html_page(og="https://cdn.test/a.jpg"),
            headers={"content-type": "text/html"}))

        result = resolve_public_url(PAGE, fetch_policy=OPEN)
        assert result.ok
        assert result.resolution_method is ResolutionMethod.OPENGRAPH
        assert result.resolved_image_url == "https://cdn.test/a.jpg"
        assert result.input_type is UrlType.WEBPAGE

    @respx.mock
    def test_webpage_with_twitter_image_resolves(self):
        """REQUIREMENT 3."""
        respx.get(PAGE).mock(return_value=httpx.Response(
            200, content=html_page(twitter="https://cdn.test/b.jpg"),
            headers={"content-type": "text/html"}))

        result = resolve_public_url(PAGE, fetch_policy=OPEN)
        assert result.ok
        assert result.resolution_method is ResolutionMethod.TWITTER_CARD

    @respx.mock
    def test_extensionless_url_serving_an_image_is_treated_as_direct(self):
        # Extensions are a hint; bytes are the truth.
        url = "https://cdn.test/render?id=9"
        respx.get(url).mock(return_value=httpx.Response(
            200, content=jpeg(), headers={"content-type": "image/jpeg"}))

        result = resolve_public_url(url, fetch_policy=OPEN)
        assert result.ok
        assert result.input_type is UrlType.DIRECT_IMAGE
        assert result.resolution_method is ResolutionMethod.DIRECT


# ==========================================================================
# 6. Honest failure
# ==========================================================================


class TestHonestFailure:
    """A platform refusal stays a refusal, and is explained in its own terms."""

    @respx.mock
    def test_instagram_403_gives_a_platform_specific_error(self):
        """REQUIREMENT 6: NOT 'That link does not return an image'."""
        url = "https://www.instagram.com/p/ABC123/"
        respx.get(url).mock(return_value=httpx.Response(403))

        result = resolve_public_url(url, fetch_policy=OPEN)

        assert not result.ok
        assert result.input_type is UrlType.SOCIAL_POST
        assert result.platform is Platform.INSTAGRAM
        assert result.resolution_status is ResolutionStatus.UNAVAILABLE
        assert "Instagram post detected" in result.reason
        assert "authentication" in result.reason
        assert "does not return an image" not in result.reason

    @respx.mock
    def test_instagram_page_without_preview_still_explains_itself(self):
        url = "https://www.instagram.com/p/NOPREVIEW/"
        respx.get(url).mock(return_value=httpx.Response(
            200, content=html_page(), headers={"content-type": "text/html"}))

        result = resolve_public_url(url, fetch_policy=OPEN)
        assert not result.ok
        assert "Instagram post detected" in result.reason

    @respx.mock
    @pytest.mark.parametrize(
        "url,platform,needle",
        [
            ("https://www.facebook.com/p/1", Platform.FACEBOOK, "Facebook post detected"),
            ("https://www.linkedin.com/posts/x", Platform.LINKEDIN, "LinkedIn post detected"),
            ("https://x.com/u/status/1", Platform.X_TWITTER, "X post detected"),
        ],
    )
    def test_each_platform_gets_its_own_message(self, url, platform, needle):
        respx.get(url).mock(return_value=httpx.Response(403))
        result = resolve_public_url(url, fetch_policy=OPEN)
        assert result.platform is platform
        assert needle in result.reason

    @respx.mock
    def test_generic_page_without_metadata_says_so(self):
        respx.get(PAGE).mock(return_value=httpx.Response(
            200, content=html_page(), headers={"content-type": "text/html"}))
        result = resolve_public_url(PAGE, fetch_policy=OPEN)
        assert not result.ok
        assert "does not publish a preview image" in result.reason

    @respx.mock
    def test_failure_never_claims_a_resolved_image(self):
        """The most important guarantee: no fabricated retrieval."""
        url = "https://www.instagram.com/p/ABC/"
        respx.get(url).mock(return_value=httpx.Response(403))
        payload = resolve_public_url(url, fetch_policy=OPEN).to_dict()

        assert payload["resolution_status"] == "unavailable"
        assert "resolved_image_url" not in payload
        assert payload["reason"]

    @respx.mock
    def test_404_is_explained_plainly(self):
        respx.get(PAGE).mock(return_value=httpx.Response(404))
        result = resolve_public_url(PAGE, fetch_policy=OPEN)
        assert "not found" in result.reason.lower()


# ==========================================================================
# 8. SSRF protection survives resolution
# ==========================================================================


class TestSsrfProtectionSurvives:
    def test_private_target_page_is_blocked(self):
        """REQUIREMENT 8, part one: the PAGE fetch is guarded."""
        result = resolve_public_url("http://169.254.169.254/latest/meta-data/")
        assert not result.ok
        assert "private network" in result.reason

    def test_localhost_page_is_blocked(self):
        result = resolve_public_url("http://127.0.0.1:8000/admin")
        assert not result.ok
        assert "private network" in result.reason

    def test_non_http_scheme_is_refused(self):
        result = resolve_public_url("file:///etc/passwd")
        assert not result.ok
        assert "valid web link" in result.reason

    @respx.mock
    def test_resolved_image_pointing_at_a_private_address_is_still_blocked(self):
        """REQUIREMENT 8, part two, and the subtle one.

        A page can publish ANY og:image, including one aimed at cloud metadata.
        The resolver returns that URL, and Stage 2.5 must block it -- which it
        does because the resolved URL is re-validated, never trusted.
        """
        respx.get(PAGE).mock(return_value=httpx.Response(
            200, content=html_page(og="http://169.254.169.254/latest/meta-data/"),
            headers={"content-type": "text/html"}))

        resolution = resolve_public_url(PAGE, fetch_policy=OPEN)
        # The resolver did its job: it found what the page published.
        assert resolution.ok
        assert "169.254.169.254" in resolution.resolved_image_url

        # Stage 2.5 refuses to fetch it.
        from tracelock.acquisition.fetcher import fetch_media

        blocked = fetch_media(resolution.resolved_image_url)
        assert not blocked.ok
        assert blocked.reason.value == "BLOCKED_URL_TARGET"

    @respx.mock
    def test_oversized_page_is_refused(self):
        respx.get(PAGE).mock(return_value=httpx.Response(
            200, content=b"x" * 200_000, headers={"content-type": "text/html"}))
        result = resolve_public_url(
            PAGE, fetch_policy=FetchPolicy(block_private_targets=False, max_bytes=1000)
        )
        assert not result.ok
        assert "too large" in result.reason


# ==========================================================================
# 4 & 10. Integration with the existing Stage 2.5 path
# ==========================================================================


class TestNoSeparateValidationPath:
    """The resolved URL goes through the SAME guard as everything else."""

    def test_resolved_url_is_handed_to_stage_2_5(self, client, monkeypatch):
        """REQUIREMENT 4."""
        calls = {}

        import tracelock.discovery.search_image as guard
        from tracelock.discovery.search_image import (
            ProbeRelationship,
            SearchImageCheck,
        )

        import sys

        app_module = sys.modules["tracelock.api.app"]
        from tracelock.ingest import adapter_for, classify_url
        from tracelock.ingest.resolve import Method, ResolvedInput

        monkeypatch.setattr(
            app_module, "resolve_input",
            lambda url, **kw: ResolvedInput(
                input_url=url,
                classification=classify_url(url),
                adapter=adapter_for(classify_url(url).platform),
                method=Method.OPENGRAPH,
                image_url="https://cdn.test/resolved.jpg",
            ),
        )

        def spy(url, engine, **kwargs):
            calls["url"] = url
            calls["probe_bytes"] = kwargs.get("probe_bytes")
            return SearchImageCheck(
                url=url, ok=True, image_format="JPEG", width=400, height=400,
                byte_size=1234, faces_detected=1, det_score=0.9,
                quality_aggregate=0.8, quality_band="EXCELLENT",
                probe_relationship=ProbeRelationship(
                    cosine_similarity=0.99, phash_distance=3, is_same_image=True),
            )

        monkeypatch.setattr(guard, "check_search_image", spy)

        selected = client.post(
            "/api/input/upload", files={"file": ("a.jpg", jpeg(), "image/jpeg")}
        ).json()
        payload = client.post("/api/input/link-url", data={
            "sha256": selected["sha256"],
            "url": "https://www.instagram.com/p/ABC/",
        }).json()

        # Stage 2.5 received the RESOLVED url, not the post url.
        assert calls["url"] == "https://cdn.test/resolved.jpg"
        assert calls["probe_bytes"] is not None
        assert payload["ok"] is True

    def test_provenance_records_method_and_input_url(self, client, monkeypatch):
        """REQUIREMENT 10."""
        import sys

        app_module = sys.modules["tracelock.api.app"]
        import tracelock.discovery.search_image as guard
        from tracelock.discovery.search_image import (
            ProbeRelationship,
            SearchImageCheck,
        )
        from tracelock.ingest import adapter_for, classify_url
        from tracelock.ingest.resolve import Method, ResolvedInput

        post_url = "https://www.instagram.com/p/ABC/"
        monkeypatch.setattr(
            app_module, "resolve_input",
            lambda url, **kw: ResolvedInput(
                input_url=url,
                classification=classify_url(url),
                adapter=adapter_for(classify_url(url).platform),
                method=Method.OPENGRAPH,
                image_url="https://cdn.test/resolved.jpg",
            ),
        )
        monkeypatch.setattr(
            guard, "check_search_image",
            lambda url, engine, **kw: SearchImageCheck(
                url=url, ok=True, image_format="JPEG", width=400, height=400,
                faces_detected=1, det_score=0.9, quality_aggregate=0.8,
                quality_band="GOOD",
                probe_relationship=ProbeRelationship(0.99, 3, True)),
        )

        selected = client.post(
            "/api/input/upload", files={"file": ("a.jpg", jpeg(), "image/jpeg")}
        ).json()
        resolution = client.post("/api/input/link-url", data={
            "sha256": selected["sha256"], "url": post_url,
        }).json()["resolution"]

        assert resolution["input_url"] == post_url
        # Platform-qualified in provenance, coarse class alongside it.
        assert resolution["input_type"] == "instagram_post"
        assert resolution["url_class"] == "social_post"
        assert resolution["platform"] == "instagram"
        assert resolution["method"] == "opengraph"
        assert resolution["image_url"] == "https://cdn.test/resolved.jpg"

    def test_unresolvable_link_returns_platform_error_not_generic(
        self, client, monkeypatch
    ):
        import sys

        app_module = sys.modules["tracelock.api.app"]
        from tracelock.ingest import adapter_for, classify_url
        from tracelock.ingest.resolve import Method, ResolvedInput

        monkeypatch.setattr(
            app_module, "resolve_input",
            lambda url, **kw: ResolvedInput(
                input_url=url,
                classification=classify_url(url),
                adapter=adapter_for(classify_url(url).platform),
                method=Method.NONE,
                reason="Instagram post detected, but its image could not be "
                       "retrieved publicly by TRACELOCK.",
            ),
        )

        selected = client.post(
            "/api/input/upload", files={"file": ("a.jpg", jpeg(), "image/jpeg")}
        ).json()
        response = client.post("/api/input/link-url", data={
            "sha256": selected["sha256"], "url": "https://www.instagram.com/p/X/",
        })
        payload = response.json()

        assert response.status_code == 400
        assert "Instagram post detected" in payload["error"]
        assert "does not return an image" not in payload["error"]
        assert payload["resolution"]["ok"] is False
        # The truthfulness guard: no image URL key at all when none was found.
        assert "image_url" not in payload["resolution"]


# ==========================================================================
# 7 & 9. Privacy architecture is unchanged
# ==========================================================================


class TestPrivacyArchitectureIntact:
    def test_local_analysis_still_makes_zero_network_requests(self, tmp_path, monkeypatch):
        """REQUIREMENT 7: Mode B remains offline."""

        def forbidden(*a, **k):
            raise AssertionError("local analysis must not touch the network")

        for target in ("get", "post", "request", "stream"):
            monkeypatch.setattr(httpx, target, forbidden, raising=False)
        monkeypatch.setattr(httpx.Client, "send", forbidden, raising=False)
        monkeypatch.setattr(httpx.Client, "stream", forbidden, raising=False)

        from tracelock.service.inputs import from_upload
        from tracelock.service.local_analysis import analyse_locally

        import tests.test_local_analysis as helpers

        trace = from_upload(jpeg(), "a.jpg", tmp_path)
        report = analyse_locally(trace, helpers.FakeEngine())
        assert report.mode == "LOCAL_ANALYSIS"

    def test_public_investigation_still_requires_explicit_action(self):
        """REQUIREMENT 9: no route resolves or investigates implicitly."""
        import importlib
        import inspect

        app_module = importlib.import_module("tracelock.api.app")

        # Resolution happens ONLY in the link-url route, which the operator
        # triggers by clicking "Check link" -- it is not reachable from
        # selecting an image.
        link = inspect.signature(app_module.link_public_url).parameters
        assert "url" in link and "sha256" in link

        # And the network is still server-owned, never client-supplied.
        for name in ("investigate", "anchor_artifact", "verify_anchor"):
            assert "network" not in inspect.signature(
                getattr(app_module, name)).parameters

    def test_selecting_an_image_does_not_resolve_anything(self, client, monkeypatch):
        import tracelock.service.url_resolver as resolver_module

        def forbidden(*a, **k):
            raise AssertionError("selecting an image must not resolve a URL")

        monkeypatch.setattr(resolver_module, "resolve_public_url", forbidden)
        response = client.post(
            "/api/input/upload", files={"file": ("a.jpg", jpeg(), "image/jpeg")}
        )
        assert response.status_code == 200


# ==========================================================================
# Support statement
# ==========================================================================


class TestSupportStatement:
    def test_does_not_claim_every_website(self):
        lowered = SUPPORT_STATEMENT.lower()
        assert "attempts to resolve" in lowered
        assert "every website" not in lowered
        assert "all websites" not in lowered

    def test_endpoint_serves_it(self, client):
        payload = client.get("/api/resolver/info").json()
        assert payload["support_statement"] == SUPPORT_STATEMENT
        assert "no credentials" in payload["note"].lower()
        assert {p["platform"] for p in payload["platforms"]} >= {
            "instagram", "x", "facebook", "linkedin", "youtube"
        }

    def test_no_authenticated_oembed_endpoints_are_configured(self):
        # Instagram and Facebook oEmbed require an app token; using them would
        # mean authenticating, which this resolver does not do.
        from tracelock.service.url_resolver import _PUBLIC_OEMBED

        assert Platform.INSTAGRAM not in _PUBLIC_OEMBED
        assert Platform.FACEBOOK not in _PUBLIC_OEMBED


class TestUiUsesTheResolver:
    def _read(self, name: str) -> str:
        from pathlib import Path

        return (Path(__file__).resolve().parents[1] / "web" / name).read_text(
            encoding="utf-8"
        )

    def test_heading_and_helper_text_updated(self):
        html = self._read("index.html")
        assert "Add the public link to this image" in html
        assert "direct image link, social-media post, or public webpage" in html
        assert "instagram.com/p/" in html

    def test_platform_specific_logic_is_not_in_app_js(self):
        """The resolver is a backend service, not UI logic.

        app.js may DETECT a platform for a visual affordance, but it must not
        contain resolution logic -- no metadata parsing, no oEmbed endpoints.
        """
        js = self._read("app.js")
        assert "og:image" not in js
        assert "twitter:image" not in js
        assert "oembed" not in js.lower()

    def test_ui_shows_the_four_resolution_steps(self):
        js = self._read("app.js")
        for step in ("Detecting link type", "Resolving publicly available image",
                     "Checking image safety", "Comparing with selected image"):
            assert step in js

    def test_ui_offers_both_failure_actions(self):
        js = self._read("app.js")
        assert "Upload the image instead" in js
        assert "Try another public source" in js
