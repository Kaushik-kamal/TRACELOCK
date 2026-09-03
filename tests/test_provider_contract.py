"""Provider interface contract.

Locks the abstraction that lets Yandex / Bing / TinEye / Bluesky be added later
without touching anything downstream.

NOTE ON TEST FIXTURES
---------------------
The payloads below are synthetic, and that is deliberate and legitimate: they
exercise the PARSER, which is pure. They are never fed to the pipeline as
discovered evidence. The gate script itself has no fixtures and no mock path --
it only ever reports what a live provider actually returned.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from tracelock.core.models import Candidate, ProbeRef
from tracelock.discovery.base import (
    DiscoveryError,
    ProviderBlockedError,
    ProviderConfigError,
    ProviderResult,
    ProviderSchemaError,
    ProviderTransportError,
    Requirement,
    SearchProvider,
    check_requirements,
)
from tracelock.discovery.serpapi_lens import SERPAPI_ENDPOINT, SerpApiProvider

VALID_KEY = "test-key-not-real"
PROBE = ProbeRef(
    local_path="probe.jpg",
    sha256="ab" * 32,
    public_url="https://example.com/probe.jpg",
)


class TestErrorTaxonomy:
    def test_all_errors_share_a_base(self):
        for error_type in (
            ProviderConfigError,
            ProviderBlockedError,
            ProviderTransportError,
            ProviderSchemaError,
        ):
            assert issubclass(error_type, DiscoveryError)

    def test_config_error_is_not_blocked_error(self):
        # The gate maps these to different exit codes: a missing key must never
        # be reported as an architecture risk.
        assert not issubclass(ProviderConfigError, ProviderBlockedError)
        assert not issubclass(ProviderBlockedError, ProviderConfigError)

    def test_schema_error_carries_observed_keys(self):
        error = ProviderSchemaError("nope", observed_keys=["search_metadata", "error"])
        assert error.observed_keys == ["search_metadata", "error"]

    def test_schema_error_defaults_to_empty_keys(self):
        assert ProviderSchemaError("nope").observed_keys == []


class TestProviderConformance:
    def test_serpapi_satisfies_the_protocol(self):
        provider = SerpApiProvider(VALID_KEY)
        assert isinstance(provider, SearchProvider)

    def test_declares_name_engine_and_requirements(self):
        provider = SerpApiProvider(VALID_KEY, engine="yandex_images")
        assert provider.name == "serpapi"
        assert provider.engine == "yandex_images"
        assert Requirement.PUBLIC_IMAGE_URL in provider.requires

    def test_rejects_empty_api_key(self):
        with pytest.raises(ProviderConfigError, match="TL_SERPAPI_API_KEY"):
            SerpApiProvider("")

    def test_rejects_whitespace_api_key(self):
        with pytest.raises(ProviderConfigError):
            SerpApiProvider("   ")

    def test_rejects_unknown_engine(self):
        with pytest.raises(ProviderConfigError, match="engine must be one of"):
            SerpApiProvider(VALID_KEY, engine="altavista")


class TestHttpStatusClassification:
    """HTTP status -> error type. The gate's exit code depends on this mapping,
    so a misclassification here reports a typo'd key as an architecture risk."""

    def _raise(self, status: int, body: str = "{}"):
        request = httpx.Request("GET", "https://serpapi.com/search")
        response = httpx.Response(status, text=body, request=request)
        SerpApiProvider._raise_for_status(response)

    def test_200_passes_through(self):
        self._raise(200)  # must not raise

    def test_401_is_a_config_error_not_a_block(self):
        # An invalid key is a SETUP problem. Reporting it as a block would
        # produce a false "architecture risk" verdict on the day-zero gate.
        with pytest.raises(ProviderConfigError, match="invalid"):
            self._raise(401, '{"error": "Invalid API key"}')

    def test_403_is_a_block(self):
        with pytest.raises(ProviderBlockedError, match="forbade"):
            self._raise(403)

    def test_429_is_a_block(self):
        with pytest.raises(ProviderBlockedError, match="rate limited"):
            self._raise(429)

    @pytest.mark.parametrize("status", [500, 502, 503])
    def test_5xx_is_transport(self, status):
        with pytest.raises(ProviderTransportError):
            self._raise(status)

    def test_unexpected_4xx_is_schema(self):
        with pytest.raises(ProviderSchemaError):
            self._raise(418)


class TestSearchErrorHandling:
    """Exercises the real search() path with the transport mocked.

    Mocking HTTP here is legitimate: it tests OUR error handling, not discovery.
    The gate script has no mock path at all -- it only reports live results.
    """

    @respx.mock
    def test_quota_message_in_200_body_is_a_block(self):
        # SerpAPI reports quota exhaustion inside a 200 response, not a 429.
        respx.get(SERPAPI_ENDPOINT).mock(
            return_value=httpx.Response(
                200, json={"error": "You have run out of searches on your plan"}
            )
        )
        with pytest.raises(ProviderBlockedError, match="quota or plan"):
            SerpApiProvider(VALID_KEY).search(PROBE)

    @respx.mock
    def test_other_error_in_200_body_is_schema(self):
        respx.get(SERPAPI_ENDPOINT).mock(
            return_value=httpx.Response(200, json={"error": "Unsupported engine"})
        )
        with pytest.raises(ProviderSchemaError, match="Unsupported engine"):
            SerpApiProvider(VALID_KEY).search(PROBE)

    @respx.mock
    def test_non_json_body_is_schema_error(self):
        respx.get(SERPAPI_ENDPOINT).mock(
            return_value=httpx.Response(200, text="<html>gateway</html>")
        )
        with pytest.raises(ProviderSchemaError, match="non-JSON"):
            SerpApiProvider(VALID_KEY).search(PROBE)

    @respx.mock
    def test_timeout_is_transport_error(self):
        respx.get(SERPAPI_ENDPOINT).mock(side_effect=httpx.ReadTimeout("slow"))
        with pytest.raises(ProviderTransportError, match="timed out"):
            SerpApiProvider(VALID_KEY).search(PROBE)

    @respx.mock
    def test_successful_search_returns_result_without_api_key(self):
        respx.get(SERPAPI_ENDPOINT).mock(
            return_value=httpx.Response(
                200,
                json={
                    "visual_matches": [
                        {
                            "position": 1,
                            "link": "https://site.example/post/1",
                            "image": "https://site.example/full.jpg",
                            "source": "site.example",
                        }
                    ]
                },
            )
        )
        result = SerpApiProvider(VALID_KEY).search(PROBE)

        assert result.count == 1
        assert len(result.with_media) == 1
        assert result.provider == "serpapi"
        assert result.engine == "google_lens"
        # The query dict is persisted to disk -- the key must never appear.
        assert "api_key" not in result.query
        assert result.query["no_cache"] == "true"
        assert result.elapsed_seconds >= 0

    @respx.mock
    def test_empty_results_is_an_honest_zero_not_an_error(self):
        respx.get(SERPAPI_ENDPOINT).mock(
            return_value=httpx.Response(200, json={"visual_matches": []})
        )
        result = SerpApiProvider(VALID_KEY).search(PROBE)
        assert result.count == 0


class TestRequirementChecking:
    def test_passes_when_probe_has_public_url(self):
        check_requirements(SerpApiProvider(VALID_KEY), PROBE)

    def test_raises_when_public_url_missing(self):
        bare = ProbeRef(local_path="probe.jpg", sha256="ab" * 32)
        with pytest.raises(ProviderConfigError, match="publicly reachable"):
            check_requirements(SerpApiProvider(VALID_KEY), bare)


class TestParser:
    """The parser must be tolerant of drift and honest about failure."""

    def _parse(self, payload, limit=20):
        return SerpApiProvider(VALID_KEY)._parse(payload, limit=limit)

    def test_parses_google_lens_visual_matches(self):
        payload = {
            "search_metadata": {"status": "Success"},
            "visual_matches": [
                {
                    "position": 1,
                    "title": "A post",
                    "link": "https://site.example/post/1",
                    "source": "site.example",
                    "thumbnail": "https://site.example/t.jpg",
                    "image": "https://site.example/full.jpg",
                }
            ],
        }
        candidates = self._parse(payload)
        assert len(candidates) == 1

        candidate = candidates[0]
        assert isinstance(candidate, Candidate)
        assert candidate.provider == "serpapi"
        assert candidate.post_url == "https://site.example/post/1"
        assert candidate.title == "A post"
        assert candidate.rank == 1
        assert candidate.has_media

    def test_falls_back_through_result_keys(self):
        payload = {"image_results": [{"link": "https://a.example/1", "image": "https://a.example/i.jpg"}]}
        assert len(self._parse(payload)) == 1

    def test_tolerates_renamed_url_field(self):
        # `url` instead of `link` -- the alias list absorbs it.
        payload = {"visual_matches": [{"url": "https://a.example/1"}]}
        assert self._parse(payload)[0].post_url == "https://a.example/1"

    def test_drops_rows_with_nothing_addressable(self):
        payload = {"visual_matches": [{"title": "no links here"}, {"link": "https://a.example/1"}]}
        assert len(self._parse(payload)) == 1

    def test_respects_limit(self):
        payload = {
            "visual_matches": [
                {"link": "https://a.example/{0}".format(i)} for i in range(50)
            ]
        }
        assert len(self._parse(payload, limit=5)) == 5

    def test_empty_results_key_is_zero_not_an_error(self):
        # An engine that genuinely found nothing is a valid, honest outcome.
        assert self._parse({"visual_matches": []}) == []

    def test_unknown_schema_raises_with_observed_keys(self):
        payload = {"search_metadata": {"status": "Success"}, "something_new": [{"a": 1}]}
        with pytest.raises(ProviderSchemaError) as caught:
            self._parse(payload)
        assert "search_metadata" in caught.value.observed_keys
        assert "something_new" in caught.value.observed_keys

    def test_ignores_non_dict_entries(self):
        payload = {"visual_matches": ["junk", {"link": "https://a.example/1"}, None]}
        assert len(self._parse(payload)) == 1

    def test_preserves_raw_metadata(self):
        item = {"link": "https://a.example/1", "undocumented_field": "keep me"}
        assert self._parse({"visual_matches": [item]})[0].raw_metadata == item


class TestYandexNestedMedia:
    """Yandex nests media URLs one level deeper than Google Lens.

    Shapes below are copied verbatim from a real captured yandex_images
    response. Google Lens returns `"thumbnail": "https://..."`; Yandex returns
    `"thumbnail": {"link": "https://...", "width": ...}`. Without a nested
    path the URL is silently dropped -- the `with usable media: 0` failure.
    """

    def _parse(self, payload, limit=100):
        return SerpApiProvider(VALID_KEY, engine="yandex_images")._parse(
            payload, limit=limit
        )

    def _image_result(self, n=1):
        return {
            "title": "Good Neighbor Devpost",
            "snippet": "felix-hh Haba",
            "link": "https://devpost.com/software/good-neighbor-{0}".format(n),
            "source": "devpost.com",
            "thumbnail": {
                "link": "https://avatars.mds.yandex.net/i?id=thumb{0}".format(n),
                "height": 90,
                "width": 148,
            },
            "original_image": {
                "link": "https://avatars.githubusercontent.com/u/{0}?v=4".format(n),
                "height": 180,
                "width": 180,
            },
        }

    def _similar_image(self, n=1):
        return {
            "image": {
                "link": "https://avatars.mds.yandex.net/i?id=sim{0}".format(n),
                "height": 320,
                "width": 320,
            },
            "link": "https://yandex.com/images/search?cbir_id=sim{0}".format(n),
        }

    def test_extracts_nested_original_image(self):
        candidate = self._parse({"image_results": [self._image_result()]})[0]
        assert candidate.image_url == "https://avatars.githubusercontent.com/u/1?v=4"

    def test_extracts_nested_thumbnail(self):
        candidate = self._parse({"image_results": [self._image_result()]})[0]
        assert candidate.thumbnail_url == "https://avatars.mds.yandex.net/i?id=thumb1"

    def test_nested_media_makes_the_candidate_usable(self):
        # The exact regression: previously has_media was False for every
        # Yandex candidate, so the gate reported zero usable media.
        assert self._parse({"image_results": [self._image_result()]})[0].has_media

    def test_extracts_real_source_page(self):
        candidate = self._parse({"image_results": [self._image_result()]})[0]
        assert candidate.post_url == "https://devpost.com/software/good-neighbor-1"
        assert candidate.source == "devpost.com"

    def test_similar_images_nested_under_image_key(self):
        candidate = self._parse({"similar_images": [self._similar_image()]})[0]
        assert candidate.image_url == "https://avatars.mds.yandex.net/i?id=sim1"
        assert candidate.has_media

    def test_both_arrays_are_merged_not_first_match_wins(self):
        payload = {
            "image_results": [self._image_result(n) for n in range(1, 4)],
            "similar_images": [self._similar_image(n) for n in range(1, 6)],
        }
        candidates = self._parse(payload)
        assert len(candidates) == 8

    def test_result_group_is_recorded(self):
        payload = {
            "image_results": [self._image_result(1)],
            "similar_images": [self._similar_image(1)],
        }
        groups = [c.result_group for c in self._parse(payload)]
        assert groups == ["image_results", "similar_images"]

    def test_declared_order_is_preserved(self):
        # image_results carries real source pages and must come first.
        payload = {
            "similar_images": [self._similar_image(1)],
            "image_results": [self._image_result(1)],
        }
        assert self._parse(payload)[0].result_group == "image_results"

    def test_overlapping_entries_are_deduped_across_arrays(self):
        shared = {
            "image": {"link": "https://cdn.test/same.jpg"},
            "link": "https://page.test/a",
        }
        payload = {"image_results": [shared], "similar_images": [dict(shared)]}
        assert len(self._parse(payload)) == 1

    def test_limit_applies_across_merged_arrays(self):
        payload = {
            "image_results": [self._image_result(n) for n in range(1, 10)],
            "similar_images": [self._similar_image(n) for n in range(1, 10)],
        }
        assert len(self._parse(payload, limit=5)) == 5

    def test_google_lens_still_uses_only_visual_matches(self):
        # Engine isolation: organic_results must not pollute a Lens run.
        payload = {
            "visual_matches": [{"link": "https://a.test/1", "image": "https://a.test/i.jpg"}],
            "organic_results": [{"link": "https://b.test/2"}],
        }
        candidates = SerpApiProvider(VALID_KEY, engine="google_lens")._parse(
            payload, limit=50
        )
        assert len(candidates) == 1
        assert candidates[0].result_group == "visual_matches"

    def test_google_lens_bare_string_media_still_works(self):
        payload = {
            "visual_matches": [
                {
                    "link": "https://a.test/1",
                    "thumbnail": "https://a.test/t.jpg",
                    "image": "https://a.test/full.jpg",
                }
            ]
        }
        candidate = SerpApiProvider(VALID_KEY, engine="google_lens")._parse(
            payload, limit=10
        )[0]
        assert candidate.thumbnail_url == "https://a.test/t.jpg"
        assert candidate.has_media

    def test_dict_valued_field_does_not_leak_through(self):
        # A dict must be skipped so the NEXT alias resolves, rather than being
        # handed to Candidate where the validator would silently null it.
        payload = {"image_results": [{"link": "https://p.test/1", "thumbnail": {"no_link_key": 1}}]}
        candidate = self._parse(payload)[0]
        assert candidate.thumbnail_url is None
        assert candidate.post_url == "https://p.test/1"

    def test_missing_nested_key_is_tolerated(self):
        payload = {"image_results": [{"link": "https://p.test/1", "original_image": {}}]}
        assert self._parse(payload)[0].image_url is None

    def test_empty_arrays_are_an_honest_zero(self):
        assert self._parse({"image_results": [], "similar_images": []}) == []

    def test_unknown_schema_still_raises_with_observed_keys(self):
        with pytest.raises(ProviderSchemaError) as caught:
            self._parse({"search_metadata": {}, "brand_new_key": [{"a": 1}]})
        assert "brand_new_key" in caught.value.observed_keys


class TestProviderResult:
    def test_counts_and_media_filter(self):
        result = ProviderResult(
            provider="serpapi",
            engine="google_lens",
            candidates=[
                Candidate(provider="serpapi", post_url="https://a.example/1"),
                Candidate(provider="serpapi", image_url="https://a.example/i.jpg"),
            ],
            raw_response={},
        )
        assert result.count == 2
        assert len(result.with_media) == 1

    def test_empty_result_is_valid(self):
        result = ProviderResult(
            provider="serpapi", engine="google_lens", candidates=[], raw_response={}
        )
        assert result.count == 0
        assert result.with_media == []
