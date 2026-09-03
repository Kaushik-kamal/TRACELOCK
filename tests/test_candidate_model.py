"""Candidate normalization.

Normalization is where provider chaos becomes a single clean type. If it is
wrong, dedup breaks and every downstream count -- including the corroboration
signal in the trust score -- is wrong with it.
"""

from __future__ import annotations

import pytest

from tracelock.core.models import (
    Candidate,
    ProbeRef,
    normalize_url,
    registrable_host,
)


class TestNormalizeUrl:
    def test_strips_tracking_params(self):
        url = "https://example.com/post?id=7&utm_source=x&fbclid=abc&utm_medium=y"
        assert normalize_url(url) == "https://example.com/post?id=7"

    def test_keeps_meaningful_params(self):
        # `ref` is deliberately NOT stripped -- real sites route on it.
        url = "https://example.com/p?ref=feed&page=2"
        assert normalize_url(url) == "https://example.com/p?page=2&ref=feed"

    def test_sorts_query_for_stable_comparison(self):
        a = normalize_url("https://example.com/p?b=2&a=1")
        b = normalize_url("https://example.com/p?a=1&b=2")
        assert a == b

    def test_lowercases_host_but_not_path(self):
        assert normalize_url("https://EXAMPLE.com/MyPost") == "https://example.com/MyPost"

    def test_drops_fragment(self):
        assert normalize_url("https://example.com/p#section") == "https://example.com/p"

    def test_drops_default_port_keeps_custom(self):
        assert normalize_url("https://example.com:443/p") == "https://example.com/p"
        assert normalize_url("https://example.com:8443/p") == "https://example.com:8443/p"

    def test_adds_scheme_to_protocol_relative(self):
        assert normalize_url("//cdn.example.com/i.jpg") == "https://cdn.example.com/i.jpg"

    def test_empty_path_becomes_root(self):
        assert normalize_url("https://example.com") == "https://example.com/"

    @pytest.mark.parametrize(
        "bad", [None, "", "   ", "not a url", "ftp://example.com/x", "javascript:alert(1)"]
    )
    def test_rejects_unusable(self, bad):
        # Missing/odd URLs are data, not exceptions -- providers omit fields
        # constantly and one bad row must not kill a run.
        assert normalize_url(bad) is None


class TestRegistrableHost:
    def test_strips_www(self):
        assert registrable_host("https://www.example.com/p") == "example.com"

    def test_plain_host(self):
        assert registrable_host("https://bsky.app/profile/x") == "bsky.app"

    def test_none_for_missing(self):
        assert registrable_host(None) is None

    def test_known_limitation_multi_label_suffix(self):
        # Documents current behaviour, which is WRONG for eTLD+1 and gets
        # fixed with tldextract in Phase 2 when corroboration starts counting
        # independent domains. Asserting it here means the fix cannot land
        # silently.
        assert registrable_host("https://user.github.io/p") == "user.github.io"


class TestCandidate:
    def test_normalizes_urls_on_construction(self):
        candidate = Candidate(
            provider="serpapi",
            post_url="https://EXAMPLE.com/post?utm_source=x",
            image_url="//cdn.example.com/i.jpg",
        )
        assert candidate.post_url == "https://example.com/post"
        assert candidate.image_url == "https://cdn.example.com/i.jpg"

    def test_derives_source_from_post_url(self):
        candidate = Candidate(provider="serpapi", post_url="https://www.bbc.co.uk/news/1")
        assert candidate.source == "bbc.co.uk"

    def test_explicit_source_wins(self):
        candidate = Candidate(
            provider="serpapi", post_url="https://example.com/p", source="BBC News"
        )
        assert candidate.source == "BBC News"

    def test_collapses_whitespace_in_text(self):
        candidate = Candidate(provider="serpapi", title="  hello \n\t world  ")
        assert candidate.title == "hello world"

    def test_blank_text_becomes_none(self):
        assert Candidate(provider="serpapi", title="   ").title is None

    def test_non_string_urls_become_none(self):
        candidate = Candidate(provider="serpapi", post_url=12345, image_url={"a": 1})
        assert candidate.post_url is None
        assert candidate.image_url is None

    def test_has_media_requires_image_or_thumb(self):
        assert not Candidate(provider="p", post_url="https://e.com/1").has_media
        assert Candidate(provider="p", image_url="https://e.com/i.jpg").has_media
        assert Candidate(provider="p", thumbnail_url="https://e.com/t.jpg").has_media

    def test_dedup_key_prefers_post_url(self):
        candidate = Candidate(
            provider="p",
            post_url="https://e.com/post?utm_source=a",
            image_url="https://e.com/i.jpg",
        )
        assert candidate.dedup_key == "https://e.com/post"

    def test_dedup_key_falls_back_to_image(self):
        candidate = Candidate(provider="p", image_url="https://e.com/i.jpg")
        assert candidate.dedup_key == "https://e.com/i.jpg"

    def test_tracking_variants_dedup_together(self):
        a = Candidate(provider="p", post_url="https://e.com/x?utm_source=twitter")
        b = Candidate(provider="p", post_url="https://e.com/x?fbclid=99")
        assert a.dedup_key == b.dedup_key

    def test_raw_metadata_is_preserved_verbatim(self):
        raw = {"weird_key": [1, 2, {"nested": True}], "position": 3}
        candidate = Candidate(provider="p", post_url="https://e.com/1", raw_metadata=raw)
        assert candidate.raw_metadata == raw

    def test_discovered_at_is_timezone_aware(self):
        assert Candidate(provider="p").discovered_at.tzinfo is not None

    def test_serializes_to_json_safe_dict(self):
        candidate = Candidate(provider="p", post_url="https://e.com/1")
        dumped = candidate.model_dump(mode="json")
        assert isinstance(dumped["discovered_at"], str)
        assert dumped["provider"] == "p"


class TestProbeRef:
    def test_normalizes_public_url(self):
        probe = ProbeRef(
            local_path="a.jpg", sha256="ab" * 32, public_url="https://E.com/i.jpg?utm_id=1"
        )
        assert probe.public_url == "https://e.com/i.jpg"

    def test_public_url_optional(self):
        assert ProbeRef(local_path="a.jpg", sha256="ab" * 32).public_url is None

    def test_is_frozen(self):
        probe = ProbeRef(local_path="a.jpg", sha256="ab" * 32)
        with pytest.raises(Exception):
            probe.local_path = "b.jpg"
