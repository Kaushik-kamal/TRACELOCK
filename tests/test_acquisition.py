"""Secure acquisition: fetching, validation, and the CAS.

All HTTP is mocked with respx. The suite must never touch the live internet --
live behaviour is exercised separately by scripts/verify_candidates.py.
"""

from __future__ import annotations

import hashlib
import io

import httpx
import pytest
import respx

from tracelock.acquisition.cas import ContentAddressedStore, SourceReference
from tracelock.acquisition.fetcher import (
    FetchPolicy,
    fetch_media,
    is_blocked_target,
)
from tracelock.acquisition.provenance import (
    describe_source,
    distinct_registrable_domains,
    registrable_domain,
)
from tracelock.acquisition.validation import (
    looks_like_html,
    sniff_format,
    validate_image_bytes,
)
from tracelock.core.reasons import RejectionReason, Stage

pytest.importorskip("cv2")
URL = "https://example.test/photo.jpg"


def jpeg_bytes(width: int = 200, height: int = 200, colour=(120, 90, 60)) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def png_bytes(width: int = 100, height: int = 100) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 200, 40)).save(buffer, format="PNG")
    return buffer.getvalue()


# ==========================================================================
# Format sniffing
# ==========================================================================


class TestSniffFormat:
    def test_detects_jpeg(self):
        assert sniff_format(jpeg_bytes()) == "JPEG"

    def test_detects_png(self):
        assert sniff_format(png_bytes()) == "PNG"

    def test_detects_gif(self):
        assert sniff_format(b"GIF89a" + b"\x00" * 20) == "GIF"

    def test_detects_webp(self):
        assert sniff_format(b"RIFF" + b"\x00" * 4 + b"WEBP" + b"\x00" * 8) == "WEBP"

    def test_rejects_html(self):
        assert sniff_format(b"<!DOCTYPE html><html></html>") is None

    def test_rejects_short_input(self):
        assert sniff_format(b"ab") is None

    def test_rejects_empty(self):
        assert sniff_format(b"") is None


class TestLooksLikeHtml:
    @pytest.mark.parametrize(
        "payload",
        [
            b"<!DOCTYPE html>",
            b"<html><body>404</body></html>",
            b"  \n  <HTML>",
            b"<?xml version='1.0'?>",
            b"<script>alert(1)</script>",
        ],
    )
    def test_detects_markup(self, payload):
        assert looks_like_html(payload)

    def test_does_not_flag_a_jpeg(self):
        assert not looks_like_html(jpeg_bytes())


# ==========================================================================
# Image validation
# ==========================================================================


class TestValidateImageBytes:
    def test_accepts_a_real_jpeg(self):
        result = validate_image_bytes(jpeg_bytes(300, 200))
        assert result.ok
        assert result.detected_format == "JPEG"
        assert (result.width, result.height) == (300, 200)

    def test_accepts_a_real_png(self):
        assert validate_image_bytes(png_bytes()).ok

    def test_rejects_empty(self):
        result = validate_image_bytes(b"")
        assert not result.ok
        assert result.reason is RejectionReason.EMPTY_CONTENT

    def test_rejects_html_served_as_jpeg(self):
        # THE headline case: HTTP 200 + Content-Type: image/jpeg + an HTML
        # error page. Magic bytes are authoritative; the header is not.
        result = validate_image_bytes(
            b"<!DOCTYPE html><html><body>Not found</body></html>",
            declared_content_type="image/jpeg",
        )
        assert not result.ok
        assert result.reason is RejectionReason.NOT_AN_IMAGE
        assert "HTML" in result.detail
        assert "image/jpeg" in result.detail

    def test_rejects_arbitrary_bytes(self):
        result = validate_image_bytes(b"\x00\x01\x02\x03 random junk")
        assert not result.ok
        assert result.reason is RejectionReason.NOT_AN_IMAGE

    def test_rejects_truncated_jpeg(self):
        truncated = jpeg_bytes()[:120]
        result = validate_image_bytes(truncated)
        assert not result.ok
        assert result.reason in (
            RejectionReason.CORRUPT_IMAGE,
            RejectionReason.NOT_AN_IMAGE,
        )

    def test_rejects_image_below_minimum_size(self):
        result = validate_image_bytes(jpeg_bytes(16, 16))
        assert not result.ok
        assert result.reason is RejectionReason.IMAGE_TOO_SMALL

    def test_records_content_type_mismatch_as_a_warning(self):
        result = validate_image_bytes(png_bytes(), declared_content_type="image/jpeg")
        assert result.ok  # a real image, just mislabelled
        assert result.warnings
        assert not result.content_type_was_honest

    def test_honest_content_type_is_recorded(self):
        result = validate_image_bytes(jpeg_bytes(), declared_content_type="image/jpeg")
        assert result.content_type_was_honest

    def test_does_not_mutate_input_bytes(self):
        # ORIGINAL EVIDENCE must survive validation untouched.
        data = jpeg_bytes()
        before = hashlib.sha256(data).hexdigest()
        validate_image_bytes(data)
        assert hashlib.sha256(data).hexdigest() == before

    def test_reason_maps_to_the_validated_stage(self):
        assert validate_image_bytes(b"junk").reason.stage is Stage.VALIDATED


# ==========================================================================
# SSRF guard
# ==========================================================================


class TestBlockedTargets:
    @pytest.mark.parametrize(
        "host",
        [
            "127.0.0.1",
            "localhost",
            "169.254.169.254",  # cloud metadata
            "10.0.0.5",
            "192.168.1.1",
            "172.16.0.1",
            "0.0.0.0",
            "::1",
        ],
    )
    def test_blocks_non_public_targets(self, host):
        blocked, why = is_blocked_target(host)
        assert blocked, why

    def test_blocks_empty_host(self):
        assert is_blocked_target("")[0]

    def test_allows_a_public_address(self):
        assert not is_blocked_target("93.184.216.34")[0]


# ==========================================================================
# Fetching
# ==========================================================================


class TestFetchMedia:
    @respx.mock
    def test_successful_download(self):
        payload = jpeg_bytes()
        respx.get(URL).mock(
            return_value=httpx.Response(
                200, content=payload, headers={"content-type": "image/jpeg"}
            )
        )
        result = fetch_media(URL, policy=FetchPolicy(block_private_targets=False))

        assert result.ok
        assert result.status_code == 200
        assert result.byte_size == len(payload)
        assert result.sha256 == hashlib.sha256(payload).hexdigest()
        assert result.declared_content_type == "image/jpeg"

    @respx.mock
    def test_sha256_is_computed_from_received_bytes(self):
        payload = jpeg_bytes(64, 64)
        respx.get(URL).mock(return_value=httpx.Response(200, content=payload))
        result = fetch_media(URL, policy=FetchPolicy(block_private_targets=False))
        assert result.sha256 == hashlib.sha256(payload).hexdigest()

    @respx.mock
    def test_http_404_is_an_error(self):
        respx.get(URL).mock(return_value=httpx.Response(404, text="gone"))
        result = fetch_media(URL, policy=FetchPolicy(block_private_targets=False))
        assert not result.ok
        assert result.reason is RejectionReason.HTTP_ERROR
        assert result.status_code == 404

    @respx.mock
    def test_http_403_is_an_error(self):
        respx.get(URL).mock(return_value=httpx.Response(403))
        assert fetch_media(
            URL, policy=FetchPolicy(block_private_targets=False)
        ).reason is RejectionReason.HTTP_ERROR

    @respx.mock
    def test_timeout(self):
        respx.get(URL).mock(side_effect=httpx.ReadTimeout("slow"))
        result = fetch_media(URL, policy=FetchPolicy(block_private_targets=False))
        assert not result.ok
        assert result.reason is RejectionReason.DOWNLOAD_TIMEOUT

    @respx.mock
    def test_connection_error(self):
        respx.get(URL).mock(side_effect=httpx.ConnectError("refused"))
        result = fetch_media(URL, policy=FetchPolicy(block_private_targets=False))
        assert not result.ok
        assert result.reason is RejectionReason.DOWNLOAD_FAILED

    @respx.mock
    def test_follows_a_redirect_and_records_the_final_url(self):
        final = "https://cdn.example.test/real.jpg"
        respx.get(URL).mock(
            return_value=httpx.Response(302, headers={"location": final})
        )
        respx.get(final).mock(return_value=httpx.Response(200, content=jpeg_bytes()))

        result = fetch_media(URL, policy=FetchPolicy(block_private_targets=False))
        assert result.ok
        assert result.redirect_count == 1
        assert result.final_url == final
        assert result.url_changed

    @respx.mock
    def test_redirect_loop_is_bounded(self):
        a, b = "https://example.test/a", "https://example.test/b"
        respx.get(a).mock(return_value=httpx.Response(302, headers={"location": b}))
        respx.get(b).mock(return_value=httpx.Response(302, headers={"location": a}))

        result = fetch_media(
            a, policy=FetchPolicy(block_private_targets=False, max_redirects=4)
        )
        assert not result.ok
        assert result.reason is RejectionReason.TOO_MANY_REDIRECTS

    @respx.mock
    def test_oversized_declared_content_length_is_refused_early(self):
        respx.get(URL).mock(
            return_value=httpx.Response(
                200, content=jpeg_bytes(), headers={"content-length": "999999999"}
            )
        )
        result = fetch_media(
            URL, policy=FetchPolicy(block_private_targets=False, max_bytes=1000)
        )
        assert not result.ok
        assert result.reason is RejectionReason.CONTENT_TOO_LARGE
        assert "Content-Length" in result.detail

    @respx.mock
    def test_lying_content_length_is_caught_by_the_byte_counter(self):
        # A server UNDERSTATES Content-Length so the early check passes, then
        # sends a much larger body. This is why the header is advisory and the
        # per-chunk counter is what actually enforces the cap.
        big = jpeg_bytes(800, 800)
        assert len(big) > 5000

        respx.get(URL).mock(
            return_value=httpx.Response(
                200, content=big, headers={"content-length": "100"}
            )
        )
        result = fetch_media(
            URL, policy=FetchPolicy(block_private_targets=False, max_bytes=5000)
        )
        assert not result.ok
        assert result.reason is RejectionReason.CONTENT_TOO_LARGE
        assert "aborted" in result.detail
        # The abort happened mid-stream, so more than the cap was never buffered.
        assert result.byte_size > 5000

    @respx.mock
    def test_empty_body_with_http_200(self):
        respx.get(URL).mock(return_value=httpx.Response(200, content=b""))
        result = fetch_media(URL, policy=FetchPolicy(block_private_targets=False))
        assert not result.ok
        assert result.reason is RejectionReason.EMPTY_CONTENT

    @respx.mock
    def test_html_masquerading_as_jpeg_downloads_then_fails_validation(self):
        # Acquisition succeeds -- the bytes arrived. Validation is what rejects.
        respx.get(URL).mock(
            return_value=httpx.Response(
                200,
                content=b"<!DOCTYPE html><html>login</html>",
                headers={"content-type": "image/jpeg"},
            )
        )
        acquired = fetch_media(URL, policy=FetchPolicy(block_private_targets=False))
        assert acquired.ok

        validated = validate_image_bytes(
            acquired.content, declared_content_type=acquired.declared_content_type
        )
        assert not validated.ok
        assert validated.reason is RejectionReason.NOT_AN_IMAGE

    def test_rejects_non_http_scheme(self):
        for url in ("file:///etc/passwd", "ftp://x.test/a.jpg", "data:image/png;base64,AA"):
            result = fetch_media(url)
            assert not result.ok
            assert result.reason is RejectionReason.INVALID_URL

    def test_rejects_empty_url(self):
        assert fetch_media("").reason is RejectionReason.INVALID_URL

    def test_blocks_ssrf_target(self):
        result = fetch_media("http://127.0.0.1:8080/admin")
        assert not result.ok
        assert result.reason is RejectionReason.BLOCKED_URL_TARGET

    @respx.mock
    def test_only_safe_headers_are_retained(self):
        respx.get(URL).mock(
            return_value=httpx.Response(
                200,
                content=jpeg_bytes(),
                headers={
                    "content-type": "image/jpeg",
                    "set-cookie": "session=SECRET",
                    "authorization": "Bearer SECRET",
                },
            )
        )
        result = fetch_media(URL, policy=FetchPolicy(block_private_targets=False))
        # These headers reach the run artifact on disk.
        assert "set-cookie" not in result.headers
        assert "authorization" not in result.headers
        assert "content-type" in result.headers

    @respx.mock
    def test_response_body_is_excluded_from_serialization(self):
        respx.get(URL).mock(return_value=httpx.Response(200, content=jpeg_bytes()))
        result = fetch_media(URL, policy=FetchPolicy(block_private_targets=False))
        assert "content" not in result.to_dict()


# ==========================================================================
# CAS
# ==========================================================================


class TestContentAddressedStore:
    def test_stores_by_content_hash(self, tmp_path):
        store = ContentAddressedStore(tmp_path)
        payload = jpeg_bytes()
        entry = store.put(payload)

        assert entry.sha256 == hashlib.sha256(payload).hexdigest()
        assert entry.path.is_file()
        assert entry.path.read_bytes() == payload
        assert not entry.was_duplicate

    def test_identical_bytes_store_once(self, tmp_path):
        store = ContentAddressedStore(tmp_path)
        payload = jpeg_bytes()

        first = store.put(payload, SourceReference("https://a.test/1.jpg"))
        second = store.put(payload, SourceReference("https://b.test/2.jpg"))

        assert first.sha256 == second.sha256
        assert not first.was_duplicate
        assert second.was_duplicate
        assert store.stats()["blob_count"] == 1

    def test_both_source_references_are_preserved(self, tmp_path):
        store = ContentAddressedStore(tmp_path)
        payload = jpeg_bytes()

        store.put(payload, SourceReference("https://a.test/1.jpg", candidate_id="c1"))
        entry = store.put(payload, SourceReference("https://b.test/2.jpg", candidate_id="c2"))

        urls = {r.requested_url for r in entry.references}
        assert urls == {"https://a.test/1.jpg", "https://b.test/2.jpg"}
        assert entry.reference_count == 2

    def test_same_url_twice_is_not_double_counted(self, tmp_path):
        store = ContentAddressedStore(tmp_path)
        payload = jpeg_bytes()
        store.put(payload, SourceReference("https://a.test/1.jpg"))
        entry = store.put(payload, SourceReference("https://a.test/1.jpg"))
        assert entry.reference_count == 1

    def test_different_bytes_store_separately(self, tmp_path):
        store = ContentAddressedStore(tmp_path)
        store.put(jpeg_bytes(colour=(10, 10, 10)))
        store.put(jpeg_bytes(colour=(200, 200, 200)))
        assert store.stats()["blob_count"] == 2

    def test_round_trips_content(self, tmp_path):
        store = ContentAddressedStore(tmp_path)
        payload = png_bytes()
        entry = store.put(payload)
        assert store.get(entry.sha256) == payload

    def test_contains(self, tmp_path):
        store = ContentAddressedStore(tmp_path)
        entry = store.put(jpeg_bytes())
        assert store.contains(entry.sha256)
        assert not store.contains("0" * 64)

    def test_get_missing_returns_none(self, tmp_path):
        assert ContentAddressedStore(tmp_path).get("a" * 64) is None

    def test_integrity_check_passes_for_intact_blob(self, tmp_path):
        store = ContentAddressedStore(tmp_path)
        entry = store.put(jpeg_bytes())
        assert store.verify_integrity(entry.sha256)

    def test_integrity_check_detects_corruption(self, tmp_path):
        store = ContentAddressedStore(tmp_path)
        entry = store.put(jpeg_bytes())
        entry.path.write_bytes(b"tampered")
        assert not store.verify_integrity(entry.sha256)

    def test_shards_by_hash_prefix(self, tmp_path):
        store = ContentAddressedStore(tmp_path)
        entry = store.put(jpeg_bytes())
        assert entry.path.parent.name == entry.sha256[:2]

    def test_rejects_a_malformed_digest(self, tmp_path):
        store = ContentAddressedStore(tmp_path)
        for bad in ("short", "X" * 64, "ZZ" * 32):
            with pytest.raises(ValueError, match="SHA-256"):
                store.blob_path(bad)

    def test_survives_a_reopened_store(self, tmp_path):
        entry = ContentAddressedStore(tmp_path).put(
            jpeg_bytes(), SourceReference("https://a.test/1.jpg")
        )
        reopened = ContentAddressedStore(tmp_path)
        assert reopened.contains(entry.sha256)
        assert len(reopened.references(entry.sha256)) == 1


# ==========================================================================
# Provenance
# ==========================================================================


class TestProvenance:
    def test_simple_domain(self):
        assert registrable_domain("https://example.com/a") == "example.com"

    def test_strips_subdomain(self):
        assert registrable_domain("https://media.licdn.com/x") == "licdn.com"

    def test_handles_multi_label_public_suffix(self):
        # The Phase 1 carry-forward defect. The naive `www.`-stripping
        # approximation returned "www.bbc.co.uk" here.
        assert registrable_domain("https://www.bbc.co.uk/news") == "bbc.co.uk"
        assert registrable_domain("https://a.b.co.uk/") == "b.co.uk"

    def test_private_suffix_default_is_conservative(self):
        # Default under-counts distinct publishers, which is the only safe
        # direction: over-counting would inflate a Phase 3 trust score.
        assert registrable_domain("https://user.github.io/p") == "github.io"

    def test_private_suffix_can_be_enabled(self):
        assert (
            registrable_domain("https://user.github.io/p", include_private=True)
            == "user.github.io"
        )

    def test_decomposes_a_url(self):
        provenance = describe_source("https://media.licdn.com/dms/image/x")
        assert provenance.scheme == "https"
        assert provenance.host == "media.licdn.com"
        assert provenance.subdomain == "media"
        assert provenance.domain == "licdn"
        assert provenance.suffix == "com"
        assert provenance.is_resolvable

    def test_ip_literal_has_no_registrable_domain(self):
        assert not describe_source("https://93.184.216.34/x").is_resolvable

    def test_empty_url_does_not_raise(self):
        assert describe_source("").registrable_domain == ""

    def test_counts_distinct_domains(self):
        urls = [
            "https://media.licdn.com/a",
            "https://www.licdn.com/b",   # same registrable domain
            "https://pbs.twimg.com/c",
            "https://assets.skool.com/d",
        ]
        assert distinct_registrable_domains(urls) == {
            "licdn.com", "twimg.com", "skool.com"
        }

    def test_offline_mode_records_psl_source(self):
        assert describe_source("https://example.com").psl_source == "bundled-snapshot"
