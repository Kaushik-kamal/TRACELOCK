"""Secure media acquisition.

Candidate URLs come from a third-party search API. They are UNTRUSTED INPUT and
are treated as hostile, not merely unreliable.

CONTROLS
--------
  scheme allowlist       http/https only -- blocks file://, data://, ftp://
  SSRF guard             refuses private/loopback/link-local targets
  explicit timeouts      connect and read, separately
  redirect limit         bounded; also covers redirect loops
  streaming download     size enforced DURING the stream, not after
  size cap               a declared Content-Length is advisory, the byte
                         counter is authoritative
  magic-byte validation  in `validation.py`; headers are never trusted
  SHA-256                computed over the bytes we actually received

WHY THE SIZE CAP IS ENFORCED MID-STREAM
---------------------------------------
Checking `Content-Length` before downloading is trivially defeated: a server
can omit it, understate it, or use chunked encoding. The only reliable limit is
a counter incremented per chunk that aborts the connection when exceeded.
"""

from __future__ import annotations

import hashlib
import ipaddress
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import httpx

from tracelock.core.reasons import RejectionReason

DEFAULT_TIMEOUT = 20.0
DEFAULT_MAX_BYTES = 25 * 1024 * 1024  # 25 MB
DEFAULT_MAX_REDIRECTS = 5
DEFAULT_MAX_DURATION = 30.0
DEFAULT_CHUNK_SIZE = 64 * 1024

USER_AGENT = "TRACELOCK/0.2 (evidence verification; +research)"

# Response headers worth keeping. Deliberately excludes Set-Cookie and any
# auth-bearing header -- this dict is written to disk in the run artifact.
SAFE_RESPONSE_HEADERS: frozenset[str] = frozenset(
    {
        "content-type",
        "content-length",
        "last-modified",
        "etag",
        "date",
        "server",
        "cache-control",
        "content-disposition",
    }
)


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    """Outcome of one download attempt. Produced on failure as well as success."""

    requested_url: str
    ok: bool
    final_url: str | None = None
    status_code: int | None = None
    content: bytes | None = None
    sha256: str | None = None
    byte_size: int = 0
    declared_content_type: str | None = None
    redirect_count: int = 0
    elapsed_seconds: float = 0.0
    headers: dict[str, str] = field(default_factory=dict)
    reason: RejectionReason | None = None
    detail: str = ""

    @property
    def was_redirected(self) -> bool:
        return self.redirect_count > 0

    @property
    def url_changed(self) -> bool:
        """Did the URL we ended at differ from the one the provider gave us?

        A provenance signal: the bytes came from somewhere other than the
        advertised location.
        """
        return bool(self.final_url) and self.final_url != self.requested_url

    def to_dict(self) -> dict[str, Any]:
        """Serializable form. The response BODY is deliberately excluded."""
        return {
            "requested_url": self.requested_url,
            "ok": self.ok,
            "final_url": self.final_url,
            "url_changed": self.url_changed,
            "status_code": self.status_code,
            "sha256": self.sha256,
            "byte_size": self.byte_size,
            "declared_content_type": self.declared_content_type,
            "redirect_count": self.redirect_count,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "headers": self.headers,
            "reason": self.reason.value if self.reason else None,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class FetchPolicy:
    """Acquisition limits. Config-driven so the run artifact can record them."""

    timeout: float = DEFAULT_TIMEOUT
    max_bytes: int = DEFAULT_MAX_BYTES
    max_redirects: int = DEFAULT_MAX_REDIRECTS
    chunk_size: int = DEFAULT_CHUNK_SIZE
    block_private_targets: bool = True
    user_agent: str = USER_AGENT

    # Wall-clock ceiling for the WHOLE download, not per read.
    #
    # httpx's timeout is per socket operation, so a server that dribbles a few
    # bytes every second satisfies every individual read and the transfer never
    # times out. Measured: a server sending 16 bytes/second was still being
    # read after 120 seconds, with the size cap nowhere near reached. That is
    # both an indefinite hang and a slowloris-style way to tie up a worker.
    max_duration: float = DEFAULT_MAX_DURATION

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeout": self.timeout,
            "max_bytes": self.max_bytes,
            "max_duration": self.max_duration,
            "max_redirects": self.max_redirects,
            "block_private_targets": self.block_private_targets,
            "user_agent": self.user_agent,
        }


def _fail(url: str, reason: RejectionReason, detail: str, **extra) -> AcquisitionResult:
    return AcquisitionResult(
        requested_url=url, ok=False, reason=reason, detail=detail, **extra
    )


def is_blocked_target(host: str) -> tuple[bool, str]:
    """Would fetching this host reach our own infrastructure?

    Resolves the name and inspects every address it returns. A hostile or
    compromised provider result could point at cloud metadata
    (169.254.169.254), an internal service, or localhost. Since the URL comes
    from an external API, this is a genuine SSRF vector and not a theoretical
    one.

    Known limitation: this is a check-then-use race (DNS rebinding). Closing it
    properly requires pinning the resolved IP into the connection, which httpx
    does not expose cleanly. Documented rather than silently ignored.
    """
    if not host:
        return True, "empty host"

    # Literal IPs skip resolution.
    try:
        address = ipaddress.ip_address(host)
        if _is_private(address):
            return True, "URL targets a non-public address: {0}".format(address)
        return False, ""
    except ValueError:
        pass

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        return True, "host does not resolve: {0}".format(exc)

    for info in infos:
        raw = info[4][0]
        try:
            address = ipaddress.ip_address(raw)
        except ValueError:
            continue
        if _is_private(address):
            return True, "{0} resolves to a non-public address: {1}".format(
                host, address
            )

    return False, ""


# Names that always mean "this machine" or "this network" and never need a
# resolver to recognise. Not exhaustive -- it does not have to be, because
# `is_blocked_target` still runs with full DNS before anything is fetched.
_LOCAL_SUFFIXES = (
    "localhost", ".localhost", ".local", ".internal", ".localdomain",
)


def is_obviously_private(host: str) -> tuple[bool, str]:
    """Offline SSRF pre-filter: literal private IPs and local-only names.

    Deliberately does NO name resolution, so callers that only need to decide
    whether to CLAIM a target is reachable can use it without a DNS round trip
    and without becoming untestable offline.

    This does not replace `is_blocked_target`, which stays the authoritative
    check and still runs inside `fetch_media` before any connection. This one
    exists so a component that returns a URL without fetching it -- the URL
    resolver -- cannot report success for an address we would refuse to fetch.
    """
    host = (host or "").strip().lower().strip("[]")
    if not host:
        return True, "empty host"

    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        for suffix in _LOCAL_SUFFIXES:
            if host == suffix.lstrip(".") or host.endswith(suffix):
                return True, "local-only hostname: {0}".format(host)
        return False, ""

    if _is_private(address):
        return True, "URL targets a non-public address: {0}".format(address)
    return False, ""


def _is_private(address) -> bool:
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )


def fetch_media(
    url: str,
    *,
    policy: FetchPolicy | None = None,
    client: httpx.Client | None = None,
) -> AcquisitionResult:
    """Download one candidate URL under the security policy.

    Never raises for an unreachable or hostile URL -- every failure comes back
    as an AcquisitionResult carrying a RejectionReason, because a failure is a
    finding that must reach the artifact.
    """
    policy = policy or FetchPolicy()

    if not url or not url.strip():
        return _fail(url or "", RejectionReason.INVALID_URL, "empty URL")

    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        return _fail(
            url,
            RejectionReason.INVALID_URL,
            "scheme {0!r} is not http or https".format(parts.scheme),
        )
    if not parts.hostname:
        return _fail(url, RejectionReason.INVALID_URL, "URL has no host")

    if policy.block_private_targets:
        blocked, why = is_blocked_target(parts.hostname)
        if blocked:
            return _fail(url, RejectionReason.BLOCKED_URL_TARGET, why)

    owns_client = client is None
    if owns_client:
        client = httpx.Client(
            timeout=httpx.Timeout(policy.timeout),
            follow_redirects=True,
            max_redirects=policy.max_redirects,
            headers={"User-Agent": policy.user_agent},
        )

    started = time.perf_counter()
    try:
        return _stream_download(url, client, policy, started)
    finally:
        if owns_client:
            client.close()


def _stream_download(
    url: str, client: httpx.Client, policy: FetchPolicy, started: float
) -> AcquisitionResult:
    try:
        with client.stream("GET", url) as response:
            elapsed = time.perf_counter() - started
            redirects = len(response.history)
            headers = {
                key: value
                for key, value in response.headers.items()
                if key.lower() in SAFE_RESPONSE_HEADERS
            }
            content_type = response.headers.get("content-type")

            if response.status_code != 200:
                return AcquisitionResult(
                    requested_url=url,
                    ok=False,
                    final_url=str(response.url),
                    status_code=response.status_code,
                    declared_content_type=content_type,
                    redirect_count=redirects,
                    elapsed_seconds=elapsed,
                    headers=headers,
                    reason=RejectionReason.HTTP_ERROR,
                    detail="server returned HTTP {0}".format(response.status_code),
                )

            # Content-Length is advisory. Used only for an early abort;
            # the byte counter below is what actually enforces the cap.
            declared = response.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > policy.max_bytes:
                return AcquisitionResult(
                    requested_url=url,
                    ok=False,
                    final_url=str(response.url),
                    status_code=response.status_code,
                    declared_content_type=content_type,
                    redirect_count=redirects,
                    elapsed_seconds=elapsed,
                    headers=headers,
                    reason=RejectionReason.CONTENT_TOO_LARGE,
                    detail="declared Content-Length {0} exceeds the {1} byte "
                    "cap".format(declared, policy.max_bytes),
                )

            digest = hashlib.sha256()
            chunks: list[bytes] = []
            total = 0

            # Wall-clock ceiling for the WHOLE transfer.
            #
            # A per-chunk check is not enough: iter_bytes(chunk_size) BLOCKS
            # until it has buffered a full chunk, so a server dribbling 16
            # bytes/second never reaches the loop body at all. Measured, that
            # ran past 120s with the size cap nowhere near reached.
            #
            # httpx has no total-deadline option, so the response is closed
            # from a watchdog timer. Closing breaks the iterator, which is
            # caught below and reported as a timeout.
            deadline = threading.Timer(
                max(0.1, policy.max_duration - (time.perf_counter() - started)),
                response.close,
            )
            deadline.daemon = True
            deadline.start()

            timed_out = False
            try:
                for chunk in response.iter_bytes(policy.chunk_size):
                    total += len(chunk)
                    if total > policy.max_bytes:
                        return AcquisitionResult(
                            requested_url=url,
                            ok=False,
                            final_url=str(response.url),
                            status_code=response.status_code,
                            declared_content_type=content_type,
                            byte_size=total,
                            redirect_count=redirects,
                            elapsed_seconds=time.perf_counter() - started,
                            headers=headers,
                            reason=RejectionReason.CONTENT_TOO_LARGE,
                            detail="stream exceeded the {0} byte cap and was "
                            "aborted".format(policy.max_bytes),
                        )
                    digest.update(chunk)
                    chunks.append(chunk)
            except (httpx.HTTPError, httpx.StreamError, OSError, RuntimeError):
                # The watchdog closed the response mid-read, or the transport
                # failed. Either way the bytes are incomplete and must not be
                # treated as content.
                timed_out = True
            finally:
                deadline.cancel()

            if timed_out or time.perf_counter() - started > policy.max_duration:
                return AcquisitionResult(
                    requested_url=url,
                    ok=False,
                    final_url=str(response.url),
                    status_code=response.status_code,
                    declared_content_type=content_type,
                    byte_size=total,
                    redirect_count=redirects,
                    elapsed_seconds=time.perf_counter() - started,
                    headers=headers,
                    reason=RejectionReason.DOWNLOAD_TIMEOUT,
                    detail="transfer exceeded the {0:.0f}s ceiling and was "
                    "aborted after {1} bytes".format(policy.max_duration, total),
                )

            content = b"".join(chunks)
            elapsed = time.perf_counter() - started

            if not content:
                return AcquisitionResult(
                    requested_url=url,
                    ok=False,
                    final_url=str(response.url),
                    status_code=response.status_code,
                    declared_content_type=content_type,
                    redirect_count=redirects,
                    elapsed_seconds=elapsed,
                    headers=headers,
                    reason=RejectionReason.EMPTY_CONTENT,
                    detail="HTTP 200 with an empty body",
                )

            return AcquisitionResult(
                requested_url=url,
                ok=True,
                final_url=str(response.url),
                status_code=response.status_code,
                content=content,
                sha256=digest.hexdigest(),
                byte_size=len(content),
                declared_content_type=content_type,
                redirect_count=redirects,
                elapsed_seconds=elapsed,
                headers=headers,
            )

    except httpx.TooManyRedirects as exc:
        return _fail(
            url,
            RejectionReason.TOO_MANY_REDIRECTS,
            "exceeded {0} redirects (this also covers redirect loops): "
            "{1}".format(policy.max_redirects, exc),
            elapsed_seconds=time.perf_counter() - started,
        )
    except httpx.TimeoutException:
        return _fail(
            url,
            RejectionReason.DOWNLOAD_TIMEOUT,
            "no response within {0}s".format(policy.timeout),
            elapsed_seconds=time.perf_counter() - started,
        )
    except httpx.HTTPError as exc:
        return _fail(
            url,
            RejectionReason.DOWNLOAD_FAILED,
            "{0}: {1}".format(type(exc).__name__, exc),
            elapsed_seconds=time.perf_counter() - started,
        )
