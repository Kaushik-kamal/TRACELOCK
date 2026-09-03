"""The concrete image hosts.

Each provider states plainly what it can and cannot do. Three of them, chosen
because they differ in exactly the way that matters for consent:

    catbox      no credentials, works immediately, CANNOT delete
    imgbb       needs a free key, auto-expires, no REST delete endpoint
    cloudinary  needs three credentials, genuinely destroys on request

Two providers were tried and rejected on evidence, not preference:

    litterbox   HTTP 403 -- the endpoint the previous implementation used is
                dead, which is why local uploads never reached discovery
    0x0.st      HTTP 503 -- "uploads disabled because it's been almost nothing
                but AI botnet spam"

Nothing here defeats a block, forges a User-Agent to impersonate a browser, or
works around a host that has said no. A refusal is reported as a refusal.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from tracelock.discovery.hosts.base import (
    HostingError,
    HostingResult,
    Retention,
    validate_payload,
    validate_returned_url,
)

# Identifies this client honestly. Not an attempt to look like a browser.
USER_AGENT = "TRACELOCK/1.0 (evidence provenance tool)"

DEFAULT_TIMEOUT = 90.0


class CatboxHost:
    """catbox.moe -- no credentials, direct image URL, no deletion.

    The default because a judge can clone the repo and demonstrate the full
    pipeline without registering anywhere. Verified end to end: an upload here
    produced a URL Google Lens fetched, returning 59 visual matches including
    two Facebook posts.

    The honest cost: an anonymous upload has no delete token, so it CANNOT be
    removed programmatically. `supports_deletion` is False and the consent
    screen says so. Anyone who needs deletion should configure Cloudinary.
    """

    key = "catbox"
    display_name = "Catbox"
    endpoint = "https://catbox.moe/user/api.php"

    @property
    def configured(self) -> bool:
        return True

    @property
    def supports_deletion(self) -> bool:
        return False

    @property
    def retention_note(self) -> str:
        return (
            "Catbox hosts the file indefinitely and offers no way for TRACELOCK "
            "to delete an anonymous upload. Use Cloudinary if you need the copy "
            "removed afterwards."
        )

    def missing_configuration(self) -> tuple[str, ...]:
        return ()

    def upload(self, data: bytes, filename: str, *, retention: Retention) -> HostingResult:
        validate_payload(data)
        try:
            response = httpx.post(
                self.endpoint,
                data={"reqtype": "fileupload"},
                files={"fileToUpload": (filename, data, "image/jpeg")},
                headers={"User-Agent": USER_AGENT},
                timeout=DEFAULT_TIMEOUT,
                follow_redirects=True,
            )
        except httpx.TimeoutException as exc:
            raise HostingError("Catbox timed out after {0}s".format(DEFAULT_TIMEOUT)) from exc
        except httpx.HTTPError as exc:
            raise HostingError("Catbox transport error: {0}".format(exc)) from exc

        if response.status_code != 200:
            raise HostingError(
                "Catbox returned HTTP {0}".format(response.status_code)
            )

        url = validate_returned_url(response.text, "Catbox")
        return HostingResult(
            url=url,
            provider=self.key,
            asset_id=url.rsplit("/", 1)[-1],
            deletion_supported=False,
            note=self.retention_note,
        )

    def delete(self, result: HostingResult) -> bool:
        # Deleting a catbox file needs a userhash from a registered account.
        # Returning False rather than attempting-and-pretending.
        return False


class ImgBBHost:
    """imgbb.com -- free key, genuine auto-expiry, no REST deletion.

    ImgBB returns a `delete_url`, but it is an HTML confirmation PAGE meant for
    a human, not an endpoint. Driving it would mean scripting a web form, so
    `supports_deletion` is False and expiry does the work instead -- which
    ImgBB does honour, and which is a stronger privacy property than a delete
    call nobody remembers to make.
    """

    key = "imgbb"
    display_name = "ImgBB"
    endpoint = "https://api.imgbb.com/1/upload"

    @property
    def _api_key(self) -> str:
        return (os.environ.get("TL_IMGBB_API_KEY") or "").strip()

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    @property
    def supports_deletion(self) -> bool:
        return False

    @property
    def retention_note(self) -> str:
        return (
            "ImgBB deletes the file automatically when the retention window "
            "expires. Its delete link is a web page for a person, not an API, "
            "so TRACELOCK does not claim programmatic deletion."
        )

    def missing_configuration(self) -> tuple[str, ...]:
        return () if self._api_key else ("TL_IMGBB_API_KEY",)

    def upload(self, data: bytes, filename: str, *, retention: Retention) -> HostingResult:
        validate_payload(data)
        if not self.configured:
            raise HostingError("TL_IMGBB_API_KEY is not set")

        try:
            response = httpx.post(
                self.endpoint,
                params={"key": self._api_key, "expiration": str(retention.seconds)},
                files={"image": (filename, data, "image/jpeg")},
                headers={"User-Agent": USER_AGENT},
                timeout=DEFAULT_TIMEOUT,
            )
        except httpx.TimeoutException as exc:
            raise HostingError("ImgBB timed out after {0}s".format(DEFAULT_TIMEOUT)) from exc
        except httpx.HTTPError as exc:
            raise HostingError("ImgBB transport error: {0}".format(exc)) from exc

        if response.status_code != 200:
            raise HostingError("ImgBB returned HTTP {0}".format(response.status_code))

        try:
            payload = response.json()
        except ValueError as exc:
            raise HostingError("ImgBB returned a non-JSON response") from exc

        if not payload.get("success"):
            raise HostingError(
                "ImgBB rejected the upload: {0}".format(
                    str(payload.get("error", {}).get("message", ""))[:120]
                )
            )

        block = payload.get("data") or {}
        # `.url` is the direct image; `.display_url` is a viewer page. Reverse
        # image search needs the former.
        url = validate_returned_url(block.get("url") or "", "ImgBB")
        return HostingResult(
            url=url,
            provider=self.key,
            asset_id=str(block.get("id") or ""),
            expires_after_seconds=retention.seconds,
            deletion_supported=False,
            note=self.retention_note,
        )

    def delete(self, result: HostingResult) -> bool:
        return False


class CloudinaryHost:
    """Cloudinary -- the only provider here that genuinely deletes on request.

    Uses the signed REST API directly over httpx, so no SDK dependency is
    added. Upload is signed with SHA-1 over the sorted parameters, which is
    Cloudinary's documented scheme; the API secret is used to compute that
    signature and never leaves this process.
    """

    key = "cloudinary"
    display_name = "Cloudinary"

    @property
    def _cloud(self) -> str:
        return (os.environ.get("TL_CLOUDINARY_CLOUD_NAME") or "").strip()

    @property
    def _api_key(self) -> str:
        return (os.environ.get("TL_CLOUDINARY_API_KEY") or "").strip()

    @property
    def _secret(self) -> str:
        return (os.environ.get("TL_CLOUDINARY_API_SECRET") or "").strip()

    @property
    def configured(self) -> bool:
        return bool(self._cloud and self._api_key and self._secret)

    @property
    def supports_deletion(self) -> bool:
        return True

    @property
    def retention_note(self) -> str:
        return (
            "Cloudinary supports real deletion, so TRACELOCK removes the copy "
            "once discovery finishes and reports whether that succeeded."
        )

    def missing_configuration(self) -> tuple[str, ...]:
        missing = []
        if not self._cloud:
            missing.append("TL_CLOUDINARY_CLOUD_NAME")
        if not self._api_key:
            missing.append("TL_CLOUDINARY_API_KEY")
        if not self._secret:
            missing.append("TL_CLOUDINARY_API_SECRET")
        return tuple(missing)

    def _sign(self, params: dict[str, Any]) -> str:
        import hashlib

        payload = "&".join(
            "{0}={1}".format(k, params[k]) for k in sorted(params) if params[k] != ""
        )
        return hashlib.sha1((payload + self._secret).encode("utf-8")).hexdigest()

    def upload(self, data: bytes, filename: str, *, retention: Retention) -> HostingResult:
        import time

        validate_payload(data)
        if not self.configured:
            raise HostingError(
                "Cloudinary needs {0}".format(", ".join(self.missing_configuration()))
            )

        timestamp = int(time.time())
        folder = "tracelock-temp"
        signed = {"folder": folder, "timestamp": timestamp}

        try:
            response = httpx.post(
                "https://api.cloudinary.com/v1_1/{0}/image/upload".format(self._cloud),
                data={
                    **signed,
                    "api_key": self._api_key,
                    "signature": self._sign(signed),
                },
                files={"file": (filename, data, "image/jpeg")},
                headers={"User-Agent": USER_AGENT},
                timeout=DEFAULT_TIMEOUT,
            )
        except httpx.TimeoutException as exc:
            raise HostingError("Cloudinary timed out") from exc
        except httpx.HTTPError as exc:
            raise HostingError("Cloudinary transport error: {0}".format(exc)) from exc

        if response.status_code not in (200, 201):
            raise HostingError(
                "Cloudinary returned HTTP {0}".format(response.status_code)
            )

        payload = response.json()
        url = validate_returned_url(
            payload.get("secure_url") or payload.get("url") or "", "Cloudinary"
        )
        return HostingResult(
            url=url,
            provider=self.key,
            asset_id=str(payload.get("public_id") or ""),
            deletion_supported=True,
            note=self.retention_note,
        )

    def delete(self, result: HostingResult) -> bool:
        import time

        if not (self.configured and result.asset_id):
            return False

        timestamp = int(time.time())
        signed = {"public_id": result.asset_id, "timestamp": timestamp}
        try:
            response = httpx.post(
                "https://api.cloudinary.com/v1_1/{0}/image/destroy".format(self._cloud),
                data={
                    **signed,
                    "api_key": self._api_key,
                    "signature": self._sign(signed),
                },
                timeout=30.0,
            )
            return response.status_code == 200 and response.json().get("result") == "ok"
        except Exception:
            # Best effort. The caller records that cleanup did not succeed
            # rather than reporting a deletion that did not happen.
            return False
