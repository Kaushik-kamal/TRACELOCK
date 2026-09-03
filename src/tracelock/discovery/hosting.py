"""Temporary public hosting for a probe image.

WHY THIS MODULE EXISTS
----------------------
No production reverse-image API accepts a local file upload; they all fetch a
URL you supply.  So "search by local image" has an unstated prerequisite:
the image must be publicly reachable first.

That prerequisite is a SEPARATE failure mode from the search itself, and the
viability gate has to be able to tell them apart -- "the host rejected us" and
"the search engine found nothing" demand completely different responses.

PRIVACY POSTURE
---------------
Uploading a face image to an anonymous third-party host is a real biometric
disclosure.  Three deliberate constraints:

  1. Never automatic.  The caller must pass --allow-upload.
  2. Shortest sensible retention (1h default); the host deletes it after.
  3. The preferred path skips this module entirely -- pass --image-url for an
     image that is already public.  Fewer moving parts AND no new disclosure.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

import httpx

LITTERBOX_ENDPOINT = "https://litterbox.catbox.moe/resources/internals/api.php"

VALID_RETENTIONS = ("1h", "12h", "24h", "72h")

# Refuse to upload anything implausible as a probe photo.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class HostingError(Exception):
    """Upload failed. Distinct from DiscoveryError: a different pipeline stage."""


def upload_temporary(
    image_path: Path,
    *,
    retention: str = "1h",
    timeout: float = 60.0,
) -> str:
    """Upload an image to a temporary host and return its public URL.

    Raises HostingError on any failure. The caller is responsible for having
    obtained explicit consent before calling this.
    """
    if retention not in VALID_RETENTIONS:
        raise HostingError(
            "retention must be one of {0}, got {1!r}".format(VALID_RETENTIONS, retention)
        )
    if not image_path.is_file():
        raise HostingError("not a file: {0}".format(image_path))

    size = image_path.stat().st_size
    if size == 0:
        raise HostingError("refusing to upload an empty file")
    if size > MAX_UPLOAD_BYTES:
        raise HostingError(
            "file is {0} bytes, over the {1} byte cap".format(size, MAX_UPLOAD_BYTES)
        )

    try:
        with image_path.open("rb") as handle:
            response = httpx.post(
                LITTERBOX_ENDPOINT,
                data={"reqtype": "fileupload", "time": retention},
                files={"fileToUpload": (image_path.name, handle)},
                timeout=timeout,
                follow_redirects=True,
            )
    except httpx.TimeoutException as exc:
        raise HostingError("image host timed out after {0}s".format(timeout)) from exc
    except httpx.HTTPError as exc:
        raise HostingError("image host transport error: {0}".format(exc)) from exc

    if response.status_code != 200:
        raise HostingError(
            "image host returned HTTP {0}: {1}".format(
                response.status_code, response.text[:200]
            )
        )

    url = response.text.strip()
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise HostingError(
            "image host did not return a URL. Body was: {0!r}".format(url[:200])
        )

    return url
