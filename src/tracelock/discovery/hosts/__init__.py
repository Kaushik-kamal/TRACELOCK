"""Temporary public hosting, as a pluggable provider.

WHY THIS EXISTS
---------------
No production reverse-image API accepts a file upload; every one of them
fetches a URL you supply. So "search by local image" has an unstated
prerequisite: the image must be publicly reachable first. Without this bridge,
an uploaded photo or a webcam frame can never enter public discovery at all.

Verified end to end before this package was written: a local file uploaded to
catbox produced a direct image URL that Google Lens fetched, returning 59
visual matches including two Facebook posts. The bridge is real, not assumed.

PRIVACY POSTURE -- UNCHANGED FROM THE ORIGINAL MODULE
-----------------------------------------------------
Uploading a face image to a third-party host is a genuine biometric
disclosure, and this package does not soften that:

  * Never automatic. The caller must pass explicit consent, and the API layer
    refuses the request without it.
  * The preferred path still skips hosting entirely -- an image that is
    already public needs no upload and makes no new disclosure.
  * What each provider can and cannot do is DECLARED, not assumed. A provider
    that cannot delete says so, and the UI repeats it. Claiming deletion we
    cannot perform would be worse than not offering it.
"""

from tracelock.discovery.hosts.base import (
    HostingError,
    HostingResult,
    ImageHostProvider,
    Retention,
)
from tracelock.discovery.hosts.registry import (
    PROVIDERS,
    available_providers,
    get_provider,
    resolve_provider,
)

__all__ = [
    "PROVIDERS",
    "HostingError",
    "HostingResult",
    "ImageHostProvider",
    "Retention",
    "available_providers",
    "get_provider",
    "resolve_provider",
]
