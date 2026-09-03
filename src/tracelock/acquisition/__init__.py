"""Secure acquisition: re-downloading and validating untrusted candidate media.

Provider URLs are hostile input. Nothing here trusts a filename, a
Content-Type, or a provider's claim about what a URL contains.
"""

from tracelock.acquisition.cas import (
    CasEntry,
    ContentAddressedStore,
    SourceReference,
)
from tracelock.acquisition.fetcher import (
    AcquisitionResult,
    FetchPolicy,
    fetch_media,
    is_blocked_target,
)
from tracelock.acquisition.provenance import (
    SourceProvenance,
    describe_source,
    distinct_registrable_domains,
    registrable_domain,
)
from tracelock.acquisition.validation import (
    ImageValidation,
    looks_like_html,
    sniff_format,
    validate_image_bytes,
)

__all__ = [
    "AcquisitionResult", "FetchPolicy", "fetch_media", "is_blocked_target",
    "ImageValidation", "validate_image_bytes", "sniff_format", "looks_like_html",
    "ContentAddressedStore", "CasEntry", "SourceReference",
    "SourceProvenance", "describe_source", "registrable_domain",
    "distinct_registrable_domains",
]
