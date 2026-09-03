"""Universal input ingestion.

Every source -- upload, webcam, direct image URL, webpage, social post, cloud
share link, demo example -- is classified here, resolved here, and leaves here
as the same normalized thing. Downstream code has exactly one shape to handle.
"""

from tracelock.ingest.adapters import (
    ADAPTERS,
    PlatformAdapter,
    Support,
    adapter_for,
    capability_matrix,
)
from tracelock.ingest.candidates import ImageCandidate, rank_candidates
from tracelock.ingest.classify import (
    Classification,
    InputType,
    Platform,
    classify_url,
    looks_like_image_path,
    platform_for_host,
    refine_with_response,
)
from tracelock.ingest.resolve import Method, ResolvedInput, resolve

__all__ = [
    "ADAPTERS",
    "Classification",
    "ImageCandidate",
    "InputType",
    "Method",
    "Platform",
    "PlatformAdapter",
    "ResolvedInput",
    "Support",
    "adapter_for",
    "capability_matrix",
    "classify_url",
    "looks_like_image_path",
    "platform_for_host",
    "rank_candidates",
    "refine_with_response",
    "resolve",
]
