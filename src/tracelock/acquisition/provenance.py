"""Source-domain provenance.

Phase 1 carried forward a known defect: `core.models.registrable_host()`
approximates eTLD+1 by stripping a `www.` prefix, which is wrong for
multi-label public suffixes (`bbc.co.uk`, `user.github.io`).

This module supplies the correct version, backed by the Public Suffix List.
Phase 0's helper is deliberately left alone -- it is a display-field fallback
whose name honestly says *host*, and its contract has not changed. Scoring uses
this module instead.

Phase 2 only MEASURES provenance. Corroboration scoring is Phase 3.

OFFLINE BY DEFAULT
------------------
`tldextract` will fetch a fresh Public Suffix List over the network on first
use unless told not to. That would make the test suite depend on the internet
and make results vary with when they were run. We pin it to the bundled
snapshot via `suffix_list_urls=()` and record the mode in the output.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from urllib.parse import urlsplit

import tldextract

# ---------------------------------------------------------------------------
# PRIVATE PSL SECTION -- a real trade-off, defaulted to the safe direction.
#
# The PSL has a "private" section listing hosting platforms (github.io,
# blogspot.com, ...). Whether to honour it changes what counts as one source:
#
#   include_private=False (DEFAULT)
#       user-a.github.io and user-b.github.io  ->  both "github.io"
#       UNDER-counts distinct publishers.
#
#   include_private=True
#       user-a.github.io and user-b.github.io  ->  two separate domains
#       Counts platform tenants as independent, which is gameable: an
#       adversary can register many free subdomains cheaply.
#
# Phase 3 multiplies trust by the number of independent domains, so
# over-counting inflates a trust score while under-counting only makes us
# more conservative. We default to the direction that cannot inflate trust
# and leave the decision explicit for Phase 3.
# ---------------------------------------------------------------------------
DEFAULT_INCLUDE_PRIVATE_SUFFIXES = False


@lru_cache(maxsize=4)
def _extractor(include_private: bool) -> tldextract.TLDExtract:
    return tldextract.TLDExtract(
        suffix_list_urls=(),  # offline: bundled snapshot only
        include_psl_private_domains=include_private,
    )


@dataclass(frozen=True, slots=True)
class SourceProvenance:
    """Where a candidate came from, decomposed for Phase 3."""

    url: str
    scheme: str
    host: str
    subdomain: str
    domain: str
    suffix: str
    registrable_domain: str
    include_private_suffixes: bool
    psl_source: str = "bundled-snapshot"

    @property
    def is_resolvable(self) -> bool:
        """False for IP literals and malformed hosts, where eTLD+1 is undefined."""
        return bool(self.registrable_domain)

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "scheme": self.scheme,
            "host": self.host,
            "subdomain": self.subdomain,
            "domain": self.domain,
            "suffix": self.suffix,
            "registrable_domain": self.registrable_domain,
            "include_private_suffixes": self.include_private_suffixes,
            "psl_source": self.psl_source,
        }


def describe_source(
    url: str, *, include_private: bool = DEFAULT_INCLUDE_PRIVATE_SUFFIXES
) -> SourceProvenance:
    """Decompose a URL into its provenance parts. Never raises on bad input."""
    parts = urlsplit(url or "")
    host = (parts.hostname or "").lower()

    extracted = _extractor(include_private)(host)
    suffix = extracted.suffix
    domain = extracted.domain

    # `top_domain_under_public_suffix` replaced the deprecated
    # `registered_domain` in tldextract 5.x; fall back for older installs.
    registrable = getattr(extracted, "top_domain_under_public_suffix", None)
    if registrable is None:
        registrable = getattr(extracted, "registered_domain", "")

    return SourceProvenance(
        url=url or "",
        scheme=parts.scheme,
        host=host,
        subdomain=extracted.subdomain,
        domain=domain,
        suffix=suffix,
        registrable_domain=registrable or "",
        include_private_suffixes=include_private,
    )


def registrable_domain(
    url: str, *, include_private: bool = DEFAULT_INCLUDE_PRIVATE_SUFFIXES
) -> str:
    """eTLD+1 for a URL, or "" when undefined (IP literal, malformed host).

    This is the function Phase 3 corroboration must use. Do NOT substitute
    `core.models.registrable_host()`, which is a display-only approximation.
    """
    return describe_source(url, include_private=include_private).registrable_domain


def distinct_registrable_domains(
    urls: list[str], *, include_private: bool = DEFAULT_INCLUDE_PRIVATE_SUFFIXES
) -> set[str]:
    """Distinct eTLD+1 values across URLs, ignoring the undefined ones.

    Provided for Phase 3. Phase 2 records provenance but does not score it.
    """
    found = set()
    for url in urls:
        domain = registrable_domain(url, include_private=include_private)
        if domain:
            found.add(domain)
    return found
