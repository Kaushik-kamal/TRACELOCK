"""Which host to use, and what to tell the operator about it.

Selection is server-side configuration (`TL_IMAGE_HOST`), never a client
parameter -- the same rule the chain layer follows, and for the same reason:
letting a browser choose where a face image gets published is not a decision
that belongs in a browser.
"""

from __future__ import annotations

import os
from typing import Any

from tracelock.discovery.hosts.base import HostingError, ImageHostProvider
from tracelock.discovery.hosts.providers import CatboxHost, CloudinaryHost, ImgBBHost

# Catbox first: it is the only one that works with no configuration at all, so
# a fresh clone can demonstrate the whole pipeline. The others are opt-in.
PROVIDERS: dict[str, ImageHostProvider] = {
    provider.key: provider
    for provider in (CatboxHost(), ImgBBHost(), CloudinaryHost())
}

DEFAULT_PROVIDER = "catbox"


def get_provider(key: str) -> ImageHostProvider:
    if key not in PROVIDERS:
        raise HostingError(
            "unknown image host {0!r}; known: {1}".format(key, ", ".join(sorted(PROVIDERS)))
        )
    return PROVIDERS[key]


def resolve_provider(preferred: str | None = None) -> ImageHostProvider:
    """The provider that will actually be used.

    Falls back to the default when the configured one cannot run, so a missing
    ImgBB key degrades to a working catbox upload rather than to a dead end --
    and the consent screen names whichever one is actually selected.
    """
    from tracelock.chain.config import _load_dotenv

    _load_dotenv()
    key = (preferred or os.environ.get("TL_IMAGE_HOST") or DEFAULT_PROVIDER).strip()

    if key not in PROVIDERS:
        return PROVIDERS[DEFAULT_PROVIDER]

    provider = PROVIDERS[key]
    if provider.configured:
        return provider

    fallback = PROVIDERS[DEFAULT_PROVIDER]
    return fallback if fallback.configured else provider


def available_providers() -> list[dict[str, Any]]:
    """Every provider with its real capabilities, for the consent screen.

    `configured` and `supports_deletion` are read from the provider itself, so
    the screen cannot promise something the code will not do.
    """
    rows = []
    for provider in PROVIDERS.values():
        rows.append({
            "key": provider.key,
            "display_name": provider.display_name,
            "configured": provider.configured,
            "supports_deletion": provider.supports_deletion,
            "retention_note": provider.retention_note,
            "missing": list(provider.missing_configuration()),
        })
    return rows
