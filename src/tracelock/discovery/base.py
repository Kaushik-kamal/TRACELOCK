"""Search provider protocol and error taxonomy.

The error types are load-bearing.  The viability gate must distinguish
"we are not set up correctly" from "this environment cannot do dynamic
discovery" -- treating a missing API key as an architecture risk would be a
false negative on the most important decision in the project.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from tracelock.core.models import Candidate, ProbeRef


class DiscoveryError(Exception):
    """Base for all discovery failures."""


class ProviderConfigError(DiscoveryError):
    """Setup problem: missing key, bad engine name, unmet requirement.

    NOT a viability failure. The gate exits 2 and says so.
    """


class ProviderBlockedError(DiscoveryError):
    """The provider refused us: 401/403/429, CAPTCHA, quota exhausted.

    This IS a viability signal -- it is the architecture risk we are testing for.
    """


class ProviderTransportError(DiscoveryError):
    """Network-level failure: DNS, TLS, timeout, connection reset."""


class ProviderSchemaError(DiscoveryError):
    """We reached the provider and got 200, but could not find results.

    Carries the observed top-level keys so schema drift is a two-minute fix
    rather than a dead end.
    """

    def __init__(self, message: str, observed_keys: list[str] | None = None) -> None:
        super().__init__(message)
        self.observed_keys = observed_keys or []


class Requirement(str, Enum):
    """What a provider needs from a ProbeRef in order to run."""

    PUBLIC_IMAGE_URL = "public_image_url"  # Google Lens, Yandex, TinEye
    LOCAL_IMAGE = "local_image"            # future: on-device index
    TEXT_QUERY = "text_query"              # future: Bluesky, Mastodon


@dataclass(slots=True)
class ProviderResult:
    """Everything one provider call produced.

    The raw response is retained deliberately: the gate must persist it, and a
    later phase hashes the discovery provenance into the evidence bundle.
    """

    provider: str
    engine: str
    candidates: list[Candidate]
    raw_response: dict[str, Any]
    query: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    @property
    def count(self) -> int:
        return len(self.candidates)

    @property
    def with_media(self) -> list[Candidate]:
        return [c for c in self.candidates if c.has_media]


@runtime_checkable
class SearchProvider(Protocol):
    """The contract every provider implements.

    Adding Yandex, Bing, TinEye or Bluesky later means writing one class that
    satisfies this and registering it. Nothing downstream changes.
    """

    name: str
    engine: str
    requires: frozenset[Requirement]

    def search(self, probe: ProbeRef, *, limit: int) -> ProviderResult:
        """Execute one live search. Raises a DiscoveryError subclass on failure.

        MUST NOT return fabricated or cached results. An empty candidate list
        is a valid, honest outcome and must be reported as such.
        """
        ...


def check_requirements(provider: SearchProvider, probe: ProbeRef) -> None:
    """Raise ProviderConfigError if the probe lacks what the provider needs."""
    if Requirement.PUBLIC_IMAGE_URL in provider.requires and not probe.public_url:
        raise ProviderConfigError(
            f"{provider.name} requires a publicly reachable image URL. "
            "Pass --image-url, or --allow-upload to host it temporarily."
        )
    if Requirement.TEXT_QUERY in provider.requires and not probe.hint_text:
        raise ProviderConfigError(f"{provider.name} requires --hint-text.")


class Stopwatch:
    """Minimal elapsed-time helper so providers report honest timings."""

    __slots__ = ("_start",)

    def __enter__(self) -> "Stopwatch":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc: object) -> None:
        pass

    @property
    def elapsed(self) -> float:
        return round(time.perf_counter() - self._start, 3)
