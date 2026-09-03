"""SerpAPI provider (Google Lens / Yandex Images reverse image search).

Chosen for Phase 0 because it is the only option that is simultaneously:
  - genuinely dynamic (live query, results unknown in advance)
  - legal and documented (an API, not scraping)
  - free at low volume (100 searches/month)
  - stable JSON

PARSER POSTURE
--------------
The `google_lens` response shape (visual_matches[]) is well established.
The `yandex_images` reverse-image shape is less certain.  So parsing is
defensive: try each known results key in turn and, on a miss, raise
ProviderSchemaError carrying the top-level keys that WERE present.  A schema
change should cost two minutes, not an evening -- and it must never be
mistaken for "dynamic discovery is impossible here".
"""

from __future__ import annotations

from typing import Any

import httpx

from tracelock.core.models import Candidate, ProbeRef
from tracelock.discovery.base import (
    ProviderBlockedError,
    ProviderConfigError,
    ProviderResult,
    ProviderSchemaError,
    ProviderTransportError,
    Requirement,
    Stopwatch,
)

SERPAPI_ENDPOINT = "https://serpapi.com/search"

SUPPORTED_ENGINES = ("google_lens", "yandex_images")

# Result arrays to look for, in priority order. Retained as the FALLBACK for
# an engine we have no explicit mapping for, and for schema drift.
RESULT_KEYS: tuple[str, ...] = (
    "visual_matches",
    "image_results",
    "exact_matches",
    "inline_images",
    "organic_results",
)

# ---------------------------------------------------------------------------
# Per-engine result arrays, verified against real captured responses.
#
# google_lens  ships ONE array; merging `organic_results` would pollute it with
#              text results that are not visual matches.
#
# yandex_images ships TWO arrays carrying genuinely DIFFERENT candidate sets,
#              so both are merged rather than first-match-wins:
#
#   image_results (67 observed)  pages that contain the image. Carries a REAL
#                                source page (`link` -> devpost.com, ...) and
#                                the original host's image
#                                (`original_image.link`). Strong provenance.
#
#   similar_images (40 observed) visually similar imagery. `image.link` points
#                                at Yandex's OWN CDN cache
#                                (avatars.mds.yandex.net) and `link` goes back
#                                into Yandex search -- there is no original
#                                source page. Weak provenance, and Phase 3 must
#                                not count 40 of these as 40 independent
#                                domains.
# ---------------------------------------------------------------------------
ENGINE_RESULT_KEYS: dict[str, tuple[str, ...]] = {
    "google_lens": ("visual_matches",),
    "yandex_images": ("image_results", "similar_images"),
}

# Provider key -> Candidate field. Several aliases per field, tried in order,
# so one renamed key does not lose the whole record.
#
# A dotted alias ("original_image.link") walks one level into a nested object.
# Yandex returns media as {"link": ..., "width": ..., "height": ...} objects
# where Google Lens returns a bare string; without the nested path the URL is
# silently dropped, which is exactly the `with usable media: 0` failure.
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "post_url": ("link", "url", "source_url", "page_url"),
    "image_url": (
        "original",                 # google_lens: bare string
        "original_image.link",      # yandex image_results: nested
        "image.link",               # yandex similar_images: nested
        "image",                    # google_lens fallback: bare string
        "original_image_link",
        "image_link",
    ),
    "thumbnail_url": (
        "thumbnail",                # google_lens: bare string
        "thumbnail.link",           # yandex: nested
        "thumbnail_url",
        "image.link",
        "image",
    ),
    "title": ("title", "name", "snippet_title"),
    "text": ("snippet", "description", "text"),
    "source": ("source", "displayed_link", "domain", "site"),
    "author": ("author", "channel", "uploader"),
    "timestamp": ("date", "published_date", "timestamp"),
}

# Substrings that mark a SerpAPI logical error as a quota/billing block
# rather than a schema problem.
_QUOTA_MARKERS = ("run out", "quota", "credit", "limit", "plan")


def _lookup(item: dict[str, Any], path: str) -> Any:
    """Read a key, or a dotted path one level into a nested object.

    Returns None rather than raising on a missing or wrongly-typed node: a
    provider changing a field's shape must lose that field, not the record.
    """
    if "." not in path:
        return item.get(path)

    head, _, tail = path.partition(".")
    nested = item.get(head)
    if not isinstance(nested, dict):
        return None
    return nested.get(tail)


def _first_present(item: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """First alias yielding a usable STRING value.

    The string requirement is load-bearing. Yandex returns
    `{"thumbnail": {"link": ...}}` where Google Lens returns
    `{"thumbnail": "https://..."}`. Accepting the dict would hand a dict to
    Candidate, whose validator correctly discards non-strings -- producing a
    silently empty media URL. Skipping non-strings lets the NEXT alias (the
    dotted path) resolve it instead.
    """
    for key in keys:
        value = _lookup(item, key)
        if isinstance(value, str) and value.strip():
            return value
    return None
    return None


class SerpApiProvider:
    """Live reverse-image search via SerpAPI."""

    name = "serpapi"

    def __init__(
        self,
        api_key: str,
        *,
        engine: str = "google_lens",
        timeout: float = 30.0,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ProviderConfigError(
                "TL_SERPAPI_API_KEY is empty. Get a free key at "
                "https://serpapi.com/manage-api-key and put it in .env"
            )
        if engine not in SUPPORTED_ENGINES:
            raise ProviderConfigError(
                "engine must be one of {0}, got {1!r}".format(SUPPORTED_ENGINES, engine)
            )
        self._api_key = api_key.strip()
        self.engine = engine
        self._timeout = timeout

    @property
    def requires(self) -> frozenset[Requirement]:
        return frozenset({Requirement.PUBLIC_IMAGE_URL})

    def search(self, probe: ProbeRef, *, limit: int = 20) -> ProviderResult:
        if not probe.public_url:
            raise ProviderConfigError("probe.public_url is required")

        params = {
            "engine": self.engine,
            "url": probe.public_url,
            "api_key": self._api_key,
            "no_cache": "true",  # force a live query; never serve a cached hit
        }

        with Stopwatch() as watch:
            try:
                response = httpx.get(
                    SERPAPI_ENDPOINT,
                    params=params,
                    timeout=self._timeout,
                    follow_redirects=True,
                )
            except httpx.TimeoutException as exc:
                raise ProviderTransportError(
                    "SerpAPI timed out after {0}s".format(self._timeout)
                ) from exc
            except httpx.HTTPError as exc:
                raise ProviderTransportError(
                    "SerpAPI transport error: {0}".format(exc)
                ) from exc

            self._raise_for_status(response)

            try:
                payload = response.json()
            except ValueError as exc:
                raise ProviderSchemaError(
                    "SerpAPI returned non-JSON (HTTP {0})".format(response.status_code)
                ) from exc

            # SerpAPI reports logical failures inside a 200 body.
            error = payload.get("error")
            if error:
                lowered = str(error).lower()
                if any(marker in lowered for marker in _QUOTA_MARKERS):
                    raise ProviderBlockedError(
                        "SerpAPI quota or plan limit: {0}".format(error)
                    )
                raise ProviderSchemaError("SerpAPI error: {0}".format(error))

            candidates = self._parse(payload, limit=limit)
            elapsed = watch.elapsed

        # api_key deliberately excluded -- this dict gets written to disk.
        return ProviderResult(
            provider=self.name,
            engine=self.engine,
            candidates=candidates,
            raw_response=payload,
            query={k: v for k, v in params.items() if k != "api_key"},
            elapsed_seconds=elapsed,
        )

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        status = response.status_code
        if status == 200:
            return
        body = response.text[:400]
        # 401 is a CREDENTIAL problem -> setup error, not a viability verdict.
        # 403/429 mean the key is recognised but we are being refused -> block.
        # Conflating these would report "architecture risk" for a typo'd key.
        if status == 401:
            raise ProviderConfigError(
                "SerpAPI rejected the key as invalid (HTTP 401): {0}".format(body)
            )
        if status == 403:
            raise ProviderBlockedError(
                "SerpAPI forbade the request (HTTP 403): {0}".format(body)
            )
        if status == 429:
            raise ProviderBlockedError(
                "SerpAPI rate limited (HTTP 429): {0}".format(body)
            )
        if status >= 500:
            raise ProviderTransportError(
                "SerpAPI server error (HTTP {0}): {1}".format(status, body)
            )
        raise ProviderSchemaError(
            "Unexpected HTTP {0} from SerpAPI: {1}".format(status, body)
        )

    def _result_groups(self, payload: dict[str, Any]) -> list[tuple[str, list]]:
        """Every populated result array for this engine, in declared order.

        Engine-specific keys are tried first. If none are present we fall back
        to the generic RESULT_KEYS scan, which keeps the parser tolerant of
        schema drift and of engines we have not mapped explicitly.
        """
        groups: list[tuple[str, list]] = []

        for key in ENGINE_RESULT_KEYS.get(self.engine, ()):
            value = payload.get(key)
            if isinstance(value, list) and value:
                groups.append((key, [e for e in value if isinstance(e, dict)]))

        if groups:
            return groups

        for key in RESULT_KEYS:
            value = payload.get(key)
            if isinstance(value, list) and value:
                return [(key, [e for e in value if isinstance(e, dict)])]

        return []

    def _parse(self, payload: dict[str, Any], *, limit: int) -> list[Candidate]:
        """Normalize every result array this engine returned. Invents nothing.

        Multiple arrays are MERGED rather than first-match-wins, because for
        yandex_images they hold different candidate sets (see
        ENGINE_RESULT_KEYS). Each candidate records which array it came from
        so Phase 3 can weight real source pages above CDN-cached thumbnails.
        """
        groups = self._result_groups(payload)

        if not groups:
            # Distinguish "the engine genuinely found nothing" from "we cannot
            # read the schema". Both matter; conflating them misleads the gate.
            known = set(RESULT_KEYS) | set(ENGINE_RESULT_KEYS.get(self.engine, ()))
            if any(key in payload for key in known):
                return []
            raise ProviderSchemaError(
                "No known results key in SerpAPI response for engine "
                "{0!r}. Looked for: {1}".format(
                    self.engine, ", ".join(sorted(known))
                ),
                observed_keys=sorted(payload.keys()),
            )

        candidates: list[Candidate] = []
        seen: set[str] = set()
        rank = 0

        for group_name, items in groups:
            for item in items:
                if len(candidates) >= limit:
                    return candidates

                rank += 1
                fields = {
                    field: _first_present(item, aliases)
                    for field, aliases in FIELD_ALIASES.items()
                }
                candidate = Candidate(
                    provider=self.name,
                    rank=item.get("position", rank),
                    result_group=group_name,
                    raw_metadata=item,
                    **fields,
                )

                # Drop rows with nothing addressable. Not filtering for
                # quality -- only for records that cannot be acted on at all.
                if not (
                    candidate.post_url
                    or candidate.image_url
                    or candidate.thumbnail_url
                ):
                    continue

                # The two Yandex arrays overlap; dedupe on normalized identity
                # so a result present in both is not counted twice.
                key = candidate.dedup_key
                if key in seen:
                    continue
                seen.add(key)

                candidates.append(candidate)

        return candidates
