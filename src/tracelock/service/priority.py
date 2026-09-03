"""Candidate ordering from signals we genuinely have.

Ordering decides which candidates get the expensive 1.73s embedding first. It
does NOT decide what any of them concludes -- a candidate that is verified last
gets exactly the verdict it would have got first, and reordering can never
change a similarity, a probability, or a trust score.

WHAT IS AND IS NOT A SIGNAL
---------------------------
Every factor below is something measured or reported, not guessed:

  discovery rank        the position the index returned it at -- the engine's
                        own relevance, used as a hint, never as evidence
  multi-engine agreement two independent indexes surfaced the same URL
  full image vs thumb   thumbnails are small and produce worse embeddings
  declared dimensions   when the provider reports them
  publisher novelty     a domain we have not verified yet

There is deliberately no "source reliability" table. Ranking sites by assumed
trustworthiness would be a fabricated number dressed as a measurement, and it
would quietly bias which evidence gets found.

WHY PUBLISHER NOVELTY MATTERS MOST
----------------------------------
Corroboration counts INDEPENDENT publishers. Three candidates from three
domains reach the evidence threshold; three from one domain do not, however
good each is. So preferring an unseen domain is not a heuristic about quality --
it is the ordering that reaches a defensible conclusion in the fewest
embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

# Weights are ordering-only. They are not thresholds, they never enter a score
# a person sees, and changing them changes speed, never a verdict.
W_RANK = 30.0
W_MULTI_ENGINE = 25.0
W_FULL_IMAGE = 20.0
W_DIMENSIONS = 15.0
W_NEW_PUBLISHER = 40.0

# The requirement asks specifically for a social-media post, so a social
# candidate is examined before a news article carrying the same photograph.
# This changes ORDER only: a news result is still verified, still counted, and
# still reported -- it simply is not the thing being looked for first.
W_SOCIAL = 35.0


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    """A candidate with its ordering score and the reasons behind it."""

    candidate: Any
    index: int
    score: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "priority": round(self.score, 2),
            "signals": list(self.reasons),
        }


def _publisher(candidate) -> str:
    from tracelock.acquisition.provenance import describe_source

    url = candidate.post_url or candidate.image_url or candidate.thumbnail_url or ""
    if not url:
        return ""
    try:
        source = describe_source(url)
    except Exception:
        return ""

    # Same key StopTracker counts by, including its host fallback. If these
    # two disagreed, prioritisation would optimise for a notion of "publisher"
    # that the stopping rule does not actually use -- and an unusual TLD or an
    # IP-literal host would silently lose the strongest ordering signal.
    return source.registrable_domain or source.host or ""


def score_candidates(
    candidates: Sequence[Any],
    *,
    found_by: dict[str, list[str]] | None = None,
    seen_publishers: set[str] | None = None,
) -> list[ScoredCandidate]:
    """Order candidates by how quickly they could produce defensible evidence.

    Stable: equal scores keep discovery order, so the same inputs always
    produce the same processing order. A forensic tool that shuffles its own
    work between identical runs cannot be reasoned about.
    """
    found_by = found_by or {}
    already = set(seen_publishers or ())
    total = max(1, len(candidates))
    scored: list[ScoredCandidate] = []

    # Publishers claimed by earlier (higher-priority) candidates in THIS pass,
    # so we do not hand the top three slots to three copies from one domain.
    claimed: set[str] = set()

    for index, candidate in enumerate(candidates):
        score = 0.0
        reasons: list[str] = []

        # 1. The index's own ranking, normalised. A hint, not evidence.
        rank_score = W_RANK * (1.0 - index / total)
        score += rank_score
        if index < 5:
            reasons.append("ranked #{0} by the search index".format(index + 1))

        # 2. Two independent indexes surfacing the same URL is a real signal.
        engines = found_by.get(_key(candidate), [])
        if len(engines) > 1:
            score += W_MULTI_ENGINE
            reasons.append("found by {0}".format(" + ".join(engines)))

        # 3. A full image beats a thumbnail: more pixels on the face.
        if candidate.image_url:
            score += W_FULL_IMAGE
        else:
            reasons.append("thumbnail only")

        # 4. Declared dimensions, when the provider reports them. Candidate
        #    has no dimension fields, so this reads the verbatim provider
        #    payload -- present for some engines, absent for others, and simply
        #    inert when absent rather than substituted with a guess.
        width, height = _declared_dimensions(candidate)
        if width and height:
            if width >= 400 and height >= 400:
                score += W_DIMENSIONS
                reasons.append("{0}x{1}".format(width, height))
            elif width < 150 or height < 150:
                score -= W_DIMENSIONS
                reasons.append("small ({0}x{1})".format(width, height))

        # 5. A genuine social platform is what the requirement asks for.
        from tracelock.ingest.social import classify_source

        source_url = (
            candidate.post_url or candidate.image_url or candidate.thumbnail_url
        )
        classification = classify_source(source_url)
        if classification.is_social:
            score += W_SOCIAL
            reasons.append("{0} post".format(classification.platform))

        # 6. An unseen publisher is what corroboration is actually made of.
        publisher = _publisher(candidate)
        if publisher and publisher not in already and publisher not in claimed:
            score += W_NEW_PUBLISHER
            claimed.add(publisher)
            reasons.append("new publisher {0}".format(publisher))
        elif publisher:
            reasons.append("publisher already seen")

        scored.append(ScoredCandidate(
            candidate=candidate, index=index, score=score, reasons=tuple(reasons),
        ))

    # Descending score, then original order -- deterministic for equal scores.
    return sorted(scored, key=lambda s: (-s.score, s.index))


def _declared_dimensions(candidate) -> tuple[int | None, int | None]:
    """Dimensions from the raw provider payload, if it reported any."""
    raw = getattr(candidate, "raw_metadata", None) or {}

    for width_key, height_key in (
        ("original_width", "original_height"),
        ("image_width", "image_height"),
        ("width", "height"),
    ):
        width, height = raw.get(width_key), raw.get(height_key)
        if isinstance(width, int) and isinstance(height, int):
            return width, height

    thumbnail = raw.get("thumbnail")
    if isinstance(thumbnail, dict):
        width, height = thumbnail.get("width"), thumbnail.get("height")
        if isinstance(width, int) and isinstance(height, int):
            return width, height

    return None, None


def _key(candidate) -> str:
    from tracelock.discovery.multi import _canonical_url

    return (
        _canonical_url(candidate.image_url or candidate.thumbnail_url)
        or _canonical_url(candidate.post_url)
    )


def prioritise(
    candidates: Sequence[Any],
    *,
    found_by: dict[str, list[str]] | None = None,
    seen_publishers: set[str] | None = None,
) -> list[Any]:
    """Just the reordered candidates, for callers that do not need the scores."""
    return [s.candidate for s in score_candidates(
        candidates, found_by=found_by, seen_publishers=seen_publishers
    )]
