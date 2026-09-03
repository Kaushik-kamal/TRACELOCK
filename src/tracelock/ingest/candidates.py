"""Extract and rank every image candidate on a public page.

A webpage rarely holds one image. It holds a preview image, a hero, a byline
portrait, a sprite sheet, three social icons and a tracking pixel. Taking the
first one, or only `og:image`, throws away the picture the operator was
actually looking at.

So: collect them all, score them, and hand back a ranked list. The runner takes
the top one automatically; the UI offers the rest so a person can override when
the ranking guesses wrong. Ranking is a CONVENIENCE, never a claim -- the
scores below predict "is this the subject of the page", not "is this a face"
and certainly not "is this the same person". Only the face engine answers the
first, and only calibrated verification answers the second.

Everything here is pure and offline: it parses HTML that has already been
fetched. No candidate is downloaded during ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlsplit

from tracelock.ingest.classify import looks_like_image_path

# Below this, an image cannot hold a usable face -- SCRFD needs roughly 50px of
# face, which needs a meaningfully larger frame. Applied only to DECLARED
# dimensions; an undeclared image is kept and judged on other signals.
MIN_USEFUL_DIMENSION = 100

MAX_CANDIDATES = 12

# Formats the pipeline can never decode, so ranking one is guaranteed-to-fail
# work. SVG is vector and ICO is a multi-resolution icon container; the face
# engine needs a raster frame, and `validation.SUPPORTED_FORMATS` lists neither.
#
# Found on iana.org, whose only in-page image is an SVG logo: it was offered as
# the top candidate, fetched, then rejected with "the link returned a web page,
# not an image" -- which was also untrue. It returned an SVG.
UNDECODABLE_EXTENSIONS = (".svg", ".svgz", ".ico", ".cur", ".pdf")

# URL/alt fragments that suggest page furniture rather than content.
_NEGATIVE = (
    ("sprite", -60), ("favicon", -60), ("tracking", -60), ("pixel.gif", -60),
    ("spacer", -60), ("1x1", -60), ("blank.", -50), ("placeholder", -45),
    ("logo", -40), ("icon", -35), ("badge", -30), ("button", -30),
    ("banner", -25), ("advert", -40), ("/ads/", -40), ("emoji", -35),
    ("watermark", -25), ("arrow", -30), ("chevron", -30), ("bullet", -25),
)

# Fragments that suggest a photograph of a person.
_POSITIVE = (
    ("avatar", 45), ("profile", 40), ("portrait", 40), ("headshot", 45),
    ("photo", 25), ("selfie", 35), ("face", 30), ("people", 20),
    ("hero", 20), ("featured", 20), ("main", 15), ("large", 12),
    ("original", 12), ("full", 10), ("/media/", 10), ("upload", 8),
)


@dataclass(frozen=True, slots=True)
class ImageCandidate:
    """One image found on a page, with why it ranked where it did."""

    url: str
    score: int
    source: str                       # og:image | twitter:image | schema.org | img
    alt: str = ""
    width: int | None = None
    height: int | None = None
    signals: tuple[str, ...] = field(default_factory=tuple)

    @property
    def label(self) -> str:
        if self.alt:
            return self.alt[:80]
        name = urlsplit(self.url).path.rsplit("/", 1)[-1]
        return name[:80] or "image"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "url": self.url,
            "score": self.score,
            "source": self.source,
            "label": self.label,
            "signals": list(self.signals),
        }
        if self.alt:
            payload["alt"] = self.alt
        if self.width and self.height:
            payload["width"] = self.width
            payload["height"] = self.height
        return payload


class _ImageHarvester(HTMLParser):
    """Collect preview metadata AND in-page <img> tags in one pass.

    Deliberately tolerant: malformed markup is normal on the open web, and a
    missing attribute must never raise.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.images: list[dict[str, str]] = []
        self.title: str = ""
        self._in_title = False
        self._order = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        attributes = {k.lower(): (v or "") for k, v in attrs}

        if tag == "meta":
            key = (attributes.get("property") or attributes.get("name") or "").lower()
            content = attributes.get("content", "").strip()
            if key and content and key not in self.meta:
                self.meta[key] = content
            return

        if tag == "link":
            if "image_src" in (attributes.get("rel") or "").lower():
                href = attributes.get("href", "").strip()
                if href:
                    self.meta.setdefault("link:image_src", href)
            return

        if tag == "title":
            self._in_title = True
            return

        if tag in ("img", "source"):
            source = (
                attributes.get("src")
                or attributes.get("data-src")
                or attributes.get("data-original")
                or _widest_from_srcset(attributes.get("srcset", ""))
                or ""
            ).strip()
            if source:
                self._order += 1
                self.images.append({
                    "src": source,
                    "alt": attributes.get("alt", "").strip(),
                    "width": attributes.get("width", ""),
                    "height": attributes.get("height", ""),
                    "order": str(self._order),
                })

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title and not self.title:
            self.title = data.strip()[:200]


def _widest_from_srcset(srcset: str) -> str:
    """Pick the highest-resolution entry from a srcset.

    Bigger is better here: more pixels on the face means a better embedding.
    """
    best_url, best_width = "", -1
    for entry in srcset.split(","):
        parts = entry.strip().split()
        if not parts:
            continue
        url = parts[0]
        width = -1
        if len(parts) > 1 and parts[1].endswith("w"):
            try:
                width = int(parts[1][:-1])
            except ValueError:
                width = -1
        if width > best_width:
            best_url, best_width = url, width
    return best_url


def _as_int(value: str) -> int | None:
    try:
        return int(str(value).strip().rstrip("px"))
    except (ValueError, AttributeError):
        return None


def _score_url_text(text: str) -> tuple[int, list[str]]:
    """Score the URL + alt text against the hint tables."""
    lowered = text.lower()
    score = 0
    signals: list[str] = []

    for fragment, weight in _NEGATIVE:
        if fragment in lowered:
            score += weight
            signals.append("looks like {0}".format(fragment.strip("/.")))

    for fragment, weight in _POSITIVE:
        if fragment in lowered:
            score += weight
            signals.append("named '{0}'".format(fragment.strip("/.")))

    return score, signals


def rank_candidates(html: str, base_url: str) -> tuple[list[ImageCandidate], str]:
    """Return (ranked candidates, page title). Highest score first.

    Deterministic: equal scores break on discovery order, so the same page
    always produces the same ranking. A demo that reorders itself between runs
    looks broken even when it is correct.
    """
    parser = _ImageHarvester()
    try:
        parser.feed(html)
    except Exception:  # malformed markup must never propagate
        pass

    found: dict[str, ImageCandidate] = {}

    def offer(
        raw_url: str, base_score: int, source: str,
        *, alt: str = "", width: int | None = None, height: int | None = None,
        order: int = 0,
    ) -> None:
        if not raw_url or raw_url.startswith("data:"):
            return

        absolute = urljoin(base_url, raw_url.strip())
        if urlsplit(absolute).scheme not in ("http", "https"):
            return

        # A format we cannot decode is not a candidate, however well it scores.
        path = urlsplit(absolute).path.lower()
        if path.endswith(UNDECODABLE_EXTENSIONS):
            return

        # A declared tiny image cannot carry a usable face.
        if width and height and (width < MIN_USEFUL_DIMENSION or height < MIN_USEFUL_DIMENSION):
            return

        text_score, signals = _score_url_text("{0} {1}".format(absolute, alt))
        score = base_score + text_score

        # Later in the document is likelier to be furniture.
        score -= min(order, 20)

        if width and height:
            pixels = width * height
            if pixels >= 640 * 480:
                score += 25
                signals.append("large ({0}x{1})".format(width, height))
            elif pixels >= 300 * 300:
                score += 12
        if looks_like_image_path(absolute):
            score += 8

        existing = found.get(absolute)
        if existing and existing.score >= score:
            return

        found[absolute] = ImageCandidate(
            url=absolute, score=score, source=source, alt=alt,
            width=width, height=height, signals=tuple(dict.fromkeys(signals)),
        )

    # Preview metadata first: the page's own statement of what it is about.
    declared_width = _as_int(parser.meta.get("og:image:width", ""))
    declared_height = _as_int(parser.meta.get("og:image:height", ""))

    for key, base_score, source in (
        ("og:image:secure_url", 100, "og:image"),
        ("og:image:url", 100, "og:image"),
        ("og:image", 100, "og:image"),
        ("twitter:image:src", 90, "twitter:image"),
        ("twitter:image", 90, "twitter:image"),
        ("link:image_src", 70, "link:image_src"),
    ):
        value = parser.meta.get(key)
        if value:
            offer(
                value, base_score, source,
                width=declared_width, height=declared_height,
            )

    # Then in-page images, which often include the one actually being viewed.
    for image in parser.images:
        offer(
            image["src"], 40, "img",
            alt=image["alt"],
            width=_as_int(image["width"]),
            height=_as_int(image["height"]),
            order=_as_int(image["order"]) or 0,
        )

    ranked = sorted(found.values(), key=lambda c: (-c.score, c.url))
    return ranked[:MAX_CANDIDATES], parser.title
