"""Perceptual hashing (DCT pHash).

Implemented here rather than pulled from `imagehash` for the same reason the
ROC code is hand-written: in a forensic system the definition of a measurement
should be readable in the repository, not delegated to a library the reviewer
has to go and check. It also avoids a dependency for ~30 lines of arithmetic.

THE DEFINITION (pin this -- changing it invalidates stored hashes)
-----------------------------------------------------------------
  1. decode to greyscale
  2. resize to 32x32, INTER_AREA
  3. 2-D DCT-II
  4. keep the top-left 8x8 low-frequency block
  5. drop the DC term (element [0,0]) when computing the median
  6. bit = 1 where coefficient > median
  7. 64 bits, packed MSB-first, rendered as 16 hex characters

Dropping the DC term matters: it carries overall brightness, so including it
would make the hash sensitive to exposure changes it should ignore.

WHAT pHash IS FOR HERE
----------------------
Distinguishing "the same photograph, republished" from "a different photograph
of the same person". Those two carry very different evidentiary weight and must
never be collapsed into one number -- see `relation.py`.

FAILURE MODES (state them; a metric you cannot fault is one you cannot trust)
  - Not rotation invariant. A 90-degree rotation produces an unrelated hash.
  - Not flip invariant. Mirrored images read as unrelated.
  - Weak on low-detail images. Flat or near-uniform pictures push most
    coefficients toward the median, so unrelated flat images can collide.
  - Heavy cropping changes the frequency layout and reads as unrelated, even
    though the crop is derived from the original.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

HASH_BITS = 64
DCT_SIZE = 32
LOW_FREQ_SIZE = 8

# Hamming distance thresholds over 64 bits. WORKING DEFAULTS, chosen from the
# widely used pHash convention (<=10 similar, >=24 unrelated) and deliberately
# leaving a gap where the answer is "we cannot tell". Not calibrated here.
NEAR_DUPLICATE_MAX_DISTANCE = 10
UNRELATED_MIN_DISTANCE = 24


@dataclass(frozen=True, slots=True)
class PerceptualHash:
    """A 64-bit pHash."""

    bits: int
    hex_digest: str
    algorithm: str = "phash-dct-8x8/1"

    def distance(self, other: "PerceptualHash") -> int:
        """Hamming distance in bits, 0..64."""
        return int(bin(self.bits ^ other.bits).count("1"))

    def similarity(self, other: "PerceptualHash") -> float:
        """Distance mapped to [0, 1], where 1.0 means identical hashes."""
        return 1.0 - (self.distance(other) / HASH_BITS)

    def to_dict(self) -> dict[str, Any]:
        return {"algorithm": self.algorithm, "hex": self.hex_digest}

    def __str__(self) -> str:
        return self.hex_digest


def compute_phash(image: np.ndarray) -> PerceptualHash:
    """Compute the pHash of a BGR or greyscale array.

    Takes a decoded array, not bytes: the caller already decoded for validation
    and face analysis, and re-decoding would be wasted work.
    """
    import cv2

    if image is None or image.size == 0:
        raise ValueError("cannot hash an empty image")

    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(grey, (DCT_SIZE, DCT_SIZE), interpolation=cv2.INTER_AREA)
    coefficients = cv2.dct(resized.astype(np.float32))
    block = coefficients[:LOW_FREQ_SIZE, :LOW_FREQ_SIZE]

    # Exclude the DC term from the median: it encodes overall brightness, and
    # including it would make the hash track exposure rather than structure.
    without_dc = block.flatten()[1:]
    median = float(np.median(without_dc))

    bits = 0
    for value in block.flatten():
        bits = (bits << 1) | (1 if float(value) > median else 0)

    return PerceptualHash(bits=bits, hex_digest="{0:016x}".format(bits))


def phash_from_bytes(data: bytes) -> PerceptualHash:
    """Decode bytes and hash them. Convenience for tests and one-off use."""
    import cv2

    array = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if array is None:
        raise ValueError("bytes could not be decoded as an image")
    return compute_phash(array)


def hamming_distance(left: PerceptualHash, right: PerceptualHash) -> int:
    return left.distance(right)


def is_near_duplicate(
    left: PerceptualHash,
    right: PerceptualHash,
    *,
    max_distance: int = NEAR_DUPLICATE_MAX_DISTANCE,
) -> bool:
    """Visual near-duplicate under a PROVISIONAL distance threshold.

    Visual only. Says nothing about identity, and must never be read as one.
    """
    return left.distance(right) <= max_distance
