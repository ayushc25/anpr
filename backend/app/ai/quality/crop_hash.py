"""A cheap perceptual hash for plate crops, used to tell frames apart.

Two questions in the cascade need "are these two crops actually different?",
and neither can afford a real image comparison in the hot path:

  * Top-K retention — keeping the five best observations by quality alone
    happily fills the set with five consecutive frames of a stopped vehicle,
    which is one observation stored five times.
  * Evidence weighting — the validator treats every read as an independent
    observation. Four near-identical frames are not four independent looks at
    a plate; they are one look, sampled four times, and they carry nowhere
    near four frames' worth of information.

A 64-bit difference hash answers it for a few microseconds: downscale to
9x8 grayscale, compare each pixel with its right-hand neighbour, pack the
comparisons into a bitmask. Robust to brightness and mild noise (it encodes
gradient direction, not intensity), sensitive to the vehicle actually moving
or the crop actually changing — which is exactly the distinction wanted.

Deliberately NOT a similarity metric for deciding whether two plates are the
same vehicle. That is what plate text and ``postprocess.same_vehicle`` are
for. This only ever answers "did the picture change".
"""
from __future__ import annotations

import cv2
import numpy as np

#: dHash grid. 9x8 comparisons give 8x8 = 64 bits.
_WIDTH, _HEIGHT = 9, 8

#: Hamming distance at or above which two crops count as different views.
#:
#: Out of 64 bits. Empirically, consecutive frames of a stationary vehicle sit
#: at 0-3 (sensor noise flipping a few gradient comparisons), while a vehicle
#: that has moved enough to change the plate's scale or angle sits well above
#: 8. Six is inside that gap and biased toward calling frames SIMILAR, because
#: the cost of wrongly discarding a diverse frame (less evidence) is higher
#: than the cost of wrongly keeping a redundant one (a little wasted slot).
DEFAULT_MIN_DISTANCE = 6


def crop_hash(crop: np.ndarray) -> int:
    """64-bit dHash of a plate crop. 0 for an unusable crop."""
    if crop is None or crop.size == 0:
        return 0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    small = cv2.resize(gray, (_WIDTH, _HEIGHT), interpolation=cv2.INTER_AREA)
    # Gradient sign between horizontally adjacent pixels.
    bits = small[:, 1:] > small[:, :-1]
    return _pack(bits)


def _pack(bits: np.ndarray) -> int:
    """Pack 64 booleans into an int. Two numpy calls beat a Python loop and
    this runs once per plate observation."""
    packed = np.packbits(bits.reshape(-1).astype(np.uint8))
    return int.from_bytes(packed.tobytes(), "big")


def hamming(a: int, b: int) -> int:
    """Bits that differ. 64 is the maximum."""
    return int(bin(a ^ b).count("1"))


def is_distinct(a: int, b: int, min_distance: int = DEFAULT_MIN_DISTANCE) -> bool:
    """Whether two crops are different enough to count as separate views.

    A zero hash means "unknown" (an unusable crop), and unknown is treated as
    distinct: refusing to store or count evidence on the strength of a hash we
    could not compute would be the wrong way to be wrong.
    """
    if a == 0 or b == 0:
        return True
    return hamming(a, b) >= min_distance
