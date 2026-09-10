"""Indian number-plate grammar, normalization and confusion repair.

Two jobs, and it is worth being precise about the difference:

*Validation* answers "is this string shaped like a real Indian plate?" — used
to weight a read down before it votes.

*Repair* answers "is there a single-character OCR confusion that turns this
invalid string into a valid one?" — applied only at positions where the format
is unambiguous about alpha-vs-digit, and only when it actually fixes the
string. It never rewrites a plate that already parses, because doing so is how
a system silently invents plates that were never on the road.

*Trimming* answers the same question for the characters the recognizer read
from outside the number: the hologram, the emblem, the plate border. Same
guard — it only ever acts on a string that is already invalid, and only when
exactly one answer comes out valid.

``resolve`` is the two of them in the order a caller wants them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

ALPHA = "A"
DIGIT = "N"

#: State / UT codes as of the current RTO series. A two-letter prefix outside
#: this set is a strong reject signal and cheaply catches a whole class of
#: misreads (``UP`` read as ``VP``, ``DL`` as ``OL``).
STATE_CODES = frozenset(
    """AN AP AR AS BR CG CH DD DL DN GA GJ HP HR JH JK KA KL LA LD MH ML MN MP
       MZ NL OD OR PB PY RJ SK TN TR TS UK UP UA WB""".split()
)

#: OCR confusions, split by direction. Only these substitutions are ever
#: applied by repair, and only where the grammar demands the other type.
#:
#: Each entry lists candidates in descending likelihood. Keep the two
#: directions mirror images of each other: a one-way entry silently disables
#: repair in the other direction, which is how a 7 misread for the T in
#: UP23AT0097 went uncorrected while T->7 was handled.
DIGIT_TO_ALPHA = {
    "0": ("O", "D", "Q"),
    "1": ("I", "L", "J"),
    "2": ("Z",),
    "4": ("A",),
    "5": ("S",),
    "6": ("G",),
    "7": ("T",),
    "8": ("B",),
    "9": ("G", "Q"),
}
ALPHA_TO_DIGIT = {
    "O": ("0",), "D": ("0",), "Q": ("0",),
    "I": ("1",), "L": ("1",), "J": ("1",),
    "Z": ("2",), "A": ("4",), "S": ("5",),
    "G": ("6",), "T": ("7",), "B": ("8",),
}

#: Confusions BETWEEN characters of the same type. The grammar cannot catch
#: these — an alpha misread as another alpha still satisfies the format mask —
#: so ``repair`` never uses them. ``confusable`` does, because registry
#: snapping has a much stronger constraint available: the result must be a
#: plate the society has actually registered.
SAME_TYPE_CONFUSIONS: frozenset[frozenset[str]] = frozenset(
    frozenset(pair) for pair in (
        "BD", "B8", "DO", "D0", "O0", "OQ", "OD", "IL", "I1", "L1",
        "S5", "Z2", "G6", "GC", "EF", "MN", "UV", "VY", "68", "96", "35", "57", "89",
    )
)

#: Characters an Indian plate never contains, dropped during normalization.
_STRIP = re.compile(r"[^A-Z0-9]")

#: Prefixes some recognizers emit from the "IND" hologram strip.
_LEADING_NOISE = re.compile(r"^(IND|1ND|IN|LND)(?=[A-Z]{2}\d)")


@dataclass(frozen=True)
class PlateFormat:
    """One accepted plate shape, as a regex plus a positional type mask."""

    name: str
    pattern: re.Pattern
    mask: str  # same length as the match: 'A' alpha, 'N' digit
    example: str


def _fmt(name: str, regex: str, mask: str, example: str) -> PlateFormat:
    return PlateFormat(name, re.compile(regex + r"$"), mask, example)


#: Every civilian series still on the road puts at least this many characters
#: on the plate: two for the state, one or two for the district, up to three
#: for the series, and three or four for the number. A shorter string is a
#: fragment of a plate, not a plate — unless it validates outright, which the
#: few legitimately short older series do.
MIN_COMPLETE_LEN = 8

#: The shortest shape ``_standard_formats`` may generate.
#:
#: Seven, not eight, and the one character between them is the whole point.
#: Seven-character no-series registrations are real — ``DL81234`` is a genuine
#: old Delhi plate (state + single-digit district + four-digit number) and
#: rejecting it would trade one wrong answer for another. Six-character ones
#: are not: no registration on any Indian road is two letters, one digit and
#: three digits.
#:
#: That single shape (``std_d1_s0_n3``) was the hole. ``is_fragment`` declares
#: anything under ``MIN_COMPLETE_LEN`` a fragment *unless it validates*, and
#: this format made ``UP1606`` validate — a textbook truncated read, produced
#: whenever the vehicle in front occludes half the plate. It matched, passed
#: ``is_valid``, scored a grammar factor of 1.0, and could be emitted as a
#: CONFIRMED plate at 0.95 while the recognizer's own 94% was merely honest
#: about the six glyphs it could see.
MIN_GENERATED_LEN = 7


def _standard_formats() -> list[PlateFormat]:
    """The civilian format is STATE + DISTRICT + SERIES + NUMBER, where the
    district is one or two digits (Delhi's DL8C is a real single-digit
    district, not a misread) and the series is zero to three letters.

    Generating the combinations beats hand-listing them: the hand-written list
    is exactly how DL8CAF5010 came to be rejected as invalid.

    Combinations shorter than ``MIN_GENERATED_LEN`` are NOT generated; see
    that constant for why the cut is at seven characters and what went wrong
    when there was no cut at all. Dropping the six-character shape also makes
    ``repair`` and ``trim_noise`` strictly more conservative: neither can now
    "fix" or trim a string into a six-character plate, because there is no
    longer such a thing.
    """
    formats: list[PlateFormat] = []
    for district in (2, 1):  # two-digit districts are far more common
        for series in (2, 3, 1, 0):
            for number in (4, 3):
                if 2 + district + series + number < MIN_GENERATED_LEN:
                    continue
                mask = ALPHA * 2 + DIGIT * district + ALPHA * series + DIGIT * number
                regex = rf"[A-Z]{{2}}\d{{{district}}}" + (rf"[A-Z]{{{series}}}" if series else "") + rf"\d{{{number}}}"
                formats.append(
                    _fmt(f"std_d{district}_s{series}_n{number}", regex, mask, "UP32AB1234")
                )
    return formats


#: Ordered most-specific first; the first match wins.
FORMATS: tuple[PlateFormat, ...] = (
    _fmt("bh_series", r"\d{2}BH\d{4}[A-Z]{2}", "NNAANNNNAA", "21BH2345AA"),
    _fmt("military", r"\d{2}[A-Z]\d{6}[A-Z]", "NNANNNNNNA", "09B123456X"),
    *_standard_formats(),
)

#: A plate shorter or longer than this is not a plate.
MIN_LEN, MAX_LEN = 6, 11


def normalize(text: str) -> str:
    """Uppercase, strip separators and known hologram noise.

    Deliberately conservative: it removes characters that cannot be part of a
    plate, and nothing else. Character substitution belongs in ``repair``,
    where it is grammar-guided and logged.
    """
    if not text:
        return ""
    cleaned = _STRIP.sub("", text.upper())
    cleaned = _LEADING_NOISE.sub("", cleaned)
    return cleaned


def match_format(plate: str) -> Optional[PlateFormat]:
    for fmt in FORMATS:
        if fmt.pattern.match(plate):
            return fmt
    return None


def is_valid(plate: str) -> bool:
    """Full validity: a known shape AND a real state code where one applies."""
    fmt = match_format(plate)
    if fmt is None:
        return False
    if fmt.mask.startswith("AA") and plate[:2] not in STATE_CODES:
        return False
    return True


def is_fragment(plate: str) -> bool:
    """True when the read is too short to be a whole registration.

    This is the signature of a plate the camera only half saw: occluded by the
    vehicle in front, cut by the frame edge, or clipped by a plate box that
    stopped at the obstruction. The recognizer reports what it saw with full
    confidence — it has no way to know the characters continue past the
    occlusion — so confidence alone never catches this. Length does.

    The ``is_valid`` escape hatch stays, because the seven-character
    no-series series really do exist. What it no longer admits is a
    SIX-character string: ``_standard_formats`` stops at
    ``MIN_GENERATED_LEN``, so ``UP1606`` can no longer validate its way out of
    being called a fragment.
    """
    if not plate:
        return True
    return len(plate) < MIN_COMPLETE_LEN and not is_valid(plate)


def grammar_factor(plate: str) -> float:
    """The ``g`` term in the read weight.

    A structurally valid plate votes at full strength; a plate that is the
    right length and roughly the right shape but fails validation votes at
    0.6; anything else votes at 0.35. These are not thresholds — a low factor
    still votes, it just needs more agreement to win.
    """
    if not plate:
        return 0.0
    if is_valid(plate):
        return 1.0
    if is_fragment(plate):
        # Below the "not even plate-ish" floor on purpose. A fragment is not a
        # noisy read of the whole plate that more agreement could rescue — it
        # is a confident, clean read of a part of one, and every extra frame
        # of the same occlusion agrees with it. Only real evidence of the
        # whole plate should be able to outvote it.
        return 0.25
    if match_format(plate) is not None:
        return 0.75  # right shape, unknown state code
    if MIN_LEN <= len(plate) <= MAX_LEN and _looks_plate_ish(plate):
        return 0.6
    return 0.35


def _looks_plate_ish(plate: str) -> bool:
    """Two leading letters and at least three digits overall — the coarse
    shape every Indian civilian format shares."""
    return bool(re.match(r"^[A-Z]{2}", plate)) and sum(c.isdigit() for c in plate) >= 3


def _candidates_for(char: str, want: str) -> Iterable[str]:
    """Substitutions that could turn ``char`` into the required type, most
    likely first."""
    if want == ALPHA and char.isdigit():
        yield from DIGIT_TO_ALPHA.get(char, ())
    elif want == DIGIT and char.isalpha():
        yield from ALPHA_TO_DIGIT.get(char, ())


@dataclass
class RepairResult:
    text: str
    corrections: list[str] = field(default_factory=list)
    repaired: bool = False


def repair(plate: str, max_fixes: int = 2) -> RepairResult:
    """Try to turn an invalid plate into a valid one using only confusion-map
    substitutions at positions where the format demands the other character
    type.

    Returns the input unchanged when it is already valid, when no format is
    close enough to guide the repair, or when more than one format would
    accept different repairs (ambiguous — better to stay wrong-but-honest than
    to guess).
    """
    if not plate:
        return RepairResult("", [], False)
    if is_valid(plate):
        return RepairResult(plate, [], False)

    solutions: list[tuple[str, list[str]]] = []
    for fmt in FORMATS:
        if len(fmt.mask) != len(plate):
            continue
        chars = list(plate)
        fixes: list[str] = []
        ok = True
        for i, want in enumerate(fmt.mask):
            char = chars[i]
            correct = (want == ALPHA and char.isalpha()) or (want == DIGIT and char.isdigit())
            if correct:
                continue
            alt = next(iter(_candidates_for(char, want)), None)
            if alt is None or len(fixes) >= max_fixes:
                ok = False
                break
            fixes.append(f"pos{i}: {char}->{alt}")
            chars[i] = alt
        if not ok:
            continue
        candidate = "".join(chars)
        if is_valid(candidate):
            solutions.append((candidate, fixes))

    if not solutions:
        return RepairResult(plate, [], False)

    # Prefer the repair that needed the fewest substitutions; a tie between
    # two different results is ambiguous, so change nothing.
    solutions.sort(key=lambda s: len(s[1]))
    if len(solutions) > 1 and len(solutions[0][1]) == len(solutions[1][1]) and solutions[0][0] != solutions[1][0]:
        return RepairResult(plate, [], False)

    text, fixes = solutions[0]
    return RepairResult(text, fixes, True)


#: How many characters ``trim_noise`` may drop. The hologram strip, the state
#: emblem and the plate's own border are all to the LEFT of the number on an
#: Indian plate, and a bolt head or the frame edge occasionally adds one more
#: glyph on the right — so the leading budget is larger than the trailing one.
MAX_TRIM_LEAD = 2
MAX_TRIM_TAIL = 1


def _trim_options() -> Iterable[tuple[int, int]]:
    """(lead, tail) pairs, fewest characters dropped first."""
    pairs = [
        (lead, tail)
        for lead in range(MAX_TRIM_LEAD + 1)
        for tail in range(MAX_TRIM_TAIL + 1)
        if lead or tail
    ]
    pairs.sort(key=lambda lt: (lt[0] + lt[1], lt[1]))  # prefer trimming the left
    return pairs


def trim_noise(plate: str, max_fixes: int = 2) -> RepairResult:
    """Drop leading/trailing glyphs that are not part of the plate.

    The recognizer reads whatever is inside the plate crop, and the crop
    frequently includes a sliver of what sits beside the number: the IND
    hologram, the state emblem, a bolt, the raised border. That shows up as an
    extra character glued to the front — TUP1GEJ0364 for UP1GEJ0364.

    ``normalize`` only removes the few whole tokens we can name (``IND``); it
    cannot know that a stray ``T`` is not part of the plate. This can, but only
    under a constraint strong enough to make it safe:

    * the input must NOT already be a valid plate — a plate that parses is
      never trimmed, so this can only ever act on a string that is wrong;
    * the trimmed remainder must be FULLY valid, state code included, on its
      own or after ``repair``;
    * exactly one trim of the minimum size may produce a valid plate. Two
      different valid answers means we cannot tell which characters were noise,
      and inventing one is worse than reporting the misread.
    """
    if not plate or is_valid(plate):
        return RepairResult(plate, [], False)

    solutions: list[tuple[int, str, list[str]]] = []
    best_cost: Optional[int] = None
    for lead, tail in _trim_options():
        cost = lead + tail
        if best_cost is not None and cost > best_cost:
            break  # a cheaper trim already worked; deeper ones cannot win
        candidate = plate[lead: len(plate) - tail if tail else None]
        if len(candidate) < MIN_LEN:
            continue
        dropped = []
        if lead:
            dropped.append(f"leading {plate[:lead]!r}")
        if tail:
            dropped.append(f"trailing {plate[len(plate) - tail:]!r}")
        fixes = ["trim: dropped " + " and ".join(dropped)]
        if is_valid(candidate):
            solutions.append((cost, candidate, fixes))
            best_cost = cost
            continue
        repaired = repair(candidate, max_fixes=max_fixes)
        if repaired.repaired:
            solutions.append((cost, repaired.text, fixes + repaired.corrections))
            best_cost = cost

    if not solutions:
        return RepairResult(plate, [], False)
    if len({text for _, text, _ in solutions}) > 1:
        return RepairResult(plate, [], False)  # ambiguous — leave it alone
    _, text, fixes = solutions[0]
    return RepairResult(text, fixes, True)


def resolve(plate: str, max_fixes: int = 2) -> RepairResult:
    """The full cleanup a read goes through before it votes.

    ``repair`` first — a same-length confusion fix is a smaller claim than
    deleting a character — then ``trim_noise`` for the reads repair cannot
    reach because the string is the wrong length to begin with.
    """
    if not plate or is_valid(plate):
        return RepairResult(plate, [], False)
    repaired = repair(plate, max_fixes=max_fixes)
    if repaired.repaired:
        return repaired
    return trim_noise(plate, max_fixes=max_fixes)


def confusable(a: str, b: str) -> bool:
    """True when ``a`` and ``b`` differ at exactly one position and that
    difference is a known OCR confusion — cross-type (8/B) or same-type (B/D).

    Used for registry snapping, which must never fire on an arbitrary
    edit-distance-1 neighbour: UP32AB1234 and UP32AX1234 are one edit apart but
    B and X do not look alike, so treating them as the same plate would invent
    a vehicle.
    """
    if len(a) != len(b):
        return False
    diffs = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    if len(diffs) != 1:
        return False
    x, y = a[diffs[0]], b[diffs[0]]
    if y in DIGIT_TO_ALPHA.get(x, ()) or y in ALPHA_TO_DIGIT.get(x, ()):
        return True
    if x in DIGIT_TO_ALPHA.get(y, ()) or x in ALPHA_TO_DIGIT.get(y, ()):
        return True
    return frozenset((x, y)) in SAME_TYPE_CONFUSIONS


# --- registry-constrained matching ------------------------------------------
#
# Free recognition has to pick the right string out of 36^10 possibilities.
# Matching against a society's registry is a far easier problem: the answer is
# one of a few hundred plates that are known in advance. That prior is strong
# enough to recover a read the recognizer got wrong in two or three places —
# but only if the scoring knows WHICH errors a recognizer actually makes, and
# only if it refuses to answer when two registered plates fit equally well.

#: Substituting a character for one it is genuinely confusable with (8 for B,
#: 0 for O) is weak evidence of a different vehicle. Substituting it for an
#: unrelated one is strong evidence.
COST_CONFUSABLE_SUB = 0.4
COST_SUB = 1.0
#: A dropped or inserted character — the hologram glyph, a character lost to
#: glare. Costed just under a substitution: it is common, but a length
#: difference is still a real difference.
COST_INDEL = 0.9

#: The most total cost a match may carry. Two unrelated substitutions, or five
#: confusions, or two missing characters.
MAX_MATCH_COST = 2.0
#: How much worse the runner-up must be before the winner is trusted. This,
#: not the cost ceiling, is what makes registry matching safe: in a closed set
#: the question is never "is this close to a plate" but "is this closer to ONE
#: plate than to any other".
MIN_MATCH_MARGIN = 0.8
#: Beyond this length difference the strings are not the same plate, whatever
#: the alignment cost says.
MAX_LENGTH_DELTA = 2


def _sub_cost(x: str, y: str) -> float:
    if x == y:
        return 0.0
    return COST_CONFUSABLE_SUB if confusable(x, y) else COST_SUB


def registry_cost(read: str, registered: str) -> float:
    """Weighted edit distance between a noisy read and a registered plate.

    Plain Levenshtein treats ``UP32A81234`` -> ``UP32AB1234`` (a textbook 8/B
    misread) and ``UP32AX1234`` -> ``UP32AB1234`` (a different vehicle) as
    the same distance of 1. Weighting the substitution by whether the two
    characters actually look alike is what separates them.
    """
    if read == registered:
        return 0.0
    if not read or not registered:
        return float(len(read or registered)) * COST_INDEL

    previous = [j * COST_INDEL for j in range(len(registered) + 1)]
    for i, a in enumerate(read, 1):
        current = [i * COST_INDEL]
        for j, b in enumerate(registered, 1):
            current.append(min(
                previous[j] + COST_INDEL,          # deletion
                current[j - 1] + COST_INDEL,       # insertion
                previous[j - 1] + _sub_cost(a, b),  # substitution
            ))
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class RegistryMatch:
    plate: str
    cost: float
    runner_up_cost: float

    @property
    def margin(self) -> float:
        return self.runner_up_cost - self.cost


def best_registry_match(
    read: str,
    registered: Iterable[str],
    max_cost: float = MAX_MATCH_COST,
    min_margin: float = MIN_MATCH_MARGIN,
) -> Optional[RegistryMatch]:
    """The one registered plate this read is closest to, or None.

    None when nothing is close enough, and — importantly — also when two
    registered plates are near-equally close. A society with both UP32AB1234
    and UP32AB1284 on its list gets no answer for a read that sits between
    them, which is correct: the system does not know which resident arrived,
    and guessing puts the wrong name on the gate log.
    """
    if not read:
        return None

    best: Optional[tuple[str, float]] = None
    runner_up = float("inf")
    for plate in registered:
        if not plate or abs(len(plate) - len(read)) > MAX_LENGTH_DELTA:
            continue
        if plate == read:
            return RegistryMatch(plate, 0.0, runner_up)
        cost = registry_cost(read, plate)
        if best is None or cost < best[1]:
            if best is not None:
                runner_up = best[1]
            best = (plate, cost)
        elif cost < runner_up:
            runner_up = cost

    if best is None or best[1] > max_cost:
        return None
    if runner_up - best[1] < min_margin:
        return None
    return RegistryMatch(best[0], best[1], runner_up)


def format_display(plate: str) -> str:
    """Group a plate the way it is painted, for the UI: UP32AB1234 ->
    'UP 32 AB 1234'. Storage always keeps the compact form."""
    fmt = match_format(plate)
    if fmt is None:
        return plate
    # Split on each alpha<->digit transition in the mask.
    groups, start = [], 0
    for i in range(1, len(fmt.mask)):
        if fmt.mask[i] != fmt.mask[i - 1]:
            groups.append(plate[start:i])
            start = i
    groups.append(plate[start:])
    return " ".join(groups)


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance. Small strings only — plates are <= 11 chars."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(
                previous[j] + 1,          # deletion
                current[j - 1] + 1,       # insertion
                previous[j - 1] + (ca != cb),  # substitution
            ))
        previous = current
    return previous[-1]


def similarity(a: str, b: str) -> float:
    """1.0 identical, 0.0 nothing in common."""
    if not a and not b:
        return 1.0
    longest = max(len(a), len(b))
    if longest == 0:
        return 0.0
    return 1.0 - edit_distance(a, b) / longest


#: Two reads of the SAME vehicle, seconds apart on one camera, rarely agree
#: exactly when the recognizer is weak: TUP14F49450 and TP1449450 are one car.
#: Exact-string de-duplication cannot see that and emits a row for each, which
#: is what fills the events table with repeats. Matching on similarity within a
#: short window on a single camera is a much better model of "same vehicle".
SAME_VEHICLE_SIMILARITY = 0.65


def same_vehicle(a: str, b: str, threshold: float = SAME_VEHICLE_SIMILARITY) -> bool:
    """Whether two plate reads plausibly describe the same vehicle.

    Only ever apply this within a short time window on ONE camera. Across
    cameras or across hours it would merge genuinely different vehicles.
    """
    if not a or not b:
        return False
    if a == b:
        return True
    # A long common prefix or suffix is strong evidence: misreads usually
    # corrupt or insert characters rather than shuffle the whole plate.
    if len(a) >= 5 and len(b) >= 5 and (a[:5] == b[:5] or a[-5:] == b[-5:]):
        return True
    return similarity(a, b) >= threshold
