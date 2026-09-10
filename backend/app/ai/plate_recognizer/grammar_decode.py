"""Grammar-constrained re-decoding, driven by the recognizer's own evidence.

The problem this solves, from a real gate misread:

    visible plate   DL7CP8161
    OCR output      CLZCP0161

Three errors, and they need three different mechanisms:

    pos 0  D -> C   both letters, mask satisfied. Only the STATE CODE knows
                    ``CL`` is not a real prefix.
    pos 2  7 -> Z   the mask demands a DIGIT here and ``Z`` is not one. The
                    grammar can see the error precisely.
    pos 5  8 -> 0   both digits, mask satisfied, prefix fine. Nothing here can
                    help; only agreement across frames can.

Position 2 is the interesting one, because the existing repair machinery
cannot fix it. ``repair`` substitutes from a hand-written confusion map, and
``ALPHA_TO_DIGIT['Z']`` is ``('2',)`` — so a map-driven fix yields ``DL2CP...``
which is confidently wrong. The answer, ``7``, is not in any map and never
will be: the map encodes which glyph pairs a human thinks look alike, not what
this model actually confused on this crop.

The model knows. Its distribution at that timestep has ``7`` somewhere below
``Z``, and until now ``ctc_greedy_decode`` discarded it. So:

    the grammar says WHICH positions are wrong
    the recognizer says WHAT they should be instead

Neither alone is enough, and this module is only the join between them. It is
strictly narrower than ``repair``:

  * it acts ONLY where the winner's TYPE is illegal under a candidate mask, so
    a confident legal character is never touched — the one exception being the
    state-code pass below, which is also type-preserving;
  * the replacement must come from the model's own top-k at that position, not
    from a map;
  * the replacement must clear an absolute probability floor AND a ratio
    against the character it displaces, so a 0.001 tail candidate cannot
    rewrite a plate;
  * the result must fully validate, or nothing changes.

When it declines, the read is left exactly as the recognizer produced it and
the existing ``resolve`` path runs unchanged. Declining is the common case and
is meant to be: an unresolvable plate must stay UNRESOLVED rather than become
a confident guess.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

from . import postprocess

logger = logging.getLogger("anpr.ai.grammar_decode")

DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# --- defaults ---------------------------------------------------------------
#
#: Smallest probability an alternative may have and still be allowed to
#: replace a character. A plate crop that produces a legitimate second choice
#: puts real mass on it; below a tenth the candidate is distribution tail and
#: substituting from it is guessing with extra steps.
DEFAULT_MIN_ALTERNATIVE_PROB = 0.10

#: How strong the alternative must be relative to the character it displaces.
#: 0.25 means "at least a quarter as likely as the illegal winner". Note the
#: winner here is ALREADY known to be impossible, so this is not a fair
#: contest — it is a sanity floor to stop a near-zero candidate winning by
#: default when the model was simply lost at that position.
DEFAULT_MIN_ALTERNATIVE_RATIO = 0.25

#: The most positions this may change. Two matches ``repair``'s budget. A read
#: needing three evidence-backed type fixes is not a misread of a plate, it is
#: a bad crop, and it should stay UNRESOLVED.
DEFAULT_MAX_SUBSTITUTIONS = 2


@dataclass
class DecodeResult:
    text: str
    corrections: list[str] = field(default_factory=list)
    changed: bool = False
    #: Set when a candidate was found but rejected by the guards, so the
    #: audit trail can show that a correction was considered and refused.
    declined: str = ""


@dataclass(frozen=True)
class GrammarDecodeConfig:
    enabled: bool = True
    min_alternative_prob: float = DEFAULT_MIN_ALTERNATIVE_PROB
    min_alternative_ratio: float = DEFAULT_MIN_ALTERNATIVE_RATIO
    max_substitutions: int = DEFAULT_MAX_SUBSTITUTIONS
    #: Constrain the two leading characters to a real state code, using the
    #: model's alternatives. Type-preserving (letter for letter).
    state_code_constraint: bool = True


def _allowed_for(mask_char: str) -> str:
    if mask_char == postprocess.ALPHA:
        return LETTERS
    if mask_char == postprocess.DIGIT:
        return DIGITS
    return LETTERS + DIGITS


def _fits(char: str, mask_char: str) -> bool:
    return char in _allowed_for(mask_char)


def decode(
    text: str,
    alternatives: Sequence[Sequence[tuple[str, float]]],
    per_char_confidence: Sequence[float],
    cfg: GrammarDecodeConfig | None = None,
) -> DecodeResult:
    """Try to turn an invalid read into a valid one using model evidence.

    ``alternatives[i]`` is the recognizer's runner-up list at position i, best
    first. Recognizers that cannot produce one pass an empty sequence, and
    this returns the input unchanged — which is what every recognizer except
    the CTC ones does today.
    """
    cfg = cfg or GrammarDecodeConfig()
    if not cfg.enabled or not text:
        return DecodeResult(text)
    if postprocess.is_valid(text):
        # Never rewrite a plate that already parses. Same rule as `repair`,
        # and for the same reason: that is how a system invents plates.
        return DecodeResult(text)
    if not any(alternatives):
        return DecodeResult(text)

    # The two constraints must COMPOSE, because the misread that motivated
    # this module needs both: CLZCP0161 has an illegal type at position 2 AND
    # a prefix that is not a real state code. Running them independently fixes
    # neither — the type pass yields CL7CP8161 which still fails validation,
    # and the state pass cannot even look at CLZCP8161 because the illegal Z
    # means no format matches, so there is no mask to tell it where the state
    # code ends.
    #
    # So: fix types first to get a SHAPE, then fix the state code within that
    # shape, sharing one substitution budget across both.
    solutions: list[tuple[int, str, list[str]]] = []
    declined = ""

    for used, shaped, fixes in _shape_candidates(text, alternatives, per_char_confidence, cfg):
        if postprocess.is_valid(shaped):
            solutions.append((used, shaped, fixes))
            continue
        if not cfg.state_code_constraint:
            continue
        remaining = cfg.max_substitutions - used
        if remaining <= 0:
            continue
        state = _fix_state_code(shaped, alternatives, per_char_confidence, cfg, remaining)
        if state.changed:
            solutions.append((used + len(state.corrections), state.text, fixes + state.corrections))
        elif state.declined and not declined:
            declined = state.declined

    if not solutions:
        return DecodeResult(
            text,
            declined=declined or _rejection_note(text, alternatives, per_char_confidence, cfg),
        )

    solutions.sort(key=lambda s: s[0])
    cheapest = solutions[0][0]
    distinct = {c for used, c, _ in solutions if used == cheapest}
    if len(distinct) > 1:
        # Two different plates fit the evidence equally cheaply. Choosing one
        # would be a guess dressed up as a decode.
        return DecodeResult(text, declined=f"ambiguous: {sorted(distinct)} all fit the evidence")

    _, candidate, fixes = solutions[0]
    if candidate == text:
        return DecodeResult(text, declined=declined)
    return DecodeResult(candidate, fixes, changed=True)


def _acceptable(prob: float, displaced_prob: float, cfg: GrammarDecodeConfig) -> bool:
    if prob < cfg.min_alternative_prob:
        return False
    if displaced_prob > 0 and (prob / displaced_prob) < cfg.min_alternative_ratio:
        return False
    return True


def _shape_candidates(
    text: str,
    alternatives: Sequence[Sequence[tuple[str, float]]],
    per_char_confidence: Sequence[float],
    cfg: GrammarDecodeConfig,
) -> list[tuple[int, str, list[str]]]:
    """Every (substitutions, text, fixes) whose SHAPE matches a known format.

    Shape only — the state code is not checked here. That separation is what
    lets the state-code pass run afterwards on a string that now has a mask to
    interpret.

    Includes the zero-substitution case when the input already matches a
    format, so a read whose only fault is an unknown prefix reaches the state
    pass without being modified first.

    A format is abandoned the moment it needs a substitution the evidence will
    not support: no alternative of the required type, or one that fails the
    probability guards.
    """
    out: list[tuple[int, str, list[str]]] = []
    if postprocess.match_format(text) is not None:
        out.append((0, text, []))

    for fmt in postprocess.FORMATS:
        if len(fmt.mask) != len(text):
            continue
        chars = list(text)
        fixes: list[str] = []
        ok = True
        for i, mask_char in enumerate(fmt.mask):
            if _fits(chars[i], mask_char):
                continue  # legal already — never touched
            if len(fixes) >= cfg.max_substitutions:
                ok = False
                break
            pick = _best_allowed(alternatives, i, _allowed_for(mask_char))
            if pick is None:
                ok = False
                break
            char, prob = pick
            displaced = per_char_confidence[i] if i < len(per_char_confidence) else 0.0
            if not _acceptable(prob, displaced, cfg):
                ok = False
                break
            fixes.append(f"grammar-decode pos{i}: {chars[i]}->{char} (p={prob:.2f})")
            chars[i] = char
        if not ok or not fixes:
            continue
        out.append((len(fixes), "".join(chars), fixes))
    return out


def _rejection_note(
    text: str,
    alternatives: Sequence[Sequence[tuple[str, float]]],
    per_char_confidence: Sequence[float],
    cfg: GrammarDecodeConfig,
) -> str:
    """First guard failure, for the audit trail.

    Recomputed only when nothing was decoded, so the hot path never pays for
    it. Its whole purpose is to distinguish "we never looked at this position"
    from "we looked and the evidence did not support a change".
    """
    for fmt in postprocess.FORMATS:
        if len(fmt.mask) != len(text):
            continue
        for i, mask_char in enumerate(fmt.mask):
            if _fits(text[i], mask_char):
                continue
            pick = _best_allowed(alternatives, i, _allowed_for(mask_char))
            if pick is None:
                return f"pos{i}: no {'letter' if mask_char == postprocess.ALPHA else 'digit'} among the model's alternatives"
            char, prob = pick
            displaced = per_char_confidence[i] if i < len(per_char_confidence) else 0.0
            if not _acceptable(prob, displaced, cfg):
                return (
                    f"pos{i}: {text[i]}->{char} rejected "
                    f"(p={prob:.3f} vs displaced {displaced:.3f})"
                )
    return ""


def _best_allowed(
    alternatives: Sequence[Sequence[tuple[str, float]]], index: int, allowed: str
) -> Optional[tuple[str, float]]:
    if index >= len(alternatives):
        return None
    for char, prob in alternatives[index]:
        if char in allowed:
            return char, float(prob)
    return None


def _fix_state_code(
    text: str,
    alternatives: Sequence[Sequence[tuple[str, float]]],
    per_char_confidence: Sequence[float],
    cfg: GrammarDecodeConfig,
    max_subs: int | None = None,
) -> DecodeResult:
    """Repair a two-letter prefix that is not a real state code.

    This is the ``CL`` -> ``DL`` half of the example. The mask is satisfied —
    both are letters — so the type pass above cannot see the error at all. The
    RTO list can: a prefix outside it is either a misread or a vehicle from a
    state that does not exist.

    Type-preserving: only letters are considered, so this can never turn a
    letter into a digit and change the plate's shape. Both leading positions
    may be substituted, but the combined result must be a real state code and
    the whole plate must then validate.
    """
    budget = cfg.max_substitutions if max_subs is None else max_subs
    if budget <= 0 or len(text) < 3 or text[:2] in postprocess.STATE_CODES:
        return DecodeResult(text)
    fmt = postprocess.match_format(text)
    if fmt is None or not fmt.mask.startswith("AA"):
        # Only the civilian STATE+... shapes carry a state code at all.
        return DecodeResult(text)

    candidates_0 = _letter_options(text, alternatives, per_char_confidence, 0, cfg)
    candidates_1 = _letter_options(text, alternatives, per_char_confidence, 1, cfg)

    solutions: list[tuple[float, str, list[str]]] = []
    declined = ""
    for c0, p0, n0 in candidates_0:
        for c1, p1, n1 in candidates_1:
            if n0 + n1 == 0 or n0 + n1 > budget:
                continue
            prefix = c0 + c1
            if prefix not in postprocess.STATE_CODES:
                continue
            candidate = prefix + text[2:]
            if not postprocess.is_valid(candidate):
                continue
            fixes = []
            if n0:
                fixes.append(f"grammar-decode pos0: {text[0]}->{c0} (state code, p={p0:.2f})")
            if n1:
                fixes.append(f"grammar-decode pos1: {text[1]}->{c1} (state code, p={p1:.2f})")
            # Rank by joint evidence, so the most probable real prefix wins.
            solutions.append((p0 * p1, candidate, fixes))

    if not solutions:
        return DecodeResult(text, declined=declined)
    solutions.sort(key=lambda s: s[0], reverse=True)
    if len(solutions) > 1 and solutions[1][0] > 0 and solutions[0][0] / solutions[1][0] < 1.5:
        # Two plausible state codes. Which resident arrived is now a guess,
        # and a guess about the state prefix is a guess about the whole plate.
        return DecodeResult(
            text,
            declined=(
                f"ambiguous state code: {solutions[0][1][:2]} vs {solutions[1][1][:2]}"
            ),
        )
    _, candidate, fixes = solutions[0]
    return DecodeResult(candidate, fixes, changed=True)


def _letter_options(
    text: str,
    alternatives: Sequence[Sequence[tuple[str, float]]],
    per_char_confidence: Sequence[float],
    index: int,
    cfg: GrammarDecodeConfig,
) -> list[tuple[str, float, int]]:
    """(char, probability, substitutions_used) for one prefix position.

    Always includes keeping the character the recognizer chose, at zero cost,
    so a prefix with only ONE wrong letter is repaired by changing only that
    letter.
    """
    own = per_char_confidence[index] if index < len(per_char_confidence) else 1.0
    options: list[tuple[str, float, int]] = [(text[index], float(own), 0)]
    displaced = own
    for char, prob in (alternatives[index] if index < len(alternatives) else ()):
        if not char.isalpha():
            continue
        if _acceptable(float(prob), displaced, cfg):
            options.append((char, float(prob), 1))
    return options
