"""CTC decoding, shared by the LPRNet and PP-OCR recognizers.

Kept separate because the two models differ only in preprocessing and in
where the blank symbol sits — the decode itself is identical, and having one
implementation means one place to be right about repeat collapsing.

TWO DECODERS, AND WHY
---------------------
``ctc_greedy_decode`` returns the winning string and one probability per
character. That is all it ever returned, and it is a lossy summary of what the
model actually produced: the network emits a full distribution over every
character at every timestep, and argmax keeps one number from each column and
discards the rest inside this function, where nothing downstream can reach it.

That discarded evidence is exactly what is needed to fix a misread the grammar
can SEE but cannot CORRECT. Given ``DL7CP8161`` read as ``CLZCP0161``, the
plate format says position 2 must be a digit and ``Z`` is not one — so we know
the character is wrong, but not what it should be. No confusion map helps
(``ALPHA_TO_DIGIT['Z']`` is ``('2',)``, and the answer is ``7``). The model's
own second choice at that timestep almost certainly is ``7``, and it was
thrown away.

So ``ctc_decode_with_alternatives`` keeps the top few characters per emitted
position. Downstream, ``grammar_decode`` may substitute an alternative ONLY at
a position where the winner's type is illegal — evidence-driven, never a
rewrite of a confident legal choice.

``ctc_greedy_decode`` is kept unchanged: LPRNet uses it, and so do its tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Alternatives retained per emitted position. Four covers the realistic
#: confusion sets (a digit misread as one of two or three letters) without
#: carrying noise: beyond the fourth candidate the probabilities on a plate
#: crop are indistinguishable from zero.
DEFAULT_TOP_K = 4


@dataclass(frozen=True)
class CharPosterior:
    """What the model thought at one emitted character position.

    ``alternatives`` are the other characters at the SAME timestep, best
    first, excluding the winner and the blank. Same timestep is the right
    comparison set: CTC emits this character because that column's argmax
    said so, and the runner-up in that column is the character the model
    nearly chose instead.
    """

    char: str
    prob: float
    timestep: int
    alternatives: tuple[tuple[str, float], ...] = field(default_factory=tuple)

    def best_of(self, allowed: str) -> tuple[str, float] | None:
        """Highest-probability candidate whose character is in ``allowed``.

        Considers the winner too, so a position that is already legal returns
        its own character and nothing changes.
        """
        if self.char in allowed:
            return self.char, self.prob
        for char, prob in self.alternatives:
            if char in allowed:
                return char, prob
        return None


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _orient(probs: np.ndarray, charset: str) -> np.ndarray:
    """Normalize a model output to (T, C). Shared by both decoders."""
    if probs.ndim == 3:
        probs = probs[0]
    if probs.shape[0] < probs.shape[1] and probs.shape[1] > len(charset) + 1:
        probs = probs.T  # some exports emit (C, T)
    return probs


def _char_at(index: int, charset: str, blank_index: int, blank_first: bool) -> str | None:
    if index == blank_index:
        return None
    offset = index - 1 if blank_first else index
    return charset[offset] if 0 <= offset < len(charset) else None


def ctc_decode_with_alternatives(
    probs: np.ndarray,
    charset: str,
    blank_index: int = 0,
    blank_first: bool = True,
    top_k: int = DEFAULT_TOP_K,
) -> list[CharPosterior]:
    """Greedy CTC decode that KEEPS the runner-up characters.

    Identical collapsing rules to ``ctc_greedy_decode`` — same string comes
    out — so the two can never disagree about what was read. The only
    difference is how much of the evidence survives the call.
    """
    probs = _orient(probs, charset)
    if probs.size == 0:
        return []

    best = probs.argmax(axis=1)
    out: list[CharPosterior] = []
    previous = -1
    for timestep, index in enumerate(best):
        index = int(index)
        # Collapse repeats and drop blanks — the CTC rules, unchanged.
        if index != previous and index != blank_index:
            char = _char_at(index, charset, blank_index, blank_first)
            if char is not None:
                out.append(
                    CharPosterior(
                        char=char,
                        prob=float(probs[timestep, index]),
                        timestep=timestep,
                        alternatives=_alternatives(
                            probs[timestep], index, charset, blank_index, blank_first, top_k
                        ),
                    )
                )
        previous = index
    return out


def _alternatives(
    column: np.ndarray,
    chosen: int,
    charset: str,
    blank_index: int,
    blank_first: bool,
    top_k: int,
) -> tuple[tuple[str, float], ...]:
    if top_k <= 0:
        return ()
    # +2 candidates of slack so the chosen index and the blank can both be
    # skipped without shortening the result.
    count = min(len(column), top_k + 2)
    ranked = np.argpartition(-column, count - 1)[:count]
    ranked = ranked[np.argsort(-column[ranked])]

    out: list[tuple[str, float]] = []
    for index in ranked:
        index = int(index)
        if index in (chosen, blank_index):
            continue
        char = _char_at(index, charset, blank_index, blank_first)
        if char is None:
            continue
        out.append((char, float(column[index])))
        if len(out) >= top_k:
            break
    return tuple(out)


def summarize(posteriors: list[CharPosterior]) -> tuple[str, float, list[float]]:
    """Collapse a posterior list back to the ``ctc_greedy_decode`` triple, so
    a caller can use the richer decoder without changing its own contract."""
    if not posteriors:
        return "", 0.0, []
    text = "".join(p.char for p in posteriors)
    confs = [p.prob for p in posteriors]
    return text, float(np.mean(confs)), confs


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def ctc_greedy_decode(
    probs: np.ndarray, charset: str, blank_index: int = 0, blank_first: bool = True
) -> tuple[str, float, list[float]]:
    """Decode a (T, C) probability matrix.

    Returns the collapsed string, the mean per-character probability, and the
    per-character probabilities themselves. The per-character values are what
    let the validator vote position-by-position rather than string-by-string,
    so they matter as much as the text.
    """
    probs = _orient(probs, charset)

    best = probs.argmax(axis=1)
    scores = probs[np.arange(probs.shape[0]), best]

    chars: list[str] = []
    confs: list[float] = []
    previous = -1
    for t, index in enumerate(best):
        index = int(index)
        if index != previous and index != blank_index:
            offset = index - 1 if blank_first else index
            if 0 <= offset < len(charset):
                chars.append(charset[offset])
                confs.append(float(scores[t]))
        previous = index

    if not chars:
        return "", 0.0, []
    return "".join(chars), float(np.mean(confs)), confs


def load_charset(path: str | None, default: str) -> str:
    """Charset file: one character per line (PP-OCR style) or a single line.

    Falls back to ``default`` when no path is given so a recognizer is always
    constructible in a test without shipping a data file.
    """
    if not path:
        return default
    from pathlib import Path

    text = Path(path).read_text(encoding="utf-8")
    lines = [line.rstrip("\n\r") for line in text.splitlines()]
    if len(lines) > 1:
        return "".join(line for line in lines if line != "")
    return text.strip()
