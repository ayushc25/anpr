"""Qwen3-VL-2B (INT4) on ONNX Runtime GenAI, as a plate recognizer.

A generative model, not a CTC head, so two things the rest of the pipeline is
built around need an explicit answer here rather than being assumed.

CONFIDENCE IS NEVER INVENTED
----------------------------
Every number this class reports is a probability the model itself produced:

  * ``confidence``        geometric mean of the probabilities of the tokens that
                          make up the plate (``exp(mean(log p))``). A long
                          plate with one doubtful token is pulled down by it.
  * ``per_char_confidence``  the probability of the token each character came
                          from. Qwen's tokenizer emits digits one at a time but
                          merges some letters (``HR``, ``BG``), so two
                          characters can share one value. That is coarser than
                          a CTC timestep and is reported as it is.
  * ``per_char_alternatives``  the runner-up SINGLE characters at positions
                          whose chosen token was itself a single character.
                          Positions inside a merged token get none: the model
                          never scored "H" against "N" there, only "HR"
                          against its rivals, and fabricating a per-letter
                          split would hand ``grammar_decode`` evidence the
                          model did not give.

These are token probabilities, not calibrated accuracies. The validator
weights them against cross-frame agreement, which is the signal that actually
catches a fluent hallucination.

``recognize`` returns ``None`` for ``UNREADABLE`` and for anything that does
not normalize to a plausible length; it never guesses a plate to fill a gap.

``verify`` answers "does this crop read exactly <candidate>?" from the
probability mass the model puts on Yes versus No at the first answer token.
It returns ``None`` — not 0.5 — when neither word got meaningful mass.

onnxruntime_genai is imported lazily so that importing the registry on a box
without it (or in a test) costs nothing.
"""
from __future__ import annotations

import json
import logging
import math
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..inference.threading import ThreadBudget
from ..types import PlateRead
from .base import PlateRecognizer

logger = logging.getLogger("anpr.ai.qwen_vl")

DEFAULT_PROMPT = (
    "Read the Indian vehicle registration plate. "
    "Return ONLY the registration number. "
    "No explanation. "
    "No spaces. "
    "No punctuation. "
    "If unreadable, return UNREADABLE."
)

DEFAULT_VERIFY_PROMPT = (
    "This is an Indian vehicle registration plate. "
    'Does it read exactly "{candidate}"? '
    "Ignore spaces, hyphens and the IND emblem. "
    "Answer only Yes or No."
)

UNREADABLE = "UNREADABLE"

#: Below this combined Yes+No probability the model answered something else,
#: and a Yes/No ratio computed from the crumbs would be noise.
MIN_YES_NO_MASS = 0.30


@dataclass
class _Token:
    piece: str
    prob: float
    #: Runner-up single characters at this step, best first. Empty unless the
    #: chosen piece was itself a single character.
    alternatives: tuple[tuple[str, float], ...] = ()


@dataclass
class _Generation:
    tokens: list[_Token] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(t.piece for t in self.tokens)


class QwenVlOnnxRecognizer(PlateRecognizer):
    name = "qwen_vl_onnx"
    #: Generative: there is no fixed output alphabet to declare.
    charset = ""

    def __init__(
        self,
        artifact: str,
        prompt: str = DEFAULT_PROMPT,
        verify_prompt: str = DEFAULT_VERIFY_PROMPT,
        max_new_tokens: int = 24,
        alternatives_top_k: int = 6,
        threads: Optional[ThreadBudget] = None,
        backend: str = "cpu",
        cache_dir=None,
        **_ignored,
    ):
        model_dir = Path(artifact)
        if not (model_dir / "genai_config.json").exists():
            raise FileNotFoundError(
                f"{model_dir} is not an ORT-GenAI model directory (no genai_config.json). "
                "Fetch it with: python scripts/fetch_vlm.py"
            )
        if backend not in ("cpu", "auto"):
            logger.warning("qwen_vl_onnx runs on CPU; ignoring backend=%r", backend)

        import sys

        try:
            import onnxruntime_genai as og  # lazy: optional dependency
        except ImportError as exc:  # DLL failures surface as ImportError on Windows
            hint = (
                "onnxruntime_genai needs a much newer onnxruntime (>=1.30) than the detectors' "
                "pinned 1.20.1 and cannot share an interpreter with it. Run the VLM only in its "
                "own environment (backend/requirements-vlm.txt) via `anpr vlm-service`, and set "
                "vlm.python so the supervisor launches it with that interpreter."
            )
            loaded = " (the onnxruntime package was already loaded in this process)" if "onnxruntime" in sys.modules else ""
            raise RuntimeError(f"onnxruntime_genai could not be loaded: {exc}{loaded}. {hint}") from exc

        self._og = og
        self.prompt = prompt
        self.verify_prompt = verify_prompt
        self.max_new_tokens = int(max_new_tokens)
        self.alternatives_top_k = int(alternatives_top_k)

        config = og.Config(str(model_dir))
        if threads is not None and threads.intra_op > 0:
            # All three sessions, not just the decoder: the vision tower is a
            # large share of the per-crop time and would otherwise default to
            # one thread per core regardless of the budget.
            n = int(threads.intra_op)
            session = {"session_options": {"intra_op_num_threads": n}}
            config.overlay(json.dumps({"model": {"decoder": session, "vision": session, "embedding": session}}))
        self._model = og.Model(config)
        self._processor = self._model.create_multimodal_processor()
        self._tokenizer = og.Tokenizer(self._model)
        self._eos = set(int(t) for t in self._model_eos_ids())

        self._yes_ids = self._first_token_ids(("Yes", "yes", "YES"))
        self._no_ids = self._first_token_ids(("No", "no", "NO"))
        self._piece_cache: dict[int, str] = {}

    # -- public interface --------------------------------------------------
    def recognize(self, plate_image: np.ndarray) -> Optional[PlateRead]:
        generation = self._generate(plate_image, self.prompt, self.max_new_tokens, want_alternatives=True)
        if generation is None or not generation.tokens:
            return None
        raw = generation.text
        if not raw.strip() or UNREADABLE in raw.upper():
            return None

        per_char: list[float] = []
        alternatives: list[tuple[tuple[str, float], ...]] = []
        for token in generation.tokens:
            for _ in token.piece:
                per_char.append(token.prob)
                alternatives.append(token.alternatives)

        log_mean = sum(math.log(max(t.prob, 1e-9)) for t in generation.tokens) / len(generation.tokens)
        confidence = math.exp(log_mean)
        return self.finalize(raw, confidence, per_char, alternatives)

    def verify(self, plate_image: np.ndarray, candidate: str) -> Optional[float]:
        """Probability the crop reads exactly ``candidate``, or None."""
        if not candidate:
            return None
        prompt = self.verify_prompt.format(candidate=candidate)
        probs = self._first_step_probs(plate_image, prompt)
        if probs is None:
            return None
        yes = float(sum(probs[i] for i in self._yes_ids))
        no = float(sum(probs[i] for i in self._no_ids))
        if yes + no < MIN_YES_NO_MASS:
            return None
        return yes / (yes + no)

    @property
    def supports_verification(self) -> bool:
        return bool(self._yes_ids and self._no_ids)

    @property
    def expects_grayscale(self) -> bool:
        return False

    def warmup(self, n: int = 1) -> float:
        import time

        dummy = np.full((48, 160, 3), 200, dtype=np.uint8)
        started = time.perf_counter()
        for _ in range(max(1, n)):
            self._generate(dummy, self.prompt, 4, want_alternatives=False)
        return (time.perf_counter() - started) * 1000 / max(1, n)

    def close(self) -> None:
        self._model = None
        self._processor = None
        self._tokenizer = None

    # -- generation --------------------------------------------------------
    def _prepare(self, plate_image: np.ndarray, prompt_text: str):
        import cv2

        ok, buffer = cv2.imencode(".png", plate_image)
        if not ok:
            raise ValueError("could not encode the plate crop")
        images = self._og.Images.open_bytes(buffer.tobytes())
        messages = json.dumps(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
        )
        chat = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        return self._processor(chat, images=images)

    def _new_generator(self, inputs, max_tokens: int):
        og = self._og
        params = og.GeneratorParams(self._model)
        # Greedy, bounded by the PROMPT length too: max_length counts the
        # image tokens, so a flat 24 would stop generation before it began.
        params.set_search_options(max_length=2000, do_sample=False)
        generator = og.Generator(self._model, params)
        generator.set_inputs(inputs)
        return generator

    def _generate(
        self, plate_image: np.ndarray, prompt_text: str, max_tokens: int, want_alternatives: bool
    ) -> Optional[_Generation]:
        if plate_image is None or plate_image.size == 0:
            return None
        generator = self._new_generator(self._prepare(plate_image, prompt_text), max_tokens)
        result = _Generation()
        for _ in range(max_tokens):
            if generator.is_done():
                break
            generator.generate_next_token()
            token_id = int(generator.get_next_tokens()[0])
            if token_id in self._eos:
                break
            logits = np.asarray(generator.get_output("logits"), dtype=np.float32)
            row = logits.reshape(-1, logits.shape[-1])[-1]
            probs = _softmax(row)
            piece = self._piece(token_id)
            alternatives: tuple[tuple[str, float], ...] = ()
            if want_alternatives and len(piece.strip()) == 1 and piece.strip().isalnum():
                alternatives = self._runner_ups(probs, token_id, piece.strip().upper())
            result.tokens.append(_Token(piece=piece, prob=float(probs[token_id]), alternatives=alternatives))
        return result

    def _first_step_probs(self, plate_image: np.ndarray, prompt_text: str) -> Optional[np.ndarray]:
        if plate_image is None or plate_image.size == 0:
            return None
        generator = self._new_generator(self._prepare(plate_image, prompt_text), 1)
        if generator.is_done():
            return None
        generator.generate_next_token()
        logits = np.asarray(generator.get_output("logits"), dtype=np.float32)
        return _softmax(logits.reshape(-1, logits.shape[-1])[-1])

    # -- helpers -----------------------------------------------------------
    def _runner_ups(
        self, probs: np.ndarray, chosen: int, chosen_char: str
    ) -> tuple[tuple[str, float], ...]:
        k = max(2, self.alternatives_top_k)
        top = np.argpartition(probs, -k)[-k:]
        merged: dict[str, float] = {}
        for idx in top:
            if int(idx) == chosen:
                continue
            piece = self._piece(int(idx)).strip()
            if len(piece) != 1 or not piece.isalnum():
                continue
            char = piece.upper()
            if char == chosen_char:
                continue
            merged[char] = merged.get(char, 0.0) + float(probs[idx])
        return tuple(sorted(merged.items(), key=lambda kv: kv[1], reverse=True))

    def _piece(self, token_id: int) -> str:
        cached = self._piece_cache.get(token_id)
        if cached is None:
            cached = _ascii_piece(self._tokenizer.decode([token_id]))
            self._piece_cache[token_id] = cached
        return cached

    def _first_token_ids(self, words) -> list[int]:
        ids = []
        for word in words:
            try:
                encoded = self._tokenizer.encode(word)
                if len(encoded):
                    ids.append(int(encoded[0]))
            except Exception:  # pragma: no cover - tokenizer-specific
                logger.debug("could not encode %r", word, exc_info=True)
        return sorted(set(ids))

    def _model_eos_ids(self):
        try:
            return list(self._tokenizer.eos_token_ids)
        except Exception:  # pragma: no cover
            return []


def _ascii_piece(text: str) -> str:
    """A decoded token as plain ASCII letters and digits, or "-" where it is not.

    Seen in practice: the model emits FULLWIDTH letters (U+FF23 for "C") on some
    crops, and ``str.isalnum`` accepts them, so they would sail through
    normalization into a plate string no registry lookup could ever match. NFKC
    folds them to ASCII; whatever is still not ASCII alphanumeric (a stray
    glyph, half of a split multi-byte character) becomes "-", which every
    downstream filter drops. One output character per input character, so the
    per-character arrays built from the pieces stay aligned.
    """
    folded = unicodedata.normalize("NFKC", text)
    return "".join(ch if (ch.isascii() and ch.isalnum()) else "-" for ch in folded)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()
