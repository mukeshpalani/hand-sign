"""
caption_generator.py

MODULE 4 — Caption Generation

Single Responsibility
----------------------
This module's ONLY job is: take the raw word/sentence stream produced by
continuous_recognition.py and turn it into polished, DISPLAYABLE captions —
correcting basic grammar, punctuating and capitalizing sentences, and
attaching timestamps (project requirement 5-6: "Convert recognized signs
into words and sentences" / "Display live captions").

It does NOT recognize signs itself, does not speak captions aloud, and
does not render any UI — it is a pure text-transformation stage sitting
between continuous_recognition.py and both voice_assistant.py (which
speaks `Caption.text`) and main.py's GUI (which displays it), plus
avatar_module.py (which synchronizes animation to it).

Design notes
------------
- STRATEGY PATTERN for grammar correction: the actual correction logic is
  hidden behind a `GrammarCorrector` abstract interface. The default
  implementation (`RuleBasedGrammarCorrector`) does simple, explainable,
  dependency-free polishing (capitalization, spacing, punctuation, a small
  fixed set of common substitutions). A more advanced implementation
  (e.g. wrapping an external NLP grammar-correction library or a small
  language model) can be swapped in later via the SAME interface, without
  any changes to this module's public API (Open/Closed Principle).
- This module maintains a bounded HISTORY of finalized captions, since
  main.py's GUI and voice_assistant.py may want to show/read back recent
  captions, not just the current one.
- Live (in-progress) captions and finalized captions are represented with
  the same `Caption` DTO but different `is_final` flags, so consumers can
  decide how to treat each (e.g. voice_assistant.py should typically only
  speak FINAL captions, to avoid narrating every partial word as it's
  still being built).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

from utils.data_types import Caption, RecognitionResult
from utils.logger import get_logger

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# Pluggable grammar correction interface (Strategy Pattern)
# ----------------------------------------------------------------------


class GrammarCorrector(ABC):
    """
    Abstract interface for turning a raw list of recognized word tokens
    (often in sign-language gloss order, e.g. "STORE I GO YESTERDAY")
    into fluent, grammatically-reasonable text (e.g. "I went to the store
    yesterday.").

    Concrete implementations can range from simple rule-based heuristics
    (this file's default) to a full NLP model. Swapping implementations
    never requires changes to CaptionGenerator, satisfying the Dependency
    Inversion Principle.
    """

    @abstractmethod
    def correct(self, tokens: List[str]) -> str:
        """
        Args:
            tokens: The raw recognized word sequence, in recognition order.

        Returns:
            A grammar-corrected, capitalized, punctuated string.
        """
        raise NotImplementedError


class RuleBasedGrammarCorrector(GrammarCorrector):
    """
    A lightweight, dependency-free grammar corrector using simple,
    explainable rules. This is intentionally NOT a full NLP solution —
    it exists so the pipeline produces reasonably readable captions
    out-of-the-box without requiring an external grammar-correction
    service, while still being trivially replaceable later.

    Rules applied, in order:
      1. Drop empty/whitespace-only tokens.
      2. Capitalize the pronoun "i" wherever it appears (common issue
         since sign vocabularies are usually lowercase).
      3. Capitalize the first word of the sentence.
      4. Join tokens with single spaces.
      5. Append terminal punctuation (defaults to a period) if the
         sentence doesn't already end in one, unless it looks like a
         question (ends with a recognized question word) in which case
         a question mark is used instead.
    """

    # Small, fixed set of interrogative words that (per common sign
    # language gloss conventions) often appear as the FIRST or LAST token
    # of a question, used only to decide "." vs "?" terminal punctuation.
    _QUESTION_WORDS = {
        "what", "who", "where", "when", "why", "how", "which", "whose",
    }

    def correct(self, tokens: List[str]) -> str:
        """See GrammarCorrector.correct for the general contract."""
        cleaned_tokens = [t.strip() for t in tokens if t and t.strip()]
        if not cleaned_tokens:
            return ""

        normalized_tokens = [
            "I" if t.lower() == "i" else t for t in cleaned_tokens
        ]

        sentence = " ".join(normalized_tokens)
        sentence = sentence[0].upper() + sentence[1:] if sentence else sentence

        if sentence[-1] not in ".?!":
            is_question = (
                normalized_tokens[0].lower() in self._QUESTION_WORDS
                or normalized_tokens[-1].lower() in self._QUESTION_WORDS
            )
            sentence += "?" if is_question else "."

        return sentence


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@dataclass
class CaptionGeneratorConfig:
    """Tunable parameters for caption generation and history retention."""

    # Maximum number of FINALIZED captions kept in history (older ones are
    # dropped). Prevents unbounded memory growth in long-running sessions.
    max_history_size: int = 100

    # Whether to emit an updated "live" (in-progress) caption on every
    # single new word, or only recompute it when explicitly requested.
    # True is more responsive for a live-captioning UI; False saves a
    # little compute if main.py prefers to poll instead.
    emit_live_captions: bool = True


# ----------------------------------------------------------------------
# Caption generator
# ----------------------------------------------------------------------


class CaptionGenerator:
    """
    Converts RecognitionResult updates (from continuous_recognition.py)
    into polished Caption objects, maintaining a bounded history of
    finalized captions.

    Usage:
        generator = CaptionGenerator()
        for recognition_update in stream_of_recognition_results:
            caption = generator.process(recognition_update)
            if caption:
                # Display caption.text on screen; if caption.is_final,
                # also hand it to voice_assistant.py / avatar_module.py.
                ...
    """

    def __init__(
        self,
        grammar_corrector: Optional[GrammarCorrector] = None,
        config: Optional[CaptionGeneratorConfig] = None,
    ) -> None:
        """
        Args:
            grammar_corrector: Optional GrammarCorrector implementation.
                Defaults to RuleBasedGrammarCorrector if omitted.
            config: Optional CaptionGeneratorConfig; defaults used if
                omitted.
        """
        self._config = config or CaptionGeneratorConfig()
        self._grammar_corrector = grammar_corrector or RuleBasedGrammarCorrector()

        self._history: Deque[Caption] = deque(maxlen=self._config.max_history_size)
        self._current_live_caption: Optional[Caption] = None

        logger.info(
            "CaptionGenerator initialized (grammar_corrector=%s, "
            "max_history_size=%d)",
            type(self._grammar_corrector).__name__,
            self._config.max_history_size,
        )

    def process(self, recognition: RecognitionResult) -> Optional[Caption]:
        """
        Convert one RecognitionResult update into a Caption, if there's
        anything new to report this frame.

        Args:
            recognition: The latest output from
                ContinuousRecognizer.process().

        Returns:
            A Caption if a new word was added and/or the sentence was
            just finalized this frame; None if nothing changed (e.g. a
            frame with no new word and no completed sentence — most
            frames, in practice, since recognition only fires every
            `stride` frames).
        """
        try:
            return self._process_impl(recognition)
        except Exception:
            logger.exception(
                "CaptionGenerator failed on frame_index=%d; skipping caption "
                "update for this frame.",
                recognition.frame_index,
            )
            return None

    def _process_impl(self, recognition: RecognitionResult) -> Optional[Caption]:
        if recognition.is_sentence_complete:
            return self._finalize_caption(recognition)

        if recognition.new_word is not None and self._config.emit_live_captions:
            return self._build_live_caption(recognition)

        # Nothing new this frame.
        return None

    def _build_live_caption(self, recognition: RecognitionResult) -> Caption:
        """
        Build an in-progress ("live") caption reflecting the sentence as
        recognized so far. Live captions use a lighter-weight formatting
        pass (capitalize first word only) rather than full grammar
        correction + terminal punctuation, since the sentence may still
        be growing and a period mid-sentence would look wrong.
        """
        raw_text = recognition.sentence_text()
        display_text = raw_text[0].upper() + raw_text[1:] if raw_text else raw_text

        caption = Caption(
            text=display_text,
            raw_text=raw_text,
            timestamp_ms=recognition.timestamp_ms,
            frame_index=recognition.frame_index,
            is_final=False,
        )
        self._current_live_caption = caption
        logger.debug("Live caption updated: '%s'", caption.text)
        return caption

    def _finalize_caption(self, recognition: RecognitionResult) -> Caption:
        """
        Build the FINAL, grammar-corrected caption for a just-completed
        sentence, add it to history, and clear the live-caption state.
        """
        raw_text = recognition.sentence_text()
        corrected_text = self._grammar_corrector.correct(recognition.sentence_tokens)

        caption = Caption(
            text=corrected_text,
            raw_text=raw_text,
            timestamp_ms=recognition.timestamp_ms,
            frame_index=recognition.frame_index,
            is_final=True,
        )

        self._history.append(caption)
        self._current_live_caption = None

        logger.info(
            "Finalized caption: '%s' (raw: '%s')", caption.text, caption.raw_text
        )
        return caption

    def get_current_live_caption(self) -> Optional[Caption]:
        """Returns the most recent in-progress caption, or None if there
        is no sentence currently being built."""
        return self._current_live_caption

    def get_history(self, count: Optional[int] = None) -> List[Caption]:
        """
        Retrieve finalized captions from history, most-recent-last.

        Args:
            count: If provided, return only the last `count` captions.
                If None, return the entire retained history.
        """
        history_list = list(self._history)
        if count is None:
            return history_list
        return history_list[-count:]

    def reset(self) -> None:
        """
        Clear all caption history and in-progress state.

        Should be called when starting a new, unrelated video session, so
        stale captions from a previous session don't linger in history.
        """
        self._history.clear()
        self._current_live_caption = None
        logger.info("CaptionGenerator history and live state reset.")


if __name__ == "__main__":
    # Minimal manual smoke-test: feeds a scripted sequence of
    # RecognitionResult-like updates through CaptionGenerator directly
    # (no camera/model needed), to verify grammar correction and history
    # behave as expected. Run via: `python caption_generator.py`.
    from utils.data_types import RecognitionResult

    logger.info("Running caption_generator.py standalone demo.")

    generator = CaptionGenerator()

    scripted_words = ["store", "i", "go", "yesterday"]
    tokens_so_far: List[str] = []

    for idx, word in enumerate(scripted_words):
        tokens_so_far.append(word)
        update = RecognitionResult(
            frame_index=idx,
            timestamp_ms=idx * 500.0,
            sentence_tokens=list(tokens_so_far),
            new_word=None,  # not needed for this smoke test's fields used
        )
        # Simulate continuous_recognition.py setting new_word on this frame.
        from utils.data_types import WordPrediction

        update.new_word = WordPrediction(word=word, confidence=0.9, timestamp_ms=idx * 500.0)

        caption = generator.process(update)
        if caption:
            print(f"[LIVE] {caption.formatted_timestamp()} -> {caption.text}")

    # Now simulate the pause that finalizes the sentence.
    final_update = RecognitionResult(
        frame_index=len(scripted_words),
        timestamp_ms=len(scripted_words) * 500.0,
        sentence_tokens=list(tokens_so_far),
        is_sentence_complete=True,
    )
    final_caption = generator.process(final_update)
    if final_caption:
        print(f"[FINAL] {final_caption.formatted_timestamp()} -> {final_caption.text}")

    print("History:", [c.text for c in generator.get_history()])
