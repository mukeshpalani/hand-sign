"""
voice_assistant.py

MODULE 6 — Voice Assistant (Text-to-Speech)

Single Responsibility
----------------------
This module's ONLY job is: take finalized captions (or arbitrary text, e.g.
from teacher_module.py's feedback) and SPEAK them aloud, with support for
pausing, resuming, and volume control (project requirement 7: "Read the
captions aloud using a voice assistant").

It does NOT recognize signs, does NOT generate captions, and does NOT
render any UI — it purely consumes text/Caption objects and produces audio
output.

Design notes
------------
- STRATEGY PATTERN for the TTS backend: the actual speech engine is hidden
  behind a `TTSEngine` abstract interface. The default implementation
  wraps `pyttsx3` (a fully offline, cross-platform TTS engine — chosen
  deliberately over cloud TTS APIs so the voice assistant keeps working
  without network access). If `pyttsx3` isn't installed, or audio hardware
  isn't available (e.g. a headless server), we fall back to a
  `NullTTSEngine` that logs what WOULD have been spoken instead of
  crashing the whole pipeline (Null Object Pattern, same approach used in
  continuous_recognition.py for the model backend).
- SPEECH RUNS ON A DEDICATED BACKGROUND THREAD: speaking text is
  inherently blocking (audio playback takes real time), so if we called
  the TTS engine directly from main.py's video loop, captions would freeze
  the camera feed for as long as speech takes. Instead, `VoiceAssistant`
  owns a queue + worker thread: callers just call `speak_caption()` (a
  fast, non-blocking enqueue) and the worker thread handles actual
  playback independently.
- PAUSE/RESUME LIMITATION (documented honestly rather than hidden): most
  offline TTS engines, including pyttsx3, do not support pausing and later
  resuming mid-utterance — once interrupted, an utterance cannot be
  resumed from the exact word it stopped at. This module implements
  pause/resume at the QUEUE level: `pause()` immediately stops any
  currently-playing utterance and halts the worker from starting new ones;
  `resume()` lets the worker continue with the NEXT queued item (the
  interrupted utterance itself is not replayed automatically, but remains
  discoverable via `get_last_interrupted_text()` so main.py can choose to
  re-queue it if desired).
"""

from __future__ import annotations

import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from utils.data_types import Caption
from utils.logger import get_logger

logger = get_logger(__name__)


try:
    import pyttsx3
except ImportError:
    pyttsx3 = None  # Handled gracefully by Pyttsx3TTSEngine / VoiceAssistant.


# ----------------------------------------------------------------------
# Pluggable TTS engine interface (Strategy Pattern)
# ----------------------------------------------------------------------


class TTSEngine(ABC):
    """
    Abstract interface for any text-to-speech backend.

    VoiceAssistant depends only on this interface, never on a concrete
    TTS library — so pyttsx3 can be swapped for another engine (a cloud
    TTS API, a different offline engine, etc.) later without touching
    VoiceAssistant's threading/queueing logic at all.
    """

    @abstractmethod
    def speak(self, text: str) -> None:
        """
        Synchronously speak the given text (this call is expected to
        block for the duration of playback — VoiceAssistant's worker
        thread is what makes this non-blocking for the rest of the app).
        """
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Immediately interrupt any currently-playing speech."""
        raise NotImplementedError

    @abstractmethod
    def set_volume(self, volume: float) -> None:
        """Set playback volume, where 0.0 is silent and 1.0 is full volume."""
        raise NotImplementedError

    @abstractmethod
    def set_rate(self, words_per_minute: int) -> None:
        """Set the speaking rate in approximate words per minute."""
        raise NotImplementedError


class Pyttsx3TTSEngine(TTSEngine):
    """
    TTSEngine implementation backed by pyttsx3 — a fully offline,
    cross-platform (Windows SAPI5 / macOS NSSpeechSynthesizer / Linux
    espeak) text-to-speech library. Chosen as the default so the voice
    assistant works without an internet connection or API keys.
    """

    def __init__(self, default_volume: float = 1.0, default_rate: int = 175) -> None:
        if pyttsx3 is None:
            raise RuntimeError(
                "pyttsx3 is not installed. Install it via: pip install pyttsx3 "
                "(Linux additionally requires the 'espeak' system package)."
            )

        try:
            self._engine = pyttsx3.init()
        except Exception as exc:
            # Common on headless machines with no audio backend at all.
            raise RuntimeError(
                "Failed to initialize pyttsx3 (no audio backend available?). "
                f"Underlying error: {exc}"
            ) from exc

        self.set_volume(default_volume)
        self.set_rate(default_rate)
        logger.info(
            "Pyttsx3TTSEngine initialized (volume=%.2f, rate=%d wpm).",
            default_volume,
            default_rate,
        )

    def speak(self, text: str) -> None:
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception:
            logger.exception("Pyttsx3TTSEngine failed to speak text: %r", text)

    def stop(self) -> None:
        try:
            self._engine.stop()
        except Exception:
            logger.exception("Pyttsx3TTSEngine failed to stop playback.")

    def set_volume(self, volume: float) -> None:
        clamped = max(0.0, min(1.0, volume))
        self._engine.setProperty("volume", clamped)

    def set_rate(self, words_per_minute: int) -> None:
        self._engine.setProperty("rate", max(50, words_per_minute))


class NullTTSEngine(TTSEngine):
    """
    A safe, no-op TTS engine that logs what would have been spoken instead
    of producing audio. Used automatically as a fallback when pyttsx3 is
    unavailable or fails to initialize (e.g. CI environments, headless
    servers, or containers without audio hardware), so the rest of the
    pipeline (captions, avatar, recognition) can still be developed and
    tested without a working audio stack.
    """

    def __init__(self) -> None:
        logger.warning(
            "VoiceAssistant is using NullTTSEngine — no audio will be "
            "produced. Captions will be logged instead of spoken."
        )
        self._volume = 1.0
        self._rate = 175

    def speak(self, text: str) -> None:
        # Simulate the passage of time real speech would take, VERY
        # roughly (~150 wpm), so downstream pause/resume/queueing logic
        # can still be exercised meaningfully in tests without real audio.
        logger.info("[NullTTSEngine] Would speak: '%s'", text)
        approx_seconds = max(0.3, len(text.split()) / 2.5)
        time.sleep(approx_seconds)

    def stop(self) -> None:
        logger.debug("[NullTTSEngine] stop() called (no-op).")

    def set_volume(self, volume: float) -> None:
        self._volume = max(0.0, min(1.0, volume))
        logger.debug("[NullTTSEngine] volume set to %.2f", self._volume)

    def set_rate(self, words_per_minute: int) -> None:
        self._rate = max(50, words_per_minute)
        logger.debug("[NullTTSEngine] rate set to %d wpm", self._rate)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@dataclass
class VoiceAssistantConfig:
    """Tunable parameters for the voice assistant's playback behaviour."""

    default_volume: float = 1.0
    default_rate: int = 175

    # If True, only FINAL captions (Caption.is_final=True) are spoken —
    # in-progress/live captions are skipped, avoiding narrating every
    # single word as the sentence is still being built. Recommended True
    # for a natural-sounding assistant.
    speak_only_final_captions: bool = True

    # Maximum number of pending texts the speech queue will hold before
    # dropping the oldest entry. Prevents unbounded memory growth and
    # runaway speech backlog if captions arrive faster than they can be
    # spoken.
    max_queue_size: int = 20


# ----------------------------------------------------------------------
# Voice assistant
# ----------------------------------------------------------------------


class VoiceAssistant:
    """
    Speaks captions (or arbitrary text) aloud on a dedicated background
    thread, with pause/resume/volume/stop controls.

    Usage:
        voice = VoiceAssistant()
        voice.speak_caption(caption)      # non-blocking enqueue
        voice.set_volume(0.5)
        voice.pause()
        voice.resume()
        voice.shutdown()                  # call when done with the app
    """

    def __init__(
        self,
        engine: Optional[TTSEngine] = None,
        config: Optional[VoiceAssistantConfig] = None,
    ) -> None:
        """
        Args:
            engine: Optional TTSEngine implementation. If omitted, attempts
                Pyttsx3TTSEngine first and falls back to NullTTSEngine if
                that fails for any reason (missing library, no audio
                hardware, etc.) — the voice assistant should never crash
                the whole application just because TTS isn't available.
            config: Optional VoiceAssistantConfig; defaults used if omitted.
        """
        self._config = config or VoiceAssistantConfig()
        self._engine = engine or self._create_default_engine()

        self._engine.set_volume(self._config.default_volume)
        self._engine.set_rate(self._config.default_rate)
        self._current_volume = self._config.default_volume

        self._queue: "queue.Queue[str]" = queue.Queue(
            maxsize=self._config.max_queue_size
        )
        self._pause_event = threading.Event()
        self._shutdown_event = threading.Event()
        self._last_interrupted_text: Optional[str] = None

        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="VoiceAssistantWorker", daemon=True
        )
        self._worker_thread.start()

        logger.info(
            "VoiceAssistant started (engine=%s, speak_only_final=%s).",
            type(self._engine).__name__,
            self._config.speak_only_final_captions,
        )

    def _create_default_engine(self) -> TTSEngine:
        """Try to construct a real TTS engine; fall back to NullTTSEngine
        on any failure so TTS being unavailable never crashes the app."""
        try:
            return Pyttsx3TTSEngine(
                default_volume=self._config.default_volume,
                default_rate=self._config.default_rate,
            )
        except Exception as exc:
            logger.warning(
                "Falling back to NullTTSEngine because Pyttsx3TTSEngine "
                "could not be created: %s",
                exc,
            )
            return NullTTSEngine()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak_caption(self, caption: Caption) -> None:
        """
        Enqueue a Caption's text to be spoken, respecting
        `speak_only_final_captions`. Non-blocking.

        Args:
            caption: The Caption to (potentially) speak. Ignored if it's a
                live/in-progress caption and speak_only_final_captions is
                True.
        """
        if self._config.speak_only_final_captions and not caption.is_final:
            return
        self.speak_text(caption.text)

    def speak_text(self, text: str) -> None:
        """
        Enqueue arbitrary text to be spoken (e.g. teacher_module.py
        feedback like "Straighten your index finger."). Non-blocking.

        If the queue is full, the oldest pending item is dropped to make
        room — we prioritize keeping the assistant responsive to NEW
        speech over guaranteeing every queued utterance is eventually
        spoken.
        """
        if not text or not text.strip():
            return

        try:
            self._queue.put_nowait(text)
        except queue.Full:
            logger.warning(
                "VoiceAssistant queue full (max=%d); dropping oldest "
                "pending utterance to make room for new text.",
                self._config.max_queue_size,
            )
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(text)

    def pause(self) -> None:
        """
        Pause speech: immediately interrupts any currently-playing
        utterance and prevents the worker from starting new ones until
        resume() is called.

        Note: per this module's documented limitation, the interrupted
        utterance is NOT automatically resumed from where it left off;
        see get_last_interrupted_text() if you want to re-queue it.
        """
        self._pause_event.set()
        self._engine.stop()
        logger.info("VoiceAssistant paused.")

    def resume(self) -> None:
        """Resume speech: the worker will continue processing the queue
        starting with the next pending item."""
        self._pause_event.clear()
        logger.info("VoiceAssistant resumed.")

    def is_paused(self) -> bool:
        """Returns True if the assistant is currently paused."""
        return self._pause_event.is_set()

    def set_volume(self, volume: float) -> None:
        """
        Set playback volume.

        Args:
            volume: A value in [0.0, 1.0]; out-of-range values are
                clamped rather than raising, since volume sliders in a
                GUI can easily send transient out-of-range values.
        """
        clamped = max(0.0, min(1.0, volume))
        self._current_volume = clamped
        self._engine.set_volume(clamped)
        logger.debug("Volume set to %.2f", clamped)

    def get_volume(self) -> float:
        """Returns the currently configured volume, in [0.0, 1.0]."""
        return self._current_volume

    def set_rate(self, words_per_minute: int) -> None:
        """Set the speaking rate, in approximate words per minute."""
        self._engine.set_rate(words_per_minute)

    def clear_queue(self) -> None:
        """Discard all pending (not-yet-spoken) queued text without
        affecting anything currently playing."""
        cleared = 0
        while True:
            try:
                self._queue.get_nowait()
                cleared += 1
            except queue.Empty:
                break
        if cleared:
            logger.info("Cleared %d pending utterance(s) from queue.", cleared)

    def get_last_interrupted_text(self) -> Optional[str]:
        """
        Returns the text that was actively being spoken (if any) at the
        moment pause() was last called, so a caller (e.g. main.py) can
        choose to re-queue it explicitly after resume() — since, per this
        module's TTS-engine limitation, it is not resumed automatically.
        """
        return self._last_interrupted_text

    def shutdown(self, wait: bool = True) -> None:
        """
        Stop the worker thread and release engine resources. Should be
        called once when the application is closing.

        Args:
            wait: If True, block until the worker thread has fully exited.
        """
        logger.info("Shutting down VoiceAssistant...")
        self._shutdown_event.set()
        self._pause_event.clear()  # ensure worker isn't stuck waiting
        self._engine.stop()
        self.clear_queue()

        if wait and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)

        logger.info("VoiceAssistant shut down.")

    def __enter__(self) -> "VoiceAssistant":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        """
        Background worker: pulls text from the queue and speaks it one
        item at a time, respecting pause state. Runs until shutdown() is
        called.
        """
        while not self._shutdown_event.is_set():
            if self._pause_event.is_set():
                # Don't start new utterances while paused; poll cheaply.
                time.sleep(0.1)
                continue

            try:
                text = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if self._shutdown_event.is_set():
                break

            self._last_interrupted_text = text
            try:
                self._engine.speak(text)
                # Successfully finished speaking (wasn't interrupted) —
                # clear the "last interrupted" marker.
                self._last_interrupted_text = None
            except Exception:
                logger.exception("Error while speaking text: %r", text)

        logger.debug("VoiceAssistant worker thread exiting.")


if __name__ == "__main__":
    # Minimal manual smoke-test / usage example. Run via:
    #   python voice_assistant.py
    # Uses whatever TTS engine is available (falls back to NullTTSEngine
    # if pyttsx3/audio isn't available in this environment), so this is
    # safe to run in headless containers too.
    logger.info("Running voice_assistant.py standalone demo.")

    with VoiceAssistant() as assistant:
        assistant.set_volume(0.8)
        assistant.speak_text("Hello, this is the AI Sign Language Assistant.")
        assistant.speak_text("This second sentence should play right after the first.")

        time.sleep(1.0)
        logger.info("Pausing...")
        assistant.pause()
        time.sleep(1.0)
        logger.info("Resuming...")
        assistant.resume()

        # Give the worker thread time to finish speaking the queued items
        # before the context manager shuts everything down.
        time.sleep(4.0)

    logger.info("Demo complete.")
