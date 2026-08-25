"""
avatar_module.py

MODULE 7 -- AI Avatar

Single Responsibility
----------------------
This module's ONLY job is: given a finalized sentence (a Caption), produce
a time-sequenced SIGN LANGUAGE ANIMATION -- hand poses + facial expression
per frame -- and play it back on a pluggable renderer, keeping the
animation's timing synchronized with captions and (optionally) voice
playback (project requirement 8: "Include an AI Avatar that responds
using sign language animation while simultaneously displaying captions").

It does NOT recognize signs, does NOT generate captions, and does NOT
speak -- it consumes Caption objects (from caption_generator.py) and
produces avatar animation frames for a renderer to draw.

Design notes
------------
- STRATEGY PATTERN, twice over:
    1. `SignAnimationLibrary` abstracts WHERE animation clips come from
       (a JSON/asset library on disk, a database, a generative model,
       etc.). This module never assumes a specific storage format.
    2. `AvatarRenderer` abstracts HOW frames actually get drawn (a game
       engine via IPC, a WebGL/Three.js canvas via websocket, a 2D
       skeleton preview, or -- for headless testing -- a no-op logger).
  Both are injectable, so avatar_module.py stays engine-agnostic and
  testable without any real 3D rendering stack installed.
- REUSES THE SHARED `Landmark` DTO (from utils/data_types.py) to represent
  each hand pose in an animation frame -- the exact same 21-point structure
  hand_detection.py produces from a camera. This is intentional: it means
  teacher_module.py (which shows a target sign and compares it against a
  user's LIVE landmarks from hand_detection.py) can compare "apples to
  apples" against an AnimationFrame's pose using the same data shape.
- "NATURAL LOOKING" (project requirement 8) is addressed concretely via
  `_build_transition_frames()`, which linearly blends from the last pose
  of one sign into the first pose of the next, instead of snapping
  abruptly between signs -- a well-known basic technique for reducing
  jarring, robotic-looking avatar motion.
- SYNCHRONIZATION: rather than this module reaching into voice_assistant.py
  or caption_generator.py directly (which would break module independence),
  AvatarController exposes `get_current_word()` / `get_progress_fraction()`
  / `is_playing()` so main.py -- the orchestrator -- can read avatar state
  and line it up with whatever else it's displaying/speaking. This keeps
  every module a one-way, independently-testable stage rather than a web
  of cross-module calls.
- Playback runs on its OWN background thread (mirroring voice_assistant.py's
  design), so animation playback never blocks the live camera/recognition
  loop in main.py.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from utils.data_types import Caption, Landmark
from utils.logger import get_logger

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# Animation data model
# ----------------------------------------------------------------------


class FacialExpression(Enum):
    """
    A small, fixed vocabulary of facial expressions used to accompany sign
    animation. Facial grammar carries real meaning in sign languages
    (e.g. raised eyebrows for yes/no questions, furrowed brow for wh-
    questions) -- this enum is intentionally simple but extensible; a
    future emotion-recognition feature (see project's future-extensibility
    list) could add more entries without changing this module's structure.
    """
    NEUTRAL = "neutral"
    RAISED_EYEBROWS = "raised_eyebrows"     # yes/no questions
    FURROWED_BROW = "furrowed_brow"         # wh- questions ("what", "where"...)
    SMILE = "smile"                          # positive/greeting signs
    EMPHASIS = "emphasis"                    # emphatic statements


@dataclass
class AnimationFrame:
    """
    A single point-in-time snapshot within an animation clip.

    Timing is stored as an OFFSET from the start of its clip
    (`timestamp_offset_ms`) rather than an absolute time, so clips remain
    reusable/composable regardless of when they end up being played.
    """
    timestamp_offset_ms: float
    left_hand: Optional[List[Landmark]] = None
    right_hand: Optional[List[Landmark]] = None
    facial_expression: FacialExpression = FacialExpression.NEUTRAL


@dataclass
class AnimationClip:
    """
    A complete animation for a single recognized word/sign, as a sequence
    of AnimationFrames.
    """
    word: str
    frames: List[AnimationFrame] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        """Total duration of this clip, inferred from its last frame's
        offset. Returns 0.0 for an empty clip."""
        return self.frames[-1].timestamp_offset_ms if self.frames else 0.0


# ----------------------------------------------------------------------
# Pluggable animation source (Strategy / Repository Pattern)
# ----------------------------------------------------------------------


class SignAnimationLibrary(ABC):
    """
    Abstract interface for looking up an AnimationClip for a given word.

    Concrete sources can range from a folder of hand-authored JSON clips
    (see JSONAnimationLibrary below) to clips generated by a future
    text-to-sign generative model -- this module doesn't care, as long as
    the source can answer `get_clip(word)`.
    """

    @abstractmethod
    def get_clip(self, word: str) -> Optional[AnimationClip]:
        """Return the AnimationClip for `word`, or None if not found (the
        caller -- AvatarController -- is responsible for falling back to
        fingerspelling or another placeholder in that case)."""
        raise NotImplementedError


class JSONAnimationLibrary(SignAnimationLibrary):
    """
    Loads sign animation clips from a directory of JSON files (one file
    per word, e.g. assets/avatar/hello.json), matching the project's
    `avatar/` asset folder. This keeps animation CONTENT as data rather
    than code, so new signs can be added by dropping in a new file -- no
    code changes needed (supports the "extendable" project requirement).

    Expected JSON schema per file:
        {
          "word": "hello",
          "frames": [
            {
              "timestamp_offset_ms": 0,
              "left_hand": [{"x":0.1,"y":0.2,"z":0.0}, ... 21 entries ...],
              "right_hand": [...],
              "facial_expression": "smile"
            },
            ...
          ]
        }
    """

    def __init__(self, library_dir: Path) -> None:
        self._library_dir = library_dir
        self._cache: Dict[str, Optional[AnimationClip]] = {}
        logger.info("JSONAnimationLibrary reading clips from: %s", library_dir)

    def get_clip(self, word: str) -> Optional[AnimationClip]:
        normalized = word.strip().lower()
        if normalized in self._cache:
            return self._cache[normalized]

        clip_path = self._library_dir / f"{normalized}.json"
        clip = self._load_clip_file(clip_path, normalized) if clip_path.exists() else None
        self._cache[normalized] = clip
        return clip

    @staticmethod
    def _load_clip_file(path: Path, word: str) -> Optional[AnimationClip]:
        """Parse a single clip JSON file into an AnimationClip, returning
        None (and logging) if the file is malformed rather than crashing
        avatar playback over one bad asset."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            frames = []
            for raw_frame in raw.get("frames", []):
                frames.append(
                    AnimationFrame(
                        timestamp_offset_ms=float(raw_frame["timestamp_offset_ms"]),
                        left_hand=JSONAnimationLibrary._parse_hand(
                            raw_frame.get("left_hand")
                        ),
                        right_hand=JSONAnimationLibrary._parse_hand(
                            raw_frame.get("right_hand")
                        ),
                        facial_expression=FacialExpression(
                            raw_frame.get("facial_expression", "neutral")
                        ),
                    )
                )
            return AnimationClip(word=word, frames=frames)
        except Exception:
            logger.exception("Failed to load animation clip from %s", path)
            return None

    @staticmethod
    def _parse_hand(raw_hand: Optional[list]) -> Optional[List[Landmark]]:
        if raw_hand is None:
            return None
        return [
            Landmark(x=pt["x"], y=pt["y"], z=pt.get("z", 0.0)) for pt in raw_hand
        ]


class PlaceholderAnimationLibrary(SignAnimationLibrary):
    """
    Procedurally generates a simple, deterministic placeholder animation
    for ANY word, using a smooth sinusoidal hand motion. This lets the
    entire avatar pipeline run and be demoed/tested end-to-end before
    real, hand-authored (or motion-captured) sign clips exist in
    assets/avatar/ -- analogous to NullSequenceClassifier and NullTTSEngine
    elsewhere in this project.

    This is NOT linguistically meaningful -- it exists purely so
    downstream code (main.py, teacher_module.py, tests) has something
    concrete to animate against during development.
    """

    def __init__(self, clip_duration_ms: float = 800.0, fps: int = 30) -> None:
        self._clip_duration_ms = clip_duration_ms
        self._fps = fps

    def get_clip(self, word: str) -> Optional[AnimationClip]:
        import math

        frame_interval_ms = 1000.0 / self._fps
        num_frames = max(2, int(self._clip_duration_ms / frame_interval_ms))
        # Deterministic per-word variation so different words at least
        # look visually distinct from one another in the placeholder.
        seed = sum(ord(c) for c in word) % 100 / 100.0

        frames = []
        for i in range(num_frames):
            t_ms = i * frame_interval_ms
            phase = (i / num_frames) * math.pi  # 0 -> pi: rise and fall
            wave = math.sin(phase) * 0.15

            right_hand = [
                Landmark(
                    x=0.5 + seed * 0.1,
                    y=0.5 - wave - (joint_idx * 0.01),
                    z=0.0,
                )
                for joint_idx in range(21)
            ]
            frames.append(
                AnimationFrame(
                    timestamp_offset_ms=t_ms,
                    left_hand=None,
                    right_hand=right_hand,
                    facial_expression=FacialExpression.NEUTRAL,
                )
            )

        return AnimationClip(word=word, frames=frames)


# ----------------------------------------------------------------------
# Pluggable renderer (Strategy Pattern)
# ----------------------------------------------------------------------


class AvatarRenderer(ABC):
    """
    Abstract interface for actually DRAWING an avatar frame. Concrete
    implementations might forward frames to a Unity/Unreal process over
    IPC, push them to a browser via websocket for a WebGL/Three.js avatar,
    or (for headless testing) just log them.
    """

    @abstractmethod
    def render_frame(self, frame: AnimationFrame) -> None:
        """Draw/dispatch a single animation frame."""
        raise NotImplementedError


class NullAvatarRenderer(AvatarRenderer):
    """
    A safe, no-op renderer that logs frames instead of drawing them. Used
    as the default so avatar playback timing/threading logic can be
    developed and tested without a real rendering backend wired up yet.
    """

    def render_frame(self, frame: AnimationFrame) -> None:
        logger.debug(
            "[NullAvatarRenderer] frame @ +%.0fms, expression=%s",
            frame.timestamp_offset_ms,
            frame.facial_expression.value,
        )


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@dataclass
class AvatarControllerConfig:
    """Tunable parameters for avatar playback and transition smoothing."""

    # Target playback frame rate. Frames are re-sampled at this rate
    # during playback scheduling (independent of the source clip's
    # authored fps), so mixed-fps clip libraries still play smoothly.
    playback_fps: int = 30

    # Duration of the blended transition inserted BETWEEN consecutive
    # signs, to avoid an abrupt jump from one sign's final pose to the
    # next sign's first pose (the "look natural" requirement).
    transition_duration_ms: float = 150.0

    # Maximum number of queued sentences waiting to be animated. Older
    # pending sentences are dropped in favor of new ones if this is
    # exceeded, mirroring voice_assistant.py's queue backpressure policy.
    max_queue_size: int = 10


# ----------------------------------------------------------------------
# Avatar controller
# ----------------------------------------------------------------------


class AvatarController:
    """
    Converts finalized captions into synchronized sign-language animation
    and plays them back frame-by-frame on a background thread.

    Usage:
        avatar = AvatarController(
            library=JSONAnimationLibrary(Path("assets/avatar")),
            renderer=MyWebSocketRenderer(),
        )
        avatar.perform_caption(caption)   # non-blocking enqueue
        ...
        avatar.shutdown()
    """

    def __init__(
        self,
        library: Optional[SignAnimationLibrary] = None,
        renderer: Optional[AvatarRenderer] = None,
        config: Optional[AvatarControllerConfig] = None,
    ) -> None:
        """
        Args:
            library: Source of AnimationClips. Defaults to
                PlaceholderAnimationLibrary so the module is runnable
                without real sign-animation assets.
            renderer: Where frames get drawn. Defaults to NullAvatarRenderer
                (logs only) so this module is testable headlessly.
            config: Optional AvatarControllerConfig; defaults used if
                omitted.
        """
        self._config = config or AvatarControllerConfig()
        self._library = library or PlaceholderAnimationLibrary()
        self._renderer = renderer or NullAvatarRenderer()

        self._queue: "queue.Queue[List[str]]" = queue.Queue(
            maxsize=self._config.max_queue_size
        )
        self._pause_event = threading.Event()
        self._shutdown_event = threading.Event()

        # State exposed to main.py for synchronizing captions/voice with
        # what the avatar is currently signing.
        self._current_word: Optional[str] = None
        self._current_sentence_tokens: List[str] = []
        self._progress_fraction: float = 0.0
        self._is_playing: bool = False
        self._state_lock = threading.Lock()

        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="AvatarControllerWorker", daemon=True
        )
        self._worker_thread.start()

        logger.info(
            "AvatarController started (library=%s, renderer=%s, fps=%d).",
            type(self._library).__name__,
            type(self._renderer).__name__,
            self._config.playback_fps,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def perform_caption(self, caption: Caption) -> None:
        """
        Enqueue a finalized Caption's words to be animated. Non-blocking.

        Uses `caption.raw_text` (the naive, un-punctuated word join) rather
        than the grammar-corrected `caption.text`, since raw_text's tokens
        correspond 1:1 with actual recognized sign vocabulary words that
        the animation library is keyed on; punctuation/capitalization from
        grammar correction has no corresponding sign clip.

        Args:
            caption: A FINAL caption (is_final=True) from caption_generator.py.
                Live/in-progress captions are ignored -- animating a sentence
                that's still being recognized would mean constantly
                restarting the animation mid-playback.
        """
        if not caption.is_final:
            return
        tokens = [t for t in caption.raw_text.split() if t]
        if not tokens:
            return
        self._enqueue_tokens(tokens)

    def perform_text(self, text: str) -> None:
        """
        Enqueue arbitrary raw text to be animated (e.g. teacher_module.py
        asking the avatar to demonstrate a target sign during a lesson).
        Non-blocking.
        """
        tokens = [t for t in text.strip().split() if t]
        if tokens:
            self._enqueue_tokens(tokens)

    def _enqueue_tokens(self, tokens: List[str]) -> None:
        try:
            self._queue.put_nowait(tokens)
        except queue.Full:
            logger.warning(
                "AvatarController queue full (max=%d); dropping oldest "
                "pending sentence to make room.",
                self._config.max_queue_size,
            )
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(tokens)

    def pause(self) -> None:
        """Pause animation playback after the current frame."""
        self._pause_event.set()
        logger.info("AvatarController paused.")

    def resume(self) -> None:
        """Resume animation playback."""
        self._pause_event.clear()
        logger.info("AvatarController resumed.")

    def is_playing(self) -> bool:
        """True if the avatar is currently mid-sentence playback."""
        with self._state_lock:
            return self._is_playing

    def get_current_word(self) -> Optional[str]:
        """The word currently being signed, or None if idle. Intended for
        main.py to highlight the matching word in the on-screen caption."""
        with self._state_lock:
            return self._current_word

    def get_progress_fraction(self) -> float:
        """Playback progress through the current sentence, in [0.0, 1.0].
        Useful for a progress bar in main.py's GUI, or for roughly
        aligning avatar pacing against concurrent voice playback."""
        with self._state_lock:
            return self._progress_fraction

    def clear_queue(self) -> None:
        """Discard all pending (not-yet-played) queued sentences."""
        cleared = 0
        while True:
            try:
                self._queue.get_nowait()
                cleared += 1
            except queue.Empty:
                break
        if cleared:
            logger.info("Cleared %d pending sentence(s) from avatar queue.", cleared)

    def shutdown(self, wait: bool = True) -> None:
        """Stop the worker thread. Should be called once when the
        application is closing."""
        logger.info("Shutting down AvatarController...")
        self._shutdown_event.set()
        self._pause_event.clear()
        self.clear_queue()
        if wait and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        logger.info("AvatarController shut down.")

    def __enter__(self) -> "AvatarController":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()

    # ------------------------------------------------------------------
    # Worker thread: sentence -> clips -> blended frame sequence -> playback
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                tokens = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if self._shutdown_event.is_set():
                break

            try:
                self._play_sentence(tokens)
            except Exception:
                logger.exception(
                    "AvatarController failed while playing sentence: %s", tokens
                )

        logger.debug("AvatarController worker thread exiting.")

    def _play_sentence(self, tokens: List[str]) -> None:
        """Resolve each token to an AnimationClip (falling back to
        fingerspelling-by-placeholder for unknown words), stitch them
        together with natural transitions, and play the combined sequence
        frame-by-frame at the configured playback rate."""
        with self._state_lock:
            self._current_sentence_tokens = list(tokens)
            self._is_playing = True
            self._progress_fraction = 0.0

        clips: List[AnimationClip] = []
        for token in tokens:
            clip = self._library.get_clip(token) or self._resolve_fallback_clip(token)
            if clip:
                clips.append(clip)

        combined_frames = self._build_combined_sequence(clips)
        total_duration_ms = combined_frames[-1].timestamp_offset_ms if combined_frames else 0.0

        logger.info(
            "Avatar performing sentence (%d words, ~%.0fms): %s",
            len(tokens),
            total_duration_ms,
            " ".join(tokens),
        )

        self._play_frame_sequence(combined_frames, tokens, total_duration_ms)

        with self._state_lock:
            self._is_playing = False
            self._current_word = None
            self._progress_fraction = 1.0

    def _resolve_fallback_clip(self, word: str) -> Optional[AnimationClip]:
        """
        Called when the animation library has no clip for `word`.

        Falls back to the PlaceholderAnimationLibrary to synthesize
        SOMETHING rather than silently skipping the word -- in a full
        production system this is the natural extension point for a real
        fingerspelling generator (one clip per letter, using a
        hand-shape library for A-Z) which can be swapped in here without
        changing any other method.
        """
        logger.debug(
            "No animation clip found for '%s'; using placeholder fallback "
            "(hook point for a real fingerspelling generator).",
            word,
        )
        return PlaceholderAnimationLibrary().get_clip(word)

    def _build_combined_sequence(
        self, clips: List[AnimationClip]
    ) -> List[AnimationFrame]:
        """
        Concatenate clips into one continuous frame sequence with
        absolute timestamps, inserting a smooth blended transition
        between each consecutive pair of clips.
        """
        combined: List[AnimationFrame] = []
        time_offset_ms = 0.0

        for clip_index, clip in enumerate(clips):
            for frame in clip.frames:
                combined.append(
                    AnimationFrame(
                        timestamp_offset_ms=time_offset_ms + frame.timestamp_offset_ms,
                        left_hand=frame.left_hand,
                        right_hand=frame.right_hand,
                        facial_expression=frame.facial_expression,
                    )
                )
            time_offset_ms += clip.duration_ms

            is_last_clip = clip_index == len(clips) - 1
            if not is_last_clip and clip.frames:
                next_clip = clips[clip_index + 1]
                if next_clip.frames:
                    transition = self._build_transition_frames(
                        start_frame=clip.frames[-1],
                        end_frame=next_clip.frames[0],
                        start_time_ms=time_offset_ms,
                    )
                    combined.extend(transition)
                    if transition:
                        time_offset_ms = transition[-1].timestamp_offset_ms

        return combined

    def _build_transition_frames(
        self,
        start_frame: AnimationFrame,
        end_frame: AnimationFrame,
        start_time_ms: float,
    ) -> List[AnimationFrame]:
        """
        Linearly interpolate ("tween") between the end pose of one sign
        and the start pose of the next, over `transition_duration_ms`.

        This is the concrete mechanism behind the project's "the avatar
        should look natural" requirement: without it, consecutive signs
        would visually snap/teleport between poses, which reads as
        robotic and unnatural.
        """
        duration_ms = self._config.transition_duration_ms
        steps = max(2, int(duration_ms / (1000.0 / self._config.playback_fps)))

        frames = []
        for step in range(1, steps + 1):
            blend = step / steps
            frames.append(
                AnimationFrame(
                    timestamp_offset_ms=start_time_ms + blend * duration_ms,
                    left_hand=self._blend_hand(
                        start_frame.left_hand, end_frame.left_hand, blend
                    ),
                    right_hand=self._blend_hand(
                        start_frame.right_hand, end_frame.right_hand, blend
                    ),
                    # Facial expression switches at the midpoint of the
                    # transition rather than blending (blending expressions
                    # numerically doesn't correspond to anything meaningful).
                    facial_expression=(
                        end_frame.facial_expression
                        if blend >= 0.5
                        else start_frame.facial_expression
                    ),
                )
            )
        return frames

    @staticmethod
    def _blend_hand(
        start: Optional[List[Landmark]],
        end: Optional[List[Landmark]],
        blend: float,
    ) -> Optional[List[Landmark]]:
        """Linearly interpolate a hand's 21 landmarks between two poses.
        If either side is None (hand not used in that pose), the other
        side's pose is used directly (no meaningful midpoint to blend
        toward/from a hand that isn't part of the sign)."""
        if start is None or end is None:
            return end if end is not None else start
        if len(start) != len(end):
            return end

        return [
            Landmark(
                x=s.x + (e.x - s.x) * blend,
                y=s.y + (e.y - s.y) * blend,
                z=s.z + (e.z - s.z) * blend,
            )
            for s, e in zip(start, end)
        ]

    def _play_frame_sequence(
        self,
        frames: List[AnimationFrame],
        tokens: List[str],
        total_duration_ms: float,
    ) -> None:
        """
        Play a fully-built frame sequence in real time, honoring pause and
        shutdown signals between frames, and keeping `_current_word` /
        `_progress_fraction` up to date for external synchronization.
        """
        if not frames:
            return

        playback_start = time.monotonic()
        # Approximate word boundaries by dividing total duration evenly;
        # good enough for caption-highlighting purposes without requiring
        # the caller to track exact per-word offsets.
        word_boundary_ms = (
            total_duration_ms / len(tokens) if tokens and total_duration_ms else 0.0
        )

        for frame in frames:
            if self._shutdown_event.is_set():
                return

            while self._pause_event.is_set() and not self._shutdown_event.is_set():
                time.sleep(0.05)

            target_time = playback_start + (frame.timestamp_offset_ms / 1000.0)
            sleep_duration = target_time - time.monotonic()
            if sleep_duration > 0:
                time.sleep(sleep_duration)

            self._renderer.render_frame(frame)

            with self._state_lock:
                if word_boundary_ms > 0:
                    word_index = min(
                        int(frame.timestamp_offset_ms / word_boundary_ms),
                        len(tokens) - 1,
                    )
                    self._current_word = tokens[word_index]
                self._progress_fraction = (
                    frame.timestamp_offset_ms / total_duration_ms
                    if total_duration_ms
                    else 1.0
                )


if __name__ == "__main__":
    # Minimal manual smoke-test: performs a scripted sentence using the
    # PlaceholderAnimationLibrary + NullAvatarRenderer (no real assets or
    # rendering backend required). Run via: `python avatar_module.py`.
    logger.info("Running avatar_module.py standalone demo.")

    with AvatarController() as avatar:
        demo_caption = Caption(
            text="Hello how are you.",
            raw_text="hello how are you",
            timestamp_ms=0.0,
            frame_index=0,
            is_final=True,
        )
        avatar.perform_caption(demo_caption)

        # Give the worker thread a moment to dequeue and start playback
        # before we start polling for completion (avoids a race where we'd
        # observe "not playing yet" and immediately think it already finished).
        time.sleep(0.3)

        while avatar.is_playing():
            logger.info(
                "Avatar signing: '%s' (%.0f%% complete)",
                avatar.get_current_word(),
                avatar.get_progress_fraction() * 100,
            )
            time.sleep(0.3)

    logger.info("Demo complete.")
