"""
pipeline.py

Shared Sign-Recognition Pipeline

Single Responsibility
----------------------
Owns everything that is IDENTICAL between every front-end this project
ships (main.py's desktop OpenCV GUI, web_app.py's browser GUI, and any
future front-end): constructing every pipeline module from `AppConfig`,
and running one captured frame through
detect -> stabilize -> recognize -> caption -> voice/avatar dispatch.

This module exists specifically to avoid duplicate code: that
orchestration used to live directly inside `main.py`. Once a second
front-end (the web UI) needed the exact same logic, keeping two copies
in sync would have violated this project's core "avoid duplicate code"
rule. Now `main.py` and `web_app.py` are both thin, front-end-specific
shells that own ONLY how a frame gets DISPLAYED (an OpenCV window vs. an
MJPEG stream) and how captions get PRESENTED (a drawn overlay vs. an
HTML/JS panel) -- neither contains any detection/recognition/caption
logic of its own; both call into `SignPipeline`.

This module also owns the `FrameSource` abstraction (camera vs.
synthetic input), since where frames come FROM is likewise identical
across front-ends.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from config import AppConfig
from evaluation import PerformanceMetrics, PerformanceProfiler
from utils.data_types import Caption, HandDetectionResult, RecognitionResult
from utils.logger import get_logger

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# Frame source abstraction (Strategy / Null-Object Pattern)
# ----------------------------------------------------------------------


class FrameSource(ABC):
    """
    Abstract interface for wherever video frames come from. Front-ends
    depend only on this interface, never on cv2.VideoCapture directly, so
    the camera can be swapped for a video file, a different capture
    backend, or a synthetic source for testing.
    """

    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """Return the next BGR frame, or None if the stream has ended /
        a frame couldn't be captured."""
        raise NotImplementedError

    @abstractmethod
    def is_opened(self) -> bool:
        """True if the source is ready to produce frames."""
        raise NotImplementedError

    @abstractmethod
    def release(self) -> None:
        """Release any underlying hardware/resource handles."""
        raise NotImplementedError


class OpenCVCameraSource(FrameSource):
    """Real webcam input via OpenCV's VideoCapture."""

    def __init__(self, camera_index: int = 0) -> None:
        import cv2  # local import: keeps this class's dependency scoped to where it's used

        self._cv2 = cv2
        self._capture = cv2.VideoCapture(camera_index)
        if not self._capture.isOpened():
            logger.error("Could not open camera at index %d.", camera_index)

    def read(self) -> Optional[np.ndarray]:
        success, frame = self._capture.read()
        return frame if success else None

    def is_opened(self) -> bool:
        return self._capture.isOpened()

    def release(self) -> None:
        self._capture.release()


class SyntheticFrameSource(FrameSource):
    """
    Generates synthetic BGR frames (plain noise images) instead of
    reading from real hardware. Used for automated testing, CI, and
    headless demos/documentation where no physical camera is available --
    lets the full pipeline (detection will simply find zero hands on
    these frames, which is a valid, well-handled state per
    hand_detection.py) be exercised end-to-end without requiring camera
    access.
    """

    def __init__(self, width: int = 640, height: int = 480, num_frames: Optional[int] = 100) -> None:
        self._width = width
        self._height = height
        self._num_frames = num_frames
        self._frames_produced = 0
        self._rng = np.random.default_rng(seed=0)

    def read(self) -> Optional[np.ndarray]:
        if self._num_frames is not None and self._frames_produced >= self._num_frames:
            return None
        self._frames_produced += 1
        return self._rng.integers(
            0, 255, size=(self._height, self._width, 3), dtype=np.uint8
        )

    def is_opened(self) -> bool:
        return self._num_frames is None or self._frames_produced < self._num_frames

    def release(self) -> None:
        pass  # nothing to release for a synthetic source


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------


@dataclass
class FrameResult:
    """Everything SignPipeline.process_frame() produces for one frame,
    for a front-end to display however it likes."""

    frame: np.ndarray  # the already-mirrored BGR frame, ready to draw on/encode
    stabilized_result: HandDetectionResult
    recognition_update: RecognitionResult
    caption: Optional[Caption]
    # Always the CURRENT persisted teacher feedback (empty list if no
    # active lesson or no feedback yet), not just "feedback from this
    # exact frame" -- so a front-end can always just display this list
    # without needing to remember the last non-empty value itself.
    teacher_feedback: List[str] = field(default_factory=list)


class SignPipeline:
    """
    Constructs and runs the full detect -> stabilize -> recognize ->
    caption pipeline, shared by every front-end.

    Usage:
        pipeline = SignPipeline(config)
        frame_result = pipeline.process_frame(raw_frame)
        if frame_result.caption:
            pipeline.dispatch_caption(frame_result.caption)
        # ... front-end-specific display of frame_result ...
        pipeline.shutdown()
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._profiler = PerformanceProfiler()
        self._frame_index = 0
        self._voice_muted = False
        self._last_teacher_feedback_messages: List[str] = []

        self._build_modules()

    def _build_modules(self) -> None:
        """Construct every pipeline stage from its corresponding AppConfig
        section. This is the one place in the whole project allowed to
        know about every module at once (the Facade)."""
        from hand_detection import HandDetector
        from overlap_resolution import OverlapResolver
        from continuous_recognition import ContinuousRecognizer
        from caption_generator import CaptionGenerator
        from dataset_manager import DatasetManager

        self._hand_detector = HandDetector(config=self._config.hand_detector)
        self._overlap_resolver = OverlapResolver(config=self._config.overlap_resolver)
        self._dataset_manager = DatasetManager(config=self._config.dataset_manager)

        classifier = self._try_load_trained_classifier()
        self._has_trained_model = classifier is not None
        self._continuous_recognizer = ContinuousRecognizer(
            classifier=classifier, config=self._config.continuous_recognizer
        )
        self._caption_generator = CaptionGenerator(config=self._config.caption_generator)

        self._voice_assistant = None
        if self._config.app.enable_voice_assistant:
            from voice_assistant import VoiceAssistant

            self._voice_assistant = VoiceAssistant(config=self._config.voice_assistant)

        self._avatar_controller = None
        if self._config.app.enable_avatar:
            from avatar_module import AvatarController

            self._avatar_controller = AvatarController(config=self._config.avatar_controller)

        self._teacher_module = None
        if self._config.app.enable_teacher_mode:
            from teacher_module import TeacherModule

            self._teacher_module = TeacherModule(
                avatar=self._avatar_controller, config=self._config.teacher_module
            )

    def _try_load_trained_classifier(self):
        """
        Attempt to load a real trained model (produced by
        model_training.py) for live recognition. Falls back to
        NullSequenceClassifier -- exactly as continuous_recognition.py's
        ContinuousRecognizer already does by default -- if TensorFlow
        isn't installed or no trained model exists yet at
        <models_dir>/current/. This lets every front-end run the full
        pipeline end-to-end before any model has been trained, per the
        project's core requirement.
        """
        model_dir = Path(self._config.paths.models_dir) / "current"
        if not model_dir.exists():
            logger.info(
                "No trained model found at %s; running with no-op recognition "
                "(train one via model_training.py to enable real recognition).",
                model_dir,
            )
            return None  # ContinuousRecognizer defaults to NullSequenceClassifier

        try:
            from model_training import TrainedSequenceClassifier

            return TrainedSequenceClassifier.load(
                model_dir, window_size=self._config.continuous_recognizer.window_size
            )
        except Exception:
            logger.exception(
                "Failed to load trained model from %s; falling back to "
                "no-op recognition.",
                model_dir,
            )
            return None

    # ------------------------------------------------------------------
    # Public accessors (front-ends read these to render/control state)
    # ------------------------------------------------------------------

    @property
    def has_trained_model(self) -> bool:
        return self._has_trained_model

    @property
    def caption_generator(self):
        return self._caption_generator

    @property
    def teacher_module(self):
        return self._teacher_module

    @property
    def voice_assistant(self):
        return self._voice_assistant

    @property
    def avatar_controller(self):
        return self._avatar_controller

    @property
    def continuous_recognizer(self):
        return self._continuous_recognizer

    @property
    def is_voice_muted(self) -> bool:
        return self._voice_muted

    def get_performance_metrics(self) -> Optional[PerformanceMetrics]:
        return self._profiler.get_metrics()

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def process_frame(self, frame_bgr: np.ndarray) -> FrameResult:
        """
        Mirror, detect, stabilize, recognize, and caption one frame.

        Deliberately does NOT dispatch the resulting caption to
        voice/avatar (see dispatch_caption()) or to anything
        front-end-specific like a chatbot -- callers decide what to do
        with a finalized caption, since different front-ends want
        different things (e.g. web_app.py additionally routes final
        captions to chatbot_module.py, which this shared module has no
        business knowing about).

        Args:
            frame_bgr: A raw BGR frame from any FrameSource.

        Returns:
            A FrameResult with the already-mirrored frame plus every
            pipeline stage's output.
        """
        # Mirror the frame horizontally before doing anything else with it.
        # This serves two purposes at once:
        #   1. A natural "selfie view" -- the displayed image moves the way
        #      a mirror would, which is what people intuitively expect from
        #      a front-facing camera preview.
        #   2. CORRECTNESS of left/right hand labeling: MediaPipe's Hand
        #      Landmarker explicitly documents that its handedness output
        #      assumes the input image is mirrored (i.e. captured with a
        #      front-facing/selfie camera and flipped horizontally). Since
        #      a raw cv2.VideoCapture frame is NOT mirrored, detecting on
        #      the unflipped frame reliably reports your right hand as
        #      "Left" and vice versa. Flipping once here, before detection,
        #      fixes it -- and since every downstream stage (drawing,
        #      captions, avatar) only ever sees this already-flipped frame,
        #      nothing else needs to change.
        frame = np.ascontiguousarray(frame_bgr[:, ::-1, :])

        with self._profiler.measure():
            raw_result = self._hand_detector.detect(frame, frame_index=self._frame_index)
            stabilized_result = self._overlap_resolver.resolve(raw_result)
            recognition_update = self._continuous_recognizer.process(stabilized_result)
            caption = self._caption_generator.process(recognition_update)

            if self._teacher_module is not None and self._teacher_module.get_target_word():
                feedback = self._teacher_module.evaluate_attempt(stabilized_result)
                if feedback is not None:
                    self._last_teacher_feedback_messages = feedback.summary_messages()

        self._frame_index += 1

        return FrameResult(
            frame=frame,
            stabilized_result=stabilized_result,
            recognition_update=recognition_update,
            caption=caption,
            teacher_feedback=list(self._last_teacher_feedback_messages),
        )

    def dispatch_caption(self, caption: Caption) -> None:
        """Route a caption to voice + avatar output (the outputs every
        front-end wants identically). Respects mute state. Both are
        optional per config, and both are non-blocking (their own
        background threads handle playback), so this never stalls a
        video loop."""
        if self._voice_assistant is not None and not self._voice_muted:
            self._voice_assistant.speak_caption(caption)
        if self._avatar_controller is not None:
            self._avatar_controller.perform_caption(caption)

    def trigger_demo_caption(self, text: str = "hello this is a test") -> Caption:
        """
        Build and dispatch a synthetic FINAL caption, bypassing
        recognition entirely -- lets any front-end verify voice/avatar
        output is actually working before a real model has been trained
        (recognition alone never produces a caption until then, which
        otherwise looks identical to voice/captions being broken).
        """
        demo_caption = Caption(
            text=text[0].upper() + text[1:] + ".",
            raw_text=text,
            timestamp_ms=time.time() * 1000.0,
            frame_index=self._frame_index,
            is_final=True,
        )
        logger.info("Triggering demo caption: '%s'", demo_caption.text)
        self.dispatch_caption(demo_caption)
        return demo_caption

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def toggle_voice_mute(self) -> bool:
        """Mute/unmute the voice assistant (the recognition/caption
        pipeline keeps running either way). Returns the new mute state."""
        self._voice_muted = not self._voice_muted
        if self._voice_assistant is not None:
            if self._voice_muted:
                self._voice_assistant.pause()
            else:
                self._voice_assistant.resume()
        logger.info("Voice assistant %s.", "muted" if self._voice_muted else "unmuted")
        return self._voice_muted

    def clear_current_sentence(self) -> None:
        """Reset the in-progress recognized sentence and caption state."""
        self._continuous_recognizer.reset()
        self._caption_generator.reset()
        logger.info("Cleared current sentence.")

    def start_teacher_lesson(self, word: str) -> bool:
        """Start a teacher_module.py lesson for `word`, if teacher mode
        is enabled. Returns True if the lesson started successfully."""
        if self._teacher_module is None:
            logger.warning("Teacher mode is not enabled; cannot start a lesson.")
            return False
        self._last_teacher_feedback_messages = []
        return self._teacher_module.start_lesson(word)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Cleanly release every module's resources. Safe to call even if
        no frames were ever processed."""
        logger.info("Shutting down SignPipeline...")

        for name, closer in (
            ("hand_detector", getattr(self._hand_detector, "close", lambda: None)),
            ("voice_assistant", getattr(self._voice_assistant, "shutdown", lambda: None)),
            ("avatar_controller", getattr(self._avatar_controller, "shutdown", lambda: None)),
        ):
            try:
                closer()
            except Exception:
                logger.exception("Error while shutting down %s.", name)

        metrics = self._profiler.get_metrics()
        if metrics:
            logger.info(
                "Session performance: mean=%.1fms, fps=%.1f, frames=%d",
                metrics.mean_latency_ms,
                metrics.fps,
                metrics.num_samples,
            )

        logger.info("SignPipeline shut down.")
