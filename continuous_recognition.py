"""
continuous_recognition.py

MODULE 3 — Continuous Sign Recognition

Single Responsibility
----------------------
This module's ONLY job is: consume a stream of (stabilized) per-frame hand
landmark results and turn them into RECOGNIZED WORDS and, over time, a
SENTENCE — i.e. continuous, sequence-based recognition rather than
single-frame classification (project requirement 4).

It does NOT know about cameras, MediaPipe, occlusion recovery, captions
rendering, grammar correction, or voice output — it only:
  1. Converts each frame's landmarks into a numeric feature vector.
  2. Buffers a sliding window of feature vectors over time.
  3. Feeds full windows to a sequence classifier (pluggable backend).
  4. Applies temporal de-duplication/cooldown so a single held sign isn't
     reported as the same word over and over.
  5. Accumulates recognized words into a growing sentence, and finalizes
     ("completes") that sentence after a configurable pause with no hands
     detected.

Design notes
------------
- STRATEGY PATTERN for the model backend: the actual neural network
  (TensorFlow / PyTorch / ONNX Runtime) is hidden behind the
  `SequenceClassifier` abstract interface. This module only depends on
  that interface, never on a specific ML framework — satisfying the
  Dependency Inversion Principle and keeping this file usable even before
  model_training.py has produced a trained model (via `NullSequenceClassifier`,
  a safe no-op default).
- Feature extraction is deterministic and framework-agnostic: each frame
  becomes a flat vector of both hands' 21 landmarks * (x, y, z) = 126
  values, with a missing hand represented as zeros. This exact same
  feature layout should be used by model_training.py when building
  training data, so the two modules stay in sync (documented clearly here
  as the shared contract).
- Sentence segmentation uses a simple, explainable heuristic (a pause of
  N consecutive no-hand frames ends a sentence) rather than a second ML
  model, keeping this module lightweight and debuggable. This can be
  swapped for a learned segmentation model later without changing the
  public interface (Open/Closed Principle).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np

from utils.data_types import (
    HandDetectionResult,
    Handedness,
    RecognitionResult,
    WordPrediction,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# Number of (x, y, z) values per hand (21 landmarks * 3 coordinates).
LANDMARKS_PER_HAND = 21
FEATURES_PER_HAND = LANDMARKS_PER_HAND * 3
# Total feature vector size per frame: both hands concatenated.
FEATURE_VECTOR_SIZE = FEATURES_PER_HAND * 2  # = 126


# ----------------------------------------------------------------------
# Pluggable sequence classifier interface (Strategy Pattern)
# ----------------------------------------------------------------------


class SequenceClassifier(ABC):
    """
    Abstract interface for any model that can classify a buffered sequence
    of per-frame feature vectors into a recognized word/sign.

    Concrete implementations (e.g. a TensorFlow LSTM/Transformer loaded
    from model_training.py's checkpoints, or an ONNX Runtime session for
    edge deployment) live OUTSIDE this file and are injected into
    ContinuousRecognizer at construction time. This module never imports
    TensorFlow/PyTorch/ONNX directly, keeping it lightweight and testable
    without any ML framework installed.
    """

    @abstractmethod
    def predict(self, sequence: np.ndarray) -> Tuple[str, float]:
        """
        Classify a buffered window of frames.

        Args:
            sequence: A numpy array of shape (window_size, FEATURE_VECTOR_SIZE)
                containing the per-frame feature vectors, oldest frame first.

        Returns:
            A (predicted_word, confidence) tuple. `predicted_word` should be
            an empty string "" if the model predicts "no sign" / background
            class (recommended: always include an explicit background/idle
            class in training so the model can signal "nothing recognized"
            rather than being forced to guess).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def required_window_size(self) -> int:
        """The exact number of frames this classifier expects per call."""
        raise NotImplementedError


class NullSequenceClassifier(SequenceClassifier):
    """
    A safe, no-op default classifier that never recognizes anything.

    This lets ContinuousRecognizer (and the rest of the pipeline in
    main.py) run end-to-end and be tested BEFORE a real model has been
    trained via model_training.py — a real classifier can be swapped in
    later with zero changes to this module. This follows the Null Object
    Pattern, avoiding scattered `if model is None` checks throughout the
    recognition logic.
    """

    def __init__(self, window_size: int = 30) -> None:
        self._window_size = window_size
        logger.warning(
            "ContinuousRecognizer is using NullSequenceClassifier — "
            "no real model is loaded, so no words will be recognized. "
            "Inject a trained SequenceClassifier (see model_training.py) "
            "for real recognition."
        )

    def predict(self, sequence: np.ndarray) -> Tuple[str, float]:
        return "", 0.0

    @property
    def required_window_size(self) -> int:
        return self._window_size


# ----------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------


class LandmarkFeatureExtractor:
    """
    Converts a single frame's HandDetectionResult into a fixed-size,
    numeric feature vector suitable for sequence modeling.

    This is a standalone class (rather than a free function) so that more
    elaborate feature engineering (e.g. normalizing relative to wrist
    position, or adding finger-angle features) can be added later behind
    the same `extract()` interface without touching ContinuousRecognizer.
    """

    @staticmethod
    def extract(result: HandDetectionResult) -> Tuple[np.ndarray, float]:
        """
        Build a flat feature vector for one frame.

        Layout (fixed, must match model_training.py's training data):
            [0:63]   = LEFT hand's 21 landmarks * (x, y, z), zeros if absent
            [63:126] = RIGHT hand's 21 landmarks * (x, y, z), zeros if absent

        Args:
            result: A single frame's (ideally already stabilized by
                overlap_resolution.py) HandDetectionResult.

        Returns:
            A tuple of:
              - feature vector, shape (FEATURE_VECTOR_SIZE,), dtype float32
              - estimated_fraction: fraction of the present hands' landmarks
                that were flagged is_estimated (0.0 if no hands present or
                none estimated). Used upstream to track how much a
                prediction relied on occlusion-recovered data.
        """
        features = np.zeros(FEATURE_VECTOR_SIZE, dtype=np.float32)
        estimated_count = 0
        total_landmark_count = 0

        for handedness, offset in (
            (Handedness.LEFT, 0),
            (Handedness.RIGHT, FEATURES_PER_HAND),
        ):
            hand = result.get_hand(handedness)
            if hand is None:
                continue

            for i, lm in enumerate(hand.landmarks[:LANDMARKS_PER_HAND]):
                base = offset + i * 3
                features[base] = lm.x
                features[base + 1] = lm.y
                features[base + 2] = lm.z
                total_landmark_count += 1
                if lm.is_estimated:
                    estimated_count += 1

        estimated_fraction = (
            estimated_count / total_landmark_count if total_landmark_count else 0.0
        )
        return features, estimated_fraction


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@dataclass
class ContinuousRecognizerConfig:
    """Tunable parameters governing buffering, thresholds, and sentence
    segmentation behaviour."""

    # Number of frames the classifier expects per prediction. Should match
    # the injected SequenceClassifier's required_window_size, but is kept
    # here too so ContinuousRecognizer can validate consistency at init.
    window_size: int = 30

    # How many NEW frames must arrive before we re-run the classifier
    # (a "sliding window with stride"). stride=1 re-predicts every frame
    # (most responsive, most compute); larger strides reduce compute at
    # the cost of latency.
    stride: int = 5

    # Minimum confidence for a prediction to be accepted as a real word.
    min_confidence: float = 0.75

    # Minimum number of frames that must pass after accepting a word
    # before the SAME word can be accepted again. Prevents a single held
    # sign from being reported as many repeated words.
    repeat_cooldown_frames: int = 20

    # Number of consecutive frames with NO hands detected at all before we
    # consider the current sentence "finished" (the signer paused/stopped).
    sentence_end_no_hand_frames: int = 45

    # Predictions whose window relied on estimated (occluded) landmarks for
    # more than this fraction are discarded, since they're more likely to
    # be inaccurate (project requirement: gracefully handle occlusion
    # without propagating bad guesses into the recognized sentence).
    max_estimated_frame_ratio: float = 0.5


# ----------------------------------------------------------------------
# Continuous recognizer
# ----------------------------------------------------------------------


class ContinuousRecognizer:
    """
    Buffers landmark sequences over time and produces recognized words and
    growing sentences, using an injected SequenceClassifier for the actual
    per-window classification.

    Usage:
        classifier = MyTrainedSequenceClassifier(...)  # from elsewhere
        recognizer = ContinuousRecognizer(classifier=classifier)

        for stabilized_result in stream_of_frames:
            update = recognizer.process(stabilized_result)
            if update.new_word:
                print("Recognized:", update.new_word.word)
            if update.is_sentence_complete:
                print("Sentence:", update.sentence_text())
    """

    def __init__(
        self,
        classifier: Optional[SequenceClassifier] = None,
        config: Optional[ContinuousRecognizerConfig] = None,
    ) -> None:
        """
        Args:
            classifier: A SequenceClassifier implementation. If omitted,
                falls back to NullSequenceClassifier (recognizes nothing,
                but keeps the pipeline runnable before a model exists).
            config: Optional ContinuousRecognizerConfig; defaults used if
                omitted.
        """
        self._config = config or ContinuousRecognizerConfig()
        self._classifier = classifier or NullSequenceClassifier(
            window_size=self._config.window_size
        )

        if self._classifier.required_window_size != self._config.window_size:
            logger.warning(
                "Configured window_size (%d) does not match classifier's "
                "required_window_size (%d); using the classifier's value "
                "to avoid shape mismatches.",
                self._config.window_size,
                self._classifier.required_window_size,
            )
            self._config.window_size = self._classifier.required_window_size

        self._feature_buffer: Deque[np.ndarray] = deque(
            maxlen=self._config.window_size
        )
        self._estimated_flags_buffer: Deque[float] = deque(
            maxlen=self._config.window_size
        )

        self._frames_since_last_prediction_attempt = 0
        self._frames_since_last_accepted_word = 0
        self._frames_with_no_hands = 0
        self._last_accepted_word: Optional[str] = None

        self._sentence_tokens: List[str] = []

        logger.info(
            "ContinuousRecognizer initialized (window_size=%d, stride=%d, "
            "min_confidence=%.2f)",
            self._config.window_size,
            self._config.stride,
            self._config.min_confidence,
        )

    def process(self, result: HandDetectionResult) -> RecognitionResult:
        """
        Process one (stabilized) frame and update recognition state.

        Args:
            result: The frame's HandDetectionResult — ideally the output
                of overlap_resolution.py rather than raw hand_detection.py
                output, so occlusion has already been smoothed/recovered.

        Returns:
            A RecognitionResult describing the current sentence state,
            with `new_word` populated only on frames where a fresh word
            was just accepted, and `is_sentence_complete` True only on the
            frame where a pause finalized the sentence.
        """
        try:
            return self._process_impl(result)
        except Exception:
            logger.exception(
                "ContinuousRecognizer failed on frame_index=%d; "
                "returning current sentence state unchanged.",
                result.frame_index,
            )
            return RecognitionResult(
                frame_index=result.frame_index,
                timestamp_ms=result.timestamp_ms,
                sentence_tokens=list(self._sentence_tokens),
            )

    def _process_impl(self, result: HandDetectionResult) -> RecognitionResult:
        update = RecognitionResult(
            frame_index=result.frame_index,
            timestamp_ms=result.timestamp_ms,
            sentence_tokens=list(self._sentence_tokens),
        )

        # --- Track hand presence for sentence-pause segmentation ---
        if result.num_hands() == 0:
            self._frames_with_no_hands += 1
        else:
            self._frames_with_no_hands = 0

        # --- Feature extraction + buffering ---
        features, estimated_fraction = LandmarkFeatureExtractor.extract(result)
        self._feature_buffer.append(features)
        self._estimated_flags_buffer.append(estimated_fraction)

        self._frames_since_last_prediction_attempt += 1
        self._frames_since_last_accepted_word += 1

        buffer_full = len(self._feature_buffer) == self._config.window_size
        stride_elapsed = (
            self._frames_since_last_prediction_attempt >= self._config.stride
        )

        if buffer_full and stride_elapsed:
            self._frames_since_last_prediction_attempt = 0
            new_word = self._attempt_prediction(result.timestamp_ms)
            if new_word is not None:
                self._sentence_tokens.append(new_word.word)
                update.sentence_tokens = list(self._sentence_tokens)
                update.new_word = new_word
                self._frames_since_last_accepted_word = 0
                self._last_accepted_word = new_word.word
                logger.info(
                    "Recognized word '%s' (confidence=%.2f) at frame %d.",
                    new_word.word,
                    new_word.confidence,
                    result.frame_index,
                )

        # --- Sentence-pause segmentation ---
        if (
            self._frames_with_no_hands >= self._config.sentence_end_no_hand_frames
            and self._sentence_tokens
        ):
            logger.info(
                "Sentence finalized after %d frames of no hands: '%s'",
                self._frames_with_no_hands,
                " ".join(self._sentence_tokens),
            )
            update.sentence_tokens = list(self._sentence_tokens)
            update.is_sentence_complete = True
            self._start_new_sentence()

        return update

    def _attempt_prediction(self, timestamp_ms: float) -> Optional[WordPrediction]:
        """
        Run the classifier on the current full buffer and decide whether
        to accept the result as a new recognized word, applying the
        confidence threshold, repeat-cooldown, and occlusion-reliance
        checks.

        Returns:
            A WordPrediction if a new word was accepted, else None.
        """
        sequence = np.stack(list(self._feature_buffer), axis=0)
        word, confidence = self._classifier.predict(sequence)

        if not word:
            # Classifier predicted "background"/idle — nothing to do.
            return None

        if confidence < self._config.min_confidence:
            logger.debug(
                "Prediction '%s' below confidence threshold (%.2f < %.2f); discarding.",
                word,
                confidence,
                self._config.min_confidence,
            )
            return None

        avg_estimated_ratio = float(np.mean(self._estimated_flags_buffer))
        if avg_estimated_ratio > self._config.max_estimated_frame_ratio:
            logger.debug(
                "Prediction '%s' relied too heavily on estimated/occluded "
                "landmarks (%.0f%% of window); discarding to avoid "
                "propagating an unreliable guess.",
                word,
                avg_estimated_ratio * 100,
            )
            return None

        is_repeat = word == self._last_accepted_word
        cooldown_active = (
            self._frames_since_last_accepted_word < self._config.repeat_cooldown_frames
        )
        if is_repeat and cooldown_active:
            logger.debug(
                "Prediction '%s' is a repeat within cooldown window "
                "(%d/%d frames); discarding.",
                word,
                self._frames_since_last_accepted_word,
                self._config.repeat_cooldown_frames,
            )
            return None

        return WordPrediction(
            word=word,
            confidence=confidence,
            timestamp_ms=timestamp_ms,
            window_size=self._config.window_size,
            estimated_frame_ratio=avg_estimated_ratio,
        )

    def _start_new_sentence(self) -> None:
        """Reset sentence-building state to begin a fresh utterance,
        WITHOUT discarding the frame/feature buffers (motion tracking
        across a brief pause can remain informative)."""
        self._sentence_tokens = []
        self._last_accepted_word = None

    def get_current_sentence(self) -> str:
        """Public accessor for the sentence built so far (joined with
        spaces, un-punctuated — grammar/punctuation is
        caption_generator.py's job)."""
        return " ".join(self._sentence_tokens)

    def reset(self) -> None:
        """
        Fully reset all recognition state: buffers, sentence, cooldowns.

        Should be called when starting a new video session, or when the
        user explicitly asks to clear the current sentence (e.g. a UI
        "clear" button in main.py).
        """
        self._feature_buffer.clear()
        self._estimated_flags_buffer.clear()
        self._frames_since_last_prediction_attempt = 0
        self._frames_since_last_accepted_word = 0
        self._frames_with_no_hands = 0
        self._last_accepted_word = None
        self._sentence_tokens = []
        logger.info("ContinuousRecognizer state fully reset.")


if __name__ == "__main__":
    # Minimal manual smoke-test chaining all three modules built so far.
    # Run via: `python continuous_recognition.py`. Requires a webcam.
    # Uses NullSequenceClassifier, so no real words will be recognized —
    # this demo only verifies the pipeline runs end-to-end without errors
    # and shows buffering/hand-presence behaviour in the console.
    import cv2

    from hand_detection import HandDetector
    from overlap_resolution import OverlapResolver

    logger.info(
        "Running continuous_recognition.py standalone demo "
        "(NullSequenceClassifier — no real recognition). Press 'q' to quit."
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam for demo.")
    else:
        resolver = OverlapResolver()
        recognizer = ContinuousRecognizer()
        with HandDetector() as detector:
            frame_idx = 0
            while True:
                success, frame = cap.read()
                if not success:
                    logger.warning("Failed to read frame from webcam.")
                    break

                raw_result = detector.detect(frame, frame_index=frame_idx)
                stable_result = resolver.resolve(raw_result)
                update = recognizer.process(stable_result)
                frame_idx += 1

                cv2.putText(
                    frame,
                    f"Sentence: {update.sentence_text()}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow("continuous_recognition.py demo", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()
