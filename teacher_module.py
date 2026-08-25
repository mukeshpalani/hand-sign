"""
teacher_module.py

MODULE 5 -- Teacher Module

Single Responsibility
----------------------
This module's ONLY job is: run an interactive sign-language LESSON --
show the learner a target sign, compare their live hand landmarks against
that target, produce a correction score plus specific, actionable
feedback ("Straighten your index finger.", "Rotate your wrist slightly."),
and keep a learning history / progress record (project requirement:
teacher_module.py -- "Teach sign language / Show target sign / Compare
user landmarks / Detect incorrect finger positions / Provide correction
score / Give feedback / Display progress / Keep learning history").

It does NOT detect hands itself, does NOT recognize continuous sentences,
and does NOT render any UI -- it consumes:
  - `HandDetectionResult` (from hand_detection.py, ideally after
    overlap_resolution.py has stabilized it) representing the LEARNER'S
    live attempt, and
  - an `AnimationClip` (from avatar_module.py) representing the TARGET
    sign to compare against.

Design notes
------------
- DELIBERATE, DOCUMENTED CROSS-MODULE REUSE: this module imports
  `Landmark`/`HandLandmarks`/`Handedness` from utils/data_types.py (the
  same shared contract every module uses) AND `AnimationClip` /
  `SignAnimationLibrary` / `AvatarController` from avatar_module.py. This
  is intentional, not a violation of module independence: teaching a sign
  means comparing against exactly the same "target sign" representation
  the avatar uses to perform it, so re-defining a second, parallel target-
  pose format here would be duplicate code (explicitly disallowed by the
  project's coding rules). teacher_module.py still has ONE responsibility
  (lesson orchestration, comparison, scoring, feedback, history) -- it
  just happens to consume, rather than redefine, avatar_module.py's types.
- INTERPRETABLE, RULE-BASED COMPARISON (not a black-box distance number):
  `FingerAnalyzer` computes a per-finger "extension ratio" (how curled vs.
  straight each finger is) and a wrist-rotation angle from landmark
  geometry, and compares each against the target. This is what lets the
  module generate NAMED, ACTIONABLE feedback per finger rather than just
  "72% correct" -- directly satisfying the project's example feedback
  strings.
- STRATEGY-FRIENDLY: `FingerAnalyzer`'s thresholds live in
  `TeacherModuleConfig`, and the comparison logic is isolated in its own
  class so a future, more sophisticated pose-comparison model (e.g. a
  learned similarity metric) could replace it behind the same
  `evaluate_attempt()` call.
- The learning history is kept in-memory here (a `deque`-backed list per
  word); the project's future-extensibility list calls out "personalized
  user profiles" -- this module's `LessonAttempt` records are already
  structured so a future `user_profile` field could be added without
  reshaping anything, and persistence can be added by handing
  `get_history()`'s output to dataset_manager.py or a database later.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

from avatar_module import AnimationClip, AvatarController, SignAnimationLibrary
from utils.data_types import HandDetectionResult, HandLandmarks, Handedness, Landmark
from utils.logger import get_logger

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# Hand topology constants (standard MediaPipe 21-point layout, matching
# the convention documented in utils/data_types.py's HandLandmarks)
# ----------------------------------------------------------------------


class Finger(Enum):
    """The five fingers, used to key per-finger analysis and feedback."""
    THUMB = "thumb"
    INDEX = "index finger"
    MIDDLE = "middle finger"
    RING = "ring finger"
    PINKY = "pinky finger"


WRIST_INDEX = 0

# (MCP, PIP/IP, DIP, TIP) landmark indices per finger. The thumb's joints
# are named differently anatomically (CMC/MCP/IP) but we reuse the same
# 4-tuple shape for uniform processing across all five fingers.
FINGER_LANDMARK_INDICES: Dict[Finger, Tuple[int, int, int, int]] = {
    Finger.THUMB: (1, 2, 3, 4),
    Finger.INDEX: (5, 6, 7, 8),
    Finger.MIDDLE: (9, 10, 11, 12),
    Finger.RING: (13, 14, 15, 16),
    Finger.PINKY: (17, 18, 19, 20),
}

# Landmark used, together with the wrist, to estimate overall wrist/hand
# rotation in the image plane (the middle finger's MCP is a stable,
# central reference point across most hand poses).
WRIST_ORIENTATION_REFERENCE_INDEX = 9  # MIDDLE_MCP


# ----------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------


@dataclass
class FingerFeedback:
    """A single piece of feedback about one finger's positioning."""
    finger: Finger
    user_extension: float
    target_extension: float
    message: str


@dataclass
class LessonFeedback:
    """
    The full result of comparing one learner attempt against a target
    sign: an overall correction score plus specific, per-finger and
    wrist-level feedback messages.
    """
    word: str
    score: float  # 0.0 - 100.0
    is_correct: bool
    finger_feedback: List[FingerFeedback] = field(default_factory=list)
    wrist_feedback: Optional[str] = None
    timestamp_ms: float = 0.0

    def summary_messages(self) -> List[str]:
        """All human-readable feedback strings for this attempt, in a
        natural order (wrist first, then fingers), ready for display or
        for voice_assistant.py to speak aloud."""
        messages = []
        if self.wrist_feedback:
            messages.append(self.wrist_feedback)
        messages.extend(fb.message for fb in self.finger_feedback)
        if not messages:
            messages.append("Great job! Your sign looks correct.")
        return messages


@dataclass
class LessonAttempt:
    """One historical record of a learner attempting a specific sign,
    kept for progress tracking."""
    word: str
    score: float
    is_correct: bool
    timestamp_ms: float


@dataclass
class WordProgress:
    """Aggregated progress statistics for a single word/sign, derived
    from LessonAttempt history."""
    word: str
    attempts: int
    best_score: float
    average_score: float
    is_mastered: bool


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@dataclass
class TeacherModuleConfig:
    """Tunable thresholds governing comparison sensitivity and mastery
    criteria."""

    # Minimum absolute difference in finger extension ratio (roughly,
    # "how curled the finger is", see FingerAnalyzer) before feedback is
    # generated for that finger. Smaller = stricter/more feedback.
    finger_extension_tolerance: float = 0.15

    # Minimum wrist-rotation angle difference (degrees) before wrist
    # feedback is generated.
    wrist_rotation_tolerance_degrees: float = 20.0

    # A score at or above this threshold counts the attempt as "correct".
    correct_score_threshold: float = 80.0

    # A word is considered "mastered" once the average of its last N
    # attempts reaches this threshold.
    mastery_score_threshold: float = 85.0
    mastery_lookback_attempts: int = 5

    # How many historical attempts to retain per word (older ones are
    # dropped), to bound memory in long-running learning sessions.
    max_history_per_word: int = 200


# ----------------------------------------------------------------------
# Finger / pose analysis
# ----------------------------------------------------------------------


class FingerAnalyzer:
    """
    Computes interpretable geometric features from a hand's 21 landmarks
    -- per-finger extension ratios and overall wrist orientation -- used
    to generate specific, named feedback rather than an opaque similarity
    score alone.

    Kept as a standalone class (rather than inline in TeacherModule) so
    it can be unit-tested independently and reused (e.g. by a future
    evaluation.py metric, or dataset_manager.py's data-quality checks).
    """

    @staticmethod
    def compute_extension_ratio(landmarks: List[Landmark], finger: Finger) -> float:
        """
        Compute how EXTENDED (straight) vs. CURLED a finger is.

        Defined as the straight-line distance from the wrist to the
        fingertip, divided by the distance from the wrist to that
        finger's MCP (knuckle) joint. A fully extended finger has a
        ratio well above 1.0; a tightly curled finger (tip folded back
        toward the palm) has a ratio close to or below 1.0. This is
        scale-invariant (works regardless of hand size/distance from
        camera) since it's a ratio of two distances on the same hand.

        Args:
            landmarks: The 21 landmarks of one hand.
            finger: Which finger to analyze.

        Returns:
            The extension ratio (unitless, typically ~0.5 to ~2.0).
        """
        mcp_idx, _, _, tip_idx = FINGER_LANDMARK_INDICES[finger]
        wrist = landmarks[WRIST_INDEX]
        mcp = landmarks[mcp_idx]
        tip = landmarks[tip_idx]

        wrist_to_mcp = FingerAnalyzer._distance(wrist, mcp)
        wrist_to_tip = FingerAnalyzer._distance(wrist, tip)

        if wrist_to_mcp < 1e-6:
            # Degenerate hand geometry (shouldn't normally happen with
            # real detections); avoid a division-by-zero crash.
            return 1.0
        return wrist_to_tip / wrist_to_mcp

    @staticmethod
    def compute_wrist_orientation_degrees(landmarks: List[Landmark]) -> float:
        """
        Estimate the hand's rotation in the image plane, in degrees, as
        the angle of the vector from the wrist to the middle finger's
        MCP joint. This is a simple 2D proxy for wrist rotation -- a full
        3D orientation would need the landmark z-values combined with a
        camera model, which is out of scope for this lightweight,
        explainable comparison.
        """
        wrist = landmarks[WRIST_INDEX]
        reference = landmarks[WRIST_ORIENTATION_REFERENCE_INDEX]
        dx = reference.x - wrist.x
        dy = reference.y - wrist.y
        return math.degrees(math.atan2(dy, dx))

    @staticmethod
    def _distance(a: Landmark, b: Landmark) -> float:
        """Euclidean distance between two landmarks in normalized (x, y)
        space (z is excluded since its scale/units aren't directly
        comparable to x/y for this 2D-plane-based analysis)."""
        return math.hypot(a.x - b.x, a.y - b.y)


# ----------------------------------------------------------------------
# Teacher module
# ----------------------------------------------------------------------


class TeacherModule:
    """
    Orchestrates an interactive sign-teaching session: presents a target
    sign, scores the learner's live attempts against it, generates
    actionable feedback, and tracks progress/history over time.

    Usage:
        teacher = TeacherModule(
            animation_library=JSONAnimationLibrary(Path("assets/avatar")),
            avatar=avatar_controller,   # optional, for visual demonstration
        )
        teacher.start_lesson("hello")

        # In the main video loop, for each stabilized HandDetectionResult:
        feedback = teacher.evaluate_attempt(stabilized_result)
        if feedback:
            print(feedback.score, feedback.summary_messages())
    """

    def __init__(
        self,
        animation_library: Optional[SignAnimationLibrary] = None,
        avatar: Optional[AvatarController] = None,
        config: Optional[TeacherModuleConfig] = None,
    ) -> None:
        """
        Args:
            animation_library: Source of target-sign AnimationClips to
                teach from and compare against. Defaults to
                PlaceholderAnimationLibrary (via AvatarController's own
                default) if omitted -- see _default_library().
            avatar: Optional AvatarController used purely to VISUALLY
                DEMONSTRATE the target sign when a lesson starts (calls
                its perform_text()). Comparison/scoring does not depend
                on this -- it works from animation_library directly, so
                the module is fully usable without an avatar wired up
                (e.g. in a text-only/CLI teaching mode).
            config: Optional TeacherModuleConfig; defaults used if omitted.
        """
        self._config = config or TeacherModuleConfig()
        self._library = animation_library or self._default_library()
        self._avatar = avatar

        self._current_word: Optional[str] = None
        self._current_target_clip: Optional[AnimationClip] = None

        self._history: Dict[str, Deque[LessonAttempt]] = defaultdict(
            lambda: deque(maxlen=self._config.max_history_per_word)
        )

        logger.info(
            "TeacherModule initialized (library=%s, avatar_demo=%s).",
            type(self._library).__name__,
            self._avatar is not None,
        )

    @staticmethod
    def _default_library() -> SignAnimationLibrary:
        """Import locally to avoid a hard dependency at module load time
        if avatar_module's placeholder generator is ever reorganized."""
        from avatar_module import PlaceholderAnimationLibrary

        return PlaceholderAnimationLibrary()

    # ------------------------------------------------------------------
    # Lesson control
    # ------------------------------------------------------------------

    def start_lesson(self, word: str) -> bool:
        """
        Begin teaching a specific sign: loads its target AnimationClip and
        (if an avatar was provided) triggers a visual demonstration.

        Args:
            word: The sign/word to teach, e.g. "hello".

        Returns:
            True if a target clip was found/generated and the lesson
            started successfully; False if no target could be resolved
            for this word (e.g. an empty/invalid word).
        """
        normalized = word.strip().lower()
        if not normalized:
            logger.warning("start_lesson() called with an empty word; ignoring.")
            return False

        clip = self._library.get_clip(normalized)
        if clip is None or not clip.frames:
            logger.warning(
                "No target animation available for '%s'; cannot start lesson.",
                normalized,
            )
            return False

        self._current_word = normalized
        self._current_target_clip = clip

        if self._avatar is not None:
            self._avatar.perform_text(normalized)

        logger.info("Lesson started for word: '%s'", normalized)
        return True

    def get_target_word(self) -> Optional[str]:
        """Returns the word currently being taught, or None if no lesson
        is active."""
        return self._current_word

    def end_lesson(self) -> None:
        """Clear the current lesson state (does not affect history)."""
        self._current_word = None
        self._current_target_clip = None
        logger.info("Lesson ended.")

    # ------------------------------------------------------------------
    # Attempt evaluation
    # ------------------------------------------------------------------

    def evaluate_attempt(
        self, result: HandDetectionResult
    ) -> Optional[LessonFeedback]:
        """
        Compare the learner's current live hand landmarks against the
        active lesson's target pose, producing a score and feedback.

        Args:
            result: The learner's current frame, ideally already
                stabilized by overlap_resolution.py.

        Returns:
            A LessonFeedback if a lesson is active and at least one
            relevant hand was detected; None if there's no active lesson
            or no comparable hand was found this frame (e.g. the
            learner's hand isn't currently visible).
        """
        if self._current_target_clip is None or self._current_word is None:
            logger.debug("evaluate_attempt() called with no active lesson.")
            return None

        try:
            return self._evaluate_impl(result)
        except Exception:
            logger.exception(
                "TeacherModule failed to evaluate attempt for word '%s'.",
                self._current_word,
            )
            return None

    def _evaluate_impl(self, result: HandDetectionResult) -> Optional[LessonFeedback]:
        target_frame = self._current_target_clip.frames[-1]  # canonical held pose

        # Compare whichever hand(s) the target pose actually uses,
        # preferring the right hand (the convention used by
        # PlaceholderAnimationLibrary and typical single-hand signs), then
        # falling back to the left hand. Two-handed sign comparison is a
        # natural extension point (see module docstring) -- currently we
        # score the first comparable hand found, which covers the common
        # one-handed-sign teaching case cleanly.
        for handedness, target_landmarks in (
            (Handedness.RIGHT, target_frame.right_hand),
            (Handedness.LEFT, target_frame.left_hand),
        ):
            if target_landmarks is None:
                continue
            user_hand = result.get_hand(handedness)
            if user_hand is None:
                continue
            return self._compare_hand(user_hand, target_landmarks)

        logger.debug(
            "No comparable hand detected for lesson word '%s' this frame.",
            self._current_word,
        )
        return None

    def _compare_hand(
        self, user_hand: HandLandmarks, target_landmarks: List[Landmark]
    ) -> LessonFeedback:
        """Run the actual per-finger + wrist comparison and assemble the
        LessonFeedback + score, recording it into history."""
        if not user_hand.is_complete() or len(target_landmarks) != 21:
            logger.debug(
                "Incomplete landmark set for word '%s'; skipping comparison "
                "this frame.",
                self._current_word,
            )
            return LessonFeedback(
                word=self._current_word,
                score=0.0,
                is_correct=False,
                wrist_feedback="Hold your hand steady in view of the camera.",
                timestamp_ms=time.time() * 1000.0,
            )

        finger_feedback: List[FingerFeedback] = []
        finger_deviations: List[float] = []

        for finger in Finger:
            user_ext = FingerAnalyzer.compute_extension_ratio(
                user_hand.landmarks, finger
            )
            target_ext = FingerAnalyzer.compute_extension_ratio(
                target_landmarks, finger
            )
            deviation = abs(user_ext - target_ext)
            finger_deviations.append(deviation)

            if deviation > self._config.finger_extension_tolerance:
                message = self._build_finger_message(finger, user_ext, target_ext)
                finger_feedback.append(
                    FingerFeedback(
                        finger=finger,
                        user_extension=user_ext,
                        target_extension=target_ext,
                        message=message,
                    )
                )

        user_wrist_angle = FingerAnalyzer.compute_wrist_orientation_degrees(
            user_hand.landmarks
        )
        target_wrist_angle = FingerAnalyzer.compute_wrist_orientation_degrees(
            target_landmarks
        )
        wrist_deviation = self._angular_difference(
            user_wrist_angle, target_wrist_angle
        )
        wrist_feedback = None
        if wrist_deviation > self._config.wrist_rotation_tolerance_degrees:
            wrist_feedback = "Rotate your wrist slightly."

        score = self._compute_score(finger_deviations, wrist_deviation)
        is_correct = score >= self._config.correct_score_threshold

        feedback = LessonFeedback(
            word=self._current_word,
            score=score,
            is_correct=is_correct,
            finger_feedback=finger_feedback,
            wrist_feedback=wrist_feedback,
            timestamp_ms=time.time() * 1000.0,
        )

        self._record_attempt(feedback)
        return feedback

    @staticmethod
    def _build_finger_message(
        finger: Finger, user_extension: float, target_extension: float
    ) -> str:
        """
        Turn a finger's extension deviation into a specific, actionable
        instruction, matching the style of the project's example
        feedback ("Raise your thumb.", "Straighten your index finger.").
        """
        needs_more_extension = user_extension < target_extension

        if finger == Finger.THUMB:
            return "Raise your thumb." if needs_more_extension else "Lower your thumb."

        finger_name = finger.value
        if needs_more_extension:
            return f"Straighten your {finger_name}."
        return f"Curl your {finger_name} more."

    @staticmethod
    def _angular_difference(angle_a_degrees: float, angle_b_degrees: float) -> float:
        """Smallest absolute difference between two angles in degrees,
        correctly handling wraparound at +/-180 degrees."""
        diff = (angle_a_degrees - angle_b_degrees + 180.0) % 360.0 - 180.0
        return abs(diff)

    def _compute_score(
        self, finger_deviations: List[float], wrist_deviation_degrees: float
    ) -> float:
        """
        Combine per-finger and wrist deviations into a single 0-100
        correction score. Finger accuracy is weighted more heavily than
        wrist rotation, since finger shape is usually the primary
        distinguishing feature between signs.
        """
        # Normalize each finger deviation against the tolerance so a
        # deviation right at the tolerance boundary contributes ~0 score
        # loss, and larger deviations scale down smoothly rather than
        # cutting off sharply.
        finger_penalty = sum(
            min(1.0, dev / max(self._config.finger_extension_tolerance, 1e-6))
            for dev in finger_deviations
        ) / max(len(finger_deviations), 1)

        wrist_penalty = min(
            1.0,
            wrist_deviation_degrees
            / max(self._config.wrist_rotation_tolerance_degrees, 1e-6),
        )

        combined_penalty = (0.8 * finger_penalty) + (0.2 * wrist_penalty)
        score = max(0.0, 100.0 * (1.0 - combined_penalty))
        return round(score, 1)

    # ------------------------------------------------------------------
    # History / progress
    # ------------------------------------------------------------------

    def _record_attempt(self, feedback: LessonFeedback) -> None:
        attempt = LessonAttempt(
            word=feedback.word,
            score=feedback.score,
            is_correct=feedback.is_correct,
            timestamp_ms=feedback.timestamp_ms,
        )
        self._history[feedback.word].append(attempt)
        logger.debug(
            "Recorded attempt for '%s': score=%.1f, correct=%s",
            attempt.word,
            attempt.score,
            attempt.is_correct,
        )

    def get_history(
        self, word: Optional[str] = None, count: Optional[int] = None
    ) -> List[LessonAttempt]:
        """
        Retrieve historical attempts, most-recent-last.

        Args:
            word: If provided, only attempts for this word are returned.
                If None, attempts for ALL words are returned (concatenated
                in per-word insertion order, not globally time-sorted).
            count: If provided, return only the last `count` attempts
                (applied per-word before concatenation if word is None).
        """
        if word is not None:
            attempts = list(self._history.get(word.strip().lower(), []))
            return attempts[-count:] if count else attempts

        all_attempts: List[LessonAttempt] = []
        for attempts in self._history.values():
            recent = list(attempts)[-count:] if count else list(attempts)
            all_attempts.extend(recent)
        return all_attempts

    def get_progress(self, word: str) -> Optional[WordProgress]:
        """
        Compute aggregated progress statistics for a single word.

        Returns:
            A WordProgress summary, or None if the word has never been
            attempted.
        """
        normalized = word.strip().lower()
        attempts = list(self._history.get(normalized, []))
        if not attempts:
            return None

        scores = [a.score for a in attempts]
        recent_scores = scores[-self._config.mastery_lookback_attempts :]
        average_recent = sum(recent_scores) / len(recent_scores)

        return WordProgress(
            word=normalized,
            attempts=len(attempts),
            best_score=max(scores),
            average_score=sum(scores) / len(scores),
            is_mastered=average_recent >= self._config.mastery_score_threshold,
        )

    def get_all_progress(self) -> List[WordProgress]:
        """Progress summaries for every word ever attempted, useful for
        rendering a full progress dashboard in main.py's GUI."""
        progress_list = []
        for word in self._history:
            progress = self.get_progress(word)
            if progress:
                progress_list.append(progress)
        return progress_list

    def reset_history(self, word: Optional[str] = None) -> None:
        """
        Clear learning history.

        Args:
            word: If provided, clears only that word's history. If None,
                clears ALL history for ALL words.
        """
        if word is not None:
            self._history.pop(word.strip().lower(), None)
            logger.info("Cleared history for word: '%s'", word)
        else:
            self._history.clear()
            logger.info("Cleared all learning history.")


if __name__ == "__main__":
    # Minimal manual smoke-test: teaches a word using PlaceholderAnimationLibrary
    # (no real assets needed) and evaluates a few synthetic "attempts" built
    # by perturbing the target pose, to verify scoring/feedback behave
    # sensibly. Run via: `python teacher_module.py`.
    from utils.data_types import HandDetectionResult, HandLandmarks, Handedness

    logger.info("Running teacher_module.py standalone demo.")

    teacher = TeacherModule()
    teacher.start_lesson("hello")

    target_clip = teacher._current_target_clip  # test-only introspection
    target_pose = target_clip.frames[-1].right_hand

    def make_attempt(curl_fraction: float) -> HandDetectionResult:
        """Build a synthetic user attempt by curling the INDEX finger's
        tip toward the wrist by `curl_fraction` (0.0 = matches target
        exactly, higher = more curled than the target), to exercise the
        extension-ratio comparison meaningfully. (A uniform translation
        of every landmark would NOT change anything here, since extension
        ratio is a ratio of distances and therefore translation-invariant
        by design -- that's correct scale/position independence, not a
        bug, but it means our synthetic test needs to actually bend a
        finger rather than just shift the whole hand.)
        """
        wrist = target_pose[WRIST_INDEX]
        perturbed = list(target_pose)
        index_tip_idx = FINGER_LANDMARK_INDICES[Finger.INDEX][3]
        original_tip = target_pose[index_tip_idx]
        perturbed[index_tip_idx] = Landmark(
            x=original_tip.x + (wrist.x - original_tip.x) * curl_fraction,
            y=original_tip.y + (wrist.y - original_tip.y) * curl_fraction,
            z=original_tip.z,
        )
        hand = HandLandmarks(
            landmarks=perturbed,
            handedness=Handedness.RIGHT,
            detection_confidence=0.9,
            handedness_confidence=0.9,
        )
        return HandDetectionResult(
            frame_index=0, timestamp_ms=0.0, hands=[hand], frame_width=640, frame_height=480
        )

    for label, curl_fraction in [
        ("matches target", 0.0),
        ("index finger slightly curled", 0.3),
        ("index finger fully curled", 0.7),
    ]:
        result = make_attempt(curl_fraction)
        feedback = teacher.evaluate_attempt(result)
        print(f"\n[{label}] score={feedback.score}, correct={feedback.is_correct}")
        for msg in feedback.summary_messages():
            print("  -", msg)

    progress = teacher.get_progress("hello")
    print(f"\nProgress for 'hello': {progress}")
