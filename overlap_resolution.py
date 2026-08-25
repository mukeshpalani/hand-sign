"""
overlap_resolution.py

MODULE 2 — Overlap Resolution

Single Responsibility
----------------------
This module's ONLY job is: take the raw, per-frame output of
hand_detection.py and produce a STABILIZED version of it, where:
  1. Overlapping/intersecting hands are detected (bounding-box IoU).
  2. A hand that vanishes from detection *because it was just occluded by
     the other hand* is not simply dropped — its landmarks are estimated
     from recent motion history instead of losing tracking.
  3. Real (observed) landmarks are lightly smoothed over time to reduce
     frame-to-frame jitter, without masking genuine motion.

This module knows NOTHING about gesture recognition, captions, or
rendering — it only consumes HandDetectionResult objects and produces
improved HandDetectionResult objects. This keeps it a pure, composable
"middleware" stage in the pipeline (main.py wires:
    hand_detection -> overlap_resolution -> continuous_recognition
), and it can be unit-tested independently by feeding it synthetic
HandDetectionResult sequences.

Design notes
------------
- The resolver is STATEFUL by necessity (it needs history across frames to
  detect motion and occlusion), so it is implemented as a class instance
  that main.py constructs once per video stream and calls repeatedly per
  frame, exactly mirroring HandDetector's lifecycle pattern.
- We only attempt to "recover" a hand that disappears if, in the frame
  immediately before it vanished, its bounding box was actually
  overlapping with the other hand's bounding box. This distinguishes real
  occlusion (spec requirement) from a hand simply leaving the camera view
  (where predicting phantom landmarks would be actively harmful/misleading).
- Predicted (estimated) landmarks are explicitly flagged via
  `Landmark.is_estimated = True` so downstream modules (e.g.
  continuous_recognition.py) can choose to weight or discount them.
- We cap how many consecutive frames we're willing to predict for
  (`max_occlusion_frames`) to avoid unbounded drift if a hand is occluded
  for a long time — after that, we honestly report the hand as absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from utils.data_types import (
    HandDetectionResult,
    HandLandmarks,
    Handedness,
    Landmark,
)
from utils.logger import get_logger

logger = get_logger(__name__)


BoundingBox = Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)


@dataclass
class OverlapResolverConfig:
    """
    Tunable parameters for occlusion detection and recovery.

    Kept as a dataclass so behaviour can be tuned per-deployment (e.g. a
    noisier low-power camera might want a lower IoU threshold or fewer
    max_occlusion_frames) without touching the resolution logic itself.
    """

    # Bounding-box Intersection-over-Union above this value is considered
    # a genuine "hands overlapping" event.
    overlap_iou_threshold: float = 0.05

    # How many CONSECUTIVE frames we are willing to predict a missing
    # hand's position for, before giving up and reporting it as truly
    # absent. Prevents unbounded drift during long occlusions.
    max_occlusion_frames: int = 15

    # Exponential Moving Average smoothing factor for REAL (observed)
    # landmarks, in [0, 1]. Higher = more responsive to new detections,
    # lower = smoother but more lag. 1.0 disables smoothing entirely.
    smoothing_alpha: float = 0.7

    # If the gap between the current frame's timestamp and the hand's last
    # real detection exceeds this many milliseconds, we treat any
    # further prediction as unreliable and drop the hand, regardless of
    # max_occlusion_frames (protects against irregular/dropped frame rates).
    max_occlusion_duration_ms: float = 1000.0


@dataclass
class _HandTrackState:
    """
    Internal per-hand (per LEFT/RIGHT) tracking state kept between calls to
    OverlapResolver.resolve(). Not part of the public API — external
    modules never see this class.
    """

    # Most recent REAL (i.e. actually observed, non-estimated) landmarks.
    last_real_landmarks: Optional[HandLandmarks] = None
    last_real_timestamp_ms: Optional[float] = None

    # The real observation before `last_real_landmarks`, kept purely to
    # compute per-landmark velocity for linear extrapolation.
    previous_real_landmarks: Optional[HandLandmarks] = None
    previous_real_timestamp_ms: Optional[float] = None

    # EMA-smoothed landmark positions, updated every time a real
    # observation arrives. This is what we actually output for "real"
    # detections, to reduce jitter.
    smoothed_positions: Optional[List[Tuple[float, float, float]]] = None

    # How many consecutive frames this hand has been missing from raw
    # detection and had its position predicted instead.
    frames_since_real_detection: int = 0

    # Whether, in the last frame this hand WAS actually seen, its bounding
    # box overlapped with the other hand's bounding box. This is the key
    # signal used to decide "is this disappearance likely occlusion?".
    was_overlapping_when_last_seen: bool = False


class OverlapResolver:
    """
    Stabilizes hand-detection results across frames by detecting
    hand-to-hand overlap and recovering landmarks for hands that briefly
    disappear due to occlusion.

    Usage:
        resolver = OverlapResolver()
        stabilized_result = resolver.resolve(raw_detection_result)
    """

    def __init__(self, config: Optional[OverlapResolverConfig] = None) -> None:
        """
        Args:
            config: Optional OverlapResolverConfig; sensible defaults are
                used if omitted.
        """
        self._config = config or OverlapResolverConfig()

        # Only LEFT and RIGHT are tracked individually (UNKNOWN handedness
        # detections are passed through unmodified, since we can't reliably
        # associate them with a track across frames).
        self._track_state: Dict[Handedness, _HandTrackState] = {
            Handedness.LEFT: _HandTrackState(),
            Handedness.RIGHT: _HandTrackState(),
        }

        logger.info(
            "OverlapResolver initialized (iou_threshold=%.2f, "
            "max_occlusion_frames=%d, smoothing_alpha=%.2f)",
            self._config.overlap_iou_threshold,
            self._config.max_occlusion_frames,
            self._config.smoothing_alpha,
        )

    def resolve(self, result: HandDetectionResult) -> HandDetectionResult:
        """
        Process one frame's raw HandDetectionResult and return a stabilized
        version with occlusion recovery and smoothing applied.

        Args:
            result: The raw output from HandDetector.detect() for this frame.

        Returns:
            A new HandDetectionResult (the input is not mutated) with:
              - Real hands present: smoothed landmark positions.
              - Hands missing due to detected occlusion: estimated
                landmarks filled in (marked is_estimated=True), up to the
                configured occlusion limits.
              - Hands missing for other reasons (e.g. left the frame, or
                occlusion limit exceeded): left absent, as-is.
        """
        try:
            return self._resolve_impl(result)
        except Exception:
            # Defensive: a bug in stabilization logic should never crash
            # the live pipeline. Worst case, we fall back to passing the
            # raw (unstabilized) result through untouched.
            logger.exception(
                "OverlapResolver failed on frame_index=%d; "
                "falling back to raw (unstabilized) detection result.",
                result.frame_index,
            )
            return result

    def _resolve_impl(self, result: HandDetectionResult) -> HandDetectionResult:
        """Core stabilization logic, separated so resolve() can wrap it in
        a single top-level try/except (see resolve() docstring)."""

        stabilized = HandDetectionResult(
            frame_index=result.frame_index,
            timestamp_ms=result.timestamp_ms,
            frame_width=result.frame_width,
            frame_height=result.frame_height,
            hands=[],
        )

        observed_by_handedness: Dict[Handedness, HandLandmarks] = {}
        passthrough_hands: List[HandLandmarks] = []

        for hand in result.hands:
            if hand.handedness in (Handedness.LEFT, Handedness.RIGHT):
                observed_by_handedness[hand.handedness] = hand
            else:
                # UNKNOWN handedness: can't be tracked across frames
                # reliably, so we pass it through unchanged rather than
                # guessing which track it belongs to.
                passthrough_hands.append(hand)

        # Determine current-frame overlap between the two tracked hands,
        # if both are present in RAW detection this frame.
        currently_overlapping = self._detect_overlap(
            observed_by_handedness.get(Handedness.LEFT),
            observed_by_handedness.get(Handedness.RIGHT),
        )

        for handedness in (Handedness.LEFT, Handedness.RIGHT):
            state = self._track_state[handedness]
            observed_hand = observed_by_handedness.get(handedness)

            if observed_hand is not None:
                # --- Case 1: hand WAS observed this frame ---
                stabilized_hand = self._update_with_real_observation(
                    state=state,
                    observed_hand=observed_hand,
                    timestamp_ms=result.timestamp_ms,
                )
                state.was_overlapping_when_last_seen = currently_overlapping
                state.frames_since_real_detection = 0
                stabilized.hands.append(stabilized_hand)

            else:
                # --- Case 2: hand MISSING this frame; consider prediction ---
                predicted_hand = self._try_predict_missing_hand(
                    state=state,
                    handedness=handedness,
                    timestamp_ms=result.timestamp_ms,
                )
                if predicted_hand is not None:
                    stabilized.hands.append(predicted_hand)
                    state.frames_since_real_detection += 1
                # else: hand stays absent from `stabilized`, i.e. we
                # honestly report "not visible" rather than guessing.

        stabilized.hands.extend(passthrough_hands)

        if currently_overlapping:
            logger.debug(
                "Hand overlap detected on frame_index=%d.", result.frame_index
            )

        return stabilized

    # ------------------------------------------------------------------
    # Real observation handling (smoothing)
    # ------------------------------------------------------------------

    def _update_with_real_observation(
        self,
        state: _HandTrackState,
        observed_hand: HandLandmarks,
        timestamp_ms: float,
    ) -> HandLandmarks:
        """
        Update the track's history with a genuine observation, apply EMA
        smoothing to reduce jitter, and return the smoothed HandLandmarks.

        Args:
            state: The mutable track state for this hand (LEFT or RIGHT).
            observed_hand: The raw HandLandmarks just observed this frame.
            timestamp_ms: Timestamp of the current frame.

        Returns:
            A new HandLandmarks with smoothed (but not estimated)
            positions.
        """
        alpha = self._config.smoothing_alpha
        raw_positions = [(lm.x, lm.y, lm.z) for lm in observed_hand.landmarks]

        if state.smoothed_positions is None or len(
            state.smoothed_positions
        ) != len(raw_positions):
            # First observation for this track (or landmark count changed
            # unexpectedly) — nothing to smooth against yet, so seed with
            # the raw values.
            smoothed_positions = raw_positions
        else:
            smoothed_positions = [
                (
                    alpha * rx + (1 - alpha) * sx,
                    alpha * ry + (1 - alpha) * sy,
                    alpha * rz + (1 - alpha) * sz,
                )
                for (rx, ry, rz), (sx, sy, sz) in zip(
                    raw_positions, state.smoothed_positions
                )
            ]

        smoothed_landmarks = [
            Landmark(x=x, y=y, z=z, is_estimated=False)
            for (x, y, z) in smoothed_positions
        ]

        smoothed_hand = HandLandmarks(
            landmarks=smoothed_landmarks,
            handedness=observed_hand.handedness,
            detection_confidence=observed_hand.detection_confidence,
            handedness_confidence=observed_hand.handedness_confidence,
            bounding_box=self._compute_bounding_box(smoothed_landmarks),
        )

        # Roll the history window forward by one real observation.
        state.previous_real_landmarks = state.last_real_landmarks
        state.previous_real_timestamp_ms = state.last_real_timestamp_ms
        state.last_real_landmarks = smoothed_hand
        state.last_real_timestamp_ms = timestamp_ms
        state.smoothed_positions = smoothed_positions

        return smoothed_hand

    # ------------------------------------------------------------------
    # Missing-hand prediction (occlusion recovery)
    # ------------------------------------------------------------------

    def _try_predict_missing_hand(
        self,
        state: _HandTrackState,
        handedness: Handedness,
        timestamp_ms: float,
    ) -> Optional[HandLandmarks]:
        """
        Attempt to estimate a missing hand's landmarks from recent motion
        history, but ONLY if we believe the disappearance is due to
        occlusion by the other hand (per project requirement 3: "Resolve
        overlapping hands intelligently by estimating missing joints
        instead of losing tracking").

        Returns:
            An estimated HandLandmarks (with every landmark flagged
            is_estimated=True), or None if recovery is not attempted
            (e.g. no history, occlusion limit exceeded, or the hand
            appears to have simply left the frame rather than been
            occluded).
        """
        # Nothing to extrapolate from if we've never seen this hand.
        if state.last_real_landmarks is None or state.last_real_timestamp_ms is None:
            return None

        # Only attempt recovery if the hand disappeared right after being
        # detected as overlapping with the other hand — this is the
        # signal that distinguishes "occluded" from "left the frame".
        if not state.was_overlapping_when_last_seen:
            return None

        # Respect the hard cap on consecutive predicted frames.
        if state.frames_since_real_detection >= self._config.max_occlusion_frames:
            logger.debug(
                "%s hand exceeded max_occlusion_frames (%d); "
                "reporting as truly lost.",
                handedness.value,
                self._config.max_occlusion_frames,
            )
            return None

        # Respect the hard cap on wall-clock occlusion duration, guarding
        # against irregular frame timing.
        elapsed_ms = timestamp_ms - state.last_real_timestamp_ms
        if elapsed_ms > self._config.max_occlusion_duration_ms:
            logger.debug(
                "%s hand occlusion duration (%.0fms) exceeded max (%.0fms); "
                "reporting as truly lost.",
                handedness.value,
                elapsed_ms,
                self._config.max_occlusion_duration_ms,
            )
            return None

        velocity_per_ms = self._estimate_velocity(state)
        dt_ms = elapsed_ms if state.previous_real_timestamp_ms is not None else 0.0

        predicted_landmarks: List[Landmark] = []
        for idx, lm in enumerate(state.last_real_landmarks.landmarks):
            vx, vy, vz = velocity_per_ms[idx] if velocity_per_ms else (0.0, 0.0, 0.0)
            predicted_landmarks.append(
                Landmark(
                    x=self._clamp01(lm.x + vx * dt_ms),
                    y=self._clamp01(lm.y + vy * dt_ms),
                    z=lm.z + vz * dt_ms,
                    is_estimated=True,
                )
            )

        logger.debug(
            "Estimated %s hand landmarks for %d consecutive occluded frame(s).",
            handedness.value,
            state.frames_since_real_detection + 1,
        )

        return HandLandmarks(
            landmarks=predicted_landmarks,
            handedness=handedness,
            # Confidence decays the longer we go without a real observation,
            # so downstream modules can appropriately discount stale
            # predictions instead of trusting them as much as fresh ones.
            detection_confidence=self._decayed_confidence(state),
            handedness_confidence=self._decayed_confidence(state),
            bounding_box=self._compute_bounding_box(predicted_landmarks),
        )

    def _estimate_velocity(
        self, state: _HandTrackState
    ) -> Optional[List[Tuple[float, float, float]]]:
        """
        Estimate per-landmark linear velocity (units: normalized-coords
        per millisecond) from the last two REAL observations.

        Returns:
            A list of (vx, vy, vz) tuples aligned with landmark indices,
            or None if there isn't enough history (only one observation
            so far) — in which case the caller should treat velocity as
            zero (i.e. predict a static/frozen last-known pose).
        """
        if (
            state.previous_real_landmarks is None
            or state.previous_real_timestamp_ms is None
            or state.last_real_landmarks is None
            or state.last_real_timestamp_ms is None
        ):
            return None

        dt_ms = state.last_real_timestamp_ms - state.previous_real_timestamp_ms
        if dt_ms <= 0:
            # Guard against duplicate/out-of-order timestamps producing a
            # division by zero or nonsensical velocity.
            return None

        velocities = []
        for prev_lm, last_lm in zip(
            state.previous_real_landmarks.landmarks,
            state.last_real_landmarks.landmarks,
        ):
            vx = (last_lm.x - prev_lm.x) / dt_ms
            vy = (last_lm.y - prev_lm.y) / dt_ms
            vz = (last_lm.z - prev_lm.z) / dt_ms
            velocities.append((vx, vy, vz))

        return velocities

    def _decayed_confidence(self, state: _HandTrackState) -> float:
        """
        Compute a confidence score for a predicted (estimated) hand that
        decays linearly the longer we've gone without a real observation.

        This lets downstream consumers (e.g. continuous_recognition.py)
        weight predicted frames less heavily as occlusion drags on.
        """
        base_confidence = (
            state.last_real_landmarks.detection_confidence
            if state.last_real_landmarks
            else 0.5
        )
        decay_fraction = 1.0 - (
            state.frames_since_real_detection
            / max(self._config.max_occlusion_frames, 1)
        )
        return max(0.0, base_confidence * decay_fraction)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _detect_overlap(
        self,
        left_hand: Optional[HandLandmarks],
        right_hand: Optional[HandLandmarks],
    ) -> bool:
        """
        Determine whether two simultaneously-detected hands' bounding
        boxes overlap enough to be considered "intersecting" per the
        configured IoU threshold.

        Returns:
            False if either hand is missing (nothing to compare), or if
            the computed IoU is below overlap_iou_threshold; True
            otherwise.
        """
        if left_hand is None or right_hand is None:
            return False
        if left_hand.bounding_box is None or right_hand.bounding_box is None:
            return False

        iou = self._compute_iou(left_hand.bounding_box, right_hand.bounding_box)
        return iou >= self._config.overlap_iou_threshold

    @staticmethod
    def _compute_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
        """
        Compute Intersection-over-Union between two normalized
        (xmin, ymin, xmax, ymax) bounding boxes.

        Returns:
            IoU in [0.0, 1.0]. 0.0 if the boxes don't intersect at all.
        """
        ax_min, ay_min, ax_max, ay_max = box_a
        bx_min, by_min, bx_max, by_max = box_b

        inter_x_min = max(ax_min, bx_min)
        inter_y_min = max(ay_min, by_min)
        inter_x_max = min(ax_max, bx_max)
        inter_y_max = min(ay_max, by_max)

        inter_width = max(0.0, inter_x_max - inter_x_min)
        inter_height = max(0.0, inter_y_max - inter_y_min)
        intersection_area = inter_width * inter_height

        area_a = max(0.0, ax_max - ax_min) * max(0.0, ay_max - ay_min)
        area_b = max(0.0, bx_max - bx_min) * max(0.0, by_max - by_min)
        union_area = area_a + area_b - intersection_area

        if union_area <= 0.0:
            return 0.0
        return intersection_area / union_area

    @staticmethod
    def _compute_bounding_box(landmarks: List[Landmark]) -> Optional[BoundingBox]:
        """Recompute a bounding box after smoothing/prediction has moved
        landmark positions (mirrors the helper in hand_detection.py, but
        kept local here since this module must not import concrete logic
        from hand_detection.py — only the shared DTOs — to preserve
        module independence)."""
        if not landmarks:
            return None
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        return (min(xs), min(ys), max(xs), max(ys))

    @staticmethod
    def _clamp01(value: float) -> float:
        """Clamp a normalized coordinate prediction into the valid [0, 1]
        range, since linear extrapolation could otherwise predict a point
        drifting outside the visible frame."""
        return max(0.0, min(1.0, value))

    def reset(self) -> None:
        """
        Clear all tracking history for both hands.

        Should be called when starting a new, unrelated video session
        (e.g. a new user steps in front of the camera) so stale motion
        history from a previous session doesn't leak into predictions.
        """
        self._track_state = {
            Handedness.LEFT: _HandTrackState(),
            Handedness.RIGHT: _HandTrackState(),
        }
        logger.info("OverlapResolver tracking history reset.")


if __name__ == "__main__":
    # Minimal manual smoke-test combining this module with hand_detection.py.
    # Run via: `python overlap_resolution.py`. Requires a webcam.
    #
    # As with hand_detection.py's demo block, this is for local sanity
    # checking only and is not imported/used by main.py.
    import cv2

    from hand_detection import HandDetector

    logger.info(
        "Running overlap_resolution.py standalone demo. Press 'q' to quit."
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam for demo.")
    else:
        resolver = OverlapResolver()
        with HandDetector() as detector:
            frame_idx = 0
            while True:
                success, frame = cap.read()
                if not success:
                    logger.warning("Failed to read frame from webcam.")
                    break

                raw_result = detector.detect(frame, frame_index=frame_idx)
                stable_result = resolver.resolve(raw_result)
                frame_idx += 1

                for hand in stable_result.hands:
                    color = (0, 0, 255) if hand.landmarks[0].is_estimated else (0, 255, 0)
                    for lm in hand.landmarks:
                        px = int(lm.x * stable_result.frame_width)
                        py = int(lm.y * stable_result.frame_height)
                        cv2.circle(frame, (px, py), 3, color, -1)

                cv2.imshow("overlap_resolution.py demo", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()
