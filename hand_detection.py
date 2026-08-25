"""
hand_detection.py

MODULE 1 -- Hand Detection

Single Responsibility
----------------------
This module's ONLY job is: given a video frame, detect up to two hands and
return their 21 landmarks each, along with handedness (left/right) and
confidence scores. It knows NOTHING about occlusion recovery, gesture
recognition, captions, voice, or avatars -- those concerns live in their own
modules (overlap_resolution.py, continuous_recognition.py, etc.), which is
what keeps this module independent, testable, and reusable in isolation.

Design notes
------------
- We wrap MediaPipe behind a small class (`HandDetector`) implementing a
  narrow interface (`detect`). This means that if we ever want to swap
  MediaPipe for a custom-trained detector or a different vendor SDK, only
  this file needs to change -- no other module in the project references
  MediaPipe directly (Dependency Inversion / Open-Closed Principle).
- All output is expressed using the shared DTOs in utils/data_types.py
  (HandDetectionResult, HandLandmarks, Landmark, Handedness) so that
  downstream modules have a stable, typed contract to depend on.
- The class manages the MediaPipe model's lifecycle (init/close) so callers
  don't need to know MediaPipe-specific setup/teardown details.

MediaPipe API note (read this if you hit an AttributeError)
-------------------------------------------------------------
Google has DEPRECATED the legacy `mediapipe.solutions.hands` API in favor
of the newer Tasks API (`mediapipe.tasks.python.vision.HandLandmarker`).
Many current `pip install mediapipe` wheels no longer ship the legacy
`solutions` module at all, which raises
`AttributeError: module 'mediapipe' has no attribute 'solutions'` if code
tries to use it. This module is therefore built entirely on the current
Tasks API, which is what Google's own documentation now recommends:
https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python

The Tasks API requires a small (~10MB) model asset file
(`hand_landmarker.task`) rather than bundling weights inside the pip
package itself. `HandDetector` downloads this automatically to
`models/hand_landmarker.task` the first time it's needed (see
`_resolve_model_path()`) and reuses the cached copy afterward. If your
environment has no internet access, download it manually from the URL in
`DEFAULT_MODEL_URL` below and place it at that path (or point
`HandDetectorConfig.model_asset_path` at wherever you put it).
"""

from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        HandLandmarker,
        HandLandmarkerOptions,
        RunningMode,
    )
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "mediapipe is required for hand_detection.py. "
        "Install it via: pip install mediapipe"
    ) from exc

from utils.data_types import (
    HandDetectionResult,
    HandLandmarks,
    Handedness,
    Landmark,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# Official Google-hosted MediaPipe Hand Landmarker model asset (float16,
# "full" variant -- balances accuracy and speed for general use). See the
# module docstring above for background on why this download exists.
DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


@dataclass
class HandDetectorConfig:
    """
    Configuration for the HandDetector.

    Kept as an explicit dataclass (rather than scattering magic numbers
    through the code) so that tuning detection behaviour -- e.g. for a
    research experiment, or for a low-power edge deployment -- is a matter
    of changing one object, not hunting through the file.
    """

    # Maximum number of hands to detect simultaneously. The project
    # requirement is two-hand detection, but this is kept configurable for
    # future extensibility (e.g. multi-person scenarios).
    max_num_hands: int = 2

    # Minimum confidence for the initial hand detection step to be
    # considered successful.
    min_detection_confidence: float = 0.6

    # Minimum confidence for a hand's continued presence to be trusted
    # between frames.
    min_presence_confidence: float = 0.5

    # Minimum confidence for the landmark tracker to keep tracking a hand
    # between frames.
    min_tracking_confidence: float = 0.5

    # Path to the Hand Landmarker .task model asset. If None (default),
    # HandDetector looks for/downloads it to models/hand_landmarker.task
    # relative to this project's root -- see _resolve_model_path().
    model_asset_path: Optional[Path] = None

    # NOTE: model "complexity" (lite vs. full accuracy/speed tradeoff) is
    # now selected by WHICH model asset file is used, not a runtime
    # parameter -- see DEFAULT_MODEL_URL. This field is kept only so old
    # config.json files with this key don't error on load; it currently
    # has no effect.
    model_complexity: int = 1


def _resolve_model_path(configured_path: Optional[Path]) -> Path:
    """
    Determine where the Hand Landmarker .task model asset lives on disk,
    downloading it from Google's model repository if it isn't present yet.

    Args:
        configured_path: An explicit path from HandDetectorConfig, or None
            to use the project-relative default (models/hand_landmarker.task).

    Returns:
        The resolved, existing path to a usable .task file.

    Raises:
        RuntimeError: If the model isn't present locally and couldn't be
            downloaded (e.g. no internet access), with clear instructions
            for resolving it manually.
    """
    path = configured_path or (
        Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
    )
    path = Path(path)

    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Hand landmark model not found at %s; downloading from %s ...",
        path,
        DEFAULT_MODEL_URL,
    )
    try:
        urllib.request.urlretrieve(DEFAULT_MODEL_URL, str(path))
        logger.info("Downloaded hand landmark model to %s", path)
    except Exception as exc:
        raise RuntimeError(
            "Could not find or download the MediaPipe Hand Landmarker model. "
            f"Please download it manually from {DEFAULT_MODEL_URL} and place "
            f"it at {path} (or set HandDetectorConfig.model_asset_path to "
            f"wherever you saved it). Underlying download error: {exc}"
        ) from exc

    return path


class HandDetector:
    """
    Detects hands and their 21 landmarks in video frames using MediaPipe's
    Hand Landmarker task.

    Usage:
        detector = HandDetector()
        result = detector.detect(frame_bgr, frame_index=0)
        detector.close()

    Or as a context manager:
        with HandDetector() as detector:
            result = detector.detect(frame_bgr, frame_index=0)
    """

    def __init__(self, config: Optional[HandDetectorConfig] = None) -> None:
        """
        Initialize the underlying MediaPipe Hand Landmarker model.

        Args:
            config: Optional HandDetectorConfig. If not provided, sensible
                defaults are used (see HandDetectorConfig).
        """
        self._config = config or HandDetectorConfig()

        model_path = _resolve_model_path(self._config.model_asset_path)

        logger.info(
            "Initializing MediaPipe Hand Landmarker (max_hands=%d, "
            "detection_conf=%.2f, tracking_conf=%.2f, model=%s)",
            self._config.max_num_hands,
            self._config.min_detection_confidence,
            self._config.min_tracking_confidence,
            model_path,
        )

        try:
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=RunningMode.VIDEO,
                num_hands=self._config.max_num_hands,
                min_hand_detection_confidence=self._config.min_detection_confidence,
                min_hand_presence_confidence=self._config.min_presence_confidence,
                min_tracking_confidence=self._config.min_tracking_confidence,
            )
            self._hands_model = HandLandmarker.create_from_options(options)
        except Exception:
            logger.exception("Failed to initialize MediaPipe Hand Landmarker model.")
            raise

        # The Tasks API's VIDEO running mode requires strictly increasing
        # timestamps across calls; we track the last one used so detect()
        # can defend against a caller passing a non-increasing or duplicate
        # timestamp (e.g. two frames captured in the same millisecond).
        self._last_timestamp_ms: int = -1
        self._is_closed = False

    def detect(
        self,
        frame_bgr: np.ndarray,
        frame_index: int = 0,
        timestamp_ms: Optional[float] = None,
    ) -> HandDetectionResult:
        """
        Run hand detection on a single BGR video frame (as produced by
        OpenCV's VideoCapture).

        Args:
            frame_bgr: The input frame in BGR color order, shape
                (height, width, 3), dtype uint8. This is OpenCV's native
                format, chosen so callers don't need to do color conversion
                themselves before calling this method -- we do it internally.
            frame_index: Sequential index of this frame in the video stream.
                Useful for downstream temporal modules (overlap_resolution,
                continuous_recognition) to reason about frame order.
            timestamp_ms: Optional explicit timestamp in milliseconds. If
                not provided, wall-clock time is used, which is fine for
                live-camera use but callers processing recorded video
                should pass the video's own timestamp for accuracy.

        Returns:
            A HandDetectionResult containing zero, one, or two detected
            hands with their landmarks, handedness, and confidence scores.

        Raises:
            ValueError: If the input frame is not a valid 3-channel image.
            RuntimeError: If called after the detector has been closed.
        """
        if self._is_closed:
            raise RuntimeError(
                "HandDetector.detect() called after close(). "
                "Create a new HandDetector instance to continue detecting."
            )

        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(
                "frame_bgr must be a valid (H, W, 3) BGR image ndarray; "
                f"got shape={None if frame_bgr is None else frame_bgr.shape}"
            )

        if timestamp_ms is None:
            timestamp_ms = time.time() * 1000.0

        frame_height, frame_width = frame_bgr.shape[:2]

        result = HandDetectionResult(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        try:
            # MediaPipe expects RGB input; OpenCV frames are BGR by default.
            rgb_frame = np.ascontiguousarray(frame_bgr[:, :, ::-1])
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # VIDEO running mode requires strictly increasing integer
            # timestamps; clamp forward by 1ms if the caller's timestamp
            # didn't advance, rather than letting the Tasks API raise.
            timestamp_ms_int = max(int(timestamp_ms), self._last_timestamp_ms + 1)
            self._last_timestamp_ms = timestamp_ms_int

            mp_result = self._hands_model.detect_for_video(mp_image, timestamp_ms_int)
        except Exception:
            # We deliberately do not let a single bad frame crash a live
            # video pipeline. We log the error and return an empty result
            # so main.py's loop can continue to the next frame.
            logger.exception(
                "Hand detection failed on frame_index=%d; returning empty result.",
                frame_index,
            )
            return result

        if not mp_result.hand_landmarks:
            # No hands detected this frame -- a perfectly valid, common case
            # (e.g. hands outside frame, occluded, or between signs).
            return result

        for hand_idx, hand_landmark_list in enumerate(mp_result.hand_landmarks):
            try:
                hand_data = self._convert_single_hand(
                    hand_landmark_list=hand_landmark_list,
                    handedness_list=mp_result.handedness,
                    hand_idx=hand_idx,
                )
                result.hands.append(hand_data)
            except Exception:
                logger.exception(
                    "Failed to convert landmarks for hand index %d on "
                    "frame_index=%d; skipping this hand.",
                    hand_idx,
                    frame_index,
                )
                continue

        return result

    def _convert_single_hand(
        self,
        hand_landmark_list,
        handedness_list: List,
        hand_idx: int,
    ) -> HandLandmarks:
        """
        Convert a single MediaPipe Tasks API hand result into our internal
        HandLandmarks DTO.

        This is a private helper (leading underscore) since it is an
        implementation detail of how we adapt MediaPipe's output format to
        our own -- external modules should never need to call this directly.

        Args:
            hand_landmark_list: MediaPipe's list of 21 NormalizedLandmark
                objects for one hand (from HandLandmarkerResult.hand_landmarks).
            handedness_list: MediaPipe's HandLandmarkerResult.handedness --
                a list (per detected hand) of lists of Category objects.
            hand_idx: Index of this hand within the current frame's detections.

        Returns:
            A populated HandLandmarks instance.
        """
        landmarks: List[Landmark] = [
            Landmark(x=lm.x, y=lm.y, z=lm.z, is_estimated=False)
            for lm in hand_landmark_list
        ]

        # Determine handedness + confidence for this specific hand index.
        # The Tasks API's `handedness` list is aligned by index with
        # `hand_landmarks`, but we defensively guard against index
        # mismatches (e.g. if a future MediaPipe version changes ordering).
        handedness = Handedness.UNKNOWN
        handedness_confidence = 0.0

        if hand_idx < len(handedness_list) and handedness_list[hand_idx]:
            top_category = handedness_list[hand_idx][0]
            raw_label = top_category.category_name  # "Left" or "Right"
            handedness_confidence = float(top_category.score)

            # IMPORTANT: MediaPipe reports handedness from the camera's
            # perspective, which is mirrored relative to the person facing
            # the camera (i.e. what MediaPipe calls "Left" is the user's
            # right hand in a typical un-mirrored webcam feed). We keep the
            # raw MediaPipe label here and leave any mirroring correction
            # to the caller/config layer (e.g. main.py, depending on
            # whether the preview is flipped), so this module stays a
            # faithful, unopinionated wrapper around the raw detector.
            try:
                handedness = Handedness(raw_label)
            except ValueError:
                logger.warning(
                    "Unrecognized handedness label '%s' from MediaPipe; "
                    "defaulting to UNKNOWN.",
                    raw_label,
                )
                handedness = Handedness.UNKNOWN

        # Detection confidence: the Tasks API doesn't expose a separate
        # per-landmark-set detection score distinct from the handedness
        # score in the Python API, so we use the handedness classification
        # score as a practical proxy for overall detection confidence.
        detection_confidence = handedness_confidence

        bounding_box = self._compute_bounding_box(landmarks)

        return HandLandmarks(
            landmarks=landmarks,
            handedness=handedness,
            detection_confidence=detection_confidence,
            handedness_confidence=handedness_confidence,
            bounding_box=bounding_box,
        )

    @staticmethod
    def _compute_bounding_box(
        landmarks: List[Landmark],
    ) -> Optional[tuple]:
        """
        Compute a normalized axis-aligned bounding box (xmin, ymin, xmax,
        ymax) around a set of landmarks.

        This is provided so overlap_resolution.py can cheaply check for
        intersecting hands without recomputing this from scratch.

        Args:
            landmarks: List of Landmark points (normalized coordinates).

        Returns:
            (xmin, ymin, xmax, ymax) tuple, or None if landmarks is empty.
        """
        if not landmarks:
            return None

        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        return (min(xs), min(ys), max(xs), max(ys))

    def close(self) -> None:
        """
        Release the underlying MediaPipe model resources.

        Should be called when the detector is no longer needed (e.g. when
        shutting down main.py's pipeline) to free native resources
        cleanly. Safe to call multiple times.
        """
        if not self._is_closed:
            try:
                self._hands_model.close()
            except Exception:
                logger.exception("Error while closing MediaPipe Hand Landmarker model.")
            finally:
                self._is_closed = True
                logger.info("HandDetector closed.")

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort cleanup in case the caller forgot to call close()
        # explicitly or use the context manager. Errors here are
        # intentionally swallowed since __del__ runs during interpreter
        # teardown, where logging/resources may already be unavailable.
        try:
            self.close()
        except Exception:
            pass


if __name__ == "__main__":
    # Minimal manual smoke-test / usage example when running this file
    # directly: `python hand_detection.py`. Requires a webcam and, on
    # first run, internet access to download the model asset once.
    #
    # This block deliberately uses OpenCV only for local demonstration; it
    # is NOT part of the module's public API and is not imported by other
    # modules (main.py implements its own, fuller camera loop).
    import cv2

    logger.info("Running hand_detection.py standalone demo. Press 'q' to quit.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam for demo.")
    else:
        with HandDetector() as detector:
            frame_idx = 0
            while True:
                success, frame = cap.read()
                if not success:
                    logger.warning("Failed to read frame from webcam.")
                    break

                detection_result = detector.detect(frame, frame_index=frame_idx)
                frame_idx += 1

                # Draw simple circles at each landmark for visual sanity-check.
                for hand in detection_result.hands:
                    for lm in hand.landmarks:
                        px = int(lm.x * detection_result.frame_width)
                        py = int(lm.y * detection_result.frame_height)
                        cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)
                    cv2.putText(
                        frame,
                        f"{hand.handedness.value} ({hand.detection_confidence:.2f})",
                        (10, 30 if hand.handedness == Handedness.LEFT else 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2,
                    )

                cv2.imshow("hand_detection.py demo", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()
