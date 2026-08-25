"""
main.py

MODULE 11 -- Main / Desktop GUI Front-End

Single Responsibility
----------------------
This file's ONLY job is: present the DESKTOP (OpenCV window) front-end
for the AI Sign Language Assistant -- own the camera loop, draw the video
feed with overlays, and handle keyboard controls. All actual detection,
recognition, captioning, voice, and avatar logic lives in `pipeline.py`'s
`SignPipeline`, which this file only calls into.

For the browser-based front-end (camera feed + caption/voice panel +
Ollama chatbot), see `web_app.py` -- it uses the exact same
`SignPipeline`, so recognition behavior is identical between the two;
only how things are DISPLAYED differs.

Design notes
------------
- THIN GUI SHELL: this file contains NO recognition/caption/voice/avatar
  logic of its own -- only OpenCV-specific rendering (drawing landmarks,
  text overlays with readable backgrounds) and keyboard handling. This is
  what keeps it trivially swappable for -- or usable alongside -- other
  front-ends like web_app.py without duplicating pipeline logic.
- GRACEFUL GUI DEGRADATION: if the OpenCV build/environment has no
  display backend available (e.g. a headless server/container), window
  rendering calls are caught and disabled after the first failure rather
  than crashing the whole application -- the recognition/caption/voice
  pipeline keeps running headlessly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from config import AppConfig, load_config
from pipeline import FrameSource, OpenCVCameraSource, SignPipeline, SyntheticFrameSource
from utils.logger import get_logger

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# Application
# ----------------------------------------------------------------------


class SignLanguageAssistantApp:
    """
    The desktop application: runs SignPipeline against a camera feed,
    displays an OpenCV window with live overlays, and handles keyboard
    controls.

    Usage:
        app = SignLanguageAssistantApp()
        app.run()
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        frame_source: Optional[FrameSource] = None,
        headless: bool = False,
    ) -> None:
        """
        Args:
            config: Optional AppConfig. Defaults to config.load_config()
                (which reads config.json, creating it with defaults on
                first run) if omitted.
            frame_source: Optional FrameSource. Defaults to a real
                OpenCVCameraSource using config.app.camera_index.
                Inject a SyntheticFrameSource for testing.
            headless: If True, never attempts to open a GUI window
                (useful for servers/CI). If False (default), a GUI window
                is attempted, but automatically falls back to headless
                mode if the OpenCV build/environment has no display
                backend available.
        """
        self._config = config or load_config()
        self._config.paths.create_all()

        self._headless = headless
        self._gui_available = not headless
        self._frame_source = frame_source or OpenCVCameraSource(
            self._config.app.camera_index
        )

        self._pipeline = SignPipeline(self._config)
        self._running = False

        logger.info(
            "SignLanguageAssistantApp initialized (headless=%s, "
            "voice=%s, avatar=%s, teacher_mode=%s).",
            headless,
            self._config.app.enable_voice_assistant,
            self._config.app.enable_avatar,
            self._config.app.enable_teacher_mode,
        )
        if not headless:
            logger.info(
                "Controls: 'q' quit | 'v' mute/unmute voice | 'c' clear sentence | "
                "'d' inject a demo caption (tests voice/avatar/captions without a trained model)"
            )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Run the live pipeline loop until the frame source is exhausted or
        the user quits. Handles setup/teardown of all modules.
        """
        self._running = True
        logger.info("Starting main application loop.")

        try:
            while self._running and self._frame_source.is_opened():
                frame = self._frame_source.read()
                if frame is None:
                    logger.info("Frame source exhausted; stopping.")
                    break

                frame_result = self._pipeline.process_frame(frame)

                if frame_result.caption is not None:
                    self._pipeline.dispatch_caption(frame_result.caption)

                if not self._headless:
                    self._render_frame(frame_result)
                    self._handle_keyboard_input()
        except KeyboardInterrupt:
            logger.info("Interrupted by user (Ctrl+C).")
        finally:
            self.shutdown()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_frame(self, frame_result) -> None:
        """Draw landmarks, captions, FPS, and teacher feedback onto the
        frame and display it. Silently (once) disables further rendering
        if the environment has no GUI/display backend available."""
        if not self._gui_available:
            return

        frame = frame_result.frame
        try:
            import cv2

            self._draw_landmarks(cv2, frame, frame_result.stabilized_result)
            self._draw_captions(cv2, frame)
            self._draw_status_overlay(cv2, frame, frame_result)

            cv2.imshow(self._config.app.window_title, frame)
            cv2.waitKey(1)
        except Exception:
            logger.warning(
                "GUI rendering unavailable in this environment (no display "
                "backend?); continuing headlessly for the rest of the session.",
                exc_info=True,
            )
            self._gui_available = False

    def _draw_landmarks(self, cv2, frame: np.ndarray, result) -> None:
        for hand in result.hands:
            color = (0, 0, 255) if hand.landmarks and hand.landmarks[0].is_estimated else (0, 255, 0)
            for lm in hand.landmarks:
                px = int(lm.x * result.frame_width)
                py = int(lm.y * result.frame_height)
                cv2.circle(frame, (px, py), 3, color, -1)

    def _draw_text_with_background(
        self,
        cv2,
        frame: np.ndarray,
        text: str,
        origin,
        font_scale: float = 0.6,
        text_color=(255, 255, 255),
        bg_color=(0, 0, 0),
        thickness: int = 2,
        padding: int = 6,
    ) -> None:
        """
        Draw text on a solid background rectangle, so it stays readable
        regardless of what's behind it in the camera feed (a plain
        cv2.putText call with no background can disappear entirely against
        a similarly-colored/bright background).
        """
        if not text:
            return
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = origin

        cv2.rectangle(
            frame,
            (x - padding, y - text_h - padding),
            (x + text_w + padding, y + baseline + padding),
            bg_color,
            thickness=-1,  # filled
        )
        cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    def _draw_captions(self, cv2, frame: np.ndarray) -> None:
        live_caption = self._pipeline.caption_generator.get_current_live_caption()
        text = live_caption.text if live_caption else ""
        if not text:
            return
        self._draw_text_with_background(
            cv2,
            frame,
            text,
            origin=(10, frame.shape[0] - 20),
            font_scale=0.8,
            text_color=(0, 255, 255),  # bright yellow text
            bg_color=(0, 0, 0),         # solid black backing box
            thickness=2,
        )

    def _draw_status_overlay(self, cv2, frame: np.ndarray, frame_result) -> None:
        metrics = self._pipeline.get_performance_metrics()
        fps_text = f"FPS: {metrics.fps:.1f}" if metrics else "FPS: --"
        self._draw_text_with_background(
            cv2, frame, fps_text, origin=(10, 30), font_scale=0.6,
            text_color=(255, 255, 0), bg_color=(0, 0, 0), thickness=2,
        )

        # Make it OBVIOUS when recognition is a no-op because no model has
        # been trained yet -- otherwise "no captions/voice ever happen"
        # looks like a bug rather than the expected state before training.
        if not self._pipeline.has_trained_model:
            self._draw_text_with_background(
                cv2, frame,
                "No trained model loaded -- press 'd' to test voice/avatar/captions",
                origin=(10, 60), font_scale=0.5,
                text_color=(255, 255, 255), bg_color=(0, 0, 180), thickness=1,
            )

        teacher_module = self._pipeline.teacher_module
        if teacher_module is not None and teacher_module.get_target_word():
            target = f"Lesson: {teacher_module.get_target_word()}"
            self._draw_text_with_background(
                cv2, frame, target, origin=(10, 90), font_scale=0.6,
                text_color=(255, 200, 0), bg_color=(0, 0, 0), thickness=2,
            )
            for i, message in enumerate(frame_result.teacher_feedback[:3]):
                self._draw_text_with_background(
                    cv2, frame, message, origin=(10, 115 + 22 * i), font_scale=0.5,
                    text_color=(200, 220, 255), bg_color=(0, 0, 0), thickness=1,
                )

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def _handle_keyboard_input(self) -> None:
        """Poll for a small set of keyboard controls. Only meaningful
        when a GUI window is active (cv2.waitKey is what captures key
        presses)."""
        if not self._gui_available:
            return

        try:
            import cv2

            key = cv2.waitKey(1) & 0xFF
        except Exception:
            return

        if key == ord("q"):
            logger.info("Quit requested via keyboard.")
            self._running = False
        elif key == ord("v"):
            self._pipeline.toggle_voice_mute()
        elif key == ord("c"):
            self._pipeline.clear_current_sentence()
        elif key == ord("d"):
            self._pipeline.trigger_demo_caption()

    def start_teacher_lesson(self, word: str) -> bool:
        """Start a teacher_module.py lesson for `word`, if teacher mode
        is enabled. Returns True if the lesson started successfully."""
        return self._pipeline.start_teacher_lesson(word)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Cleanly release every module's resources. Safe to call even if
        run() was never started or was only partially set up."""
        logger.info("Shutting down SignLanguageAssistantApp...")

        try:
            self._frame_source.release()
        except Exception:
            logger.exception("Error while releasing frame source.")

        self._pipeline.shutdown()

        if not self._headless and self._gui_available:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:
                pass

        logger.info("Shutdown complete.")


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Sign Language Assistant (desktop GUI)")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.json")
    parser.add_argument("--camera-index", type=int, default=None, help="Override camera index")
    parser.add_argument("--no-voice", action="store_true", help="Disable voice assistant")
    parser.add_argument("--no-avatar", action="store_true", help="Disable avatar")
    parser.add_argument("--teacher", type=str, default=None, help="Start in teacher mode for this word")
    parser.add_argument("--headless", action="store_true", help="Run without a GUI window")
    parser.add_argument(
        "--synthetic-frames",
        action="store_true",
        help="Use a synthetic (non-camera) frame source, for testing/CI",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _parse_args(argv)

    config = load_config(path=args.config)
    if args.camera_index is not None:
        config.app.camera_index = args.camera_index
    if args.no_voice:
        config.app.enable_voice_assistant = False
    if args.no_avatar:
        config.app.enable_avatar = False
    if args.teacher:
        config.app.enable_teacher_mode = True

    frame_source = SyntheticFrameSource() if args.synthetic_frames else None

    app = SignLanguageAssistantApp(
        config=config, frame_source=frame_source, headless=args.headless
    )

    if args.teacher:
        app.start_teacher_lesson(args.teacher)

    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
