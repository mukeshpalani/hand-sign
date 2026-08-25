"""
web_app.py

MODULE -- Web GUI Front-End

Single Responsibility
----------------------
This file's ONLY job is: present a BROWSER-BASED front-end for the AI
Sign Language Assistant -- serve a webpage with a live camera feed, a
live caption + voice status panel, and a chat panel wired to
chatbot_module.py's Ollama-backed chatbot, which automatically receives
every finalized signed sentence and replies in text.

Exactly like main.py, this file contains NO detection/recognition/
caption/voice/avatar logic of its own -- all of that lives in
`pipeline.py`'s `SignPipeline`, which both front-ends call into
identically. This file only owns: the Flask HTTP routes, the background
capture thread that feeds them, and routing finalized captions into
chatbot_module.py's ChatbotSession (a front-end-specific choice that
pipeline.py deliberately knows nothing about).

Design notes
------------
- REUSES `SignPipeline` (see pipeline.py): recognition behavior is
  IDENTICAL to the desktop app in main.py; only presentation differs.
- MJPEG STREAMING for video: a background thread continuously reads
  frames, runs them through SignPipeline, draws hand landmarks, and
  JPEG-encodes the result. The browser's `<img src="/video_feed">` tag
  consumes a `multipart/x-mixed-replace` stream -- a well-established,
  dependency-free way to show live video in a plain <img> tag with no
  WebRTC/WebSocket machinery required.
- POLLING, NOT WEBSOCKETS, for text updates: the browser polls
  `/status` and `/chat/history` every few hundred milliseconds. This
  keeps the whole web layer dependency-light (just Flask) at the cost of
  slightly higher latency than a websocket push -- an acceptable
  trade-off for captions/chat text, which don't need frame-perfect timing
  the way video does.
- CAPTIONS ARE KEPT OUT OF THE VIDEO IMAGE: unlike main.py's OpenCV
  overlay, the video stream here shows ONLY the camera feed + hand
  landmark dots. Captions, FPS, and status all render as real HTML/CSS in
  the side panel instead -- which is what makes them controllable,
  styleable, and guaranteed-readable (no more "burned into pixels and
  blending into the background" contrast problem).
- CHAT INTEGRATION: whenever SignPipeline finalizes a caption (a
  completed signed sentence), it's automatically forwarded to
  ChatbotSession.send() on a background thread (so a slow LLM reply never
  stalls the video loop), and the resulting conversation is exposed via
  `/chat/history` for the browser to render. A person can also type
  directly into the chat panel; both paths share the same
  ChatbotSession/history.
- FLASK IS A HARD DEPENDENCY OF THIS FILE ONLY: guarded exactly like
  hand_detection.py's mediapipe requirement -- importing web_app.py
  without Flask installed raises a clear, actionable ImportError, but
  every OTHER module in this project (including main.py, the desktop
  front-end) remains entirely unaffected.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "opencv-python is required for web_app.py. Install it via: "
        "pip install opencv-python"
    ) from exc

try:
    from flask import Flask, Response, jsonify, request, send_from_directory
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "Flask is required for web_app.py (the browser-based front-end). "
        "Install it via: pip install flask. Note: main.py (the desktop "
        "front-end) does NOT need Flask -- this dependency is scoped to "
        "web_app.py only."
    ) from exc

from config import AppConfig, load_config
from pipeline import FrameSource, OpenCVCameraSource, SignPipeline, SyntheticFrameSource
from utils.data_types import Caption
from utils.logger import get_logger

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@dataclass
class WebAppConfig:
    """Tunable parameters for the web server itself. Defined independent
    of any Flask import so config.py can construct this even in an
    environment that hasn't installed Flask -- only actually RUNNING
    web_app.py requires it (see the guarded import above)."""

    host: str = "127.0.0.1"
    port: int = 5000
    jpeg_quality: int = 80  # 0-100, JPEG encoding quality for the video stream
    max_chat_messages: int = 200  # bounds memory for long-running sessions
    debug: bool = False


# ----------------------------------------------------------------------
# Web application
# ----------------------------------------------------------------------


class WebApp:
    """
    Runs SignPipeline against a camera feed in a background thread and
    serves a Flask app exposing: the live video as MJPEG, live
    caption/status as JSON, and a chatbot panel wired to
    chatbot_module.py.

    Usage:
        app = WebApp(config=load_config())
        app.run()   # blocks, serving until Ctrl+C
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        frame_source: Optional[FrameSource] = None,
        chatbot_session: Optional[Any] = None,
    ) -> None:
        """
        Args:
            config: Optional AppConfig. Defaults to config.load_config()
                if omitted.
            frame_source: Optional FrameSource. Defaults to a real
                OpenCVCameraSource. Inject a SyntheticFrameSource for
                testing.
            chatbot_session: Optional chatbot_module.ChatbotSession.
                Defaults to constructing one from config.chatbot (which
                itself falls back to a safe NullChatbotEngine if Ollama
                isn't reachable -- see chatbot_module.py).
        """
        from chatbot_module import ChatbotSession

        self._config = config or load_config()
        self._config.paths.create_all()
        self._web_config: WebAppConfig = self._config.web

        self._pipeline = SignPipeline(self._config)
        self._frame_source = frame_source or OpenCVCameraSource(
            self._config.app.camera_index
        )
        self._chatbot_session = chatbot_session or ChatbotSession(config=self._config.chatbot)

        # Shared state between the background capture thread and Flask's
        # (possibly multiple, since we run threaded) request-handling
        # threads. One lock is enough here: nothing held under it does
        # meaningful work, it just guards a few small pieces of state.
        self._state_lock = threading.Lock()
        self._frame_condition = threading.Condition()
        self._latest_jpeg: Optional[bytes] = None
        self._chat_messages: List[Dict[str, Any]] = []
        self._status: Dict[str, Any] = {
            "fps": None,
            "has_trained_model": self._pipeline.has_trained_model,
            "voice_muted": False,
            "live_caption": "",
            "teacher_word": None,
            "teacher_feedback": [],
        }

        self._stop_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None

        self._verify_web_assets()
        self._flask_app = self._build_flask_app()

        logger.info(
            "WebApp initialized (host=%s, port=%d, has_trained_model=%s).",
            self._web_config.host,
            self._web_config.port,
            self._pipeline.has_trained_model,
        )

    # ------------------------------------------------------------------
    # Flask app / routes
    # ------------------------------------------------------------------

    def _verify_web_assets(self) -> None:
        """
        Check that web/templates/index.html and web/static/{style.css,
        app.js} exist at the expected paths, and log a clear, actionable
        error immediately at startup if any are missing -- rather than
        letting the person only discover a broken page (via a bare 404)
        after they've already opened a browser. A very common way to hit
        this: downloading the project's files individually rather than
        as a single folder, which can silently flatten or rearrange the
        web/templates/ and web/static/ subfolder structure.
        """
        web_root = Path(__file__).resolve().parent / "web"
        expected_files = [
            web_root / "templates" / "index.html",
            web_root / "static" / "style.css",
            web_root / "static" / "app.js",
        ]
        missing = [str(p) for p in expected_files if not p.exists()]

        if missing:
            logger.error(
                "Missing web front-end file(s):\n  %s\n"
                "The folder structure must be EXACTLY (relative to web_app.py):\n"
                "  web/templates/index.html\n"
                "  web/static/style.css\n"
                "  web/static/app.js\n"
                "If you downloaded these files individually rather than as a "
                "whole folder, double-check they landed in these exact "
                "subfolders -- the page will 404 until this is fixed.",
                "\n  ".join(missing),
            )
        else:
            logger.info("Web front-end assets found at %s", web_root)

    def _build_flask_app(self) -> Flask:
        web_root = Path(__file__).resolve().parent / "web"
        app = Flask(
            __name__,
            template_folder=str(web_root / "templates"),
            static_folder=str(web_root / "static"),
        )

        @app.route("/")
        def index():
            index_path = web_root / "templates" / "index.html"
            if not index_path.exists():
                logger.error(
                    "index.html not found at %s -- the web/ folder structure "
                    "must be exactly: web/templates/index.html, "
                    "web/static/style.css, web/static/app.js (relative to "
                    "web_app.py). If you downloaded files individually, "
                    "double-check they landed in these exact subfolders.",
                    index_path,
                )
                return (
                    f"<h1>500: index.html not found</h1>"
                    f"<p>Expected it at: <code>{index_path}</code></p>"
                    f"<p>Check that your folder structure is exactly "
                    f"<code>web/templates/index.html</code>, "
                    f"<code>web/static/style.css</code>, and "
                    f"<code>web/static/app.js</code>, relative to "
                    f"<code>web_app.py</code>.</p>",
                    500,
                )
            return send_from_directory(str(web_root / "templates"), "index.html")

        @app.route("/video_feed")
        def video_feed():
            return Response(
                self._mjpeg_generator(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @app.route("/status")
        def status():
            with self._state_lock:
                return jsonify(dict(self._status))

        @app.route("/chat/history")
        def chat_history():
            with self._state_lock:
                return jsonify({"messages": list(self._chat_messages)})

        @app.route("/chat/send", methods=["POST"])
        def chat_send():
            payload = request.get_json(silent=True) or {}
            text = str(payload.get("text", "")).strip()
            if not text:
                return jsonify({"error": "text is required"}), 400

            reply = self._handle_chat_text(text)
            return jsonify({"reply": reply})

        @app.route("/chat/reset", methods=["POST"])
        def chat_reset():
            self._chatbot_session.reset()
            with self._state_lock:
                self._chat_messages = []
            return jsonify({"ok": True})

        @app.route("/control/mute", methods=["POST"])
        def control_mute():
            muted = self._pipeline.toggle_voice_mute()
            with self._state_lock:
                self._status["voice_muted"] = muted
            return jsonify({"muted": muted})

        @app.route("/control/clear", methods=["POST"])
        def control_clear():
            self._pipeline.clear_current_sentence()
            return jsonify({"ok": True})

        @app.route("/control/demo", methods=["POST"])
        def control_demo():
            demo_caption = self._pipeline.trigger_demo_caption()
            self._dispatch_to_chatbot_async(demo_caption.raw_text)
            return jsonify({"ok": True, "text": demo_caption.raw_text})

        return app

    # ------------------------------------------------------------------
    # Background capture loop
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        logger.info("Web capture loop starting.")
        while not self._stop_event.is_set() and self._frame_source.is_opened():
            frame = self._frame_source.read()
            if frame is None:
                logger.info("Frame source exhausted; stopping capture loop.")
                break

            try:
                self._process_and_publish(frame)
            except Exception:
                logger.exception("Error processing a frame in the web capture loop.")

        logger.info("Web capture loop exiting.")

    def _process_and_publish(self, frame: np.ndarray) -> None:
        frame_result = self._pipeline.process_frame(frame)

        # Draw ONLY hand landmarks into the video image -- captions and
        # status are rendered as real HTML in the side panel instead (see
        # module docstring), so the video stream stays clean.
        annotated = frame_result.frame
        for hand in frame_result.stabilized_result.hands:
            color = (0, 0, 255) if hand.landmarks and hand.landmarks[0].is_estimated else (0, 255, 0)
            for lm in hand.landmarks:
                px = int(lm.x * frame_result.stabilized_result.frame_width)
                py = int(lm.y * frame_result.stabilized_result.frame_height)
                cv2.circle(annotated, (px, py), 3, color, -1)

        success, encoded = cv2.imencode(
            ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, self._web_config.jpeg_quality]
        )
        if success:
            with self._frame_condition:
                self._latest_jpeg = encoded.tobytes()
                self._frame_condition.notify_all()

        if frame_result.caption is not None:
            self._pipeline.dispatch_caption(frame_result.caption)
            if frame_result.caption.is_final:
                # Only FINAL captions go to the chatbot -- sending every
                # in-progress partial word would spam the LLM with
                # incomplete sentences.
                self._dispatch_to_chatbot_async(frame_result.caption.raw_text)

        live_caption = self._pipeline.caption_generator.get_current_live_caption()
        metrics = self._pipeline.get_performance_metrics()
        teacher_module = self._pipeline.teacher_module

        with self._state_lock:
            self._status["fps"] = metrics.fps if metrics else None
            self._status["live_caption"] = live_caption.text if live_caption else ""
            self._status["voice_muted"] = self._pipeline.is_voice_muted
            self._status["teacher_word"] = (
                teacher_module.get_target_word() if teacher_module else None
            )
            self._status["teacher_feedback"] = list(frame_result.teacher_feedback)

    def _mjpeg_generator(self):
        """Generator yielding multipart/x-mixed-replace JPEG chunks,
        waking up efficiently via a Condition variable each time a new
        frame is published rather than busy-polling."""
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while not self._stop_event.is_set():
            with self._frame_condition:
                self._frame_condition.wait(timeout=1.0)
                jpeg = self._latest_jpeg
            if jpeg is None:
                continue
            yield boundary + jpeg + b"\r\n"

    # ------------------------------------------------------------------
    # Chat handling
    # ------------------------------------------------------------------

    def _append_chat_message(self, role: str, text: str) -> None:
        with self._state_lock:
            self._chat_messages.append(
                {"role": role, "text": text, "timestamp_ms": time.time() * 1000.0}
            )
            overflow = len(self._chat_messages) - self._web_config.max_chat_messages
            if overflow > 0:
                self._chat_messages = self._chat_messages[overflow:]

    def _handle_chat_text(self, user_text: str) -> str:
        """Send text to the chatbot and record both sides of the
        exchange. Safe to call from any thread; ChatbotSession.send()
        itself never raises (see chatbot_module.py)."""
        self._append_chat_message("user", user_text)
        reply = self._chatbot_session.send(user_text)
        self._append_chat_message("assistant", reply)
        return reply

    def _dispatch_to_chatbot_async(self, user_text: str) -> None:
        """Fire-and-forget chatbot dispatch for text originating from the
        capture loop -- runs on its own daemon thread so a slow LLM reply
        never stalls frame processing."""
        thread = threading.Thread(
            target=self._handle_chat_text, args=(user_text,), daemon=True
        )
        thread.start()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the background capture thread and serve the Flask app.
        Blocks until interrupted (Ctrl+C) or the server is stopped."""
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        logger.info(
            "Starting web server at http://%s:%d", self._web_config.host, self._web_config.port
        )
        try:
            # threaded=True is required: /video_feed holds a connection
            # open indefinitely, so status/chat requests must be served
            # concurrently rather than queued behind it.
            self._flask_app.run(
                host=self._web_config.host,
                port=self._web_config.port,
                debug=self._web_config.debug,
                threaded=True,
                use_reloader=False,
            )
        except KeyboardInterrupt:
            logger.info("Interrupted by user (Ctrl+C).")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Cleanly stop the capture thread and release every module's
        resources. Safe to call even if run() was never started."""
        logger.info("Shutting down WebApp...")
        self._stop_event.set()

        with self._frame_condition:
            self._frame_condition.notify_all()  # wake the MJPEG generator so it can exit

        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=5.0)

        try:
            self._frame_source.release()
        except Exception:
            logger.exception("Error while releasing frame source.")

        self._pipeline.shutdown()
        logger.info("WebApp shutdown complete.")


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Sign Language Assistant (web GUI)")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.json")
    parser.add_argument("--camera-index", type=int, default=None, help="Override camera index")
    parser.add_argument("--host", type=str, default=None, help="Override web server host")
    parser.add_argument("--port", type=int, default=None, help="Override web server port")
    parser.add_argument("--no-voice", action="store_true", help="Disable voice assistant")
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
    if args.host is not None:
        config.web.host = args.host
    if args.port is not None:
        config.web.port = args.port
    if args.no_voice:
        config.app.enable_voice_assistant = False

    # The web layout requested is camera + caption/voice + chat only, so
    # the avatar isn't shown here by default; SignPipeline still supports
    # it (dispatch_caption() would call it if enabled) for anyone who
    # wants to extend the web UI with an avatar panel later.
    config.app.enable_avatar = False

    frame_source = SyntheticFrameSource() if args.synthetic_frames else None

    app = WebApp(config=config, frame_source=frame_source)
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
