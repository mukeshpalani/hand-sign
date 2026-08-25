# AI Sign Language Assistant

A modular, end-to-end sign language recognition system: live two-hand
detection and tracking, occlusion-aware landmark recovery, continuous
(sentence-level) sign recognition, live captioning, spoken output, an
animated signing avatar, and an interactive teaching mode -- built from
scratch in Python with a strict clean-architecture, SOLID-principles
approach.

Every stage of the pipeline runs **before any ML model is trained or
real hardware is present**, thanks to Null-Object fallbacks throughout
(a `NullSequenceClassifier`, `NullTTSEngine`, `NullAvatarRenderer`,
`PlaceholderAnimationLibrary`, etc.) -- you can run `main.py` on day
one and get a fully working (if not yet "smart") pipeline end to end.

---

## Table of contents

1. [Architecture](#architecture)
2. [Project structure](#project-structure)
3. [Installation](#installation)
4. [Quick start](#quick-start)
5. [Configuration](#configuration)
6. [Building a real recognizer: collect -> augment -> train -> evaluate](#building-a-real-recognizer)
7. [Teacher mode](#teacher-mode)
8. [Web front-end (camera + captions + Ollama chat)](#web-front-end-camera--captions--ollama-chat)
9. [Design principles](#design-principles)
10. [Extending the system](#extending-the-system)
11. [Known limitations](#known-limitations)

---

## Architecture

```
Camera Frame
   |
   v
hand_detection.py        -> HandDetectionResult (21 landmarks x up to 2 hands)
   |
   v
overlap_resolution.py    -> stabilized HandDetectionResult
   |                          (occlusion-recovered, jitter-smoothed)
   v
continuous_recognition.py -> RecognitionResult (recognized words -> sentence)
   |
   v
caption_generator.py     -> Caption (live + grammar-corrected final captions)
   |
   +--------------------------+--------------------------+
   v                          v                          v
voice_assistant.py       avatar_module.py         chatbot_module.py
(speaks final captions)  (signs the sentence      (sends final captions to a
                          back, synced facial       local Ollama model, gets
                          expressions + natural     a text reply back --
                          blended transitions)       web_app.py only)
```

All of the above is orchestrated by **`pipeline.py`**'s `SignPipeline` --
the one place that constructs every module and runs one frame through
the full chain. Two independent front-ends sit on top of it:

- **`main.py`** -- desktop OpenCV window (captions/status drawn as
  overlays on the video).
- **`web_app.py`** -- browser UI (camera feed left, live caption + voice
  status panel right, Ollama chat panel below it -- see
  [Web front-end](#web-front-end-camera--captions--ollama-chat)).

Neither front-end contains any detection/recognition/caption logic of
its own -- both call into the same `SignPipeline`, so recognition
behavior is identical between them; only *how things are displayed*
differs. This is also why the "GUI" project requirement is satisfied
twice over without duplicating any pipeline code.

Supporting subsystems, used around (not inside) the live loop:

- **`teacher_module.py`** -- shows a target sign (via `avatar_module.py`),
  compares the learner's live landmarks against it, scores the attempt,
  and gives specific feedback ("Straighten your index finger.").
- **`dataset_manager.py`** -- collects, labels, augments, splits, and
  versions training data.
- **`model_training.py`** -- trains the recognition model from
  `dataset_manager.py`'s data and exports it (TFLite / ONNX). Its output
  plugs directly into `continuous_recognition.py` as a real
  `SequenceClassifier`.
- **`evaluation.py`** -- accuracy/precision/recall/F1/confusion matrix
  for model quality, plus latency/FPS profiling for runtime performance.
- **`config.py`** -- the single source of truth for every module's
  tunable parameters and the shared folder layout.

Every arrow above is a **one-way, typed data contract** defined in
`utils/data_types.py` (`HandDetectionResult`, `RecognitionResult`,
`Caption`, etc.) -- no module reaches backward into an earlier stage, and
no module imports a concrete third-party library (MediaPipe, TensorFlow,
pyttsx3, OpenCV, Flask) except the one file responsible for wrapping it.

---

## Project structure

```
AI_Sign_Assistant/
    main.py                  # desktop front-end: OpenCV window, GUI, keyboard controls
    web_app.py                # browser front-end: Flask server, MJPEG stream, chat routes
    pipeline.py                # SignPipeline -- shared by BOTH front-ends (avoids duplicate code)
    hand_detection.py        # Module 1: MediaPipe Hands wrapper
    overlap_resolution.py    # Module 2: occlusion recovery + smoothing
    continuous_recognition.py# Module 3: sliding-window sign recognition
    teacher_module.py        # Module 5: lessons, scoring, feedback
    avatar_module.py         # Module 7: sign animation playback
    caption_generator.py     # Module 4: live + grammar-corrected captions
    voice_assistant.py       # Module 6: threaded TTS with pause/resume
    chatbot_module.py         # Ollama-backed chat assistant (web_app.py only)
    dataset_manager.py       # Module 8: data collection/augmentation/versioning
    model_training.py        # Module 9: training, checkpoints, export
    evaluation.py             # Module 10: metrics + performance profiling
    config.py                 # central configuration hub
    requirements.txt
    README.md
    models/                   # trained model exports (.tflite/.onnx) live here
    datasets/                 # versioned training data lives here
    assets/                   # general static assets
    avatar/                   # per-word animation clip JSON files
    sounds/                   # any bundled audio assets
    checkpoints/              # training checkpoints (resume support)
    web/
        templates/index.html   # web_app.py's page structure
        static/style.css        # web_app.py's dark theme + layout
        static/app.js            # web_app.py's status/chat polling logic
    utils/
        __init__.py
        data_types.py          # shared DTOs used across every module
        logger.py               # centralized logging setup
```

---

## Installation

Requires **Python 3.9+**.

```bash
git clone <this-repo>
cd AI_Sign_Assistant
pip install -r requirements.txt
```

Notes:
- `mediapipe` and `opencv-python` are **required** -- there's no
  fallback for camera-based hand detection itself.
- `pyttsx3` (voice) and `tensorflow`/`tf2onnx` (training/export) are
  **soft** dependencies -- the app runs and logs a clear warning instead
  of crashing if they're missing; see [Known limitations](#known-limitations).
- Linux users: `pyttsx3` needs the system `espeak` package:
  `sudo apt-get install espeak`.

---

## Quick start

```bash
python main.py
```

Prefer a browser UI with a built-in Ollama chatbot instead of the
desktop window? See [Web front-end](#web-front-end-camera--captions--ollama-chat)
-- run `python web_app.py` instead and open http://127.0.0.1:5000.

On first run this creates `config.json` (see [Configuration](#configuration))
and opens a webcam window showing live hand-landmark tracking, a caption
bar, and an FPS counter. Since no model has been trained yet, recognition
runs in a safe no-op mode (`NullSequenceClassifier`) -- the pipeline is
fully wired and running, it just won't recognize real words until you
[train a model](#building-a-real-recognizer).

Useful flags:

```bash
python main.py --no-voice --no-avatar     # detection/captions only
python main.py --camera-index 1           # use a different webcam
python main.py --teacher hello            # start a teaching lesson for "hello"
python main.py --headless --synthetic-frames  # for CI/testing, no camera/display needed
```

In-window controls: `q` quit, `v` mute/unmute voice, `c` clear the
current sentence, `d` inject a demo caption to test voice/avatar/caption
output immediately -- useful before you've trained a model, since
recognition itself won't produce any captions until then (see
[Building a real recognizer](#building-a-real-recognizer)).

---

## Configuration

All tunable parameters live in `config.json` (auto-created on first run
from every module's built-in defaults). Edit it directly, or override a
couple of common settings via environment variables:

```bash
SIGN_ASSISTANT_CAMERA_INDEX=1 SIGN_ASSISTANT_LOG_LEVEL=DEBUG python main.py
```

Each top-level section of `config.json` maps 1:1 to a module's own
`Config` dataclass (e.g. `continuous_recognizer.min_confidence` controls
`continuous_recognition.py`'s `ContinuousRecognizerConfig.min_confidence`)
-- see each module's file for the full, documented list of parameters.
This includes `chatbot` (Ollama host/model/system prompt -- see
[Web front-end](#web-front-end-camera--captions--ollama-chat)) and `web`
(the Flask server's host/port/JPEG quality), even though those only take
effect when running `web_app.py`.

---

## Building a real recognizer

The pipeline runs with no-op recognition out of the box. To make it
actually recognize signs:

**1. Collect data** (per sign, record several performances as landmark
feature sequences and add them via `dataset_manager.py`):

```python
from dataset_manager import DatasetManager
manager = DatasetManager()
manager.add_sample("hello", feature_sequence)   # shape: (num_frames, 126)
```

Feature sequences use the exact layout `continuous_recognition.py`
produces at inference time (`LandmarkFeatureExtractor`) -- both hands'
21 landmarks x (x, y, z), zero-padded for a missing hand. Remember to
also record samples under the `"_background_"` label (no sign being
performed), so the trained model can output "nothing recognized" rather
than always guessing a word.

**2. Augment** (optional but recommended for small datasets):

```python
from dataset_manager import GaussianNoiseAugmentation, HorizontalFlipAugmentation
manager.augment_dataset([GaussianNoiseAugmentation(), HorizontalFlipAugmentation()])
```

**3. Split and train:**

```python
from model_training import ModelTrainer
trainer = ModelTrainer(manager, run_id="v1")
metrics = trainer.train()          # resumable: re-run to continue from last checkpoint
trainer.export_tflite()
```

**4. Wire the trained model into live recognition** by placing the
exported model + labels under `models/current/` and re-launching
`main.py` -- it auto-detects and loads a trained model if present (see
`SignLanguageAssistantApp._try_load_trained_classifier`).

**5. Evaluate:**

```python
from evaluation import ClassificationEvaluator
evaluator = ClassificationEvaluator(labels=manager.get_labels())
report = evaluator.evaluate_classifier_on_dataset(
    trainer.get_sequence_classifier(), manager, split.test_sample_ids
)
print(report.confusion_matrix_as_text())
```

---

## Teacher mode

```bash
python main.py --teacher hello
```

The avatar demonstrates the target sign; as you attempt it, the on-screen
overlay shows a correction score and specific feedback ("Raise your
thumb.", "Rotate your wrist slightly."). Progress and history are tracked
per word (`TeacherModule.get_progress("hello")`).

---

## Web front-end (camera + captions + Ollama chat)

In addition to `main.py`'s desktop OpenCV window, this project includes a
browser-based front-end: camera feed on the left, a live caption + voice
status panel on the right, and a chat panel below it wired to a
**locally-running Ollama model**. Whenever you finish signing a sentence,
it's automatically sent to Ollama, and the reply appears in the chat
panel as text.

This reuses every pipeline module exactly as-is -- `web_app.py` and
`main.py` both build on the same `pipeline.SignPipeline` (see
[Design principles](#design-principles)), so recognition behavior is
identical between the two; only how things are *displayed* differs.
`chatbot_module.py` is the new module handling the conversational side.

**Setup:**

```bash
pip install -r requirements.txt   # now includes flask
ollama serve                      # in a separate terminal, if not already running
ollama pull llama3.2              # or whichever model you want (update config.json's chatbot.model to match)
python web_app.py
```

Then open **http://127.0.0.1:5000** in a browser.

**How it works:**
- A background thread reads frames from your webcam (server-side, via
  the same `OpenCVCameraSource` `main.py` uses), runs them through
  `SignPipeline`, and streams the annotated result to the browser's
  `<img src="/video_feed">` tag as an MJPEG stream -- no WebRTC/WebSocket
  setup required.
- The video image shows only the camera feed plus hand-landmark dots
  (green = directly observed, red = being reconstructed by
  `overlap_resolution.py` during a hand-occlusion event). Captions and
  status render as real HTML in the side panel instead of being drawn
  into the video pixels -- this is also what fixes the "caption text
  blending into the background" contrast problem from the desktop GUI.
- The browser polls `GET /status` (~2-3x/sec) for the live caption, FPS,
  and whether a trained model is loaded, and polls `GET /chat/history`
  for new chat messages -- deliberately simple HTTP polling rather than
  WebSockets, keeping the whole web layer's only dependency at Flask.
- When `caption_generator.py` finalizes a sentence, `web_app.py`
  automatically forwards it to `chatbot_module.py`'s `ChatbotSession` on
  a background thread (so a slow LLM reply never stalls the video loop);
  both your sentence and the reply show up in the chat panel.
- **Voice output is server-side** (the same `voice_assistant.py`/pyttsx3
  used by `main.py`) -- appropriate since this is designed for local use
  (you and the browser are on the same machine), not a multi-user remote
  deployment.
- You can also type directly into the chat box -- useful for testing the
  Ollama connection before you've trained a recognition model, since (as
  with `main.py`) recognition itself won't produce any captions until you
  have. The **"Test Voice / Chat" button** does the same thing `main.py`'s
  `d` key does: injects a demo caption that exercises voice + chat output
  immediately, without needing a trained model or performing a real sign.

**Files added for this:**

```
pipeline.py                # SignPipeline -- shared by main.py AND web_app.py (see Design principles)
chatbot_module.py          # Ollama chat client (Strategy + Null-Object pattern, like this project's other backends)
web_app.py                 # Flask backend: capture thread, MJPEG stream, status/chat routes
web/templates/index.html   # page structure
web/static/style.css       # dark theme, camera-left / caption+chat-right layout
web/static/app.js          # status/chat polling, button + chat-form wiring
```

**Note:** the avatar isn't shown in the web layout by default (the
requested design was camera + caption/voice + chat only) -- `web_app.py`
sets `config.app.enable_avatar = False` for its own session. `SignPipeline`
still fully supports it, so a future avatar panel can be added to
`web_app.py` without touching any pipeline code.

---

## Design principles

- **Single Responsibility** -- each file does exactly one job (see each
  module's own docstring for its declared responsibility).
- **Dependency Inversion / Strategy Pattern** -- every swappable backend
  (TTS engine, sequence classifier, animation renderer, grammar
  corrector) sits behind an abstract interface; concrete third-party
  libraries are wrapped in exactly one file each.
- **Null-Object Pattern** -- every pluggable interface has a safe,
  no-op default (`NullSequenceClassifier`, `NullTTSEngine`,
  `NullAvatarRenderer`, `PlaceholderAnimationLibrary`), so the full
  pipeline runs end-to-end before real models/assets/hardware exist.
- **Shared DTOs, not duplicated structs** -- `utils/data_types.py` is
  the one place inter-module data shapes are defined, extended
  incrementally as new modules needed new contracts.
- **Fail-safe hot paths** -- every per-frame method wraps its core logic
  in a top-level `try/except` so one bad frame never crashes the live
  video loop.

---

## Extending the system

- **New sign language / vocabulary**: add labels via
  `DatasetManager.add_label()` and drop new animation clips into
  `avatar/<word>.json` -- no code changes needed.
- **Different model architecture**: change only
  `ModelTrainer._build_model()`; every other part of training
  (checkpointing, export, evaluation) is architecture-agnostic.
- **Different TTS/avatar backend**: implement `TTSEngine` /
  `AvatarRenderer` and pass an instance into `VoiceAssistant` /
  `AvatarController` -- nothing else changes.
- **Real fingerspelling**: `avatar_module.py`'s
  `AvatarController._resolve_fallback_clip()` is the documented
  extension point for replacing the placeholder fallback with a real
  letter-by-letter fingerspelling generator.
- **Mobile / edge deployment**: `model_training.py` already exports
  TensorFlow Lite; `continuous_recognition.py`'s `SequenceClassifier`
  interface is backend-agnostic, so a TFLite-Runtime-based
  implementation can be swapped in for constrained devices.

---

## Known limitations

- **MediaPipe model download on first run**: `hand_detection.py` uses
  Google's current, recommended Hand Landmarker **Tasks API**
  (`mediapipe.tasks.python.vision.HandLandmarker`) rather than the
  legacy, now-deprecated `mediapipe.solutions.hands` API (many current
  `pip install mediapipe` builds no longer ship `solutions` at all,
  which raised `AttributeError: module 'mediapipe' has no attribute
  'solutions'` under the old implementation). The Tasks API needs a
  small (~10MB) `hand_landmarker.task` model file, which
  `HandDetector` downloads automatically to `models/hand_landmarker.task`
  the first time it runs, and reuses afterward. If your machine has no
  internet access, download it manually from the URL in
  `hand_detection.DEFAULT_MODEL_URL` and place it at that path (or point
  `HandDetectorConfig.model_asset_path` at wherever you saved it).
- **Teacher mode compares against a single static target pose** (the
  target clip's final frame) rather than the full dynamic motion of a
  sign -- fine for held/static signs, less precise for highly dynamic
  ones.
- **Two-handed sign comparison in teacher mode** currently scores the
  first comparable hand found rather than combining both hands into one
  score -- see `TeacherModule._evaluate_impl` for the documented
  extension point.
- **Avatar animation library ships with a procedural placeholder only**
  (`PlaceholderAnimationLibrary`) -- real signing requires authoring (or
  motion-capturing) actual clips into `avatar/*.json` per the schema
  documented in `avatar_module.py`.
- **No speech-to-text / voice command input** is implemented (the
  project's tech-stack notes mention `SpeechRecognition` as available
  tooling, but no requirement called for voice-driven control of the
  app itself).
