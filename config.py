"""
config.py

Central Configuration

Single Responsibility
----------------------
This file's ONLY job is: be the SINGLE SOURCE OF TRUTH for every tunable
parameter across the whole AI Sign Language Assistant, and for the
project's shared folder layout. It does not contain any pipeline logic
itself -- it just aggregates each module's own `Config` dataclass (every
module already defines one, per this project's cross-cutting design
rules) into one `AppConfig`, and provides load/save helpers so the whole
system can be re-tuned by editing a JSON file instead of touching code.

main.py is expected to call `load_config()` once at startup and pass the
relevant section of the resulting `AppConfig` into each module's
constructor (e.g. `HandDetector(config=app_config.hand_detector)`).

Design notes
------------
- COMPOSITION, NOT DUPLICATION: this file does not redefine any tunable
  parameter that already lives in a module's own Config dataclass (that
  would violate the project's "avoid duplicate code" rule and risk the
  two copies drifting apart). It only adds ONE new thing modules don't
  already have: `PathsConfig`, the shared directory layout, plus a small
  top-level `AppSettings` for options that donapply to any single module
  (camera index, target FPS, feature toggles).
- PATH RECONCILIATION: several modules' own Config dataclasses (e.g.
  DatasetManagerConfig.dataset_root, ModelTrainingConfig.checkpoint_root)
  default to relative paths like "datasets/" or "checkpoints/" defined
  independently in their own files. `AppConfig.create_default()`
  overwrites those with paths derived from ONE shared `PathsConfig`, so
  there is exactly one place that decides where things live on disk.
- LAZY, GUARDED IMPORTS FOR HARD-DEPENDENCY MODULES: hand_detection.py
  raises ImportError immediately at module load if `mediapipe` isn't
  installed (appropriately, since hand detection cannot function at all
  without it). To keep config.py itself always importable -- even in an
  environment that only wants to inspect/edit configuration, not run the
  live camera pipeline -- HandDetectorConfig is imported lazily inside
  `AppConfig.create_default()` rather than at module load time, with a
  clear, actionable error if mediapipe truly isn't available when a
  default config is actually being constructed.
- GENERIC, TYPE-SAFE OVERRIDES: `_apply_overrides()` merges a loaded
  JSON dict onto a dataclass instance by inspecting each field's ACTUAL
  RUNTIME VALUE (not its string type annotation, which several imported
  modules' `from __future__ import annotations` would otherwise turn
  into unreliable strings) -- so Path-typed fields are correctly
  re-wrapped in `Path(...)` after a JSON round-trip, generically, without
  needing a hand-written loader per module.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# Shared paths
# ----------------------------------------------------------------------


@dataclass
class PathsConfig:
    """
    The project's shared on-disk folder layout, matching the structure
    specified for AI_Sign_Assistant/. This is the ONE place that decides
    where each kind of file lives -- every module-specific Config that
    also has a path field (DatasetManagerConfig, ModelTrainingConfig)
    gets that field synchronized to these values by
    AppConfig.create_default(), rather than deciding independently.
    """

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)

    models_dir: Optional[Path] = None
    datasets_dir: Optional[Path] = None
    assets_dir: Optional[Path] = None
    avatar_assets_dir: Optional[Path] = None
    sounds_dir: Optional[Path] = None
    checkpoints_dir: Optional[Path] = None
    logs_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        """Fill in any unset directory as a sensible subfolder of
        project_root, so callers only need to override what they
        actually want to change."""
        self.models_dir = self.models_dir or self.project_root / "models"
        self.datasets_dir = self.datasets_dir or self.project_root / "datasets"
        self.assets_dir = self.assets_dir or self.project_root / "assets"
        self.avatar_assets_dir = self.avatar_assets_dir or self.project_root / "avatar"
        self.sounds_dir = self.sounds_dir or self.project_root / "sounds"
        self.checkpoints_dir = self.checkpoints_dir or self.project_root / "checkpoints"
        self.logs_dir = self.logs_dir or self.project_root / "logs"

    def create_all(self) -> None:
        """Create every configured directory if it doesn't already
        exist. Safe to call repeatedly (idempotent)."""
        for directory in (
            self.models_dir,
            self.datasets_dir,
            self.assets_dir,
            self.avatar_assets_dir,
            self.sounds_dir,
            self.checkpoints_dir,
            self.logs_dir,
        ):
            Path(directory).mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# App-wide settings that don't belong to any single module
# ----------------------------------------------------------------------


@dataclass
class AppSettings:
    """Top-level options that apply to the application as a whole rather
    than to any one pipeline module."""

    camera_index: int = 0
    target_fps: int = 30
    window_title: str = "AI Sign Language Assistant"
    log_level: str = "INFO"

    # MediaPipe's handedness classifier assumes the input image is
    # mirrored (i.e. a front-facing/selfie camera flipped horizontally --
    # this is documented MediaPipe behavior, not a bug in this project).
    # If the raw, unmirrored camera frame is fed to it directly, LEFT and
    # RIGHT hands come out swapped. Mirroring the frame here (before both
    # detection AND display) fixes handedness AND gives the natural
    # "selfie view" users expect from a webcam app -- moving your right
    # hand right also moves it right on screen. Set to False only if your
    # camera feed is already mirrored upstream, or you specifically want
    # the raw, unmirrored feed.
    mirror_camera: bool = True

    # Feature toggles -- lets main.py (or a lightweight deployment) skip
    # constructing modules that aren't needed, e.g. running headless
    # without voice output or without the avatar.
    enable_voice_assistant: bool = True
    enable_avatar: bool = True
    enable_teacher_mode: bool = False


# ----------------------------------------------------------------------
# Aggregated application configuration
# ----------------------------------------------------------------------


@dataclass
class AppConfig:
    """
    The full application configuration: shared paths, app-wide settings,
    and every pipeline module's own Config dataclass, all in one place.

    main.py should construct this ONCE at startup via `load_config()`
    and pass each `app_config.<module>` section into the corresponding
    module's constructor.
    """

    paths: PathsConfig = field(default_factory=PathsConfig)
    app: AppSettings = field(default_factory=AppSettings)

    # NOTE: these are typed as `Any` rather than each module's concrete
    # Config class, specifically so this file can be IMPORTED without
    # requiring every optional dependency (mediapipe, tensorflow, etc.)
    # to be installed -- concrete types are only touched inside
    # create_default(), where they're imported lazily. main.py, which
    # DOES have all dependencies installed, still gets full concrete
    # types at the call site since create_default() returns real
    # instances, not `Any` values.
    hand_detector: Any = None
    overlap_resolver: Any = None
    continuous_recognizer: Any = None
    caption_generator: Any = None
    teacher_module: Any = None
    voice_assistant: Any = None
    avatar_controller: Any = None
    dataset_manager: Any = None
    model_training: Any = None
    chatbot: Any = None
    web: Any = None

    @classmethod
    def create_default(cls, paths: Optional[PathsConfig] = None) -> "AppConfig":
        """
        Build a fully-populated AppConfig using every module's own
        default Config values, with path-bearing fields synchronized to
        a shared PathsConfig.

        Args:
            paths: Optional PathsConfig to use instead of the default
                (project-root-relative) layout.

        Returns:
            A ready-to-use AppConfig.

        Raises:
            ImportError: If a module with a hard external dependency
                (currently only hand_detection.py's mediapipe
                requirement) can't be imported. The error message from
                that module's own guard (see hand_detection.py) explains
                how to install it.
        """
        paths = paths or PathsConfig()

        # Lazy imports: each of these pulls in a real module file, some
        # of which have hard (hand_detection.py) or soft/guarded
        # (voice_assistant.py, model_training.py) external dependencies.
        # Importing here rather than at this file's top level keeps
        # config.py itself always importable.
        from hand_detection import HandDetectorConfig
        from overlap_resolution import OverlapResolverConfig
        from continuous_recognition import ContinuousRecognizerConfig
        from caption_generator import CaptionGeneratorConfig
        from teacher_module import TeacherModuleConfig
        from voice_assistant import VoiceAssistantConfig
        from avatar_module import AvatarControllerConfig
        from dataset_manager import DatasetManagerConfig
        from model_training import ModelTrainingConfig
        from chatbot_module import ChatbotModuleConfig
        from web_app import WebAppConfig

        dataset_manager_config = DatasetManagerConfig(root_dir=paths.datasets_dir)
        model_training_config = ModelTrainingConfig(
            dataset_root=paths.datasets_dir,
            checkpoint_root=paths.checkpoints_dir,
            model_output_root=paths.models_dir,
        )

        return cls(
            paths=paths,
            app=AppSettings(),
            hand_detector=HandDetectorConfig(),
            overlap_resolver=OverlapResolverConfig(),
            continuous_recognizer=ContinuousRecognizerConfig(),
            caption_generator=CaptionGeneratorConfig(),
            teacher_module=TeacherModuleConfig(),
            voice_assistant=VoiceAssistantConfig(),
            avatar_controller=AvatarControllerConfig(),
            dataset_manager=dataset_manager_config,
            model_training=model_training_config,
            chatbot=ChatbotModuleConfig(),
            web=WebAppConfig(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this AppConfig into a plain, JSON-safe dict (Path
        objects become strings)."""
        return _to_json_safe(dataclasses.asdict(self))

    def save(self, path: Path) -> None:
        """Write this configuration to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved configuration to %s", path)


# ----------------------------------------------------------------------
# Serialization helpers
# ----------------------------------------------------------------------


def _to_json_safe(value: Any) -> Any:
    """Recursively convert Path objects (and anything nested in
    dicts/lists) into JSON-serializable equivalents."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(v) for v in value]
    return value


def _apply_overrides(instance: Any, overrides: Dict[str, Any]) -> Any:
    """
    Return a COPY of `instance` (a dataclass) with fields from
    `overrides` applied, converting values back to `Path` wherever the
    instance's CURRENT value for that field is a Path -- this sidesteps
    needing each field's static type annotation (which several imported
    modules' `from __future__ import annotations` turn into plain
    strings at runtime, making naive type-based reconstruction unreliable).

    Unknown keys in `overrides` (fields that don't exist on `instance`,
    e.g. from a config file written by an older version of a module) are
    logged and skipped rather than raising, so old config files degrade
    gracefully instead of crashing startup.
    """
    updates: Dict[str, Any] = {}
    for key, raw_value in overrides.items():
        if not hasattr(instance, key):
            logger.warning(
                "Ignoring unknown config field '%s' on %s (config file may be "
                "from a different version of this project).",
                key,
                type(instance).__name__,
            )
            continue

        current_value = getattr(instance, key)
        updates[key] = Path(raw_value) if isinstance(current_value, Path) else raw_value

    return dataclasses.replace(instance, **updates)


# ----------------------------------------------------------------------
# Load / save entry points
# ----------------------------------------------------------------------


def get_default_config_path() -> Path:
    """The conventional location for this project's config file:
    <project_root>/config.json."""
    return Path(__file__).resolve().parent / "config.json"


def load_config(path: Optional[Path] = None, create_if_missing: bool = True) -> AppConfig:
    """
    Load the application configuration, starting from every module's
    built-in defaults and overlaying any values found in a JSON config
    file.

    Args:
        path: Path to the config JSON file. Defaults to
            get_default_config_path() if omitted.
        create_if_missing: If True (default) and no config file exists
            yet, a fresh one is written to `path` using pure defaults --
            so the first run of the application always produces an
            editable config.json, and subsequent runs pick up any
            manual edits.

    Returns:
        A fully populated AppConfig.
    """
    path = path or get_default_config_path()
    default_config = AppConfig.create_default()

    if not path.exists():
        if create_if_missing:
            default_config.save(path)
            logger.info(
                "No config file found at %s; created one with default values.", path
            )
        else:
            logger.info("No config file found at %s; using in-memory defaults.", path)
        return default_config

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.exception(
            "Failed to read config file at %s; falling back to defaults.", path
        )
        return default_config

    try:
        merged = AppConfig(
            paths=_apply_overrides(default_config.paths, raw.get("paths", {})),
            app=_apply_overrides(default_config.app, raw.get("app", {})),
            hand_detector=_apply_overrides(
                default_config.hand_detector, raw.get("hand_detector", {})
            ),
            overlap_resolver=_apply_overrides(
                default_config.overlap_resolver, raw.get("overlap_resolver", {})
            ),
            continuous_recognizer=_apply_overrides(
                default_config.continuous_recognizer, raw.get("continuous_recognizer", {})
            ),
            caption_generator=_apply_overrides(
                default_config.caption_generator, raw.get("caption_generator", {})
            ),
            teacher_module=_apply_overrides(
                default_config.teacher_module, raw.get("teacher_module", {})
            ),
            voice_assistant=_apply_overrides(
                default_config.voice_assistant, raw.get("voice_assistant", {})
            ),
            avatar_controller=_apply_overrides(
                default_config.avatar_controller, raw.get("avatar_controller", {})
            ),
            dataset_manager=_apply_overrides(
                default_config.dataset_manager, raw.get("dataset_manager", {})
            ),
            model_training=_apply_overrides(
                default_config.model_training, raw.get("model_training", {})
            ),
            chatbot=_apply_overrides(default_config.chatbot, raw.get("chatbot", {})),
            web=_apply_overrides(default_config.web, raw.get("web", {})),
        )
    except Exception:
        logger.exception(
            "Failed to merge config file at %s with defaults; falling back to "
            "pure defaults.",
            path,
        )
        return default_config

    _apply_env_overrides(merged)
    logger.info("Loaded configuration from %s", path)
    return merged


# ----------------------------------------------------------------------
# Environment variable overrides (small, explicit allowlist)
# ----------------------------------------------------------------------


def _apply_env_overrides(config: AppConfig) -> None:
    """
    Apply a small, deliberately limited set of environment-variable
    overrides on top of a loaded config -- useful for containerized/CI
    deployments where re-specifying a whole JSON file just to change the
    camera index or log level is inconvenient. Mutates `config` in place.

    Recognized variables:
        SIGN_ASSISTANT_CAMERA_INDEX -- overrides app.camera_index
        SIGN_ASSISTANT_LOG_LEVEL    -- overrides app.log_level
    """
    camera_index_env = os.environ.get("SIGN_ASSISTANT_CAMERA_INDEX")
    if camera_index_env is not None:
        try:
            config.app.camera_index = int(camera_index_env)
            logger.info(
                "Overriding camera_index from environment: %d", config.app.camera_index
            )
        except ValueError:
            logger.warning(
                "Ignoring invalid SIGN_ASSISTANT_CAMERA_INDEX value: %r", camera_index_env
            )

    log_level_env = os.environ.get("SIGN_ASSISTANT_LOG_LEVEL")
    if log_level_env is not None:
        config.app.log_level = log_level_env.upper()
        logger.info("Overriding log_level from environment: %s", config.app.log_level)


if __name__ == "__main__":
    # Minimal manual smoke-test: builds a default config in a temp
    # directory, saves it, reloads it, tweaks a value, saves again, and
    # verifies the override round-trips correctly. Run via:
    #   python config.py
    import shutil
    import tempfile

    logger.info("Running config.py standalone demo.")

    temp_dir = Path(tempfile.mkdtemp(prefix="config_demo_"))
    try:
        config_path = temp_dir / "config.json"
        paths = PathsConfig(project_root=temp_dir)

        config = AppConfig.create_default(paths=paths)
        config.paths.create_all()
        config.save(config_path)
        print("Saved default config to:", config_path)
        print("Datasets dir:", config.dataset_manager.root_dir)
        print("Camera index (default):", config.app.camera_index)

        # Simulate a user manually editing config.json to change the
        # camera index and a recognition threshold.
        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        raw["app"]["camera_index"] = 2
        raw["continuous_recognizer"]["min_confidence"] = 0.9
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2)

        reloaded = load_config(config_path)
        print("\nAfter simulated manual edit + reload:")
        print("Camera index:", reloaded.app.camera_index)
        print("Recognition min_confidence:", reloaded.continuous_recognizer.min_confidence)
        print("Datasets dir still synced to paths:", reloaded.dataset_manager.root_dir)

        # Verify an unknown/stale field in the file is skipped gracefully.
        raw["app"]["some_field_from_a_future_version"] = 123
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2)
        reloaded_again = load_config(config_path)
        print("\nGracefully handled unknown field; camera_index still:", reloaded_again.app.camera_index)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    logger.info("Demo complete.")
