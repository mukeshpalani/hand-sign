"""
model_training.py

MODULE 9 -- Model Training

Single Responsibility
----------------------
This module's ONLY job is: train the sign-recognition SEQUENCE MODEL from
data owned by dataset_manager.py, manage checkpoints (including resuming
an interrupted run), export the trained model to deployment formats
(TensorFlow Lite for edge/mobile, ONNX for cross-framework inference), and
report training/evaluation metrics (project requirement: model_training.py
-- "Train recognition model / Save checkpoints / Resume training / Export
ONNX / TensorFlow Lite / Metrics").

It does NOT collect or store data (dataset_manager.py's job) and does NOT
run live inference (continuous_recognition.py's job) -- but it is the
BRIDGE between the two: it reads samples via a DatasetManager instance,
and produces a `TrainedSequenceClassifier` that implements
continuous_recognition.py's `SequenceClassifier` interface, ready to be
injected straight into `ContinuousRecognizer` to close the full pipeline
loop (collect -> train -> recognize).

Design notes
------------
- FRAMEWORK CHOICE: TensorFlow/Keras, chosen specifically because the
  project requires TensorFlow Lite export -- TFLite conversion is native
  to TensorFlow's toolchain, so building the model in Keras avoids an
  extra cross-framework conversion step. ONNX export is still supported
  (via the optional `tf2onnx` package) for the project's "ONNX (optional)"
  requirement and for teams wanting a framework-agnostic deployment path.
- GUARDED IMPORT (Null-Object-adjacent pattern, but for a hard
  dependency): unlike voice_assistant.py or continuous_recognition.py,
  training genuinely CANNOT happen without a real ML framework -- there's
  no meaningful "null" trainer. So `tensorflow` is imported guardedly at
  module load time (this file stays importable even without TensorFlow
  installed, e.g. so other modules' type hints/tests aren't blocked), but
  `ModelTrainer.__init__` raises a clear, actionable RuntimeError if
  TensorFlow is missing when someone actually tries to train -- mirroring
  the exact pattern voice_assistant.py's Pyttsx3TTSEngine uses.
- CHECKPOINT/RESUME: every epoch's weights are saved to
  checkpoints/<run_id>/, alongside a small `training_state.json` tracking
  the last completed epoch and best validation accuracy so far. Calling
  `train()` again on the same run_id automatically resumes from the last
  checkpoint rather than starting over -- important for long training
  jobs on unreliable hardware.
- SHARED CONTRACT: the model's input shape is
  (window_size, FEATURE_VECTOR_SIZE), and window_size is taken from
  continuous_recognition.py's ContinuousRecognizerConfig default (30) by
  default -- keeping training and live-inference windowing in sync is
  documented explicitly, since a mismatch here would silently break
  real-time recognition.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from continuous_recognition import (
    FEATURE_VECTOR_SIZE,
    ContinuousRecognizerConfig,
    SequenceClassifier,
)
from dataset_manager import DatasetManager
from utils.logger import get_logger

logger = get_logger(__name__)


try:
    import tensorflow as tf
except ImportError:
    tf = None  # Handled gracefully; see ModelTrainer.__init__ / TrainedSequenceClassifier.


# Background/idle label: training data SHOULD include samples of "no sign
# being performed" under this label, so the trained model can output an
# explicit "nothing recognized" prediction (mirroring
# NullSequenceClassifier's contract of returning "" for the idle case)
# rather than being forced to always guess a real word.
BACKGROUND_LABEL = "_background_"


# ----------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------


@dataclass
class ModelTrainingConfig:
    """Tunable hyperparameters and paths for training."""

    dataset_root: Path = field(default_factory=lambda: Path("datasets"))
    checkpoint_root: Path = field(default_factory=lambda: Path("checkpoints"))
    model_output_root: Path = field(default_factory=lambda: Path("models"))

    # MUST match continuous_recognition.py's live-inference window size,
    # or the trained model's input shape won't match what
    # ContinuousRecognizer feeds it at inference time.
    window_size: int = ContinuousRecognizerConfig().window_size

    lstm_units: int = 64
    dense_units: int = 32
    dropout_rate: float = 0.3
    learning_rate: float = 1e-3
    batch_size: int = 16
    max_epochs: int = 50
    early_stopping_patience: int = 8


@dataclass
class TrainingMetrics:
    """Summary metrics from a training/evaluation run, saved alongside
    each checkpoint for evaluation.py and dashboards to consume."""

    run_id: str
    epochs_completed: int
    final_train_loss: float
    final_train_accuracy: float
    best_val_loss: float
    best_val_accuracy: float
    test_loss: Optional[float] = None
    test_accuracy: Optional[float] = None
    trained_at_ms: float = 0.0
    dataset_version: int = 0
    labels: List[str] = field(default_factory=list)


# ----------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------


class SequenceDataPreparer:
    """
    Converts DatasetManager samples (variable-length feature sequences)
    into fixed-length (window_size, FEATURE_VECTOR_SIZE) arrays suitable
    for batched training, using the same uniform-resampling approach as
    dataset_manager.py's TimeWarpAugmentation for consistency.

    Kept as its own class so the (framework-agnostic) resampling logic is
    unit-testable without TensorFlow installed.
    """

    def __init__(self, window_size: int) -> None:
        self._window_size = window_size

    def resample_to_window(self, sequence: np.ndarray) -> np.ndarray:
        """
        Resample a (num_frames, FEATURE_VECTOR_SIZE) sequence to exactly
        `window_size` frames via linear interpolation along the time
        axis -- handles both shorter sequences (stretched/upsampled) and
        longer ones (compressed/downsampled) uniformly, preserving the
        overall motion shape better than naive padding/truncation would.
        """
        num_frames = sequence.shape[0]
        if num_frames == self._window_size:
            return sequence.astype(np.float32)
        if num_frames < 2:
            # Degenerate single-frame sample: just repeat it.
            return np.repeat(sequence, self._window_size, axis=0).astype(np.float32)

        source_indices = np.arange(num_frames)
        target_indices = np.linspace(0, num_frames - 1, self._window_size)

        output = np.zeros((self._window_size, sequence.shape[1]), dtype=np.float32)
        for feature_idx in range(sequence.shape[1]):
            output[:, feature_idx] = np.interp(
                target_indices, source_indices, sequence[:, feature_idx]
            )
        return output

    def build_arrays(
        self,
        dataset_manager: DatasetManager,
        sample_ids: List[str],
        label_to_id: Dict[str, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and resample a list of samples into batched X/y arrays.

        Args:
            dataset_manager: Source of sample features.
            sample_ids: Which samples to include (typically one of
                DatasetSplit's train/val/test id lists).
            label_to_id: Word -> integer class id mapping.

        Returns:
            (X, y): X has shape (num_samples, window_size,
            FEATURE_VECTOR_SIZE); y has shape (num_samples,) of integer
            class ids.
        """
        features_list = []
        labels_list = []

        for sample_id in sample_ids:
            try:
                metadata = next(
                    s for s in dataset_manager.get_samples() if s.sample_id == sample_id
                )
                raw_features = dataset_manager.load_sample_features(sample_id)
            except (StopIteration, KeyError, FileNotFoundError):
                logger.warning(
                    "Skipping sample %s: not found in current dataset version.",
                    sample_id,
                )
                continue

            if metadata.word not in label_to_id:
                logger.warning(
                    "Skipping sample %s: label '%s' not in label_to_id mapping.",
                    sample_id,
                    metadata.word,
                )
                continue

            resampled = self.resample_to_window(raw_features)
            features_list.append(resampled)
            labels_list.append(label_to_id[metadata.word])

        if not features_list:
            return (
                np.zeros((0, self._window_size, FEATURE_VECTOR_SIZE), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )

        return np.stack(features_list, axis=0), np.array(labels_list, dtype=np.int64)


# ----------------------------------------------------------------------
# Model trainer
# ----------------------------------------------------------------------


class ModelTrainer:
    """
    Trains the sign-recognition sequence model, with checkpointing,
    resume support, metrics reporting, and TFLite/ONNX export.

    Usage:
        dataset_manager = DatasetManager()
        trainer = ModelTrainer(dataset_manager, run_id="v1_run1")
        metrics = trainer.train()
        trainer.export_tflite()
        classifier = trainer.get_sequence_classifier()  # for ContinuousRecognizer
    """

    def __init__(
        self,
        dataset_manager: DatasetManager,
        run_id: str,
        config: Optional[ModelTrainingConfig] = None,
    ) -> None:
        """
        Args:
            dataset_manager: The DatasetManager to train from.
            run_id: A unique identifier for this training run (used as
                the checkpoint subfolder name). Reusing a run_id resumes
                that run's training instead of starting fresh.
            config: Optional ModelTrainingConfig; defaults used if omitted.

        Raises:
            RuntimeError: If TensorFlow is not installed. Training is a
                hard dependency on an ML framework -- there is no
                meaningful "null" trainer, unlike this project's other
                pluggable-with-a-safe-default components.
        """
        if tf is None:
            raise RuntimeError(
                "TensorFlow is not installed. Install it via: "
                "pip install tensorflow. (Model training requires a real "
                "ML framework -- unlike other modules in this project, "
                "there's no safe no-op fallback here.)"
            )

        self._dataset_manager = dataset_manager
        self._run_id = run_id
        self._config = config or ModelTrainingConfig()
        self._preparer = SequenceDataPreparer(self._config.window_size)

        self._checkpoint_dir = Path(self._config.checkpoint_root) / run_id
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._model: Optional["tf.keras.Model"] = None
        self._label_to_id: Dict[str, int] = {}
        self._labels: List[str] = []

        logger.info(
            "ModelTrainer initialized (run_id=%s, window_size=%d).",
            run_id,
            self._config.window_size,
        )

    # ------------------------------------------------------------------
    # Model architecture
    # ------------------------------------------------------------------

    def _build_model(self, num_classes: int) -> "tf.keras.Model":
        """
        Build a straightforward two-layer LSTM sequence classifier.

        Architecture rationale: LSTMs are a natural fit for landmark
        sequences (variable-speed hand motion over time); two stacked
        layers give enough capacity to learn multi-phase gestures without
        being excessive for a from-scratch, modestly-sized sign dataset.
        This is intentionally a reasonable BASELINE -- swapping in a
        Transformer or a more elaborate architecture only requires
        changing this one method, since everything else (data prep,
        checkpointing, export) is architecture-agnostic.
        """
        inputs = tf.keras.Input(
            shape=(self._config.window_size, FEATURE_VECTOR_SIZE), name="landmark_sequence"
        )
        x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
        x = tf.keras.layers.LSTM(self._config.lstm_units, return_sequences=True)(x)
        x = tf.keras.layers.Dropout(self._config.dropout_rate)(x)
        x = tf.keras.layers.LSTM(self._config.lstm_units // 2)(x)
        x = tf.keras.layers.Dropout(self._config.dropout_rate)(x)
        x = tf.keras.layers.Dense(self._config.dense_units, activation="relu")(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="sign_sequence_classifier")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self._config.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # ------------------------------------------------------------------
    # Checkpoint / resume
    # ------------------------------------------------------------------

    def _training_state_path(self) -> Path:
        return self._checkpoint_dir / "training_state.json"

    def _latest_weights_path(self) -> Path:
        return self._checkpoint_dir / "latest.weights.h5"

    def _best_weights_path(self) -> Path:
        return self._checkpoint_dir / "best.weights.h5"

    def _load_training_state(self) -> Dict:
        path = self._training_state_path()
        if not path.exists():
            return {"epochs_completed": 0, "best_val_accuracy": 0.0, "best_val_loss": float("inf")}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logger.exception("Failed to read training_state.json; starting fresh.")
            return {"epochs_completed": 0, "best_val_accuracy": 0.0, "best_val_loss": float("inf")}

    def _save_training_state(self, state: Dict) -> None:
        with open(self._training_state_path(), "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> TrainingMetrics:
        """
        Train (or RESUME training) the model on the dataset's current
        split, saving checkpoints as it goes.

        If a previous run with this run_id exists, training resumes from
        its last completed epoch and best validation metrics rather than
        starting over -- calling train() again after an interruption is
        the intended way to continue.

        Returns:
            TrainingMetrics summarizing the completed run.
        """
        self._label_to_id = self._dataset_manager.get_label_to_id()
        self._labels = self._dataset_manager.get_labels()
        if not self._labels:
            raise ValueError(
                "No labels registered in the dataset; add samples via "
                "DatasetManager.add_sample() before training."
            )

        split = self._dataset_manager.get_split() or self._dataset_manager.split_dataset()

        x_train, y_train = self._preparer.build_arrays(
            self._dataset_manager, split.train_sample_ids, self._label_to_id
        )
        x_val, y_val = self._preparer.build_arrays(
            self._dataset_manager, split.val_sample_ids, self._label_to_id
        )

        if x_train.shape[0] == 0:
            raise ValueError(
                "No training samples available after preparing the split; "
                "collect more data via DatasetManager.add_sample()."
            )

        state = self._load_training_state()
        self._model = self._build_model(num_classes=len(self._labels))

        if self._latest_weights_path().exists():
            logger.info(
                "Resuming run '%s' from epoch %d.",
                self._run_id,
                state["epochs_completed"],
            )
            self._model.load_weights(str(self._latest_weights_path()))
        else:
            logger.info("Starting new training run '%s'.", self._run_id)

        remaining_epochs = max(0, self._config.max_epochs - state["epochs_completed"])
        if remaining_epochs == 0:
            logger.info(
                "Run '%s' already completed %d/%d configured epochs; nothing to do.",
                self._run_id,
                state["epochs_completed"],
                self._config.max_epochs,
            )
            return self._build_metrics_from_state(state)

        checkpoint_callback = _CheckpointAndStateCallback(
            trainer=self, initial_epoch=state["epochs_completed"], state=state
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self._config.early_stopping_patience,
            restore_best_weights=True,
        )

        history = self._model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val) if x_val.shape[0] > 0 else None,
            batch_size=self._config.batch_size,
            epochs=self._config.max_epochs,
            initial_epoch=state["epochs_completed"],
            callbacks=[checkpoint_callback, early_stopping],
            verbose=0,  # we log per-epoch ourselves via the callback, per project's "use logging not print" rule
        )

        final_state = self._load_training_state()
        metrics = TrainingMetrics(
            run_id=self._run_id,
            epochs_completed=final_state["epochs_completed"],
            final_train_loss=float(history.history.get("loss", [float("nan")])[-1]),
            final_train_accuracy=float(history.history.get("accuracy", [float("nan")])[-1]),
            best_val_loss=final_state["best_val_loss"],
            best_val_accuracy=final_state["best_val_accuracy"],
            trained_at_ms=time.time() * 1000.0,
            dataset_version=self._dataset_manager.get_current_version(),
            labels=self._labels,
        )
        self._save_metrics(metrics)
        return metrics

    def _build_metrics_from_state(self, state: Dict) -> TrainingMetrics:
        return TrainingMetrics(
            run_id=self._run_id,
            epochs_completed=state["epochs_completed"],
            final_train_loss=float("nan"),
            final_train_accuracy=float("nan"),
            best_val_loss=state["best_val_loss"],
            best_val_accuracy=state["best_val_accuracy"],
            trained_at_ms=time.time() * 1000.0,
            dataset_version=self._dataset_manager.get_current_version(),
            labels=self._labels,
        )

    def _save_metrics(self, metrics: TrainingMetrics) -> None:
        path = self._checkpoint_dir / "metrics.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(metrics), f, indent=2)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_on_test_set(self) -> Tuple[float, float]:
        """
        Evaluate the current model against the dataset's held-out test
        split.

        Returns:
            (test_loss, test_accuracy). See evaluation.py for a fuller
            metrics suite (precision/recall/F1/confusion matrix) built on
            top of a trained model's predictions.
        """
        if self._model is None:
            raise RuntimeError("No model loaded; call train() (or load a checkpoint) first.")

        split = self._dataset_manager.get_split()
        if split is None:
            raise RuntimeError(
                "No dataset split found; call DatasetManager.split_dataset() first."
            )

        x_test, y_test = self._preparer.build_arrays(
            self._dataset_manager, split.test_sample_ids, self._label_to_id
        )
        if x_test.shape[0] == 0:
            logger.warning("Test split is empty; cannot evaluate.")
            return float("nan"), float("nan")

        test_loss, test_accuracy = self._model.evaluate(x_test, y_test, verbose=0)
        logger.info("Test evaluation: loss=%.4f, accuracy=%.4f", test_loss, test_accuracy)
        return float(test_loss), float(test_accuracy)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_tflite(self, output_path: Optional[Path] = None) -> Path:
        """
        Export the trained model to TensorFlow Lite format, for
        mobile/edge deployment (per the future-extensibility list's
        "Mobile deployment" / "Edge AI deployment" goals).

        Returns:
            The path the .tflite file was written to.
        """
        if self._model is None:
            raise RuntimeError("No model loaded; call train() first.")

        output_path = output_path or (
            Path(self._config.model_output_root) / f"{self._run_id}.tflite"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        converter = tf.lite.TFLiteConverter.from_keras_model(self._model)
        tflite_model = converter.convert()
        output_path.write_bytes(tflite_model)

        self._write_label_map(output_path.with_suffix(".labels.json"))
        logger.info("Exported TensorFlow Lite model to %s", output_path)
        return output_path

    def export_onnx(self, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Export the trained model to ONNX format (optional per the
        project's tech stack notes). Requires the optional `tf2onnx`
        package; if it's not installed, this logs a warning and returns
        None rather than failing the whole training pipeline over an
        optional export format.
        """
        if self._model is None:
            raise RuntimeError("No model loaded; call train() first.")

        try:
            import tf2onnx
        except ImportError:
            logger.warning(
                "tf2onnx is not installed; skipping ONNX export "
                "(install via: pip install tf2onnx). ONNX export is "
                "optional per the project's tech stack."
            )
            return None

        output_path = output_path or (
            Path(self._config.model_output_root) / f"{self._run_id}.onnx"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        input_signature = [
            tf.TensorSpec(
                (None, self._config.window_size, FEATURE_VECTOR_SIZE),
                tf.float32,
                name="landmark_sequence",
            )
        ]
        model_proto, _ = tf2onnx.convert.from_keras(
            self._model, input_signature=input_signature, output_path=str(output_path)
        )

        self._write_label_map(output_path.with_suffix(".labels.json"))
        logger.info("Exported ONNX model to %s", output_path)
        return output_path

    def _write_label_map(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"labels": self._labels, "window_size": self._config.window_size}, f, indent=2
            )

    # ------------------------------------------------------------------
    # Bridge to continuous_recognition.py
    # ------------------------------------------------------------------

    def get_sequence_classifier(self) -> "TrainedSequenceClassifier":
        """
        Wrap the currently trained model as a `SequenceClassifier`, ready
        to be injected directly into `ContinuousRecognizer` for live
        inference -- this is what closes the loop between training and
        the real-time pipeline.
        """
        if self._model is None:
            raise RuntimeError("No model loaded; call train() first.")
        return TrainedSequenceClassifier(
            model=self._model,
            labels=self._labels,
            window_size=self._config.window_size,
        )


class _CheckpointAndStateCallback:
    """
    A Keras callback (duck-typed rather than subclassing tf.keras.callbacks.Callback
    at import time, so this file stays importable without TensorFlow) that
    saves model weights and training progress after every epoch, enabling
    resume-from-interruption.
    """

    def __new__(cls, trainer: "ModelTrainer", initial_epoch: int, state: Dict):
        # Defined as a factory returning a real tf.keras.callbacks.Callback
        # instance, constructed lazily so `tf` is guaranteed available by
        # the time this runs (ModelTrainer.__init__ already validated that).
        base = tf.keras.callbacks.Callback

        class _Impl(base):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                completed = epoch + 1
                state["epochs_completed"] = completed

                val_loss = logs.get("val_loss", logs.get("loss"))
                val_accuracy = logs.get("val_accuracy", logs.get("accuracy", 0.0))

                trainer._model.save_weights(str(trainer._latest_weights_path()))

                if val_loss is not None and val_loss < state.get("best_val_loss", float("inf")):
                    state["best_val_loss"] = float(val_loss)
                    state["best_val_accuracy"] = float(val_accuracy)
                    trainer._model.save_weights(str(trainer._best_weights_path()))
                    logger.info(
                        "Epoch %d: new best val_loss=%.4f, val_accuracy=%.4f (checkpoint saved).",
                        completed,
                        val_loss,
                        val_accuracy,
                    )
                else:
                    logger.info(
                        "Epoch %d: loss=%.4f, accuracy=%.4f%s",
                        completed,
                        logs.get("loss", float("nan")),
                        logs.get("accuracy", float("nan")),
                        (
                            f", val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}"
                            if val_loss is not None
                            else ""
                        ),
                    )

                trainer._save_training_state(state)

        return _Impl()


# ----------------------------------------------------------------------
# Trained classifier: bridges into continuous_recognition.py
# ----------------------------------------------------------------------


class TrainedSequenceClassifier(SequenceClassifier):
    """
    Implements continuous_recognition.py's `SequenceClassifier` interface
    using a trained Keras model -- this is the concrete classifier that
    replaces `NullSequenceClassifier` once training has produced a real
    model, closing the collect -> train -> recognize loop.
    """

    def __init__(
        self,
        model: "tf.keras.Model",
        labels: List[str],
        window_size: int,
        min_confidence_for_background_override: float = 0.0,
    ) -> None:
        """
        Args:
            model: A trained Keras model with input shape
                (window_size, FEATURE_VECTOR_SIZE) and softmax output over
                len(labels) classes.
            labels: The class labels, in the same order as the model's
                output layer (see DatasetManager.get_label_to_id()).
            window_size: Expected input sequence length.
            min_confidence_for_background_override: Reserved for future
                use (e.g. forcing a background prediction below a
                threshold even if BACKGROUND_LABEL isn't the argmax);
                defaults to 0.0 (no override), since
                ContinuousRecognizer already applies its own
                min_confidence filtering downstream.
        """
        if tf is None:
            raise RuntimeError("TensorFlow is not installed; cannot run TrainedSequenceClassifier.")

        self._model = model
        self._labels = labels
        self._window_size = window_size
        self._min_confidence_override = min_confidence_for_background_override

    @property
    def required_window_size(self) -> int:
        return self._window_size

    def predict(self, sequence: np.ndarray) -> Tuple[str, float]:
        try:
            batched = np.expand_dims(sequence.astype(np.float32), axis=0)
            probabilities = self._model.predict(batched, verbose=0)[0]
            best_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[best_idx])
            predicted_label = self._labels[best_idx]

            if predicted_label == BACKGROUND_LABEL:
                return "", confidence
            return predicted_label, confidence
        except Exception:
            logger.exception("TrainedSequenceClassifier failed to predict; returning idle.")
            return "", 0.0

    @classmethod
    def load(cls, model_dir: Path, window_size: int) -> "TrainedSequenceClassifier":
        """
        Load a previously exported/saved Keras model + label map from
        disk (e.g. models/<run_id>/) for use at inference time, without
        needing a ModelTrainer instance.
        """
        if tf is None:
            raise RuntimeError("TensorFlow is not installed; cannot load a trained model.")

        model = tf.keras.models.load_model(str(model_dir / "model.keras"))
        with open(model_dir / "labels.json", "r", encoding="utf-8") as f:
            labels = json.load(f)["labels"]
        return cls(model=model, labels=labels, window_size=window_size)


if __name__ == "__main__":
    # Minimal manual smoke-test. If TensorFlow is installed, this runs a
    # tiny end-to-end training loop on synthetic data. If not, it still
    # exercises the framework-agnostic data-preparation logic
    # (SequenceDataPreparer) so that part of the module is verified even
    # without TensorFlow available. Run via: `python model_training.py`.
    import shutil
    import tempfile

    from dataset_manager import DatasetManagerConfig

    logger.info("Running model_training.py standalone demo.")

    temp_dir = Path(tempfile.mkdtemp(prefix="model_training_demo_"))
    try:
        dataset_manager = DatasetManager(DatasetManagerConfig(root_dir=temp_dir))
        rng = np.random.default_rng(seed=0)
        for word, count in [("hello", 8), ("thanks", 8), (BACKGROUND_LABEL, 8)]:
            for _ in range(count):
                num_frames = rng.integers(20, 40)
                fake_sequence = rng.random((num_frames, FEATURE_VECTOR_SIZE)).astype(np.float32)
                dataset_manager.add_sample(word, fake_sequence)

        # Always exercise the framework-agnostic resampling logic.
        preparer = SequenceDataPreparer(window_size=30)
        split = dataset_manager.split_dataset(test_ratio=0.2, val_ratio=0.2, seed=1)
        x_train, y_train = preparer.build_arrays(
            dataset_manager, split.train_sample_ids, dataset_manager.get_label_to_id()
        )
        print(f"Prepared training arrays: X={x_train.shape}, y={y_train.shape}")

        if tf is None:
            logger.warning(
                "TensorFlow is not installed in this environment; skipping the "
                "actual training/export steps. Data preparation logic above "
                "was verified successfully. Install TensorFlow "
                "(`pip install tensorflow`) to exercise full training."
            )
        else:
            trainer = ModelTrainer(
                dataset_manager,
                run_id="demo_run",
                config=ModelTrainingConfig(
                    window_size=30, max_epochs=2, batch_size=4, dataset_root=temp_dir
                ),
            )
            metrics = trainer.train()
            print("Training metrics:", metrics)
            test_loss, test_acc = trainer.evaluate_on_test_set()
            print(f"Test loss={test_loss:.4f}, accuracy={test_acc:.4f}")
            tflite_path = trainer.export_tflite()
            print("Exported TFLite model to:", tflite_path)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    logger.info("Demo complete.")
