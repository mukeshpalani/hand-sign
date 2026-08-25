"""
evaluation.py

MODULE 10 -- Evaluation

Single Responsibility
----------------------
This module's ONLY job is: measure how well the system performs, along
two independent axes (project requirement: evaluation.py -- "Accuracy /
Precision / Recall / F1 Score / Latency / FPS / Confusion Matrix"):

  1. CLASSIFICATION QUALITY -- given predicted vs. true labels, compute
     accuracy, per-class and aggregate precision/recall/F1, and a full
     confusion matrix.
  2. RUNTIME PERFORMANCE -- given a stream of measured operation
     durations (e.g. one hand-detection call, one recognition-window
     inference), compute latency statistics (mean/percentiles) and
     effective FPS.

It does NOT train models, does NOT run inference itself in the live
pipeline, and does NOT collect data -- it consumes predictions/timings
produced elsewhere (model_training.py's trained classifier, or timed
calls from main.py's live loop) and turns them into interpretable
metrics and reports.

Design notes
------------
- NO HEAVY ML-METRICS DEPENDENCY: precision/recall/F1/confusion-matrix
  are implemented directly with NumPy rather than pulling in scikit-learn
  just for this, keeping the project's dependency footprint small and the
  math fully transparent/auditable in one place.
- `ClassificationEvaluator` works from plain label lists (`evaluate()`),
  so it's usable completely independently of any specific model backend
  -- including with `NullSequenceClassifier`/synthetic data for testing,
  which is how this module's own smoke test below exercises it without
  needing TensorFlow installed. `evaluate_classifier_on_dataset()` is a
  thin convenience layer on top that wires in a real
  `SequenceClassifier` + `DatasetManager` when you DO have both.
- `PerformanceProfiler` is a general-purpose timing utility (context
  manager based) that any module's hot path can wrap -- e.g. main.py can
  profile `hand_detector.detect()`, `resolver.resolve()`, and
  `recognizer.process()` all independently to find bottlenecks, without
  those modules needing any awareness of evaluation.py at all (this
  module only ever depends on others' PUBLIC interfaces, never the other
  way around, keeping the dependency graph one-directional).
"""

from __future__ import annotations

import json
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# Classification metrics data model
# ----------------------------------------------------------------------


@dataclass
class PerClassMetrics:
    """Precision/recall/F1/support for a single class label."""
    label: str
    precision: float
    recall: float
    f1_score: float
    support: int  # number of true instances of this label in the evaluated set


@dataclass
class ClassificationMetrics:
    """
    Full classification-quality report: overall accuracy, per-class
    metrics, macro/weighted aggregates, and the confusion matrix.

    Macro averages treat every class equally (simple mean across
    classes); weighted averages weight each class by its support (number
    of true instances) -- both are reported since sign-language datasets
    commonly have imbalanced per-word sample counts, and the two
    averages can tell quite different stories in that case.
    """
    labels: List[str]
    accuracy: float
    per_class: List[PerClassMetrics] = field(default_factory=list)
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    weighted_precision: float = 0.0
    weighted_recall: float = 0.0
    weighted_f1: float = 0.0
    confusion_matrix: List[List[int]] = field(default_factory=list)
    num_samples: int = 0

    def confusion_matrix_as_text(self) -> str:
        """Render the confusion matrix as a simple, readable text table
        (rows = true label, columns = predicted label) -- no plotting
        library required, so this can be logged directly."""
        if not self.confusion_matrix:
            return "(empty confusion matrix)"

        col_width = max(4, max(len(label) for label in self.labels) + 1)
        header = " " * col_width + "".join(
            f"{label[:col_width - 1]:>{col_width}}" for label in self.labels
        )
        lines = [header]
        for row_label, row in zip(self.labels, self.confusion_matrix):
            row_text = "".join(f"{value:>{col_width}}" for value in row)
            lines.append(f"{row_label[:col_width - 1]:<{col_width}}{row_text}")
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Classification evaluator
# ----------------------------------------------------------------------


class ClassificationEvaluator:
    """
    Computes accuracy, precision, recall, F1, and a confusion matrix from
    predicted vs. true labels.

    Usage:
        evaluator = ClassificationEvaluator(labels=dataset_manager.get_labels())
        metrics = evaluator.evaluate(y_true=["hello", "thanks", ...],
                                      y_pred=["hello", "hello", ...])
        print(metrics.confusion_matrix_as_text())
    """

    def __init__(self, labels: List[str]) -> None:
        """
        Args:
            labels: The full, fixed set of possible class labels, in a
                stable order (typically DatasetManager.get_labels()).
                Determines the confusion matrix's row/column order.
        """
        if not labels:
            raise ValueError("ClassificationEvaluator requires a non-empty label list.")
        self._labels = list(labels)
        self._label_to_index = {label: idx for idx, label in enumerate(self._labels)}

    def evaluate(
        self, y_true: Sequence[str], y_pred: Sequence[str]
    ) -> ClassificationMetrics:
        """
        Compute the full classification report for a batch of predictions.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels, same length and order as y_true.

        Returns:
            A ClassificationMetrics report.

        Raises:
            ValueError: If y_true and y_pred have different lengths, or
                either is empty.
        """
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"y_true (len={len(y_true)}) and y_pred (len={len(y_pred)}) "
                "must have the same length."
            )
        if not y_true:
            raise ValueError("Cannot evaluate on an empty set of predictions.")

        confusion = self._build_confusion_matrix(y_true, y_pred)
        per_class = self._compute_per_class_metrics(confusion)

        accuracy = float(np.trace(confusion) / confusion.sum())
        macro_precision = float(np.mean([m.precision for m in per_class]))
        macro_recall = float(np.mean([m.recall for m in per_class]))
        macro_f1 = float(np.mean([m.f1_score for m in per_class]))

        total_support = sum(m.support for m in per_class)
        weighted_precision, weighted_recall, weighted_f1 = 0.0, 0.0, 0.0
        if total_support > 0:
            weighted_precision = sum(m.precision * m.support for m in per_class) / total_support
            weighted_recall = sum(m.recall * m.support for m in per_class) / total_support
            weighted_f1 = sum(m.f1_score * m.support for m in per_class) / total_support

        return ClassificationMetrics(
            labels=self._labels,
            accuracy=accuracy,
            per_class=per_class,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            macro_f1=macro_f1,
            weighted_precision=float(weighted_precision),
            weighted_recall=float(weighted_recall),
            weighted_f1=float(weighted_f1),
            confusion_matrix=confusion.tolist(),
            num_samples=len(y_true),
        )

    def _build_confusion_matrix(
        self, y_true: Sequence[str], y_pred: Sequence[str]
    ) -> np.ndarray:
        """
        Build an (num_labels x num_labels) confusion matrix, where
        matrix[i][j] = count of samples with true label i predicted as
        label j. Unknown labels (not in self._labels, e.g. an unexpected
        prediction string) are logged and skipped rather than crashing
        the whole evaluation over one bad entry.
        """
        num_labels = len(self._labels)
        matrix = np.zeros((num_labels, num_labels), dtype=np.int64)

        skipped = 0
        for true_label, pred_label in zip(y_true, y_pred):
            true_idx = self._label_to_index.get(true_label)
            pred_idx = self._label_to_index.get(pred_label)
            if true_idx is None or pred_idx is None:
                skipped += 1
                continue
            matrix[true_idx][pred_idx] += 1

        if skipped:
            logger.warning(
                "Skipped %d prediction(s) with a label outside the known "
                "label set during confusion matrix construction.",
                skipped,
            )
        return matrix

    def _compute_per_class_metrics(self, confusion: np.ndarray) -> List[PerClassMetrics]:
        """Derive precision/recall/F1/support for every class from the
        confusion matrix's rows (true) and columns (predicted)."""
        per_class = []
        for idx, label in enumerate(self._labels):
            true_positives = confusion[idx][idx]
            predicted_positives = confusion[:, idx].sum()
            actual_positives = confusion[idx, :].sum()

            precision = (
                true_positives / predicted_positives if predicted_positives > 0 else 0.0
            )
            recall = true_positives / actual_positives if actual_positives > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            per_class.append(
                PerClassMetrics(
                    label=label,
                    precision=float(precision),
                    recall=float(recall),
                    f1_score=float(f1),
                    support=int(actual_positives),
                )
            )
        return per_class

    def evaluate_classifier_on_dataset(
        self,
        classifier,  # continuous_recognition.SequenceClassifier -- typed loosely to avoid a hard import cycle
        dataset_manager,
        sample_ids: List[str],
    ) -> ClassificationMetrics:
        """
        Convenience wrapper: run a trained SequenceClassifier over a set
        of dataset samples and evaluate its predictions in one call.

        Reuses model_training.py's SequenceDataPreparer for the exact
        same fixed-length resampling used during training, so evaluation
        sees the model under the same input distribution it was trained
        on (rather than accidentally evaluating on differently-shaped
        inputs).

        Args:
            classifier: Any object implementing
                continuous_recognition.SequenceClassifier's predict()
                method (e.g. model_training.TrainedSequenceClassifier).
            dataset_manager: Source of the samples to evaluate on.
            sample_ids: Which samples to evaluate (typically a
                DatasetSplit's test_sample_ids).

        Returns:
            A ClassificationMetrics report for this sample set.
        """
        from model_training import SequenceDataPreparer  # local import avoids a hard TF dependency at module load

        preparer = SequenceDataPreparer(window_size=classifier.required_window_size)
        y_true: List[str] = []
        y_pred: List[str] = []

        for sample_id in sample_ids:
            try:
                metadata = next(
                    s for s in dataset_manager.get_samples() if s.sample_id == sample_id
                )
                raw_features = dataset_manager.load_sample_features(sample_id)
            except (StopIteration, KeyError, FileNotFoundError):
                logger.warning("Skipping missing sample %s during evaluation.", sample_id)
                continue

            resampled = preparer.resample_to_window(raw_features)
            predicted_word, _confidence = classifier.predict(resampled)

            y_true.append(metadata.word)
            # An empty prediction ("nothing recognized") is compared
            # against the background label if it exists in our label set,
            # so "correctly predicted nothing" can still score as correct
            # when evaluating against background/idle samples.
            y_pred.append(predicted_word if predicted_word else "_background_")

        return self.evaluate(y_true, y_pred)


# ----------------------------------------------------------------------
# Runtime performance profiling
# ----------------------------------------------------------------------


@dataclass
class PerformanceMetrics:
    """Latency and throughput statistics collected over a window of
    timed operations."""
    num_samples: int
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    min_latency_ms: float
    fps: float  # effective frames/operations per second, from mean latency


class PerformanceProfiler:
    """
    A lightweight, general-purpose latency/FPS profiler. Any module's hot
    path can be wrapped in `with profiler.measure(): ...` to record its
    duration -- e.g. main.py might keep one profiler per pipeline stage
    (detection, overlap resolution, recognition) to find bottlenecks.

    Usage:
        profiler = PerformanceProfiler()
        for frame in video_stream:
            with profiler.measure():
                result = hand_detector.detect(frame)
        print(profiler.get_metrics())
    """

    def __init__(self, window_size: int = 500) -> None:
        """
        Args:
            window_size: How many recent measurements to retain for
                computing rolling statistics (older ones are dropped),
                bounding memory in long-running sessions.
        """
        self._durations_ms: Deque[float] = deque(maxlen=window_size)

    @contextmanager
    def measure(self):
        """Context manager that times the wrapped block and records its
        duration in milliseconds."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.record(elapsed_ms)

    def record(self, duration_ms: float) -> None:
        """Manually record a duration (in milliseconds), for cases where
        wrapping in `measure()` isn't convenient (e.g. timing something
        that happens across multiple function calls)."""
        self._durations_ms.append(duration_ms)

    def get_metrics(self) -> Optional[PerformanceMetrics]:
        """
        Compute latency/FPS statistics from all currently retained
        measurements.

        Returns:
            A PerformanceMetrics summary, or None if no measurements have
            been recorded yet.
        """
        if not self._durations_ms:
            return None

        durations = np.array(self._durations_ms, dtype=np.float64)
        mean_latency = float(np.mean(durations))

        return PerformanceMetrics(
            num_samples=len(durations),
            mean_latency_ms=mean_latency,
            median_latency_ms=float(np.median(durations)),
            p95_latency_ms=float(np.percentile(durations, 95)),
            p99_latency_ms=float(np.percentile(durations, 99)),
            max_latency_ms=float(np.max(durations)),
            min_latency_ms=float(np.min(durations)),
            fps=(1000.0 / mean_latency) if mean_latency > 0 else 0.0,
        )

    def reset(self) -> None:
        """Clear all recorded measurements."""
        self._durations_ms.clear()


# ----------------------------------------------------------------------
# Combined evaluation report
# ----------------------------------------------------------------------


@dataclass
class EvaluationReport:
    """A combined classification + performance report, suitable for
    saving to disk (e.g. alongside model_training.py's checkpoints) or
    displaying on a dashboard."""
    classification: Optional[ClassificationMetrics]
    performance: Optional[PerformanceMetrics]
    generated_at_ms: float = field(default_factory=lambda: time.time() * 1000.0)
    notes: str = ""

    def save(self, path: Path) -> None:
        """Serialize this report to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
        logger.info("Saved evaluation report to %s", path)

    @classmethod
    def load(cls, path: Path) -> "EvaluationReport":
        """Load a previously saved EvaluationReport from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        classification = None
        if raw.get("classification"):
            c = raw["classification"]
            classification = ClassificationMetrics(
                labels=c["labels"],
                accuracy=c["accuracy"],
                per_class=[PerClassMetrics(**pc) for pc in c["per_class"]],
                macro_precision=c["macro_precision"],
                macro_recall=c["macro_recall"],
                macro_f1=c["macro_f1"],
                weighted_precision=c["weighted_precision"],
                weighted_recall=c["weighted_recall"],
                weighted_f1=c["weighted_f1"],
                confusion_matrix=c["confusion_matrix"],
                num_samples=c["num_samples"],
            )

        performance = None
        if raw.get("performance"):
            performance = PerformanceMetrics(**raw["performance"])

        return cls(
            classification=classification,
            performance=performance,
            generated_at_ms=raw.get("generated_at_ms", 0.0),
            notes=raw.get("notes", ""),
        )


if __name__ == "__main__":
    # Minimal manual smoke-test: exercises BOTH classification metrics
    # (on synthetic predictions -- no trained model needed) and the
    # performance profiler (on a trivially timed synthetic workload).
    # Run via: `python evaluation.py`.
    import random

    logger.info("Running evaluation.py standalone demo.")

    labels = ["hello", "thanks", "yes", "no", "_background_"]
    evaluator = ClassificationEvaluator(labels)

    # Build a synthetic prediction set that's mostly correct, with a
    # deliberate, consistent confusion between "yes" and "no" so the
    # confusion matrix has something interesting to show.
    rng = random.Random(0)
    y_true: List[str] = []
    y_pred: List[str] = []
    for _ in range(200):
        true_label = rng.choice(labels)
        if true_label == "yes" and rng.random() < 0.3:
            pred_label = "no"
        elif rng.random() < 0.1:
            pred_label = rng.choice(labels)
        else:
            pred_label = true_label
        y_true.append(true_label)
        y_pred.append(pred_label)

    metrics = evaluator.evaluate(y_true, y_pred)
    print(f"Accuracy: {metrics.accuracy:.3f}")
    print(f"Macro F1: {metrics.macro_f1:.3f}  |  Weighted F1: {metrics.weighted_f1:.3f}")
    print("\nPer-class metrics:")
    for pc in metrics.per_class:
        print(
            f"  {pc.label:<12} precision={pc.precision:.2f}  recall={pc.recall:.2f}  "
            f"f1={pc.f1_score:.2f}  support={pc.support}"
        )
    print("\nConfusion matrix (rows=true, cols=predicted):")
    print(metrics.confusion_matrix_as_text())

    # Performance profiling demo.
    profiler = PerformanceProfiler()
    for _ in range(50):
        with profiler.measure():
            time.sleep(rng.uniform(0.005, 0.02))  # simulate variable per-frame work

    perf = profiler.get_metrics()
    print(
        f"\nPerformance: mean={perf.mean_latency_ms:.2f}ms  "
        f"p95={perf.p95_latency_ms:.2f}ms  fps={perf.fps:.1f}"
    )

    report = EvaluationReport(
        classification=metrics, performance=perf, notes="evaluation.py smoke test"
    )
    import tempfile

    report_path = Path(tempfile.mkdtemp()) / "report.json"
    report.save(report_path)
    reloaded = EvaluationReport.load(report_path)
    print(f"\nReloaded report accuracy matches: {reloaded.classification.accuracy == metrics.accuracy}")

    logger.info("Demo complete.")
