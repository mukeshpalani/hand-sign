"""
dataset_manager.py

MODULE 8 -- Dataset Manager

Single Responsibility
----------------------
This module's ONLY job is: own the on-disk sign-language TRAINING DATASET
-- collecting new samples, managing the label (vocabulary) registry,
splitting data into train/validation/test sets, augmenting samples, and
versioning the dataset over time (project requirement: dataset_manager.py
-- "Collect data / Add new signs / Manage labels / Split train/test /
Augment dataset / Version datasets").

It does NOT train any model (that's model_training.py's job) and does NOT
recognize signs live -- it is a REPOSITORY that model_training.py reads
from, and that a data-collection tool (or main.py in a "recording mode")
writes to.

Design notes
------------
- REPOSITORY PATTERN: DatasetManager is the single gatekeeper for all
  dataset reads/writes. Nothing else in the project touches the
  datasets/ folder's file layout directly -- if the storage format ever
  changes (e.g. moving from local .npy files to a cloud bucket, per the
  future-extensibility list's "cloud synchronization"), only this file
  needs to change.
- SHARED FEATURE CONTRACT: samples are stored using the EXACT SAME
  126-value-per-frame feature layout that continuous_recognition.py's
  `LandmarkFeatureExtractor` produces at inference time (imported as
  `FEATURE_VECTOR_SIZE` rather than re-hardcoded, to guarantee the two
  never drift apart). Training data that doesn't match the shape the
  model will see in production is a classic, easy-to-miss bug -- this
  keeps that contract explicit and enforced (validated on every write).
- VERSIONING: each dataset version lives in its own folder
  (datasets/v1/, datasets/v2/, ...), containing that version's label
  registry, sample manifest, and feature files. Creating a new version
  snapshots the previous one forward, so old versions remain intact and
  reproducible (e.g. for comparing model performance across dataset
  revisions in evaluation.py).
- STRATEGY PATTERN for augmentation: `AugmentationStrategy` is an
  abstract interface with a few concrete, sign-language-appropriate
  implementations (Gaussian jitter, horizontal mirroring, time-warping).
  New augmentation techniques can be added later without changing
  DatasetManager's orchestration logic.
"""

from __future__ import annotations

import json
import random
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from continuous_recognition import FEATURE_VECTOR_SIZE, FEATURES_PER_HAND
from utils.logger import get_logger

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------


@dataclass
class SampleMetadata:
    """
    Metadata describing one recorded (or augmented) training sample: a
    sequence of feature vectors for one performed sign, plus its label
    and provenance.
    """
    sample_id: str
    word: str
    num_frames: int
    created_at_ms: float
    source: str  # "live_capture" | "augmented" | "imported"
    parent_sample_id: Optional[str] = None       # set for augmented samples
    augmentation_strategy: Optional[str] = None   # e.g. "gaussian_noise"

    def feature_filename(self) -> str:
        """Filename (not full path) of this sample's stored feature array."""
        return f"{self.sample_id}.npy"


@dataclass
class DatasetSplit:
    """The result of splitting a dataset version into train/val/test sets,
    recorded so the exact split is reproducible later (e.g. for a fair
    comparison across model_training.py runs)."""
    seed: int
    test_ratio: float
    val_ratio: float
    train_sample_ids: List[str] = field(default_factory=list)
    val_sample_ids: List[str] = field(default_factory=list)
    test_sample_ids: List[str] = field(default_factory=list)
    created_at_ms: float = 0.0


@dataclass
class DatasetStats:
    """Summary statistics for a dataset version, useful for a dashboard
    or sanity-checking before training."""
    version: int
    total_samples: int
    samples_per_label: Dict[str, int]
    labels: List[str]


# ----------------------------------------------------------------------
# Pluggable augmentation strategies (Strategy Pattern)
# ----------------------------------------------------------------------


class AugmentationStrategy(ABC):
    """
    Abstract interface for a data-augmentation technique operating on a
    sample's feature sequence, shape (num_frames, FEATURE_VECTOR_SIZE).

    Every strategy must be shape-preserving in the sense that it returns
    a valid (num_frames', FEATURE_VECTOR_SIZE) array (num_frames' may
    differ for time-based augmentations, though the ones provided here
    resample back to the original length for training-batch simplicity).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """A short, filesystem/log-safe identifier for this strategy,
        recorded in SampleMetadata.augmentation_strategy."""
        raise NotImplementedError

    @abstractmethod
    def apply(self, sequence: np.ndarray) -> np.ndarray:
        """Return an augmented COPY of `sequence` (must not mutate the
        input in place, since the same original sample may be reused
        across multiple augmentation calls)."""
        raise NotImplementedError


class GaussianNoiseAugmentation(AugmentationStrategy):
    """
    Adds small random jitter to every landmark coordinate, simulating
    natural camera/detector noise and sensor imprecision. This is the
    simplest and most broadly applicable augmentation for landmark-based
    data -- it teaches the model to be robust to minor tracking noise
    rather than overfitting to exact coordinates.
    """

    def __init__(self, std: float = 0.01) -> None:
        self._std = std

    @property
    def name(self) -> str:
        return "gaussian_noise"

    def apply(self, sequence: np.ndarray) -> np.ndarray:
        noise = np.random.normal(loc=0.0, scale=self._std, size=sequence.shape)
        return (sequence + noise).astype(np.float32)


class HorizontalFlipAugmentation(AugmentationStrategy):
    """
    Mirrors a sample horizontally, simulating a left-handed signer
    performing what a right-handed signer signed (or vice versa). This
    requires TWO operations on our [LEFT(63) | RIGHT(63)] feature layout
    (see continuous_recognition.py's LandmarkFeatureExtractor):
      1. Mirror the x-coordinate of every landmark (x -> 1.0 - x).
      2. Swap the LEFT-hand block and RIGHT-hand block, since a
         horizontally mirrored left hand looks like a right hand's
         mirror image, not a left hand's.
    """

    @property
    def name(self) -> str:
        return "horizontal_flip"

    def apply(self, sequence: np.ndarray) -> np.ndarray:
        flipped = sequence.copy()

        # Mirror x-coordinates: every 3rd value starting at index 0 is an
        # x-coordinate (layout per hand is [x,y,z, x,y,z, ...]).
        flipped[:, 0::3] = 1.0 - flipped[:, 0::3]

        # Swap the LEFT-hand block [0:FEATURES_PER_HAND] with the
        # RIGHT-hand block [FEATURES_PER_HAND:FEATURE_VECTOR_SIZE].
        left_block = flipped[:, :FEATURES_PER_HAND].copy()
        right_block = flipped[:, FEATURES_PER_HAND:FEATURE_VECTOR_SIZE].copy()
        flipped[:, :FEATURES_PER_HAND] = right_block
        flipped[:, FEATURES_PER_HAND:FEATURE_VECTOR_SIZE] = left_block

        return flipped.astype(np.float32)


class TimeWarpAugmentation(AugmentationStrategy):
    """
    Speeds up or slows down a sample's motion by resampling it along the
    time axis, then interpolating back to the ORIGINAL frame count. This
    simulates natural variation in how fast different people sign the
    same word, without changing the sequence length the model expects.
    """

    def __init__(self, warp_factor_range: tuple = (0.8, 1.25)) -> None:
        self._warp_factor_range = warp_factor_range

    @property
    def name(self) -> str:
        return "time_warp"

    def apply(self, sequence: np.ndarray) -> np.ndarray:
        num_frames = sequence.shape[0]
        if num_frames < 2:
            # Nothing meaningful to time-warp with fewer than 2 frames.
            return sequence.copy()

        warp_factor = random.uniform(*self._warp_factor_range)
        original_indices = np.arange(num_frames)

        # "Virtual" warped timeline: e.g. warp_factor=0.8 (slower) stretches
        # the sample over more virtual time; we then resample back onto
        # exactly `num_frames` points, which is what actually speeds up or
        # slows down the perceived motion once played back at a fixed
        # frame rate.
        warped_length = max(2, int(round(num_frames * warp_factor)))
        warped_indices = np.linspace(0, num_frames - 1, warped_length)
        resample_indices = np.linspace(0, warped_length - 1, num_frames)

        output = np.zeros_like(sequence)
        for feature_idx in range(sequence.shape[1]):
            # Interpolate original -> warped timeline, then warped -> back
            # to the original frame count.
            warped_values = np.interp(
                warped_indices, original_indices, sequence[:, feature_idx]
            )
            output[:, feature_idx] = np.interp(
                resample_indices, np.arange(warped_length), warped_values
            )

        return output.astype(np.float32)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@dataclass
class DatasetManagerConfig:
    """Tunable parameters for dataset storage and splitting behaviour."""

    # Root directory where all dataset versions are stored, matching the
    # project's top-level datasets/ folder.
    root_dir: Path = field(default_factory=lambda: Path("datasets"))

    # Default split ratios used by split_dataset() if not overridden.
    default_test_ratio: float = 0.15
    default_val_ratio: float = 0.15
    default_split_seed: int = 42


# ----------------------------------------------------------------------
# Dataset manager
# ----------------------------------------------------------------------


class DatasetManager:
    """
    Owns the on-disk sign-language dataset: sample collection, label
    registry, train/val/test splitting, augmentation, and versioning.

    Usage:
        manager = DatasetManager()
        manager.add_sample("hello", feature_sequence)          # collect
        manager.augment_dataset([GaussianNoiseAugmentation()])  # augment
        split = manager.split_dataset()                         # split
        new_version = manager.create_new_version("added 20 new signs")
    """

    def __init__(self, config: Optional[DatasetManagerConfig] = None) -> None:
        self._config = config or DatasetManagerConfig()
        self._root_dir = Path(self._config.root_dir)
        self._root_dir.mkdir(parents=True, exist_ok=True)

        self._current_version = self._load_or_init_current_version()
        self._ensure_version_dir(self._current_version)

        self._labels: List[str] = self._load_labels(self._current_version)
        self._manifest: Dict[str, SampleMetadata] = self._load_manifest(
            self._current_version
        )

        logger.info(
            "DatasetManager initialized (root=%s, version=%d, samples=%d, labels=%d).",
            self._root_dir,
            self._current_version,
            len(self._manifest),
            len(self._labels),
        )

    # ------------------------------------------------------------------
    # Version/path bookkeeping
    # ------------------------------------------------------------------

    def _metadata_path(self) -> Path:
        return self._root_dir / "metadata.json"

    def _version_dir(self, version: int) -> Path:
        return self._root_dir / f"v{version}"

    def _features_dir(self, version: int) -> Path:
        return self._version_dir(version) / "features"

    def _labels_path(self, version: int) -> Path:
        return self._version_dir(version) / "labels.json"

    def _manifest_path(self, version: int) -> Path:
        return self._version_dir(version) / "manifest.json"

    def _splits_path(self, version: int) -> Path:
        return self._version_dir(version) / "splits.json"

    def _load_or_init_current_version(self) -> int:
        path = self._metadata_path()
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return int(json.load(f)["current_version"])
            except Exception:
                logger.exception(
                    "Failed to read dataset metadata.json; defaulting to version 1."
                )
        return 1

    def _save_current_version_pointer(self) -> None:
        with open(self._metadata_path(), "w", encoding="utf-8") as f:
            json.dump({"current_version": self._current_version}, f, indent=2)

    def _ensure_version_dir(self, version: int) -> None:
        self._version_dir(version).mkdir(parents=True, exist_ok=True)
        self._features_dir(version).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Label management
    # ------------------------------------------------------------------

    def _load_labels(self, version: int) -> List[str]:
        path = self._labels_path(version)
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f).get("labels", [])
        except Exception:
            logger.exception("Failed to read labels.json for version %d.", version)
            return []

    def _save_labels(self) -> None:
        with open(self._labels_path(self._current_version), "w", encoding="utf-8") as f:
            json.dump({"labels": self._labels}, f, indent=2)

    def add_label(self, word: str) -> bool:
        """
        Register a new sign/word in the label vocabulary.

        Args:
            word: The word to register.

        Returns:
            True if this was a genuinely new label; False if it was
            already registered (idempotent -- safe to call repeatedly).
        """
        normalized = word.strip().lower()
        if not normalized:
            raise ValueError("Cannot add an empty label.")

        if normalized in self._labels:
            return False

        self._labels.append(normalized)
        self._save_labels()
        logger.info("Added new sign label: '%s'", normalized)
        return True

    def get_labels(self) -> List[str]:
        """Return the current label vocabulary, in registration order."""
        return list(self._labels)

    def get_label_to_id(self) -> Dict[str, int]:
        """Return a stable word -> integer-id mapping (registration
        order), for model_training.py's classifier output layer."""
        return {label: idx for idx, label in enumerate(self._labels)}

    # ------------------------------------------------------------------
    # Sample collection
    # ------------------------------------------------------------------

    def _load_manifest(self, version: int) -> Dict[str, SampleMetadata]:
        path = self._manifest_path(version)
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_samples = json.load(f).get("samples", [])
            return {s["sample_id"]: SampleMetadata(**s) for s in raw_samples}
        except Exception:
            logger.exception("Failed to read manifest.json for version %d.", version)
            return {}

    def _save_manifest(self) -> None:
        with open(self._manifest_path(self._current_version), "w", encoding="utf-8") as f:
            json.dump(
                {"samples": [asdict(s) for s in self._manifest.values()]}, f, indent=2
            )

    def add_sample(
        self,
        word: str,
        feature_sequence: np.ndarray,
        source: str = "live_capture",
        parent_sample_id: Optional[str] = None,
        augmentation_strategy: Optional[str] = None,
    ) -> SampleMetadata:
        """
        Collect and persist one training sample.

        Args:
            word: The sign this sample demonstrates. Auto-registered as a
                new label if not already known (fulfills "Add new signs").
            feature_sequence: A numpy array of shape
                (num_frames, FEATURE_VECTOR_SIZE) -- the same layout
                continuous_recognition.py's LandmarkFeatureExtractor
                produces at inference time.
            source: Provenance tag, e.g. "live_capture" or "augmented".
            parent_sample_id: For augmented samples, the sample_id this
                one was derived from (lineage tracking).
            augmentation_strategy: For augmented samples, the name of the
                AugmentationStrategy used.

        Returns:
            The SampleMetadata record for the newly stored sample.

        Raises:
            ValueError: If feature_sequence has the wrong shape.
        """
        if feature_sequence.ndim != 2 or feature_sequence.shape[1] != FEATURE_VECTOR_SIZE:
            raise ValueError(
                f"feature_sequence must have shape (num_frames, {FEATURE_VECTOR_SIZE}); "
                f"got {feature_sequence.shape}. This must match "
                f"continuous_recognition.py's feature layout exactly."
            )

        normalized_word = word.strip().lower()
        self.add_label(normalized_word)  # no-op if already registered

        sample_id = str(uuid.uuid4())
        metadata = SampleMetadata(
            sample_id=sample_id,
            word=normalized_word,
            num_frames=int(feature_sequence.shape[0]),
            created_at_ms=time.time() * 1000.0,
            source=source,
            parent_sample_id=parent_sample_id,
            augmentation_strategy=augmentation_strategy,
        )

        feature_path = self._features_dir(self._current_version) / metadata.feature_filename()
        try:
            np.save(feature_path, feature_sequence.astype(np.float32))
        except OSError:
            logger.exception("Failed to write feature file for sample %s.", sample_id)
            raise

        self._manifest[sample_id] = metadata
        self._save_manifest()

        logger.info(
            "Collected sample %s for '%s' (%d frames, source=%s).",
            sample_id,
            normalized_word,
            metadata.num_frames,
            source,
        )
        return metadata

    def get_samples(self, word: Optional[str] = None) -> List[SampleMetadata]:
        """Return sample metadata, optionally filtered to a single word."""
        samples = list(self._manifest.values())
        if word is not None:
            normalized = word.strip().lower()
            samples = [s for s in samples if s.word == normalized]
        return samples

    def load_sample_features(self, sample_id: str) -> np.ndarray:
        """
        Load a sample's stored feature array from disk.

        Raises:
            KeyError: If sample_id isn't in the current version's manifest.
            FileNotFoundError: If the manifest references a feature file
                that's missing from disk (indicates dataset corruption).
        """
        if sample_id not in self._manifest:
            raise KeyError(f"Unknown sample_id: {sample_id}")

        metadata = self._manifest[sample_id]
        path = self._features_dir(self._current_version) / metadata.feature_filename()
        if not path.exists():
            raise FileNotFoundError(
                f"Feature file missing for sample {sample_id} at {path}"
            )
        return np.load(path)

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    def augment_dataset(
        self,
        strategies: List[AugmentationStrategy],
        words: Optional[List[str]] = None,
        multiplier: int = 1,
    ) -> List[SampleMetadata]:
        """
        Generate augmented copies of existing samples and add them to the
        dataset as new samples, preserving lineage back to the original.

        Args:
            strategies: Which augmentation techniques to apply. Each
                selected source sample gets ONE augmented copy per
                strategy per multiplier repetition (so len(strategies) *
                multiplier new samples per source sample).
            words: If provided, only augment samples for these words.
                If None, augments the entire current dataset.
            multiplier: How many times to repeat the augmentation set per
                source sample (useful for stochastic strategies like
                GaussianNoiseAugmentation, where each repetition yields a
                different random result).

        Returns:
            The list of newly created SampleMetadata records.
        """
        source_samples = []
        if words:
            for w in words:
                source_samples.extend(self.get_samples(word=w))
        else:
            source_samples = [
                s for s in self._manifest.values() if s.source != "augmented"
            ]

        new_samples: List[SampleMetadata] = []
        for source in source_samples:
            try:
                original_features = self.load_sample_features(source.sample_id)
            except (KeyError, FileNotFoundError):
                logger.exception(
                    "Skipping augmentation for missing/corrupt sample %s.",
                    source.sample_id,
                )
                continue

            for _ in range(multiplier):
                for strategy in strategies:
                    try:
                        augmented_features = strategy.apply(original_features)
                        new_metadata = self.add_sample(
                            word=source.word,
                            feature_sequence=augmented_features,
                            source="augmented",
                            parent_sample_id=source.sample_id,
                            augmentation_strategy=strategy.name,
                        )
                        new_samples.append(new_metadata)
                    except Exception:
                        logger.exception(
                            "Augmentation strategy '%s' failed on sample %s.",
                            strategy.name,
                            source.sample_id,
                        )

        logger.info(
            "Augmentation complete: %d new sample(s) created from %d source sample(s).",
            len(new_samples),
            len(source_samples),
        )
        return new_samples

    # ------------------------------------------------------------------
    # Train/val/test splitting
    # ------------------------------------------------------------------

    def split_dataset(
        self,
        test_ratio: Optional[float] = None,
        val_ratio: Optional[float] = None,
        seed: Optional[int] = None,
        stratify: bool = True,
    ) -> DatasetSplit:
        """
        Partition the current version's samples into train/validation/test
        sets and persist the split to disk for reproducibility.

        Args:
            test_ratio: Fraction of samples reserved for testing. Defaults
                to config.default_test_ratio.
            val_ratio: Fraction reserved for validation. Defaults to
                config.default_val_ratio.
            seed: Random seed for reproducibility. Defaults to
                config.default_split_seed.
            stratify: If True (recommended, and the default), the split
                is performed PER LABEL so every word is proportionally
                represented in train/val/test -- important for sign
                language data where some words may have far fewer
                recorded samples than others.

        Returns:
            The DatasetSplit describing which sample_ids landed in which
            partition.
        """
        test_ratio = test_ratio if test_ratio is not None else self._config.default_test_ratio
        val_ratio = val_ratio if val_ratio is not None else self._config.default_val_ratio
        seed = seed if seed is not None else self._config.default_split_seed

        if test_ratio + val_ratio >= 1.0:
            raise ValueError(
                f"test_ratio ({test_ratio}) + val_ratio ({val_ratio}) must be < 1.0"
            )

        rng = random.Random(seed)
        train_ids: List[str] = []
        val_ids: List[str] = []
        test_ids: List[str] = []

        if stratify:
            samples_by_label: Dict[str, List[str]] = defaultdict(list)
            for sample in self._manifest.values():
                samples_by_label[sample.word].append(sample.sample_id)

            for label, sample_ids in samples_by_label.items():
                shuffled = list(sample_ids)
                rng.shuffle(shuffled)
                n = len(shuffled)
                n_test = round(n * test_ratio)
                n_val = round(n * val_ratio)

                test_ids.extend(shuffled[:n_test])
                val_ids.extend(shuffled[n_test : n_test + n_val])
                train_ids.extend(shuffled[n_test + n_val :])

                if n > 0 and (n_test == 0 or n_val == 0):
                    logger.warning(
                        "Label '%s' has only %d sample(s); train/val/test "
                        "split may be degenerate for this label.",
                        label,
                        n,
                    )
        else:
            all_ids = list(self._manifest.keys())
            rng.shuffle(all_ids)
            n = len(all_ids)
            n_test = round(n * test_ratio)
            n_val = round(n * val_ratio)
            test_ids = all_ids[:n_test]
            val_ids = all_ids[n_test : n_test + n_val]
            train_ids = all_ids[n_test + n_val :]

        split = DatasetSplit(
            seed=seed,
            test_ratio=test_ratio,
            val_ratio=val_ratio,
            train_sample_ids=train_ids,
            val_sample_ids=val_ids,
            test_sample_ids=test_ids,
            created_at_ms=time.time() * 1000.0,
        )

        with open(self._splits_path(self._current_version), "w", encoding="utf-8") as f:
            json.dump(asdict(split), f, indent=2)

        logger.info(
            "Dataset split complete: %d train / %d val / %d test (seed=%d).",
            len(train_ids),
            len(val_ids),
            len(test_ids),
            seed,
        )
        return split

    def get_split(self) -> Optional[DatasetSplit]:
        """Load the most recently saved split for the current version, if
        one exists."""
        path = self._splits_path(self._current_version)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return DatasetSplit(**json.load(f))
        except Exception:
            logger.exception("Failed to read splits.json for version %d.", self._current_version)
            return None

    # ------------------------------------------------------------------
    # Versioning
    # ------------------------------------------------------------------

    def get_current_version(self) -> int:
        """Return the currently active dataset version number."""
        return self._current_version

    def create_new_version(self, description: str = "") -> int:
        """
        Create a new dataset version by snapshotting the current one
        forward (labels + manifest carried over; feature files are
        hard-linked/copied so both versions remain independently valid
        and browsable).

        This lets the dataset evolve (new samples, new labels, cleanup)
        while old versions stay intact for reproducibility -- e.g. so
        evaluation.py can report which dataset version a given trained
        model was trained against.

        Args:
            description: Optional human-readable note about why this
                version was created (e.g. "added 20 new ASL signs"),
                stored in the new version's metadata.

        Returns:
            The new version number.
        """
        new_version = self._current_version + 1
        self._ensure_version_dir(new_version)

        # Carry labels and manifest forward.
        new_labels = list(self._labels)
        new_manifest = dict(self._manifest)

        with open(self._labels_path(new_version), "w", encoding="utf-8") as f:
            json.dump({"labels": new_labels}, f, indent=2)
        with open(self._manifest_path(new_version), "w", encoding="utf-8") as f:
            json.dump(
                {"samples": [asdict(s) for s in new_manifest.values()]}, f, indent=2
            )

        # Copy feature files forward so the new version is fully
        # self-contained (doesn't rely on the old version's files
        # remaining on disk).
        old_features_dir = self._features_dir(self._current_version)
        new_features_dir = self._features_dir(new_version)
        for sample in new_manifest.values():
            src = old_features_dir / sample.feature_filename()
            dst = new_features_dir / sample.feature_filename()
            if src.exists() and not dst.exists():
                dst.write_bytes(src.read_bytes())

        version_info_path = self._version_dir(new_version) / "version_info.json"
        with open(version_info_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": new_version,
                    "created_at_ms": time.time() * 1000.0,
                    "description": description,
                    "carried_forward_from_version": self._current_version,
                },
                f,
                indent=2,
            )

        self._current_version = new_version
        self._labels = new_labels
        self._manifest = new_manifest
        self._save_current_version_pointer()

        logger.info(
            "Created dataset version %d (from v%d): %s",
            new_version,
            new_version - 1,
            description or "(no description)",
        )
        return new_version

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> DatasetStats:
        """Return summary statistics for the current dataset version."""
        counts: Dict[str, int] = defaultdict(int)
        for sample in self._manifest.values():
            counts[sample.word] += 1

        return DatasetStats(
            version=self._current_version,
            total_samples=len(self._manifest),
            samples_per_label=dict(counts),
            labels=list(self._labels),
        )


if __name__ == "__main__":
    # Minimal manual smoke-test using a temporary directory, so running
    # this file never touches the real project datasets/ folder. Run via:
    #   python dataset_manager.py
    import shutil
    import tempfile

    logger.info("Running dataset_manager.py standalone demo.")

    temp_dir = Path(tempfile.mkdtemp(prefix="dataset_manager_demo_"))
    try:
        manager = DatasetManager(DatasetManagerConfig(root_dir=temp_dir))

        # Simulate collecting a few samples for two words.
        rng = np.random.default_rng(seed=0)
        for word, count in [("hello", 5), ("thanks", 2)]:
            for _ in range(count):
                fake_sequence = rng.random((30, FEATURE_VECTOR_SIZE)).astype(np.float32)
                manager.add_sample(word, fake_sequence)

        print("Labels:", manager.get_labels())
        print("Stats before augmentation:", manager.get_stats())

        manager.augment_dataset(
            strategies=[GaussianNoiseAugmentation(), HorizontalFlipAugmentation()],
            multiplier=1,
        )
        print("Stats after augmentation:", manager.get_stats())

        split = manager.split_dataset(test_ratio=0.2, val_ratio=0.2, seed=1)
        print(
            f"Split: train={len(split.train_sample_ids)}, "
            f"val={len(split.val_sample_ids)}, test={len(split.test_sample_ids)}"
        )

        new_version = manager.create_new_version("demo version bump")
        print("New version:", new_version)
        print("Stats in new version:", manager.get_stats())
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    logger.info("Demo complete.")
