"""
Data structures for neural spike analysis.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import numpy as np
from enum import Enum


class Orientation(Enum):
    """Stimulus orientation types."""
    DEGREES_0 = 0
    DEGREES_90 = 90
    ALL = -1


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable experiment configuration."""
    trial_start_time: float
    trial_end_time: float
    grating_on_time: float
    grating_off_time: float
    bin_size: float
    
    def __post_init__(self):
        """Validate timing configuration."""
        if self.trial_start_time >= self.trial_end_time:
            raise ValueError("trial_start_time must be before trial_end_time")
        if not (self.trial_start_time <= self.grating_on_time <= self.trial_end_time):
            raise ValueError("grating_on_time must be within trial window")
        if not (self.trial_start_time <= self.grating_off_time <= self.trial_end_time):
            raise ValueError("grating_off_time must be within trial window")
        if self.bin_size <= 0:
            raise ValueError("bin_size must be positive")
    
    @property
    def trial_duration(self) -> float:
        return self.trial_end_time - self.trial_start_time
    
    @property
    def grating_duration(self) -> float:
        return self.grating_off_time - self.grating_on_time
    
    @property
    def n_bins(self) -> int:
        return int(np.ceil(self.trial_duration / self.bin_size))


@dataclass
class SpikeData:
    """Container for neural spike data."""
    trials: np.ndarray
    channels: np.ndarray
    orientations: np.ndarray
    spike_times: np.ndarray
    spike_binned: np.ndarray
    times: np.ndarray
    config: ExperimentConfig
    
    def __post_init__(self):
        """Validate data consistency."""
        n_trials = len(self.trials)
        n_channels = len(self.channels)
        n_bins = len(self.times)
        
        if self.spike_binned.shape != (n_trials, n_channels, n_bins):
            raise ValueError(
                f"spike_binned shape {self.spike_binned.shape} doesn't match "
                f"expected ({n_trials}, {n_channels}, {n_bins})"
            )
    
    @property
    def n_trials(self) -> int:
        return len(self.trials)
    
    @property
    def n_channels(self) -> int:
        return len(self.channels)
    
    @property
    def n_bins(self) -> int:
        return len(self.times)


@dataclass
class Feature:
    """Feature representation for machine learning."""
    X: np.ndarray                    # Feature matrix (n_samples, n_features)
    y: np.ndarray                    # Labels (n_samples,)
    sample_ids: np.ndarray           # Sample identifiers
    feature_names: List[str]         # Feature names
    class_names: List[str] = field(default_factory=list)  # Class names
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate feature data."""
        if self.X.shape[0] != len(self.y):
            raise ValueError("X and y must have same number of samples")
        if self.X.shape[0] != len(self.sample_ids):
            raise ValueError("X and sample_ids must have same number of samples")
    
    @property
    def n_samples(self) -> int:
        return self.X.shape[0]
    
    @property
    def n_features(self) -> int:
        return int(np.prod(self.X.shape[1:]))


@dataclass
class PSTH:
    """Peri-Stimulus Time Histogram data."""
    psth: List[np.ndarray]
    orientations: List[int]
    channels: np.ndarray
    times: np.ndarray
    bin_size: float
    n_trials_per_orientation: List[int]


@dataclass
class ConnectivityMatrix:
    """Connectivity or similarity matrix between channels."""
    matrices: List[np.ndarray]
    conditions: List[Tuple[int, int]]
    channels: np.ndarray
    method: str
    normalization: bool
    metadata: Dict = field(default_factory=dict)


@dataclass
class StatisticalResult:
    """Results from statistical testing."""
    p_values: np.ndarray
    effect_sizes: np.ndarray
    significant_mask: np.ndarray
    method: str
    correction: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class ClassificationResult:
    """Results from classification analysis."""
    train_accuracies: np.ndarray
    test_accuracies: np.ndarray
    feature_importances: Optional[np.ndarray] = None
    permutation_importance: Optional[np.ndarray] = None
    model_name: str = "Unknown"
    metadata: Dict = field(default_factory=dict)
    
    @property
    def mean_train_acc(self) -> float:
        return float(np.mean(self.train_accuracies))
    
    @property
    def mean_test_acc(self) -> float:
        return float(np.mean(self.test_accuracies))
