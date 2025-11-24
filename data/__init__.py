"""
Data module for neural spike analysis.
"""

from .structures import (
    Orientation,
    ExperimentConfig,
    SpikeData,
    Feature,
    PSTH,
    ConnectivityMatrix,
    StatisticalResult,
    ClassificationResult
)

from .loader import DataLoader

__all__ = [
    'Orientation',
    'ExperimentConfig',
    'SpikeData',
    'Feature',
    'PSTH',
    'ConnectivityMatrix',
    'StatisticalResult',
    'ClassificationResult',
    'DataLoader'
]
