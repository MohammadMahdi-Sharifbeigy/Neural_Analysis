"""
Data preprocessing for neural spike analysis.

This module provides preprocessing operations for SpikeData objects including
normalization, filtering, channel removal, and data transformations.
"""

import numpy as np
from typing import Optional, List, Tuple, Union
from copy import deepcopy
import logging

from .structures import SpikeData, ExperimentConfig

logger = logging.getLogger(__name__)


class SpikeDataPreprocessor:
    """
    Preprocessor for neural spike data with various normalization and filtering options.
    
    This class provides methods to preprocess SpikeData objects including:
    - Channel removal
    - Z-score normalization (global or per-condition)
    - Noise subtraction (trial-averaged baseline)
    - Time window extraction
    - Data quality filtering
    """
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        """
        Initialize preprocessor.
        
        Args:
            config: Experiment configuration (optional)
        """
        self.config = config
        self._normalization_params = {}
        self._noise_baseline = {}
    
    def remove_channels(
        self, 
        data: SpikeData, 
        channels_to_remove: List[int],
        inplace: bool = False
    ) -> SpikeData:
        """
        Remove specified channels from data.
        
        Args:
            data: Input spike data
            channels_to_remove: List of channel IDs to remove
            inplace: Modify data in place (default: False)
        
        Returns:
            SpikeData with channels removed
        """
        channels_to_remove = np.array(channels_to_remove)
        all_channels = data.channels
        
        # Find channels to keep
        keep_mask = ~np.isin(all_channels, channels_to_remove)
        
        if not np.any(keep_mask):
            raise ValueError("Cannot remove all channels")
        
        # Create new or modify existing data
        if inplace:
            result = data
        else:
            result = deepcopy(data)
        
        # Update arrays
        result.channels = all_channels[keep_mask]
        result.spike_times = data.spike_times[:, keep_mask]
        result.spike_binned = data.spike_binned[:, keep_mask, :]
        
        n_removed = len(channels_to_remove)
        n_remaining = np.sum(keep_mask)
        logger.info(f"Removed {n_removed} channels, {n_remaining} remaining")
        
        return result
    
    def extract_time_window(
        self,
        data: SpikeData,
        start_time: float,
        end_time: float,
        inplace: bool = False
    ) -> SpikeData:
        """
        Extract a specific time window from data.
        
        Args:
            data: Input spike data
            start_time: Window start time (seconds)
            end_time: Window end time (seconds)
            inplace: Modify data in place (default: False)
        
        Returns:
            SpikeData with extracted time window
        """
        if start_time >= end_time:
            raise ValueError("start_time must be less than end_time")
        
        # Find time bins within window
        time_mask = (data.times >= start_time) & (data.times <= end_time)
        
        if not np.any(time_mask):
            raise ValueError(f"No time bins found in window [{start_time}, {end_time}]")
        
        # Create new or modify existing data
        if inplace:
            result = data
        else:
            result = deepcopy(data)
        
        # Update arrays
        result.times = data.times[time_mask]
        result.spike_binned = data.spike_binned[:, :, time_mask]
        
        n_bins = np.sum(time_mask)
        logger.info(f"Extracted time window [{start_time}, {end_time}] with {n_bins} bins")
        
        return result
    
    def normalize_zscore(
        self,
        data: SpikeData,
        method: str = 'global',
        epsilon: float = 1e-8,
        inplace: bool = False,
        fit: bool = True,
        use_stored_params: bool = False
    ) -> SpikeData:
        """
        Apply z-score normalization to spike data.
        
        Args:
            data: Input spike data
            method: Normalization method
                - 'global': Normalize across all trials and time bins
                - 'per_channel': Normalize each channel independently
                - 'per_condition': Normalize within each stimulus condition
            epsilon: Small value to avoid division by zero
            inplace: Modify data in place (default: False)
            fit: Compute normalization parameters (default: True)
            use_stored_params: Use previously stored parameters (default: False)
        
        Returns:
            Normalized SpikeData
        """
        if method not in ['global', 'per_channel', 'per_condition']:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Create new or modify existing data
        if inplace:
            result = data
        else:
            result = deepcopy(data)
        
        spike_binned = result.spike_binned.copy()
        
        if use_stored_params and method in self._normalization_params:
            # Use stored parameters
            params = self._normalization_params[method]
            mean = params['mean']
            std = params['std']
            logger.info(f"Using stored normalization parameters for method '{method}'")
        elif fit:
            # Compute normalization parameters
            if method == 'global':
                # Normalize across all trials and time
                mean = np.mean(spike_binned, axis=(0, 2), keepdims=True)
                std = np.std(spike_binned, axis=(0, 2), keepdims=True)
            
            elif method == 'per_channel':
                # Normalize each channel independently
                mean = np.mean(spike_binned, axis=(0, 2), keepdims=True)
                std = np.std(spike_binned, axis=(0, 2), keepdims=True)
            
            elif method == 'per_condition':
                # Normalize within each orientation condition
                mean = np.zeros_like(spike_binned)
                std = np.zeros_like(spike_binned)
                
                for orientation in np.unique(data.orientations):
                    mask = data.orientations == orientation
                    mean[mask] = np.mean(spike_binned[mask], axis=(0, 2), keepdims=True)
                    std[mask] = np.std(spike_binned[mask], axis=(0, 2), keepdims=True)
            
            # Store parameters
            self._normalization_params[method] = {
                'mean': mean,
                'std': std,
                'method': method
            }
        else:
            raise ValueError("Must either fit=True or use_stored_params=True")
        
        # Apply normalization
        spike_binned_normalized = (spike_binned - mean) / (std + epsilon)
        result.spike_binned = spike_binned_normalized
        
        logger.info(f"Applied z-score normalization (method: {method})")
        
        return result
    
    def subtract_noise(
        self,
        data: SpikeData,
        method: str = 'trial_mean',
        inplace: bool = False,
        fit: bool = True,
        use_stored_baseline: bool = False
    ) -> SpikeData:
        """
        Subtract noise baseline from spike data.
        
        Args:
            data: Input spike data
            method: Noise estimation method
                - 'trial_mean': Subtract trial-averaged activity
                - 'baseline_window': Subtract mean activity in baseline window
            inplace: Modify data in place (default: False)
            fit: Compute noise baseline (default: True)
            use_stored_baseline: Use previously stored baseline (default: False)
        
        Returns:
            Noise-subtracted SpikeData
        """
        if method not in ['trial_mean', 'baseline_window']:
            raise ValueError(f"Unknown noise subtraction method: {method}")
        
        # Create new or modify existing data
        if inplace:
            result = data
        else:
            result = deepcopy(data)
        
        spike_binned = result.spike_binned.copy()
        
        if use_stored_baseline and method in self._noise_baseline:
            # Use stored baseline
            baseline = self._noise_baseline[method]
            logger.info(f"Using stored noise baseline for method '{method}'")
        elif fit:
            # Compute noise baseline
            if method == 'trial_mean':
                # Average across trials
                baseline = np.mean(spike_binned, axis=0, keepdims=True)
            
            elif method == 'baseline_window':
                # Use pre-stimulus baseline window
                if self.config is None:
                    raise ValueError("Config required for baseline_window method")
                
                baseline_mask = data.times < self.config.grating_on_time
                if not np.any(baseline_mask):
                    raise ValueError("No baseline window found before grating onset")
                
                baseline = np.mean(
                    spike_binned[:, :, baseline_mask], 
                    axis=(0, 2), 
                    keepdims=True
                )
            
            # Store baseline
            self._noise_baseline[method] = baseline
        else:
            raise ValueError("Must either fit=True or use_stored_baseline=True")
        
        # Subtract baseline
        spike_binned_denoised = spike_binned - baseline
        result.spike_binned = spike_binned_denoised
        
        logger.info(f"Subtracted noise baseline (method: {method})")
        
        return result
    
    def filter_low_firing_channels(
        self,
        data: SpikeData,
        min_rate: float = 1.0,
        inplace: bool = False
    ) -> SpikeData:
        """
        Remove channels with mean firing rate below threshold.
        
        Args:
            data: Input spike data
            min_rate: Minimum mean firing rate (Hz)
            inplace: Modify data in place (default: False)
        
        Returns:
            Filtered SpikeData
        """
        # Calculate mean firing rates
        mean_rates = np.mean(data.spike_binned, axis=(0, 2))
        
        # Find channels to keep
        keep_mask = mean_rates >= min_rate
        
        if not np.any(keep_mask):
            raise ValueError(f"No channels have mean rate >= {min_rate} Hz")
        
        # Create new or modify existing data
        if inplace:
            result = data
        else:
            result = deepcopy(data)
        
        # Update arrays
        result.channels = data.channels[keep_mask]
        result.spike_times = data.spike_times[:, keep_mask]
        result.spike_binned = data.spike_binned[:, keep_mask, :]
        
        n_removed = np.sum(~keep_mask)
        n_remaining = np.sum(keep_mask)
        logger.info(
            f"Filtered {n_removed} low-firing channels "
            f"(< {min_rate} Hz), {n_remaining} remaining"
        )
        
        return result
    
    def filter_trials_by_orientation(
        self,
        data: SpikeData,
        orientations: List[int],
        inplace: bool = False
    ) -> SpikeData:
        """
        Keep only trials with specified orientations.
        
        Args:
            data: Input spike data
            orientations: List of orientations to keep
            inplace: Modify data in place (default: False)
        
        Returns:
            Filtered SpikeData
        """
        orientations = np.array(orientations)
        
        # Find trials to keep
        keep_mask = np.isin(data.orientations, orientations)
        
        if not np.any(keep_mask):
            raise ValueError(f"No trials found with orientations {orientations}")
        
        # Create new or modify existing data
        if inplace:
            result = data
        else:
            result = deepcopy(data)
        
        # Update arrays
        result.trials = data.trials[keep_mask]
        result.orientations = data.orientations[keep_mask]
        result.spike_times = data.spike_times[keep_mask, :]
        result.spike_binned = data.spike_binned[keep_mask, :, :]
        
        n_kept = np.sum(keep_mask)
        logger.info(f"Kept {n_kept} trials with orientations {orientations}")
        
        return result
    
    def reset_normalization_params(self):
        """Reset stored normalization parameters."""
        self._normalization_params = {}
        logger.info("Reset normalization parameters")
    
    def reset_noise_baseline(self):
        """Reset stored noise baseline."""
        self._noise_baseline = {}
        logger.info("Reset noise baseline")
    
    def get_preprocessing_summary(self) -> dict:
        """
        Get summary of preprocessing state.
        
        Returns:
            Dictionary with preprocessing information
        """
        summary = {
            'normalization_methods': list(self._normalization_params.keys()),
            'noise_methods': list(self._noise_baseline.keys()),
            'has_config': self.config is not None
        }
        
        return summary


class DataQualityChecker:
    """
    Quality control checks for spike data.
    """
    
    @staticmethod
    def check_silent_channels(data: SpikeData, threshold: float = 0.0) -> List[int]:
        """
        Find channels with no or very low activity.
        
        Args:
            data: Spike data to check
            threshold: Maximum mean rate to consider silent (Hz)
        
        Returns:
            List of silent channel IDs
        """
        mean_rates = np.mean(data.spike_binned, axis=(0, 2))
        silent_mask = mean_rates <= threshold
        silent_channels = data.channels[silent_mask].tolist()
        
        if silent_channels:
            logger.warning(f"Found {len(silent_channels)} silent channels: {silent_channels}")
        
        return silent_channels
    
    @staticmethod
    def check_outlier_trials(
        data: SpikeData, 
        n_std: float = 3.0
    ) -> List[int]:
        """
        Find trials with outlier activity levels.
        
        Args:
            data: Spike data to check
            n_std: Number of standard deviations for outlier threshold
        
        Returns:
            List of outlier trial IDs
        """
        # Calculate total spike count per trial
        trial_counts = np.sum(data.spike_binned, axis=(1, 2))
        
        mean_count = np.mean(trial_counts)
        std_count = np.std(trial_counts)
        
        # Find outliers
        outlier_mask = np.abs(trial_counts - mean_count) > n_std * std_count
        outlier_trials = data.trials[outlier_mask].tolist()
        
        if outlier_trials:
            logger.warning(
                f"Found {len(outlier_trials)} outlier trials "
                f"(>{n_std} std from mean): {outlier_trials}"
            )
        
        return outlier_trials
    
    @staticmethod
    def check_data_balance(data: SpikeData) -> dict:
        """
        Check balance of trials across conditions.
        
        Args:
            data: Spike data to check
        
        Returns:
            Dictionary with balance information
        """
        balance = {}
        
        for orientation in np.unique(data.orientations):
            n_trials = np.sum(data.orientations == orientation)
            balance[int(orientation)] = int(n_trials)
        
        is_balanced = len(set(balance.values())) == 1
        
        if not is_balanced:
            logger.warning(f"Data is imbalanced across orientations: {balance}")
        else:
            logger.info(f"Data is balanced across orientations: {balance}")
        
        return {
            'trials_per_orientation': balance,
            'is_balanced': is_balanced
        }


# Convenience function for common preprocessing pipeline
def preprocess_pipeline(
    data: SpikeData,
    config: Optional[ExperimentConfig] = None,
    channels_to_remove: Optional[List[int]] = None,
    min_firing_rate: Optional[float] = None,
    time_window: Optional[Tuple[float, float]] = None,
    normalize: bool = False,
    normalization_method: str = 'global',
    subtract_noise: bool = False,
    noise_method: str = 'trial_mean',
    inplace: bool = False
) -> SpikeData:
    """
    Apply common preprocessing pipeline to spike data.
    
    Args:
        data: Input spike data
        config: Experiment configuration
        channels_to_remove: List of channel IDs to remove (optional)
        min_firing_rate: Minimum firing rate threshold (optional)
        time_window: (start, end) time window to extract (optional)
        normalize: Apply z-score normalization
        normalization_method: Method for normalization
        subtract_noise: Subtract noise baseline
        noise_method: Method for noise subtraction
        inplace: Modify data in place
    
    Returns:
        Preprocessed SpikeData
    """
    preprocessor = SpikeDataPreprocessor(config)
    
    result = data if inplace else deepcopy(data)
    
    # Remove specific channels
    if channels_to_remove is not None:
        result = preprocessor.remove_channels(result, channels_to_remove, inplace=True)
    
    # Filter low-firing channels
    if min_firing_rate is not None:
        result = preprocessor.filter_low_firing_channels(
            result, min_firing_rate, inplace=True
        )
    
    # Extract time window
    if time_window is not None:
        start_time, end_time = time_window
        result = preprocessor.extract_time_window(
            result, start_time, end_time, inplace=True
        )
    
    # Normalize
    if normalize:
        result = preprocessor.normalize_zscore(
            result, method=normalization_method, inplace=True
        )
    
    # Subtract noise
    if subtract_noise:
        result = preprocessor.subtract_noise(
            result, method=noise_method, inplace=True
        )
    
    logger.info("Preprocessing pipeline complete")
    
    return result
