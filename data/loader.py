"""
Data loading and validation for neural spike data.
"""

import numpy as np
import pandas as pd
from typing import Optional, List
import logging

from .structures import SpikeData, ExperimentConfig

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads and validates neural spike data from CSV files."""
    
    REQUIRED_COLUMNS = ['trial', 'channel', 'orientation', 'time']
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
    
    def load_csv(
        self,
        filepath: str,
        filter_trials: Optional[List[int]] = None,
        filter_channels: Optional[List[int]] = None,
        filter_orientations: Optional[List[int]] = None
    ) -> SpikeData:
        """Load spike data from CSV file."""
        logger.info(f"Loading data from {filepath}")
        
        # Read CSV
        df = pd.read_csv(filepath)
        
        # Validate columns
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Apply filters
        if filter_trials is not None:
            df = df[df['trial'].isin(filter_trials)]
        if filter_channels is not None:
            df = df[df['channel'].isin(filter_channels)]
        if filter_orientations is not None:
            df = df[df['orientation'].isin(filter_orientations)]
        
        if len(df) == 0:
            raise ValueError("No data remaining after filtering")
        
        # Get unique values
        trials = np.sort(df['trial'].unique())
        channels = np.sort(df['channel'].unique())
        orientations_per_trial = df.groupby('trial')['orientation'].first()
        orientations = orientations_per_trial.loc[trials].values
        
        # Create time bins
        times = np.arange(
            self.config.trial_start_time,
            self.config.trial_end_time,
            self.config.bin_size
        )
        n_bins = len(times)
        
        # Initialize arrays
        n_trials = len(trials)
        n_channels = len(channels)
        
        spike_times = [[[] for _ in range(n_channels)] for _ in range(n_trials)]
        spike_binned = np.zeros((n_trials, n_channels, n_bins), dtype=np.float32)
        
        # Process spikes
        logger.info("Binning spike times...")
        
        for trial_idx, trial_id in enumerate(trials):
            trial_data = df[df['trial'] == trial_id]
            
            for ch_idx, ch_id in enumerate(channels):
                ch_data = trial_data[trial_data['channel'] == ch_id]
                times_ch = ch_data['time'].values
                
                spike_times[trial_idx][ch_idx] = times_ch
                
                if len(times_ch) > 0:
                    # Filter spikes within trial window first
                    valid_times = (times_ch >= self.config.trial_start_time) & (times_ch < self.config.trial_end_time)
                    times_ch_valid = times_ch[valid_times]
                    
                    if len(times_ch_valid) > 0:
                        # Bin the spikes
                        bins = np.digitize(times_ch_valid, times) - 1
                        # Clip to ensure last bin catches spikes at the end
                        bins = np.clip(bins, 0, n_bins - 1)
                        
                        counts = np.bincount(bins, minlength=n_bins)
                        spike_binned[trial_idx, ch_idx, :] = counts / self.config.bin_size   
        
        # Convert spike_times to numpy array
        spike_times_array = np.empty((n_trials, n_channels), dtype=object)
        for i in range(n_trials):
            for j in range(n_channels):
                spike_times_array[i, j] = np.array(spike_times[i][j])
        
        logger.info(f"Loaded {n_trials} trials, {n_channels} channels, {n_bins} bins")
        
        return SpikeData(
            trials=trials,
            channels=channels,
            orientations=orientations,
            spike_times=spike_times_array,
            spike_binned=spike_binned,
            times=times,
            config=self.config
        )
