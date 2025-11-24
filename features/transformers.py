"""
Feature extraction transformers for neural spike data.
"""

import numpy as np
from typing import Optional, Literal
from copy import deepcopy
import logging

# Import from parent data module
import sys
sys.path.append('/content')
from data.structures import SpikeData, Feature
import numpy as np
from copy import deepcopy
import logging
from typing import Optional, Dict, Any, List, Tuple

from data.structures import SpikeData

logger = logging.getLogger(__name__)


class BaseFeatureTransformer:
    """Base class for feature transformers."""
    
    def __init__(self):
        self.is_fitted = False
        self._fit_params = {}
    
    def fit(self, data: SpikeData) -> 'BaseFeatureTransformer':
        raise NotImplementedError("Subclasses must implement fit()")
    
    def transform(self, data: SpikeData) -> Feature:
        raise NotImplementedError("Subclasses must implement transform()")
    
    def fit_transform(self, data: SpikeData) -> Feature:
        return self.fit(data).transform(data)
    
    def _check_is_fitted(self):
        if not self.is_fitted:
            raise RuntimeError(f"{self.__class__.__name__} is not fitted")


class FreqTransformer(BaseFeatureTransformer):
    """Extract mean firing rate features."""
    
    def __init__(self, normalize: bool = False, per_condition: bool = False):
        super().__init__()
        self.normalize = normalize
        self.per_condition = per_condition
    
    def fit(self, data: SpikeData) -> 'FreqTransformer':
        if not self.normalize:
            self.is_fitted = True
            return self
        
        spike_binned = data.spike_binned
        
        if self.per_condition:
            self._fit_params = {}
            for orientation in np.unique(data.orientations):
                mask = data.orientations == orientation
                data_subset = spike_binned[mask]
                mean = np.mean(data_subset, axis=(0, 2), keepdims=True)
                std = np.std(data_subset, axis=(0, 2), keepdims=True)
                self._fit_params[int(orientation)] = {'mean': mean, 'std': std}
        else:
            mean = np.mean(spike_binned, axis=(0, 2), keepdims=True)
            std = np.std(spike_binned, axis=(0, 2), keepdims=True)
            self._fit_params['global'] = {'mean': mean, 'std': std}
        
        self.is_fitted = True
        logger.info(
            f"FreqTransformer fitted (normalize={self.normalize}, "
            f"per_condition={self.per_condition})"
        )
        
        return self
    
    def transform(self, data: SpikeData) -> Feature:
        self._check_is_fitted()
        
        spike_binned = data.spike_binned.copy()
        
        if self.normalize:
            if self.per_condition:
                for orientation in np.unique(data.orientations):
                    mask = data.orientations == orientation
                    if int(orientation) not in self._fit_params:
                        raise ValueError(f"Orientation {orientation} not seen during fit")
                    params = self._fit_params[int(orientation)]
                    spike_binned[mask] = (
                        (spike_binned[mask] - params['mean']) / (params['std'] + 1e-8)
                    )
            else:
                params = self._fit_params['global']
                spike_binned = (spike_binned - params['mean']) / (params['std'] + 1e-8)
        
        # Compute mean firing rate
        freq = np.mean(spike_binned, axis=2)
        
        # Create feature names
        feature_names = [f"freq_ch{ch}" for ch in data.channels]
        
        # Create Feature object
        feature = Feature(
            X=freq,
            y=data.orientations,
            sample_ids=data.trials,
            feature_names=feature_names,
            class_names=['0', '90'],
            metadata={
                'transformer': 'FreqTransformer',
                'normalize': self.normalize,
                'per_condition': self.per_condition,
                'n_channels': len(data.channels),
                'bin_size': data.config.bin_size
            }
        )
        
        logger.info(f"Extracted {freq.shape[1]} frequency features")
        return feature


class ConecTransformer(BaseFeatureTransformer):
    """Extract connectivity features using cross-correlation."""
    
    def __init__(
        self,
        lag: float = 0.0,
        normalize: bool = False,
        method: Literal['Pearson', 'Covariance', 'Dot product', 'Cosine'] = 'Dot product',
        reduction: Optional[Literal['max', 'demean', 'percent', 'denorm']] = 'max',
        noise: bool = False,
        per_condition: bool = False
    ):
        super().__init__()
        self.lag = lag
        self.normalize = normalize
        self.method = method
        self.reduction = reduction
        self.noise = noise
        self.per_condition = per_condition
    
    @staticmethod
    def _shift_2d(X: np.ndarray, shift: int) -> np.ndarray:
        """Shift array along last axis."""
        if shift == 0:
            return X
        out = np.zeros_like(X)
        if shift > 0:
            out[..., shift:] = X[..., :-shift]
        else:
            out[..., :shift] = X[..., -shift:]
        return out
    
    def _compute_cross_correlation(self, data: SpikeData, bin_size: float) -> tuple:
        spike_binned = data.spike_binned.copy()
        n_trials, n_channels, n_bins = spike_binned.shape
        
        if self.noise:
            if self.per_condition:
                for orientation in np.unique(data.orientations):
                    mask = data.orientations == orientation
                    noise_mean = np.mean(spike_binned[mask], axis=0, keepdims=True)
                    spike_binned[mask] = spike_binned[mask] - noise_mean
            else:
                noise_mean = np.mean(spike_binned, axis=0, keepdims=True)
                spike_binned = spike_binned - noise_mean
                
        if self.normalize:
            if self.per_condition:
                for orientation in np.unique(data.orientations):
                    mask = data.orientations == orientation
                    data_subset = spike_binned[mask]
                    mean = np.mean(data_subset, axis=(0, 2), keepdims=True)
                    std = np.std(data_subset, axis=(0, 2), keepdims=True)
                    if self.noise:
                        spike_binned[mask] = (spike_binned[mask]) / (std + 1e-8)
                    else:
                        spike_binned[mask] = (spike_binned[mask] - mean) / (std + 1e-8)

            else:
                mean = np.mean(spike_binned, axis=(0, 2), keepdims=True)
                std = np.std(spike_binned, axis=(0, 2), keepdims=True)
                if self.noise:
                    spike_binned = (spike_binned) / (std + 1e-8)
                else:
                    spike_binned = (spike_binned - mean) / (std + 1e-8)
        
        
        max_shift_bins = int(np.ceil(self.lag / bin_size))
        shifts = np.arange(-max_shift_bins, max_shift_bins + 1)
        lags = shifts * bin_size
        n_shifts = len(shifts)
        
        cross_corr_all = np.zeros((n_trials, n_shifts, n_channels, n_channels), dtype=np.float32)
        
        for shift_idx, shift in enumerate(shifts):
            spike_shifted = self._shift_2d(spike_binned, shift)
            
            if self.method == 'Dot product':
                cross_corr_all[:, shift_idx] = np.einsum('tik,tjk->tij', spike_binned, spike_shifted)
            elif self.method == 'Pearson':
                X_c = spike_binned - spike_binned.mean(axis=2, keepdims=True)
                Y_c = spike_shifted - spike_shifted.mean(axis=2, keepdims=True)
                dot = np.einsum('tik,tjk->tij', X_c, Y_c)
                norm_x = np.sqrt(np.sum(X_c ** 2, axis=2))
                norm_y = np.sqrt(np.sum(Y_c ** 2, axis=2))
                cross_corr_all[:, shift_idx] = dot / (norm_x[:, :, None] * norm_y[:, None, :] + 1e-8)
            elif self.method == 'Covariance':
                X_c = spike_binned - spike_binned.mean(axis=2, keepdims=True)
                Y_c = spike_shifted - spike_shifted.mean(axis=2, keepdims=True)
                cross_corr_all[:, shift_idx] = np.einsum('tik,tjk->tij', X_c, Y_c) / n_bins
            elif self.method == 'Cosine':
                dot = np.einsum('tik,tjk->tij', spike_binned, spike_shifted)
                norm_x = np.sqrt(np.sum(spike_binned ** 2, axis=2))
                norm_y = np.sqrt(np.sum(spike_shifted ** 2, axis=2))
                cross_corr_all[:, shift_idx] = dot / (norm_x[:, :, None] * norm_y[:, None, :] + 1e-8)
        
        cross_corr_mean = np.mean(cross_corr_all, axis=0)
        return cross_corr_all, cross_corr_mean, lags

    
    def fit(self, data: SpikeData) -> 'ConecTransformer':
        bin_size = data.times[1] - data.times[0] if len(data.times) > 1 else 0.01
        cc_all, cc_mean, lags = self._compute_cross_correlation(data, bin_size)
        
        self._fit_params = {
            'bin_size': bin_size,
            'cross_corr_mean': cc_mean,
            'lags': lags,
            'reduction_mean': np.mean(cc_mean, axis=0),
            'reduction_std': np.std(cc_mean, axis=0)
        }
        
        self.is_fitted = True
        logger.info(
            f"ConecTransformer fitted (lag={self.lag}, method={self.method}, "
            f"reduction={self.reduction})"
        )
        
        return self
    
    def transform(self, data: SpikeData) -> Feature:
        self._check_is_fitted()
        
        bin_size = self._fit_params['bin_size']
        cc_all, cc_mean, lags = self._compute_cross_correlation(data, bin_size)
        
        # cc_all shape: (n_trials, n_shifts, n_channels, n_channels)
        n_trials, n_shifts, n_channels, _ = cc_all.shape
        
        if self.reduction is False:
            # Return all shifts: (n_trials, n_channels*n_channels, n_shifts)
            # Reshape to (trials, channel_pairs, shifts)
            result = cc_all.transpose(0, 2, 3, 1)  # (n_trials, n_channels, n_channels, n_shifts)
            features_3d = result.reshape(n_trials, n_channels * n_channels, n_shifts)
            
            # Create feature names with lag information
            feature_names = [
                f"conec_ch{i}_ch{j}_lag{lag:.3f}s" 
                for i in data.channels 
                for j in data.channels
                for lag in lags
            ]
            
            metadata = {
                'transformer': 'ConecTransformer',
                'lag': self.lag,
                'method': self.method,
                'reduction': False,
                'bin_size': data.config.bin_size,
                'n_shifts': n_shifts,
                'n_channels': n_channels,
                'lags': lags,
                'shape_3d': (n_trials, n_channels * n_channels, n_shifts),
                'normalize': self.normalize,
                'per_condition': self.per_condition,
                'noise': self.noise,
            }
            
            logger.info(f"Extracted {features_3d.shape[1]} x {features_3d.shape[2]} connectivity features (no reduction)")
            
        elif self.reduction is True:
            # Return zero-lag only: (n_trials, n_channels*n_channels)
            zero_lag_idx = len(lags) // 2
            result = cc_all[:, zero_lag_idx, :, :]  # (n_trials, n_channels, n_channels)
            features_3d = result.reshape(n_trials, n_channels * n_channels)
            
            # Create feature names
            feature_names = [
                f"conec_ch{i}_ch{j}" 
                for i in data.channels 
                for j in data.channels
            ]
            
            metadata = {
                'transformer': 'ConecTransformer',
                'lag': self.lag,
                'method': self.method,
                'reduction': True,
                'bin_size': data.config.bin_size,
                'n_channels': n_channels,
                'shape_3d': (n_trials, n_channels * n_channels),
                'normalize': self.normalize,
                'per_condition': self.per_condition,
                'noise': self.noise,
            }
            
            logger.info(f"Extracted {features_3d.shape[1]} connectivity features (zero-lag)")
        
        else:
            raise ValueError(f"reduction must be True or False, got {self.reduction}")
        
        # Create Feature object
        feature = Feature(
            X=features_3d,
            y=data.orientations,
            sample_ids=data.trials,
            feature_names=feature_names,
            class_names=['0', '90'],
            metadata=metadata
        )
        
        return feature



class BinTransformer(BaseFeatureTransformer):
    """Extract temporal bin features."""
    
    def __init__(self, normalize: bool = False, noise: bool = False, per_condition: bool = False):
        super().__init__()
        self.normalize = normalize
        self.noise = noise
        self.per_condition = per_condition
    
    def fit(self, data: SpikeData) -> 'BinTransformer':
        """Fit the transformer by computing normalization parameters."""
        
        # Always initialize _fit_params, even if we don't need normalization
        spike_binned = data.spike_binned
        params = {}
        
        if self.normalize:
            params['mean'] = np.mean(spike_binned, axis=(0, 2), keepdims=True)
            params['std'] = np.std(spike_binned, axis=(0, 2), keepdims=True)
        
        if self.noise:
            params['noise_mean'] = np.mean(spike_binned, axis=0, keepdims=True)
        
        # Always set the params, even if empty dict
        self._fit_params['global'] = params
        self.is_fitted = True
        
        logger.info(
            f"BinTransformer fitted (normalize={self.normalize}, "
            f"noise={self.noise}, per_condition={self.per_condition})"
        )
        
        return self
    
    def transform(self, data: SpikeData) -> Feature:
        """Transform spike data to temporal bin features."""
        self._check_is_fitted()
        
        spike_binned = data.spike_binned.copy()
        params = self._fit_params['global']
        
        # Apply normalization if fitted
        if self.normalize and 'mean' in params:
            spike_binned = (spike_binned - params['mean']) / (params['std'] + 1e-8)
        
        # Apply noise subtraction if fitted
        if self.noise and 'noise_mean' in params:
            spike_binned = spike_binned - params['noise_mean']
        
        # Flatten
        n_trials = spike_binned.shape[0]
        features_flat = spike_binned.reshape(n_trials, -1)
        
        # Create feature names
        feature_names = [
            f"bin_ch{ch}_t{int(t*1000)}ms" 
            for ch in data.channels 
            for t in data.times
        ]
        
        # Create Feature object
        feature = Feature(
            X=features_flat,
            y=data.orientations,
            sample_ids=data.trials,
            feature_names=feature_names,
            class_names=['0', '90'],
            metadata={
                'transformer': 'BinTransformer',
                'normalize': self.normalize,
                'noise': self.noise,
                'per_condition': self.per_condition,
                'n_channels': data.n_channels,
                'n_bins': data.n_bins,
                'bin_size': data.config.bin_size
            }
        )
        
        logger.info(f"Extracted {features_flat.shape[1]} temporal features")
        return feature





class SubspaceRemovalTransformer:
    """
    Subspace-based removal of orientation-related variance via orthogonal-complement projection.

    Fit:
      - Builds an orientation subspace from training data by stacking orientation-conditioned
        mean firing patterns (flattened channel×timebin), optionally centered by the grand mean.
      - Uses SVD to obtain an orthonormal basis of the orientation subspace and selects K
        components either explicitly or by explained-variance threshold.

    Transform:
      - Applies the same fixed projection to any SpikeData without flattening the public API:
        internally flattens per-trial vectors, projects, then reshapes back to (channels, time).
      - Preserves grand mean by default (center trials by grand mean, project, add mean back).

    Output:
      - Returns a new SpikeData with spike_binned cleaned plus a diagnostics dict rich enough
        for later plotting (basis vectors, singular values/ratios, per-component scores and
        correlations with orientation, variance removed globally and per-channel, checks).
    """
    def __init__(
        self,
        center_by_grand_mean: bool = True,
        n_components: Optional[int] = None,
        evr_threshold: Optional[float] = 0.95,
        min_components: int = 1,
        max_components: Optional[int] = None,
        compute_transform_diagnostics: bool = True,
        store_projection_matrix: bool = False,  # if False, store basis and project as X - (XU)U^T
        random_state: Optional[int] = None,
    ):
        super().__init__()
        """
        Args:
            center_by_grand_mean: Center orientation means by grand mean to preserve overall mean on projection.
            n_components: Explicit number of subspace components to remove; overrides evr_threshold if set.
            evr_threshold: Explained-variance ratio threshold to choose K from SVD (sum of top-K >= threshold).
            min_components: Minimum K to remove.
            max_components: Optional cap on K.
            compute_transform_diagnostics: If True, compute diagnostics also on each transform() call.
            store_projection_matrix: If True, store dense P_perp; else store orthonormal basis U and use X - (XU)U^T.
            random_state: Unused placeholder (kept for API symmetry/logging).
        """
        self.center_by_grand_mean = center_by_grand_mean
        self.n_components = n_components
        self.evr_threshold = evr_threshold
        self.min_components = min_components
        self.max_components = max_components
        self.compute_transform_diagnostics = compute_transform_diagnostics
        self.store_projection_matrix = store_projection_matrix
        self.random_state = random_state

        # Fitted attributes
        self.is_fitted_: bool = False
        self.feature_dim_: Optional[int] = None
        self.grand_mean_: Optional[np.ndarray] = None  # (n_features,)
        self.basis_: Optional[np.ndarray] = None       # (n_features, K) orthonormal
        self.P_perp_: Optional[np.ndarray] = None      # (n_features, n_features) if stored
        self.singular_values_: Optional[np.ndarray] = None
        self.evr_: Optional[np.ndarray] = None
        self.selected_components_: Optional[int] = None
        self.orientation_means_: Optional[Dict[int, np.ndarray]] = None  # raw means per orientation

        # Diagnostics aggregated at fit
        self.fit_diagnostics_: Dict[str, Any] = {}

    @staticmethod
    def _flatten_trials(spike_binned: np.ndarray) -> np.ndarray:
        # (n_trials, n_channels, n_bins) -> (n_trials, n_features)
        n_trials = spike_binned.shape[0]
        return spike_binned.reshape(n_trials, -1)

    @staticmethod
    def _unflatten_trials(X: np.ndarray, shape_ref: Tuple[int, int, int]) -> np.ndarray:
        # (n_trials, n_features) -> (n_trials, n_channels, n_bins)
        n_trials = X.shape[0]
        _, n_channels, n_bins = shape_ref
        return X.reshape(n_trials, n_channels, n_bins)

    def _compute_orientation_means(self, data: SpikeData) -> Dict[int, np.ndarray]:
        X = self._flatten_trials(data.spike_binned)  # (T, F)
        means = {}
        for ori in np.unique(data.orientations):
            mask = (data.orientations == ori)
            if np.sum(mask) < 1:
                raise ValueError(f"No trials found for orientation {ori}")
            means[int(ori)] = X[mask].mean(axis=0)
        return means

    def _svd_basis(self, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # M shape (n_orientations, F); use economy SVD; we want right singular vectors (feature space)
        # M = U_svd S Vt  => V (columns) are basis in feature space
        U_svd, S, Vt = np.linalg.svd(M, full_matrices=False)
        V = Vt.T  # (F, r)
        return S, Vt, V

    def _select_k(self, S: np.ndarray) -> int:
        # Explained-variance ratios from singular values
        power = S**2
        evr = power / np.sum(power) if np.sum(power) > 0 else np.zeros_like(power)
        self.singular_values_ = S
        self.evr_ = evr
        if self.n_components is not None:
            k = int(self.n_components)
        else:
            # pick smallest k with cumulative EVR >= threshold
            cum = np.cumsum(evr)
            k = int(np.searchsorted(cum, self.evr_threshold, side="left") + 1) if self.evr_threshold is not None else len(S)
            k = max(k, self.min_components)
        if self.max_components is not None:
            k = min(k, self.max_components)
        k = max(1, min(k, len(S)))
        return k

    def fit(self, data: SpikeData) -> "SubspaceRemovalTransformer":
        """
        Fit the orientation subspace on training data ONLY.

        Steps:
          1) Compute per-orientation mean patterns over trials, flattened to (F,).
          2) Optionally center these means by the grand mean (to preserve overall mean).
          3) SVD of the stacked matrix to get an orthonormal basis in feature space.
          4) Select K components and (optionally) form the perpendicular projector.
        """
        # Flattened dimensionality
        X = self._flatten_trials(data.spike_binned)  # (T, F)
        T, F = X.shape
        self.feature_dim_ = F

        # Orientation means (on training)
        ori_means = self._compute_orientation_means(data)  # dict[int] -> (F,)
        self.orientation_means_ = {k: v.copy() for k, v in ori_means.items()}

        # Grand mean for preservation
        grand_mean = X.mean(axis=0)
        self.grand_mean_ = grand_mean.copy()

        # Stack orientation means to matrix (n_orientations, F)
        M = np.vstack([ori_means[k] for k in sorted(ori_means.keys())])  # (O, F)

        # Center by grand mean to isolate deviation subspace (preserves global mean on projection)
        if self.center_by_grand_mean:
            M_centered = M - grand_mean[None, :]
        else:
            M_centered = M

        # SVD and basis selection
        S, Vt, V = self._svd_basis(M_centered)  # V: (F, r)
        k = self._select_k(S)
        U_basis = V[:, :k]  # (F, k); orthonormal when from SVD

        self.basis_ = U_basis
        self.selected_components_ = k

        # Optionally build dense projector P_perp = I - UU^T (since U is orthonormal)
        if self.store_projection_matrix:
            I = np.eye(F, dtype=np.float32)
            self.P_perp_ = (I - U_basis @ U_basis.T).astype(np.float32)
        else:
            self.P_perp_ = None

        # Fit diagnostics computed on training data
        fit_scores = (X - grand_mean) @ U_basis  # (T, k) if centered projection semantics
        # Correlation with orientation (numeric) per component
        # Use raw degrees as numeric labels
        y_num = data.orientations.astype(float)
        comp_corr = []
        for j in range(k):
            sj = fit_scores[:, j]
            # Pearson r (robust to small variance guard)
            num = np.corrcoef(sj, y_num)[0, 1] if np.std(sj) > 0 and np.std(y_num) > 0 else 0.0
            comp_corr.append(float(num))

        # Variance removed (project training and compare)
        Xc = X - grand_mean if self.center_by_grand_mean else X
        Xc_clean = self._apply_projection_matrix(Xc, use_centered=True)
        # Add grand mean back for reporting parity
        X_train_clean = Xc_clean + (grand_mean if self.center_by_grand_mean else 0.0)

        total_var_before = float(np.var(X, axis=0, ddof=0).sum())
        total_var_after = float(np.var(X_train_clean, axis=0, ddof=0).sum())
        frac_removed = float((total_var_before - total_var_after) / (total_var_before + 1e-12))

        # Per-channel variance removed
        n_channels, n_bins = data.spike_binned.shape[1], data.spike_binned.shape[2]
        before_3d = self._unflatten_trials(X, data.spike_binned.shape)
        after_3d = self._unflatten_trials(X_train_clean, data.spike_binned.shape)
        var_removed_per_channel = []
        for ch in range(n_channels):
            vb = float(np.var(before_3d[:, ch, :]))
            va = float(np.var(after_3d[:, ch, :]))
            var_removed_per_channel.append(float((vb - va) / (vb + 1e-12)))

        self.fit_diagnostics_ = {
            "n_features": int(F),
            "n_trials_fit": int(T),
            "selected_components": int(k),
            "singular_values": self.singular_values_.tolist() if self.singular_values_ is not None else [],
            "explained_variance_ratio": self.evr_.tolist() if self.evr_ is not None else [],
            "component_corr_with_orientation": comp_corr,
            "total_variance_before": total_var_before,
            "total_variance_after": total_var_after,
            "fraction_variance_removed": frac_removed,
            "variance_removed_per_channel": var_removed_per_channel,
            "orientation_means_keys": [int(k) for k in sorted(ori_means.keys())],
            "center_by_grand_mean": bool(self.center_by_grand_mean),
            "projection_stored_dense": bool(self.P_perp_ is not None),
            "sanity": {
                "idempotent_if_dense": (bool(self.P_perp_ is not None) and
                                        np.allclose(self.P_perp_ @ self.P_perp_, self.P_perp_, atol=1e-6)),
                "trace_if_dense": (float(np.trace(self.P_perp_)) if self.P_perp_ is not None else None),
            },
        }

        self.is_fitted_ = True
        logger.info(
            f"SubspaceRemovalTransformer fitted with K={k}, EVR cum={np.sum(self.evr_[:k]) if self.evr_ is not None else 'NA'}."
        )
        return self

    def _apply_projection_matrix(self, X: np.ndarray, use_centered: bool) -> np.ndarray:
        """
        Project onto orthogonal complement using either stored dense projector or basis.
        If use_centered is True, assumes X has already been centered by grand mean when appropriate.
        """
        if self.P_perp_ is not None:
            return X @ self.P_perp_.T
        else:
            # X - (XU)U^T (U is orthonormal)
            U = self.basis_
            return X - (X @ U) @ U.T

    def transform(self, data: SpikeData) -> Tuple[SpikeData, Dict[str, Any]]:
        """
        Apply the learned projection to remove orientation subspace from provided data.

        Returns:
            cleaned_data: SpikeData with spike_binned projected to the orthogonal complement.
            diagnostics: dict with transform-time metrics for this dataset.
        """
        if not self.is_fitted_:
            raise RuntimeError("SubspaceRemovalTransformer must be fitted before transform().")

        # Prepare input X
        X_in = self._flatten_trials(data.spike_binned)  # (T2, F)
        T2, F2 = X_in.shape
        if F2 != self.feature_dim_:
            raise ValueError(f"Feature dimension mismatch: fitted {self.feature_dim_}, got {F2}")

        # Center trials consistently if requested
        if self.center_by_grand_mean and self.grand_mean_ is not None:
            Xc = X_in - self.grand_mean_[None, :]
        else:
            Xc = X_in

        # Project
        Xc_clean = self._apply_projection_matrix(Xc, use_centered=self.center_by_grand_mean)

        # Add grand mean back if centered
        X_out = Xc_clean + (self.grand_mean_[None, :] if self.center_by_grand_mean else 0.0)

        # Reshape and package SpikeData (deep copy structure, replace spike_binned)
        cleaned = deepcopy(data)
        cleaned.spike_binned = self._unflatten_trials(X_out, data.spike_binned.shape)

        # Diagnostics at transform-time (optional)
        diagnostics = {
            "n_trials_transform": int(T2),
            "used_same_projection": True,
            "selected_components": int(self.selected_components_ or 0),
        }

        if self.compute_transform_diagnostics:
            # Variance removal on this dataset
            total_var_before = float(np.var(X_in, axis=0, ddof=0).sum())
            total_var_after = float(np.var(X_out, axis=0, ddof=0).sum())
            frac_removed = float((total_var_before - total_var_after) / (total_var_before + 1e-12))

            # Per-channel variance removed
            before_3d = self._unflatten_trials(X_in, data.spike_binned.shape)
            after_3d = self._unflatten_trials(X_out, data.spike_binned.shape)
            n_channels = data.spike_binned.shape[1]
            var_removed_per_channel = []
            for ch in range(n_channels):
                vb = float(np.var(before_3d[:, ch, :]))
                va = float(np.var(after_3d[:, ch, :]))
                var_removed_per_channel.append(float((vb - va) / (vb + 1e-12)))

            # Per-component scores and corr with this dataset's orientations (if available)
            comp_scores = None
            comp_corr = None
            if self.basis_ is not None:
                U = self.basis_
                comp_scores = (X_in - (self.grand_mean_ if self.center_by_grand_mean else 0.0)) @ U
                y_num = data.orientations.astype(float)
                comp_corr = []
                for j in range(U.shape[1]):
                    sj = comp_scores[:, j]
                    r = np.corrcoef(sj, y_num)[0, 1] if np.std(sj) > 0 and np.std(y_num) > 0 else 0.0
                    comp_corr.append(float(r))

            diagnostics.update({
                "total_variance_before": total_var_before,
                "total_variance_after": total_var_after,
                "fraction_variance_removed": frac_removed,
                "variance_removed_per_channel": var_removed_per_channel,
                "component_scores_shape": (None if comp_scores is None else list(comp_scores.shape)),
                "component_corr_with_orientation": comp_corr,
            })

        return cleaned, diagnostics

    def get_fit_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostics computed during fit()."""
        return deepcopy(self.fit_diagnostics_)


class SequentialTransformer:
    def __init__(
        self,
        subspace_remover,
        feature_transformer,
        store_intermediate: bool = False
    ):
        super().__init__()

        self.subspace_remover = subspace_remover
        self.feature_transformer = feature_transformer
        self.store_intermediate = store_intermediate

        self.is_fitted_ = False
        self.intermediate_data_train_ = None

    def fit(self, data: SpikeData) -> "SequentialTransformer":

        logger.info("SequentialTransformer: Starting fit on training data")

        logger.info("Step 1/2: Fitting SubspaceRemovalTransformer...")
        self.subspace_remover.fit(data)

        logger.info("Step 1/2: Transforming training data with fitted subspace remover...")
        data_cleaned, _ = self.subspace_remover.transform(data)

        if self.store_intermediate:
            self.intermediate_data_train_ = deepcopy(data_cleaned)

        logger.info("Step 2/2: Fitting feature transformer on cleaned data...")
        self.feature_transformer.fit(data_cleaned)

        self.is_fitted_ = True
        logger.info("SequentialTransformer: Fit complete (no data leakage)")

        return self

    def transform(self, data: SpikeData) -> Feature:

        if not self.is_fitted_:
            raise RuntimeError("SequentialTransformer must be fitted before transform()")

        logger.debug("SequentialTransformer: Transforming new data")

        data_cleaned, _ = self.subspace_remover.transform(data)

        features = self.feature_transformer.transform(data_cleaned)

        return features

    def fit_transform(self, data: SpikeData) -> Feature:

        return self.fit(data).transform(data)

    def get_subspace_diagnostics(self) -> Dict[str, Any]:

        if not self.is_fitted_:
            raise RuntimeError("Transformer not fitted yet")

        if hasattr(self.subspace_remover, 'get_fit_diagnostics'):
            return self.subspace_remover.get_fit_diagnostics()
        else:
            return {}

    def get_intermediate_data(self) -> Optional[SpikeData]:

        if not self.store_intermediate:
            raise ValueError("store_intermediate was set to False during initialization")

        return self.intermediate_data_train_