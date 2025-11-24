"""
Statistical analysis functionality for neural data.

This module provides comprehensive statistical testing including t-tests,
permutation tests, effect sizes, and multiple comparison corrections.
"""

import numpy as np
from scipy import stats
from typing import Optional, Tuple, List, Dict, Literal, Callable
from dataclasses import dataclass
import logging

import sys
sys.path.append('/content')

from data.structures import SpikeData, Feature, StatisticalResult

logger = logging.getLogger(__name__)


@dataclass
class TTestResult:
    """Results from t-test."""
    statistic: float
    p_value: float
    effect_size: float
    mean_group1: float
    mean_group2: float
    std_group1: float
    std_group2: float
    n_group1: int
    n_group2: int
    test_type: str


class FDRCorrection:
    """
    False Discovery Rate correction for multiple comparisons.
    
    Implements Benjamini-Hochberg procedure for controlling FDR.
    """
    
    @staticmethod
    def benjamini_hochberg(
        p_values: np.ndarray,
        alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Apply Benjamini-Hochberg FDR correction.
        
        Args:
            p_values: Array of p-values
            alpha: Desired FDR level
        
        Returns:
            Tuple of (corrected_p_values, rejected, threshold)
        """
        p_values = np.asarray(p_values)
        n = len(p_values)
        
        # Sort p-values and keep track of original indices
        sort_idx = np.argsort(p_values)
        p_sorted = p_values[sort_idx]
        
        # Compute thresholds
        thresholds = (np.arange(1, n + 1) / n) * alpha
        
        # Find largest i where p(i) <= (i/m)*alpha
        comparison = p_sorted <= thresholds
        
        if np.any(comparison):
            max_idx = np.where(comparison)[0][-1]
            threshold = thresholds[max_idx]
        else:
            threshold = 0.0
        
        # Determine which are rejected
        rejected = p_values <= threshold
        
        # Compute adjusted p-values
        adjusted_p = np.minimum(1, p_sorted * n / np.arange(1, n + 1))
        
        # Ensure monotonicity
        for i in range(n - 2, -1, -1):
            adjusted_p[i] = min(adjusted_p[i], adjusted_p[i + 1])
        
        # Unsort to match original order
        corrected_p = np.empty_like(adjusted_p)
        corrected_p[sort_idx] = adjusted_p
        
        logger.info(f"FDR correction: {np.sum(rejected)}/{n} tests significant at α={alpha}")
        
        return corrected_p, rejected, threshold
    
    @staticmethod
    def bonferroni(
        p_values: np.ndarray,
        alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Bonferroni correction (more conservative).
        
        Args:
            p_values: Array of p-values
            alpha: Desired significance level
        
        Returns:
            Tuple of (corrected_p_values, rejected)
        """
        p_values = np.asarray(p_values)
        n = len(p_values)
        
        corrected_p = np.minimum(p_values * n, 1.0)
        rejected = corrected_p <= alpha
        
        logger.info(f"Bonferroni correction: {np.sum(rejected)}/{n} tests significant at α={alpha}")
        
        return corrected_p, rejected


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis for neural data.
    
    Provides t-tests, permutation tests, effect sizes, and multiple
    comparison corrections.
    """
    
    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize statistical analyzer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
    @staticmethod
    def cohens_d(
        group1: np.ndarray,
        group2: np.ndarray,
        pooled: bool = True
    ) -> float:
        """
        Compute Cohen's d effect size.
        
        Args:
            group1: First group data
            group2: Second group data
            pooled: Use pooled standard deviation
        
        Returns:
            Cohen's d effect size
        """
        mean1, mean2 = np.mean(group1), np.mean(group2)
        
        if pooled:
            n1, n2 = len(group1), len(group2)
            var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            d = (mean1 - mean2) / pooled_std
        else:
            std1 = np.std(group1, ddof=1)
            d = (mean1 - mean2) / std1
        
        return d
    
    def ttest_independent(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        equal_var: bool = True
    ) -> TTestResult:
        """
        Perform independent samples t-test.
        
        Args:
            group1: First group data
            group2: Second group data
            equal_var: Assume equal variances (True = Student's, False = Welch's)
        
        Returns:
            TTestResult object
        """
        statistic, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)
        effect_size = self.cohens_d(group1, group2, pooled=equal_var)
        
        result = TTestResult(
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=float(effect_size),
            mean_group1=float(np.mean(group1)),
            mean_group2=float(np.mean(group2)),
            std_group1=float(np.std(group1, ddof=1)),
            std_group2=float(np.std(group2, ddof=1)),
            n_group1=len(group1),
            n_group2=len(group2),
            test_type='independent' + ('_welch' if not equal_var else '')
        )
        
        return result
    
    def ttest_paired(
        self,
        group1: np.ndarray,
        group2: np.ndarray
    ) -> TTestResult:
        """
        Perform paired samples t-test.
        
        Args:
            group1: First group data
            group2: Second group data (must be same length)
        
        Returns:
            TTestResult object
        """
        if len(group1) != len(group2):
            raise ValueError("Paired t-test requires equal sample sizes")
        
        statistic, p_value = stats.ttest_rel(group1, group2)
        effect_size = self.cohens_d(group1, group2, pooled=False)
        
        result = TTestResult(
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=float(effect_size),
            mean_group1=float(np.mean(group1)),
            mean_group2=float(np.mean(group2)),
            std_group1=float(np.std(group1, ddof=1)),
            std_group2=float(np.std(group2, ddof=1)),
            n_group1=len(group1),
            n_group2=len(group2),
            test_type='paired'
        )
        
        return result
    
    def ttest_features(
        self,
        feature: Feature,
        test_type: Literal['independent', 'paired'] = 'independent',
        equal_var: bool = True,
        fdr_correction: bool = True,
        alpha: float = 0.05
    ) -> StatisticalResult:
        """
        Perform t-tests on all features comparing two classes.
        
        Args:
            feature: Feature object with two classes
            test_type: 'independent' or 'paired'
            equal_var: Assume equal variances
            fdr_correction: Apply FDR correction
            alpha: Significance level
        
        Returns:
            StatisticalResult object
        """
        unique_labels = np.unique(feature.y)
        if len(unique_labels) != 2:
            raise ValueError("t-test requires exactly 2 classes")
        
        mask1 = feature.y == unique_labels[0]
        mask2 = feature.y == unique_labels[1]
        
        n_features = feature.n_features
        p_values = np.zeros(n_features)
        effect_sizes = np.zeros(n_features)
        
        for i in range(n_features):
            data1 = feature.X[mask1, i]
            data2 = feature.X[mask2, i]
            
            if test_type == 'independent':
                result = self.ttest_independent(data1, data2, equal_var=equal_var)
            else:
                result = self.ttest_paired(data1, data2)
            
            p_values[i] = result.p_value
            effect_sizes[i] = result.effect_size
        
        # Apply FDR correction if requested
        if fdr_correction:
            corrected_p, significant, threshold = FDRCorrection.benjamini_hochberg(
                p_values, alpha=alpha
            )
            correction_method = 'benjamini_hochberg'
        else:
            corrected_p = p_values
            significant = p_values < alpha
            threshold = alpha
            correction_method = 'none'
        
        result = StatisticalResult(
            p_values=p_values,
            effect_sizes=effect_sizes,
            significant_mask=significant,
            method=f"t-test_{test_type}",
            correction=correction_method,
            metadata={
                'alpha': alpha,
                'fdr_threshold': threshold,
                'n_significant': int(np.sum(significant)),
                'n_tests': n_features,
                'corrected_p_values': corrected_p,
                'feature_names': feature.feature_names
            }
        )
        
        logger.info(
            f"T-test complete: {np.sum(significant)}/{n_features} features significant "
            f"(α={alpha}, FDR={fdr_correction})"
        )
        
        return result
    
    def permutation_test(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        statistic_func: Callable = np.mean,
        n_permutations: int = 10000,
        alternative: Literal['two-sided', 'greater', 'less'] = 'two-sided'
    ) -> Tuple[float, float, np.ndarray]:
        """
        Perform permutation test.
        
        Args:
            group1: First group data
            group2: Second group data
            statistic_func: Function to compute test statistic
            n_permutations: Number of permutations
            alternative: Test alternative
        
        Returns:
            Tuple of (observed_statistic, p_value, null_distribution)
        """
        # Observed statistic
        observed = statistic_func(group1) - statistic_func(group2)
        
        # Combined data
        combined = np.concatenate([group1, group2])
        n1 = len(group1)
        n_total = len(combined)
        
        # Permutation distribution
        null_dist = np.zeros(n_permutations)
        
        for i in range(n_permutations):
            # Shuffle and split
            shuffled = np.random.permutation(combined)
            perm_group1 = shuffled[:n1]
            perm_group2 = shuffled[n1:]
            
            # Compute statistic
            null_dist[i] = statistic_func(perm_group1) - statistic_func(perm_group2)
        
        # Compute p-value
        if alternative == 'two-sided':
            p_value = np.mean(np.abs(null_dist) >= np.abs(observed))
        elif alternative == 'greater':
            p_value = np.mean(null_dist >= observed)
        else:  # less
            p_value = np.mean(null_dist <= observed)
        
        logger.info(f"Permutation test: observed={observed:.4f}, p={p_value:.4f}")
        
        return float(observed), float(p_value), null_dist
    
    def bootstrap_confidence_interval(
        self,
        data: np.ndarray,
        statistic_func: Callable = np.mean,
        n_bootstrap: int = 10000,
        confidence_level: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Compute bootstrap confidence interval.
        
        Args:
            data: Input data
            statistic_func: Statistic to compute
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (e.g., 0.95 for 95%)
        
        Returns:
            Tuple of (estimate, lower_bound, upper_bound)
        """
        n = len(data)
        bootstrap_stats = np.zeros(n_bootstrap)
        
        for i in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats[i] = statistic_func(sample)
        
        # Compute percentiles
        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        estimate = statistic_func(data)
        
        logger.info(
            f"Bootstrap CI ({confidence_level*100:.0f}%): "
            f"{estimate:.4f} [{lower:.4f}, {upper:.4f}]"
        )
        
        return float(estimate), float(lower), float(upper)
    
    def anova_oneway(
        self,
        *groups: np.ndarray
    ) -> Tuple[float, float]:
        """
        Perform one-way ANOVA.
        
        Args:
            *groups: Variable number of group arrays
        
        Returns:
            Tuple of (F_statistic, p_value)
        """
        f_stat, p_value = stats.f_oneway(*groups)
        
        logger.info(f"One-way ANOVA: F={f_stat:.4f}, p={p_value:.4f}")
        
        return float(f_stat), float(p_value)
    
    def compare_firing_rates(
        self,
        data: SpikeData,
        channel_idx: int,
        time_window: Optional[Tuple[float, float]] = None,
        test_type: str = 'independent',
        fdr_correction: bool = True
    ) -> TTestResult:
        """
        Compare firing rates between orientations for a single channel.
        
        Args:
            data: Spike data
            channel_idx: Channel index to test
            time_window: Optional (start, end) time window
            test_type: 'independent' or 'paired'
            fdr_correction: Apply FDR correction
        
        Returns:
            TTestResult
        """
        # Extract data
        if time_window is not None:
            time_mask = (data.times >= time_window[0]) & (data.times <= time_window[1])
            spike_data = data.spike_binned[:, channel_idx, time_mask]
        else:
            spike_data = data.spike_binned[:, channel_idx, :]
        
        # Compute mean firing rates per trial
        rates = np.mean(spike_data, axis=1)
        
        # Split by orientation
        orientations = np.unique(data.orientations)
        if len(orientations) != 2:
            raise ValueError("Requires exactly 2 orientations")
        
        rates1 = rates[data.orientations == orientations[0]]
        rates2 = rates[data.orientations == orientations[1]]
        
        # Perform t-test
        if test_type == 'independent':
            result = self.ttest_independent(rates1, rates2)
        else:
            result = self.ttest_paired(rates1, rates2)
        
        return result
