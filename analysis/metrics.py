"""
Performance metrics and significance testing for classification.

This module provides custom metrics and statistical tests for
evaluating classification performance.
"""

import numpy as np
from scipy import stats
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    Custom performance metrics for neural data classification.
    """
    
    @staticmethod
    def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute balanced accuracy."""
        from sklearn.metrics import balanced_accuracy_score
        return float(balanced_accuracy_score(y_true, y_pred))
    
    @staticmethod
    def matthews_corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Matthews correlation coefficient."""
        from sklearn.metrics import matthews_corrcoef
        return float(matthews_corrcoef(y_true, y_pred))
    
    @staticmethod
    def classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Generate classification report as dictionary."""
        from sklearn.metrics import classification_report
        return classification_report(y_true, y_pred, output_dict=True)


class SignificanceTest:
    """
    Statistical significance tests for classification performance.
    """
    
    @staticmethod
    def binomial_test(
        accuracy: float,
        n_samples: int,
        chance_level: float = 0.5,
        alternative: str = 'greater'
    ) -> float:
        """
        Test if accuracy is significantly above chance using binomial test.
        
        Args:
            accuracy: Observed accuracy
            n_samples: Number of samples
            chance_level: Chance performance level
            alternative: 'greater', 'less', or 'two-sided'
        
        Returns:
            P-value
        """
        n_correct = int(accuracy * n_samples)
        p_value = stats.binom_test(
            n_correct,
            n_samples,
            chance_level,
            alternative=alternative
        )
        
        return float(p_value)
    
    @staticmethod
    def mcnemar_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray) -> Tuple[float, float]:
        """
        McNemar's test for comparing two classifiers.
        
        Args:
            y_true: True labels
            y_pred1: Predictions from classifier 1
            y_pred2: Predictions from classifier 2
        
        Returns:
            Tuple of (statistic, p_value)
        """
        # Create contingency table
        n_01 = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))
        n_10 = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
        
        # McNemar statistic
        statistic = (abs(n_01 - n_10) - 1) ** 2 / (n_01 + n_10 + 1e-10)
        p_value = 1 - stats.chi2.cdf(statistic, df=1)
        
        return float(statistic), float(p_value)
