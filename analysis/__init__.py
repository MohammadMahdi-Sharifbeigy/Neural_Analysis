"""
Analysis module for neural spike data.

This module provides statistical analysis, machine learning pipelines,
and performance metrics for neural data analysis.
"""

from .statistics import StatisticalAnalyzer, FDRCorrection
from .classification import (
    AdvancedClassificationPipeline,
    classify_with_transformer,
    ComprehensiveClassificationResult,
    FeatureImportanceResult,
    FoldResult,
    ShapLikeImportance,
)
from .metrics import PerformanceMetrics, SignificanceTest

__all__ = [
    'StatisticalAnalyzer',
    'FDRCorrection',
    'AdvancedClassificationPipeline',
    'classify_with_transformer',
    'ComprehensiveClassificationResult',
    'FeatureImportanceResult',
    'FoldResult',
    'ShapLikeImportance',
    'PerformanceMetrics',
    'SignificanceTest',
]

__version__ = '1.1.0'
