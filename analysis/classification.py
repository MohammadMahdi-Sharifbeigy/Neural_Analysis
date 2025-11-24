"""
Advanced classification pipeline for neural data analysis with transformer integration.

This module provides a comprehensive machine learning pipeline that:
- Accepts SpikeData and any BaseFeatureTransformer
- Performs transformation inside each cross-validation fold (no data leakage)
- Computes multiple feature importance metrics
- Supports optional scaling after transformation
- Returns comprehensive results with full metadata
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, 
    confusion_matrix, roc_auc_score, f1_score,
    precision_score, recall_score, matthews_corrcoef
)
from sklearn.inspection import permutation_importance
from typing import Optional, List, Dict, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
from copy import deepcopy
import warnings

import sys
sys.path.append('/content')

from data.structures import SpikeData, Feature, ClassificationResult
from features import BaseFeatureTransformer

logger = logging.getLogger(__name__)


@dataclass
class FeatureImportanceResult:
    """Container for feature importance results."""
    permutation_importance_mean: np.ndarray
    permutation_importance_std: np.ndarray
    shap_like_importance_mean: Optional[np.ndarray] = None
    shap_like_importance_std: Optional[np.ndarray] = None
    builtin_importance: Optional[np.ndarray] = None
    feature_names: List[str] = field(default_factory=list)
    method_metadata: Dict = field(default_factory=dict)


@dataclass
class FoldResult:
    """Results from a single cross-validation fold."""
    fold_idx: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_score: float
    test_score: float
    train_balanced_acc: float
    test_balanced_acc: float
    confusion_matrix: np.ndarray
    predictions: np.ndarray
    true_labels: np.ndarray
    feature_importance: Optional[FeatureImportanceResult] = None
    estimator: Any = None
    scaler: Any = None
    transformer_state: Dict = field(default_factory=dict)


@dataclass
class ComprehensiveClassificationResult(ClassificationResult):
    """Extended classification result with comprehensive metrics and metadata."""
    # Additional metrics
    balanced_accuracies_train: Optional[np.ndarray] = None
    balanced_accuracies_test: Optional[np.ndarray] = None
    f1_scores: Optional[np.ndarray] = None
    precision_scores: Optional[np.ndarray] = None
    recall_scores: Optional[np.ndarray] = None
    roc_auc_scores: Optional[np.ndarray] = None
    mcc_scores: Optional[np.ndarray] = None
    
    # Feature importance
    feature_importance_folds: Optional[List[FeatureImportanceResult]] = None
    feature_importance_mean: Optional[FeatureImportanceResult] = None
    
    # Fold-level results
    fold_results: Optional[List[FoldResult]] = None
    
    # Confusion matrices
    confusion_matrices: Optional[List[np.ndarray]] = None
    confusion_matrix_mean: Optional[np.ndarray] = None
    
    # Additional attributes
    cv_scores_detailed: Optional[Dict] = None
    best_fold_idx: Optional[int] = None
    
    @property
    def mean_balanced_acc_test(self) -> float:
        if self.balanced_accuracies_test is not None:
            return float(np.mean(self.balanced_accuracies_test))
        return 0.0
    
    @property
    def std_balanced_acc_test(self) -> float:
        if self.balanced_accuracies_test is not None:
            return float(np.std(self.balanced_accuracies_test))
        return 0.0
    
    @property
    def mean_f1(self) -> float:
        if self.f1_scores is not None:
            return float(np.mean(self.f1_scores))
        return 0.0


class ShapLikeImportance:
    """
    Custom feature importance inspired by SHAP values.
    
    Computes feature importance by measuring the impact of removing
    each feature on model predictions across multiple perturbations.
    """
    
    @staticmethod
    def compute(
        estimator: Any,
        X: np.ndarray,
        y: np.ndarray,
        n_repeats: int = 10,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute SHAP-like feature importance.
        
        Method: For each feature, we replace its values with random samples
        from the feature's distribution and measure the drop in prediction
        accuracy. This is similar to permutation importance but uses
        sampling from the marginal distribution.
        
        Args:
            estimator: Trained estimator
            X: Feature matrix (n_samples, n_features)
            y: True labels
            n_repeats: Number of perturbation repeats
            random_state: Random seed
            
        Returns:
            Tuple of (importance_mean, importance_std)
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        n_samples, n_features = X.shape
        
        # Baseline score
        y_pred_baseline = estimator.predict(X)
        baseline_score = accuracy_score(y, y_pred_baseline)
        
        # Storage for importance values
        importances = np.zeros((n_features, n_repeats))
        
        for feature_idx in range(n_features):
            for repeat in range(n_repeats):
                # Create perturbed data
                X_perturbed = X.copy()
                
                # Sample from marginal distribution (with replacement)
                sampled_values = np.random.choice(
                    X[:, feature_idx], 
                    size=n_samples, 
                    replace=True
                )
                
                # Replace feature values
                X_perturbed[:, feature_idx] = sampled_values
                
                # Compute score with perturbed feature
                y_pred_perturbed = estimator.predict(X_perturbed)
                perturbed_score = accuracy_score(y, y_pred_perturbed)
                
                # Importance is the drop in performance
                importances[feature_idx, repeat] = baseline_score - perturbed_score
        
        # Compute mean and std across repeats
        importance_mean = np.mean(importances, axis=1)
        importance_std = np.std(importances, axis=1)
        
        return importance_mean, importance_std


class AdvancedClassificationPipeline:
    """
    Advanced classification pipeline with transformer integration.
    
    Key features:
    - Transforms data inside each CV fold (prevents leakage)
    - Supports any BaseFeatureTransformer
    - Optional scaling after transformation
    - Multiple feature importance methods
    - Comprehensive result tracking
    - Full metadata storage
    """
    
    # Default classifiers
    DEFAULT_CLASSIFIERS = {
        'SVM_linear': SVC(kernel='linear', C=1.0, random_state=42, probability=True),
        'SVM_rbf': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True),
        'LogisticRegression': LogisticRegression(max_iter=2000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'NaiveBayes': GaussianNB()
    }
    
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = 42,
        n_jobs: int = -1
    ):
        """
        Initialize advanced classification pipeline.
        
        Args:
            n_splits: Number of CV folds
            shuffle: Shuffle data before splitting
            random_state: Random seed
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )
        
        logger.info(
            f"Initialized AdvancedClassificationPipeline with {n_splits} folds"
        )
    
    def classify_with_transformer(
        self,
        data: SpikeData,
        transformer: BaseFeatureTransformer,
        estimator: Optional[Any] = None,
        use_scaler: bool = True,
        compute_feature_importance: bool = True,
        permutation_repeats: int = 10,
        shaplike_repeats: int = 10,
        return_estimators: bool = True
    ) -> ComprehensiveClassificationResult:
        """
        Perform classification with transformer applied inside each CV fold.
        
        This method ensures no data leakage by:
        1. Splitting data into train/test
        2. Fitting transformer on train data only
        3. Transforming both train and test
        4. Optionally scaling after transformation
        5. Training classifier
        6. Computing feature importance on test set
        
        Args:
            data: SpikeData object
            transformer: BaseFeatureTransformer (e.g., ConecTransformer, BinTransformer)
            estimator: Sklearn estimator (if None, uses SVM_linear)
            use_scaler: Apply StandardScaler after transformation
            compute_feature_importance: Compute feature importance metrics
            permutation_repeats: Number of permutation repeats
            shaplike_repeats: Number of SHAP-like repeats
            return_estimators: Store trained estimators
            
        Returns:
            ComprehensiveClassificationResult with all metrics and metadata
        """
        # Default estimator
        if estimator is None:
            estimator = self.DEFAULT_CLASSIFIERS['SVM_linear']
        
        # Get labels (we'll use these for CV splitting)
        y = data.orientations
        n_samples = len(y)
        
        # Storage for results
        fold_results = []
        train_accuracies = []
        test_accuracies = []
        train_balanced_accs = []
        test_balanced_accs = []
        f1_scores_list = []
        precision_scores_list = []
        recall_scores_list = []
        mcc_scores_list = []
        roc_auc_scores_list = []
        confusion_matrices = []
        feature_importance_folds = []
        
        logger.info(f"Starting {self.n_splits}-fold cross-validation with {transformer.__class__.__name__}")
        
        # Perform cross-validation
        for fold_idx, (train_idx, test_idx) in enumerate(self.cv.split(np.arange(n_samples), y)):
            logger.info(f"Processing fold {fold_idx + 1}/{self.n_splits}")
            
            # 1. Split SpikeData into train/test
            data_train = self._subset_spikedata(data, train_idx)
            data_test = self._subset_spikedata(data, test_idx)
            
            # 2. Clone transformer for this fold
            transformer_fold = deepcopy(transformer)
            
            # 3. Fit transformer on train data ONLY
            transformer_fold.fit(data_train)
            
            # 4. Transform train and test data
            feature_train = transformer_fold.transform(data_train)
            feature_test = transformer_fold.transform(data_test)
            
            X_train = feature_train.X
            y_train = feature_train.y
            X_test = feature_test.X
            y_test = feature_test.y
            
            # 5. Optional scaling (after transformation)
            scaler = None
            if use_scaler:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            
            # 6. Clone and train estimator
            estimator_fold = deepcopy(estimator)
            estimator_fold.fit(X_train, y_train)
            
            # 7. Predictions and metrics
            y_train_pred = estimator_fold.predict(X_train)
            y_test_pred = estimator_fold.predict(X_test)
            
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            train_balanced = balanced_accuracy_score(y_train, y_train_pred)
            test_balanced = balanced_accuracy_score(y_test, y_test_pred)
            
            # Additional metrics
            f1 = f1_score(y_test, y_test_pred, average='weighted')
            precision = precision_score(y_test, y_test_pred, average='weighted')
            recall = recall_score(y_test, y_test_pred, average='weighted')
            mcc = matthews_corrcoef(y_test, y_test_pred)
            
            # ROC AUC (only for binary classification with probability estimates)
            roc_auc = None
            if len(np.unique(y)) == 2 and hasattr(estimator_fold, 'predict_proba'):
                try:
                    y_test_proba = estimator_fold.predict_proba(X_test)[:, 1]
                    roc_auc = roc_auc_score(y_test, y_test_proba)
                except:
                    logger.warning("Could not compute ROC AUC")
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_test_pred)
            
            # 8. Feature importance
            feat_importance = None
            if compute_feature_importance:
                feat_importance = self._compute_feature_importance(
                    estimator_fold,
                    X_test,
                    y_test,
                    feature_test.feature_names,
                    permutation_repeats=permutation_repeats,
                    shaplike_repeats=shaplike_repeats
                )
                feature_importance_folds.append(feat_importance)
            
            # Store fold result
            fold_result = FoldResult(
                fold_idx=fold_idx,
                train_indices=train_idx,
                test_indices=test_idx,
                train_score=train_acc,
                test_score=test_acc,
                train_balanced_acc=train_balanced,
                test_balanced_acc=test_balanced,
                confusion_matrix=cm,
                predictions=y_test_pred,
                true_labels=y_test,
                feature_importance=feat_importance,
                estimator=estimator_fold if return_estimators else None,
                scaler=scaler if return_estimators else None,
                transformer_state={
                    'transformer_class': transformer_fold.__class__.__name__,
                    'transformer_params': transformer_fold.__dict__,
                    'n_features_out': feature_test.n_features
                }
            )
            
            fold_results.append(fold_result)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            train_balanced_accs.append(train_balanced)
            test_balanced_accs.append(test_balanced)
            f1_scores_list.append(f1)
            precision_scores_list.append(precision)
            recall_scores_list.append(recall)
            mcc_scores_list.append(mcc)
            if roc_auc is not None:
                roc_auc_scores_list.append(roc_auc)
            confusion_matrices.append(cm)
            
            logger.info(
                f"Fold {fold_idx + 1}: train_acc={train_acc:.4f}, "
                f"test_acc={test_acc:.4f}, test_balanced={test_balanced:.4f}"
            )
        
        # Aggregate feature importance across folds
        feature_importance_mean = self._aggregate_feature_importance(
            feature_importance_folds
        ) if compute_feature_importance else None
        
        # Find best fold
        best_fold_idx = int(np.argmax(test_accuracies))
        
        # Compute mean confusion matrix
        cm_mean = np.mean(confusion_matrices, axis=0)
        
        # Build comprehensive metadata
        metadata = {
            'transformer': {
                'class': transformer.__class__.__name__,
                'params': {k: v for k, v in transformer.__dict__.items() 
                          if not k.startswith('_')},
            },
            'estimator': {
                'class': estimator.__class__.__name__,
                'params': estimator.get_params() if hasattr(estimator, 'get_params') else {}
            },
            'cv': {
                'n_splits': self.n_splits,
                'shuffle': self.shuffle,
                'random_state': self.random_state
            },
            'preprocessing': {
                'use_scaler': use_scaler,
                'scaler_type': 'StandardScaler' if use_scaler else None
            },
            'data': {
                'n_samples': n_samples,
                'n_channels': data.n_channels,
                'n_bins': data.n_bins,
                'bin_size': data.config.bin_size,
                'trial_duration': data.config.trial_duration
            },
            'feature_importance': {
                'computed': compute_feature_importance,
                'permutation_repeats': permutation_repeats,
                'shaplike_repeats': shaplike_repeats
            }
        }
        
        # Create comprehensive result
        result = ComprehensiveClassificationResult(
            train_accuracies=np.array(train_accuracies),
            test_accuracies=np.array(test_accuracies),
            balanced_accuracies_train=np.array(train_balanced_accs),
            balanced_accuracies_test=np.array(test_balanced_accs),
            f1_scores=np.array(f1_scores_list) if f1_scores_list else None,
            precision_scores=np.array(precision_scores_list) if precision_scores_list else None,
            recall_scores=np.array(recall_scores_list) if recall_scores_list else None,
            mcc_scores=np.array(mcc_scores_list) if mcc_scores_list else None,
            roc_auc_scores=np.array(roc_auc_scores_list) if roc_auc_scores_list else None,
            feature_importance_folds=feature_importance_folds,
            feature_importance_mean=feature_importance_mean,
            fold_results=fold_results,
            confusion_matrices=confusion_matrices,
            confusion_matrix_mean=cm_mean,
            best_fold_idx=best_fold_idx,
            model_name=estimator.__class__.__name__,
            metadata=metadata
        )
        
        # Print summary
        self._print_summary(result)
        
        return result
    
    def compare_transformers(
        self,
        data: SpikeData,
        transformers: Dict[str, BaseFeatureTransformer],
        estimator: Optional[Any] = None,
        use_scaler: bool = True,
        compute_feature_importance: bool = False
    ) -> Dict[str, ComprehensiveClassificationResult]:
        """
        Compare multiple transformers with the same classifier.
        
        Args:
            data: SpikeData object
            transformers: Dict of {name: transformer}
            estimator: Sklearn estimator
            use_scaler: Apply scaling
            compute_feature_importance: Compute importance (slower)
            
        Returns:
            Dictionary of {transformer_name: result}
        """
        results = {}
        
        logger.info(f"Comparing {len(transformers)} transformers")
        
        for name, transformer in transformers.items():
            logger.info(f"\n{'='*70}")
            logger.info(f"Testing transformer: {name}")
            logger.info(f"{'='*70}")
            
            result = self.classify_with_transformer(
                data=data,
                transformer=transformer,
                estimator=estimator,
                use_scaler=use_scaler,
                compute_feature_importance=compute_feature_importance
            )
            
            results[name] = result
        
        # Print comparison summary
        self._print_comparison_summary(results)
        
        return results
    
    def compare_classifiers(
        self,
        data: SpikeData,
        transformer: BaseFeatureTransformer,
        classifiers: Optional[Dict[str, Any]] = None,
        use_scaler: bool = True,
        compute_feature_importance: bool = False
    ) -> Dict[str, ComprehensiveClassificationResult]:
        """
        Compare multiple classifiers with the same transformer.
        
        Args:
            data: SpikeData object
            transformer: BaseFeatureTransformer
            classifiers: Dict of {name: estimator} (uses defaults if None)
            use_scaler: Apply scaling
            compute_feature_importance: Compute importance (slower)
            
        Returns:
            Dictionary of {classifier_name: result}
        """
        if classifiers is None:
            classifiers = self.DEFAULT_CLASSIFIERS
        
        results = {}
        
        logger.info(f"Comparing {len(classifiers)} classifiers")
        
        for name, estimator in classifiers.items():
            logger.info(f"\n{'='*70}")
            logger.info(f"Testing classifier: {name}")
            logger.info(f"{'='*70}")
            
            result = self.classify_with_transformer(
                data=data,
                transformer=transformer,
                estimator=estimator,
                use_scaler=use_scaler,
                compute_feature_importance=compute_feature_importance
            )
            
            results[name] = result
        
        # Print comparison summary
        self._print_comparison_summary(results)
        
        return results
    
    def _subset_spikedata(self, data: SpikeData, indices: np.ndarray) -> SpikeData:
        """Create subset of SpikeData for given trial indices."""
        return SpikeData(
            trials=data.trials[indices],
            channels=data.channels,  # Keep all channels
            orientations=data.orientations[indices],
            spike_times=data.spike_times,  # Keep all (filtered during processing)
            spike_binned=data.spike_binned[indices],
            times=data.times,
            config=data.config
        )
    
    def _compute_feature_importance(
        self,
        estimator: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        permutation_repeats: int = 10,
        shaplike_repeats: int = 10
    ) -> FeatureImportanceResult:
        """
        Compute multiple types of feature importance.
        
        1. Permutation importance (sklearn)
        2. SHAP-like importance (custom)
        3. Built-in importance (if available)
        """
        # 1. Permutation importance (sklearn standard)
        perm_importance = permutation_importance(
            estimator,
            X,
            y,
            n_repeats=permutation_repeats,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        perm_mean = perm_importance.importances_mean
        perm_std = perm_importance.importances_std
        
        # 2. SHAP-like importance (custom method)
        shap_mean, shap_std = ShapLikeImportance.compute(
            estimator,
            X,
            y,
            n_repeats=shaplike_repeats,
            random_state=self.random_state
        )
        
        # 3. Built-in importance (if available)
        builtin_importance = None
        if hasattr(estimator, 'coef_'):
            builtin_importance = np.abs(estimator.coef_[0]) if estimator.coef_.ndim > 1 else np.abs(estimator.coef_)
        elif hasattr(estimator, 'feature_importances_'):
            builtin_importance = estimator.feature_importances_
        
        return FeatureImportanceResult(
            permutation_importance_mean=perm_mean,
            permutation_importance_std=perm_std,
            shap_like_importance_mean=shap_mean,
            shap_like_importance_std=shap_std,
            builtin_importance=builtin_importance,
            feature_names=feature_names,
            method_metadata={
                'permutation_repeats': permutation_repeats,
                'shaplike_repeats': shaplike_repeats,
                'has_builtin': builtin_importance is not None
            }
        )
    
    def _aggregate_feature_importance(
        self,
        fold_importances: List[FeatureImportanceResult]
    ) -> FeatureImportanceResult:
        """Aggregate feature importance across folds."""
        if not fold_importances:
            return None
        
        # Stack all permutation importances
        perm_means = np.stack([fi.permutation_importance_mean for fi in fold_importances])
        perm_stds = np.stack([fi.permutation_importance_std for fi in fold_importances])
        
        # Stack SHAP-like importances
        shap_means = np.stack([fi.shap_like_importance_mean for fi in fold_importances])
        shap_stds = np.stack([fi.shap_like_importance_std for fi in fold_importances])
        
        # Stack built-in if available
        builtin_list = [fi.builtin_importance for fi in fold_importances 
                       if fi.builtin_importance is not None]
        builtin_mean = np.mean(builtin_list, axis=0) if builtin_list else None
        
        # Aggregate
        return FeatureImportanceResult(
            permutation_importance_mean=np.mean(perm_means, axis=0),
            permutation_importance_std=np.std(perm_means, axis=0),
            shap_like_importance_mean=np.mean(shap_means, axis=0),
            shap_like_importance_std=np.std(shap_means, axis=0),
            builtin_importance=builtin_mean,
            feature_names=fold_importances[0].feature_names,
            method_metadata={
                'aggregated_over_folds': len(fold_importances),
                **fold_importances[0].method_metadata
            }
        )
    
    def _print_summary(self, result: ComprehensiveClassificationResult):
        """Print summary of classification results."""
        print("\n" + "="*80)
        print("CLASSIFICATION RESULTS SUMMARY")
        print("="*80)
        print(f"Model: {result.model_name}")
        print(f"Transformer: {result.metadata['transformer']['class']}")
        print("-"*80)
        print(f"Accuracy (Test):        {result.mean_test_acc:.4f} ± {np.std(result.test_accuracies):.4f}")
        print(f"Balanced Acc (Test):    {result.mean_balanced_acc_test:.4f} ± {result.std_balanced_acc_test:.4f}")
        if result.f1_scores is not None:
            print(f"F1 Score:               {result.mean_f1:.4f} ± {np.std(result.f1_scores):.4f}")
        if result.roc_auc_scores is not None and len(result.roc_auc_scores) > 0:
            print(f"ROC AUC:                {np.mean(result.roc_auc_scores):.4f} ± {np.std(result.roc_auc_scores):.4f}")
        print("-"*80)
        print(f"Best Fold:              {result.best_fold_idx + 1}")
        print(f"Best Test Accuracy:     {result.test_accuracies[result.best_fold_idx]:.4f}")
        print("="*80 + "\n")
    
    def _print_comparison_summary(self, results: Dict[str, ComprehensiveClassificationResult]):
        """Print comparison summary."""
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        print(f"{'Name':<25} {'Test Acc':<15} {'Balanced Acc':<15} {'F1 Score':<15}")
        print("-"*80)
        
        for name, result in results.items():
            test_acc_str = f"{result.mean_test_acc:.4f}±{np.std(result.test_accuracies):.4f}"
            bal_acc_str = f"{result.mean_balanced_acc_test:.4f}±{result.std_balanced_acc_test:.4f}"
            f1_str = f"{result.mean_f1:.4f}±{np.std(result.f1_scores):.4f}" if result.f1_scores is not None else "N/A"
            print(f"{name:<25} {test_acc_str:<15} {bal_acc_str:<15} {f1_str:<15}")
        
        print("="*80 + "\n")


def classify_with_transformer(
    data: SpikeData,
    transformer: BaseFeatureTransformer,
    estimator: Optional[Any] = None,
    n_splits: int = 5,
    use_scaler: bool = True,
    compute_feature_importance: bool = True,
    random_state: Optional[int] = 42,
    **kwargs
) -> ComprehensiveClassificationResult:
    """
    Convenience function for classification with transformer.
    
    This is the main entry point for users.
    
    Args:
        data: SpikeData object
        transformer: BaseFeatureTransformer instance
        estimator: Sklearn estimator (default: SVM linear)
        n_splits: Number of CV folds
        use_scaler: Apply StandardScaler after transformation
        compute_feature_importance: Compute feature importance
        random_state: Random seed
        **kwargs: Additional arguments for AdvancedClassificationPipeline
        
    Returns:
        ComprehensiveClassificationResult
        
    Example:
        >>> from features import ConecTransformer
        >>> from analysis import classify_with_transformer
        >>> 
        >>> transformer = ConecTransformer(lag=0.1, method='Pearson')
        >>> result = classify_with_transformer(
        ...     data_001_clipped,
        ...     transformer,
        ...     use_scaler=True,
        ...     compute_feature_importance=True
        ... )
        >>> print(f"Accuracy: {result.mean_test_acc:.4f}")
    """
    pipeline = AdvancedClassificationPipeline(
        n_splits=n_splits,
        random_state=random_state,
        **kwargs
    )
    
    return pipeline.classify_with_transformer(
        data=data,
        transformer=transformer,
        estimator=estimator,
        use_scaler=use_scaler,
        compute_feature_importance=compute_feature_importance
    )


