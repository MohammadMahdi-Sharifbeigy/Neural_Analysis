#%%
import re
import math
import random
import itertools

import numpy as np
import pandas as pd

from scipy.stats import ttest_ind, ttest_1samp

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap

from copy import deepcopy
from typing import List, Dict, Any, Union, Optional

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import DecisionBoundaryDisplay

#%%
from data import (
    Orientation,
    ExperimentConfig,
    SpikeData,
    Feature,
    PSTH,
    ConnectivityMatrix,
    StatisticalResult,
    ClassificationResult,
    DataLoader,
)

from data.preprocessor import (
    SpikeDataPreprocessor,
    DataQualityChecker,
    preprocess_pipeline,
)

from features import (
    BaseFeatureTransformer,
    FreqTransformer,
    ConecTransformer,
    BinTransformer,
    SubspaceRemovalTransformer,
    SequentialTransformer
)

from analysis import (
    StatisticalAnalyzer,
    FDRCorrection,
    AdvancedClassificationPipeline,
    classify_with_transformer,
    ComprehensiveClassificationResult,
    FeatureImportanceResult,
    FoldResult,
    ShapLikeImportance,
    PerformanceMetrics,
    SignificanceTest,
)

#%%
from Plot_Functions import plot_freq_per_chan, plot_freq_per_cond, plot_bin_per_chan, plot_conec_per_chan_pair, plot_connec_matrix_static, plot_connectivity_importance, plot_importance_distribution, plot_fold_consistency, plot_all_subspace_diagnostics

#%%
config_01 = ExperimentConfig(trial_start_time=-0.5, trial_end_time=2.5, grating_on_time=0.0, grating_off_time=2.0, bin_size=0.1)
config_001 = ExperimentConfig(trial_start_time=-0.5, trial_end_time=2.5, grating_on_time=0.0, grating_off_time=2.0, bin_size=0.01)
loader_01 = DataLoader(config_01)
loader_001 = DataLoader(config_001)
data_01 = loader_01.load_csv('multielectrode_data.csv')
data_001 = loader_001.load_csv('multielectrode_data.csv')

#%%
freq_transformer = FreqTransformer(normalize=False, per_condition=False)
freq_feature = freq_transformer.fit_transform(data_01)
plot_freq_per_chan(freq_feature, condition='all', bins=51, x_lim=False, log=False, alpha=0.7)
plot_freq_per_cond(freq_feature, bins=51, x_lim=True, log=False)

#%%
channels_to_remove = [4, 7, 10, 32, 35, 45, 63, 73, 86, 91, 95]
channels_to_remove = [4, 7, 32, 35, 45, 91, 95]

preprocessor = SpikeDataPreprocessor()
data_01_cleaned = preprocessor.remove_channels(data_01, channels_to_remove=channels_to_remove, inplace=False)
preprocessor = SpikeDataPreprocessor()
data_001_cleaned = preprocessor.remove_channels(data_001, channels_to_remove=channels_to_remove, inplace=False)

#%%
freq_transformer = FreqTransformer(normalize=True, per_condition=True)
freq_feature = freq_transformer.fit_transform(data_01_cleaned)
plot_freq_per_chan(freq_feature, condition='separate', bins=51, x_lim=True, log=False, alpha=0.7)
plot_freq_per_cond(freq_feature, bins=51, x_lim=True, log=False)

#%%
bin_transformer = BinTransformer(normalize=True, per_condition=False)
bin_feature = bin_transformer.fit_transform(data_001_cleaned)
plot_bin_per_chan(bin_feature, condition='separate', y_lim=False, conf_int=False)

#%%
preprocessor = SpikeDataPreprocessor(config=data_01_cleaned.config)
data_01_clipped = preprocessor.extract_time_window(data_01_cleaned, start_time=0.0, end_time=2.0, inplace=False)
preprocessor = SpikeDataPreprocessor(config=data_001_cleaned.config)
data_001_clipped = preprocessor.extract_time_window(data_001_cleaned, start_time=0.0, end_time=2.0, inplace=False)

#%%
conec_transformer = ConecTransformer(lag=0.1, normalize=True, method='Pearson', reduction=False, noise=True, per_condition=True)
conec_feature = conec_transformer.fit_transform(data_001_clipped)
plot_conec_per_chan_pair(conec_feature, condition='separate', y_lim=True, conf_int=False, remove_diag=True)

#%%
conec_transformer = ConecTransformer(lag=0, normalize=True, method='Pearson', reduction=True, noise=False, per_condition=False)
conec_01_feature = conec_transformer.fit_transform(data_01_clipped)
conec_001_transformer = ConecTransformer(lag=0, normalize=True, method='Pearson', reduction=True, noise=True, per_condition=True)
conec_001_feature = conec_001_transformer.fit_transform(data_001_clipped)

plot_connec_matrix_static(conec_01_feature, n_permutations=5000, p_threshold=0.05, d_threshold=0.2, remove_diag=False, remove_upper=False)
plot_connec_matrix_static(conec_001_feature, n_permutations=5000, p_threshold=0.05, d_threshold=0.2, remove_diag=False, remove_upper=False)

#%%
conec_01_transformer = ConecTransformer(
    lag=0,
    normalize=True,
    method='Pearson',
    reduction=True,
    noise=False,
    per_condition=False
)

result_01 = classify_with_transformer(
    data=data_01_clipped,
    transformer=conec_01_transformer,
    estimator=LogisticRegression(),
    n_splits=5,
    use_scaler=True,
    compute_feature_importance=True,
    random_state=42
)

#%%
plot_connectivity_importance(result_01, data=data_01_clipped)
plot_importance_distribution(result_01)
plot_fold_consistency(result_01, top_k=15)

#%%
conec_001_transformer = ConecTransformer(
    lag=0,
    normalize=True,
    method='Pearson',
    reduction=True,
    noise=True,
    per_condition=True
)

result_001 = classify_with_transformer(
    data=data_001_clipped,
    transformer=conec_001_transformer,
    estimator=LogisticRegression(),
    n_splits=5,
    use_scaler=True,
    compute_feature_importance=True,
    random_state=42
)

#%%
plot_connectivity_importance(result_001, data=data_01_clipped)
plot_importance_distribution(result_001)
plot_fold_consistency(result_001, top_k=15)

#%%
sub_transformer_01 = SubspaceRemovalTransformer(
    center_by_grand_mean=False,
    n_components=2,
    evr_threshold=0.95,
    min_components=1,
    max_components=100,
    compute_transform_diagnostics=True,
    store_projection_matrix=False
)
sub_transformer_01.fit(data_01_clipped)
data_01_sub, diag_transform_01 = sub_transformer_01.transform(data_01_clipped)

print(f"  Selected components: {sub_transformer_01.selected_components_}")
print(f"  Feature dimension: {sub_transformer_01.feature_dim_}")
print(f"  Stored projection matrix: {sub_transformer_01.P_perp_ is not None}")
print(f"  Basis shape: {sub_transformer_01.basis_.shape if sub_transformer_01.basis_ is not None else 'None'}")

figs = plot_all_subspace_diagnostics(
    sub_transformer_01, 
    data_01_clipped, 
    data_01_sub,
    save_prefix=None
)

plt.show()

#%%
sub_transformer_001 = SubspaceRemovalTransformer(
    center_by_grand_mean=False,
    n_components=2,
    evr_threshold=0.95,
    min_components=1,
    max_components=100,
    compute_transform_diagnostics=True,
    store_projection_matrix=False
)
sub_transformer_001.fit(data_001_clipped)
data_001_sub, diag_transform_001 = sub_transformer_001.transform(data_001_clipped)

print(f"  Selected components: {sub_transformer_001.selected_components_}")
print(f"  Feature dimension: {sub_transformer_001.feature_dim_}")
print(f"  Stored projection matrix: {sub_transformer_001.P_perp_ is not None}")
print(f"  Basis shape: {sub_transformer_001.basis_.shape if sub_transformer_001.basis_ is not None else 'None'}")

figs = plot_all_subspace_diagnostics(
    sub_transformer_001, 
    data_001_clipped, 
    data_001_sub,
    save_prefix=None
)

plt.show()

#%%
freq_transformer = FreqTransformer(normalize=False, per_condition=False)
freq_feature = freq_transformer.fit_transform(data_001_clipped)
plot_freq_per_chan(freq_feature, condition='all', bins=51, x_lim=True, log=False, alpha=0.7)

freq_transformer = FreqTransformer(normalize=False, per_condition=False)
freq_feature = freq_transformer.fit_transform(data_001_sub)
plot_freq_per_chan(freq_feature, condition='all', bins=51, x_lim=True, log=False, alpha=0.7)

#%%
bin_transformer = BinTransformer(normalize=True, per_condition=False)
bin_feature = bin_transformer.fit_transform(data_001_clipped)
plot_bin_per_chan(bin_feature, condition='separate', y_lim=False, conf_int=True)

bin_transformer = BinTransformer(normalize=True, per_condition=False)
bin_feature = bin_transformer.fit_transform(data_001_sub)
plot_bin_per_chan(bin_feature, condition='separate', y_lim=False, conf_int=True)


#%%
conec_01_transformer = ConecTransformer(lag=0, normalize=True, method='Pearson', reduction=True, noise=False, per_condition=False)
conec_01_feature = conec_01_transformer.fit_transform(data_01_clipped)
conec_01_sub_transformer = ConecTransformer(lag=0, normalize=True, method='Pearson', reduction=True, noise=False, per_condition=False)
conec_01_sub_feature = conec_01_sub_transformer.fit_transform(data_01_sub)

plot_connec_matrix_static(conec_01_feature, n_permutations=5000, p_threshold=0.05, d_threshold=0.2, remove_diag=False, remove_upper=False)
plot_connec_matrix_static(conec_01_sub_feature, n_permutations=5000, p_threshold=0.05, d_threshold=0.2, remove_diag=False, remove_upper=False)

conec_001_transformer = ConecTransformer(lag=0, normalize=True, method='Pearson', reduction=True, noise=True, per_condition=True)
conec_001_feature = conec_001_transformer.fit_transform(data_001_clipped)
conec_001_sub_transformer = ConecTransformer(lag=0, normalize=True, method='Pearson', reduction=True, noise=True, per_condition=True)
conec_001_sub_feature = conec_001_sub_transformer.fit_transform(data_001_sub)

plot_connec_matrix_static(conec_001_feature, n_permutations=5000, p_threshold=0.05, d_threshold=0.2, remove_diag=False, remove_upper=False)
plot_connec_matrix_static(conec_001_sub_feature, n_permutations=5000, p_threshold=0.05, d_threshold=0.2, remove_diag=False, remove_upper=False)

#%%
conec_001_fire_transformer = ConecTransformer(lag=0, normalize=True, method='Pearson', reduction=True, noise=False, per_condition=False)
conec_001_fire_feature = conec_001_fire_transformer.fit_transform(data_001_clipped)
conec_001_fire_sub_transformer = ConecTransformer(lag=0, normalize=True, method='Pearson', reduction=True, noise=False, per_condition=False)
conec_001_fire_sub_feature = conec_001_fire_sub_transformer.fit_transform(data_001_sub)

plot_connec_matrix_static(conec_001_fire_feature, n_permutations=5000, p_threshold=0.05, d_threshold=0.2, remove_diag=False, remove_upper=False)
plot_connec_matrix_static(conec_001_fire_sub_feature, n_permutations=5000, p_threshold=0.05, d_threshold=0.2, remove_diag=False, remove_upper=False)

#%%
sub_transformer_001 = SubspaceRemovalTransformer(
    center_by_grand_mean=False,
    n_components=2,
    evr_threshold=0.95,
    min_components=1,
    max_components=100,
    compute_transform_diagnostics=True,
    store_projection_matrix=False
)
conec_001_transformer = ConecTransformer(
    lag=0,
    normalize=True,
    method='Pearson',
    reduction=True,
    noise=False,
    per_condition=False
)

sequential_001 = SequentialTransformer(
    subspace_remover=sub_transformer_001,
    feature_transformer=conec_001_transformer
)

result_001 = classify_with_transformer(
    data=data_001_clipped,
    transformer=sequential_001,
    estimator=LogisticRegression(),
    n_splits=20,
    use_scaler=True,
    compute_feature_importance=True,
    random_state=42
)

#%%
plot_connectivity_importance(result_001, data=data_001_clipped)
plot_importance_distribution(result_001)
plot_fold_consistency(result_001, top_k=15)

#%%
sub_transformer_01 = SubspaceRemovalTransformer(
    center_by_grand_mean=False,
    n_components=2,
    evr_threshold=0.95,
    min_components=1,
    max_components=100,
    compute_transform_diagnostics=True,
    store_projection_matrix=False
)
conec_01_transformer = ConecTransformer(
    lag=0,
    normalize=True,
    method='Pearson',
    reduction=True,
    noise=False,
    per_condition=False
)

sequential_01 = SequentialTransformer(
    subspace_remover=sub_transformer_01,
    feature_transformer=conec_01_transformer
)

result_01 = classify_with_transformer(
    data=data_01_clipped,
    transformer=sequential_01,
    estimator=LogisticRegression(),
    n_splits=20,
    use_scaler=True,
    compute_feature_importance=True,
    random_state=42
)

#%%
plot_connectivity_importance(result_01, data=data_01_clipped)
plot_importance_distribution(result_01)
plot_fold_consistency(result_001, top_k=15)