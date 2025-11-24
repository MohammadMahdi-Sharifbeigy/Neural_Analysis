# @title Plot Functions

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
from matplotlib.gridspec import GridSpec

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
def plot_freq_per_chan(freq_feature, condition='separate', bins=101, x_lim=False, log=False, alpha=0.7):
    
    normalize = freq_feature.metadata['normalize']
    per_condition = freq_feature.metadata['per_condition']
    n_channels = freq_feature.metadata['n_channels']
    
    channel_names = [int(float(name.split('_ch')[1])) for name in freq_feature.feature_names]
    
    n_cols = math.ceil(np.sqrt(n_channels + 1))
    n_rows = math.ceil((n_channels + 1) / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2))
    axes = np.atleast_2d(axes).flatten()
    
    color_0 = '#1f77b4'
    color_90 = '#ff7f0e'
    color_all = '#9467bd'
    
    if x_lim:
        min_freq = np.min(freq_feature.X)
        max_freq = np.max(freq_feature.X)
        xlim_range = (min_freq, max_freq * 1.05)
    
    if condition == 'separate':
        mask_0 = freq_feature.y == 0
        mask_90 = freq_feature.y == 90
        freq_0 = freq_feature.X[mask_0]
        freq_90 = freq_feature.X[mask_90]
        
        ax = axes[0]
        ax.hist(freq_0.flatten(), color=color_0, bins=bins, log=log, density=True, alpha=alpha, label='0°')
        ax.hist(freq_90.flatten(), color=color_90, bins=bins, log=log, density=True, alpha=alpha, label='90°')
        ax.axvline(x=np.mean(freq_0), linestyle='--', color=color_0, alpha=0.8)
        ax.axvline(x=np.mean(freq_90), linestyle='--', color=color_90, alpha=0.8)
        ax.set_title('All', fontsize=10, fontweight='bold')
        if x_lim:
            ax.set_xlim(xlim_range)
        ax.legend(fontsize=8)
        
        for i in range(n_channels):
            ax = axes[i + 1]
            ax.hist(freq_0[:, i], color=color_0, bins=bins, log=log, density=True, alpha=alpha, label='0°')
            ax.hist(freq_90[:, i], color=color_90, bins=bins, log=log, density=True, alpha=alpha, label='90°')
            ax.axvline(x=np.mean(freq_0[:, i]), linestyle='--', color=color_0, alpha=0.8)
            ax.axvline(x=np.mean(freq_90[:, i]), linestyle='--', color=color_90, alpha=0.8)
            ax.set_title(channel_names[i], fontsize=10)
            if x_lim:
                ax.set_xlim(xlim_range)
        
        title = f'Frequencies Per Channel {"with" if normalize else "without"} Normalization {"Per-Condition" if per_condition else ""}'
    
    elif condition == 0 or condition == 90:
        mask = freq_feature.y == condition
        freq = freq_feature.X[mask]
        color = color_0 if condition == 0 else color_90
        
        ax = axes[0]
        ax.hist(freq.flatten(), color=color, bins=bins, log=log, density=True)
        ax.axvline(x=np.mean(freq), linestyle='--', color='gray')
        ax.set_title('All', fontsize=10, fontweight='bold')
        if x_lim:
            ax.set_xlim(xlim_range)
        
        for i in range(n_channels):
            ax = axes[i + 1]
            ax.hist(freq[:, i], color=color, bins=bins, log=log, density=True)
            ax.axvline(x=np.mean(freq[:, i]), linestyle='--', color='gray')
            ax.set_title(channel_names[i], fontsize=10)
            if x_lim:
                ax.set_xlim(xlim_range)
        
        title = f'Frequencies Per Channel {condition}° Orientation {"with" if normalize else "without"} Normalization {"Per-Condition" if per_condition else ""}'
    
    elif condition == 'all':
        freq = freq_feature.X
        
        ax = axes[0]
        ax.hist(freq.flatten(), color=color_all, bins=bins, log=log, density=True)
        ax.axvline(x=np.mean(freq), linestyle='--', color='gray')
        ax.set_title('All', fontsize=10, fontweight='bold')
        if x_lim:
            ax.set_xlim(xlim_range)
        
        for i in range(n_channels):
            ax = axes[i + 1]
            ax.hist(freq[:, i], color=color_all, bins=bins, log=log, density=True)
            ax.axvline(x=np.mean(freq[:, i]), linestyle='--', color='gray')
            ax.set_title(channel_names[i], fontsize=10)
            if x_lim:
                ax.set_xlim(xlim_range)
        
        title = f'Frequencies Per Channel All Trials {"with" if normalize else "without"} Normalization {"Per-Condition" if per_condition else ""}'
    
    for j in range(n_channels + 1, len(axes)):
        fig.delaxes(axes[j])
    
    fig.text(0.5, 0.04, 'Frequencies (Hz)', ha='center', va='center')
    fig.text(0.04, 0.5, 'Density', ha='center', va='center', rotation='vertical')
    
    plt.tight_layout(rect=[0.05, 0.05, 1, 1])
    fig.suptitle(title, fontsize=16, y=1.02)
    
    plt.show()

#%%
def plot_freq_per_cond(freq_feature, bins=51, x_lim=False, log=False):
    
    normalize = freq_feature.metadata['normalize']
    per_condition = freq_feature.metadata['per_condition']
    
    mask_0 = freq_feature.y == 0
    mask_90 = freq_feature.y == 90
    
    freq_all = freq_feature.X.flatten()
    freq_0 = freq_feature.X[mask_0].flatten()
    freq_90 = freq_feature.X[mask_90].flatten()
    
    color_0 = '#1f77b4'
    color_90 = '#ff7f0e'
    color_all = '#9467bd'
    
    if x_lim:
        min_freq = np.min(freq_feature.X)
        max_freq = np.max(freq_feature.X)
        xlim_range = (min_freq, max_freq * 1.05)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    
    axes[0].hist(freq_all, color=color_all, bins=bins, log=log, density=True)
    axes[0].axvline(x=np.mean(freq_all), linestyle='--', color='gray')
    axes[0].set_title('All', fontsize=12, fontweight='bold')
    if x_lim:
        axes[0].set_xlim(xlim_range)
    
    axes[1].hist(freq_0, color=color_0, bins=bins, log=log, density=True)
    axes[1].axvline(x=np.mean(freq_0), linestyle='--', color='gray')
    axes[1].set_title('0°', fontsize=12, fontweight='bold')
    if x_lim:
        axes[1].set_xlim(xlim_range)
    
    axes[2].hist(freq_90, color=color_90, bins=bins, log=log, density=True)
    axes[2].axvline(x=np.mean(freq_90), linestyle='--', color='gray')
    axes[2].set_title('90°', fontsize=12, fontweight='bold')
    if x_lim:
        axes[2].set_xlim(xlim_range)
    
    fig.text(0.5, 0.04, 'Frequencies (Hz)', ha='center', va='center')
    fig.text(0.04, 0.5, 'Density', ha='center', va='center', rotation='vertical')
    
    title = f'Frequencies Per Condition {"with" if normalize else "without"} Normalization {"Per-Condition" if per_condition else ""}'
    
    plt.tight_layout(rect=[0.05, 0.05, 1, 1])
    fig.suptitle(title, fontsize=16, y=1.05)
    
    plt.show()

#%%
def plot_bin_per_chan(bin_feature, condition='all', y_lim=False, conf_int=False):
    normalize = bin_feature.metadata['normalize']
    per_condition = bin_feature.metadata['per_condition']
    n_channels = bin_feature.metadata['n_channels']
    n_bins = bin_feature.X.shape[1] // n_channels
    
    data_3d = bin_feature.X.reshape(bin_feature.X.shape[0], n_channels, n_bins)
    
    channel_names = [int(float(name.split('_ch')[1].split('_')[0])) for name in bin_feature.feature_names[::n_bins]]
    bin_size = bin_feature.metadata['bin_size']
    times = np.linspace(0, n_bins * bin_size, n_bins, endpoint=False)
    
    n_cols = math.ceil(np.sqrt(n_channels + 1))
    n_rows = math.ceil((n_channels + 1) / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 2))
    axes = np.atleast_2d(axes).flatten()
    
    def conf_interval(data, axis=0):
        if data.shape[axis] <= 1:
            return np.zeros(data.shape[1:] if axis == 0 else data.shape[:axis] + data.shape[axis+1:])
        sem = np.std(data, axis=axis, ddof=1) / np.sqrt(data.shape[axis])
        return 1.96 * sem
    
    if condition == 'all':
        mean_all = data_3d.mean(axis=(0, 1))
        ax0 = axes[0]
        ax0.plot(times, mean_all, color='#9467bd', linewidth=1)
        if conf_int:
            ci_all = conf_interval(data_3d.reshape(-1, n_bins), axis=0)
            ax0.fill_between(times, mean_all - ci_all, mean_all + ci_all, color='#9467bd', alpha=0.3)
        ax0.set_title('All', fontsize=10, fontweight='bold')
        ax0.set_xlim(times[0], times[-1])
        
        for i in range(n_channels):
            ax = axes[i + 1]
            mean_chan = data_3d[:, i, :].mean(axis=0)
            ax.plot(times, mean_chan, color='#9467bd', linewidth=1)
            if conf_int:
                ci = conf_interval(data_3d[:, i, :], axis=0)
                ax.fill_between(times, mean_chan - ci, mean_chan + ci, color='#9467bd', alpha=0.3)
            ax.set_title(str(channel_names[i]), fontsize=10)
            ax.set_xlim(times[0], times[-1])
    
    elif condition == 'separate':
        mask_0 = bin_feature.y == 0
        mask_90 = bin_feature.y == 90
        data_0 = data_3d[mask_0]
        data_90 = data_3d[mask_90]
        
        mean_0_all = data_0.mean(axis=(0, 1))
        mean_90_all = data_90.mean(axis=(0, 1))
        
        ax0 = axes[0]
        ax0.plot(times, mean_0_all, color='#1f77b4', linewidth=1, label='0°')
        ax0.plot(times, mean_90_all, color='#ff7f0e', linewidth=1, label='90°')
        if conf_int:
            ci_0_all = conf_interval(data_0.reshape(-1, n_bins), axis=0)
            ci_90_all = conf_interval(data_90.reshape(-1, n_bins), axis=0)
            ax0.fill_between(times, mean_0_all - ci_0_all, mean_0_all + ci_0_all, color='#1f77b4', alpha=0.3)
            ax0.fill_between(times, mean_90_all - ci_90_all, mean_90_all + ci_90_all, color='#ff7f0e', alpha=0.3)
        ax0.set_title('All', fontsize=10, fontweight='bold')
        ax0.legend(fontsize=8)
        ax0.set_xlim(times[0], times[-1])
        
        for i in range(n_channels):
            ax = axes[i + 1]
            mean_0 = data_0[:, i, :].mean(axis=0)
            mean_90 = data_90[:, i, :].mean(axis=0)
            ax.plot(times, mean_0, color='#1f77b4', linewidth=1, label='0°')
            ax.plot(times, mean_90, color='#ff7f0e', linewidth=1, label='90°')
            if conf_int:
                ci_0 = conf_interval(data_0[:, i, :], axis=0)
                ci_90 = conf_interval(data_90[:, i, :], axis=0)
                ax.fill_between(times, mean_0 - ci_0, mean_0 + ci_0, color='#1f77b4', alpha=0.3)
                ax.fill_between(times, mean_90 - ci_90, mean_90 + ci_90, color='#ff7f0e', alpha=0.3)
            ax.set_title(str(channel_names[i]), fontsize=10)
            ax.set_xlim(times[0], times[-1])
    
    else:
        mask = bin_feature.y == condition
        data_cond = data_3d[mask]
        mean_all = data_cond.mean(axis=(0, 1))
        color = '#1f77b4' if condition == 0 else '#ff7f0e'
        
        ax0 = axes[0]
        ax0.plot(times, mean_all, color=color, linewidth=1)
        if conf_int:
            ci_all = conf_interval(data_cond.reshape(-1, n_bins), axis=0)
            ax0.fill_between(times, mean_all - ci_all, mean_all + ci_all, color=color, alpha=0.3)
        ax0.set_title('All', fontsize=10, fontweight='bold')
        ax0.set_xlim(times[0], times[-1])
        
        for i in range(n_channels):
            ax = axes[i + 1]
            mean_chan = data_cond[:, i, :].mean(axis=0)
            ax.plot(times, mean_chan, color=color, linewidth=1)
            if conf_int:
                ci = conf_interval(data_cond[:, i, :], axis=0)
                ax.fill_between(times, mean_chan - ci, mean_chan + ci, color=color, alpha=0.3)
            ax.set_title(str(channel_names[i]), fontsize=10)
            ax.set_xlim(times[0], times[-1])
    
    if y_lim:
        all_means = [data_3d[:, i, :].mean(axis=0) for i in range(n_channels)]
        y_min = min(m.min() for m in all_means)
        y_max = max(m.max() for m in all_means)
        y_limits = (y_min, y_max * 1.05)
        for j in range(n_channels + 1):
            if j < len(axes):
                axes[j].set_ylim(y_limits)
    
    for j in range(n_channels + 1):
        if j < len(axes):
            ax = axes[j]
            row = j // n_cols
            if row < n_rows - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xticks(np.linspace(times[0], times[-1], num=5))
            ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], num=5))
    
    for j in range(n_channels + 1, len(axes)):
        fig.delaxes(axes[j])
    
    fig.text(0.5, 0.04, 'Time (s)', ha='center', va='center')
    fig.text(0.04, 0.5, 'Firing rate', ha='center', va='center', rotation='vertical')
    
    title = f'Binned Spike Data Per Channel {"with" if normalize else "without"} Normalization {"Per-Condition" if per_condition else ""}'
    fig.suptitle(title, fontsize=16, y=1.02)
    
    plt.tight_layout(rect=[0.05, 0.05, 1, 1])
    plt.show()

#%%
def plot_conec_per_chan_pair(conec_feature, condition='all', y_lim=False, conf_int=False, remove_diag=False):
    normalize = conec_feature.metadata['normalize']
    per_condition = conec_feature.metadata['per_condition']
    method = conec_feature.metadata['method']
    n_channels = conec_feature.metadata['n_channels']
    n_shifts = conec_feature.metadata['n_shifts']
    lags = conec_feature.metadata['lags']
    
    data_4d = conec_feature.X.reshape(conec_feature.X.shape[0], n_channels, n_channels, n_shifts)
    
    channel_ids = []
    for name in conec_feature.feature_names:
        parts = name.split('_ch')
        if len(parts) >= 2:
            ch_id = int(float(parts[1].split('_')[0]))
            if ch_id not in channel_ids:
                channel_ids.append(ch_id)
    
    fig, axes = plt.subplots(n_channels, n_channels, figsize=(n_channels * 4, n_channels * 3))
    axes = np.atleast_2d(axes)
    
    def conf_interval(data, axis=0):
        if data.shape[axis] <= 1:
            return np.zeros(data.shape[1:] if axis == 0 else data.shape[:axis] + data.shape[axis+1:])
        sem = np.std(data, axis=axis, ddof=1) / np.sqrt(data.shape[axis])
        return 1.96 * sem
    
    if remove_diag:
        off_diag_data = []
        for i in range(n_channels):
            for j in range(n_channels):
                if i != j:
                    off_diag_data.append(data_4d[:, i, j, :])
        off_diag_array = np.array(off_diag_data)
        if y_lim:
            min_val = np.min(np.mean(off_diag_array, axis=1))
            max_val = np.max(np.mean(off_diag_array, axis=1))
            y_limits = (min_val, max_val * 1.05)
    else:
        if y_lim:
            min_val = np.min(np.mean(data_4d, axis=0))
            max_val = np.max(np.mean(data_4d, axis=0))
            y_limits = (min_val, max_val * 1.05)
    
    if condition == 'all':
        mean_data = data_4d.mean(axis=0)
        
        for i in range(n_channels):
            for j in range(n_channels):
                ax = axes[i, j]
                
                if remove_diag and i == j:
                    ax.axis('off')
                    continue
                
                mean_cc = mean_data[i, j, :]
                
                ax.plot(lags, mean_cc, color='#9467bd', linewidth=1)
                if conf_int:
                    ci = conf_interval(data_4d[:, i, j, :], axis=0)
                    ax.fill_between(lags, mean_cc - ci, mean_cc + ci, color='#9467bd', alpha=0.3)
                
                if y_lim:
                    ax.set_ylim(y_limits)
                ax.set_xlim(lags[0], lags[-1])
                ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
                
                if i == 0:
                    ax.set_title(str(channel_ids[j]), fontsize=10, fontweight='bold')
                
                if j == 0:
                    ax.set_ylabel(str(channel_ids[i]), fontsize=10, fontweight='bold')
                
                if i < n_channels - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xticks(np.linspace(lags[0], lags[-1], num=5))
                    ax.tick_params(axis='x', labelsize=8)
                
                ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], num=5))
                ax.tick_params(axis='y', labelsize=8)
    
    elif condition == 'separate':
        mask_0 = conec_feature.y == 0
        mask_90 = conec_feature.y == 90
        data_0 = data_4d[mask_0]
        data_90 = data_4d[mask_90]
        
        mean_0 = data_0.mean(axis=0)
        mean_90 = data_90.mean(axis=0)
        
        for i in range(n_channels):
            for j in range(n_channels):
                ax = axes[i, j]
                
                if remove_diag and i == j:
                    ax.axis('off')
                    continue
                
                ax.plot(lags, mean_0[i, j, :], color='#1f77b4', linewidth=0.5, label='0°')
                ax.plot(lags, mean_90[i, j, :], color='#ff7f0e', linewidth=0.5, label='90°')
                
                if conf_int:
                    ci_0 = conf_interval(data_0[:, i, j, :], axis=0)
                    ci_90 = conf_interval(data_90[:, i, j, :], axis=0)
                    ax.fill_between(lags, mean_0[i, j, :] - ci_0, mean_0[i, j, :] + ci_0, 
                                    color='#1f77b4', alpha=0.3)
                    ax.fill_between(lags, mean_90[i, j, :] - ci_90, mean_90[i, j, :] + ci_90, 
                                    color='#ff7f0e', alpha=0.3)
                
                if y_lim:
                    ax.set_ylim(y_limits)
                ax.set_xlim(lags[0], lags[-1])
                ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
                
                if i == 0 and j == 0:
                    ax.legend(fontsize=8, loc='best')
                
                if i == 0:
                    ax.set_title(str(channel_ids[j]), fontsize=10, fontweight='bold')
                
                if j == 0:
                    ax.set_ylabel(str(channel_ids[i]), fontsize=10, fontweight='bold')
                
                if i < n_channels - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xticks(np.linspace(lags[0], lags[-1], num=5))
                    ax.tick_params(axis='x', labelsize=8)
                
                ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], num=5))
                ax.tick_params(axis='y', labelsize=8)
    
    else:
        mask = conec_feature.y == condition
        data_cond = data_4d[mask]
        mean_data = data_cond.mean(axis=0)
        
        color = '#1f77b4' if condition == 0 else '#ff7f0e'
        
        for i in range(n_channels):
            for j in range(n_channels):
                ax = axes[i, j]
                
                if remove_diag and i == j:
                    ax.axis('off')
                    continue
                
                mean_cc = mean_data[i, j, :]
                
                ax.plot(lags, mean_cc, color=color, linewidth=0.5)
                if conf_int:
                    ci = conf_interval(data_cond[:, i, j, :], axis=0)
                    ax.fill_between(lags, mean_cc - ci, mean_cc + ci, color=color, alpha=0.3)
                
                if y_lim:
                    ax.set_ylim(y_limits)
                ax.set_xlim(lags[0], lags[-1])
                ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
                
                if i == 0:
                    ax.set_title(str(channel_ids[j]), fontsize=10, fontweight='bold')
                
                if j == 0:
                    ax.set_ylabel(str(channel_ids[i]), fontsize=10, fontweight='bold')
                
                if i < n_channels - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xticks(np.linspace(lags[0], lags[-1], num=5))
                    ax.tick_params(axis='x', labelsize=8)
                
                ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], num=5))
                ax.tick_params(axis='y', labelsize=8)
    
    fig.text(0.5, 0.04, 'Lag (s)', ha='center', va='center', fontsize=14)
    fig.text(0.04, 0.5, f'Cross-correlation ({method})', ha='center', va='center', 
             rotation='vertical', fontsize=14)
    
    condition_str = 'All' if condition == 'all' else ('Separate' if condition == 'separate' else f'{condition}°')
    title = f'Cross-Correlation Per Channel Pair ({condition_str}) {"with" if normalize else "without"} Normalization {"Per-Condition" if per_condition else ""}'
    fig.suptitle(title, fontsize=16, y=0.98)
    
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.96])
    plt.show()

#%%
def plot_connec_matrix_static(conec_feature, n_permutations=5000, p_threshold=0.05,
                               d_threshold=0.2, remove_diag=False, remove_upper=False):
    n_channels = conec_feature.metadata['n_channels']
    normalize = conec_feature.metadata['normalize']
    noise = conec_feature.metadata['noise']
    per_condition = conec_feature.metadata['per_condition']
    method = conec_feature.metadata['method']
    bin_size = conec_feature.metadata['bin_size']

    X = conec_feature.X.reshape(conec_feature.X.shape[0], n_channels, n_channels)
    Y = conec_feature.y

    channel_ids = []
    for name in conec_feature.feature_names:
        parts = name.split('_ch')
        if len(parts) >= 2:
            ch_id = int(float(parts[1].split('_')[0]))
            if ch_id not in channel_ids:
                channel_ids.append(ch_id)

    if remove_upper:
        keep_mask = np.tril(np.ones((n_channels, n_channels), dtype=bool))
    else:
        keep_mask = np.ones((n_channels, n_channels), dtype=bool)

    if remove_diag:
        keep_mask[np.diag_indices(n_channels)] = False

    pvals = np.ones((n_channels, n_channels))
    ds = np.zeros((n_channels, n_channels))

    group0_mask = (Y == 0)
    group1_mask = (Y == 90)

    from analysis.statistics import StatisticalAnalyzer

    stat = StatisticalAnalyzer()

    def _perm_test(x, y, n_perm=5000):
        _, p, _ = stat.permutation_test(x, y, statistic_func=np.mean, n_permutations=n_perm, alternative="two-sided")
        return p

    def _cohens_d(x, y):
        return stat.cohens_d(x, y, pooled=True)

    for i in range(n_channels):
        for j in range(n_channels):
            if not keep_mask[i, j]:
                continue
            vals0 = X[group0_mask, i, j]
            vals1 = X[group1_mask, i, j]
            if vals0.size == 0 or vals1.size == 0:
                pvals[i, j] = 1.0
                ds[i, j] = 0.0
                continue
            pvals[i, j] = _perm_test(vals0.copy(), vals1.copy(), n_perm=n_permutations)
            ds[i, j] = _cohens_d(vals0, vals1)

    p_sig_mask = pvals < p_threshold
    d_sig_mask = np.abs(ds) >= d_threshold

    if not remove_upper:
        pvals[~keep_mask] = np.nan
        ds[~keep_mask] = np.nan

    if remove_diag:
        pvals[np.diag_indices(n_channels)] = np.nan
        ds[np.diag_indices(n_channels)] = np.nan

    matrices = [pvals, ds]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for idx, ax in enumerate(axes):
        mat = matrices[idx].copy()

        if idx == 0:
            im = ax.imshow(mat - 1e-8, cmap='magma_r', origin='lower', aspect='auto',
                           norm='log', vmin=1e-4, vmax=1)
        else:
            im = ax.imshow(mat, cmap='coolwarm', origin='lower', aspect='auto')

        title = f'p-values (α={p_threshold})' if idx == 0 else f"Cohen's d (|d|≥{d_threshold})"
        ax.set_title(title, fontsize=12, loc='right')
        ax.set_yticks(range(n_channels))
        ax.set_yticklabels([int(ch) for ch in channel_ids])
        ax.set_xticks(range(n_channels))
        ax.set_xticklabels([int(ch) for ch in channel_ids], rotation=90)

        for i in range(n_channels):
            for j in range(n_channels):
                if not keep_mask[i, j]:
                    continue
                if idx == 0 and p_sig_mask[i, j]:
                    ax.text(j, i, "*", ha='center', va='center',
                            fontsize=12, fontweight='bold', color='black')
                elif idx == 1 and d_sig_mask[i, j]:
                    ax.text(j, i, "*", ha='center', va='center',
                            fontsize=12, fontweight='bold', color='black')

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    fig.text(0.5, 0.04, 'Channel ID', ha='center', fontsize=12)
    fig.text(0.04, 0.5, 'Channel ID', va='center', rotation='vertical', fontsize=12)

    corr_type = 'Noise Correlation' if noise else 'Signal Correlation'
    norm_str = 'Normalized' if normalize else 'Raw'
    cond_str = 'Per-Condition' if per_condition else 'All Trials'
    title = f'{corr_type} Statistics ({method}, bin={int(bin_size*1000)}ms, {norm_str}, {cond_str})'
    fig.suptitle(title, fontsize=14, y=0.98)

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.96])
    plt.show()
    
#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import re
from typing import Optional, Dict, Tuple
from analysis import ComprehensiveClassificationResult
from data.structures import SpikeData


def plot_connectivity_importance(
    result: ComprehensiveClassificationResult,
    data: Optional[SpikeData] = None,
    figsize: tuple = (25, 12),
    cmap_diverging: str = 'RdBu_r',
    cmap_sequential: str = 'viridis',
    save_path: Optional[str] = None
):
    feat_imp = result.feature_importance_mean
    
    if feat_imp is None:
        raise ValueError("No feature importance computed. Set compute_feature_importance=True")
    
    channel_ids, channel_map = _extract_channel_mapping(feat_imp.feature_names, data)
    n_channels = len(channel_ids)
    
    perm_mean_mat = _parse_connectivity_matrix(
        feat_imp.permutation_importance_mean, 
        feat_imp.feature_names, 
        channel_map
    )
    perm_std_mat = _parse_connectivity_matrix(
        feat_imp.permutation_importance_std, 
        feat_imp.feature_names, 
        channel_map
    )
    
    shap_mean_mat = _parse_connectivity_matrix(
        feat_imp.shap_like_importance_mean, 
        feat_imp.feature_names, 
        channel_map
    )
    shap_std_mat = _parse_connectivity_matrix(
        feat_imp.shap_like_importance_std, 
        feat_imp.feature_names, 
        channel_map
    )
    
    builtin_mat = None
    if feat_imp.builtin_importance is not None:
        builtin_mat = _parse_connectivity_matrix(
            feat_imp.builtin_importance, 
            feat_imp.feature_names, 
            channel_map
        )
    
    n_methods = 3 if builtin_mat is not None else 2
    n_stats = 5
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_methods, n_stats, figure=fig, hspace=0.35, wspace=0.4)
    
    vmin_perm = min(perm_mean_mat.min(), shap_mean_mat.min())
    vmax_perm = max(perm_mean_mat.max(), shap_mean_mat.max())
    
    row = 0
    _plot_matrix_row(
        fig, gs, row, perm_mean_mat, perm_std_mat,
        "Permutation Importance (sklearn)", 
        channel_ids, vmin_perm, vmax_perm, cmap_sequential
    )
    
    row = 1
    _plot_matrix_row(
        fig, gs, row, shap_mean_mat, shap_std_mat,
        "SHAP-like Importance (custom)", 
        channel_ids, vmin_perm, vmax_perm, cmap_sequential
    )
    
    if builtin_mat is not None:
        row = 2
        _plot_matrix_row(
            fig, gs, row, builtin_mat, None,
            "Built-in Importance (model)", 
            channel_ids, builtin_mat.min(), builtin_mat.max(), cmap_diverging
        )
    
    fig.suptitle(
        f"Feature Importance Analysis: {result.model_name} with {result.metadata['transformer']['class']}\n"
        f"Test Accuracy: {result.mean_test_acc:.4f} ± {np.std(result.test_accuracies):.4f}",
        fontsize=16, fontweight='bold', y=0.98
    )
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def _plot_matrix_row(fig, gs, row, mean_mat, std_mat, title, channel_ids, vmin, vmax, cmap):
    n_channels = len(channel_ids)
    
    ax_mean = fig.add_subplot(gs[row, 0])
    im_mean = ax_mean.imshow(mean_mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax_mean.set_title(f"{title}\nMean", fontsize=11, fontweight='bold')
    ax_mean.set_xlabel("Target Channel")
    ax_mean.set_ylabel("Source Channel")
    _set_ticks(ax_mean, channel_ids)
    plt.colorbar(im_mean, ax=ax_mean, fraction=0.046, pad=0.04)
    
    if std_mat is not None:
        ax_std = fig.add_subplot(gs[row, 1])
        im_std = ax_std.imshow(std_mat, cmap='YlOrRd', vmin=0, vmax=std_mat.max(), aspect='auto')
        ax_std.set_title("Std Dev", fontsize=11, fontweight='bold')
        ax_std.set_xlabel("Target Channel")
        ax_std.set_ylabel("Source Channel")
        _set_ticks(ax_std, channel_ids)
        plt.colorbar(im_std, ax=ax_std, fraction=0.046, pad=0.04)
    else:
        ax_std = fig.add_subplot(gs[row, 1])
        ax_std.axis('off')
    
    ax_max = fig.add_subplot(gs[row, 2])
    max_per_source = mean_mat.max(axis=1)
    y_pos = np.arange(n_channels)
    ax_max.barh(y_pos, max_per_source, color='#2ca02c', alpha=0.7)
    ax_max.set_title("Max per Source", fontsize=11, fontweight='bold')
    ax_max.set_xlabel("Importance")
    ax_max.set_ylabel("Source Channel")
    ax_max.set_yticks(y_pos)
    ax_max.set_yticklabels(channel_ids, fontsize=7)
    ax_max.invert_yaxis()
    ax_max.grid(True, alpha=0.3, axis='x')
    
    ax_sum = fig.add_subplot(gs[row, 3])
    sum_per_source = mean_mat.sum(axis=1)
    ax_sum.barh(y_pos, sum_per_source, color='#d62728', alpha=0.7)
    ax_sum.set_title("Sum per Source", fontsize=11, fontweight='bold')
    ax_sum.set_xlabel("Total Importance")
    ax_sum.set_ylabel("Source Channel")
    ax_sum.set_yticks(y_pos)
    ax_sum.set_yticklabels(channel_ids, fontsize=7)
    ax_sum.invert_yaxis()
    ax_sum.grid(True, alpha=0.3, axis='x')
    
    ax_top = fig.add_subplot(gs[row, 4])
    flat_vals = mean_mat.flatten()
    top_10_indices = np.argsort(flat_vals)[-10:][::-1]
    top_10_coords = [(idx // n_channels, idx % n_channels) for idx in top_10_indices]
    top_10_values = flat_vals[top_10_indices]
    
    y_pos = np.arange(10)
    labels = [f"Ch{channel_ids[src]}→Ch{channel_ids[tgt]}" for src, tgt in top_10_coords]
    ax_top.barh(y_pos, top_10_values, color='#9467bd', alpha=0.7)
    ax_top.set_yticks(y_pos)
    ax_top.set_yticklabels(labels, fontsize=9)
    ax_top.set_title("Top 10 Connections", fontsize=11, fontweight='bold')
    ax_top.set_xlabel("Importance")
    ax_top.invert_yaxis()
    ax_top.grid(True, alpha=0.3, axis='x')


def _extract_channel_mapping(feature_names, data):
    if data is not None:
        channel_ids = sorted(data.channels)
        channel_map = {ch_id: idx for idx, ch_id in enumerate(channel_ids)}
        return channel_ids, channel_map
    
    channel_set = set()
    for name in feature_names:
        matches = re.findall(r'ch(\d+)(?:\.\d+)?', name.lower())
        for match in matches:
            channel_set.add(int(match))
    
    channel_ids = sorted(list(channel_set))
    channel_map = {ch_id: idx for idx, ch_id in enumerate(channel_ids)}
    
    return channel_ids, channel_map


def _parse_connectivity_matrix(importance_values, feature_names, channel_map):
    n_channels = len(channel_map)
    matrix = np.zeros((n_channels, n_channels))
    
    for idx, (feat_name, importance) in enumerate(zip(feature_names, importance_values)):
        match = re.search(r'ch(\d+)(?:\.\d+)?_ch(\d+)(?:\.\d+)?', feat_name.lower())
        
        if match:
            ch_id_i = int(match.group(1))
            ch_id_j = int(match.group(2))
            
            if ch_id_i in channel_map and ch_id_j in channel_map:
                idx_i = channel_map[ch_id_i]
                idx_j = channel_map[ch_id_j]
                matrix[idx_i, idx_j] = importance
    
    return matrix


def _set_ticks(ax, channel_ids):
    n_channels = len(channel_ids)
    
    if n_channels <= 20:
        step = 1
        ticks = np.arange(n_channels)
        labels = [str(ch) for ch in channel_ids]
        fontsize = 8
    elif n_channels <= 50:
        step = 5
        ticks = np.arange(0, n_channels, step)
        labels = [str(channel_ids[i]) for i in ticks]
        fontsize = 7
    else:
        step = 10
        ticks = np.arange(0, n_channels, step)
        labels = [str(channel_ids[i]) for i in ticks]
        fontsize = 6
    
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, fontsize=fontsize)
    ax.set_yticklabels(labels, fontsize=fontsize)


def plot_importance_distribution(
    result: ComprehensiveClassificationResult,
    figsize: tuple = (14, 5),
    save_path: Optional[str] = None
):
    feat_imp = result.feature_importance_mean
    
    if feat_imp is None:
        raise ValueError("No feature importance computed")
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    axes[0].hist(feat_imp.permutation_importance_mean, bins=50, color='#1f77b4', alpha=0.7, edgecolor='black')
    axes[0].set_title('Permutation Importance\nDistribution', fontweight='bold')
    axes[0].set_xlabel('Importance Value')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(feat_imp.permutation_importance_mean.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0].legend()
    
    axes[1].hist(feat_imp.shap_like_importance_mean, bins=50, color='#ff7f0e', alpha=0.7, edgecolor='black')
    axes[1].set_title('SHAP-like Importance\nDistribution', fontweight='bold')
    axes[1].set_xlabel('Importance Value')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(feat_imp.shap_like_importance_mean.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[1].legend()
    
    axes[2].scatter(
        feat_imp.permutation_importance_mean, 
        feat_imp.shap_like_importance_mean,
        alpha=0.5, s=30, c='#2ca02c'
    )
    axes[2].set_title('Importance Correlation', fontweight='bold')
    axes[2].set_xlabel('Permutation Importance')
    axes[2].set_ylabel('SHAP-like Importance')
    axes[2].grid(True, alpha=0.3)
    
    corr = np.corrcoef(feat_imp.permutation_importance_mean, feat_imp.shap_like_importance_mean)[0, 1]
    axes[2].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[2].transAxes, 
                fontsize=12, fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_fold_consistency(
    result: ComprehensiveClassificationResult,
    top_k: int = 20,
    figsize: tuple = (12, 8),
    save_path: Optional[str] = None
):
    if not result.feature_importance_folds:
        raise ValueError("No per-fold feature importance available")
    
    n_folds = len(result.feature_importance_folds)
    feat_imp_mean = result.feature_importance_mean
    
    top_indices = np.argsort(feat_imp_mean.permutation_importance_mean)[-top_k:][::-1]
    
    perm_matrix = np.zeros((top_k, n_folds))
    for fold_idx, fold_imp in enumerate(result.feature_importance_folds):
        perm_matrix[:, fold_idx] = fold_imp.permutation_importance_mean[top_indices]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    im = axes[0].imshow(perm_matrix, cmap='YlOrRd', aspect='auto')
    axes[0].set_title(f'Top {top_k} Features\nAcross Folds (Permutation)', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Fold')
    axes[0].set_ylabel('Feature Rank')
    axes[0].set_xticks(range(n_folds))
    axes[0].set_xticklabels([f'F{i+1}' for i in range(n_folds)])
    plt.colorbar(im, ax=axes[0], label='Importance')
    
    feature_names_top = [feat_imp_mean.feature_names[i] for i in top_indices]
    
    for i in range(top_k):
        axes[1].plot(range(n_folds), perm_matrix[i, :], marker='o', alpha=0.6, linewidth=1.5, label=feature_names_top[i] if i < 5 else None)
    
    axes[1].set_title(f'Top {top_k} Features\nConsistency', fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('Importance Value')
    axes[1].set_xticks(range(n_folds))
    axes[1].set_xticklabels([f'F{i+1}' for i in range(n_folds)])
    axes[1].grid(True, alpha=0.3)
    if top_k <= 5:
        axes[1].legend(fontsize=8, loc='best')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close(fig)

#%%
def plot_subspace_overview(
    transformer,
    data_before: SpikeData,
    data_after: SpikeData,
    figsize: Tuple[int, int] = (20, 12)
) -> plt.Figure:

    fig = plt.figure(figsize=figsize)
    diag = transformer.get_fit_diagnostics()

    ax1 = plt.subplot(2, 4, 1)
    svs = diag['singular_values'][:10]
    ax1.bar(range(len(svs)), svs, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(diag['selected_components'] - 0.5, color='red', linestyle='--', linewidth=2, label=f'K={diag["selected_components"]}')
    ax1.set_xlabel('Component', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Singular Value', fontsize=12, fontweight='bold')
    ax1.set_title('Orientation Subspace Structure', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = plt.subplot(2, 4, 2)
    evr = diag['explained_variance_ratio'][:10]
    cumsum = np.cumsum(evr)
    ax2.bar(range(len(evr)), evr, color='coral', alpha=0.7, edgecolor='black', label='Individual')
    ax2.plot(range(len(cumsum)), cumsum, 'o-', color='darkred', linewidth=2, markersize=8, label='Cumulative')
    ax2.axhline(0.95, color='green', linestyle='--', alpha=0.7, label='Threshold')
    ax2.set_xlabel('Component', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Explained Variance Ratio', fontsize=12, fontweight='bold')
    ax2.set_title('Variance Explained by Components', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    ax3 = plt.subplot(2, 4, 3)
    corrs = diag['component_corr_with_orientation']
    colors_corr = ['darkgreen' if abs(c) > 0.5 else 'orange' if abs(c) > 0.3 else 'gray' for c in corrs]
    ax3.bar(range(len(corrs)), corrs, color=colors_corr, alpha=0.7, edgecolor='black')
    ax3.axhline(0, color='black', linewidth=1)
    ax3.set_xlabel('Component', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Correlation with Orientation', fontsize=12, fontweight='bold')
    ax3.set_title('Component-Orientation Alignment', fontsize=14, fontweight='bold')
    ax3.set_ylim([-1, 1])
    ax3.grid(alpha=0.3)

    ax4 = plt.subplot(2, 4, 4)
    var_removed = diag['variance_removed_per_channel']
    ax4.bar(range(len(var_removed)), np.array(var_removed) * 100, color='purple', alpha=0.7, edgecolor='black')
    ax4.axhline(np.mean(var_removed) * 100, color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(var_removed)*100:.1f}%')
    ax4.set_xlabel('Channel', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Variance Removed (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Per-Channel Variance Reduction', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)

    X_before = data_before.spike_binned.reshape(data_before.spike_binned.shape[0], -1)
    X_after = data_after.spike_binned.reshape(data_after.spike_binned.shape[0], -1)
    ori_labels = data_before.orientations

    pca = PCA(n_components=2)
    X_before_pca = pca.fit_transform(X_before)
    X_after_pca = pca.transform(X_after)

    ax5 = plt.subplot(2, 4, 5)
    for ori in np.unique(ori_labels):
        mask = ori_labels == ori
        ax5.scatter(X_before_pca[mask, 0], X_before_pca[mask, 1], 
                   label=f'{ori}°', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax5.set_xlabel('PC1', fontsize=12, fontweight='bold')
    ax5.set_ylabel('PC2', fontsize=12, fontweight='bold')
    ax5.set_title('BEFORE: Trial Distribution (PCA)', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)

    ax6 = plt.subplot(2, 4, 6)
    for ori in np.unique(ori_labels):
        mask = ori_labels == ori
        ax6.scatter(X_after_pca[mask, 0], X_after_pca[mask, 1], 
                   label=f'{ori}°', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax6.set_xlabel('PC1', fontsize=12, fontweight='bold')
    ax6.set_ylabel('PC2', fontsize=12, fontweight='bold')
    ax6.set_title('AFTER: Trial Distribution (PCA)', fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(alpha=0.3)

    ax7 = plt.subplot(2, 4, 7)
    trial_mean_before = X_before.mean(axis=1)
    trial_mean_after = X_after.mean(axis=1)
    ori_numeric = ori_labels.astype(float)

    for ori in np.unique(ori_labels):
        mask = ori_labels == ori
        ax7.scatter(ori_numeric[mask], trial_mean_before[mask], 
                   label=f'{ori}°', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    z_before = np.polyfit(ori_numeric, trial_mean_before, 1)
    p_before = np.poly1d(z_before)
    x_fit = np.linspace(ori_numeric.min(), ori_numeric.max(), 100)
    ax7.plot(x_fit, p_before(x_fit), 'r--', linewidth=2, label='Fit')

    corr_before = np.corrcoef(trial_mean_before, ori_numeric)[0, 1]
    ax7.text(0.05, 0.95, f'r = {corr_before:+.4f}', transform=ax7.transAxes, 
            fontsize=12, fontweight='bold', va='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax7.set_xlabel('Orientation (°)', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Mean Firing Rate', fontsize=12, fontweight='bold')
    ax7.set_title('BEFORE: Orientation Correlation', fontsize=14, fontweight='bold')
    ax7.grid(alpha=0.3)

    ax8 = plt.subplot(2, 4, 8)
    for ori in np.unique(ori_labels):
        mask = ori_labels == ori
        ax8.scatter(ori_numeric[mask], trial_mean_after[mask], 
                   label=f'{ori}°', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    z_after = np.polyfit(ori_numeric, trial_mean_after, 1)
    p_after = np.poly1d(z_after)
    ax8.plot(x_fit, p_after(x_fit), 'r--', linewidth=2, label='Fit')

    corr_after = np.corrcoef(trial_mean_after, ori_numeric)[0, 1]
    ax8.text(0.05, 0.95, f'r = {corr_after:+.4f}', transform=ax8.transAxes, 
            fontsize=12, fontweight='bold', va='top', 
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax8.set_xlabel('Orientation (°)', fontsize=12, fontweight='bold')
    ax8.set_ylabel('Mean Firing Rate', fontsize=12, fontweight='bold')
    ax8.set_title('AFTER: Orientation Correlation', fontsize=14, fontweight='bold')
    ax8.grid(alpha=0.3)

    plt.suptitle('Subspace Removal: Complete Overview', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig


def plot_3d_trial_distribution(
    data_before: SpikeData,
    data_after: SpikeData,
    method: str = 'pca',
    figsize: Tuple[int, int] = (18, 8)
) -> plt.Figure:

    X_before = data_before.spike_binned.reshape(data_before.spike_binned.shape[0], -1)
    X_after = data_after.spike_binned.reshape(data_after.spike_binned.shape[0], -1)
    ori_labels = data_before.orientations

    if method == 'pca':
        reducer = PCA(n_components=3)
        X_before_3d = reducer.fit_transform(X_before)
        X_after_3d = reducer.transform(X_after)
        labels = ['PC1', 'PC2', 'PC3']
    else:
        X_before_3d = X_before[:, :3]
        X_after_3d = X_after[:, :3]
        labels = ['Feature 1', 'Feature 2', 'Feature 3']

    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(121, projection='3d')
    unique_ori = np.unique(ori_labels)
    colors_map = plt.cm.get_cmap('tab10', len(unique_ori))

    for idx, ori in enumerate(unique_ori):
        mask = ori_labels == ori
        ax1.scatter(X_before_3d[mask, 0], X_before_3d[mask, 1], X_before_3d[mask, 2],
                   label=f'{ori}°', alpha=0.6, s=50, c=[colors_map(idx)], 
                   edgecolors='black', linewidth=0.5)

    ax1.set_xlabel(labels[0], fontsize=12, fontweight='bold')
    ax1.set_ylabel(labels[1], fontsize=12, fontweight='bold')
    ax1.set_zlabel(labels[2], fontsize=12, fontweight='bold')
    ax1.set_title('BEFORE Removal', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(122, projection='3d')
    for idx, ori in enumerate(unique_ori):
        mask = ori_labels == ori
        ax2.scatter(X_after_3d[mask, 0], X_after_3d[mask, 1], X_after_3d[mask, 2],
                   label=f'{ori}°', alpha=0.6, s=50, c=[colors_map(idx)], 
                   edgecolors='black', linewidth=0.5)

    ax2.set_xlabel(labels[0], fontsize=12, fontweight='bold')
    ax2.set_ylabel(labels[1], fontsize=12, fontweight='bold')
    ax2.set_zlabel(labels[2], fontsize=12, fontweight='bold')
    ax2.set_title('AFTER Removal', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(alpha=0.3)

    plt.suptitle(f'3D Trial Distribution ({method.upper()})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_orientation_means_comparison(
    data_before: SpikeData,
    data_after: SpikeData,
    figsize: Tuple[int, int] = (16, 10)
) -> plt.Figure:

    ori_labels = data_before.orientations
    unique_ori = np.unique(ori_labels)
    n_channels = data_before.spike_binned.shape[1]
    n_bins = data_before.spike_binned.shape[2]

    fig = plt.figure(figsize=figsize)

    for idx, ori in enumerate(unique_ori):
        mask = ori_labels == ori

        mean_before = data_before.spike_binned[mask].mean(axis=0)
        mean_after = data_after.spike_binned[mask].mean(axis=0)

        ax1 = plt.subplot(len(unique_ori), 3, idx * 3 + 1)
        im1 = ax1.imshow(mean_before, aspect='auto', cmap='viridis', interpolation='nearest')
        ax1.set_ylabel(f'{ori}°', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time Bin', fontsize=10, fontweight='bold')
        if idx == 0:
            ax1.set_title('BEFORE', fontsize=14, fontweight='bold')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = plt.subplot(len(unique_ori), 3, idx * 3 + 2)
        im2 = ax2.imshow(mean_after, aspect='auto', cmap='viridis', interpolation='nearest')
        ax2.set_xlabel('Time Bin', fontsize=10, fontweight='bold')
        ax2.set_yticklabels([])
        if idx == 0:
            ax2.set_title('AFTER', fontsize=14, fontweight='bold')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        ax3 = plt.subplot(len(unique_ori), 3, idx * 3 + 3)
        diff = mean_before - mean_after
        vmax = np.abs(diff).max()
        im3 = ax3.imshow(diff, aspect='auto', cmap='RdBu_r', interpolation='nearest', 
                        vmin=-vmax, vmax=vmax)
        ax3.set_xlabel('Time Bin', fontsize=10, fontweight='bold')
        ax3.set_yticklabels([])
        if idx == 0:
            ax3.set_title('DIFFERENCE', fontsize=14, fontweight='bold')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    plt.suptitle('Orientation-Conditioned Mean Firing Patterns', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_component_scores_vs_orientation(
    transformer,
    data: SpikeData,
    figsize: Tuple[int, int] = (16, 10)
) -> plt.Figure:

    X = data.spike_binned.reshape(data.spike_binned.shape[0], -1)
    ori_labels = data.orientations

    if transformer.center_by_grand_mean and transformer.grand_mean_ is not None:
        X_centered = X - transformer.grand_mean_[None, :]
    else:
        X_centered = X

    scores = X_centered @ transformer.basis_
    n_comp = scores.shape[1]

    n_rows = (n_comp + 1) // 2
    n_cols = 2

    fig = plt.figure(figsize=figsize)

    for comp_idx in range(n_comp):
        ax = plt.subplot(n_rows, n_cols, comp_idx + 1)

        unique_ori = np.unique(ori_labels)
        colors_map = plt.cm.get_cmap('tab10', len(unique_ori))

        for idx, ori in enumerate(unique_ori):
            mask = ori_labels == ori
            ax.scatter(ori_labels[mask], scores[mask, comp_idx], 
                      label=f'{ori}°', alpha=0.6, s=50, c=[colors_map(idx)],
                      edgecolors='black', linewidth=0.5)

        ori_numeric = ori_labels.astype(float)
        corr = np.corrcoef(scores[:, comp_idx], ori_numeric)[0, 1]

        z = np.polyfit(ori_numeric, scores[:, comp_idx], 1)
        p = np.poly1d(z)
        x_fit = np.linspace(ori_numeric.min(), ori_numeric.max(), 100)
        ax.plot(x_fit, p(x_fit), 'r--', linewidth=2, alpha=0.7)

        ax.text(0.05, 0.95, f'r = {corr:+.4f}', transform=ax.transAxes,
               fontsize=12, fontweight='bold', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel('Orientation (°)', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'Component {comp_idx} Score', fontsize=11, fontweight='bold')
        ax.set_title(f'Component {comp_idx}', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)
        if comp_idx == 0:
            ax.legend(loc='upper right')

    plt.suptitle('Subspace Component Scores vs Orientation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_per_channel_effect(
    data_before: SpikeData,
    data_after: SpikeData,
    figsize: Tuple[int, int] = (18, 12)
) -> plt.Figure:

    n_channels = data_before.spike_binned.shape[1]
    ori_labels = data_before.orientations
    unique_ori = np.unique(ori_labels)

    fig = plt.figure(figsize=figsize)

    n_rows = int(np.ceil(np.sqrt(n_channels)))
    n_cols = int(np.ceil(n_channels / n_rows))

    for ch in range(n_channels):
        ax = plt.subplot(n_rows, n_cols, ch + 1)

        ch_before = data_before.spike_binned[:, ch, :].mean(axis=1)
        ch_after = data_after.spike_binned[:, ch, :].mean(axis=1)

        for ori in unique_ori:
            mask = ori_labels == ori
            ax.scatter(ch_before[mask], ch_after[mask], label=f'{ori}°', 
                      alpha=0.6, s=40, edgecolors='black', linewidth=0.5)

        lims = [
            min(ch_before.min(), ch_after.min()),
            max(ch_before.max(), ch_after.max())
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=2)

        corr = np.corrcoef(ch_before, ch_after)[0, 1]
        ax.text(0.05, 0.95, f'r={corr:.3f}', transform=ax.transAxes,
               fontsize=10, fontweight='bold', va='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

        ax.set_xlabel('Before', fontsize=9, fontweight='bold')
        ax.set_ylabel('After', fontsize=9, fontweight='bold')
        ax.set_title(f'Ch {ch}', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)

        if ch == 0:
            ax.legend(fontsize=8, loc='lower right')

    plt.suptitle('Per-Channel: Before vs After Mean Firing', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_variance_decomposition(
    transformer,
    data_before: SpikeData,
    data_after: SpikeData,
    figsize: Tuple[int, int] = (16, 6)
) -> plt.Figure:

    diag = transformer.get_fit_diagnostics()

    fig = plt.figure(figsize=figsize)

    ax1 = plt.subplot(1, 3, 1)
    labels = ['Kept', 'Removed']
    sizes = [
        diag['total_variance_after'],
        diag['total_variance_before'] - diag['total_variance_after']
    ]
    colors_pie = ['lightgreen', 'lightcoral']
    explode = (0.05, 0.05)

    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                                        colors=colors_pie, explode=explode,
                                        shadow=True, startangle=90)
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    ax1.set_title('Total Variance Decomposition', fontsize=14, fontweight='bold')

    ax2 = plt.subplot(1, 3, 2)
    var_removed = np.array(diag['variance_removed_per_channel']) * 100
    channels = np.arange(len(var_removed))

    colors_bar = ['darkred' if v > 40 else 'orange' if v > 20 else 'lightblue' for v in var_removed]
    ax2.barh(channels, var_removed, color=colors_bar, alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(var_removed), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(var_removed):.1f}%')
    ax2.set_xlabel('Variance Removed (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Channel', fontsize=12, fontweight='bold')
    ax2.set_title('Per-Channel Variance Removal', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3, axis='x')

    ax3 = plt.subplot(1, 3, 3)
    n_channels = data_before.spike_binned.shape[1]
    var_before_all = []
    var_after_all = []

    for ch in range(n_channels):
        var_before_all.append(np.var(data_before.spike_binned[:, ch, :]))
        var_after_all.append(np.var(data_after.spike_binned[:, ch, :]))

    ax3.scatter(var_before_all, var_after_all, s=100, alpha=0.7, 
               c=var_removed, cmap='RdYlGn_r', edgecolors='black', linewidth=1)

    lims = [
        min(min(var_before_all), min(var_after_all)),
        max(max(var_before_all), max(var_after_all))
    ]
    ax3.plot(lims, lims, 'k--', alpha=0.5, linewidth=2)

    ax3.set_xlabel('Variance Before', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Variance After', fontsize=12, fontweight='bold')
    ax3.set_title('Channel Variance: Before vs After', fontsize=14, fontweight='bold')
    ax3.grid(alpha=0.3)

    cbar = plt.colorbar(ax3.collections[0], ax=ax3)
    cbar.set_label('Variance Removed (%)', fontsize=10, fontweight='bold')

    plt.suptitle('Variance Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_time_resolved_effect(
    data_before: SpikeData,
    data_after: SpikeData,
    figsize: Tuple[int, int] = (16, 8)
) -> plt.Figure:

    ori_labels = data_before.orientations
    unique_ori = np.unique(ori_labels)
    n_bins = data_before.spike_binned.shape[2]

    fig = plt.figure(figsize=figsize)

    ax1 = plt.subplot(2, 2, 1)
    for ori in unique_ori:
        mask = ori_labels == ori
        mean_before = data_before.spike_binned[mask].mean(axis=(0, 1))
        ax1.plot(mean_before, label=f'{ori}°', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Time Bin', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Firing Rate', fontsize=12, fontweight='bold')
    ax1.set_title('BEFORE: Time Course by Orientation', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = plt.subplot(2, 2, 2)
    for ori in unique_ori:
        mask = ori_labels == ori
        mean_after = data_after.spike_binned[mask].mean(axis=(0, 1))
        ax2.plot(mean_after, label=f'{ori}°', linewidth=2, marker='o', markersize=4)
    ax2.set_xlabel('Time Bin', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Firing Rate', fontsize=12, fontweight='bold')
    ax2.set_title('AFTER: Time Course by Orientation', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    ax3 = plt.subplot(2, 2, 3)
    diff_across_ori = []
    for t in range(n_bins):
        means_at_t = []
        for ori in unique_ori:
            mask = ori_labels == ori
            means_at_t.append(data_before.spike_binned[mask, :, t].mean())
        diff_across_ori.append(np.std(means_at_t))

    ax3.plot(diff_across_ori, linewidth=2, color='darkblue', marker='o', markersize=5)
    ax3.set_xlabel('Time Bin', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Orientation Separation (SD)', fontsize=12, fontweight='bold')
    ax3.set_title('BEFORE: Orientation Discriminability', fontsize=14, fontweight='bold')
    ax3.grid(alpha=0.3)

    ax4 = plt.subplot(2, 2, 4)
    diff_across_ori_after = []
    for t in range(n_bins):
        means_at_t = []
        for ori in unique_ori:
            mask = ori_labels == ori
            means_at_t.append(data_after.spike_binned[mask, :, t].mean())
        diff_across_ori_after.append(np.std(means_at_t))

    ax4.plot(diff_across_ori_after, linewidth=2, color='darkgreen', marker='o', markersize=5)
    ax4.set_xlabel('Time Bin', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Orientation Separation (SD)', fontsize=12, fontweight='bold')
    ax4.set_title('AFTER: Orientation Discriminability', fontsize=14, fontweight='bold')
    ax4.grid(alpha=0.3)

    plt.suptitle('Time-Resolved Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_all_subspace_diagnostics(
    transformer,
    data_before: SpikeData,
    data_after: SpikeData,
    save_prefix: Optional[str] = None
):

    figs = {}

    figs['overview'] = plot_subspace_overview(transformer, data_before, data_after)

    figs['3d_pca'] = plot_3d_trial_distribution(data_before, data_after, method='pca')

    figs['ori_means'] = plot_orientation_means_comparison(data_before, data_after)

    figs['comp_scores'] = plot_component_scores_vs_orientation(transformer, data_before)

    figs['per_channel'] = plot_per_channel_effect(data_before, data_after)

    figs['variance'] = plot_variance_decomposition(transformer, data_before, data_after)

    figs['time_resolved'] = plot_time_resolved_effect(data_before, data_after)

    if save_prefix:
        for name, fig in figs.items():
            filename = f"{save_prefix}_{name}.png"
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Saved: {filename}")

    print(f"\nGenerated {len(figs)} figures total.")
    return figs











