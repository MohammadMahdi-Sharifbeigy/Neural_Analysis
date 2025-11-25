# Neural_Analysis

A comprehensive neuroelectrophysiology data analysis framework focused on spike statistics and machine learning methods for multi-electrode neural recordings.

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Module Documentation](#module-documentation)
- [Scripts Documentation](#scripts-documentation)
- [Usage Examples](#usage-examples)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

Neural_Analysis is a Python framework designed to analyze multi-electrode neural recording data with focus on:
- **Spike Statistics**: Extract and analyze spike patterns from multi-electrode recordings
- **Feature Extraction**: Generate meaningful features from raw spike data
- **Machine Learning**: Apply classification algorithms to predict neural responses
- **Statistical Analysis**: Perform rigorous statistical testing with multiple comparison correction
- **Visualization**: Comprehensive plotting functions for result presentation

This project provides a modular, extensible architecture for neurophysiology data analysis workflows.

## ✨ Features

- **Multi-Electrode Data Support**: Load and process spike data from multiple recording channels
- **Flexible Feature Extraction**: Multiple feature transformers (firing rates, connectivity, binned spikes, subspace analysis)
- **Advanced Classification Pipelines**: Support for 10+ ML classifiers with cross-validation
- **Feature Importance Analysis**: SHAP-like importance scoring and feature contribution analysis
- **Statistical Rigor**: FDR correction, hypothesis testing, and significance assessment
- **Data Quality Control**: Automated channel quality assessment and artifact detection
- **Comprehensive Visualization**: 1,500+ lines of plotting functions for all analysis outputs
- **Modular Design**: Clean separation of concerns with dedicated modules for data, features, and analysis

## 📁 Project Structure

```
Neural_Analysis/
├── data/                                  # Data loading and preprocessing
│   ├── __init__.py                       # Module exports (defines public API)
│   ├── structures.py                     # Core data structures and enums
│   ├── loader.py                         # CSV data loading and validation
│   └── preprocessor.py                   # Data preprocessing and quality control
│
├── features/                              # Feature extraction and transformation
│   ├── __init__.py                       # Module exports
│   └── transformers.py                   # Feature extraction transformers
│
├── analysis/                              # Statistical analysis and ML
│   ├── __init__.py                       # Module exports (version: 1.1.0)
│   ├── statistics.py                     # Statistical analysis and FDR correction
│   ├── classification.py                 # ML classification pipelines
│   └── metrics.py                        # Performance metrics and significance testing
│
├── main.py                                # Main analysis script (358 lines)
├── Plot_Functions.py                      # Visualization utilities (1,532 lines)
├── multielectrode_data.csv                # Neural recording dataset (21 MB)
├── README.md                              # This file
└── LICENSE                                # MIT License (© 2025 MohammadMahdi)
```

## 💻 Installation

### Prerequisites
- Python 3.6+
- pip package manager

### Required Dependencies

```bash
numpy>=1.19.0
pandas>=1.1.0
scipy>=1.5.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
```

### Setup

1. **Clone or download the repository:**
   ```bash
   cd Neural_Analysis
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy pandas scipy scikit-learn matplotlib
   ```

3. **Verify installation:**
   ```python
   python -c "import data, features, analysis; print('Installation successful!')"
   ```

## 🚀 Quick Start

### Basic Workflow

```python
from data.loader import DataLoader
from data.preprocessor import preprocess_pipeline
from features.transformers import SequentialTransformer
from analysis.classification import classify_with_transformer

# 1. Load data
loader = DataLoader()
spike_data = loader.load_csv('multielectrode_data.csv')

# 2. Preprocess
preprocessed = preprocess_pipeline(spike_data)

# 3. Extract features
transformer = SequentialTransformer()
features = transformer.fit_transform(preprocessed)

# 4. Classify
result = classify_with_transformer(features, preprocessed)

# 5. Analyze results
print(f"Accuracy: {result.metrics.accuracy:.3f}")
print(f"AUC: {result.metrics.auc:.3f}")
```

## 📚 Module Documentation

### `data` Module - Data Loading & Preprocessing

#### `structures.py` - Core Data Structures

Defines immutable, type-safe data structures using Python dataclasses:

- **`Orientation`** (enum): Recording angle orientation
  - `DEG_0`: 0-degree orientation
  - `DEG_90`: 90-degree orientation
  - `ALL`: Combined orientation

- **`ExperimentConfig`** (dataclass): Experiment metadata and trial timing
  - Trial duration, stimulus onset/offset times
  - Number of trials per condition
  - Stimulus parameters

- **`SpikeData`** (dataclass): Neural spike data container
  - `spike_times`: Neural spike time points (seconds)
  - `channels`: Recording channel indices
  - `trials`: Trial identifiers
  - `orientations`: Stimulus orientation per spike

- **`Feature`** (dataclass): Extracted feature matrix
  - `data`: NumPy array of extracted features
  - `channel_ids`: Associated channel identifiers
  - `feature_names`: Descriptive feature names

- **`PSTH`** (dataclass): Peri-stimulus time histogram
  - `time_bins`: Temporal bins
  - `spike_counts`: Spike counts per bin per condition

- **`ConnectivityMatrix`** (dataclass): Neural connectivity measures
  - `correlation_matrix`: Cross-channel correlations
  - `lags`: Associated temporal lags

- **`StatisticalResult`** (dataclass): Statistical test results
  - p-values, test statistics, effect sizes
  - Multiple comparison correction information

- **`ClassificationResult`** (dataclass): ML classification output
  - Predictions, probabilities, performance metrics

#### `loader.py` - Data Loading (115 lines)

**`DataLoader` class**: Loads neural spike data from CSV files

Key methods:
- `load_csv()`: Load and validate spike data
- `validate_columns()`: Ensure required columns present
- `filter_channels()`: Select subset of recording channels
- `filter_by_orientation()`: Filter spikes by stimulus orientation

**Required CSV columns:**
- `trial`: Trial identifier
- `channel`: Recording channel number
- `orientation`: Stimulus orientation (0 or 90 degrees)
- `time`: Spike time in seconds

#### `preprocessor.py` - Data Preprocessing (550 lines)

**`SpikeDataPreprocessor` class**: Comprehensive preprocessing utilities
- `remove_bad_channels()`: Exclude noisy channels
- `extract_trial_window()`: Extract specific time windows
- `apply_temporal_filter()`: Temporal filtering
- `normalize_rates()`: Rate normalization

**`DataQualityChecker` class**: Quality assessment
- `check_channel_quality()`: Evaluate channel signal quality
- `detect_artifacts()`: Identify noise and artifacts
- `assess_overall_quality()`: Overall dataset quality score

**`preprocess_pipeline()` function**: Complete preprocessing workflow
- Automated quality control
- Channel removal and filtering
- Normalization and standardization

---

### `features` Module - Feature Extraction

#### `transformers.py` - Feature Extraction Transformers (801 lines)

Implements sklearn-compatible transformer interface with `fit()` and `transform()` methods.

**`BaseFeatureTransformer` class**: Abstract base class
- Template for all feature extractors
- Standardized fit/transform interface

**`FreqTransformer` class**: Firing Rate Features
- Extracts mean firing rate for each channel and condition
- Per-condition analysis
- Normalization options
- Output: One feature per channel per condition

**`ConecTransformer` class**: Connectivity Features
- Computes cross-channel correlations
- Supports temporal lags
- Features: Raw correlation, normalized correlation
- Output: Channel pair correlations with lag support

**`BinTransformer` class**: Binned Spike Counts
- Discretizes spike times into temporal bins
- Computes spike count per bin per trial
- Configurable bin width
- Output: High-dimensional sparse features

**`SubspaceRemovalTransformer` class**: PCA-Based Subspace Removal
- Principal Component Analysis on spike data
- Removes principal components exceeding variance threshold
- Noise modeling and filtering
- Configurable explained variance retention

**`SequentialTransformer` class**: Feature Pipeline
- Chains multiple transformers in sequence
- First applies subspace removal
- Then applies selected feature transformer
- Reduces noise while preserving signal

---

### `analysis` Module - Statistical Analysis & ML

#### `statistics.py` - Statistical Analysis (477 lines)

**`StatisticalAnalyzer` class**: Comprehensive statistical tools
- Hypothesis testing (t-tests, ANOVA variants)
- Effect size computation (Cohen's d, eta-squared)
- Statistical result organization

**`FDRCorrection` class**: Multiple Comparison Correction
- Benjamini-Hochberg FDR correction
- Bonferroni correction
- Q-value computation
- Handles multiple statistical tests

Methods:
- `correct()`: Apply FDR correction to p-values
- `get_significant()`: Filter results by significance level
- `compute_q_values()`: Convert p-values to q-values

#### `classification.py` - ML Classification Pipelines (781 lines)

**`AdvancedClassificationPipeline` class**: Full ML pipeline
- **Supported Classifiers**:
  - k-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Logistic Regression
  - Gaussian Naive Bayes
  - Gaussian Process Classifier
  - Decision Tree
  - Random Forest
  - AdaBoost
  - Multi-Layer Perceptron (Neural Network)
  - Quadratic Discriminant Analysis (QDA)

- **Features**:
  - K-fold cross-validation (default: 5 folds)
  - Automatic feature scaling (StandardScaler)
  - Feature importance computation
  - Hyperparameter tuning options
  - Per-fold results tracking

Methods:
- `fit()`: Train pipeline on data
- `cross_validate()`: Perform k-fold cross-validation
- `get_importance()`: Compute feature importance scores

**`classify_with_transformer()` function**: Simplified ML workflow
- Chains feature extraction and classification
- Returns comprehensive results in one call

**`ComprehensiveClassificationResult` class**: Result container
- `predictions`: Model predictions
- `probabilities`: Prediction probabilities
- `metrics`: Performance metrics
- `importance`: Feature importance scores
- `folds`: Per-fold results and metrics

**`PerformanceMetrics` class**: Classification Metrics
- `accuracy`: Fraction correct
- `sensitivity`: True positive rate
- `specificity`: True negative rate
- `auc`: Area under ROC curve
- `precision`: Positive predictive value
- `f1_score`: Harmonic mean of precision/recall

#### `metrics.py` - Performance Metrics (95 lines)

**`PerformanceMetrics` dataclass**: Core performance measures
- Classification accuracy
- ROC AUC score
- Sensitivity, specificity, precision
- F1-score
- Confusion matrix

**`SignificanceTest` class**: Statistical Significance Testing
- Tests if classifier performance exceeds chance
- Null distribution generation
- P-value computation
- Bootstrapping support

---

## 🔧 Scripts Documentation

### `main.py` - Main Analysis Script (358 lines)

The primary analysis workflow implementing a cell-based execution model similar to Jupyter notebooks.

**Purpose**: Execute complete analysis pipeline from data loading through visualization

**Key Operations**:

1. **Data Loading**
   ```python
   loader = DataLoader()
   spike_data = loader.load_csv('multielectrode_data.csv')
   ```

2. **Data Preprocessing**
   ```python
   spike_data = preprocess_pipeline(spike_data)
   ```

3. **Feature Extraction**
   ```python
   # Frequency features
   freq_features = FreqTransformer().fit_transform(spike_data)

   # Connectivity features
   connectivity = ConecTransformer().fit_transform(spike_data)

   # Binned features
   binned = BinTransformer().fit_transform(spike_data)
   ```

4. **Classification**
   ```python
   pipeline = AdvancedClassificationPipeline()
   result = pipeline.fit(features, labels)
   ```

5. **Feature Importance**
   ```python
   importance = result.get_importance()
   ```

6. **Statistical Analysis**
   ```python
   analyzer = StatisticalAnalyzer()
   stats = analyzer.analyze(result)
   ```

7. **Visualization**
   ```python
   from Plot_Functions import plot_classification_results
   plot_classification_results(result)
   ```

**Running the script**:
```bash
python main.py
```

### `Plot_Functions.py` - Visualization Utilities (1,532 lines)

Comprehensive plotting library for all analysis outputs.

**Contains 30+ visualization functions** including:

- `plot_spike_raster()`: Raster plot of spike times
- `plot_psth()`: Peri-stimulus time histogram
- `plot_firing_rate()`: Firing rate over time
- `plot_connectivity_matrix()`: Correlation matrices
- `plot_classification_confusion()`: Confusion matrix
- `plot_roc_curve()`: ROC curves with AUC
- `plot_feature_importance()`: Feature importance bars
- `plot_cross_validation()`: CV fold results
- `plot_neural_trajectory()`: Neural state trajectories
- `plot_orientation_tuning()`: Orientation selectivity
- `plot_heatmaps()`: Various heatmap visualizations
- And many more...

**Features**:
- Matplotlib-based plotting
- Publication-quality figures
- Customizable styling and colors
- Support for multiple subplots
- Data validation and error handling

**Usage**:
```python
from Plot_Functions import *
import matplotlib.pyplot as plt

# Plot spike raster
plot_spike_raster(spike_data)
plt.show()

# Plot classification results
plot_classification_confusion(result)
plt.show()
```

---

## 💡 Usage Examples

### Example 1: Basic Data Loading and Exploration

```python
from data.loader import DataLoader
from data.preprocessor import DataQualityChecker

# Load data
loader = DataLoader()
spike_data = loader.load_csv('multielectrode_data.csv')

# Check data quality
quality_checker = DataQualityChecker()
quality_scores = quality_checker.check_channel_quality(spike_data)
print(f"Channel quality scores: {quality_scores}")

# Filter by orientation
oriented_data = loader.filter_by_orientation(spike_data, orientation='0')
print(f"Spikes at 0°: {len(oriented_data.spike_times)}")
```

### Example 2: Feature Extraction Pipeline

```python
from features.transformers import SequentialTransformer, FreqTransformer
from data.preprocessor import preprocess_pipeline

# Preprocess
spike_data = preprocess_pipeline(spike_data)

# Extract frequency features with subspace removal
transformer = SequentialTransformer()
features = transformer.fit_transform(spike_data)

print(f"Feature shape: {features.data.shape}")
print(f"Feature names: {features.feature_names}")
```

### Example 3: Classification with Cross-Validation

```python
from analysis.classification import AdvancedClassificationPipeline

# Create and run pipeline
pipeline = AdvancedClassificationPipeline(
    classifier='RandomForest',
    n_folds=5,
    random_state=42
)

# Train and validate
result = pipeline.fit(features, labels)

# Print results
print(f"Accuracy: {result.metrics.accuracy:.3f}")
print(f"AUC: {result.metrics.auc:.3f}")
print(f"Sensitivity: {result.metrics.sensitivity:.3f}")
```

### Example 4: Feature Importance Analysis

```python
# Get feature importance
importance = result.get_importance()

# Plot top 10 features
from Plot_Functions import plot_feature_importance
plot_feature_importance(importance, top_n=10)
```

### Example 5: Statistical Analysis with Multiple Comparison Correction

```python
from analysis.statistics import StatisticalAnalyzer, FDRCorrection

# Perform statistical tests
analyzer = StatisticalAnalyzer()
p_values = analyzer.compare_conditions(spike_data)

# Correct for multiple comparisons
fdr_corrector = FDRCorrection()
corrected_results = fdr_corrector.correct(p_values, alpha=0.05)

print(f"Significant features: {corrected_results['significant'].sum()}")
```

---

## 🏗️ Data Flow Architecture

```
Raw CSV Data
    ↓
DataLoader
    ↓
SpikeData (structured)
    ↓
Preprocessor (quality control, filtering)
    ↓
Cleaned SpikeData
    ↓
Feature Transformers (Freq, Connectivity, Binned, etc.)
    ↓
Feature Matrix
    ↓
Classification Pipeline (scaling + ML)
    ↓
Classification Result (predictions, metrics, importance)
    ↓
Statistical Analysis & Visualization
    ↓
Comprehensive Results Report
```

---

## 📊 Supported Machine Learning Models

| Classifier | Type | Best For |
|-----------|------|----------|
| **KNN** | Distance-based | Non-linear patterns |
| **SVM** | Kernel-based | High-dimensional data |
| **Logistic Regression** | Linear | Interpretability |
| **Gaussian Naive Bayes** | Probabilistic | Fast baseline |
| **Gaussian Process** | Bayesian | Uncertainty estimation |
| **Decision Tree** | Tree-based | Single variable importance |
| **Random Forest** | Ensemble | Robust predictions |
| **AdaBoost** | Boosting | Weak learner ensemble |
| **Neural Network** | Deep learning | Complex patterns |
| **QDA** | Quadratic discriminant | Non-linear boundaries |

---

## 📈 Statistical Features

- **Hypothesis Testing**: t-tests, ANOVA variants
- **Effect Sizes**: Cohen's d, eta-squared
- **Multiple Comparison Correction**: Benjamini-Hochberg FDR, Bonferroni
- **Cross-Validation**: K-fold with per-fold metrics
- **Significance Testing**: Classifier vs. chance level
- **Bootstrapping**: Confidence interval estimation

---

## 🔍 Quality Control Features

- **Channel Quality Assessment**: SNR, spike consistency metrics
- **Artifact Detection**: Automated noise identification
- **Data Validation**: Required field verification
- **Channel Filtering**: Remove bad/noisy channels
- **Time Window Extraction**: Precise temporal alignment

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Follow the existing code style and structure
2. Add docstrings to all new functions and classes
3. Ensure backward compatibility with existing code
4. Test changes with the provided example data
5. Update this README for significant new features

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

**Copyright © 2025 MohammadMahdi Sharifbeigy**

---

## 📧 Project Information

**Type**: Neuroelectrophysiology Data Analysis Framework
**Language**: Python 3.6+
**Version**: 1.1.0
**Status**: Active Development
**Use Case**: Multi-electrode spike analysis, neural classification, feature importance analysis

---

## 🙏 Acknowledgments

This project implements modern machine learning and statistical analysis techniques for neural data analysis, suitable for:
- Computational neuroscience research
- Educational coursework in neuroelectrophysiology
- Spike statistics and machine learning training
- Comparative classifier evaluation

---

## 📚 References & Further Reading

- **NumPy/SciPy**: Numerical computing https://numpy.org, https://scipy.org
- **Pandas**: Data manipulation https://pandas.pydata.org
- **scikit-learn**: Machine learning https://scikit-learn.org
- **Matplotlib**: Visualization https://matplotlib.org

---

**Last Updated**: November 2025
**Total Code**: 4,952 lines of Python
**Modules**: 3 (data, features, analysis)
**Scripts**: 2 main (main.py, Plot_Functions.py)