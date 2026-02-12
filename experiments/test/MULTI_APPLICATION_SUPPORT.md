# Multi-Application Support

## Overview

The test runner now supports **three applications** instead of just CNN:

1. **CNN** (Configuration) - Neural network hyperparameter tuning
2. **Segmentation** (Composition) - Image segmentation algorithm composition
3. **FuzzyART** (Generation) - Choice function generation for clustering

## Usage

### Basic Syntax

```bash
python runner.py --application <app_name> [options]
```

### Applications and Their Datasets

| Application | Type | Datasets |
|------------|------|----------|
| `cnn` | Configuration | mnist, fashion-mnist, cifar-10, cifar-100 |
| `segmentation` | Composition | bsd500, covid, pascal |
| `fuzzyart` | Generation | chatgpt, enron, imdb |

## Examples

### CNN Application (Original)
```bash
# Single dataset
python runner.py --application cnn --datasets mnist --timesteps 50 --runs 30

# All CNN datasets
python runner.py --application cnn --all-datasets --timesteps 50 --runs 30

# Quick test
python runner.py --application cnn --quick-test
```

### Segmentation Application (NEW)
```bash
# Single dataset
python runner.py --application segmentation --datasets bsd500 --timesteps 60 --runs 30

# All segmentation datasets
python runner.py --application segmentation --all-datasets --timesteps 60 --runs 30

# Quick test
python runner.py --application segmentation --quick-test
```

### FuzzyART Application (NEW)
```bash
# Single dataset
python runner.py --application fuzzyart --datasets chatgpt --timesteps 60 --runs 30

# All text datasets
python runner.py --application fuzzyart --all-datasets --timesteps 60 --runs 30

# Quick test
python runner.py --application fuzzyart --quick-test
```

## What Gets Tested

For **each application**, the runner tests:

### Approaches (13 variants)
1. OnMAR - Accuracy Prediction (kNN, RF, XGBoost)
2. OnMAR - Design Prediction (kNN, RF, XGBoost)
3. OffMAR - Accuracy Prediction (kNN, RF, XGBoost)
4. OffMAR - Design Prediction (kNN, RF, XGBoost)
5. AutoSklearn (baseline)

### Default Parameters
- **Timesteps**: 50 for CNN, 60 for Segmentation/FuzzyART
- **Runs**: 30 (for statistical significance)
- **Meta-learners**: kNN, Random Forest, XGBoost
- **AutoSklearn time**: 3600 seconds (1 hour)

## Output Structure

Results are saved per application:

```
experiments/test/results/
├── individual_runs/
│   ├── mnist_OnMAR-AccuracyPrediction_knn_run1.json
│   ├── bsd500_OnMAR-AccuracyPrediction_knn_run1.json
│   ├── chatgpt_OnMAR-AccuracyPrediction_knn_run1.json
│   └── ...
├── summary_TIMESTAMP.csv
├── thesis_rankings_TIMESTAMP.txt
├── boxplot_data_TIMESTAMP.json
└── all_results_TIMESTAMP.json
```

## Key Changes

### 1. Application-Agnostic Architecture
- Removed hardcoded `CNNConfigurationApplication`
- Added dynamic application class loading
- Configuration pulled from OnMAR/OffMAR config files

### 2. New Command-Line Arguments
```bash
--application {cnn,segmentation,fuzzyart}
```

### 3. Automatic Dataset Selection
- `--all-datasets` now uses datasets for the chosen application
- `--quick-test` automatically picks first dataset for the application

### 4. Unified Interface
- Same runner works for all applications
- Same metrics logged (adapted per application)
- Same analysis tools work across applications

## Technical Details

### Application Configuration

Each application has:
- **Application class**: The base application implementation
- **Datasets**: List of compatible datasets
- **Configs**: OnMAR/OffMAR configs for each meta-learner

Example configuration structure:
```python
APPLICATION_INFO = {
    'cnn': {
        'class': CNNConfigurationApplication,
        'datasets': ['mnist', 'fashion-mnist', 'cifar-10', 'cifar-100'],
        'configs': {
            'onmar_acc': CNN_CONFIG_ONMAR_ACC,
            'onmar_des': CNN_CONFIG_ONMAR_DES,
            'offmar_acc': CNN_CONFIG_OFFMAR_ACC,
            'offmar_des': CNN_CONFIG_OFFMAR_DES
        }
    },
    # ... similar for segmentation and fuzzyart
}
```

### Metrics Per Application

| Application | Primary Metric | Secondary Metrics |
|------------|---------------|-------------------|
| CNN | Test Accuracy | Training time, Design reuse % |
| Segmentation | IoU (Intersection over Union) | ARI, Training time |
| FuzzyART | Clustering Accuracy | Expression complexity, Training time |

## Backward Compatibility

The runner maintains **full backward compatibility**:

```bash
# This still works (defaults to CNN)
python runner.py --datasets mnist --timesteps 50 --runs 30

# Equivalent to:
python runner.py --application cnn --datasets mnist --timesteps 50 --runs 30
```

## Troubleshooting

### Issue: "Unknown application"
**Solution**: Use `--application {cnn,segmentation,fuzzyart}`

### Issue: Dataset not found
**Solution**: Check the dataset is compatible with the chosen application
- CNN: mnist, fashion-mnist, cifar-10, cifar-100
- Segmentation: bsd500, covid, pascal
- FuzzyART: chatgpt, enron, imdb

### Issue: Import errors
**Solution**: Ensure all application modules are available:
```bash
ls applications/configuration/cnn/
ls applications/composition/segmentation/
ls applications/generation/fuzzyart/
```

## Next Steps

To run experiments on all applications:

```bash
# CNN (6-8 hours per dataset)
python runner.py --application cnn --all-datasets --timesteps 50 --runs 30

# Segmentation (8-10 hours per dataset)
python runner.py --application segmentation --all-datasets --timesteps 60 --runs 30

# FuzzyART (6-8 hours per dataset)
python runner.py --application fuzzyart --all-datasets --timesteps 60 --runs 30
```

---

**Created**: 2026-01-11
**Version**: 2.0
**Status**: Production Ready
