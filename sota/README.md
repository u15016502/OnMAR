# Auto-sklearn Integration

**State-of-the-Art AutoML Baseline** for comparing with OnMAR and OffMAR approaches.

This module provides a wrapper that allows Auto-sklearn to work with all three applications (CNN, Segmentation, Fuzzy ART) through the BaseApplication interface.

---

## Table of Contents

1. [Overview](#overview)
2. [How It Works](#how-it-works)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Usage Examples](#usage-examples)
6. [Configuration](#configuration)
7. [Comparison with OnMAR/OffMAR](#comparison-with-onmaroffmar)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### What is Auto-sklearn?

Auto-sklearn is a state-of-the-art AutoML system that:
- Uses **Bayesian optimization** to search hyperparameter spaces
- Employs **meta-learning** to warm-start the optimization
- Builds **ensemble models** for improved performance
- Is built on top of scikit-learn

### Why Use Auto-sklearn as a Baseline?

Auto-sklearn serves as a **strong SOTA baseline** because:
1. **Well-established**: Widely used in AutoML research
2. **Meta-learning**: Similar conceptual foundation to OnMAR/OffMAR
3. **Fair comparison**: Uses sophisticated optimization (SMAC) vs GA/GE
4. **Reproducible**: Standard tool with published results

### Auto-sklearn vs. OnMAR/OffMAR

| Aspect | Auto-sklearn | OnMAR/OffMAR |
|--------|--------------|--------------|
| **Design Algorithm** | Bayesian optimization (SMAC) | GA/GE |
| **Meta-learning** | Offline (warmstart) | Online or Offline |
| **Real-time Adaptation** | ❌ No | ✅ Yes (OnMAR) |
| **Design Reuse** | ❌ No | ✅ Yes |
| **Target** | General AutoML | Real-time dynamic designs |

---

## How It Works

### Architecture

```
AutoSklearnWrapper
    │
    ├─> BaseApplication Interface
    │   ├─> get_design_space() → ConfigSpace
    │   ├─> train(design) → performance
    │   └─> evaluate(design) → test results
    │
    └─> SMAC Optimizer
        ├─> Bayesian Optimization
        ├─> Configuration sampling
        └─> Performance modeling
```

### Optimization Process

```
1. Initialize:
   - Load application
   - Create ConfigSpace from design space
   - Set up SMAC scenario

2. Optimization Loop:
   For each iteration:
       a. SMAC proposes configuration (design)
       b. Evaluate design → get performance
       c. Update SMAC model
       d. Continue until budget exhausted

3. Return:
   - Best design found
   - Optimization trajectory
   - Test set performance
```

### Key Differences from Standard Auto-sklearn

Our wrapper **replaces the design algorithm** (GA/GE) with Auto-sklearn's SMAC optimizer:

- **CNN**: Auto-sklearn searches hyperparameter space (replaces manual tuning)
- **Segmentation**: Auto-sklearn searches algorithm compositions (replaces GA)
- **Fuzzy ART**: Auto-sklearn searches choice function parameters (replaces GE)

---

## Installation

### Prerequisites

```bash
# Core dependencies
pip install numpy scikit-learn

# SMAC (Auto-sklearn's optimizer)
pip install smac

# ConfigSpace (for defining search spaces)
pip install ConfigSpace
```

### Optional

```bash
# For plotting
pip install matplotlib

# Application-specific dependencies
pip install torch torchvision  # CNN
pip install opencv-python scikit-image  # Segmentation
pip install nltk  # Fuzzy ART
```

---

## Quick Start

### Basic Usage

```python
from sota.autosklearn_wrapper import AutoSklearnWrapper
from applications.configuration.cnn.cnn_application import CNNConfigurationApplication

# Initialize application
app = CNNConfigurationApplication(dataset_name='mnist', random_seed=42)

# Initialize Auto-sklearn wrapper
autosklearn = AutoSklearnWrapper(
    application=app,
    time_budget=3600,  # 1 hour
    per_run_time_limit=300,  # 5 min per evaluation
    n_jobs=1,
    random_seed=42
)

# Run optimization
results = autosklearn.run_optimization(
    dataset_name='mnist',
    max_evaluations=50
)

# Access results
print(f"Best performance: {results['best_performance']}")
print(f"Best design: {results['best_design']}")
print(f"Test performance: {results['test_results']['test_performance']}")
```

---

## Usage Examples

### Example 1: CNN Configuration

```python
from sota.autosklearn_wrapper import AutoSklearnWrapper
from applications.configuration.cnn.cnn_application import CNNConfigurationApplication

# Initialize
app = CNNConfigurationApplication(dataset_name='mnist', random_seed=42)

autosklearn = AutoSklearnWrapper(
    application=app,
    time_budget=3600,
    max_evaluations=50
)

# Run optimization
results = autosklearn.run_optimization(dataset_name='mnist')

# Plot trajectory
performances, eval_nums = autosklearn.get_incumbent_trajectory()
import matplotlib.pyplot as plt
plt.plot(eval_nums, performances)
plt.xlabel('Evaluation Number')
plt.ylabel('Best Performance So Far')
plt.title('Auto-sklearn Optimization on MNIST')
plt.show()
```

### Example 2: Segmentation Composition

```python
from sota.autosklearn_wrapper import AutoSklearnWrapper
from applications.composition.segmentation.segmentation_application import SegmentationCompositionApplication

# Initialize
app = SegmentationCompositionApplication(dataset_name='bsd500', random_seed=42)

autosklearn = AutoSklearnWrapper(
    application=app,
    time_budget=7200,  # 2 hours
    per_run_time_limit=600,  # 10 min per evaluation (segmentation is slow)
    max_evaluations=30
)

# Run optimization
results = autosklearn.run_optimization(dataset_name='bsd500', max_evaluations=30)
```

### Example 3: Fuzzy ART Generation

```python
from sota.autosklearn_wrapper import AutoSklearnWrapper
from applications.generation.fuzzyart.fuzzyart_application import FuzzyARTGenerationApplication

# Initialize
app = FuzzyARTGenerationApplication(dataset_name='enron', random_seed=42)

autosklearn = AutoSklearnWrapper(
    application=app,
    time_budget=3600,
    max_evaluations=40
)

# Run optimization
results = autosklearn.run_optimization(dataset_name='enron', max_evaluations=40)
```

---

## Configuration

### Time Budget vs. Max Evaluations

You can control optimization either by **time** or **number of evaluations**:

```python
# Time-based (run for 1 hour)
autosklearn = AutoSklearnWrapper(app, time_budget=3600)
results = autosklearn.run_optimization('mnist')

# Evaluation-based (run for 50 evaluations)
autosklearn = AutoSklearnWrapper(app, time_budget=10000)  # Large budget
results = autosklearn.run_optimization('mnist', max_evaluations=50)

# Combined (stop at whichever comes first)
autosklearn = AutoSklearnWrapper(app, time_budget=3600)
results = autosklearn.run_optimization('mnist', max_evaluations=100)
```

### Per-Evaluation Time Limit

```python
# Fast evaluations (e.g., CNN)
autosklearn = AutoSklearnWrapper(
    app,
    time_budget=3600,
    per_run_time_limit=300  # 5 minutes max per design
)

# Slow evaluations (e.g., Segmentation)
autosklearn = AutoSklearnWrapper(
    app,
    time_budget=7200,
    per_run_time_limit=600  # 10 minutes max per design
)
```

### Parallelization

```python
# Run evaluations in parallel (if application supports it)
autosklearn = AutoSklearnWrapper(
    app,
    time_budget=3600,
    n_jobs=4  # Use 4 parallel workers
)
```

---

## Comparison with OnMAR/OffMAR

### Experimental Setup for Fair Comparison

To fairly compare Auto-sklearn with OnMAR/OffMAR:

```python
from sota.autosklearn_wrapper import AutoSklearnWrapper
from OnMAR.accuracy_prediction.onmar import OnMARAccuracyPrediction
from OffMAR.design_prediction.offmar import OffMARDesignPrediction

# Same application and dataset
app = CNNConfigurationApplication(dataset_name='mnist', random_seed=42)

# Auto-sklearn (SOTA baseline)
autosklearn = AutoSklearnWrapper(app, time_budget=3600, max_evaluations=50)
as_results = autosklearn.run_optimization('mnist')

# OnMAR (our online approach)
onmar = OnMARAccuracyPrediction(app, meta_learner_type='knn', theta_p=0.85)
onmar_results = onmar.run_onmar('mnist', timesteps=50)

# OffMAR (our offline approach)
offmar = OffMARDesignPrediction(app, meta_learner_type='rf', theta_p=0.85)
offmar_results = offmar.run_full_offmar('mnist', timesteps=50)

# Compare
print(f"Auto-sklearn: {as_results['best_performance']:.4f}")
print(f"OnMAR:        {onmar_results['best_performance']:.4f}")
print(f"OffMAR:       {offmar_results['phase_2']['best_performance']:.4f}")
```

### Key Comparison Metrics

| Metric | Auto-sklearn | OnMAR/OffMAR |
|--------|--------------|--------------|
| **Best Performance** | `results['best_performance']` | `results['best_performance']` |
| **Test Performance** | `results['test_results']['test_performance']` | `results['test_results']['test_performance']` |
| **Total Runtime** | `results['total_time']` | `results['total_time']` |
| **Evaluations** | `results['num_evaluations']` | `results['num_design_algorithm_calls']` |

### Expected Performance Differences

**Auto-sklearn advantages:**
- Sophisticated Bayesian optimization
- Meta-learning warm-start
- Well-tuned for many problems

**OnMAR/OffMAR advantages:**
- Real-time adaptation (OnMAR)
- Design reuse (reduces evaluations)
- Continuous learning from changing data (OnMAR)
- Reproducibility (OffMAR)

---

## Troubleshooting

### Problem 1: SMAC Not Installed

**Symptoms:**
```
ImportError: No module named 'smac'
```

**Solution:**
```bash
pip install smac
```

---

### Problem 2: ConfigSpace Import Error

**Symptoms:**
```
ImportError: cannot import name 'ConfigurationSpace'
```

**Solution:**
```bash
pip install ConfigSpace
```

---

### Problem 3: Optimization Runs Too Quickly

**Symptoms:** Optimization completes in seconds with very few evaluations

**Cause:** Time budget too small or max_evaluations too low

**Solution:**
```python
# Increase time budget
autosklearn = AutoSklearnWrapper(app, time_budget=7200)  # 2 hours

# Or increase max evaluations
results = autosklearn.run_optimization('mnist', max_evaluations=100)
```

---

### Problem 4: Individual Evaluations Timeout

**Symptoms:** Many evaluations fail or return high cost

**Cause:** `per_run_time_limit` too small for application

**Solution:**
```python
# Increase per-evaluation time limit
autosklearn = AutoSklearnWrapper(
    app,
    time_budget=7200,
    per_run_time_limit=900  # 15 minutes instead of 5
)
```

---

### Problem 5: Out of Memory

**Symptoms:** Process killed or memory error

**Cause:** Too many parallel jobs or large models

**Solution:**
```python
# Reduce parallel jobs
autosklearn = AutoSklearnWrapper(
    app,
    time_budget=3600,
    n_jobs=1  # Sequential instead of parallel
)
```

---

## Advanced Usage

### Accessing Optimization Trajectory

```python
# Run optimization
results = autosklearn.run_optimization('mnist', max_evaluations=50)

# Get all evaluations
all_performances, all_evals = autosklearn.get_optimization_trajectory()

# Get best-so-far trajectory
best_performances, best_evals = autosklearn.get_incumbent_trajectory()

# Plot both
import matplotlib.pyplot as plt
plt.scatter(all_evals, all_performances, alpha=0.3, label='All')
plt.plot(best_evals, best_performances, 'r-', linewidth=2, label='Best')
plt.legend()
plt.show()
```

### Analyzing Evaluation History

```python
# Access full evaluation history
history = results['evaluation_history']

for i, entry in enumerate(history):
    print(f"Eval {i+1}: Design={entry['design']}, Perf={entry['performance']:.4f}")
```

### Warm-Starting from Previous Run

```python
# First run
autosklearn = AutoSklearnWrapper(app, time_budget=1800)
results1 = autosklearn.run_optimization('mnist', max_evaluations=25)

# Get best design
best_design = results1['best_design']

# Use as starting point for OnMAR/OffMAR
onmar = OnMARAccuracyPrediction(app, meta_learner_type='knn')
onmar_results = onmar.run_onmar('mnist', timesteps=50, initial_design=best_design)
```

---

## References

- **SMAC**: Sequential Model-based Algorithm Configuration
- **Auto-sklearn**: Efficient and Robust Automated Machine Learning
- **ConfigSpace**: Python library for configuration spaces

---

## Contact & Support

For issues or questions:
- Check existing documentation
- Review example scripts in `sota/examples/`
- Compare with OnMAR/OffMAR in `COMPARISON_ALL_APPROACHES.md`
