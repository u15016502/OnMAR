# CNN Application Test Suite

Comprehensive testing framework for running OnMAR, OffMAR, and AutoSklearn on the CNN configuration application across all datasets.

## Overview

This test suite runs all four variants of meta-learning approaches plus AutoSklearn baseline:

1. **OnMAR - Accuracy Prediction**: Meta-learner predicts accuracy, decides whether to reuse
2. **OnMAR - Design Prediction**: Meta-learner directly predicts CNN designs
3. **OffMAR - Accuracy Prediction**: Two-phase offline approach predicting accuracy
4. **OffMAR - Design Prediction**: Two-phase offline approach predicting designs
5. **AutoSklearn**: Bayesian optimization baseline (SOTA)

Each approach is tested with three meta-learners (kNN, Random Forest, XGBoost) across all CNN datasets (MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100).

## Files

- **`runner.py`**: Main experiment runner
- **`results_analyzer.py`**: Results analysis and visualization utilities
- **`README.md`**: This file

## Quick Start

### 1. Quick Test (Fast, for debugging)

```bash
python runner.py --quick-test
```

This runs:
- 1 dataset (MNIST)
- 5 independent runs
- 20 timesteps
- ~30 minutes total

### 2. Single Dataset (Full experiment)

```bash
python runner.py --datasets mnist --timesteps 50 --runs 30
```

This runs:
- MNIST only
- 30 independent runs (as in thesis)
- 50 timesteps (as in thesis)
- ~6-8 hours

### 3. All Datasets (Complete replication)

```bash
python runner.py --all-datasets --timesteps 50 --runs 30
```

This runs:
- All 4 datasets (MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100)
- 30 independent runs each
- 50 timesteps
- ~24-36 hours total

## Command Line Options

```
--datasets DATASET [DATASET ...]
    Specific datasets to test (default: mnist)
    Options: mnist, fashion-mnist, cifar-10, cifar-100

--all-datasets
    Run on all available datasets

--timesteps N
    Number of timesteps/epochs for MAR approaches (default: 50)

--runs N
    Number of independent runs per configuration (default: 30)

--autosklearn-time SECONDS
    Time budget for AutoSklearn in seconds (default: 3600)

--output-dir PATH
    Output directory for results (default: experiments/test/results)

--quick-test
    Quick test mode: 1 dataset, 5 runs, 20 timesteps
```

## Output Files

Results are saved in `experiments/test/results/` with timestamps:

### Immediate Outputs (saved during execution)

- **`individual_runs/`**: Individual result JSON files for each run
  - Format: `{dataset}_{approach}_{meta_learner}_run{N}.json`
  - Saved immediately to prevent data loss on crashes

### Final Outputs (saved at completion)

- **`summary_{timestamp}.csv`**: Statistical summary of all results
  - Mean ± std for all metrics
  - Formatted for easy import into spreadsheets

- **`thesis_rankings_{timestamp}.txt`**: Rankings in thesis Table 8.2 format
  - Formatted exactly as in Chapter 8, Table 8.2
  - Easy to copy-paste into papers

- **`boxplot_data_{timestamp}.json`**: Raw data for generating box plots
  - All test performance values for each approach
  - Ready for plotting scripts

- **`all_results_{timestamp}.json`**: Complete results dictionary
  - Full raw results for detailed analysis
  - Can be reloaded for post-hoc analysis

## Metrics Logged

For each run, the following metrics are logged (matching thesis Chapter 8):

### Performance Metrics
- **Best Performance**: Highest validation accuracy achieved
- **Final Performance**: Validation accuracy at final timestep
- **Test Performance**: Accuracy on held-out test set

### Efficiency Metrics (MAR approaches only)
- **Design Algorithm Calls**: Number of times design algorithm was run
- **Design Reuses**: Number of times previous design was reused
- **Reuse Percentage**: Percentage of timesteps that reused designs
- **Knowledge Repository Size**: Number of entries in meta-learner training set

### Runtime Metrics
- **Total Time**: Wall-clock time in seconds
- **Runtime (minutes)**: Total time in minutes for reporting

## Analyzing Results

### Generate Comprehensive Report

```bash
python results_analyzer.py --results-dir experiments/test/results --generate-report
```

This creates:
- Ranking tables for each dataset
- Statistical comparisons (Mann-Whitney U tests)
- Best approach identification

### Generate Box Plots

```bash
python results_analyzer.py --results-dir experiments/test/results --generate-plots
```

This creates box plots matching thesis Figure 8.7 format for each dataset.

### Custom Analysis in Python

```python
from results_analyzer import ResultsAnalyzer

# Load results
analyzer = ResultsAnalyzer('experiments/test/results')
analyzer.load_results()

# Generate ranking table for MNIST
ranking_df = analyzer.generate_ranking_table('mnist')
print(ranking_df)

# Statistical comparison
stat, p_value, interpretation = analyzer.mann_whitney_comparison(
    dataset='mnist',
    approach1='OnMAR-Acc-knn',
    approach2='OffMAR-Acc-knn'
)
print(interpretation)

# Calculate effect sizes
effect_sizes = analyzer.calculate_effect_sizes('mnist')
print(effect_sizes)

# Generate box plot
analyzer.generate_boxplots('mnist', save_path='mnist_results.png')
```

## Expected Results

Based on thesis Chapter 8, Table 8.2, expected patterns:

### Performance Ranking (typical)
1. **OnMAR** approaches generally outperform OffMAR
2. **kNN** or **XGBoost** meta-learners typically rank highest
3. **Accuracy prediction** often performs better than design prediction
4. **AutoSklearn** provides strong baseline but higher computational cost

### Runtime Ranking (typical)
1. **OnMAR** - Fastest (selective design algorithm calls)
2. **AutoSklearn** - Medium (full Bayesian optimization)
3. **OffMAR** - Slowest (two full phases)

### Reuse Percentages
- OnMAR: 30-60% design reuse (varies by dataset)
- OffMAR: 40-70% design reuse (typically higher than OnMAR)

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors:
```bash
# Reduce batch size in CNN application
# Or run fewer datasets at once
python runner.py --datasets mnist --runs 30
```

### Long Runtime

For faster iteration during development:
```bash
# Use quick test mode
python runner.py --quick-test

# Or reduce runs/timesteps
python runner.py --datasets mnist --runs 10 --timesteps 30
```

### Missing Dependencies

```bash
pip install -r requirements.txt
```

## Results Validation

To validate your results match the thesis:

1. Check **Table 8.2 rankings**: OnMAR should generally rank 1-2
2. Check **runtime**: OnMAR < AutoSklearn < OffMAR
3. Check **reuse percentage**: Should be 30-70%
4. Check **box plots**: Patterns should match Figure 8.7

## Citation

If using this code, please cite:

```bibtex
@phdthesis{gerber2024thesis,
  title={Meta-Learning for Real-Time AutoML},
  author={Gerber, Mia},
  year={2024},
  school={Your University}
}
```

## Contact

For questions or issues, please contact or open an issue in the repository.
