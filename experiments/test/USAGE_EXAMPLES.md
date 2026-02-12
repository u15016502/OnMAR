# Usage Examples

Comprehensive examples for running experiments and analyzing results.

## Table of Contents
1. [Basic Usage](#basic-usage)
2. [Advanced Usage](#advanced-usage)
3. [Analysis Examples](#analysis-examples)
4. [Troubleshooting](#troubleshooting)

---

## Basic Usage

### Example 1: Quick Sanity Check

Before running long experiments, verify everything works:

```bash
# Step 1: Verify setup
python verify_setup.py

# Step 2: Quick test (5 runs, 20 timesteps, MNIST only)
python runner.py --quick-test

# Step 3: Check results
ls experiments/test/results/
```

Expected output:
- Runtime: ~30 minutes
- Files: individual_runs/, summary_*.csv, all_results_*.json

### Example 2: Single Dataset, Full Experiment

Run complete experiment on MNIST as in thesis:

```bash
# Run 30 independent runs with 50 timesteps
python runner.py \
    --datasets mnist \
    --timesteps 50 \
    --runs 30 \
    --output-dir experiments/test/results/mnist_experiment

# This will take approximately 6-8 hours
```

### Example 3: Multiple Datasets

Run on MNIST and CIFAR-10:

```bash
python runner.py \
    --datasets mnist cifar-10 \
    --timesteps 50 \
    --runs 30

# This will take approximately 12-16 hours
```

### Example 4: Full Thesis Replication

Replicate all results from Chapter 8:

```bash
python runner.py \
    --all-datasets \
    --timesteps 50 \
    --runs 30 \
    --autosklearn-time 3600

# This will take approximately 24-36 hours
# Recommended to run overnight or over weekend
```

---

## Advanced Usage

### Example 5: Custom Parameters

Fine-tune experimental parameters:

```bash
python runner.py \
    --datasets mnist fashion-mnist \
    --timesteps 100 \              # More timesteps for convergence
    --runs 50 \                     # More runs for statistics
    --autosklearn-time 7200 \       # 2 hours for AutoSklearn
    --output-dir experiments/test/results/custom_run
```

### Example 6: Resume After Crash

If experiment crashes, individual runs are saved. To continue:

```bash
# Check what completed
ls experiments/test/results/individual_runs/ | wc -l

# Run again with same parameters
# Already-completed runs won't be re-run
python runner.py --datasets mnist --runs 30

# Or start fresh with new output dir
python runner.py --datasets mnist --runs 30 --output-dir experiments/test/results/run2
```

### Example 7: Parallel Execution

Run multiple datasets in parallel (separate terminals):

```bash
# Terminal 1
python runner.py --datasets mnist --runs 30 --output-dir results/mnist

# Terminal 2
python runner.py --datasets fashion-mnist --runs 30 --output-dir results/fashion

# Terminal 3
python runner.py --datasets cifar-10 --runs 30 --output-dir results/cifar10

# Terminal 4
python runner.py --datasets cifar-100 --runs 30 --output-dir results/cifar100
```

---

## Analysis Examples

### Example 8: Generate Summary Report

After experiments complete:

```bash
# Generate comprehensive report
python results_analyzer.py \
    --results-dir experiments/test/results \
    --generate-report

# Output: experiments/test/results/analysis_report.txt
```

### Example 9: Generate Box Plots

Create publication-quality figures:

```bash
# Generate all box plots
python results_analyzer.py \
    --results-dir experiments/test/results \
    --generate-plots

# Output: experiments/test/results/plots/*.png
```

### Example 10: Generate Publication Tables

Create tables in multiple formats:

```bash
# Find your results file
ls experiments/test/results/all_results_*.json

# Generate tables (replace with actual filename)
python generate_tables.py \
    --results-file experiments/test/results/all_results_20240115_143022.json \
    --formats txt csv latex md

# Output: experiments/test/results/tables/
```

### Example 11: Custom Analysis in Python

```python
from results_analyzer import ResultsAnalyzer

# Load results
analyzer = ResultsAnalyzer('experiments/test/results')
analyzer.load_results()

# Get ranking table for MNIST
ranking = analyzer.generate_ranking_table('mnist')
print("\nTop 5 Approaches for MNIST:")
print(ranking.head(5))

# Compare OnMAR vs OffMAR
for ml in ['knn', 'rf', 'xgboost']:
    stat, p, interp = analyzer.mann_whitney_comparison(
        dataset='mnist',
        approach1=f'OnMAR-Acc-{ml}',
        approach2=f'OffMAR-Acc-{ml}'
    )
    print(f"\n{ml.upper()}: {interp}")

# Calculate effect sizes
effect_sizes = analyzer.calculate_effect_sizes('mnist')
large_effects = effect_sizes[effect_sizes['Effect Size'] == 'Large']
print(f"\nFound {len(large_effects)} large effect sizes")
print(large_effects)

# Generate custom plot
analyzer.generate_boxplots(
    dataset='mnist',
    metric='test_performance',
    save_path='custom_mnist_plot.png'
)
```

### Example 12: Compare Specific Approaches

```python
import json
import numpy as np

# Load results
with open('experiments/test/results/all_results_TIMESTAMP.json', 'r') as f:
    results = json.load(f)

# Extract OnMAR-kNN vs AutoSklearn for MNIST
onmar_runs = results['mnist']['OnMAR-Acc-knn']
autosklearn_runs = results['mnist']['AutoSklearn']

# Get test performances
onmar_perf = [r['test_results']['test_performance'] for r in onmar_runs]
auto_perf = [r['test_results']['test_performance'] for r in autosklearn_runs]

# Get runtimes
onmar_time = [r['total_time']/60 for r in onmar_runs]  # minutes
auto_time = [r['total_time']/60 for r in autosklearn_runs]

# Compare
print(f"OnMAR-kNN: {np.mean(onmar_perf):.4f} ± {np.std(onmar_perf):.4f}")
print(f"AutoSklearn: {np.mean(auto_perf):.4f} ± {np.std(auto_perf):.4f}")
print(f"\nOnMAR Runtime: {np.mean(onmar_time):.1f} ± {np.std(onmar_time):.1f} min")
print(f"AutoSklearn Runtime: {np.mean(auto_time):.1f} ± {np.std(auto_time):.1f} min")
print(f"\nSpeedup: {np.mean(auto_time)/np.mean(onmar_time):.2f}x")
```

### Example 13: Extract Data for External Tools

```python
import json
import pandas as pd

# Load results
with open('experiments/test/results/all_results_TIMESTAMP.json', 'r') as f:
    results = json.load(f)

# Flatten to DataFrame for R, Excel, etc.
rows = []
for dataset, approaches in results.items():
    for approach_key, runs in approaches.items():
        for run in runs:
            rows.append({
                'dataset': dataset,
                'approach': run['approach'],
                'meta_learner': run['meta_learner'],
                'run_number': run['run_number'],
                'best_performance': run['best_performance'],
                'final_performance': run['final_performance'],
                'test_performance': run.get('test_results', {}).get('test_performance', 0),
                'runtime_minutes': run['total_time'] / 60,
                'reuse_percentage': run.get('design_reuse_percentage', 0),
                'design_calls': run.get('num_design_algorithm_calls', 0)
            })

df = pd.DataFrame(rows)

# Save for external analysis
df.to_csv('experiments/test/results/flattened_results.csv', index=False)
df.to_excel('experiments/test/results/flattened_results.xlsx', index=False)

print(f"Exported {len(df)} rows to CSV and Excel")
print(f"\nSummary by approach:")
print(df.groupby('approach')['test_performance'].agg(['mean', 'std', 'count']))
```

---

## Troubleshooting

### Issue 1: Out of Memory

**Symptoms:** Process killed, OOM errors

**Solutions:**

```bash
# Option 1: Reduce batch size (edit cnn_application.py)
# Change batch_size from 128 to 64 or 32

# Option 2: Run one dataset at a time
python runner.py --datasets mnist --runs 30
python runner.py --datasets cifar-10 --runs 30  # After first completes

# Option 3: Reduce number of parallel runs
# Run fewer experiments simultaneously
```

### Issue 2: Slow Performance

**Symptoms:** Single run takes >1 hour

**Solutions:**

```bash
# Option 1: Use GPU if available
# Set CUDA_VISIBLE_DEVICES=0 before running

# Option 2: Reduce timesteps for testing
python runner.py --datasets mnist --timesteps 30 --runs 10

# Option 3: Profile to find bottleneck
python -m cProfile -o profile.stats runner.py --quick-test
```

### Issue 3: Import Errors

**Symptoms:** ModuleNotFoundError, ImportError

**Solutions:**

```bash
# Option 1: Reinstall dependencies
pip install -r requirements.txt

# Option 2: Verify PYTHONPATH
export PYTHONPATH="/Users/miagerber/Documents/Journal article experiments:$PYTHONPATH"

# Option 3: Run verification
python verify_setup.py
```

### Issue 4: Missing Results

**Symptoms:** Some approaches/datasets missing from output

**Solutions:**

```python
# Check individual runs directory
import os
runs_dir = 'experiments/test/results/individual_runs'
files = os.listdir(runs_dir)

# Count by approach
from collections import Counter
approaches = [f.split('_')[1] for f in files if f.endswith('.json')]
print(Counter(approaches))

# Identify missing runs
expected = 30  # Number of runs
for dataset in ['mnist', 'fashion-mnist', 'cifar-10', 'cifar-100']:
    for approach in ['OnMAR-AccuracyPrediction', 'OffMAR-AccuracyPrediction']:
        for ml in ['knn', 'rf', 'xgboost']:
            pattern = f"{dataset}_{approach}_{ml}"
            count = len([f for f in files if pattern in f])
            if count < expected:
                print(f"Missing runs for {pattern}: {expected - count}")
```

### Issue 5: Inconsistent Results

**Symptoms:** Results differ from thesis

**Possible causes:**

1. **Random seed variation**: Check that seeds are set correctly
2. **Different data splits**: Verify train/test splits match
3. **Hyperparameter differences**: Compare meta-learner params
4. **Version differences**: Check library versions

**Diagnostic:**

```python
# Compare configurations
from OnMAR.accuracy_prediction.config import CNN_CONFIG

for ml in ['knn', 'rf', 'xgboost']:
    config = CNN_CONFIG[ml]
    print(f"\n{ml.upper()} Config:")
    print(f"  theta_t: {config.theta_t}")
    print(f"  theta_p: {config.theta_p}")
    print(f"  params: {config.meta_learner_params}")
```

---

## Tips and Best Practices

### Tip 1: Monitor Progress

```bash
# In separate terminal, watch log
tail -f experiments/test/results/individual_runs/*.json | grep "best_performance"

# Count completed runs
watch -n 60 'ls experiments/test/results/individual_runs/*.json | wc -l'
```

### Tip 2: Checkpoint Long Runs

```bash
# Use screen or tmux for long runs
screen -S experiments
python runner.py --all-datasets --runs 30
# Ctrl+A, D to detach
# screen -r experiments to reattach
```

### Tip 3: Batch Analysis

```bash
# After each dataset completes, generate interim report
python results_analyzer.py \
    --results-dir experiments/test/results \
    --generate-report
```

### Tip 4: Compare with Thesis

```python
# Load thesis reference values (from Chapter 8, Table 8.2)
thesis_values = {
    'mnist': {
        'OnMAR-Acc-knn': 0.9200,  # Example values
        'OffMAR-Acc-knn': 0.9150,
        'AutoSklearn': 0.9100
    }
}

# Compare with your results
from results_analyzer import ResultsAnalyzer
analyzer = ResultsAnalyzer('experiments/test/results')
analyzer.load_results()

ranking = analyzer.generate_ranking_table('mnist')
for _, row in ranking.iterrows():
    approach_key = f"{row['Approach']}-{row['Meta-Learner']}"
    if approach_key in thesis_values['mnist']:
        diff = row['Mean'] - thesis_values['mnist'][approach_key]
        print(f"{approach_key}: {row['Mean']:.4f} (thesis: {thesis_values['mnist'][approach_key]:.4f}, diff: {diff:+.4f})")
```

---

## Next Steps

After running experiments successfully:

1. **Validate Results**: Compare with thesis Chapter 8 tables
2. **Generate Figures**: Create publication-quality plots
3. **Write Report**: Document findings and interpretations
4. **Archive Data**: Save raw results for reproducibility

For questions or issues, refer to the main README.md or create an issue in the repository.
