# Command Reference for All Experiment Configurations

## Individual Commands for Each Configuration

Here are the exact terminal commands for each configuration from your table:

### CNN Configuration

```bash
# 1. CNN - OnMAR (Design) - All datasets - All meta-learners
python runner.py --application cnn --all-datasets --techniques onmar-design --timesteps 50 --runs 30

# 2. CNN - OnMAR (Accuracy) - All datasets - All meta-learners
python runner.py --application cnn --all-datasets --techniques onmar-accuracy --timesteps 50 --runs 30

# 3. CNN - OffMAR (Design) - All datasets - All meta-learners
python runner.py --application cnn --all-datasets --techniques offmar-design --timesteps 50 --runs 30

# 4. CNN - OffMAR (Accuracy) - All datasets - All meta-learners
python runner.py --application cnn --all-datasets --techniques offmar-accuracy --timesteps 50 --runs 30

# 5. CNN - AutoSklearn - All datasets
python runner.py --application cnn --all-datasets --techniques autosklearn --timesteps 50 --runs 30
```

### Segmentation Composition

```bash
# 6. Segmentation - OnMAR (Design) - All datasets - All meta-learners
python runner.py --application segmentation --all-datasets --techniques onmar-design --timesteps 60 --runs 30

# 7. Segmentation - OnMAR (Accuracy) - All datasets - All meta-learners
python runner.py --application segmentation --all-datasets --techniques onmar-accuracy --timesteps 60 --runs 30

# 8. Segmentation - OffMAR (Design) - All datasets - All meta-learners
python runner.py --application segmentation --all-datasets --techniques offmar-design --timesteps 60 --runs 30

# 9. Segmentation - OffMAR (Accuracy) - All datasets - All meta-learners
python runner.py --application segmentation --all-datasets --techniques offmar-accuracy --timesteps 60 --runs 30

# 10. Segmentation - AutoSklearn - All datasets
python runner.py --application segmentation --all-datasets --techniques autosklearn --timesteps 60 --runs 30
```

### FuzzyART Generation

```bash
# 11. FuzzyART - OnMAR (Design) - All datasets - All meta-learners
python runner.py --application fuzzyart --all-datasets --techniques onmar-design --timesteps 60 --runs 30

# 12. FuzzyART - OnMAR (Accuracy) - All datasets - All meta-learners
python runner.py --application fuzzyart --all-datasets --techniques onmar-accuracy --timesteps 60 --runs 30

# 13. FuzzyART - OffMAR (Design) - All datasets - All meta-learners
python runner.py --application fuzzyart --all-datasets --techniques offmar-design --timesteps 60 --runs 30

# 14. FuzzyART - OffMAR (Accuracy) - All datasets - All meta-learners
python runner.py --application fuzzyart --all-datasets --techniques offmar-accuracy --timesteps 60 --runs 30

# 15. FuzzyART - AutoSklearn - All datasets
python runner.py --application fuzzyart --all-datasets --techniques autosklearn --timesteps 60 --runs 30
```

---

## Even More Granular Control

### Run Specific Meta-Learner Only

```bash
# OnMAR Design with only kNN on CNN
python runner.py --application cnn --all-datasets --techniques onmar-design --meta-learners knn --timesteps 50 --runs 30

# OnMAR Design with only RF on CNN
python runner.py --application cnn --all-datasets --techniques onmar-design --meta-learners rf --timesteps 50 --runs 30

# OnMAR Design with only XGBoost on CNN
python runner.py --application cnn --all-datasets --techniques onmar-design --meta-learners xgboost --timesteps 50 --runs 30
```

### Run Multiple Techniques Together

```bash
# Run both OnMAR variants (Design + Accuracy) with all meta-learners
python runner.py --application cnn --all-datasets --techniques onmar-design onmar-accuracy --timesteps 50 --runs 30

# Run all OnMAR and OffMAR Design variants
python runner.py --application cnn --all-datasets --techniques onmar-design offmar-design --timesteps 50 --runs 30

# Run everything except AutoSklearn
python runner.py --application cnn --all-datasets --techniques onmar-design onmar-accuracy offmar-design offmar-accuracy --timesteps 50 --runs 30
```

### Run Specific Dataset Only

```bash
# Test OnMAR Design on MNIST only
python runner.py --application cnn --datasets mnist --techniques onmar-design --timesteps 50 --runs 30

# Test OnMAR Design on CIFAR-10 and CIFAR-100
python runner.py --application cnn --datasets cifar-10 cifar-100 --techniques onmar-design --timesteps 50 --runs 30
```

---

## Quick Reference Table

| Flag | Options | Description |
|------|---------|-------------|
| `--application` | `cnn`, `segmentation`, `fuzzyart` | Which application to test |
| `--datasets` | Dataset names (space-separated) | Specific datasets to test |
| `--all-datasets` | (flag) | Test all datasets for the application |
| `--techniques` | `onmar-accuracy`, `onmar-design`, `offmar-accuracy`, `offmar-design`, `autosklearn` | Which techniques to run |
| `--meta-learners` | `knn`, `rf`, `xgboost` | Which meta-learners to use |
| `--timesteps` | Integer (default: 50) | Number of training epochs/generations |
| `--runs` | Integer (default: 30) | Number of independent runs per config |
| `--autosklearn-time` | Integer (default: 3600) | Time budget for AutoSklearn in seconds |
| `--quick-test` | (flag) | Run quick test (1 dataset, 1 run, 10 timesteps) |

---

## Parallel Execution Strategy

To speed up experiments, run different commands in parallel on different machines or terminals:

### Terminal 1
```bash
python runner.py --application cnn --all-datasets --techniques onmar-design --timesteps 50 --runs 30
```

### Terminal 2
```bash
python runner.py --application cnn --all-datasets --techniques onmar-accuracy --timesteps 50 --runs 30
```

### Terminal 3
```bash
python runner.py --application cnn --all-datasets --techniques offmar-design --timesteps 50 --runs 30
```

### Terminal 4
```bash
python runner.py --application cnn --all-datasets --techniques offmar-accuracy --timesteps 50 --runs 30
```

### Terminal 5
```bash
python runner.py --application cnn --all-datasets --techniques autosklearn --timesteps 50 --runs 30
```

---

## Testing Before Full Runs

Always test with `--quick-test` first to catch errors:

```bash
# Quick test for CNN OnMAR Design
python runner.py --application cnn --techniques onmar-design --quick-test

# Quick test for Segmentation OffMAR Accuracy
python runner.py --application segmentation --techniques offmar-accuracy --quick-test

# Quick test for FuzzyART AutoSklearn
python runner.py --application fuzzyart --techniques autosklearn --quick-test
```

---

## Copy-Paste Ready: All 15 Commands

```bash
# CNN
python runner.py --application cnn --all-datasets --techniques onmar-design --timesteps 50 --runs 30
python runner.py --application cnn --all-datasets --techniques onmar-accuracy --timesteps 50 --runs 30
python runner.py --application cnn --all-datasets --techniques offmar-design --timesteps 50 --runs 30
python runner.py --application cnn --all-datasets --techniques offmar-accuracy --timesteps 50 --runs 30
python runner.py --application cnn --all-datasets --techniques autosklearn --timesteps 50 --runs 30

# Segmentation
python runner.py --application segmentation --all-datasets --techniques onmar-design --timesteps 60 --runs 30
python runner.py --application segmentation --all-datasets --techniques onmar-accuracy --timesteps 60 --runs 30
python runner.py --application segmentation --all-datasets --techniques offmar-design --timesteps 60 --runs 30
python runner.py --application segmentation --all-datasets --techniques offmar-accuracy --timesteps 60 --runs 30
python runner.py --application segmentation --all-datasets --techniques autosklearn --timesteps 60 --runs 30

# FuzzyART
python runner.py --application fuzzyart --all-datasets --techniques onmar-design --timesteps 60 --runs 30
python runner.py --application fuzzyart --all-datasets --techniques onmar-accuracy --timesteps 60 --runs 30
python runner.py --application fuzzyart --all-datasets --techniques offmar-design --timesteps 60 --runs 30
python runner.py --application fuzzyart --all-datasets --techniques offmar-accuracy --timesteps 60 --runs 30
python runner.py --application fuzzyart --all-datasets --techniques autosklearn --timesteps 60 --runs 30
```

---

**Created**: 2026-01-11
**Status**: Ready to use!
