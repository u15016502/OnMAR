# OnMAR vs OffMAR: Complete Comparison Guide

This guide helps you choose between OnMAR (Online Meta-learning for AutoML in Real-time) and OffMAR (Offline Meta-learning for AutoML in Real-time) based on Chapter 8 of the thesis.

## Quick Decision Matrix

| Scenario | Recommended Approach | Why |
|----------|---------------------|-----|
| **First time experimenting** | OnMAR with kNN | Simplest, often best performance (Table 8.2) |
| **Production deployment (stable data)** | OffMAR with RF | Reproducible, reusable meta-learner |
| **Production deployment (changing data)** | OnMAR with XGBoost | Adapts to distribution shifts |
| **Limited timesteps (N < 40)** | OffMAR | Needs fewer timesteps to be effective |
| **Many timesteps (N ≥ 60)** | OnMAR | Has time to learn and adapt |
| **Want maximum performance** | OnMAR with kNN | Ranks #1 most often (Table 8.2) |
| **Want computational savings** | OnMAR | Reuses designs, fewer algorithm calls |
| **Need reproducibility** | OffMAR | Same meta-learner → same results |
| **Research/experimentation** | Both (compare!) | Best practice for papers |

## Detailed Comparison

### 1. Meta-Learning Approach

| Aspect | OnMAR | OffMAR |
|--------|-------|--------|
| **Learning Type** | **Online** (continuous) | **Offline** (batch) |
| **When learning happens** | During execution (every timestep after θt) | Before deployment (Phase 1 only) |
| **Knowledge repository** | Continuously grows and updates | Fixed after Phase 1 |
| **Adaptability** | **High** - adapts to changes | **Low** - fixed after training |
| **Meta-learner retraining** | Every timestep (after θt) | Once (end of Phase 1) |

**Winner**: **OnMAR** for adaptability, **OffMAR** for stability

---

### 2. What Does the Meta-Learner Predict?

| Aspect | OnMAR | OffMAR |
|--------|-------|--------|
| **Prediction target** | **Accuracy/Performance** | **Design itself** |
| **Input** | Meta-features + Current design | Meta-features only |
| **Output** | Scalar (predicted accuracy) | Design vector |
| **Decision** | Reuse OR create new | Direct design prediction |
| **Flexibility** | **High** (binary decision) | **Lower** (must use predicted design) |

**Winner**: **OnMAR** - more flexible decision-making

---

### 3. Computational Cost

| Phase | OnMAR | OffMAR |
|-------|-------|--------|
| **Phase 1 (t < θt)** | Run design algorithm + train meta-learner | Run design algorithm (full N timesteps) |
| **Phase 2 (t ≥ θt)** | Conditionally run design algorithm + re-train meta-learner | Only meta-learner inference (fast!) |
| **Meta-learner training** | Incremental (cheap for kNN, moderate for RF/XGBoost) | One-time (can be expensive) |
| **Design algorithm calls** | ~40-60% of timesteps | 100% in Phase 1, 0% in Phase 2 |

**Example** (N=50, θt=25, θp=0.85):
- **OnMAR**: 25 calls (Phase 1) + ~10 calls (Phase 2) = **35 calls total** (70%)
- **OffMAR Phase 1**: **50 calls** (100%)
- **OffMAR Phase 2**: **0 calls** (uses meta-learner)

**Winner**: **OnMAR** for single run, **OffMAR** if reusing across multiple runs

---

### 4. Performance (from Table 8.2)

#### CNN Configuration

| Meta-Learner | OnMAR Rank | OffMAR Rank | Winner |
|--------------|------------|-------------|--------|
| kNN | **1.6** ✓ | 2.0 | OnMAR |
| RF | **2.0** ✓ | 2.0 | Tie |
| XGBoost | **1.1** ✓ | 2.0 | OnMAR |

#### Segmentation Composition

| Meta-Learner | OnMAR Rank | OffMAR Rank | Winner |
|--------------|------------|-------------|--------|
| kNN | **1.0** ✓ | 3.0 | OnMAR |
| RF | **2.0** ✓ | 4.0 | OnMAR |
| XGBoost | **3.0** ✓ | 4.0 | OnMAR |

#### Fuzzy ART Generation

| Meta-Learner | OnMAR Rank | OffMAR Rank | Winner |
|--------------|------------|-------------|--------|
| kNN | **1.0** ✓ | 3.0 | OnMAR |
| RF | **2.0** = | 2.0 | Tie |
| XGBoost | **1.0** ✓ | 3.0 | OnMAR |

**Overall Winner**: **OnMAR** (wins or ties in almost all cases)

---

### 5. Implementation Complexity

| Aspect | OnMAR | OffMAR |
|--------|-------|--------|
| **Number of phases** | 1 (two modes) | 2 (separate phases) |
| **Code complexity** | Moderate | Moderate |
| **Hyperparameters** | θt, θp | θp |
| **Tuning difficulty** | Moderate (need to tune θp) | Easy (θp usually 0.85) |
| **Debugging** | Moderate (online behavior) | Easier (offline = reproducible) |

**Winner**: **OffMAR** - simpler to debug

---

### 6. Use Cases

#### When to Use OnMAR

✅ **Best for:**
- **Dynamic environments** where data distribution changes
- **Long runs** (N ≥ 50-60 timesteps)
- **Maximum performance** as primary goal
- **Single dataset** experiments
- **Adaptive systems** that need to respond to changes

✅ **Examples:**
- Training CNNs where later epochs have different characteristics
- Evolving segmentation algorithms where data complexity increases
- Generating choice functions for changing text corpora

#### When to Use OffMAR

✅ **Best for:**
- **Stable environments** where data doesn't change
- **Short runs** (N < 40 timesteps)
- **Production deployment** with reproducibility requirements
- **Multiple datasets** (train once on one dataset, apply to others)
- **Rapid inference** without design algorithm

✅ **Examples:**
- Production ML pipeline with fixed dataset
- Benchmarking on standardized datasets
- Transfer learning across similar datasets
- Real-time inference without computation budget for design algorithm

---

### 7. Hyperparameter Tuning

#### OnMAR Hyperparameters

| Parameter | Default | Range | Tuning Guidance |
|-----------|---------|-------|-----------------|
| **θt** | N/2 | [N/4, 3N/4] | More training data → larger θt |
| **θp** | 0.85 | [0.70, 0.95] | More reuse → lower θp |

**Tuning strategy:**
1. Run with default θt=N/2, θp=0.85
2. Check Phase 1 performance statistics (mean, std)
3. Set θp = mean(Phase1) - 0.5*std(Phase1)
4. If too many design calls → decrease θp
5. If performance degrading → increase θp

#### OffMAR Hyperparameters

| Parameter | Default | Range | Tuning Guidance |
|-----------|---------|-------|-----------------|
| **θp** | 0.85 | [0.70, 0.95] | Pruning threshold for knowledge repository |

**Tuning strategy:**
1. Run Phase 1 with θp=0.85
2. Check how many designs are pruned
3. If too few remain (< 20%) → decrease θp
4. If too many poor designs remain → increase θp
5. Analyze Phase 2 performance vs Phase 1 best

---

### 8. Meta-Learner Selection

#### For OnMAR (from Table 8.2)

**Recommendation by application:**
- **CNN Configuration**: kNN or XGBoost
- **Segmentation**: kNN
- **Fuzzy ART**: kNN or XGBoost

**General rule**: **kNN is almost always best for OnMAR**

#### For OffMAR (from Table 8.2)

**Recommendation by application:**
- **CNN Configuration**: RF or XGBoost
- **Segmentation**: RF
- **Fuzzy ART**: kNN or RF

**General rule**: **RF is most reliable for OffMAR**

---

### 9. Advantages & Disadvantages

#### OnMAR

**Advantages** ✅
- **Best performance** (Table 8.2)
- **Adapts online** to data changes
- **Computational savings** vs. always running design algorithm
- **Single phase** (simpler workflow)
- **Flexible** (can choose to reuse or create)

**Disadvantages** ❌
- Needs **sufficient timesteps** (N ≥ 40-60)
- **Non-deterministic** (different runs may differ)
- **Memory grows** with knowledge repository
- **Re-training overhead** each timestep
- Cannot **reuse meta-learner** across datasets

#### OffMAR

**Advantages** ✅
- Works with **fewer timesteps** (N ≥ 20-30)
- **Reproducible** (same meta-learner → same results)
- **Reusable** (train once, apply many times)
- **No online overhead** in Phase 2
- **Transfer learning** potential across datasets

**Disadvantages** ❌
- **Expensive Phase 1** (runs design algorithm for all N timesteps)
- **Cannot adapt** to changes after Phase 1
- **Less flexible** (must use predicted design)
- **Lower performance** in many cases (Table 8.2)
- **Two-phase workflow** (more complex)

---

### 10. Practical Recommendations

#### For Research Papers

**Best practice**: Implement and compare both!

```python
# Run OnMAR
onmar_results = run_onmar(dataset, meta_learner='knn', timesteps=50)

# Run OffMAR
offmar_results = run_offmar(dataset, meta_learner='rf', timesteps=50)

# Compare
compare_results(onmar_results, offmar_results)
```

**Report**:
- Performance metrics (best, final, test)
- Computational efficiency (design algorithm calls, runtime)
- Statistical significance tests

#### For Production

**If data is stable:**
1. Use **OffMAR** in production
2. Re-run Phase 1 periodically (e.g., monthly)
3. Deploy Phase 2 for fast inference

**If data changes:**
1. Use **OnMAR** for adaptability
2. Monitor θp effectiveness
3. Optionally adjust θp dynamically

---

### 11. Common Pitfalls

#### OnMAR Pitfalls

❌ **Setting θt too small**
- Meta-learner doesn't have enough training data
- Poor predictions in Phase 2
- **Solution**: Keep θt = N/2 or larger

❌ **Setting θp too high**
- Meta-learner almost never predicts good enough performance
- Effectively disables design reuse
- **Solution**: Start with 0.85, decrease if needed

❌ **Not re-training meta-learner**
- Knowledge repository grows but meta-learner doesn't learn
- **Solution**: Ensure `_train_meta_learner()` is called each timestep

#### OffMAR Pitfalls

❌ **Not pruning knowledge repository**
- Meta-learner learns from poor designs
- Poor Phase 2 performance
- **Solution**: Set appropriate θp threshold

❌ **Using wrong meta-learner for prediction**
- OffMAR predicts **designs** not accuracy
- Using regression instead of multi-output prediction
- **Solution**: Ensure meta-learner outputs design vector

❌ **Insufficient Phase 1 data**
- Not enough timesteps to learn meaningful patterns
- **Solution**: Use N ≥ 30-40 timesteps in Phase 1

---

### 12. Example Scenarios

#### Scenario 1: Academic Research

**Goal**: Compare meta-learning approaches for AutoML paper

**Recommendation**: **Both OnMAR and OffMAR**

```python
# Test all combinations
for approach in ['onmar', 'offmar']:
    for meta_learner in ['knn', 'rf', 'xgboost']:
        results = run_experiment(
            approach=approach,
            meta_learner=meta_learner,
            dataset='mnist',
            timesteps=60
        )
        save_results(results)

# Statistical analysis
compare_approaches(all_results)
```

#### Scenario 2: Industry Production System

**Goal**: Deploy AutoML system for stable image classification

**Recommendation**: **OffMAR with RF**

**Workflow**:
1. **Phase 1** (offline, monthly):
   - Run on representative dataset
   - Train meta-learner
   - Save model

2. **Phase 2** (online, real-time):
   - Load trained meta-learner
   - Use for fast design prediction
   - No design algorithm needed

#### Scenario 3: Adaptive Robotics System

**Goal**: Robot learning tasks with changing environments

**Recommendation**: **OnMAR with kNN**

**Rationale**:
- Environment changes → need online adaptation
- kNN updates instantly (just add to repository)
- Can adjust θp based on robot's performance

---

### 13. Code Comparison

#### OnMAR Example

```python
from OnMAR.accuracy_prediction.onmar import OnMARAccuracyPrediction

onmar = OnMARAccuracyPrediction(
    application=cnn_app,
    meta_learner_type='knn',
    theta_t=None,  # Auto-set to N/2
    theta_p=0.85,
    meta_learner_params={'k': 5}
)

# Single method call
results = onmar.run_onmar(
    dataset_name='mnist',
    timesteps=50
)

# Key metrics
print(f"Design reuse: {results['design_reuse_percentage']}%")
print(f"Best performance: {results['best_performance']}")
```

#### OffMAR Example

```python
from OffMAR.design_prediction.offmar import OffMARDesignPrediction

offmar = OffMARDesignPrediction(
    application=cnn_app,
    meta_learner_type='rf',
    theta_p=0.85,
    meta_learner_params={'n_estimators': 100}
)

# Two-phase workflow
phase_1 = offmar.phase_1_collect_data(
    dataset_name='mnist',
    timesteps=50
)

phase_2 = offmar.phase_2_predict_designs(
    dataset_name='mnist',
    timesteps=50
)

# Key metrics
print(f"Samples pruned: {phase_1['samples_collected'] - phase_1['samples_after_pruning']}")
print(f"Final performance: {phase_2['test_results']['test_performance']}")
```

---

## Summary Recommendations

### 🥇 Overall Best: **OnMAR with kNN**

**Why**: Best performance in most cases (Table 8.2), adaptive, computationally efficient

**Use when**: You have ≥50 timesteps and want maximum performance

### 🥈 Production-Ready: **OffMAR with RF**

**Why**: Reproducible, reusable, stable

**Use when**: You need reproducibility and can afford expensive Phase 1

### 🥉 Quick Start: **OnMAR with kNN (θp=0.85, θt=N/2)**

**Why**: Simplest to configure, often works best out-of-the-box

**Use when**: First time trying meta-learning for AutoML

---

## References

- Chapter 8: Meta-learning for Real-time Design
- Table 8.2: Average ranking of approaches per application
- Algorithm 9: OnMAR pseudo-code
- Section 8.2.1: OnMAR approach description
- Section 8.2.2: OffMAR approach description
