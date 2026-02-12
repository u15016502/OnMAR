# Compare all approaches on same dataset
from sota.autosklearn_wrapper import AutoSklearnWrapper
from OnMAR.accuracy_prediction.onmar import OnMARAccuracyPrediction
from OffMAR.design_prediction.offmar import OffMARDesignPrediction

app = CNNConfigurationApplication(dataset_name='mnist', random_seed=42)

# Auto-sklearn (SOTA)
as_wrapper = AutoSklearnWrapper(app, time_budget=3600)
as_results = as_wrapper.run_optimization('mnist', max_evaluations=50)

# OnMAR
onmar = OnMARAccuracyPrediction(app, meta_learner_type='knn', theta_p=0.85)
onmar_results = onmar.run_onmar('mnist', timesteps=50)

# OffMAR
offmar = OffMARDesignPrediction(app, meta_learner_type='rf', theta_p=0.85)
offmar_results = offmar.run_full_offmar('mnist', timesteps=50)

# Compare performance
print(f"Auto-sklearn: {as_results['best_performance']:.4f}")
print(f"OnMAR:        {onmar_results['best_performance']:.4f}")
print(f"OffMAR:       {offmar_results['phase_2']['best_performance']:.4f}")


All optimizations are enabled by default with sensible settings:


# Fuzzy ART - parallel GE
app = FuzzyARTGenerationApplication(
    dataset_name='enron',
    n_jobs=8  # Optional: defaults to CPU count - 1
)

# CNN - multi-GPU
app = CNNConfigurationApplication(
    dataset_name='mnist',
    use_data_parallel=True  # Default: True if multiple GPUs available
)

# OffMAR - parallel Phase 1
offmar = OffMARDesignPrediction(
    application=app,
    n_jobs=8  # Optional: defaults to CPU count - 1
)