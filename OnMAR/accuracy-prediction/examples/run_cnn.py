"""
Example script for running OnMAR Accuracy Prediction with CNN Configuration Application.

This demonstrates how to use OnMAR with the CNN configuration application
across all three meta-learners (kNN, RF, XGBoost).
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from OnMAR.accuracy_prediction.onmar import OnMARAccuracyPrediction
from OnMAR.accuracy_prediction.config import CNN_CONFIG
from applications.configuration.cnn.cnn_application import CNNConfigurationApplication
import matplotlib.pyplot as plt


def run_onmar_cnn_example(
    dataset_name: str = 'mnist',
    meta_learner_type: str = 'knn',
    timesteps: int = 50,
    save_results: bool = True
):
    """
    Run OnMAR with CNN configuration application.

    Args:
        dataset_name: Dataset to use ('mnist', 'cifar-10', 'fashion-mnist', etc.)
        meta_learner_type: Meta-learner to use ('knn', 'rf', 'xgboost')
        timesteps: Number of training epochs
        save_results: Whether to save results and plots

    Returns:
        Results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Running OnMAR Accuracy Prediction with CNN Configuration")
    print(f"Dataset: {dataset_name}")
    print(f"Meta-learner: {meta_learner_type.upper()}")
    print(f"Timesteps: {timesteps}")
    print(f"{'='*60}\n")

    # Initialize CNN application
    cnn_app = CNNConfigurationApplication(
        dataset_name=dataset_name,
        random_seed=42
    )

    # Get configuration for this meta-learner
    config = CNN_CONFIG[meta_learner_type]

    # Initialize OnMAR
    onmar = OnMARAccuracyPrediction(
        application=cnn_app,
        meta_learner_type=config.meta_learner_type,
        theta_t=config.theta_t,  # None means N/2
        theta_p=config.theta_p,
        meta_learner_params=config.meta_learner_params
    )

    # Run OnMAR
    results = onmar.run_onmar(
        dataset_name=dataset_name,
        timesteps=timesteps,
        initial_design=None  # Use random initialization
    )

    # Print results summary
    print(f"\n{'='*60}")
    print("Results Summary:")
    print(f"{'='*60}")
    print(f"Total runtime: {results['total_time']:.2f} seconds")
    print(f"Best performance: {results['best_performance']:.4f}")
    print(f"Final performance: {results['final_performance']:.4f}")
    print(f"Test performance: {results['test_results'].get('test_performance', 'N/A')}")
    print(f"\nEfficiency Metrics:")
    print(f"  Design algorithm calls: {results['num_design_algorithm_calls']} ({results['design_algorithm_percentage']:.1f}%)")
    print(f"  Design reuses: {results['num_design_reuses']} ({results['design_reuse_percentage']:.1f}%)")
    print(f"  Knowledge repository size: {results['knowledge_repository_size']}")
    print(f"{'='*60}\n")

    # Plot performance over time
    if save_results:
        plot_results(results, dataset_name, meta_learner_type)

    # Save model
    if save_results:
        model_path = f"OnMAR/accuracy-prediction/models/cnn_{dataset_name}_{meta_learner_type}.pkl"
        onmar.save_model(model_path)
        print(f"Model saved to: {model_path}")

    return results


def plot_results(results: dict, dataset_name: str, meta_learner_type: str):
    """
    Plot OnMAR results.

    Args:
        results: Results dictionary from OnMAR
        dataset_name: Dataset name
        meta_learner_type: Meta-learner type
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Performance over time
    timesteps = list(range(1, len(results['performance_history']) + 1))
    performances = results['performance_history']
    theta_t = results['theta_t']

    ax1.plot(timesteps, performances, marker='o', markersize=3, linewidth=2)
    ax1.axvline(x=theta_t, color='r', linestyle='--', label=f'θt = {theta_t} (start using meta-learner)')
    ax1.axhline(y=results['theta_p'], color='g', linestyle='--', label=f'θp = {results["theta_p"]} (performance threshold)')
    ax1.set_xlabel('Timestep (Epoch)', fontsize=12)
    ax1.set_ylabel('Performance (Accuracy)', fontsize=12)
    ax1.set_title(f'OnMAR Performance Over Time\n{dataset_name.upper()} - {meta_learner_type.upper()}', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Design algorithm usage
    phase1_calls = min(theta_t, len(timesteps))
    phase2_calls = results['num_design_algorithm_calls'] - phase1_calls
    phase2_reuses = results['num_design_reuses']

    categories = ['Phase 1\n(Before θt)', 'Phase 2\n(After θt)']
    algorithm_calls = [phase1_calls, phase2_calls]
    reuses = [0, phase2_reuses]

    x = range(len(categories))
    width = 0.35

    ax2.bar([i - width/2 for i in x], algorithm_calls, width, label='Design Algorithm Calls', color='orange')
    ax2.bar([i + width/2 for i in x], reuses, width, label='Design Reuses', color='green')

    ax2.set_xlabel('Phase', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title(f'Design Algorithm Usage vs. Reuse\n{dataset_name.upper()} - {meta_learner_type.upper()}', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save plot
    plot_path = f"OnMAR/accuracy-prediction/results/cnn_{dataset_name}_{meta_learner_type}.png"
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")

    plt.close()


def compare_meta_learners(dataset_name: str = 'mnist', timesteps: int = 50):
    """
    Compare all three meta-learners on the same dataset.

    Args:
        dataset_name: Dataset to use
        timesteps: Number of timesteps
    """
    meta_learners = ['knn', 'rf', 'xgboost']
    all_results = {}

    print(f"\n{'='*60}")
    print(f"Comparing Meta-Learners on {dataset_name.upper()}")
    print(f"{'='*60}\n")

    for meta_learner in meta_learners:
        print(f"\n--- Running with {meta_learner.upper()} ---")
        try:
            results = run_onmar_cnn_example(
                dataset_name=dataset_name,
                meta_learner_type=meta_learner,
                timesteps=timesteps,
                save_results=True
            )
            all_results[meta_learner] = results
        except Exception as e:
            print(f"Error with {meta_learner}: {e}")
            continue

    # Print comparison
    print(f"\n{'='*60}")
    print("Comparison Summary:")
    print(f"{'='*60}")
    print(f"{'Meta-Learner':<15} {'Best Perf':<12} {'Final Perf':<12} {'Test Perf':<12} {'Reuse %':<12} {'Runtime (s)':<12}")
    print("-" * 80)

    for meta_learner, results in all_results.items():
        print(f"{meta_learner.upper():<15} "
              f"{results['best_performance']:<12.4f} "
              f"{results['final_performance']:<12.4f} "
              f"{results['test_results'].get('test_performance', 0.0):<12.4f} "
              f"{results['design_reuse_percentage']:<12.1f} "
              f"{results['total_time']:<12.2f}")

    print(f"{'='*60}\n")

    return all_results


if __name__ == "__main__":
    # Example 1: Run with single meta-learner
    results = run_onmar_cnn_example(
        dataset_name='mnist',
        meta_learner_type='knn',
        timesteps=20,  # Reduced for quick testing
        save_results=True
    )

    # Example 2: Compare all meta-learners
    # comparison = compare_meta_learners(
    #     dataset_name='mnist',
    #     timesteps=20
    # )
