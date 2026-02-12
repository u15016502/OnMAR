"""
Example script for running OffMAR Accuracy Prediction with CNN Configuration Application.

This demonstrates how to use OffMAR accuracy prediction with the CNN configuration application
across all three meta-learners (kNN, RF, XGBoost).
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from OffMAR.accuracy_prediction.offmar import OffMARAccuracyPrediction
from OffMAR.accuracy_prediction.config import CNN_CONFIG
from applications.configuration.cnn.cnn_application import CNNConfigurationApplication
import matplotlib.pyplot as plt


def run_offmar_cnn_example(
    dataset_name: str = 'mnist',
    meta_learner_type: str = 'rf',
    timesteps: int = 50,
    save_results: bool = True
):
    """
    Run OffMAR accuracy prediction with CNN configuration application.

    Args:
        dataset_name: Dataset to use ('mnist', 'cifar-10', 'fashion-mnist', etc.)
        meta_learner_type: Meta-learner to use ('knn', 'rf', 'xgboost')
        timesteps: Number of training epochs (split between Phase 1 and Phase 2)
        save_results: Whether to save results and plots

    Returns:
        Results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Running OffMAR Accuracy Prediction with CNN Configuration")
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

    # Initialize OffMAR Accuracy Prediction
    offmar = OffMARAccuracyPrediction(
        application=cnn_app,
        meta_learner_type=config.meta_learner_type,
        theta_p=config.theta_p,
        meta_learner_params=config.meta_learner_params
    )

    # Run full OffMAR (both phases)
    results = offmar.run_full_offmar(
        dataset_name=dataset_name,
        timesteps=timesteps,
        initial_design=None
    )

    # Print results summary
    print(f"\n{'='*60}")
    print("Results Summary:")
    print(f"{'='*60}")
    print(f"Total runtime: {results['total_time']:.2f} seconds")
    print(f"\nPhase 1 (Data Collection):")
    print(f"  - Samples collected: {results['phase_1']['samples_collected']}")
    print(f"  - Samples after pruning: {results['phase_1']['samples_after_pruning']}")
    print(f"  - Best performance: {results['phase_1']['best_performance']:.4f}")
    print(f"  - Runtime: {results['phase_1']['phase_1_time']:.2f} seconds")
    print(f"\nPhase 2 (Accuracy Prediction & Reuse):")
    print(f"  - Reuses: {results['phase_2']['num_reuses']} ({results['phase_2']['reuse_percentage']:.1f}%)")
    print(f"  - New designs: {results['phase_2']['num_new_designs']} ({results['phase_2']['new_design_percentage']:.1f}%)")
    print(f"  - Best performance: {results['phase_2']['best_performance']:.4f}")
    print(f"  - Final performance: {results['final_performance']:.4f}")
    print(f"  - Test performance: {results['test_performance']}")
    print(f"  - Runtime: {results['phase_2']['phase_2_time']:.2f} seconds")
    print(f"{'='*60}\n")

    # Plot performance over time
    if save_results:
        plot_results(results, dataset_name, meta_learner_type)

    # Save model
    if save_results:
        model_path = f"OffMAR/accuracy-prediction/models/cnn_{dataset_name}_{meta_learner_type}.pkl"
        offmar.save_model(model_path)
        print(f"Model saved to: {model_path}")

    return results


def plot_results(results: dict, dataset_name: str, meta_learner_type: str):
    """
    Plot OffMAR accuracy prediction results.

    Args:
        results: Results dictionary from OffMAR
        dataset_name: Dataset name
        meta_learner_type: Meta-learner type
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Combine Phase 1 and Phase 2 performance
    phase_1_perf = results['phase_1']['performance_history']
    phase_2_perf = results['phase_2']['performance_history']
    all_performance = phase_1_perf + phase_2_perf
    timesteps = list(range(1, len(all_performance) + 1))
    phase_1_end = len(phase_1_perf)

    # Plot 1: Performance over time
    ax1.plot(timesteps[:phase_1_end], phase_1_perf, marker='o', markersize=3, linewidth=2, label='Phase 1 (Data Collection)', color='blue')
    ax1.plot(timesteps[phase_1_end:], phase_2_perf, marker='s', markersize=3, linewidth=2, label='Phase 2 (Accuracy Prediction)', color='green')
    ax1.axvline(x=phase_1_end, color='r', linestyle='--', label=f'Phase 1→2 transition')
    ax1.axhline(y=results['theta_p'], color='orange', linestyle='--', label=f'θp = {results["theta_p"]} (reuse threshold)')
    ax1.set_xlabel('Timestep (Epoch)', fontsize=12)
    ax1.set_ylabel('Performance (Accuracy)', fontsize=12)
    ax1.set_title(f'OffMAR Accuracy Prediction Performance\n{dataset_name.upper()} - {meta_learner_type.upper()}', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Phase 2 reuse statistics
    num_reuses = results['phase_2']['num_reuses']
    num_new = results['phase_2']['num_new_designs']

    categories = ['Reused Designs', 'New Designs']
    counts = [num_reuses, num_new]
    colors = ['green', 'orange']

    ax2.bar(categories, counts, color=colors)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title(f'Phase 2: Reuse Decisions\n{dataset_name.upper()} - {meta_learner_type.upper()}', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add percentage labels on bars
    for i, (cat, count) in enumerate(zip(categories, counts)):
        percentage = (count / sum(counts)) * 100
        ax2.text(i, count, f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    # Save plot
    plot_path = f"OffMAR/accuracy-prediction/results/cnn_{dataset_name}_{meta_learner_type}.png"
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
            results = run_offmar_cnn_example(
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
    print("-" * 90)

    for meta_learner, results in all_results.items():
        print(f"{meta_learner.upper():<15} "
              f"{results['best_performance']:<12.4f} "
              f"{results['final_performance']:<12.4f} "
              f"{results['test_performance']:<12.4f} "
              f"{results['phase_2']['reuse_percentage']:<12.1f} "
              f"{results['total_time']:<12.2f}")

    print(f"{'='*60}\n")

    return all_results


if __name__ == "__main__":
    # Example 1: Run with single meta-learner
    results = run_offmar_cnn_example(
        dataset_name='mnist',
        meta_learner_type='rf',
        timesteps=20,  # Reduced for quick testing
        save_results=True
    )

    # Example 2: Compare all meta-learners
    # comparison = compare_meta_learners(
    #     dataset_name='mnist',
    #     timesteps=20
    # )
