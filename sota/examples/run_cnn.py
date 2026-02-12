"""
Example script for running Auto-sklearn with CNN Configuration Application.

This demonstrates how to use Auto-sklearn as an alternative design algorithm
to replace manual hyperparameter tuning or GA-based optimization.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from sota.autosklearn_wrapper import AutoSklearnWrapper
from applications.configuration.cnn.cnn_application import CNNConfigurationApplication
import matplotlib.pyplot as plt
import numpy as np


def run_autosklearn_cnn_example(
    dataset_name: str = 'mnist',
    time_budget: int = 3600,
    max_evaluations: int = 50,
    save_results: bool = True
):
    """
    Run Auto-sklearn with CNN configuration application.

    Args:
        dataset_name: Dataset to use ('mnist', 'cifar-10', 'fashion-mnist', etc.)
        time_budget: Total time budget in seconds
        max_evaluations: Maximum number of design evaluations
        save_results: Whether to save results and plots

    Returns:
        Results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Running Auto-sklearn with CNN Configuration")
    print(f"Dataset: {dataset_name}")
    print(f"Time budget: {time_budget}s")
    print(f"Max evaluations: {max_evaluations}")
    print(f"{'='*60}\n")

    # Initialize CNN application
    cnn_app = CNNConfigurationApplication(
        dataset_name=dataset_name,
        random_seed=42
    )

    # Initialize Auto-sklearn wrapper
    autosklearn = AutoSklearnWrapper(
        application=cnn_app,
        time_budget=time_budget,
        per_run_time_limit=300,  # 5 minutes per design evaluation
        n_jobs=1,
        random_seed=42
    )

    # Run optimization
    results = autosklearn.run_optimization(
        dataset_name=dataset_name,
        max_evaluations=max_evaluations
    )

    # Print results summary
    print(f"\n{'='*60}")
    print("Results Summary:")
    print(f"{'='*60}")
    print(f"Total runtime: {results['total_time']:.2f} seconds")
    print(f"Number of evaluations: {results['num_evaluations']}")
    print(f"Best performance: {results['best_performance']:.4f}")
    print(f"Test performance: {results['test_results'].get('test_performance', 'N/A')}")
    print(f"\nBest Design:")
    for param, value in results['best_design'].items():
        print(f"  {param}: {value}")
    print(f"{'='*60}\n")

    # Plot optimization trajectory
    if save_results:
        plot_results(autosklearn, dataset_name)

    return results


def plot_results(autosklearn: AutoSklearnWrapper, dataset_name: str):
    """
    Plot Auto-sklearn optimization trajectory.

    Args:
        autosklearn: AutoSklearnWrapper instance
        dataset_name: Dataset name
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Get trajectories
    all_perf, all_eval = autosklearn.get_optimization_trajectory()
    best_perf, best_eval = autosklearn.get_incumbent_trajectory()

    # Plot 1: All evaluations
    ax1.scatter(all_eval, all_perf, alpha=0.5, s=20, label='All evaluations')
    ax1.plot(best_eval, best_perf, 'r-', linewidth=2, label='Best so far')
    ax1.set_xlabel('Evaluation Number', fontsize=12)
    ax1.set_ylabel('Performance (Accuracy)', fontsize=12)
    ax1.set_title(f'Auto-sklearn Optimization Trajectory\n{dataset_name.upper()}', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Incumbent trajectory (zoomed)
    ax2.plot(best_eval, best_perf, 'g-', linewidth=2, marker='o', markersize=4)
    ax2.set_xlabel('Evaluation Number', fontsize=12)
    ax2.set_ylabel('Best Performance So Far', fontsize=12)
    ax2.set_title(f'Best Performance Over Time\n{dataset_name.upper()}', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Add final best performance annotation
    final_best = best_perf[-1]
    ax2.annotate(f'Final: {final_best:.4f}',
                 xy=(best_eval[-1], final_best),
                 xytext=(10, -20),
                 textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()

    # Save plot
    plot_path = f"sota/results/autosklearn_cnn_{dataset_name}.png"
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")

    plt.close()


if __name__ == "__main__":
    # Example: Run Auto-sklearn on MNIST
    results = run_autosklearn_cnn_example(
        dataset_name='mnist',
        time_budget=1800,  # 30 minutes
        max_evaluations=30,  # Reduced for quick testing
        save_results=True
    )
