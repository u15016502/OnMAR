"""
Example script for running Auto-sklearn with Segmentation Composition Application.

This demonstrates how to use Auto-sklearn to replace GA for evolving
segmentation algorithm compositions.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from sota.autosklearn_wrapper import AutoSklearnWrapper
from applications.composition.segmentation.segmentation_application import SegmentationCompositionApplication


def run_autosklearn_segmentation_example(
    dataset_name: str = 'bsd500',
    time_budget: int = 3600,
    max_evaluations: int = 30
):
    """
    Run Auto-sklearn with segmentation composition application.

    Args:
        dataset_name: Dataset to use ('bsd500', 'covid', 'pascal')
        time_budget: Total time budget in seconds
        max_evaluations: Maximum number of design evaluations

    Returns:
        Results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Running Auto-sklearn with Segmentation Composition")
    print(f"Dataset: {dataset_name}")
    print(f"Time budget: {time_budget}s")
    print(f"Max evaluations: {max_evaluations}")
    print(f"{'='*60}\n")

    # Initialize segmentation application
    seg_app = SegmentationCompositionApplication(
        dataset_name=dataset_name,
        random_seed=42
    )

    # Initialize Auto-sklearn wrapper
    autosklearn = AutoSklearnWrapper(
        application=seg_app,
        time_budget=time_budget,
        per_run_time_limit=600,  # 10 minutes per design evaluation (segmentation can be slow)
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
    print(f"\nBest Segmentation Pipeline:")
    for param, value in results['best_design'].items():
        print(f"  {param}: {value}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    # Example: Run Auto-sklearn on BSD500
    results = run_autosklearn_segmentation_example(
        dataset_name='bsd500',
        time_budget=1800,  # 30 minutes
        max_evaluations=20  # Reduced for quick testing
    )
