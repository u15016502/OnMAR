"""
Example script for running OnMAR Design Prediction with Segmentation Composition Application.

This demonstrates how to use OnMAR design prediction with the segmentation composition application.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from OnMAR.design_prediction.onmar import OnMARDesignPrediction
from OnMAR.design_prediction.config import SEGMENTATION_CONFIG
from applications.composition.segmentation.segmentation_application import SegmentationCompositionApplication


def run_onmar_segmentation_example(
    dataset_name: str = 'bsd500',
    meta_learner_type: str = 'rf',
    timesteps: int = 30
):
    """
    Run OnMAR design prediction with segmentation composition application.

    Args:
        dataset_name: Dataset to use ('bsd500', 'covid', 'pascal')
        meta_learner_type: Meta-learner to use ('knn', 'rf', 'xgboost')
        timesteps: Number of GA generations

    Returns:
        Results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Running OnMAR Design Prediction with Segmentation Composition")
    print(f"Dataset: {dataset_name}")
    print(f"Meta-learner: {meta_learner_type.upper()}")
    print(f"Timesteps: {timesteps}")
    print(f"{'='*60}\n")

    # Initialize segmentation application
    seg_app = SegmentationCompositionApplication(
        dataset_name=dataset_name,
        random_seed=42
    )

    # Get configuration for this meta-learner
    config = SEGMENTATION_CONFIG[meta_learner_type]

    # Initialize OnMAR Design Prediction
    onmar = OnMARDesignPrediction(
        application=seg_app,
        meta_learner_type=config.meta_learner_type,
        theta_t=config.theta_t,  # None means N/2
        theta_p=config.theta_p,
        meta_learner_params=config.meta_learner_params
    )

    # Run OnMAR
    results = onmar.run_onmar(
        dataset_name=dataset_name,
        timesteps=timesteps,
        initial_design=None
    )

    # Print results summary
    print(f"\n{'='*60}")
    print("Results Summary:")
    print(f"{'='*60}")
    print(f"Total runtime: {results['total_time']:.2f} seconds")
    print(f"Best performance: {results['best_performance']:.4f}")
    print(f"Final performance: {results['final_performance']:.4f}")
    print(f"\nEfficiency Metrics:")
    print(f"  Design algorithm calls: {results['num_design_algorithm_calls']} ({results['design_algorithm_percentage']:.1f}%)")
    print(f"  Design predictions: {results['num_design_predictions']} ({results['design_prediction_percentage']:.1f}%)")
    print(f"  Knowledge repository size: {results['knowledge_repository_size']}")
    print(f"{'='*60}\n")

    # Save model
    model_path = f"OnMAR/design-prediction/models/seg_{dataset_name}_{meta_learner_type}.pkl"
    onmar.save_model(model_path)
    print(f"Model saved to: {model_path}")

    return results


if __name__ == "__main__":
    # Example: Run with Random Forest (recommended for segmentation)
    results = run_onmar_segmentation_example(
        dataset_name='bsd500',
        meta_learner_type='rf',
        timesteps=15  # Reduced for quick testing
    )
