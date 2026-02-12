"""
Example script for running OffMAR Accuracy Prediction with Segmentation Composition Application.

This demonstrates how to use OffMAR accuracy prediction with the segmentation composition application.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from OffMAR.accuracy_prediction.offmar import OffMARAccuracyPrediction
from OffMAR.accuracy_prediction.config import SEGMENTATION_CONFIG
from applications.composition.segmentation.segmentation_application import SegmentationCompositionApplication


def run_offmar_segmentation_example(
    dataset_name: str = 'bsd500',
    meta_learner_type: str = 'rf',
    timesteps: int = 30
):
    """
    Run OffMAR accuracy prediction with segmentation composition application.

    Args:
        dataset_name: Dataset to use ('bsd500', 'covid', 'pascal')
        meta_learner_type: Meta-learner to use ('knn', 'rf', 'xgboost')
        timesteps: Number of GA generations (split between phases)

    Returns:
        Results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Running OffMAR Accuracy Prediction with Segmentation Composition")
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

    # Initialize OffMAR Accuracy Prediction
    offmar = OffMARAccuracyPrediction(
        application=seg_app,
        meta_learner_type=config.meta_learner_type,
        theta_p=config.theta_p,
        meta_learner_params=config.meta_learner_params
    )

    # Run full OffMAR
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
    print(f"\nPhase 1:")
    print(f"  Best performance: {results['phase_1']['best_performance']:.4f}")
    print(f"\nPhase 2:")
    print(f"  Reuse percentage: {results['phase_2']['reuse_percentage']:.1f}%")
    print(f"  Final performance: {results['final_performance']:.4f}")
    print(f"{'='*60}\n")

    # Save model
    model_path = f"OffMAR/accuracy-prediction/models/seg_{dataset_name}_{meta_learner_type}.pkl"
    offmar.save_model(model_path)
    print(f"Model saved to: {model_path}")

    return results


if __name__ == "__main__":
    results = run_offmar_segmentation_example(
        dataset_name='bsd500',
        meta_learner_type='rf',
        timesteps=20  # Reduced for quick testing
    )
