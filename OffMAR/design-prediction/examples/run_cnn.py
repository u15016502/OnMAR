"""
Example script for running OffMAR Design Prediction with CNN Configuration Application.

This demonstrates how to use OffMAR with the CNN configuration application
across all three meta-learners (kNN, RF, XGBoost).
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from OffMAR.design_prediction.offmar import OffMARDesignPrediction
from OffMAR.design_prediction.config import CNN_CONFIG
from applications.configuration.cnn.cnn_application import CNNConfigurationApplication


def run_offmar_cnn_example(
    dataset_name: str = 'mnist',
    meta_learner_type: str = 'knn',
    timesteps: int = 50
):
    """
    Run OffMAR with CNN configuration application.

    Args:
        dataset_name: Dataset to use ('mnist', 'cifar-10', 'fashion-mnist', etc.)
        meta_learner_type: Meta-learner to use ('knn', 'rf', 'xgboost')
        timesteps: Number of training epochs
    """
    print(f"\n{'='*60}")
    print(f"Running OffMAR Design Prediction with CNN Configuration")
    print(f"Dataset: {dataset_name}")
    print(f"Meta-learner: {meta_learner_type.upper()}")
    print(f"Timesteps: {timesteps}")
    print(f"{'='*60}\n")

    # Initialize CNN application
    cnn_app = CNNConfigurationApplication(
        dataset_name=dataset_name,
        random_seed=42
    )

    # Load data
    print("Loading dataset...")
    cnn_app.load_data()

    # Get configuration for this meta-learner
    config = CNN_CONFIG[meta_learner_type]

    # Initialize OffMAR
    offmar = OffMARDesignPrediction(
        application=cnn_app,
        meta_learner_type=config.meta_learner_type,
        theta_p=config.theta_p,
        meta_learner_params=config.meta_learner_params
    )

    # Run full OffMAR (both phases)
    results = offmar.run_full_offmar(
        dataset_name=dataset_name,
        timesteps=timesteps,
        initial_design=None  # Use random initialization
    )

    # Print results
    print(f"\n{'='*60}")
    print("Results Summary:")
    print(f"{'='*60}")
    print(f"Total runtime: {results['total_time']:.2f} seconds")
    print(f"\nPhase 1 (Data Collection):")
    print(f"  - Samples collected: {results['phase_1']['samples_collected']}")
    print(f"  - Samples after pruning: {results['phase_1']['samples_after_pruning']}")
    print(f"  - Best performance: {results['phase_1']['best_performance']:.4f}")
    print(f"  - Runtime: {results['phase_1']['phase_1_time']:.2f} seconds")
    print(f"\nPhase 2 (Design Prediction):")
    print(f"  - Final performance: {results['final_performance']:.4f}")
    print(f"  - Runtime: {results['phase_2']['phase_2_time']:.2f} seconds")
    print(f"{'='*60}\n")

    # Save model
    model_path = f"OffMAR/design-prediction/models/cnn_{dataset_name}_{meta_learner_type}.pkl"
    offmar.save_model(model_path)
    print(f"Model saved to: {model_path}")

    return results


if __name__ == "__main__":
    # Example: Run with different meta-learners
    datasets = ['mnist']  # Can add more: 'cifar-10', 'fashion-mnist'
    meta_learners = ['knn', 'rf', 'xgboost']

    for dataset in datasets:
        for meta_learner in meta_learners:
            try:
                results = run_offmar_cnn_example(
                    dataset_name=dataset,
                    meta_learner_type=meta_learner,
                    timesteps=20  # Reduced for quick testing
                )
            except Exception as e:
                print(f"Error with {dataset} and {meta_learner}: {e}")
                continue
