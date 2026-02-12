"""
Test script for CNN Configuration Application

This script tests the CNN application on MNIST to verify everything works correctly.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from applications.configuration.cnn.cnn_application import CNNConfigurationApplication


def test_cnn_application():
    """Test the CNN application with a simple configuration."""
    print("="*80)
    print("Testing CNN Configuration Application")
    print("="*80)

    # Initialize application
    print("\n1. Initializing CNN application with MNIST dataset...")
    app = CNNConfigurationApplication(
        dataset_name='mnist',
        random_seed=42
    )

    # Load data
    print("\n2. Loading data...")
    app.load_data()

    # Get design space
    print("\n3. Design space:")
    design_space = app.get_design_space()
    for key, values in design_space.items():
        print(f"   {key}: {values}")

    # Get meta-features
    print("\n4. Meta-features:")
    meta_features = app.get_meta_features()
    for key, value in meta_features.items():
        print(f"   {key}: {value}")

    # Define a simple design
    print("\n5. Testing with a simple design...")
    test_design = {
        'conv_filters': 32,
        'batch_norm': 1,
        'activation': 3,  # ReLU
        'dropout': 0.2,
        'max_pool_size': 2,
        'dense_nodes': 128,
        'optimizer': 1,  # Adam
        'learning_rate': 0.001,
        'num_conv_layers': 2
    }

    print(f"   Design: {test_design}")

    # Validate design
    is_valid = app.validate_design(test_design)
    print(f"   Design is valid: {is_valid}")

    # Train model (with only 5 epochs for quick testing)
    print("\n6. Training model (5 epochs for quick test)...")
    train_metrics = app.train(test_design, timesteps=5)

    print("\n   Training metrics:")
    for key, value in train_metrics.items():
        if 'time' in key:
            print(f"   {key}: {value:.2f} seconds")
        else:
            print(f"   {key}: {value:.4f}")

    # Evaluate on test set
    print("\n7. Evaluating on test set...")
    test_metrics = app.evaluate()

    print("   Test metrics:")
    for key, value in test_metrics.items():
        print(f"   {key}: {value:.4f}")

    # Test application info methods
    print(f"\n8. Application type: {app.get_application_type()}")
    print(f"   Supports dynamic designs: {app.supports_dynamic_designs()}")
    print(f"   Default number of timesteps: {app.get_num_timesteps()}")

    print("\n" + "="*80)
    print("Test completed successfully!")
    print("="*80)


if __name__ == "__main__":
    test_cnn_application()
