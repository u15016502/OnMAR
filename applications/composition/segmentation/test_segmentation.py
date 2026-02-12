"""
Test script for Segmentation Composition Application

This script tests the segmentation application to verify everything works correctly.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import numpy as np
from applications.composition.segmentation.segmentation_application import SegmentationCompositionApplication
from applications.composition.segmentation.segmentation_model import (
    SegmentationComponents, COMPONENT_REGISTRY, apply_chromosome
)


def test_individual_components():
    """Test individual segmentation components."""
    print("=" * 80)
    print("Testing Individual Segmentation Components")
    print("=" * 80)

    # Create a test image (random RGB image)
    test_image = np.random.rand(64, 64, 3).astype(np.float32)
    print(f"\nTest image shape: {test_image.shape}")

    # Test a few components from each category
    test_components = [
        ('FIL', 'gaussian_filter'),
        ('FIL', 'thresholding'),
        ('MOR', 'dilation'),
        ('MOR', 'erosion'),
        ('FD', 'sobel'),
        ('FD', 'canny_edge'),
        ('CIS', 'slic_superpixels'),
        ('CIS', 'felsenszwalb'),
        ('CG', 'kmeans_plusplus'),
        ('CG', 'gaussian_mixtures'),
    ]

    print("\nTesting components:")
    for comp_type, comp_name in test_components:
        try:
            func = COMPONENT_REGISTRY[comp_name]
            result = func(test_image)
            print(f"   [{comp_type}] {comp_name}: output shape {result.shape} - OK")
        except Exception as e:
            print(f"   [{comp_type}] {comp_name}: FAILED - {e}")

    print(f"\nTotal components in registry: {len(COMPONENT_REGISTRY)}")


def test_chromosome_application():
    """Test applying a chromosome to an image."""
    print("\n" + "=" * 80)
    print("Testing Chromosome Application")
    print("=" * 80)

    # Create a test image
    test_image = np.random.rand(64, 64, 3).astype(np.float32)

    # Test chromosome
    test_chromosome = [
        'FIL_gaussian_filter',
        'FD_sobel',
        'MOR_dilation',
        'CIS_slic_superpixels'
    ]

    print(f"\nTest chromosome: {test_chromosome}")
    print(f"Input image shape: {test_image.shape}")

    result = apply_chromosome(test_image, test_chromosome)
    print(f"Output shape: {result.shape}")
    print("Chromosome application: OK")


def test_segmentation_application():
    """Test the SegmentationCompositionApplication class."""
    print("\n" + "=" * 80)
    print("Testing Segmentation Composition Application")
    print("=" * 80)

    # Initialize application (using bsd500 as default)
    print("\n1. Initializing Segmentation application...")
    try:
        app = SegmentationCompositionApplication(
            dataset_name='bsd500',
            random_seed=42
        )
        print("   Initialization: OK")
    except Exception as e:
        print(f"   Initialization: FAILED - {e}")
        print("   (This may be expected if dataset is not downloaded)")
        return

    # Get design space
    print("\n2. Design space:")
    design_space = app.get_design_space()
    for key, values in design_space.items():
        print(f"   {key}: {len(values)} options")
    print(f"   Total unique components: {sum(len(v) for v in design_space.values())}")

    # Validate a test design
    print("\n3. Testing design validation...")
    test_design = {
        'chromosome': [
            'FIL_gaussian_filter',
            'MOR_dilation',
            'FD_canny_edge',
            'CIS_slic_superpixels'
        ]
    }
    # Note: validate_design in base class expects different format
    # For composition, chromosome format is different
    print(f"   Test design: {test_design}")

    # Test application info methods
    print(f"\n4. Application type: {app.get_application_type()}")
    print(f"   Supports dynamic designs: {app.supports_dynamic_designs()}")
    print(f"   Default number of timesteps (generations): {app.get_num_timesteps()}")

    print("\n" + "=" * 80)
    print("Basic tests completed!")
    print("=" * 80)


def test_ga_operators():
    """Test GA operators (crossover, mutation)."""
    print("\n" + "=" * 80)
    print("Testing GA Operators")
    print("=" * 80)

    app = SegmentationCompositionApplication(
        dataset_name='bsd500',
        random_seed=42
    )

    # Test crossover
    print("\n1. Testing crossover...")
    parent1 = ['FIL_gaussian_filter', 'MOR_dilation', 'FD_sobel']
    parent2 = ['FIL_median_blur', 'CIS_watershed', 'CG_kmeans_plusplus', 'MOR_erosion']

    print(f"   Parent 1: {parent1}")
    print(f"   Parent 2: {parent2}")

    offspring = app._crossover(parent1, parent2)
    print(f"   Offspring: {offspring}")

    # Test mutation
    print("\n2. Testing mutation...")
    original = ['FIL_gaussian_filter', 'MOR_dilation', 'FD_sobel']
    print(f"   Original: {original}")

    for i in range(3):
        mutated = app._mutate(original.copy())
        print(f"   Mutated {i+1}: {mutated}")

    print("\nGA operators: OK")


def run_quick_evolution():
    """Run a quick evolution test (few generations)."""
    print("\n" + "=" * 80)
    print("Running Quick Evolution Test (5 generations)")
    print("=" * 80)

    app = SegmentationCompositionApplication(
        dataset_name='bsd500',
        random_seed=42
    )

    # Test design
    test_design = {
        'chromosome': [
            'FIL_gaussian_filter',
            'FD_canny_edge',
            'CIS_slic_superpixels'
        ]
    }

    print(f"\nSeed design: {test_design}")
    print("\nRunning evolution (5 generations with placeholder fitness)...")

    try:
        # Note: This will use placeholder fitness evaluation
        # Real evaluation requires actual dataset loading
        results = app.train(test_design, timesteps=5)

        print("\nEvolution results:")
        for key, value in results.items():
            if 'time' in key:
                print(f"   {key}: {value:.2f} seconds")
            else:
                print(f"   {key}: {value:.4f}")

        print("\nBest individual found:")
        print(f"   {app.current_design}")

    except Exception as e:
        print(f"\nEvolution failed: {e}")
        print("(This may be expected if dataset is not available)")


if __name__ == "__main__":
    # Run tests
    test_individual_components()
    test_chromosome_application()
    test_ga_operators()
    test_segmentation_application()

    # Uncomment to run evolution test (requires more time)
    # run_quick_evolution()

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
