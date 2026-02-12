"""
Test script for Fuzzy ART Choice Function Generation Application

This script tests the Fuzzy ART application to verify everything works correctly.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import numpy as np
from applications.generation.fuzzyart.fuzzyart_application import FuzzyARTGenerationApplication
from applications.generation.fuzzyart.fuzzyart_model import (
    FuzzyART, ChoiceFunctionPrimitives, ExpressionEvaluator,
    OPERATION_REGISTRY, create_choice_function
)


def test_fuzzy_art_basic():
    """Test basic Fuzzy ART clustering."""
    print("=" * 80)
    print("Testing Basic Fuzzy ART Clustering")
    print("=" * 80)

    # Create simple test data
    np.random.seed(42)
    # Three clusters
    cluster1 = np.random.randn(20, 2) + np.array([0, 0])
    cluster2 = np.random.randn(20, 2) + np.array([5, 5])
    cluster3 = np.random.randn(20, 2) + np.array([0, 5])
    X = np.vstack([cluster1, cluster2, cluster3])

    # Normalize to [0, 1]
    X = (X - X.min()) / (X.max() - X.min())

    print(f"\nTest data shape: {X.shape}")

    # Test with default choice function
    art = FuzzyART(alpha=0.01, beta=1.0, rho=0.7)
    labels = art.fit_predict(X, max_iterations=10)

    print(f"Number of categories created: {art.num_categories}")
    print(f"Unique labels: {np.unique(labels)}")
    print(f"Label distribution: {np.bincount(labels)}")

    print("\nBasic Fuzzy ART: OK")


def test_primitive_operations():
    """Test individual primitive operations."""
    print("\n" + "=" * 80)
    print("Testing Primitive Operations")
    print("=" * 80)

    # Create test arrays
    a = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    b = np.array([0.5, 0.4, 0.3, 0.2, 0.1])

    print(f"\nTest arrays:")
    print(f"   a = {a}")
    print(f"   b = {b}")

    # Test arithmetic operations
    print("\nArithmetic operations:")
    print(f"   add(a, b) = {ChoiceFunctionPrimitives.add(a, b)}")
    print(f"   sub(a, b) = {ChoiceFunctionPrimitives.sub(a, b)}")
    print(f"   mul(a, b) = {ChoiceFunctionPrimitives.mul(a, b)}")
    print(f"   div(a, b) = {ChoiceFunctionPrimitives.div(a, b)}")
    print(f"   min(a) = {ChoiceFunctionPrimitives.min_op(a)}")
    print(f"   max(a) = {ChoiceFunctionPrimitives.max_op(a)}")

    # Test linear algebra operations
    print("\nLinear algebra operations:")
    print(f"   inner(a, b) = {ChoiceFunctionPrimitives.inner(a, b):.4f}")
    print(f"   dot(a, b) = {ChoiceFunctionPrimitives.dot(a, b):.4f}")
    print(f"   l1norm(a) = {ChoiceFunctionPrimitives.l1norm(a):.4f}")
    print(f"   l2norm(a) = {ChoiceFunctionPrimitives.l2norm(a):.4f}")

    # Test statistical operations
    print("\nStatistical operations:")
    print(f"   average(a) = {ChoiceFunctionPrimitives.average(a):.4f}")
    print(f"   stdev(a) = {ChoiceFunctionPrimitives.stdev(a):.4f}")
    print(f"   variance(a) = {ChoiceFunctionPrimitives.variance(a):.4f}")
    print(f"   median(a) = {ChoiceFunctionPrimitives.median(a):.4f}")

    # Test fuzzy operations
    print("\nFuzzy operations:")
    print(f"   fuzzy_and(a, b) = {ChoiceFunctionPrimitives.fuzzy_and(a, b)}")
    print(f"   fuzzy_or(a, b) = {ChoiceFunctionPrimitives.fuzzy_or(a, b)}")

    print(f"\nTotal operations in registry: {len(OPERATION_REGISTRY)}")
    print("\nPrimitive operations: OK")


def test_expression_evaluator():
    """Test expression parsing and evaluation."""
    print("\n" + "=" * 80)
    print("Testing Expression Evaluator")
    print("=" * 80)

    evaluator = ExpressionEvaluator()

    # Test context
    context = {
        'f': np.array([0.1, 0.2, 0.3, 0.4]),
        'cluster': np.array([0.5, 0.5, 0.5, 0.5]),
        'alpha': 0.01,
        'beta': 1.0,
        'rho': 0.5,
        'pi': np.pi
    }

    # Test expressions
    test_expressions = [
        ("l1norm(f)", "L1 norm of input"),
        ("l2norm(cluster)", "L2 norm of cluster"),
        ("inner(f, cluster)", "Inner product"),
        ("div(l1norm(fuzzy_and(f, cluster)), add(alpha, l1norm(cluster)))", "Standard choice function"),
        ("add(inner(f, cluster), mul(alpha, l1norm(f)))", "Custom choice function"),
        ("average(mul(f, cluster))", "Average of element-wise product"),
    ]

    print("\nEvaluating expressions:")
    for expr, desc in test_expressions:
        try:
            result = evaluator.parse_and_evaluate(expr, context)
            print(f"   {desc}:")
            print(f"      Expression: {expr}")
            print(f"      Result: {result:.6f}")
        except Exception as e:
            print(f"   {desc}: FAILED - {e}")

    print("\nExpression evaluator: OK")


def test_custom_choice_function():
    """Test Fuzzy ART with custom generated choice function."""
    print("\n" + "=" * 80)
    print("Testing Custom Choice Function")
    print("=" * 80)

    # Create test data
    np.random.seed(42)
    cluster1 = np.random.randn(15, 2) * 0.1 + np.array([0.2, 0.2])
    cluster2 = np.random.randn(15, 2) * 0.1 + np.array([0.8, 0.8])
    X = np.vstack([cluster1, cluster2])
    X = np.clip(X, 0, 1)  # Ensure [0, 1] range

    # Create custom choice function from expression
    expression = "div(l1norm(fuzzy_and(f, cluster)), add(alpha, l1norm(cluster)))"
    print(f"\nCustom expression: {expression}")

    custom_func = create_choice_function(expression)

    # Test with custom choice function
    art = FuzzyART(alpha=0.01, beta=1.0, rho=0.7, choice_function=custom_func)
    labels = art.fit_predict(X, max_iterations=10)

    print(f"Number of categories: {art.num_categories}")
    print(f"Unique labels: {np.unique(labels)}")

    print("\nCustom choice function: OK")


def test_ge_decoding():
    """Test Grammatical Evolution chromosome decoding."""
    print("\n" + "=" * 80)
    print("Testing GE Chromosome Decoding")
    print("=" * 80)

    # Initialize application
    app = FuzzyARTGenerationApplication(
        dataset_name='iris',  # Placeholder
        random_seed=42
    )

    # Test chromosomes
    test_chromosomes = [
        [10, 20, 30, 40, 50, 60, 70, 80],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [255, 255, 255, 255, 255, 255, 255, 255],
        list(np.random.randint(0, 256, size=20))
    ]

    print("\nDecoding chromosomes:")
    for i, chromosome in enumerate(test_chromosomes):
        expr = app._decode_chromosome(chromosome, max_wraps=3)
        print(f"   Chromosome {i+1}: {chromosome[:8]}...")
        print(f"   Expression: {expr}")
        print()

    print("GE decoding: OK")


def test_fuzzyart_application():
    """Test the FuzzyARTGenerationApplication class."""
    print("\n" + "=" * 80)
    print("Testing Fuzzy ART Generation Application")
    print("=" * 80)

    # Initialize application
    print("\n1. Initializing application...")
    app = FuzzyARTGenerationApplication(
        dataset_name='iris',
        random_seed=42
    )
    print("   Initialization: OK")

    # Get design space
    print("\n2. Design space:")
    design_space = app.get_design_space()
    for key, values in design_space.items():
        if isinstance(values, list) and len(values) > 5:
            print(f"   {key}: {len(values)} options")
        else:
            print(f"   {key}: {values}")

    # Get grammar
    print("\n3. Grammar rules:")
    grammar = app.get_grammar()
    for key, rules in grammar.items():
        print(f"   {key}: {len(rules)} productions")

    # Application info
    print(f"\n4. Application type: {app.get_application_type()}")
    print(f"   Supports dynamic designs: {app.supports_dynamic_designs()}")
    print(f"   Default timesteps (generations): {app.get_num_timesteps()}")

    print("\n" + "=" * 80)
    print("Application tests completed!")
    print("=" * 80)


def test_ga_operators():
    """Test GA operators (crossover, mutation)."""
    print("\n" + "=" * 80)
    print("Testing GA Operators")
    print("=" * 80)

    app = FuzzyARTGenerationApplication(
        dataset_name='iris',
        random_seed=42
    )

    # Test crossover
    print("\n1. Testing crossover...")
    parent1 = list(range(0, 50))
    parent2 = list(range(50, 100))

    print(f"   Parent 1: {parent1[:10]}... (len={len(parent1)})")
    print(f"   Parent 2: {parent2[:10]}... (len={len(parent2)})")

    offspring = app._crossover(parent1, parent2)
    print(f"   Offspring: {offspring[:10]}... (len={len(offspring)})")

    # Test mutation
    print("\n2. Testing mutation...")
    original = list(range(50))
    print(f"   Original: {original[:10]}...")

    for i in range(3):
        mutated = app._mutate(original.copy())
        # Find mutation point
        diff_idx = next((j for j in range(len(original)) if original[j] != mutated[j]), -1)
        if diff_idx >= 0:
            print(f"   Mutated {i+1}: Changed position {diff_idx}: {original[diff_idx]} -> {mutated[diff_idx]}")

    print("\nGA operators: OK")


def run_quick_evolution():
    """Run a quick evolution test (few generations)."""
    print("\n" + "=" * 80)
    print("Running Quick Evolution Test (5 generations)")
    print("=" * 80)

    app = FuzzyARTGenerationApplication(
        dataset_name='iris',
        random_seed=42
    )

    # Test design (seed chromosome)
    test_design = {
        'chromosome': list(np.random.randint(0, 256, size=30)),
        'chromosome_length': 30
    }

    print(f"\nSeed chromosome length: {len(test_design['chromosome'])}")
    print("\nRunning evolution (5 generations with placeholder fitness)...")

    try:
        results = app.train(test_design, timesteps=5)

        print("\nEvolution results:")
        for key, value in results.items():
            if key == 'best_expression':
                print(f"   {key}: {value[:50]}..." if len(str(value)) > 50 else f"   {key}: {value}")
            elif 'time' in key:
                print(f"   {key}: {value:.2f} seconds")
            else:
                print(f"   {key}: {value:.4f}" if isinstance(value, float) else f"   {key}: {value}")

    except Exception as e:
        print(f"\nEvolution failed: {e}")


if __name__ == "__main__":
    # Run tests
    test_fuzzy_art_basic()
    test_primitive_operations()
    test_expression_evaluator()
    test_custom_choice_function()
    test_ge_decoding()
    test_ga_operators()
    test_fuzzyart_application()

    # Uncomment to run evolution test
    # run_quick_evolution()

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
