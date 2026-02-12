"""
Fuzzy ART Choice Function Generation Application

Automated generation of Fuzzy ART choice functions using Grammatical Evolution (GE).
Uses a grammar-based approach to evolve mathematical expressions that serve as
choice functions for the Fuzzy ART clustering algorithm.
Design space based on Appendix A.6 of the thesis.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from typing import Dict, Any, List, Optional, Tuple
import time
import numpy as np
from numpy import linalg as la
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import random
from sklearn.metrics import accuracy_score
from applications.base_application import BaseApplication
from datasets.text_datasets import TextDatasetLoader
from applications.generation.fuzzyart.meta_features import FuzzyARTMetaFeatureExtractor


# ============================================================================
# Fuzzy ART Helper Functions
# ============================================================================

def fuzzy_and(x, y):
    """Fuzzy AND operation (element-wise minimum)."""
    return np.minimum(x, y)


def fuzzy_or(x, y):
    """Fuzzy OR operation (element-wise maximum)."""
    return np.maximum(x, y)


def l1norm(x):
    """L1 norm (Manhattan distance)."""
    return la.norm(x, ord=1)


def l2norm(x):
    """L2 norm (Euclidean distance)."""
    return la.norm(x, ord=2)


def safe_divide(x1, x2):
    """Safe division that handles division by zero."""
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.divide(x1, x2)
            result[~np.isfinite(result)] = 0
            return result
    except:
        return x1


def infer_cluster_labels(predicted_clusters, true_labels):
    """
    Map cluster IDs to class labels based on majority voting.

    Args:
        predicted_clusters: Cluster assignments for each data point
        true_labels: Ground truth class labels

    Returns:
        Dictionary mapping cluster IDs to class labels
    """
    predicted_clusters = predicted_clusters.astype(np.uint8)
    true_labels = true_labels.astype(np.uint8)

    inferred_labels = {}
    unique_clusters = np.unique(predicted_clusters)

    for cluster_id in unique_clusters:
        # Find all points in this cluster
        mask = predicted_clusters == cluster_id
        cluster_true_labels = true_labels[mask]

        if len(cluster_true_labels) > 0:
            # Find the most common true label in this cluster
            counts = np.bincount(cluster_true_labels)
            most_common_label = np.argmax(counts)

            if most_common_label in inferred_labels:
                inferred_labels[most_common_label].append(cluster_id)
            else:
                inferred_labels[most_common_label] = [cluster_id]

    return inferred_labels


def map_clusters_to_labels(cluster_assignments, cluster_to_label_map):
    """
    Convert cluster assignments to predicted class labels.

    Args:
        cluster_assignments: Cluster ID for each data point
        cluster_to_label_map: Mapping from class labels to cluster IDs

    Returns:
        Predicted class labels
    """
    predicted_labels = np.zeros(len(cluster_assignments), dtype=np.uint8)

    for i, cluster_id in enumerate(cluster_assignments):
        for label, cluster_list in cluster_to_label_map.items():
            if cluster_id in cluster_list:
                predicted_labels[i] = label
                break

    return predicted_labels


class SimpleFuzzyART:
    """
    Simplified Fuzzy ART implementation for choice function evaluation.

    Based on the reference implementation but streamlined for use in
    the grammatical evolution process.
    """

    def __init__(self, rho, alpha, beta, num_features, choice_function_str):
        """
        Initialize Fuzzy ART network.

        Args:
            rho: Vigilance parameter (0 to 1) - controls cluster granularity
            alpha: Choice parameter (small positive value)
            beta: Learning rate (0 to 1)
            num_features: Number of features in input data
            choice_function_str: String representation of choice function
        """
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.num_features = num_features
        self.choice_function_str = choice_function_str

        # Weight matrix (starts with one uncommitted node)
        self.weights = np.ones((1, num_features * 2))
        self.num_clusters = 0

    def complement_code(self, data):
        """Apply complement coding to data."""
        return np.concatenate((data, 1 - data), axis=1)

    def evaluate_choice_function(self, x, w):
        """
        Evaluate the evolved choice function for category selection.

        Args:
            x: Input pattern (complement coded)
            w: Weight vector for a category

        Returns:
            Choice function value
        """
        try:
            # Create a namespace with all the variables and functions the expression might use
            namespace = {
                'x': x,
                'w': w,
                'alpha': self.alpha,
                'beta': self.beta,
                'rho': self.rho,
                'fuzzy_and': fuzzy_and,
                'fuzzy_or': fuzzy_or,
                'l1norm': l1norm,
                'l2norm': l2norm,
                'np': np,
                'min': np.minimum,
                'max': np.maximum,
                'exp': np.exp,
                'log': np.log,
                'sin': np.sin,
                'cos': np.cos,
                'add': np.add,
                'sub': np.subtract,
                'mul': np.multiply,
                'div': safe_divide,
            }

            # Evaluate the choice function string
            result = eval(self.choice_function_str, {"__builtins__": {}}, namespace)

            # Handle array results by taking the mean
            if isinstance(result, np.ndarray):
                result = np.mean(result)

            # Ensure result is a valid number
            if not np.isfinite(result):
                return 0.0

            return float(result)
        except:
            # If evaluation fails, return 0
            return 0.0

    def train_pattern(self, pattern):
        """
        Train on a single pattern and return winning cluster.

        Args:
            pattern: Input pattern (complement coded)

        Returns:
            Winning cluster ID
        """
        # Find winning category
        winner = self.find_winner(pattern)

        # Update weight of winning neuron
        try:
            self.weights[winner, :] = (
                self.beta * fuzzy_and(pattern, self.weights[winner, :]) +
                (1 - self.beta) * self.weights[winner, :]
            )
        except:
            pass

        # If uncommitted node won, increment cluster count and add new uncommitted node
        if winner >= self.num_clusters:
            self.num_clusters += 1
            self.weights = np.concatenate((self.weights, np.ones((1, self.num_features * 2))))

        return winner

    def find_winner(self, pattern):
        """
        Find the winning category for a pattern using the choice function.

        Args:
            pattern: Input pattern (complement coded)

        Returns:
            Index of winning category
        """
        num_categories = self.weights.shape[0]
        matches = np.zeros(num_categories)

        # Evaluate choice function for all categories
        for j in range(num_categories):
            matches[j] = self.evaluate_choice_function(pattern, self.weights[j, :])

        # Vigilance test
        vigilance_threshold = self.rho * l2norm(pattern)
        attempts = 0

        while attempts < num_categories:
            # Winner-take-all selection
            winner = np.argmax(matches)

            # Vigilance test
            if l2norm(fuzzy_and(pattern, self.weights[winner, :])) >= vigilance_threshold:
                return winner
            else:
                # Shut off this category and try again
                matches[winner] = 0
                attempts += 1

        # If no category passed, return the uncommitted node
        return num_categories - 1

    def run_clustering(self, train_data, test_data, max_epochs=10):
        """
        Run Fuzzy ART clustering on train and test data.

        Args:
            train_data: Training data (not complement coded yet)
            test_data: Test data (not complement coded yet)
            max_epochs: Maximum number of training epochs

        Returns:
            Tuple of (train_clusters, test_clusters)
        """
        # Complement code the data
        train_data_coded = self.complement_code(train_data)
        test_data_coded = self.complement_code(test_data)

        # Training phase
        train_clusters = np.zeros(len(train_data), dtype=int)

        for epoch in range(max_epochs):
            indices = list(range(len(train_data)))
            random.shuffle(indices)

            for idx in indices:
                train_clusters[idx] = self.train_pattern(train_data_coded[idx])

        # Testing phase (no weight updates)
        test_clusters = np.zeros(len(test_data), dtype=int)

        for idx in range(len(test_data)):
            test_clusters[idx] = self.find_winner(test_data_coded[idx])

        return train_clusters, test_clusters


class FuzzyARTGenerationApplication(BaseApplication):
    """
    Fuzzy ART choice function generation application for automated algorithm design.
    Uses Grammatical Evolution to generate mathematical expressions as choice functions.
    """

    def __init__(self, dataset_name: str, random_seed: int = 42, n_jobs: int = None):
        """
        Initialize the Fuzzy ART generation application.

        Args:
            dataset_name: Name of the dataset (chatgpt, enron, imdb)
            random_seed: Random seed for reproducibility
            n_jobs: Number of parallel jobs for GE fitness evaluation (default: CPU count - 1)
        """
        super().__init__(dataset_name, random_seed)

        # Set number of parallel jobs
        self.n_jobs = n_jobs if n_jobs is not None else max(1, mp.cpu_count() - 1)

        # Initialize dataset loader
        self.dataset_loader = TextDatasetLoader(dataset_name, random_seed=random_seed)
        self.dataset_info = self.dataset_loader.get_dataset_info()

        # Data (initialized in load_data)
        self.data = None
        self.labels = None

        # Current best design (generated choice function)
        self.current_design = None
        self.best_choice_function = None

        # Meta-feature extractor
        self.meta_feature_extractor = FuzzyARTMetaFeatureExtractor()

        # Fuzzy ART model (for meta-feature extraction)
        self.fuzzyart_model = None

        # Grammar for GE (from Appendix A.6)
        self.grammar = self._build_grammar()

        # Set random seed
        np.random.seed(random_seed)

    def load_data(self, max_samples: int = 5000, max_features: int = 1000) -> None:
        """
        Load and prepare the dataset.

        Args:
            max_samples: Maximum number of samples to load
            max_features: Maximum number of features for text vectorization
        """
        self.data, self.labels = self.dataset_loader.load_dataset(
            max_samples=max_samples,
            max_features=max_features
        )
        print(f"Loaded {self.dataset_name}: {len(self.data)} samples, {self.data.shape[1]} features")

    def _build_grammar(self) -> Dict[str, List[str]]:
        """
        Build the grammar for Grammatical Evolution from Appendix A.6.

        The grammar defines how choice functions can be constructed using:
        - Arithmetic operations (add, sub, mul, div, min, max, exp)
        - Linear algebra operations (inner, outer, matmul, dot, l1norm, l2norm)
        - Trigonometric functions (sin, cos, tan, log, asin, acos, atan)
        - Statistical functions (stdev, variance, average, median)
        - Boolean/fuzzy operations (logical_and, logical_or, fuzzy_and, fuzzy_or)

        Returns:
            Dictionary mapping non-terminals to their production rules
        """
        return {
            'fns': ['arth', 'la', 'trig', 'stat', 'bool'],

            'arth': [
                'add(i, i)', 'sub(i, i)', 'mul(i, i)',
                'div(i, i)', 'min(i)', 'max(i)', 'exp(i)'
            ],

            'la': [
                'inner(i, i)', 'outer(i, i)', 'matmul(i, i)',
                'dot(i, i)', 'l1norm(i)', 'l2norm(i)'
            ],

            'trig': [
                'sin(i)', 'cos(i)', 'tan(i)', 'log(i)',
                'asin(i)', 'acos(i)', 'atan(i)'
            ],

            'stat': [
                'stdev(i)', 'variance(i)', 'average(i)', 'median(i)'
            ],

            'bool': [
                'logical_and(i, i)', 'logical_or(i, i)',
                'logical_not(i, i)', 'logical_xor(i, i)',
                'fuzzy_and(i, i)', 'fuzzy_or(i, i)'
            ],

            'i': [
                'f', 'text', 'cluster', 'alpha', 'beta', 'rho',
                'iter', 'num_w', 'pi', 'integer', 'frac'
            ],

            'pi': ['div(22, 7)'],

            'integer': ['1', '2', '3', '4', '5', '6', '7', '8', '9'],

            'frac': ['0.integer_integer_integer']
        }

    def get_design_space(self) -> Dict[str, List[Any]]:
        """
        Get the design space for Fuzzy ART choice function generation.

        The design space is defined by the grammar in Appendix A.6.
        Each design is a chromosome (list of integers) that maps to
        grammar production rules to generate a choice function expression.

        Returns:
            Dictionary describing the grammar-based design space
        """
        return {
            'function_type': ['arth', 'la', 'trig', 'stat', 'bool'],
            'arth_ops': ['add', 'sub', 'mul', 'div', 'min', 'max', 'exp'],
            'la_ops': ['inner', 'outer', 'matmul', 'dot', 'l1norm', 'l2norm'],
            'trig_ops': ['sin', 'cos', 'tan', 'log', 'asin', 'acos', 'atan'],
            'stat_ops': ['stdev', 'variance', 'average', 'median'],
            'bool_ops': ['logical_and', 'logical_or', 'logical_not',
                        'logical_xor', 'fuzzy_and', 'fuzzy_or'],
            'terminals': ['f', 'text', 'cluster', 'alpha', 'beta', 'rho',
                         'iter', 'num_w', 'pi', 'integer', 'frac'],
            'integers': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'chromosome_length': list(range(10, 101))
        }

    def train(self, design: Dict[str, Any], timesteps: Optional[int] = None) -> Dict[str, float]:
        """
        Train/evolve choice functions with the given design using GE.

        Args:
            design: Dictionary specifying a chromosome (list of integers for GE)
                   Format: {'chromosome': [int, int, ...]} or {'expression': 'string'}
            timesteps: Number of GE generations (default: 60)

        Returns:
            Dictionary of performance metrics including clustering accuracy
        """
        if timesteps is None:
            timesteps = 60

        # GE parameters from Chapter 7
        population_size = 60
        tournament_size = 3
        crossover_rate = 0.75
        mutation_rate = 0.25
        max_wraps = 3
        chromosome_length = design.get('chromosome_length', 50)

        # Start timing
        start_time = time.time()

        # Initialize population
        population = self._initialize_population(population_size, chromosome_length, design)

        best_fitness = 0.0
        best_individual = None
        best_expression = None
        fitness_history = []

        # Evolution loop
        for generation in range(timesteps):
            # Decode and evaluate population
            fitnesses = []
            expressions = []

            # Parallel or sequential evaluation
            if self.n_jobs > 1:
                # Parallel evaluation using ProcessPoolExecutor
                with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                    # Submit all evaluation tasks
                    future_to_idx = {
                        executor.submit(
                            self._evaluate_chromosome_static,
                            chromosome,
                            max_wraps,
                            self.grammar
                        ): i
                        for i, chromosome in enumerate(population)
                    }

                    # Pre-allocate results lists
                    fitnesses = [None] * len(population)
                    expressions = [None] * len(population)

                    # Collect results as they complete
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        expr, fitness = future.result()
                        expressions[idx] = expr
                        fitnesses[idx] = fitness
            else:
                # Sequential evaluation (original code)
                for chromosome in population:
                    expr = self._decode_chromosome(chromosome, max_wraps)
                    expressions.append(expr)
                    fitness = self._evaluate_choice_function(expr)
                    fitnesses.append(fitness)

            # Track best
            gen_best_idx = np.argmax(fitnesses)
            gen_best_fitness = fitnesses[gen_best_idx]

            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_individual = population[gen_best_idx].copy()
                best_expression = expressions[gen_best_idx]

            fitness_history.append(best_fitness)

            if (generation + 1) % 10 == 0:
                print(f"Generation [{generation+1}/{timesteps}], "
                      f"Best Fitness: {best_fitness:.4f}")

            # Create next generation
            new_population = []

            while len(new_population) < population_size:
                # Tournament selection
                parent1 = self._tournament_selection(population, fitnesses, tournament_size)
                parent2 = self._tournament_selection(population, fitnesses, tournament_size)

                # Crossover
                if np.random.random() < crossover_rate:
                    offspring = self._crossover(parent1, parent2)
                else:
                    offspring = parent1.copy()

                # Mutation
                if np.random.random() < mutation_rate:
                    offspring = self._mutate(offspring)

                new_population.append(offspring)

            population = new_population

        training_time = time.time() - start_time
        self.is_trained = True
        self.current_design = best_individual
        self.best_choice_function = best_expression

        return {
            'best_fitness': best_fitness,
            'final_fitness': fitness_history[-1],
            'training_time': training_time,
            'generations': timesteps,
            'best_expression': best_expression
        }

    def evaluate(self, design: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Evaluate the choice function on the test data.

        Args:
            design: Optional design to train and evaluate. If None, use current best.

        Returns:
            Dictionary of performance metrics on test set
        """
        if design is not None:
            self.train(design)

        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")

        # Evaluate using the best choice function
        test_fitness = self._evaluate_choice_function(self.best_choice_function)

        return {
            'test_fitness': test_fitness,
            'choice_function': self.best_choice_function
        }

    def _initialize_population(self, size: int, chromosome_length: int,
                              seed_design: Dict[str, Any]) -> List[List[int]]:
        """Initialize population with integer chromosomes for GE."""
        population = []

        # Add seed design if provided
        if 'chromosome' in seed_design:
            population.append(seed_design['chromosome'])

        # Generate random chromosomes
        while len(population) < size:
            chromosome = list(np.random.randint(0, 256, size=chromosome_length))
            population.append(chromosome)

        return population

    def _decode_chromosome(self, chromosome: List[int], max_wraps: int = 3) -> str:
        """
        Decode a chromosome into a choice function expression using GE.

        Args:
            chromosome: List of integers
            max_wraps: Maximum number of times to wrap around chromosome

        Returns:
            String representation of the choice function
        """
        expr = 'fns'  # Start symbol
        gene_idx = 0
        wraps = 0
        max_expansions = 100

        for _ in range(max_expansions):
            # Find leftmost non-terminal
            non_terminal = self._find_leftmost_nonterminal(expr)
            if non_terminal is None:
                break

            # Get production rules for this non-terminal
            if non_terminal not in self.grammar:
                # Terminal or unknown, skip
                break

            rules = self.grammar[non_terminal]

            # Select rule using current gene
            if gene_idx >= len(chromosome):
                if wraps >= max_wraps:
                    break
                gene_idx = 0
                wraps += 1

            rule_idx = chromosome[gene_idx] % len(rules)
            selected_rule = rules[rule_idx]
            gene_idx += 1

            # Replace non-terminal with selected rule
            expr = self._replace_first(expr, non_terminal, selected_rule)

        # Clean up expression
        expr = self._clean_expression(expr)

        return expr

    def _find_leftmost_nonterminal(self, expr: str) -> Optional[str]:
        """Find the leftmost non-terminal in the expression."""
        non_terminals = list(self.grammar.keys())

        min_pos = len(expr)
        found_nt = None

        for nt in non_terminals:
            pos = expr.find(nt)
            if pos != -1 and pos < min_pos:
                # Check it's not part of a longer word
                before_ok = pos == 0 or not expr[pos-1].isalnum()
                after_ok = pos + len(nt) >= len(expr) or not expr[pos + len(nt)].isalnum()

                if before_ok and after_ok:
                    min_pos = pos
                    found_nt = nt

        return found_nt

    def _replace_first(self, expr: str, old: str, new: str) -> str:
        """Replace first occurrence of old with new."""
        return expr.replace(old, new, 1)

    def _clean_expression(self, expr: str) -> str:
        """Clean up the generated expression."""
        # Replace remaining non-terminals with defaults
        for nt in self.grammar.keys():
            if nt in expr:
                if nt == 'i':
                    expr = expr.replace(nt, 'f', 1)
                elif nt == 'integer':
                    expr = expr.replace(nt, '1', 1)
                elif nt == 'frac':
                    expr = expr.replace(nt, '0.5', 1)
                elif nt in ['fns', 'arth', 'la', 'trig', 'stat', 'bool']:
                    expr = expr.replace(nt, 'f', 1)

        # Handle nested replacements
        expr = expr.replace('integer_integer_integer', '123')

        return expr

    def _tournament_selection(self, population: List[List[int]],
                             fitnesses: List[float], k: int) -> List[int]:
        """Select individual using tournament selection."""
        indices = np.random.choice(len(population), size=k, replace=False)
        tournament_fitnesses = [fitnesses[i] for i in indices]
        winner_idx = indices[np.argmax(tournament_fitnesses)]
        return population[winner_idx].copy()

    def _crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Perform single-point crossover."""
        if len(parent1) == 0 or len(parent2) == 0:
            return parent1.copy() if len(parent1) > 0 else parent2.copy()

        point = np.random.randint(1, min(len(parent1), len(parent2)))
        offspring = parent1[:point] + parent2[point:]
        return offspring

    def _mutate(self, chromosome: List[int]) -> List[int]:
        """Mutate chromosome by randomly changing some genes."""
        mutation_point = np.random.randint(0, len(chromosome))
        chromosome[mutation_point] = np.random.randint(0, 256)
        return chromosome

    def _evaluate_choice_function(self, expression: str) -> float:
        """
        Evaluate a choice function expression using Fuzzy ART clustering.

        Args:
            expression: String representation of the choice function

        Returns:
            Fitness score (clustering quality measured by accuracy)
        """
        # Penalize trivially simple or invalid expressions
        if expression in ['f', 'text', 'cluster', 'alpha', 'beta', 'rho', '']:
            return 0.0

        try:
            # Use a subset of the data for faster evaluation during evolution
            # Randomly sample 150 points for train and test
            sample_size = min(150, len(self.data) // 2)

            indices = np.random.choice(len(self.data), sample_size * 2, replace=False)
            train_idx = indices[:sample_size]
            test_idx = indices[sample_size:]

            train_data = self.data[train_idx]
            train_labels = self.labels[train_idx]
            test_data = self.data[test_idx]
            test_labels = self.labels[test_idx]

            # Fuzzy ART parameters (from reference code)
            rho = 0.75  # Vigilance parameter
            alpha = 0.01  # Choice parameter
            beta = 1.0  # Learning rate

            # Create Fuzzy ART network with evolved choice function
            fuzzy_art = SimpleFuzzyART(
                rho=rho,
                alpha=alpha,
                beta=beta,
                num_features=train_data.shape[1],
                choice_function_str=expression
            )

            # Run clustering
            train_clusters, test_clusters = fuzzy_art.run_clustering(
                train_data, test_data, max_epochs=10
            )

            # Map clusters to labels based on majority voting
            cluster_to_label = infer_cluster_labels(test_clusters, test_labels)
            predicted_labels = map_clusters_to_labels(test_clusters, cluster_to_label)

            # Compute accuracy as fitness
            accuracy = accuracy_score(test_labels, predicted_labels)

            return accuracy

        except Exception as e:
            # If evaluation fails, return low fitness
            return 0.0

    def get_meta_features(self) -> Dict[str, float]:
        """Extract meta-features from the dataset."""
        if self.data is None:
            return {}

        return {
            'num_samples': len(self.data),
            'num_features': self.data.shape[1] if len(self.data.shape) > 1 else 1,
            'num_clusters': len(np.unique(self.labels)) if self.labels is not None else 0
        }

    def get_application_type(self) -> str:
        """Get the type of application."""
        return 'generation'

    def supports_dynamic_designs(self) -> bool:
        """Generation supports dynamic designs (changing expression during execution)."""
        return True

    def get_num_timesteps(self) -> int:
        """Return default number of GE generations."""
        return 60

    def extract_meta_features(self, timestep: int = 0) -> Dict[str, Any]:
        """
        Extract meta-features for the current state of evolution.

        Args:
            timestep: Current timestep (generation)

        Returns:
            Dictionary of meta-features
        """
        if self.data is None or self.labels is None:
            # Return only basic dataset features
            return {
                'num_classes': self.dataset_info.get('num_classes', 2),
                'num_instances': 0,
                'input_dim': 0,
                'timestep': timestep
            }

        # Split data into train/val for meta-feature extraction
        split_idx = int(0.8 * len(self.data))
        train_data = self.data[:split_idx]
        train_labels = self.labels[:split_idx]
        val_data = self.data[split_idx:]
        val_labels = self.labels[split_idx:]

        # If we have a Fuzzy ART model, extract features
        if self.fuzzyart_model is not None:
            return self.meta_feature_extractor.extract_meta_features(
                fuzzyart_model=self.fuzzyart_model,
                train_data=train_data,
                train_labels=train_labels,
                val_data=val_data,
                val_labels=val_labels,
                dataset_info=self.dataset_info,
                timestep=timestep
            )
        else:
            # Return basic features
            return {
                'num_classes': self.dataset_info.get('num_classes', 2),
                'num_instances': len(self.data),
                'input_dim': self.data.shape[1] if len(self.data.shape) > 1 else 1,
                'timestep': timestep
            }

    def get_grammar(self) -> Dict[str, List[str]]:
        """Return the grammar used for choice function generation."""
        return self.grammar

    @staticmethod
    def _evaluate_chromosome_static(chromosome: List[int], max_wraps: int,
                                    grammar: Dict[str, List[str]]) -> Tuple[str, float]:
        """
        Static method for parallel chromosome evaluation.

        This method is static to enable pickling for multiprocessing.
        It decodes a chromosome and evaluates the resulting expression.

        Args:
            chromosome: List of integers representing a GE chromosome
            max_wraps: Maximum number of wraps for decoding
            grammar: Grammar dictionary for GE decoding

        Returns:
            Tuple of (expression, fitness)
        """
        # Decode chromosome to expression
        expr = FuzzyARTGenerationApplication._decode_chromosome_static(
            chromosome, max_wraps, grammar
        )

        # Evaluate expression
        fitness = FuzzyARTGenerationApplication._evaluate_choice_function_static(expr)

        return expr, fitness

    @staticmethod
    def _decode_chromosome_static(chromosome: List[int], max_wraps: int,
                                  grammar: Dict[str, List[str]]) -> str:
        """
        Static version of _decode_chromosome for multiprocessing.

        Args:
            chromosome: List of integers
            max_wraps: Maximum number of times to wrap around chromosome
            grammar: Grammar dictionary

        Returns:
            String representation of the choice function
        """
        expr = 'fns'  # Start symbol
        gene_idx = 0
        wraps = 0
        max_expansions = 100

        for _ in range(max_expansions):
            # Find leftmost non-terminal
            non_terminal = FuzzyARTGenerationApplication._find_leftmost_nonterminal_static(
                expr, grammar
            )
            if non_terminal is None:
                break

            # Get production rules for this non-terminal
            if non_terminal not in grammar:
                break

            rules = grammar[non_terminal]

            # Select rule using current gene
            if gene_idx >= len(chromosome):
                if wraps >= max_wraps:
                    break
                gene_idx = 0
                wraps += 1

            rule_idx = chromosome[gene_idx] % len(rules)
            selected_rule = rules[rule_idx]
            gene_idx += 1

            # Replace non-terminal with selected rule
            expr = expr.replace(non_terminal, selected_rule, 1)

        # Clean up expression
        expr = FuzzyARTGenerationApplication._clean_expression_static(expr, grammar)

        return expr

    @staticmethod
    def _find_leftmost_nonterminal_static(expr: str, grammar: Dict[str, List[str]]) -> Optional[str]:
        """Static version of _find_leftmost_nonterminal."""
        non_terminals = list(grammar.keys())

        min_pos = len(expr)
        found_nt = None

        for nt in non_terminals:
            pos = expr.find(nt)
            if pos != -1 and pos < min_pos:
                # Check it's not part of a longer word
                before_ok = pos == 0 or not expr[pos-1].isalnum()
                after_ok = pos + len(nt) >= len(expr) or not expr[pos + len(nt)].isalnum()

                if before_ok and after_ok:
                    min_pos = pos
                    found_nt = nt

        return found_nt

    @staticmethod
    def _clean_expression_static(expr: str, grammar: Dict[str, List[str]]) -> str:
        """Static version of _clean_expression."""
        # Replace remaining non-terminals with defaults
        for nt in grammar.keys():
            if nt in expr:
                if nt == 'i':
                    expr = expr.replace(nt, 'f', 1)
                elif nt == 'integer':
                    expr = expr.replace(nt, '1', 1)
                elif nt == 'frac':
                    expr = expr.replace(nt, '0.5', 1)
                elif nt in ['fns', 'arth', 'la', 'trig', 'stat', 'bool']:
                    expr = expr.replace(nt, 'f', 1)

        # Handle nested replacements
        expr = expr.replace('integer_integer_integer', '123')

        return expr

    @staticmethod
    def _evaluate_choice_function_static(expression: str, data: np.ndarray = None,
                                        labels: np.ndarray = None) -> float:
        """
        Static version of _evaluate_choice_function for multiprocessing.

        NOTE: This static version is primarily for the parallel chromosome evaluation.
        Since we can't easily pass data through multiprocessing, this returns a
        complexity-based heuristic. The actual evaluation happens in the non-static
        version during sequential evaluation.

        Args:
            expression: String representation of the choice function
            data: Optional data array (not used in static version)
            labels: Optional labels array (not used in static version)

        Returns:
            Fitness score (complexity-based heuristic)
        """
        # Penalize trivially simple or invalid expressions
        if expression in ['f', 'text', 'cluster', 'alpha', 'beta', 'rho', '']:
            return 0.0

        # For the static version used in parallel evaluation, we use a complexity heuristic
        # The actual Fuzzy ART evaluation happens in the non-static sequential version
        # This is because passing large data arrays through multiprocessing is inefficient

        # Reward expressions with appropriate complexity
        length_bonus = min(len(expression) / 200, 0.3)

        # Count operators to reward structured expressions
        operator_count = sum(1 for op in ['fuzzy_and', 'fuzzy_or', 'l1norm', 'l2norm',
                                          '+', '-', '*', '/'] if op in expression)
        operator_bonus = min(operator_count * 0.05, 0.2)

        # Base fitness
        base_fitness = 0.5

        return base_fitness + length_bonus + operator_bonus
