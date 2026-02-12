"""
Auto-sklearn Wrapper for BaseApplication Interface

This module provides a wrapper that allows Auto-sklearn to work with any application
that implements the BaseApplication interface (CNN, Segmentation, Fuzzy ART).

Auto-sklearn replaces the design algorithm (GA/GE) and uses Bayesian optimization
with meta-learning to search the design space.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import time
from ConfigSpace import ConfigurationSpace, Configuration
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter
)
from smac.facade.hyperparameter_optimization_facade import HyperparameterOptimizationFacade as HPOFacade
from smac.scenario import Scenario


class AutoSklearnWrapper:
    """
    Wrapper to use Auto-sklearn's SMAC optimizer with BaseApplication interface.

    This replaces GA/GE as the design algorithm, using Bayesian optimization
    to search the design space defined by each application.
    """

    def __init__(
        self,
        application,
        time_budget: int = 3600,  # Total time budget in seconds
        per_run_time_limit: int = 300,  # Time limit per design evaluation
        n_jobs: int = 1,
        random_seed: int = 42
    ):
        """
        Initialize Auto-sklearn wrapper.

        Args:
            application: Application instance (CNN, Segmentation, or FuzzyART)
            time_budget: Total optimization time budget in seconds
            per_run_time_limit: Maximum time per single design evaluation
            n_jobs: Number of parallel jobs
            random_seed: Random seed for reproducibility
        """
        self.application = application
        self.time_budget = time_budget
        self.per_run_time_limit = per_run_time_limit
        self.n_jobs = n_jobs
        self.random_seed = random_seed

        # Optimization state
        self.config_space = None
        self.optimizer = None
        self.best_design = None
        self.best_performance = -np.inf
        self.evaluation_history = []

        # Statistics
        self.num_evaluations = 0
        self.total_time = 0.0

    def run_optimization(
        self,
        dataset_name: str,
        max_evaluations: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run Auto-sklearn optimization to find best design.

        Args:
            dataset_name: Name of dataset to use
            max_evaluations: Maximum number of design evaluations (if None, use time budget)

        Returns:
            Dictionary with optimization results
        """
        print(f"\n=== Auto-sklearn Optimization ===")
        print(f"Application: {self.application.__class__.__name__}")
        print(f"Dataset: {dataset_name}")
        print(f"Time budget: {self.time_budget}s")
        print(f"Max evaluations: {max_evaluations if max_evaluations else 'Time-based'}")
        print(f"{'='*50}\n")

        start_time = time.time()

        # Load dataset
        self.application.load_data()

        # Create configuration space from application design space
        self.config_space = self._create_config_space()

        # Set up SMAC scenario
        scenario = Scenario(
            configspace=self.config_space,
            deterministic=True,
            n_trials=max_evaluations if max_evaluations else 1000,
            walltime_limit=self.time_budget,
            n_workers=self.n_jobs,
            seed=self.random_seed
        )

        # Create SMAC optimizer
        self.optimizer = HPOFacade(
            scenario=scenario,
            target_function=self._evaluate_config,
            overwrite=True
        )

        # Run optimization
        print("Starting SMAC optimization...")
        incumbent = self.optimizer.optimize()

        self.total_time = time.time() - start_time

        # Convert best configuration to design
        self.best_design = self._config_to_design(incumbent)

        # Final evaluation on test set
        print(f"\n{'='*50}")
        print("Final Evaluation on Test Set")
        test_results = self.application.evaluate(self.best_design)
        print(f"Test Performance: {test_results.get('test_performance', 'N/A')}")
        print(f"{'='*50}\n")

        # Prepare results
        results = {
            'approach': 'Auto-sklearn',
            'best_design': self.best_design,
            'best_performance': self.best_performance,
            'num_evaluations': self.num_evaluations,
            'evaluation_history': self.evaluation_history,
            'test_results': test_results,
            'total_time': self.total_time,
            'incumbent_config': incumbent
        }

        print(f"\n{'='*50}")
        print("Auto-sklearn Optimization Complete")
        print(f"Total runtime: {self.total_time:.2f}s")
        print(f"Best performance: {self.best_performance:.4f}")
        print(f"Number of evaluations: {self.num_evaluations}")
        print(f"{'='*50}\n")

        return results

    def _create_config_space(self) -> ConfigurationSpace:
        """
        Create ConfigSpace from application's design space.

        Returns:
            ConfigurationSpace for SMAC
        """
        cs = ConfigurationSpace(seed=self.random_seed)
        design_space = self.application.get_design_space()

        for param_name, param_values in design_space.items():
            if len(param_values) == 3 and param_values[2] == 'continuous':
                # Continuous parameter
                min_val, max_val, _ = param_values
                hp = UniformFloatHyperparameter(
                    name=param_name,
                    lower=min_val,
                    upper=max_val
                )
            elif all(isinstance(v, (int, np.integer)) for v in param_values):
                # Integer categorical (treat as ordinal for some applications)
                hp = CategoricalHyperparameter(
                    name=param_name,
                    choices=param_values
                )
            else:
                # General categorical
                hp = CategoricalHyperparameter(
                    name=param_name,
                    choices=param_values
                )

            cs.add_hyperparameter(hp)

        return cs

    def _evaluate_config(self, config: Configuration, seed: int = 0) -> float:
        """
        Evaluate a configuration (design).

        This is called by SMAC for each candidate design.

        Args:
            config: Configuration from SMAC
            seed: Random seed (required by SMAC interface)

        Returns:
            Cost (negative performance, since SMAC minimizes)
        """
        # Convert configuration to design dictionary
        design = self._config_to_design(config)

        # Evaluate design
        try:
            results = self.application.train(design, timesteps=1)

            # Extract performance metric
            performance = self._extract_performance(results)

            # Track evaluation
            self.num_evaluations += 1
            self.evaluation_history.append({
                'design': design,
                'performance': performance,
                'evaluation_num': self.num_evaluations
            })

            # Update best
            if performance > self.best_performance:
                self.best_performance = performance
                self.best_design = design

            print(f"Evaluation {self.num_evaluations}: Performance = {performance:.4f}")

            # SMAC minimizes, so return negative performance
            return -performance

        except Exception as e:
            print(f"Evaluation {self.num_evaluations} failed: {e}")
            # Return large cost for failed evaluations
            return 1000.0

    def _config_to_design(self, config: Configuration) -> Dict[str, Any]:
        """
        Convert SMAC Configuration to application design dictionary.

        Args:
            config: Configuration from SMAC

        Returns:
            Design dictionary compatible with application
        """
        design = {}
        for param_name in config:
            design[param_name] = config[param_name]
        return design

    def _extract_performance(self, results: Dict[str, float]) -> float:
        """
        Extract primary performance metric from training results.

        Args:
            results: Results dictionary from application.train()

        Returns:
            Performance value (higher is better)
        """
        # Try common metric names
        for metric_name in ['accuracy', 'val_accuracy', 'dice_coefficient', 'performance']:
            if metric_name in results:
                return results[metric_name]

        # Default: return first value
        return list(results.values())[0] if results else 0.0

    def get_optimization_trajectory(self) -> Tuple[List[float], List[int]]:
        """
        Get optimization trajectory over time.

        Returns:
            Tuple of (performance_values, evaluation_numbers)
        """
        performances = [entry['performance'] for entry in self.evaluation_history]
        eval_nums = [entry['evaluation_num'] for entry in self.evaluation_history]
        return performances, eval_nums

    def get_incumbent_trajectory(self) -> Tuple[List[float], List[int]]:
        """
        Get incumbent (best so far) trajectory.

        Returns:
            Tuple of (best_performance_values, evaluation_numbers)
        """
        best_so_far = []
        current_best = -np.inf

        for entry in self.evaluation_history:
            if entry['performance'] > current_best:
                current_best = entry['performance']
            best_so_far.append(current_best)

        eval_nums = [entry['evaluation_num'] for entry in self.evaluation_history]
        return best_so_far, eval_nums
