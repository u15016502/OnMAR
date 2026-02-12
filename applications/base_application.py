"""
Base Application Interface

This module defines the abstract base class that all applications must inherit from.
It provides a standard interface for OnMAR, OffMAR, and AutoSklearn to interact with.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
from functools import lru_cache
import numpy as np


class BaseApplication(ABC):
    """
    Abstract base class for all AutoML applications.

    This provides a plug-and-play interface for meta-learning approaches
    (OnMAR, OffMAR, AutoSklearn) to interact with different application types
    (configuration, composition, generation).
    """

    def __init__(self, dataset_name: str, random_seed: int = 42):
        """
        Initialize the application.

        Args:
            dataset_name: Name of the dataset to use
            random_seed: Random seed for reproducibility
        """
        self.dataset_name = dataset_name
        self.random_seed = random_seed
        self.is_trained = False

    @abstractmethod
    def load_data(self) -> None:
        """
        Load and prepare the dataset.

        This should handle downloading if necessary and splitting into
        train/validation/test sets.
        """
        pass

    @abstractmethod
    def get_design_space(self) -> Dict[str, List[Any]]:
        """
        Get the design space for this application.

        Returns:
            Dictionary mapping design option names to their possible values.
            For continuous parameters, return [min_value, max_value, 'continuous'].
            For categorical parameters, return list of possible values.

        Example:
            {
                'learning_rate': [0.0001, 0.1, 'continuous'],
                'optimizer': ['adam', 'sgd', 'rmsprop'],
                'num_layers': [1, 2, 3, 4, 5]
            }
        """
        pass

    @abstractmethod
    def train(self, design: Dict[str, Any], timesteps: Optional[int] = None) -> Dict[str, float]:
        """
        Train the application with the given design.

        Args:
            design: Dictionary mapping design option names to chosen values
            timesteps: Optional number of timesteps/epochs for dynamic designs

        Returns:
            Dictionary of performance metrics

        Example:
            {
                'accuracy': 0.95,
                'loss': 0.123,
                'training_time': 45.6
            }
        """
        pass

    @abstractmethod
    def evaluate(self, design: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Evaluate the current or given design on the test set.

        Args:
            design: Optional design to evaluate. If None, evaluate the last trained design.

        Returns:
            Dictionary of performance metrics on test set
        """
        pass

    def validate_design(self, design: Dict[str, Any]) -> bool:
        """
        Validate that a design is within the design space.

        Args:
            design: Design to validate

        Returns:
            True if design is valid, False otherwise
        """
        design_space = self.get_design_space()

        for param_name, param_value in design.items():
            if param_name not in design_space:
                return False

            valid_values = design_space[param_name]

            # Check continuous parameters
            if len(valid_values) == 3 and valid_values[2] == 'continuous':
                min_val, max_val, _ = valid_values
                if not (min_val <= param_value <= max_val):
                    return False
            # Check categorical parameters
            elif param_value not in valid_values:
                return False

        return True

    def get_meta_features(self) -> Dict[str, float]:
        """
        Extract meta-features from the dataset for meta-learning.

        Returns:
            Dictionary of meta-features

        Example:
            {
                'num_samples': 60000,
                'num_features': 784,
                'num_classes': 10,
                'class_balance': 0.95
            }
        """
        # Default implementation - can be overridden by subclasses
        return {}

    def reset(self) -> None:
        """
        Reset the application to initial state.
        Useful for running multiple experiments.
        """
        self.is_trained = False

    @abstractmethod
    def get_application_type(self) -> str:
        """
        Get the type of application.

        Returns:
            One of: 'configuration', 'composition', 'generation'
        """
        pass

    def supports_dynamic_designs(self) -> bool:
        """
        Check if this application supports dynamic designs (designs that change over time).

        Returns:
            True if dynamic designs are supported, False otherwise
        """
        return False

    def get_num_timesteps(self) -> int:
        """
        Get the number of timesteps for dynamic designs.

        Returns:
            Number of timesteps (e.g., epochs for training)
        """
        return 1
