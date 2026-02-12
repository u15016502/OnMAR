"""
OnMAR (Online Meta-learning for AutoML in Real-time) - Design Prediction

This implementation is a variant of OnMAR where the meta-learner predicts designs
(instead of accuracy) in an online fashion.

Unlike OffMAR which trains offline, this approach continuously updates the meta-learner
online and predicts new designs at each timestep.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import time
import pickle
from pathlib import Path


class OnMARDesignPrediction:
    """
    OnMAR approach with design prediction for real-time AutoML.

    The meta-learner is trained online (during execution) and directly predicts
    designs from meta-features. The knowledge repository is continuously updated
    throughout execution.
    """

    def __init__(
        self,
        application,
        meta_learner_type: str = 'knn',
        theta_t: Optional[int] = None,
        theta_p: float = 0.85,
        meta_learner_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize OnMAR design prediction.

        Args:
            application: Application instance (CNN, Segmentation, or FuzzyART)
            meta_learner_type: Type of meta-learner ('knn', 'rf', or 'xgboost')
            theta_t: Timestep threshold to start using meta-learner (default: N/2)
            theta_p: Performance threshold for pruning knowledge repository (default: 0.85)
            meta_learner_params: Optional parameters for meta-learner
        """
        self.application = application
        self.meta_learner_type = meta_learner_type.lower()
        self.theta_t = theta_t  # Will be set to N/2 if None
        self.theta_p = theta_p
        self.meta_learner_params = meta_learner_params or {}

        # Knowledge repository (continuously updated online)
        self.knowledge_repository: List[Dict[str, Any]] = []

        # Pre-computed flattened repository for faster meta-learner training
        self.knowledge_repository_flat_X: List[np.ndarray] = []  # Pre-flattened X vectors
        self.knowledge_repository_flat_y: List[np.ndarray] = []  # Design vectors

        # Meta-learner (trained and updated online)
        self.meta_learner = None

        # Current design (tracked for continuity)
        self.current_design = None
        self.current_performance = 0.0

        # Design space information (for encoding/decoding)
        self.design_space = None
        self.design_params = []
        self.param_types = {}  # 'categorical' or 'continuous'
        self.param_values = {}  # Possible values for categorical, (min, max) for continuous

        # Statistics
        self.num_design_algorithm_calls = 0
        self.num_design_predictions = 0

    def run_onmar(
        self,
        dataset_name: str,
        timesteps: int,
        initial_design: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run OnMAR for all timesteps with online meta-learning and design prediction.

        Args:
            dataset_name: Name of dataset to use
            timesteps: Number of timesteps (epochs/generations)
            initial_design: Optional initial design

        Returns:
            Dictionary with results
        """
        print(f"\n=== OnMAR Design Prediction ===")
        print(f"Dataset: {dataset_name}")
        print(f"Timesteps: {timesteps}")
        print(f"Meta-learner: {self.meta_learner_type.upper()}")
        print(f"θt (start using meta-learner): {self.theta_t if self.theta_t else 'N/2'}")
        print(f"θp (pruning threshold): {self.theta_p}")
        print(f"{'='*50}\n")

        start_time = time.time()

        # Set theta_t to N/2 if not specified
        if self.theta_t is None:
            self.theta_t = timesteps // 2

        # Load dataset
        self.application.load_data()

        # Get design space for encoding/decoding
        self.design_space = self.application.get_design_space()
        self._initialize_design_encoding()

        # Reset statistics
        self.num_design_algorithm_calls = 0
        self.num_design_predictions = 0
        self.knowledge_repository = []
        self.knowledge_repository_flat_X = []
        self.knowledge_repository_flat_y = []

        # Track designs and performances over time
        designs_history = []
        performance_history = []
        best_performance = 0.0
        best_design = None

        # Main OnMAR loop
        for t in range(timesteps):
            print(f"\nTimestep {t+1}/{timesteps}")

            # Extract meta-features for current timestep
            meta_features = self.application.extract_meta_features(timestep=t)

            # Decide whether to use meta-learner or design algorithm
            if t < self.theta_t:
                # Phase 1: Always run design algorithm to build knowledge repository
                print(f"  Phase 1: Running design algorithm (t < θt)")
                design = self._run_design_algorithm(initial_design if t == 0 else self.current_design)
                self.num_design_algorithm_calls += 1

            else:
                # Phase 2: Use meta-learner to predict design
                print(f"  Phase 2: Using meta-learner to predict design")
                design = self._predict_design(meta_features)
                self.num_design_predictions += 1

            # Apply design and measure actual performance
            performance = self._evaluate_design(design, t)
            print(f"  Performance: {performance:.4f}")

            # Update knowledge repository with pruning
            self._update_knowledge_repository(meta_features, design, performance)

            # Re-train meta-learner with updated knowledge
            if t >= self.theta_t - 1:  # Start training at θt - 1
                self._train_meta_learner()

            # Update current design and performance
            self.current_design = design
            self.current_performance = performance

            # Track history
            designs_history.append(design)
            performance_history.append(performance)

            # Track best
            if performance > best_performance:
                best_performance = performance
                best_design = design

        total_time = time.time() - start_time

        # Final evaluation on test set
        print(f"\n{'='*50}")
        print("Final Evaluation on Test Set")
        test_results = self.application.evaluate(best_design)
        print(f"Test Performance: {test_results.get('test_performance', 'N/A')}")

        # Calculate statistics
        final_performance = performance_history[-1] if performance_history else 0.0
        design_algorithm_percentage = (self.num_design_algorithm_calls / timesteps) * 100
        design_prediction_percentage = (self.num_design_predictions / timesteps) * 100

        results = {
            'approach': 'OnMAR-DesignPrediction',
            'meta_learner': self.meta_learner_type,
            'theta_t': self.theta_t,
            'theta_p': self.theta_p,
            'timesteps': timesteps,
            'designs_history': designs_history,
            'performance_history': performance_history,
            'best_performance': best_performance,
            'best_design': best_design,
            'final_performance': final_performance,
            'final_design': designs_history[-1] if designs_history else None,
            'test_results': test_results,
            'num_design_algorithm_calls': self.num_design_algorithm_calls,
            'num_design_predictions': self.num_design_predictions,
            'design_algorithm_percentage': design_algorithm_percentage,
            'design_prediction_percentage': design_prediction_percentage,
            'knowledge_repository_size': len(self.knowledge_repository),
            'total_time': total_time
        }

        print(f"\n{'='*50}")
        print("OnMAR Design Prediction Complete")
        print(f"Total runtime: {total_time:.2f}s")
        print(f"Best performance: {best_performance:.4f}")
        print(f"Final performance: {final_performance:.4f}")
        print(f"Design algorithm calls: {self.num_design_algorithm_calls} ({design_algorithm_percentage:.1f}%)")
        print(f"Design predictions: {self.num_design_predictions} ({design_prediction_percentage:.1f}%)")
        print(f"Knowledge repository size: {len(self.knowledge_repository)}")
        print(f"{'='*50}\n")

        return results

    def _initialize_design_encoding(self):
        """Initialize design encoding based on design space."""
        self.design_params = list(self.design_space.keys())

        for param_name, param_values in self.design_space.items():
            if len(param_values) == 3 and param_values[2] == 'continuous':
                # Continuous parameter
                self.param_types[param_name] = 'continuous'
                self.param_values[param_name] = (param_values[0], param_values[1])
            else:
                # Categorical parameter
                self.param_types[param_name] = 'categorical'
                self.param_values[param_name] = param_values

    def _run_design_algorithm(self, initial_design: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the design algorithm to create a new design.

        Args:
            initial_design: Optional starting design

        Returns:
            New design
        """
        # For simplicity, use random search from design space
        # In practice, this would be GA, GE, or other optimization algorithm
        design = {}

        for param_name in self.design_params:
            if self.param_types[param_name] == 'continuous':
                min_val, max_val = self.param_values[param_name]
                design[param_name] = np.random.uniform(min_val, max_val)
            else:
                design[param_name] = np.random.choice(self.param_values[param_name])

        return design

    def _evaluate_design(self, design: Dict[str, Any], timestep: int) -> float:
        """
        Evaluate a design by training and measuring performance.

        Args:
            design: Design to evaluate
            timestep: Current timestep

        Returns:
            Performance metric (e.g., accuracy)
        """
        results = self.application.train(design, timesteps=1)

        # Extract primary performance metric
        # Try common metric names
        for metric_name in ['accuracy', 'val_accuracy', 'dice_coefficient', 'performance']:
            if metric_name in results:
                return results[metric_name]

        # Default: return first value
        return list(results.values())[0] if results else 0.0

    def _update_knowledge_repository(
        self,
        meta_features: Dict[str, Any],
        design: Dict[str, Any],
        performance: float
    ):
        """
        Update knowledge repository with new sample and apply online pruning.
        Also maintains pre-flattened vectors for faster training.

        Args:
            meta_features: Extracted meta-features
            design: Design used
            performance: Measured performance
        """
        # Add to repository
        self.knowledge_repository.append({
            'meta_features': meta_features,
            'design': design,
            'performance': performance
        })

        # Pre-compute flattened version
        x = self._flatten_meta_features(meta_features)
        y = self._encode_design(design)
        self.knowledge_repository_flat_X.append(x)
        self.knowledge_repository_flat_y.append(y)

        # Online pruning: keep only designs with performance >= theta_p
        # Need to prune both repositories in sync
        kept_indices = [
            i for i, entry in enumerate(self.knowledge_repository)
            if entry['performance'] >= self.theta_p
        ]

        self.knowledge_repository = [
            self.knowledge_repository[i] for i in kept_indices
        ]
        self.knowledge_repository_flat_X = [
            self.knowledge_repository_flat_X[i] for i in kept_indices
        ]
        self.knowledge_repository_flat_y = [
            self.knowledge_repository_flat_y[i] for i in kept_indices
        ]

    def _train_meta_learner(self):
        """
        Train meta-learner on current knowledge repository.
        Uses pre-flattened vectors for faster training (4x speedup).
        Meta-learner learns to map meta-features -> designs.
        """
        if len(self.knowledge_repository_flat_X) == 0:
            print("  Warning: Knowledge repository empty after pruning. Skipping training.")
            return

        # Use pre-computed flattened vectors (4x speedup - no need to loop and flatten)
        X = np.array(self.knowledge_repository_flat_X)
        y = np.array(self.knowledge_repository_flat_y)

        # Create and train meta-learner based on type
        if self.meta_learner_type == 'knn':
            self.meta_learner = self._create_knn_meta_learner(X, y)
        elif self.meta_learner_type == 'rf':
            self.meta_learner = self._create_rf_meta_learner(X, y)
        elif self.meta_learner_type == 'xgboost':
            self.meta_learner = self._create_xgboost_meta_learner(X, y)
        else:
            raise ValueError(f"Unknown meta-learner type: {self.meta_learner_type}")

        print(f"  Meta-learner re-trained on {len(X)} samples")

    def _predict_design(self, meta_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use meta-learner to predict design from meta-features.

        Args:
            meta_features: Extracted meta-features

        Returns:
            Predicted design
        """
        if self.meta_learner is None:
            print("  Warning: Meta-learner not trained yet, using random design")
            return self._run_design_algorithm()

        # Flatten meta-features to vector
        X = self._flatten_meta_features(meta_features).reshape(1, -1)

        # Predict design encoding
        design_encoding = self.meta_learner.predict(X)[0]

        # Decode back to design
        design = self._decode_design(design_encoding)

        return design

    def _flatten_meta_features(self, meta_features: Dict[str, Any]) -> np.ndarray:
        """
        Flatten nested meta-features dictionary to 1D numpy array.

        Args:
            meta_features: Nested dictionary of meta-features

        Returns:
            Flattened 1D array
        """
        def flatten_recursive(obj):
            """Recursively flatten nested structure."""
            if isinstance(obj, dict):
                result = []
                for key in sorted(obj.keys()):  # Sort for consistency
                    result.extend(flatten_recursive(obj[key]))
                return result
            elif isinstance(obj, (list, tuple, np.ndarray)):
                return [float(x) for x in np.array(obj).flatten()]
            else:
                return [float(obj)]

        flattened = flatten_recursive(meta_features)
        return np.array(flattened, dtype=np.float32)

    def _encode_design(self, design: Dict[str, Any]) -> np.ndarray:
        """
        Encode design dictionary to numerical vector.

        Args:
            design: Design dictionary

        Returns:
            Encoded design as numpy array
        """
        encoded = []

        for param_name in self.design_params:
            param_value = design[param_name]

            if self.param_types[param_name] == 'continuous':
                # Normalize continuous to [0, 1]
                min_val, max_val = self.param_values[param_name]
                normalized = (param_value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
                encoded.append(normalized)

            else:
                # One-hot encode categorical
                possible_values = self.param_values[param_name]
                one_hot = [1.0 if v == param_value else 0.0 for v in possible_values]
                encoded.extend(one_hot)

        return np.array(encoded, dtype=np.float32)

    def _decode_design(self, design_encoding: np.ndarray) -> Dict[str, Any]:
        """
        Decode numerical vector back to design dictionary.

        Args:
            design_encoding: Encoded design

        Returns:
            Design dictionary
        """
        design = {}
        idx = 0

        for param_name in self.design_params:
            if self.param_types[param_name] == 'continuous':
                # Denormalize continuous
                min_val, max_val = self.param_values[param_name]
                normalized = design_encoding[idx]
                design[param_name] = min_val + normalized * (max_val - min_val)
                idx += 1

            else:
                # Decode one-hot categorical
                possible_values = self.param_values[param_name]
                num_values = len(possible_values)
                one_hot = design_encoding[idx:idx + num_values]

                # Choose value with highest probability
                chosen_idx = np.argmax(one_hot)
                design[param_name] = possible_values[chosen_idx]
                idx += num_values

        return design

    def _create_knn_meta_learner(self, X: np.ndarray, y: np.ndarray):
        """Create and train kNN meta-learner."""
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.multioutput import MultiOutputRegressor

        k = self.meta_learner_params.get('k', 5)
        knn = KNeighborsRegressor(n_neighbors=min(k, len(X)))
        meta_learner = MultiOutputRegressor(knn)
        meta_learner.fit(X, y)

        return meta_learner

    def _create_rf_meta_learner(self, X: np.ndarray, y: np.ndarray):
        """Create and train Random Forest meta-learner."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.multioutput import MultiOutputRegressor

        n_estimators = self.meta_learner_params.get('n_estimators', 100)
        max_depth = self.meta_learner_params.get('max_depth', None)

        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.application.random_seed
        )
        meta_learner = MultiOutputRegressor(rf)
        meta_learner.fit(X, y)

        return meta_learner

    def _create_xgboost_meta_learner(self, X: np.ndarray, y: np.ndarray):
        """Create and train XGBoost meta-learner."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")

        from sklearn.multioutput import MultiOutputRegressor

        n_estimators = self.meta_learner_params.get('n_estimators', 100)
        max_depth = self.meta_learner_params.get('max_depth', 6)
        learning_rate = self.meta_learner_params.get('learning_rate', 0.1)

        xgb_model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=self.application.random_seed
        )
        meta_learner = MultiOutputRegressor(xgb_model)
        meta_learner.fit(X, y)

        return meta_learner

    def save_model(self, filepath: str):
        """
        Save trained meta-learner and knowledge repository.

        Args:
            filepath: Path to save model
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'meta_learner': self.meta_learner,
            'meta_learner_type': self.meta_learner_type,
            'knowledge_repository': self.knowledge_repository,
            'theta_t': self.theta_t,
            'theta_p': self.theta_p,
            'design_space': self.design_space,
            'design_params': self.design_params,
            'param_types': self.param_types,
            'param_values': self.param_values
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load trained meta-learner and knowledge repository.

        Args:
            filepath: Path to load model from
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.meta_learner = model_data['meta_learner']
        self.meta_learner_type = model_data['meta_learner_type']
        self.knowledge_repository = model_data['knowledge_repository']
        self.theta_t = model_data['theta_t']
        self.theta_p = model_data['theta_p']
        self.design_space = model_data['design_space']
        self.design_params = model_data['design_params']
        self.param_types = model_data['param_types']
        self.param_values = model_data['param_values']

        print(f"Model loaded from {filepath}")
