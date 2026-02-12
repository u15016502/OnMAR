"""
CNN Configuration Application

Automated configuration of convolutional neural networks for image classification.
Design space based on Appendix A.3 of the thesis.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from typing import Dict, Any, List, Optional
import time
import torch
import torch.nn as nn
import torch.optim as optim
from applications.base_application import BaseApplication
from datasets.image_datasets import ImageDatasetLoader
from applications.configuration.cnn.cnn_model import ConfigurableCNN
from applications.configuration.cnn.meta_features import CNNMetaFeatureExtractor


class CNNConfigurationApplication(BaseApplication):
    """
    CNN configuration application for automated hyperparameter optimization.
    """

    def __init__(self, dataset_name: str, random_seed: int = 42, device: Optional[str] = None,
                 use_data_parallel: bool = True):
        """
        Initialize the CNN configuration application.

        Args:
            dataset_name: Name of the dataset (mnist, fashion-mnist, cifar-10, cifar-100)
            random_seed: Random seed for reproducibility
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            use_data_parallel: Use DataParallel for multi-GPU training (default: True)
        """
        super().__init__(dataset_name, random_seed)

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Multi-GPU configuration
        self.use_data_parallel = use_data_parallel and torch.cuda.device_count() > 1
        if self.use_data_parallel:
            print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")

        # Initialize dataset loader
        self.dataset_loader = ImageDatasetLoader(dataset_name, random_seed=random_seed)
        self.dataset_info = self.dataset_loader.get_dataset_info()

        # Data loaders (initialized in load_data)
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # Model and training state
        self.model = None
        self.current_design = None

        # Meta-feature extractor
        self.meta_feature_extractor = None

    def load_data(self) -> None:
        """Load and prepare the dataset."""
        self.train_loader, self.val_loader, self.test_loader = self.dataset_loader.load_dataset(
            val_split=0.1,
            batch_size=128,
            num_workers=4
        )
        print(f"Loaded {self.dataset_name}: "
              f"Train={len(self.train_loader.dataset)}, "
              f"Val={len(self.val_loader.dataset)}, "
              f"Test={len(self.test_loader.dataset)}")

    def get_design_space(self) -> Dict[str, List[Any]]:
        """
        Get the design space for CNN configuration.

        Returns design space from Appendix A.3:
        - Convolutional filters
        - Batch normalization
        - Activation function
        - Dropout
        - 2D Max-pooling
        - Dense nodes
        - Optimizer
        """
        return {
            'conv_filters': [8, 16, 32, 64, 128, 256, 512, 1024, 2048],
            'batch_norm': [0, 1],  # 0 = False, 1 = True
            'activation': [1, 2, 3, 4, 5, 6, 7, 8, 9],  # ELU, GELU, ReLU, SELU, Sigmoid, Softmax, Softplus, Swish, Tanh
            'dropout': [-1.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # -1.0 means no dropout
            'max_pool_size': [2, 4, 6, 8],
            'dense_nodes': [16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
            'optimizer': [1, 2, 3, 4, 5, 6, 7, 8],  # Adam, Adamax, RMSprop, AdaGrad, AdaDelta, SGD, Nadam, Ftrl
            'learning_rate': [0.0001, 0.1, 'continuous'],
            'num_conv_layers': [1, 2, 3],  # Number of convolutional blocks
        }

    def train(self, design: Dict[str, Any], timesteps: Optional[int] = None) -> Dict[str, float]:
        """
        Train the CNN with the given design.

        Args:
            design: Dictionary specifying the CNN configuration
            timesteps: Number of training epochs (default: 50)

        Returns:
            Dictionary of performance metrics
        """
        if not self.validate_design(design):
            raise ValueError(f"Invalid design: {design}")

        # Set default timesteps
        if timesteps is None:
            timesteps = 50

        # Build model
        self.model = ConfigurableCNN(
            input_shape=self.dataset_info['input_shape'],
            num_classes=self.dataset_info['num_classes'],
            design=design
        ).to(self.device)

        # Wrap with DataParallel for multi-GPU if available
        if self.use_data_parallel:
            self.model = nn.DataParallel(self.model)

        self.current_design = design

        # Initialize meta-feature extractor
        self.meta_feature_extractor = CNNMetaFeatureExtractor(self.model, self.device)

        # Get optimizer
        optimizer = self._get_optimizer(design)

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Training loop
        start_time = time.time()
        best_val_acc = 0.0
        train_losses = []
        val_accuracies = []

        for epoch in range(timesteps):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)

            # Validation phase
            val_acc = self._evaluate_loader(self.val_loader)
            val_accuracies.append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{timesteps}], "
                      f"Loss: {avg_train_loss:.4f}, "
                      f"Val Acc: {val_acc:.4f}")

        training_time = time.time() - start_time
        self.is_trained = True

        return {
            'val_accuracy': best_val_acc,
            'final_val_accuracy': val_accuracies[-1],
            'final_train_loss': train_losses[-1],
            'training_time': training_time
        }

    def evaluate(self, design: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Evaluate the CNN on the test set.

        Args:
            design: Optional design to train and evaluate. If None, use current model.

        Returns:
            Dictionary of performance metrics on test set
        """
        if design is not None:
            # Train a new model with this design
            self.train(design)

        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")

        # Evaluate on test set
        test_acc = self._evaluate_loader(self.test_loader)

        return {
            'test_accuracy': test_acc
        }

    def _evaluate_loader(self, data_loader) -> float:
        """Evaluate the model on a given data loader."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        return correct / total

    def _get_optimizer(self, design: Dict[str, Any]) -> optim.Optimizer:
        """Create optimizer based on design specification."""
        lr = design.get('learning_rate', 0.001)
        opt_type = design.get('optimizer', 1)

        params = self.model.parameters()

        if opt_type == 1:  # Adam
            return optim.Adam(params, lr=lr)
        elif opt_type == 2:  # Adamax
            return optim.Adamax(params, lr=lr)
        elif opt_type == 3:  # RMSprop
            return optim.RMSprop(params, lr=lr)
        elif opt_type == 4:  # AdaGrad
            return optim.Adagrad(params, lr=lr)
        elif opt_type == 5:  # AdaDelta
            return optim.Adadelta(params, lr=lr)
        elif opt_type == 6:  # SGD
            return optim.SGD(params, lr=lr, momentum=0.9)
        elif opt_type == 7:  # Nadam
            return optim.NAdam(params, lr=lr)
        else:  # Default to Adam
            return optim.Adam(params, lr=lr)

    def get_meta_features(self) -> Dict[str, float]:
        """Extract meta-features from the dataset."""
        return {
            'num_train_samples': len(self.train_loader.dataset),
            'num_val_samples': len(self.val_loader.dataset),
            'num_test_samples': len(self.test_loader.dataset),
            'num_classes': self.dataset_info['num_classes'],
            'input_channels': self.dataset_info['channels'],
            'input_height': self.dataset_info['height'],
            'input_width': self.dataset_info['width'],
            'total_pixels': self.dataset_info['height'] * self.dataset_info['width'] * self.dataset_info['channels']
        }

    def get_application_type(self) -> str:
        """Get the type of application."""
        return 'configuration'

    def supports_dynamic_designs(self) -> bool:
        """CNN supports dynamic designs (changing hyperparameters during training)."""
        return True

    def get_num_timesteps(self) -> int:
        """Return default number of training epochs."""
        return 50

    def extract_meta_features(self, timestep: int = 0) -> Dict[str, Any]:
        """
        Extract meta-features for the current state of training.

        Args:
            timestep: Current training timestep (epoch)

        Returns:
            Dictionary of meta-features
        """
        if self.meta_feature_extractor is None:
            # If no model yet, return only basic dataset features
            return {
                'num_classes': self.dataset_info['num_classes'],
                'num_instances': len(self.train_loader.dataset) if self.train_loader else 0,
                'input_channels': self.dataset_info.get('channels', 3),
                'input_height': self.dataset_info.get('height', 32),
                'input_width': self.dataset_info.get('width', 32),
                'timestep': timestep
            }

        return self.meta_feature_extractor.extract_meta_features(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            dataset_info=self.dataset_info,
            timestep=timestep
        )
