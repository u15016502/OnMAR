"""
Configurable CNN Model

A flexible CNN architecture that can be configured based on design specifications.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfigurableCNN(nn.Module):
    """
    A configurable CNN that builds its architecture based on design parameters.
    """

    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int, design: Dict[str, Any]):
        """
        Initialize the configurable CNN.

        Args:
            input_shape: (channels, height, width)
            num_classes: Number of output classes
            design: Dictionary containing design parameters
        """
        super(ConfigurableCNN, self).__init__()

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.design = design

        # Extract design parameters
        self.conv_filters = design.get('conv_filters', 64)
        self.use_batch_norm = design.get('batch_norm', 1) == 1
        self.activation_type = design.get('activation', 3)  # Default to ReLU
        self.dropout_rate = design.get('dropout', -1.0)
        self.max_pool_size = design.get('max_pool_size', 2)
        self.dense_nodes = design.get('dense_nodes', 128)
        self.num_conv_layers = design.get('num_conv_layers', 2)

        # Build the network
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList() if self.use_batch_norm else None

        # Build convolutional layers
        in_channels = input_shape[0]
        current_filters = self.conv_filters

        for i in range(self.num_conv_layers):
            # Convolutional layer
            self.conv_layers.append(
                nn.Conv2d(in_channels, current_filters, kernel_size=3, padding=1)
            )

            # Batch normalization
            if self.use_batch_norm:
                self.bn_layers.append(nn.BatchNorm2d(current_filters))

            in_channels = current_filters
            # Optionally increase filters in deeper layers
            current_filters = min(current_filters * 2, 512)

        # Calculate size after conv and pooling layers
        self.feature_size = self._calculate_feature_size()

        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size, self.dense_nodes)
        if self.use_batch_norm:
            self.bn_fc = nn.BatchNorm1d(self.dense_nodes)
        self.fc2 = nn.Linear(self.dense_nodes, num_classes)

        # Dropout
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        else:
            self.dropout = None

    def _calculate_feature_size(self) -> int:
        """Calculate the size of features after conv and pooling layers."""
        with torch.no_grad():
            x = torch.zeros(1, *self.input_shape)
            for i in range(self.num_conv_layers):
                x = self.conv_layers[i](x)
                # Only apply pooling if spatial dimensions are large enough
                h, w = x.shape[2], x.shape[3]
                if h >= self.max_pool_size and w >= self.max_pool_size:
                    x = F.max_pool2d(x, self.max_pool_size)
                elif h > 1 and w > 1:
                    # Use smaller pooling if dimensions don't allow max_pool_size
                    pool_size = min(h, w, self.max_pool_size)
                    x = F.max_pool2d(x, pool_size)
            return int(torch.prod(torch.tensor(x.shape[1:])))

    def _get_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation function based on design."""
        if self.activation_type == 1:  # ELU
            return F.elu(x)
        elif self.activation_type == 2:  # GELU
            return F.gelu(x)
        elif self.activation_type == 3:  # ReLU
            return F.relu(x)
        elif self.activation_type == 4:  # SELU
            return F.selu(x)
        elif self.activation_type == 5:  # Sigmoid
            return torch.sigmoid(x)
        elif self.activation_type == 6:  # Softmax (not typically used in hidden layers)
            return F.softmax(x, dim=1)
        elif self.activation_type == 7:  # Softplus
            return F.softplus(x)
        elif self.activation_type == 8:  # Swish (SiLU)
            return F.silu(x)
        elif self.activation_type == 9:  # Tanh
            return torch.tanh(x)
        else:
            return F.relu(x)  # Default to ReLU

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Convolutional layers
        for i in range(self.num_conv_layers):
            x = self.conv_layers[i](x)

            if self.use_batch_norm:
                x = self.bn_layers[i](x)

            x = self._get_activation(x)

            # Apply adaptive pooling to prevent dimension reduction to zero
            h, w = x.shape[2], x.shape[3]
            if h >= self.max_pool_size and w >= self.max_pool_size:
                x = F.max_pool2d(x, self.max_pool_size)
            elif h > 1 and w > 1:
                # Use smaller pooling if dimensions don't allow max_pool_size
                pool_size = min(h, w, self.max_pool_size)
                x = F.max_pool2d(x, pool_size)

            if self.dropout is not None and i < self.num_conv_layers - 1:
                x = self.dropout(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)

        if self.use_batch_norm:
            x = self.bn_fc(x)

        x = self._get_activation(x)

        if self.dropout is not None:
            x = self.dropout(x)

        x = self.fc2(x)

        return x

    def get_num_parameters(self) -> int:
        """Get the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
