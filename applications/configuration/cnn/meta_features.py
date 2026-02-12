"""
Meta-feature extraction for CNN Configuration Application.

Based on Chapter 8, Section 8.4.1 of the thesis, this includes:
- Application-specific meta-features (sensitivity, specificity, loss metrics, etc.)
- Application-agnostic meta-features (ELA features, dataset properties)
"""

from typing import Dict, Any, List, Tuple
from functools import lru_cache
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import skew, kurtosis
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, cohen_kappa_score,
    mean_squared_error, mean_absolute_error, r2_score, f1_score
)
from scipy.spatial.distance import cosine, hamming, minkowski, euclidean, cityblock
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


class CNNMetaFeatureExtractor:
    """
    Extract meta-features for CNN configuration application.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize meta-feature extractor.

        Args:
            model: The CNN model
            device: Torch device (cuda/cpu)
        """
        self.model = model
        self.device = device
        self.previous_weights = None
        self._cached_dataset_features = None  # Cache for dataset-level features

    def extract_meta_features(
        self,
        train_loader,
        val_loader,
        dataset_info: Dict[str, Any],
        timestep: int
    ) -> Dict[str, Any]:
        """
        Extract all meta-features for current timestep.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            dataset_info: Dataset information dictionary
            timestep: Current timestep (epoch)

        Returns:
            Dictionary of meta-features
        """
        meta_features = {}

        # Application-agnostic meta-features
        meta_features.update(self._extract_dataset_features(train_loader, dataset_info))

        # Application-specific meta-features (from Section 8.4.1)
        if self.model is not None and timestep > 0:
            meta_features.update(self._extract_model_features(train_loader, val_loader))

        # ELA features (if we have sufficient data)
        if timestep > 5:  # Need some history for landscape analysis
            meta_features.update(self._extract_ela_features())

        # Timestep feature
        meta_features['timestep'] = timestep

        return meta_features

    def _extract_dataset_features(
        self,
        train_loader,
        dataset_info: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Extract application-agnostic dataset meta-features.
        Uses caching to avoid recomputing static dataset properties.
        """
        # Return cached features if available (dataset properties don't change)
        if self._cached_dataset_features is not None:
            return self._cached_dataset_features.copy()

        features = {}

        # Basic dataset properties (these are constant across timesteps)
        features['num_classes'] = dataset_info['num_classes']
        features['num_instances'] = len(train_loader.dataset)
        features['input_channels'] = dataset_info.get('channels', 3)
        features['input_height'] = dataset_info.get('height', 32)
        features['input_width'] = dataset_info.get('width', 32)

        # Class imbalance (this is also constant)
        features['imbalance'] = self._calculate_imbalance(train_loader)

        # Cache these features for future calls
        self._cached_dataset_features = features.copy()

        return features

    def _extract_model_features(
        self,
        train_loader,
        val_loader
    ) -> Dict[str, float]:
        """
        Extract application-specific model meta-features from Section 8.4.1.
        """
        features = {}

        # Get predictions
        train_preds, train_labels = self._get_predictions(train_loader)
        val_preds, val_labels = self._get_predictions(val_loader)

        # Confusion matrix metrics (TP, TN, FP, FN)
        train_cm_features = self._calculate_confusion_matrix_features(train_preds, train_labels)
        val_cm_features = self._calculate_confusion_matrix_features(val_preds, val_labels)

        for key, value in train_cm_features.items():
            features[f'train_{key}'] = value
        for key, value in val_cm_features.items():
            features[f'val_{key}'] = value

        # Performance metrics
        features.update(self._calculate_performance_metrics(train_preds, train_labels, 'train'))
        features.update(self._calculate_performance_metrics(val_preds, val_labels, 'val'))

        # Weight distance features (if we have previous weights)
        if self.previous_weights is not None:
            features.update(self._calculate_weight_distances())

        # Update previous weights for next timestep
        self.previous_weights = self._get_current_weights()

        return features

    def _calculate_confusion_matrix_features(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate TP, TN, FP, FN and derived metrics.
        Uses one-vs-rest for multi-class.
        """
        features = {}

        # Get unique classes
        unique_classes = np.unique(labels)
        num_classes = len(unique_classes)

        # For multi-class, compute one-vs-rest metrics and average
        tp_total = 0
        tn_total = 0
        fp_total = 0
        fn_total = 0

        for cls in unique_classes:
            # Binarize for this class
            binary_labels = (labels == cls).astype(int)
            binary_preds = (predictions == cls).astype(int)

            # Compute confusion matrix
            cm = confusion_matrix(binary_labels, binary_preds)

            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                # Handle edge case where only one class is predicted
                tp = np.sum((binary_labels == 1) & (binary_preds == 1))
                tn = np.sum((binary_labels == 0) & (binary_preds == 0))
                fp = np.sum((binary_labels == 0) & (binary_preds == 1))
                fn = np.sum((binary_labels == 1) & (binary_preds == 0))

            tp_total += tp
            tn_total += tn
            fp_total += fp
            fn_total += fn

        # Average across classes
        features['tp'] = tp_total / num_classes
        features['tn'] = tn_total / num_classes
        features['fp'] = fp_total / num_classes
        features['fn'] = fn_total / num_classes

        # Calculate accuracy: (TP + TN) / (TP + TN + FP + FN)
        total = tp_total + tn_total + fp_total + fn_total
        if total > 0:
            features['accuracy'] = (tp_total + tn_total) / total
        else:
            features['accuracy'] = 0.0

        # Sensitivity (Recall/TPR): TP / (TP + FN)
        if (tp_total + fn_total) > 0:
            features['sensitivity'] = tp_total / (tp_total + fn_total)
        else:
            features['sensitivity'] = 0.0

        # Specificity (TNR): TN / (TN + FP)
        if (tn_total + fp_total) > 0:
            features['specificity'] = tn_total / (tn_total + fp_total)
        else:
            features['specificity'] = 0.0

        # False alarm rate (FPR): FP / (FP + TN)
        if (fp_total + tn_total) > 0:
            features['false_alarm_rate'] = fp_total / (fp_total + tn_total)
        else:
            features['false_alarm_rate'] = 0.0

        # Selectivity (Precision): TP / (TP + FP)
        if (tp_total + fp_total) > 0:
            features['selectivity'] = tp_total / (tp_total + fp_total)
        else:
            features['selectivity'] = 0.0

        # Proportion of variance (POV) - squared cosine of angle
        if len(predictions) > 0 and len(labels) > 0:
            # Compute cosine similarity
            predictions_flat = predictions.flatten().astype(float)
            labels_flat = labels.flatten().astype(float)
            if np.std(predictions_flat) > 0 and np.std(labels_flat) > 0:
                cos_sim = 1 - cosine(predictions_flat, labels_flat)
                features['pov'] = cos_sim ** 2
            else:
                features['pov'] = 0.0
        else:
            features['pov'] = 0.0

        return features

    def _calculate_performance_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        prefix: str
    ) -> Dict[str, float]:
        """
        Calculate various performance metrics mentioned in Section 8.4.1.
        """
        features = {}

        # RMSE and MAE (treating as regression problem for probabilities)
        features[f'{prefix}_rmse'] = np.sqrt(mean_squared_error(labels, predictions))
        features[f'{prefix}_mae'] = mean_absolute_error(labels, predictions)

        # KL Divergence (approximation using log loss)
        try:
            # Convert to probability distribution
            pred_dist = np.bincount(predictions, minlength=np.max(labels)+1) / len(predictions)
            label_dist = np.bincount(labels, minlength=np.max(labels)+1) / len(labels)

            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            pred_dist = pred_dist + epsilon
            label_dist = label_dist + epsilon

            kl_div = np.sum(label_dist * np.log(label_dist / pred_dist))
            features[f'{prefix}_kl_divergence'] = kl_div
        except:
            features[f'{prefix}_kl_divergence'] = 0.0

        # R2 Score
        try:
            features[f'{prefix}_r2_score'] = r2_score(labels, predictions)
        except:
            features[f'{prefix}_r2_score'] = 0.0

        # F-Beta Score (using F1)
        try:
            features[f'{prefix}_f1_score'] = f1_score(labels, predictions, average='weighted')
        except:
            features[f'{prefix}_f1_score'] = 0.0

        # Pearson correlation
        try:
            if len(predictions) > 1 and np.std(predictions) > 0 and np.std(labels) > 0:
                corr, _ = pearsonr(predictions, labels)
                features[f'{prefix}_pearson'] = corr
            else:
                features[f'{prefix}_pearson'] = 0.0
        except:
            features[f'{prefix}_pearson'] = 0.0

        # Cohen Kappa
        try:
            features[f'{prefix}_cohen_kappa'] = cohen_kappa_score(labels, predictions)
        except:
            features[f'{prefix}_cohen_kappa'] = 0.0

        return features

    def _calculate_weight_distances(self) -> Dict[str, float]:
        """
        Calculate distance between current and previous weights.
        From Section 8.4.1: cosine, Hamming, Minkowski, Euclidean, Manhattan.
        """
        features = {}

        current_weights = self._get_current_weights()

        if current_weights is not None and self.previous_weights is not None:
            # Flatten weights
            current_flat = current_weights.flatten()
            previous_flat = self.previous_weights.flatten()

            # Ensure same length
            if len(current_flat) == len(previous_flat):
                # Cosine distance
                try:
                    features['weight_cosine_distance'] = cosine(current_flat, previous_flat)
                except:
                    features['weight_cosine_distance'] = 0.0

                # Euclidean distance
                features['weight_euclidean_distance'] = euclidean(current_flat, previous_flat)

                # Manhattan distance
                features['weight_manhattan_distance'] = cityblock(current_flat, previous_flat)

                # Minkowski distance (p=3)
                features['weight_minkowski_distance'] = minkowski(current_flat, previous_flat, p=3)

                # Hamming distance (for discrete comparison)
                # Binarize weights for Hamming
                current_binary = (current_flat > 0).astype(int)
                previous_binary = (previous_flat > 0).astype(int)
                features['weight_hamming_distance'] = hamming(current_binary, previous_binary)

        return features

    def _extract_ela_features(self) -> Dict[str, float]:
        """
        Extract Exploratory Landscape Analysis (ELA) features.
        From Table 8.1: y-distribution, meta-model, dispersion, information content, NBC.
        """
        features = {}

        # For now, we'll implement simplified ELA features
        # Full implementation would require maintaining a history of explored designs

        # Get current weights as a sample point
        weights = self._get_current_weights()

        if weights is not None:
            # Y-distribution features (distribution of weight values)
            features['y_skewness'] = skew(weights.flatten())
            features['y_kurtosis'] = kurtosis(weights.flatten())

            # Number of peaks (simplified: count sign changes in sorted weights)
            sorted_weights = np.sort(weights.flatten())
            diffs = np.diff(sorted_weights)
            sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
            features['y_num_peaks'] = sign_changes

        return features

    def _get_predictions(self, data_loader) -> Tuple[np.ndarray, np.ndarray]:
        """Get model predictions and true labels from data loader."""
        self.model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predictions = torch.max(outputs, 1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())

        return np.array(all_predictions), np.array(all_labels)

    def _get_current_weights(self) -> np.ndarray:
        """Get flattened current model weights."""
        if self.model is None:
            return None

        weights = []
        for param in self.model.parameters():
            weights.append(param.data.cpu().numpy().flatten())

        return np.concatenate(weights) if weights else None

    def _calculate_imbalance(self, data_loader) -> float:
        """
        Calculate class imbalance metric.
        0 = perfectly balanced, 1 = completely imbalanced.
        """
        # Collect all labels
        all_labels = []
        for _, labels in data_loader:
            all_labels.extend(labels.numpy())

        all_labels = np.array(all_labels)

        # Count class frequencies
        unique, counts = np.unique(all_labels, return_counts=True)

        if len(unique) <= 1:
            return 0.0  # Only one class, consider balanced

        # Calculate imbalance as coefficient of variation
        mean_count = np.mean(counts)
        std_count = np.std(counts)

        if mean_count > 0:
            imbalance = std_count / mean_count
        else:
            imbalance = 0.0

        return imbalance
