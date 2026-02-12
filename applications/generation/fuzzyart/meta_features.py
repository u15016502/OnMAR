"""
Meta-feature extraction for Fuzzy ART Choice Function Generation Application.

Based on Chapter 8, Section 8.5.2 of the thesis, this uses similar meta-features
to the CNN configuration application (Section 8.4.1) since both involve neural networks
for classification tasks.
"""

from typing import Dict, Any, Tuple
from functools import lru_cache
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, cohen_kappa_score,
    mean_squared_error, mean_absolute_error, r2_score, f1_score
)
from scipy.spatial.distance import cosine, hamming, minkowski, euclidean, cityblock
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


class FuzzyARTMetaFeatureExtractor:
    """
    Extract meta-features for Fuzzy ART generation application.
    Similar to CNN meta-features since both are neural network classifiers.
    """

    def __init__(self):
        """Initialize meta-feature extractor."""
        self.previous_weights = None
        self._cached_dataset_features = None  # Cache for dataset-level features

    def extract_meta_features(
        self,
        fuzzyart_model,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        val_data: np.ndarray,
        val_labels: np.ndarray,
        dataset_info: Dict[str, Any],
        timestep: int
    ) -> Dict[str, Any]:
        """
        Extract all meta-features for current timestep.

        Args:
            fuzzyart_model: The Fuzzy ART model
            train_data: Training data
            train_labels: Training labels
            val_data: Validation data
            val_labels: Validation labels
            dataset_info: Dataset information dictionary
            timestep: Current timestep (epoch/generation)

        Returns:
            Dictionary of meta-features
        """
        meta_features = {}

        # Application-agnostic meta-features
        meta_features.update(self._extract_dataset_features(dataset_info, train_data, train_labels))

        # Application-specific meta-features (similar to Section 8.4.1 for CNN)
        if fuzzyart_model is not None and fuzzyart_model.w is not None:
            meta_features.update(
                self._extract_model_features(
                    fuzzyart_model,
                    train_data,
                    train_labels,
                    val_data,
                    val_labels
                )
            )

        # ELA features (if we have sufficient history)
        if timestep > 5:
            meta_features.update(self._extract_ela_features(fuzzyart_model))

        # Timestep feature
        meta_features['timestep'] = timestep

        return meta_features

    def _extract_dataset_features(
        self,
        dataset_info: Dict[str, Any],
        train_data: np.ndarray,
        train_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract application-agnostic dataset meta-features.
        Uses caching to avoid recomputing static dataset properties.
        """
        # Return cached features if available (dataset properties don't change)
        if self._cached_dataset_features is not None:
            return self._cached_dataset_features.copy()

        features = {}

        # Basic dataset properties (constant across timesteps)
        features['num_classes'] = dataset_info.get('num_classes', len(np.unique(train_labels)))
        features['num_instances'] = len(train_data)
        features['input_dim'] = train_data.shape[1] if len(train_data.shape) > 1 else 1

        # Class imbalance (constant for static dataset)
        features['imbalance'] = self._calculate_imbalance(train_labels)

        # Cache these features for future calls
        self._cached_dataset_features = features.copy()

        return features

    def _extract_model_features(
        self,
        model,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        val_data: np.ndarray,
        val_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract application-specific model meta-features (from Section 8.4.1).
        """
        features = {}

        # Get predictions
        train_preds = model.predict(train_data)
        val_preds = model.predict(val_data)

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

        # Weight distance features (L2 layer weights)
        if self.previous_weights is not None and model.w is not None:
            features.update(self._calculate_weight_distances(model.w))

        # Update previous weights for next timestep
        if model.w is not None:
            self.previous_weights = model.w.copy()

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

        # RMSE and MAE
        features[f'{prefix}_rmse'] = np.sqrt(mean_squared_error(labels, predictions))
        features[f'{prefix}_mae'] = mean_absolute_error(labels, predictions)

        # KL Divergence (approximation)
        try:
            pred_dist = np.bincount(predictions.astype(int), minlength=np.max(labels)+1) / len(predictions)
            label_dist = np.bincount(labels.astype(int), minlength=np.max(labels)+1) / len(labels)

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

    def _calculate_weight_distances(self, current_weights: np.ndarray) -> Dict[str, float]:
        """
        Calculate distance between current and previous L2 layer weights.
        From Section 8.4.1: cosine, Hamming, Minkowski, Euclidean, Manhattan.
        """
        features = {}

        if self.previous_weights is not None:
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
                current_binary = (current_flat > 0).astype(int)
                previous_binary = (previous_flat > 0).astype(int)
                features['weight_hamming_distance'] = hamming(current_binary, previous_binary)

        return features

    def _extract_ela_features(self, model) -> Dict[str, float]:
        """
        Extract Exploratory Landscape Analysis (ELA) features.
        From Table 8.1: y-distribution, meta-model, dispersion, information content, NBC.
        """
        features = {}

        # Get current weights as a sample point
        if model.w is not None:
            weights = model.w.flatten()

            # Y-distribution features (distribution of weight values)
            features['y_skewness'] = skew(weights)
            features['y_kurtosis'] = kurtosis(weights)

            # Number of peaks (simplified: count sign changes in sorted weights)
            sorted_weights = np.sort(weights)
            diffs = np.diff(sorted_weights)
            sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
            features['y_num_peaks'] = sign_changes

        return features

    def _calculate_imbalance(self, labels: np.ndarray) -> float:
        """
        Calculate class imbalance metric.
        0 = perfectly balanced, 1 = completely imbalanced.
        """
        unique, counts = np.unique(labels, return_counts=True)

        if len(unique) <= 1:
            return 0.0

        # Calculate imbalance as coefficient of variation
        mean_count = np.mean(counts)
        std_count = np.std(counts)

        if mean_count > 0:
            imbalance = std_count / mean_count
        else:
            imbalance = 0.0

        return imbalance
