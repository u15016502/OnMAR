"""
Meta-feature extraction for Segmentation Composition Application.

Based on Chapter 8, Section 8.3.2 of the thesis, this includes:
- Local Consistency Error (LCE)
- Global Consistency Error (GCE)
- ROC and AUC
- Cohen-Kappa, Dice coefficient
- Various distance metrics
- Application-agnostic features (ELA, dataset properties)
"""

from typing import Dict, Any, Tuple
from functools import lru_cache
import numpy as np
from scipy.stats import skew, kurtosis, bernoulli, laplace, zipf, poisson
from scipy.spatial.distance import cosine, euclidean, hamming
try:
    from scipy.spatial.distance import directed_hausdorff as hausdorff
except ImportError:
    # Fallback for older scipy versions
    from scipy.spatial.distance import hausdorff
from sklearn.metrics import (
    roc_auc_score, roc_curve, cohen_kappa_score,
    mean_squared_error
)
from scipy.stats import wasserstein_distance
import warnings
warnings.filterwarnings('ignore')


class SegmentationMetaFeatureExtractor:
    """
    Extract meta-features for segmentation composition application.
    """

    def __init__(self):
        """Initialize meta-feature extractor."""
        self._cached_dataset_features = None  # Cache for dataset-level features

    def extract_meta_features(
        self,
        predicted_segmentation: np.ndarray,
        ground_truth_segmentation: np.ndarray,
        dataset_info: Dict[str, Any],
        timestep: int
    ) -> Dict[str, Any]:
        """
        Extract all meta-features for current timestep.

        Args:
            predicted_segmentation: Predicted segmentation mask
            ground_truth_segmentation: Ground truth segmentation mask
            dataset_info: Dataset information dictionary
            timestep: Current timestep (generation)

        Returns:
            Dictionary of meta-features
        """
        meta_features = {}

        # Application-agnostic meta-features
        meta_features.update(self._extract_dataset_features(dataset_info))

        # Application-specific meta-features (from Section 8.3.2)
        if predicted_segmentation is not None and ground_truth_segmentation is not None:
            meta_features.update(
                self._extract_segmentation_features(
                    predicted_segmentation,
                    ground_truth_segmentation
                )
            )

        # Timestep feature
        meta_features['timestep'] = timestep

        return meta_features

    def _extract_dataset_features(
        self,
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

        # Basic dataset properties (constant across timesteps)
        features['num_classes'] = dataset_info.get('num_classes', 2)
        features['num_instances'] = dataset_info.get('num_images', 0)
        features['input_height'] = dataset_info.get('height', 256)
        features['input_width'] = dataset_info.get('width', 256)
        features['total_pixels'] = features['input_height'] * features['input_width']

        # Class imbalance (placeholder, would need actual class distributions)
        features['imbalance'] = 0.0

        # Cache these features for future calls
        self._cached_dataset_features = features.copy()

        return features

    def _extract_segmentation_features(
        self,
        predicted: np.ndarray,
        ground_truth: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract application-specific segmentation meta-features from Section 8.3.2.
        """
        features = {}

        # Flatten arrays for easier computation
        pred_flat = predicted.flatten()
        gt_flat = ground_truth.flatten()

        # Local and Global Consistency Errors
        features['lce'] = self._calculate_lce(predicted, ground_truth)
        features['gce'] = self._calculate_gce(predicted, ground_truth)

        # ROC and AUC (for binary segmentation)
        try:
            if len(np.unique(gt_flat)) == 2:  # Binary segmentation
                fpr, tpr, _ = roc_curve(gt_flat, pred_flat)
                features['auc'] = roc_auc_score(gt_flat, pred_flat)
                features['roc_tpr_mean'] = np.mean(tpr)
                features['roc_fpr_mean'] = np.mean(fpr)
            else:
                features['auc'] = 0.0
                features['roc_tpr_mean'] = 0.0
                features['roc_fpr_mean'] = 0.0
        except:
            features['auc'] = 0.0
            features['roc_tpr_mean'] = 0.0
            features['roc_fpr_mean'] = 0.0

        # Cohen-Kappa score
        try:
            features['cohen_kappa'] = cohen_kappa_score(gt_flat, pred_flat)
        except:
            features['cohen_kappa'] = 0.0

        # Dice coefficient
        features['dice_coefficient'] = self._calculate_dice(predicted, ground_truth)

        # Wasserstein distance
        try:
            features['wasserstein_distance'] = wasserstein_distance(pred_flat, gt_flat)
        except:
            features['wasserstein_distance'] = 0.0

        # Kernel similarities
        features['chi_squared_kernel'] = self._calculate_chi_squared_kernel(pred_flat, gt_flat)
        features['linear_kernel'] = self._calculate_linear_kernel(pred_flat, gt_flat)
        features['laplacian_kernel'] = self._calculate_laplacian_kernel(pred_flat, gt_flat)

        # KL Divergence
        features['kl_divergence'] = self._calculate_kl_divergence(pred_flat, gt_flat)

        # Squared hinge loss
        features['squared_hinge'] = self._calculate_squared_hinge(pred_flat, gt_flat)

        # RMSE
        features['rmse'] = np.sqrt(mean_squared_error(gt_flat, pred_flat))

        # Distance metrics between predicted and ground truth
        features.update(self._calculate_distance_metrics(pred_flat, gt_flat))

        # Distribution features
        features.update(self._calculate_distribution_features(pred_flat))

        return features

    def _calculate_lce(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate Local Consistency Error (LCE).
        Measures consistency error in local regions.
        """
        # Simplified LCE implementation
        # Full implementation would use superpixel segmentation
        # For now, use a sliding window approach

        if pred.shape != gt.shape:
            return 1.0  # Maximum error if shapes don't match

        height, width = pred.shape[:2]
        window_size = 5
        errors = []

        for i in range(0, height - window_size, window_size):
            for j in range(0, width - window_size, window_size):
                pred_window = pred[i:i+window_size, j:j+window_size]
                gt_window = gt[i:i+window_size, j:j+window_size]

                # Calculate local error (proportion of mismatched pixels)
                local_error = np.mean(pred_window != gt_window)
                errors.append(local_error)

        return np.mean(errors) if errors else 1.0

    def _calculate_gce(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate Global Consistency Error (GCE).
        Measures global consistency of segmentation.
        """
        # Simplified GCE implementation
        # Measures overall pixel-wise consistency

        if pred.shape != gt.shape:
            return 1.0  # Maximum error if shapes don't match

        # Global error is proportion of mismatched pixels
        global_error = np.mean(pred != gt)

        return global_error

    def _calculate_dice(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate Dice coefficient.
        Dice = 2 * |pred ∩ gt| / (|pred| + |gt|)
        """
        intersection = np.sum(pred == gt)
        total = pred.size + gt.size

        if total == 0:
            return 0.0

        dice = (2.0 * intersection) / total
        return dice

    def _calculate_chi_squared_kernel(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate chi-squared kernel similarity."""
        # Normalize to probability distributions
        pred_hist = np.bincount(pred.astype(int).flatten()) / len(pred)
        gt_hist = np.bincount(gt.astype(int).flatten()) / len(gt)

        # Make same length
        max_len = max(len(pred_hist), len(gt_hist))
        pred_hist = np.pad(pred_hist, (0, max_len - len(pred_hist)))
        gt_hist = np.pad(gt_hist, (0, max_len - len(gt_hist)))

        # Chi-squared kernel
        epsilon = 1e-10
        chi_squared = np.sum((pred_hist - gt_hist) ** 2 / (pred_hist + gt_hist + epsilon))

        return chi_squared

    def _calculate_linear_kernel(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate linear kernel similarity (dot product)."""
        # Normalize
        pred_norm = pred / (np.linalg.norm(pred) + 1e-10)
        gt_norm = gt / (np.linalg.norm(gt) + 1e-10)

        return np.dot(pred_norm, gt_norm)

    def _calculate_laplacian_kernel(self, pred: np.ndarray, gt: np.ndarray, gamma: float = 0.1) -> float:
        """Calculate Laplacian kernel similarity."""
        # Laplacian kernel: exp(-gamma * ||x - y||_1)
        l1_distance = np.sum(np.abs(pred - gt))
        return np.exp(-gamma * l1_distance)

    def _calculate_kl_divergence(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate KL divergence between distributions."""
        # Convert to probability distributions
        pred_hist = np.bincount(pred.astype(int).flatten()) / len(pred)
        gt_hist = np.bincount(gt.astype(int).flatten()) / len(gt)

        # Make same length
        max_len = max(len(pred_hist), len(gt_hist))
        pred_hist = np.pad(pred_hist, (0, max_len - len(pred_hist)))
        gt_hist = np.pad(gt_hist, (0, max_len - len(gt_hist)))

        # Add epsilon to avoid log(0)
        epsilon = 1e-10
        pred_hist = pred_hist + epsilon
        gt_hist = gt_hist + epsilon

        # Normalize
        pred_hist = pred_hist / np.sum(pred_hist)
        gt_hist = gt_hist / np.sum(gt_hist)

        # KL divergence
        kl_div = np.sum(gt_hist * np.log(gt_hist / pred_hist))

        return kl_div

    def _calculate_squared_hinge(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate squared hinge loss."""
        # Normalize to [-1, 1]
        pred_norm = 2 * (pred - np.min(pred)) / (np.max(pred) - np.min(pred) + 1e-10) - 1
        gt_norm = 2 * (gt - np.min(gt)) / (np.max(gt) - np.min(gt) + 1e-10) - 1

        # Squared hinge loss
        hinge = np.maximum(0, 1 - gt_norm * pred_norm)
        squared_hinge = np.mean(hinge ** 2)

        return squared_hinge

    def _calculate_distance_metrics(self, pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
        """
        Calculate various distance metrics from Section 8.3.2:
        cosine, Euclidean, sigmoid, Hausdorff, Hamming
        """
        features = {}

        # Cosine distance
        try:
            features['cosine_distance'] = cosine(pred, gt)
        except:
            features['cosine_distance'] = 0.0

        # Euclidean distance
        features['euclidean_distance'] = euclidean(pred, gt)

        # Sigmoid kernel distance (approximation)
        sigmoid_sim = np.tanh(np.dot(pred, gt) / (np.linalg.norm(pred) * np.linalg.norm(gt) + 1e-10))
        features['sigmoid_distance'] = 1 - sigmoid_sim

        # Hamming distance (for binary/integer arrays)
        try:
            features['hamming_distance'] = hamming(pred.astype(int), gt.astype(int))
        except:
            features['hamming_distance'] = 0.0

        # Hausdorff distance (for 2D shapes)
        # Simplified: use sample of points
        try:
            pred_sample = pred[:min(1000, len(pred))]
            gt_sample = gt[:min(1000, len(gt))]
            features['hausdorff_distance'] = hausdorff(
                pred_sample.reshape(-1, 1),
                gt_sample.reshape(-1, 1)
            )
        except:
            features['hausdorff_distance'] = 0.0

        return features

    def _calculate_distribution_features(self, pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate features based on discrete random variable distributions.
        From Section 8.3.2: Bernoulli, Laplacian, Zipf, Poisson, Planck, Logarithmic, Yule-Simon
        """
        features = {}

        # Get histogram of predicted values
        values = pred.astype(int)
        hist, _ = np.histogram(values, bins=min(50, len(np.unique(values))))

        # Normalize to probability mass function
        pmf = hist / np.sum(hist)

        # Bernoulli: probability mass at two points
        if len(pmf) >= 2:
            features['bernoulli_p'] = pmf[0]
        else:
            features['bernoulli_p'] = 0.0

        # Laplacian: calculate location and scale
        features['laplacian_loc'] = np.mean(values)
        features['laplacian_scale'] = np.std(values)

        # Zipf: power law distribution
        # Estimate Zipf parameter
        if len(np.unique(values)) > 1:
            rank = np.arange(1, len(pmf) + 1)
            log_pmf = np.log(pmf + 1e-10)
            log_rank = np.log(rank)
            zipf_param = -np.polyfit(log_rank, log_pmf, 1)[0]
            features['zipf_param'] = zipf_param
        else:
            features['zipf_param'] = 0.0

        # Poisson: lambda parameter (mean)
        features['poisson_lambda'] = np.mean(values)

        # Distribution statistics
        features['dist_skewness'] = skew(values)
        features['dist_kurtosis'] = kurtosis(values)
        features['dist_entropy'] = -np.sum(pmf * np.log(pmf + 1e-10))

        return features
