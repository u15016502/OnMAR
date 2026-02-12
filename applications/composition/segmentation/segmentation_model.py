"""
Segmentation Algorithmic Components

This module implements the 50 algorithmic components from Appendix A.2:
- Filtering (FIL): 8 components
- Morphology (MOR): 12 components
- Feature Detection (FD): 14 components
- Image-specific Clustering (CIS): 6 components
- Generic Clustering (CG): 10 components

Each component is implemented as a function that takes an image and returns
a processed image or segmentation mask.
"""

from typing import Dict, Callable, Any, Tuple
import numpy as np
import cv2
from skimage import filters, morphology, feature, segmentation, measure
from skimage.filters import gabor
from skimage.transform import hough_line, hough_line_peaks
from sklearn.cluster import (
    KMeans, AffinityPropagation, MeanShift, SpectralClustering,
    AgglomerativeClustering, DBSCAN, OPTICS, BisectingKMeans, Birch
)
from sklearn.mixture import GaussianMixture
from scipy import ndimage


class SegmentationComponents:
    """
    Collection of segmentation algorithmic components.
    Each component is a callable that processes an image.
    """

    # ========== FILTERING (FIL) - 8 components ==========

    @staticmethod
    def gaussian_filter(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply Gaussian filter for noise reduction."""
        sigma = kwargs.get('sigma', 1.0)
        if len(image.shape) == 3:
            return np.stack([filters.gaussian(image[:, :, i], sigma=sigma)
                           for i in range(image.shape[2])], axis=2)
        return filters.gaussian(image, sigma=sigma)

    @staticmethod
    def adaptive_histogram_eq(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply adaptive histogram equalization (CLAHE)."""
        # Convert to uint8 for CLAHE
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        if len(image.shape) == 3:
            # Convert to LAB and apply CLAHE to L channel
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image).astype(np.float32) / 255.0

    @staticmethod
    def thresholding(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply Otsu's thresholding."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)

        thresh_val = filters.threshold_otsu(gray)
        return (gray > thresh_val).astype(np.float32)

    @staticmethod
    def convert_to_binary(image: np.ndarray, **kwargs) -> np.ndarray:
        """Convert image to binary using thresholding."""
        threshold = kwargs.get('threshold', 0.5)
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        return (gray > threshold).astype(np.float32)

    @staticmethod
    def leung_malik_filter(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply Leung-Malik filter bank (simplified version)."""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Apply multiple Gaussian derivatives at different scales
        scales = [1, 2, 4]
        responses = []
        for scale in scales:
            responses.append(filters.gaussian(gray, sigma=scale))

        # Return mean of responses
        return np.mean(responses, axis=0)

    @staticmethod
    def median_blur(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply median blur filter."""
        ksize = kwargs.get('ksize', 5)
        if len(image.shape) == 3:
            return np.stack([ndimage.median_filter(image[:, :, i], size=ksize)
                           for i in range(image.shape[2])], axis=2)
        return ndimage.median_filter(image, size=ksize)

    @staticmethod
    def brightness_enhance(image: np.ndarray, **kwargs) -> np.ndarray:
        """Enhance image brightness."""
        factor = kwargs.get('factor', 1.2)
        enhanced = image * factor
        return np.clip(enhanced, 0, 1)

    @staticmethod
    def color_space_change(image: np.ndarray, **kwargs) -> np.ndarray:
        """Change color space (RGB to HSV or LAB)."""
        if len(image.shape) != 3:
            return image

        # Convert to uint8 for OpenCV
        img_uint8 = (image * 255).astype(np.uint8)

        # Convert to HSV
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
        return hsv.astype(np.float32) / 255.0

    # ========== MORPHOLOGY (MOR) - 12 components ==========

    @staticmethod
    def vertical_area_extraction(image: np.ndarray, **kwargs) -> np.ndarray:
        """Extract vertical structures."""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Vertical structuring element
        kernel = np.ones((9, 1), np.uint8)
        binary = (gray > 0.5).astype(np.uint8)
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel).astype(np.float32)

    @staticmethod
    def horizontal_area_extraction(image: np.ndarray, **kwargs) -> np.ndarray:
        """Extract horizontal structures."""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Horizontal structuring element
        kernel = np.ones((1, 9), np.uint8)
        binary = (gray > 0.5).astype(np.uint8)
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel).astype(np.float32)

    @staticmethod
    def skeletonization(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply morphological skeletonization."""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        binary = gray > 0.5
        skeleton = morphology.skeletonize(binary)
        return skeleton.astype(np.float32)

    @staticmethod
    def dilation(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply morphological dilation."""
        selem_size = kwargs.get('selem_size', 3)
        selem = morphology.disk(selem_size)

        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        binary = gray > 0.5
        return morphology.binary_dilation(binary, selem).astype(np.float32)

    @staticmethod
    def erosion(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply morphological erosion."""
        selem_size = kwargs.get('selem_size', 3)
        selem = morphology.disk(selem_size)

        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        binary = gray > 0.5
        return morphology.binary_erosion(binary, selem).astype(np.float32)

    @staticmethod
    def opening(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply morphological opening (erosion then dilation)."""
        selem_size = kwargs.get('selem_size', 3)
        selem = morphology.disk(selem_size)

        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        binary = gray > 0.5
        return morphology.binary_opening(binary, selem).astype(np.float32)

    @staticmethod
    def closing(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply morphological closing (dilation then erosion)."""
        selem_size = kwargs.get('selem_size', 3)
        selem = morphology.disk(selem_size)

        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        binary = gray > 0.5
        return morphology.binary_closing(binary, selem).astype(np.float32)

    @staticmethod
    def union_find(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply connected component labeling using union-find."""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        binary = gray > 0.5
        labeled = measure.label(binary)
        return (labeled > 0).astype(np.float32)

    @staticmethod
    def convex_hull(image: np.ndarray, **kwargs) -> np.ndarray:
        """Compute convex hull of binary regions."""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        binary = gray > 0.5
        return morphology.convex_hull_image(binary).astype(np.float32)

    @staticmethod
    def hole_filling(image: np.ndarray, **kwargs) -> np.ndarray:
        """Fill holes in binary image."""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        binary = gray > 0.5
        return ndimage.binary_fill_holes(binary).astype(np.float32)

    @staticmethod
    def top_hat(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply white top-hat transform."""
        selem_size = kwargs.get('selem_size', 5)
        selem = morphology.disk(selem_size)

        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        result = morphology.white_tophat(gray, selem)
        return result

    @staticmethod
    def bottom_hat(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply black bottom-hat transform."""
        selem_size = kwargs.get('selem_size', 5)
        selem = morphology.disk(selem_size)

        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        result = morphology.black_tophat(gray, selem)
        return result

    # ========== FEATURE DETECTION (FD) - 14 components ==========

    @staticmethod
    def hough_transform(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply Hough line transform."""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Edge detection first
        edges = feature.canny(gray, sigma=2.0)

        # Hough transform
        h, theta, d = hough_line(edges)

        # Return edges for now (simplified)
        return edges.astype(np.float32)

    @staticmethod
    def prewitt(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply Prewitt edge detector."""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        edges = filters.prewitt(gray)
        return edges

    @staticmethod
    def sobel(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply Sobel edge detector."""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        edges = filters.sobel(gray)
        return edges

    @staticmethod
    def canny_edge(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply Canny edge detector."""
        sigma = kwargs.get('sigma', 2.0)

        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        edges = feature.canny(gray, sigma=sigma)
        return edges.astype(np.float32)

    @staticmethod
    def haar(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply Haar-like features (simplified)."""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Use integral image for Haar-like features
        integral = np.cumsum(np.cumsum(gray, axis=0), axis=1)
        return integral / integral.max() if integral.max() > 0 else integral

    @staticmethod
    def laplacian(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply Laplacian operator."""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        result = filters.laplace(gray)
        return result

    @staticmethod
    def det_of_hessian(image: np.ndarray, **kwargs) -> np.ndarray:
        """Compute determinant of Hessian."""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Compute second derivatives
        Ixx = ndimage.sobel(ndimage.sobel(gray, axis=0), axis=0)
        Iyy = ndimage.sobel(ndimage.sobel(gray, axis=1), axis=1)
        Ixy = ndimage.sobel(ndimage.sobel(gray, axis=0), axis=1)

        # Determinant of Hessian
        det_H = Ixx * Iyy - Ixy ** 2
        return det_H

    @staticmethod
    def diff_of_gaussian(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply Difference of Gaussians."""
        sigma1 = kwargs.get('sigma1', 1.0)
        sigma2 = kwargs.get('sigma2', 2.0)

        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        gauss1 = filters.gaussian(gray, sigma=sigma1)
        gauss2 = filters.gaussian(gray, sigma=sigma2)

        return gauss1 - gauss2

    @staticmethod
    def laplacian_of_gaussian(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply Laplacian of Gaussian (LoG)."""
        sigma = kwargs.get('sigma', 2.0)

        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        result = ndimage.gaussian_laplace(gray, sigma=sigma)
        return result

    @staticmethod
    def shape_index(image: np.ndarray, **kwargs) -> np.ndarray:
        """Compute shape index from Hessian eigenvalues."""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Compute second derivatives
        Ixx = ndimage.sobel(ndimage.sobel(gray, axis=0), axis=0)
        Iyy = ndimage.sobel(ndimage.sobel(gray, axis=1), axis=1)
        Ixy = ndimage.sobel(ndimage.sobel(gray, axis=0), axis=1)

        # Eigenvalues
        trace = Ixx + Iyy
        det = Ixx * Iyy - Ixy ** 2

        discriminant = trace ** 2 - 4 * det
        discriminant = np.maximum(discriminant, 0)

        lambda1 = (trace + np.sqrt(discriminant)) / 2
        lambda2 = (trace - np.sqrt(discriminant)) / 2

        # Shape index
        with np.errstate(divide='ignore', invalid='ignore'):
            shape_idx = (2 / np.pi) * np.arctan((lambda2 + lambda1) / (lambda2 - lambda1))
            shape_idx = np.nan_to_num(shape_idx)

        return shape_idx

    @staticmethod
    def hog(image: np.ndarray, **kwargs) -> np.ndarray:
        """Compute Histogram of Oriented Gradients."""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Compute HOG features
        fd = feature.hog(gray, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, channel_axis=None)

        # Return visualization
        return fd[1] if isinstance(fd, tuple) else fd

    @staticmethod
    def shape_formula(image: np.ndarray, **kwargs) -> np.ndarray:
        """Compute shape-based features."""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        binary = gray > 0.5
        labeled = measure.label(binary)

        # Create output based on region properties
        output = np.zeros_like(gray)
        for region in measure.regionprops(labeled):
            # Use eccentricity as shape measure
            output[labeled == region.label] = region.eccentricity

        return output

    @staticmethod
    def contour_finding(image: np.ndarray, **kwargs) -> np.ndarray:
        """Find contours in image."""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Find contours
        contours = measure.find_contours(gray, 0.5)

        # Draw contours on blank image
        output = np.zeros_like(gray)
        for contour in contours:
            coords = contour.astype(int)
            coords = coords[(coords[:, 0] >= 0) & (coords[:, 0] < output.shape[0]) &
                           (coords[:, 1] >= 0) & (coords[:, 1] < output.shape[1])]
            if len(coords) > 0:
                output[coords[:, 0], coords[:, 1]] = 1.0

        return output

    @staticmethod
    def local_binary_pattern(image: np.ndarray, **kwargs) -> np.ndarray:
        """Compute Local Binary Pattern."""
        n_points = kwargs.get('n_points', 8)
        radius = kwargs.get('radius', 1)

        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Convert to uint8
        gray_uint8 = (gray * 255).astype(np.uint8)

        lbp = feature.local_binary_pattern(gray_uint8, n_points, radius, method='uniform')

        # Normalize
        return lbp / lbp.max() if lbp.max() > 0 else lbp

    # ========== IMAGE-SPECIFIC CLUSTERING (CIS) - 6 components ==========

    @staticmethod
    def slic_superpixels(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply SLIC superpixel segmentation."""
        n_segments = kwargs.get('n_segments', 100)

        if len(image.shape) != 3:
            image = np.stack([image] * 3, axis=2)

        segments = segmentation.slic(image, n_segments=n_segments, compactness=10)
        return segments.astype(np.float32) / segments.max()

    @staticmethod
    def chan_vese(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply Chan-Vese active contour segmentation."""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        cv_result = segmentation.chan_vese(gray, mu=0.25, lambda1=1, lambda2=1,
                                          tol=1e-3, max_num_iter=200,
                                          dt=0.5, init_level_set="checkerboard")
        return cv_result.astype(np.float32)

    @staticmethod
    def felsenszwalb(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply Felsenszwalb's efficient graph-based segmentation."""
        scale = kwargs.get('scale', 100)

        if len(image.shape) != 3:
            image = np.stack([image] * 3, axis=2)

        segments = segmentation.felzenszwalb(image, scale=scale, sigma=0.5, min_size=50)
        return segments.astype(np.float32) / segments.max()

    @staticmethod
    def quickshift(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply Quickshift segmentation."""
        kernel_size = kwargs.get('kernel_size', 5)

        if len(image.shape) != 3:
            image = np.stack([image] * 3, axis=2)

        segments = segmentation.quickshift(image, kernel_size=kernel_size, max_dist=10, ratio=1.0)
        return segments.astype(np.float32) / segments.max()

    @staticmethod
    def gabor_filter(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply Gabor filter bank."""
        frequency = kwargs.get('frequency', 0.1)

        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Apply Gabor filter at multiple orientations
        responses = []
        for theta in range(4):
            real, imag = gabor(gray, frequency=frequency, theta=theta * np.pi / 4)
            responses.append(np.sqrt(real**2 + imag**2))

        return np.mean(responses, axis=0)

    @staticmethod
    def watershed(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply watershed segmentation."""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # Edge detection
        edges = filters.sobel(gray)

        # Markers
        markers = np.zeros_like(gray, dtype=int)
        markers[gray < 0.3] = 1
        markers[gray > 0.7] = 2

        # Watershed
        segments = segmentation.watershed(edges, markers)
        return segments.astype(np.float32) / segments.max()

    # ========== GENERIC CLUSTERING (CG) - 10 components ==========

    @staticmethod
    def kmeans_plusplus(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply K-Means++ clustering."""
        n_clusters = kwargs.get('n_clusters', 3)

        # Reshape image to feature vectors
        original_shape = image.shape
        if len(image.shape) == 3:
            pixels = image.reshape(-1, image.shape[2])
        else:
            pixels = image.reshape(-1, 1)

        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)

        return labels.reshape(original_shape[:2]).astype(np.float32) / n_clusters

    @staticmethod
    def affinity_propagation(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply Affinity Propagation clustering."""
        # Reshape image
        original_shape = image.shape
        if len(image.shape) == 3:
            pixels = image.reshape(-1, image.shape[2])
        else:
            pixels = image.reshape(-1, 1)

        # Sample for efficiency
        if len(pixels) > 1000:
            indices = np.random.choice(len(pixels), 1000, replace=False)
            pixels_sample = pixels[indices]
        else:
            pixels_sample = pixels
            indices = np.arange(len(pixels))

        # Cluster
        ap = AffinityPropagation(random_state=42, max_iter=200)
        labels_sample = ap.fit_predict(pixels_sample)

        # Map back
        labels = np.zeros(len(pixels), dtype=int)
        labels[indices] = labels_sample

        n_clusters = len(np.unique(labels))
        return labels.reshape(original_shape[:2]).astype(np.float32) / max(n_clusters, 1)

    @staticmethod
    def mean_shift(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply Mean Shift clustering."""
        # Reshape image
        original_shape = image.shape
        if len(image.shape) == 3:
            pixels = image.reshape(-1, image.shape[2])
        else:
            pixels = image.reshape(-1, 1)

        # Sample for efficiency
        if len(pixels) > 1000:
            indices = np.random.choice(len(pixels), 1000, replace=False)
            pixels_sample = pixels[indices]
        else:
            pixels_sample = pixels
            indices = np.arange(len(pixels))

        # Cluster
        ms = MeanShift(max_iter=200)
        labels_sample = ms.fit_predict(pixels_sample)

        # Map back
        labels = np.zeros(len(pixels), dtype=int)
        labels[indices] = labels_sample

        n_clusters = len(np.unique(labels))
        return labels.reshape(original_shape[:2]).astype(np.float32) / max(n_clusters, 1)

    @staticmethod
    def spectral_clustering(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply Spectral clustering."""
        n_clusters = kwargs.get('n_clusters', 3)

        # Reshape image
        original_shape = image.shape
        if len(image.shape) == 3:
            pixels = image.reshape(-1, image.shape[2])
        else:
            pixels = image.reshape(-1, 1)

        # Sample for efficiency
        if len(pixels) > 1000:
            indices = np.random.choice(len(pixels), 1000, replace=False)
            pixels_sample = pixels[indices]
        else:
            pixels_sample = pixels
            indices = np.arange(len(pixels))

        # Cluster
        sc = SpectralClustering(n_clusters=n_clusters, random_state=42, n_init=10)
        labels_sample = sc.fit_predict(pixels_sample)

        # Map back
        labels = np.zeros(len(pixels), dtype=int)
        labels[indices] = labels_sample

        return labels.reshape(original_shape[:2]).astype(np.float32) / n_clusters

    @staticmethod
    def agglomerative(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply Agglomerative clustering."""
        n_clusters = kwargs.get('n_clusters', 3)

        # Reshape image
        original_shape = image.shape
        if len(image.shape) == 3:
            pixels = image.reshape(-1, image.shape[2])
        else:
            pixels = image.reshape(-1, 1)

        # Sample for efficiency
        if len(pixels) > 1000:
            indices = np.random.choice(len(pixels), 1000, replace=False)
            pixels_sample = pixels[indices]
        else:
            pixels_sample = pixels
            indices = np.arange(len(pixels))

        # Cluster
        agg = AgglomerativeClustering(n_clusters=n_clusters)
        labels_sample = agg.fit_predict(pixels_sample)

        # Map back
        labels = np.zeros(len(pixels), dtype=int)
        labels[indices] = labels_sample

        return labels.reshape(original_shape[:2]).astype(np.float32) / n_clusters

    @staticmethod
    def dbscan(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply DBSCAN clustering."""
        eps = kwargs.get('eps', 0.1)

        # Reshape image
        original_shape = image.shape
        if len(image.shape) == 3:
            pixels = image.reshape(-1, image.shape[2])
        else:
            pixels = image.reshape(-1, 1)

        # Sample for efficiency
        if len(pixels) > 1000:
            indices = np.random.choice(len(pixels), 1000, replace=False)
            pixels_sample = pixels[indices]
        else:
            pixels_sample = pixels
            indices = np.arange(len(pixels))

        # Cluster
        dbscan = DBSCAN(eps=eps, min_samples=5)
        labels_sample = dbscan.fit_predict(pixels_sample)

        # Map back
        labels = np.zeros(len(pixels), dtype=int)
        labels[indices] = labels_sample

        # Handle noise points (-1)
        labels = labels + 1
        n_clusters = len(np.unique(labels))

        return labels.reshape(original_shape[:2]).astype(np.float32) / max(n_clusters, 1)

    @staticmethod
    def optics(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply OPTICS clustering."""
        min_samples = kwargs.get('min_samples', 5)

        # Reshape image
        original_shape = image.shape
        if len(image.shape) == 3:
            pixels = image.reshape(-1, image.shape[2])
        else:
            pixels = image.reshape(-1, 1)

        # Sample for efficiency
        if len(pixels) > 1000:
            indices = np.random.choice(len(pixels), 1000, replace=False)
            pixels_sample = pixels[indices]
        else:
            pixels_sample = pixels
            indices = np.arange(len(pixels))

        # Cluster
        optics = OPTICS(min_samples=min_samples, max_eps=0.5)
        labels_sample = optics.fit_predict(pixels_sample)

        # Map back
        labels = np.zeros(len(pixels), dtype=int)
        labels[indices] = labels_sample

        # Handle noise points (-1)
        labels = labels + 1
        n_clusters = len(np.unique(labels))

        return labels.reshape(original_shape[:2]).astype(np.float32) / max(n_clusters, 1)

    @staticmethod
    def gaussian_mixtures(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply Gaussian Mixture Model clustering."""
        n_components = kwargs.get('n_components', 3)

        # Reshape image
        original_shape = image.shape
        if len(image.shape) == 3:
            pixels = image.reshape(-1, image.shape[2])
        else:
            pixels = image.reshape(-1, 1)

        # Cluster
        gmm = GaussianMixture(n_components=n_components, random_state=42, max_iter=100)
        labels = gmm.fit_predict(pixels)

        return labels.reshape(original_shape[:2]).astype(np.float32) / n_components

    @staticmethod
    def birch(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply BIRCH clustering."""
        n_clusters = kwargs.get('n_clusters', 3)

        # Reshape image
        original_shape = image.shape
        if len(image.shape) == 3:
            pixels = image.reshape(-1, image.shape[2])
        else:
            pixels = image.reshape(-1, 1)

        # Cluster
        birch = Birch(n_clusters=n_clusters)
        labels = birch.fit_predict(pixels)

        return labels.reshape(original_shape[:2]).astype(np.float32) / n_clusters

    @staticmethod
    def bisecting_kmeans(image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply Bisecting K-Means clustering."""
        n_clusters = kwargs.get('n_clusters', 3)

        # Reshape image
        original_shape = image.shape
        if len(image.shape) == 3:
            pixels = image.reshape(-1, image.shape[2])
        else:
            pixels = image.reshape(-1, 1)

        # Cluster
        bkm = BisectingKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = bkm.fit_predict(pixels)

        return labels.reshape(original_shape[:2]).astype(np.float32) / n_clusters


# Component registry mapping names to functions
COMPONENT_REGISTRY: Dict[str, Callable] = {
    # FIL - Filtering
    'gaussian_filter': SegmentationComponents.gaussian_filter,
    'adaptive_histogram_eq': SegmentationComponents.adaptive_histogram_eq,
    'thresholding': SegmentationComponents.thresholding,
    'convert_to_binary': SegmentationComponents.convert_to_binary,
    'leung_malik_filter': SegmentationComponents.leung_malik_filter,
    'median_blur': SegmentationComponents.median_blur,
    'brightness_enhance': SegmentationComponents.brightness_enhance,
    'color_space_change': SegmentationComponents.color_space_change,

    # MOR - Morphology
    'vertical_area_extraction': SegmentationComponents.vertical_area_extraction,
    'horizontal_area_extraction': SegmentationComponents.horizontal_area_extraction,
    'skeletonization': SegmentationComponents.skeletonization,
    'dilation': SegmentationComponents.dilation,
    'erosion': SegmentationComponents.erosion,
    'opening': SegmentationComponents.opening,
    'closing': SegmentationComponents.closing,
    'union_find': SegmentationComponents.union_find,
    'convex_hull': SegmentationComponents.convex_hull,
    'hole_filling': SegmentationComponents.hole_filling,
    'top_hat': SegmentationComponents.top_hat,
    'bottom_hat': SegmentationComponents.bottom_hat,

    # FD - Feature Detection
    'hough_transform': SegmentationComponents.hough_transform,
    'prewitt': SegmentationComponents.prewitt,
    'sobel': SegmentationComponents.sobel,
    'canny_edge': SegmentationComponents.canny_edge,
    'haar': SegmentationComponents.haar,
    'laplacian': SegmentationComponents.laplacian,
    'det_of_hessian': SegmentationComponents.det_of_hessian,
    'diff_of_gaussian': SegmentationComponents.diff_of_gaussian,
    'laplacian_of_gaussian': SegmentationComponents.laplacian_of_gaussian,
    'shape_index': SegmentationComponents.shape_index,
    'hog': SegmentationComponents.hog,
    'shape_formula': SegmentationComponents.shape_formula,
    'contour_finding': SegmentationComponents.contour_finding,
    'local_binary_pattern': SegmentationComponents.local_binary_pattern,

    # CIS - Image-specific Clustering
    'slic_superpixels': SegmentationComponents.slic_superpixels,
    'chan_vese': SegmentationComponents.chan_vese,
    'felsenszwalb': SegmentationComponents.felsenszwalb,
    'quickshift': SegmentationComponents.quickshift,
    'gabor_filter': SegmentationComponents.gabor_filter,
    'watershed': SegmentationComponents.watershed,

    # CG - Generic Clustering
    'kmeans_plusplus': SegmentationComponents.kmeans_plusplus,
    'affinity_propagation': SegmentationComponents.affinity_propagation,
    'mean_shift': SegmentationComponents.mean_shift,
    'spectral_clustering': SegmentationComponents.spectral_clustering,
    'agglomerative': SegmentationComponents.agglomerative,
    'dbscan': SegmentationComponents.dbscan,
    'optics': SegmentationComponents.optics,
    'gaussian_mixtures': SegmentationComponents.gaussian_mixtures,
    'birch': SegmentationComponents.birch,
    'bisecting_kmeans': SegmentationComponents.bisecting_kmeans
}


def apply_chromosome(image: np.ndarray, chromosome: list) -> np.ndarray:
    """
    Apply a chromosome (sequence of components) to an image.

    Args:
        image: Input image (H, W) or (H, W, C)
        chromosome: List of component names like ['FIL_gaussian_filter', 'MOR_dilation', ...]

    Returns:
        Processed image/segmentation mask
    """
    result = image.copy()

    for gene in chromosome:
        # Parse gene format: 'TYPE_component_name'
        parts = gene.split('_', 1)
        if len(parts) != 2:
            continue

        component_type, component_name = parts

        # Get component function
        if component_name in COMPONENT_REGISTRY:
            component_func = COMPONENT_REGISTRY[component_name]
            try:
                result = component_func(result)
            except Exception as e:
                print(f"Error applying {component_name}: {e}")
                # Continue with current result on error

    return result
