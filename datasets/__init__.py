"""Datasets module.

Provides dataset loaders for various applications:
- ImageDatasetLoader: For CNN configuration (MNIST, CIFAR, etc.)
- SegmentationDatasetLoader: For segmentation composition (BSD500, COVID, Pascal)
- TextDatasetLoader: For Fuzzy ART generation (ChatGPT, Enron, IMDB)
"""

from .image_datasets import ImageDatasetLoader
from .segmentation_datasets import SegmentationDatasetLoader, SegmentationDataLoader
from .text_datasets import TextDatasetLoader

# Alias for backward compatibility
ClusteringDatasetLoader = TextDatasetLoader

__all__ = [
    'ImageDatasetLoader',
    'SegmentationDatasetLoader',
    'SegmentationDataLoader',
    'TextDatasetLoader',
    'ClusteringDatasetLoader'
]
