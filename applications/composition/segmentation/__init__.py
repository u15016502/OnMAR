"""
Image Segmentation Composition Application

This module provides automated composition of unsupervised image segmentation
algorithms using a genetic algorithm to evolve variable-length chromosomes
of algorithmic components.
"""

from applications.composition.segmentation.segmentation_application import SegmentationCompositionApplication
from applications.composition.segmentation.segmentation_model import (
    SegmentationComponents,
    COMPONENT_REGISTRY,
    apply_chromosome
)

__all__ = [
    'SegmentationCompositionApplication',
    'SegmentationComponents',
    'COMPONENT_REGISTRY',
    'apply_chromosome'
]
