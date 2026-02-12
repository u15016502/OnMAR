"""
OnMAR Design Prediction Module

Online Meta-learning for AutoML in Real-time with direct design prediction.
"""

from .onmar import OnMARDesignPrediction
from .config import (
    OnMARConfig,
    KNN_CONFIG,
    RF_CONFIG,
    XGBOOST_CONFIG,
    CONSERVATIVE_CONFIG,
    AGGRESSIVE_CONFIG,
    BALANCED_CONFIG,
    CNN_CONFIG,
    SEGMENTATION_CONFIG,
    FUZZYART_CONFIG
)

__all__ = [
    'OnMARDesignPrediction',
    'OnMARConfig',
    'KNN_CONFIG',
    'RF_CONFIG',
    'XGBOOST_CONFIG',
    'CONSERVATIVE_CONFIG',
    'AGGRESSIVE_CONFIG',
    'BALANCED_CONFIG',
    'CNN_CONFIG',
    'SEGMENTATION_CONFIG',
    'FUZZYART_CONFIG'
]
