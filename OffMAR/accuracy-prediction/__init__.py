"""
OffMAR Accuracy Prediction Module

Offline Meta-learning for AutoML in Real-time with accuracy prediction for reuse decisions.
"""

from .offmar import OffMARAccuracyPrediction
from .config import (
    OffMARConfig,
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
    'OffMARAccuracyPrediction',
    'OffMARConfig',
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
