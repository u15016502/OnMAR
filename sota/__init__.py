"""
SOTA (State-of-the-Art) Baselines Module

This module provides state-of-the-art AutoML baselines for comparison with
OnMAR and OffMAR approaches.
"""

from .autosklearn_wrapper import AutoSklearnWrapper

__all__ = ['AutoSklearnWrapper']
