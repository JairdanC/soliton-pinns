"""
KdV PINN Solver Module
"""
from .kdv import KDV_LEGACY
from .types import (
    TrainingDomain,
    TestingDomain,
    Solutions,
    ErrorStats,
    TrainingStats
)

__all__ = [
    "KDV_LEGACY",
    "TrainingDomain",
    "TestingDomain",
    "Solutions",
    "ErrorStats",
    "TrainingStats"
]