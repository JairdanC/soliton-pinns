"""
KdV PINN Solver Module
"""
from .kdv import KDV
from .types import (
    TrainingDomain,
    TestingDomain,
    Solutions,
    ErrorStats,
    TrainingStats
)

__all__ = [
    "KDV",
    "TrainingDomain",
    "TestingDomain",
    "Solutions",
    "ErrorStats",
    "TrainingStats"
]