# Make key classes available directly from the package
from .base import PINN, PINN1D #legacy for kp
from .network import MLP
from .kdv import KDV as KDV_LEGACY  # Rename to avoid collision
from .kp import KP
from .kdv_2.kdv import KDV_LEGACY          # Import the new refactored model


# Define what gets imported with "from pinns.models import *"
__all__ = ['PINN', 'PINN1D', 'MLP', 'KDV_LEGACY', 'KDV_LEGACY', 'KP'] 