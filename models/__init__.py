# Make key classes available directly from the package
from .base import PINN, PINN1D
from .kdv import KDV
from .kp import KP

# Define what gets imported with "from pinns.models import *"
__all__ = ['PINN', 'PINN1D', 'KDV', 'KP'] 