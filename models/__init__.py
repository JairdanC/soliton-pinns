# Make key classes available directly from the package
from .network import MLP
from .kdv.kdv import KDV


# Define what gets imported with "from pinns.models import *"
__all__ = ['MLP', 'KDV'] 
