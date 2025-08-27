"""
Media package for airship hull optimization.

Provides standard media (air, water) and custom fluid definitions for 
volume-based Reynolds number calculations.
"""

from .base import Medium
from .air import make_air
from .water import make_water
from .custom import make_custom

__all__ = [
    "Medium",
    "make_air",
    "make_water", 
    "make_custom"
]
