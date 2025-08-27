"""
Custom medium for airship hull optimization.

Provides a factory function for creating a medium with 
user-specified properties.
"""

from .base import Medium

def make_custom(rho: float = 1.0, nu: float = 1.0e-6, name: str = "custom") -> Medium:
    """
    Create custom fluid medium with specified properties.
    
    Args:
        rho: Density [kg/m^3], defaults to unity
        nu: Kinematic viscosity [m^2/s], defaults to unity
        name: Medium name/description (defaults to "custom")
    """
    return Medium(name=name, rho=rho, nu=nu)
