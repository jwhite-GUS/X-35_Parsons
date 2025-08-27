"""
Water medium for airship hull optimization.

Provides standard room-temperature water properties and factory
function for custom water conditions.
"""

from .base import Medium

def make_water(rho: float = 998.2, nu: float = 1.004e-6, name: str = "water_20C") -> Medium:
    """Create a water medium with specified properties.
    
    Args:
        rho: Density [kg/m^3], defaults to 20°C (998.2)
        nu: Kinematic viscosity [m^2/s], defaults to 20°C (~1.004e-6)
        name: Medium identifier
        
    Returns:
        Medium dataclass with specified properties
    """
    return Medium(name=name, rho=rho, nu=nu)
