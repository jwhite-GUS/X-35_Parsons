"""
Air medium for airship hull optimization.

Provides standard sea-level air properties and factory function
for custom air conditions.
"""

from .base import Medium

def make_air(rho: float = 1.225, nu: float = 1.46e-5, name: str = "air_sl") -> Medium:
    """Create an air medium with specified properties.
    
    Args:
        rho: Density [kg/m^3], defaults to sea-level (1.225)
        nu: Kinematic viscosity [m^2/s], defaults to sea-level (~1.46e-5)
        name: Medium identifier
        
    Returns:
        Medium dataclass with specified properties
    """
    return Medium(name=name, rho=rho, nu=nu)
