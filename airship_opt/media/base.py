"""
Base medium interface for airship hull optimization.

This module defines the Medium dataclass that encapsulates
fluid properties used for Reynolds number calculations.
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class Medium:
    """Physical properties of a working fluid.
    
    Args:
        name: Identifier for the fluid
        rho: Density [kg/m^3]
        nu: Kinematic viscosity [m^2/s] (primary for Re_V)
    """
    name: str
    rho: float  # density [kg/m^3]
    nu: float   # kinematic viscosity [m^2/s]  (primary for Re_V)
