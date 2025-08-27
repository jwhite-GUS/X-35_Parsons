"""
Utility functions for Reynolds number calculations.

This module provides helper functions for working with volume-based
Reynolds numbers and fluid properties in airship hull optimization.
"""

from typing import Optional

def compute_ReV(U: float | None, V: float, nu: float, ReV: float | None) -> float:
    """Compute volume-based Reynolds number.
    
    Args:
        U: Free-stream speed [m/s], optional
        V: Hull volume [m^3]
        nu: Fluid kinematic viscosity [m^2/s]
        ReV: Direct Reynolds number input, optional
        
    Returns:
        Volume-based Reynolds number Re_V = U * V^(1/3) / nu
        
    Raises:
        ValueError: If neither U nor ReV is provided
    """
    if ReV is not None:
        return ReV
    if U is None:
        raise ValueError("Either ReV or U must be provided to compute Re_V.")
    return U * (V ** (1/3.0)) / nu
