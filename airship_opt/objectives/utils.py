"""
Utility functions for objective function calculations.

This module provides helper functions for computing quantities
used in objective function evaluation.
"""

from typing import Optional

def compute_ReV(U: Optional[float], V: float, nu: float, ReV: Optional[float]) -> float:
    """Compute volume-based Reynolds number following Parsons/Young convention.

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
        return float(ReV)
    if U is None:
        raise ValueError("Either --ReV or --U must be provided to compute Re_V.")
    return float(U) * (V ** (1.0/3.0)) / float(nu)
