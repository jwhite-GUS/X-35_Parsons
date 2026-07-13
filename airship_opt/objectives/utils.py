"""
Utility functions for objective function calculations.

This module provides helper functions for computing quantities
used in objective function evaluation.
"""

import math
from typing import Optional, Tuple

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
        value = float(ReV)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Re_V must be positive and finite.")
        return value
    if U is None:
        raise ValueError("Either --ReV or --U must be provided to compute Re_V.")
    U = float(U)
    V = float(V)
    nu = float(nu)
    if not all(math.isfinite(value) for value in (U, V, nu)):
        raise ValueError("U, V, and nu must be finite.")
    if U <= 0 or V <= 0 or nu <= 0:
        raise ValueError("U, physical V, and nu must be positive.")
    return U * (V ** (1.0 / 3.0)) / nu


def resolve_reynolds(
    *,
    U: Optional[float],
    V_bar: float,
    nu: float,
    ReV: Optional[float],
    reference_length_m: Optional[float],
) -> Tuple[float, float]:
    """Resolve volume- and length-based Reynolds numbers for normalized geometry.

    ``V_bar`` is the nondimensional volume ``V/L^3``. A direct ``ReV`` needs no
    physical length. Deriving ``ReV`` from speed requires the physical reference
    length so that ``V = V_bar * L^3`` is a physical cubic volume.

    Returns:
        ``(Re_V, Re_L)`` where ``Re_L = U L / nu = Re_V / V_bar^(1/3)``.
    """
    V_bar = float(V_bar)
    nu = float(nu)
    if not math.isfinite(V_bar) or V_bar <= 0:
        raise ValueError("Nondimensional volume V_bar must be positive and finite.")
    if not math.isfinite(nu) or nu <= 0:
        raise ValueError("Kinematic viscosity nu must be positive and finite.")

    if ReV is None:
        if U is None or reference_length_m is None:
            raise ValueError(
                "Deriving Re_V from speed requires both --U and --reference-length-m; "
                "otherwise provide --ReV directly."
            )
        length = float(reference_length_m)
        if not math.isfinite(length) or length <= 0:
            raise ValueError("Reference length must be positive and finite.")
        physical_volume = V_bar * length**3
        resolved_ReV = compute_ReV(U=U, V=physical_volume, nu=nu, ReV=None)
    else:
        resolved_ReV = compute_ReV(U=None, V=1.0, nu=nu, ReV=ReV)

    Re_L = resolved_ReV / (V_bar ** (1.0 / 3.0))
    return resolved_ReV, Re_L
