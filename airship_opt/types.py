"""
Type definitions for airship hull optimization.

This module defines the core data structures used throughout the optimization
process, including parameters, coefficients, configuration, and results.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Any

Vector = List[float]

@dataclass
class Params:
    """Eight-parameter hull definition following Parsons' method."""
    rn: float   # nose curvature ratio (rn * D = R0)
    fr: float   # fineness ratio (L/D)
    xm: float   # station of max radius (0..1)
    k: float    # curvature knob at xm (semantic as in your spec)
    Xi: float   # transition from mid to tail (0..1, > xm)
    n: float    # max radius (unitless, typically <= 1)
    S: float    # slope control / segment shape knob (Parsons)
    t: float    # tail radius at x=1

@dataclass
class Coefs:
    """Polynomial coefficients for the three hull segments."""
    fore: Tuple[float, ...]  # polynomial coefficients fore segment
    mid:  Tuple[float, ...]  # polynomial coefficients mid segment
    tail: Tuple[float, ...]  # polynomial coefficients tail segment
    xm:   float              # max diameter station
    Xi:   float              # inflection point station

@dataclass
class Config:
    """Configuration for optimization and analysis."""
    vol_target: float        # target volume for constraint enforcement
    medium: Any              # working fluid properties (e.g. air, water)
    w_volume: float = 1.0    # volume penalty weight
    w_shape: float = 1.0     # shape penalty weight
    w_bounds: float = 1e6    # constraint violation penalty weight
    n_vol_steps: int = 1000  # integration steps for volume calculation
    n_bl_steps: int = 10000  # integration steps for boundary layer
    epsilon_join: float = 1e-9  # epsilon for continuity checks
    transition_point: float = None  # laminar-turbulent transition point (None = auto)
    speed_U: float = None    # free-stream speed [m/s], used to compute Re_V
    Re_V: float = None       # volume-based Reynolds number (if provided directly)

@dataclass
class IterRecord:
    """Record of a single optimization iteration."""
    iter: int
    x: Vector
    obj: float
    cd: float
    volume: float
    vol_err: float

@dataclass
class Result:
    """Complete optimization result."""
    params: Params
    coefs: Coefs
    cd: float
    volume: float
    objective: float
    history: List[IterRecord] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
