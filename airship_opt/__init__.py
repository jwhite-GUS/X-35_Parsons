"""
Airship Hull Shape Optimization - Parsons' Method Implementation

A modular package for airship hull shape optimization using Parsons' eight-parameter
method with piecewise polynomial geometry and Young's drag formulation.

Main interfaces:
- geometry: solve_forebody, solve_midbody, solve_tail, evaluate_shape
- aero: volume, compute_drag
- objective: objective function with soft constraints
- optimize: nelder_mead optimization driver
- io_viz: save/load results and visualization
"""

from .types import Params, Config, Result, Coefs
from .geometry import solve_forebody, solve_midbody, solve_tail, evaluate_shape, build_coefs, sample_profile
from .aero import volume, compute_drag
from .objective import objective, constraint_penalties, peak_penalty
from .optimize import nelder_mead
from .io_viz import save_result, load_result

__version__ = "1.0.0"
__all__ = [
    "Params", "Config", "Result", "Coefs",
    "solve_forebody", "solve_midbody", "solve_tail", "evaluate_shape", "build_coefs", "sample_profile",
    "volume", "compute_drag",
    "objective", "constraint_penalties", "peak_penalty",
    "nelder_mead",
    "save_result", "load_result"
]
