"""
Airship hull geometry and bounded optimization research tools.

The package retains Parsons' eight-parameter geometry. Its current drag path is
an explicitly selected screening surrogate, not a faithful Young/Parsons solver.

Main interfaces:
- geometry: solve_forebody, solve_midbody, solve_tail, evaluate_shape
- aero: volume, compute_drag
- objective: objective function with soft constraints
- optimize: nelder_mead optimization driver
- io_viz: save/load results and visualization
"""

from .types import Params, Config, Result, Coefs
from .geometry import solve_forebody, solve_midbody, solve_tail, evaluate_shape, build_coefs, sample_profile
from .aero import (
    CONSTANT_EDGE_SURROGATE_V1,
    SURROGATE_MODEL_STATUS,
    compute_drag,
    laminar_flat_plate_benchmark,
    surrogate_claim_metadata,
    surrogate_static_boundary,
    volume,
)
from .objective import objective, constraint_penalties, peak_penalty
from .optimize import nelder_mead
from .io_viz import save_result, load_result

__version__ = "1.0.0"
__all__ = [
    "Params", "Config", "Result", "Coefs",
    "solve_forebody", "solve_midbody", "solve_tail", "evaluate_shape", "build_coefs", "sample_profile",
    "CONSTANT_EDGE_SURROGATE_V1", "SURROGATE_MODEL_STATUS",
    "volume", "compute_drag", "laminar_flat_plate_benchmark",
    "surrogate_claim_metadata", "surrogate_static_boundary",
    "objective", "constraint_penalties", "peak_penalty",
    "nelder_mead",
    "save_result", "load_result"
]
