"""Geometry volume and explicitly bounded drag-surrogate calculations.

The repository does not yet contain the external-flow and axisymmetric
boundary-layer solvers used by Parsons. The available drag calculation is a
constant-edge-velocity screening surrogate and must be selected explicitly.
"""

from __future__ import annotations
import math
from typing import Tuple
from .types import Params, Coefs, Config
from .geometry import build_coefs, evaluate_shape
from .media.base import Medium
from .objectives.utils import resolve_reynolds

CONSTANT_EDGE_SURROGATE_V1 = "constant-edge-velocity-surrogate-v1"
SURROGATE_MODEL_STATUS = "screening-surrogate-open"
SURROGATE_RE_L_MIN = 1.0e5
SURROGATE_RE_L_MAX = 1.0e9
SURROGATE_TURBULENT_RE_S_MIN = 5.0e5
SURROGATE_TURBULENT_RE_S_MAX = 1.0e9


def surrogate_static_boundary() -> dict:
    """Return the machine-readable boundary that bars a value from hull evidence."""
    return {
        "quantity": "screening_coefficient",
        "model_id": CONSTANT_EDGE_SURROGATE_V1,
        "model_status": SURROGATE_MODEL_STATUS,
        "hull_evidence_eligible": False,
        "physical_magnitude_validated": False,
        "ranking_scope": "same-model same-flow relative software screening only",
    }


def _transition_point(params: Params, cfg: Config) -> float:
    value = (
        float(cfg.transition_point)
        if cfg.transition_point is not None
        else params.xm + 0.5 * (params.Xi - params.xm)
    )
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("Surrogate transition point must be finite and in [0, 1].")
    return value


def _validate_reynolds_envelope(Re_L: float) -> None:
    if not math.isfinite(Re_L) or not SURROGATE_RE_L_MIN <= Re_L <= SURROGATE_RE_L_MAX:
        raise ValueError(
            "Screening surrogate is bounded to "
            f"{SURROGATE_RE_L_MIN:.0e} <= Re_L <= {SURROGATE_RE_L_MAX:.0e}; "
            "this is a computational screening envelope, not physical validation."
        )


def laminar_flat_plate_benchmark(Re_L: float, n_steps: int = 50000) -> dict:
    """Compare the retained laminar integration with its closed-form integral.

    For a unit flat plate with constant edge velocity and
    ``Cf=1.328/sqrt(Re_L*s)``, the implemented momentum-thickness operation
    ``theta = integral(Cf/2 ds)`` has the closed form
    ``theta=1.328/sqrt(Re_L)``. This verifies arithmetic only.
    """
    Re_L = float(Re_L)
    if not math.isfinite(Re_L) or Re_L <= 0 or n_steps <= 0:
        raise ValueError("Positive finite Re_L and positive benchmark steps are required.")
    ds = 1.0 / n_steps
    numeric = 0.0
    for index in range(1, n_steps + 1):
        Re_s = Re_L * index * ds
        numeric += 0.5 * (1.328 / math.sqrt(Re_s)) * ds
    closed_form = 1.328 / math.sqrt(Re_L)
    relative_error = abs(numeric - closed_form) / closed_form
    return {
        "id": "unit-laminar-flat-plate-momentum-thickness",
        "Re_L": Re_L,
        "steps": n_steps,
        "numeric_theta_over_L": numeric,
        "closed_form_theta_over_L": closed_form,
        "relative_error": relative_error,
        "tolerance": 5.0e-3,
        "passed": relative_error <= 5.0e-3,
        "claim": "arithmetic benchmark only; not hull-drag validation",
    }


def surrogate_claim_metadata(
    params: Params, cfg: Config, medium: Medium, V_bar: float
) -> dict:
    """Build the complete non-evidence flag, envelope, and arithmetic check."""
    Re_V, Re_L = resolve_reynolds(
        U=cfg.speed_U,
        V_bar=V_bar,
        nu=medium.nu,
        ReV=cfg.Re_V,
        reference_length_m=cfg.reference_length_m,
    )
    transition = _transition_point(params, cfg)
    _validate_reynolds_envelope(Re_L)
    result = surrogate_static_boundary()
    result.update(
        {
            "screening_optimization_acknowledged": bool(
                cfg.allow_screening_optimization
            ),
            "validity_envelope": {
                "assessment": "computational-envelope-pass",
                "Re_V": Re_V,
                "Re_L": Re_L,
                "Re_L_min": SURROGATE_RE_L_MIN,
                "Re_L_max": SURROGATE_RE_L_MAX,
                "transition_point": transition,
                "transition_min": 0.0,
                "transition_max": 1.0,
                "first_turbulent_Re_s_min": SURROGATE_TURBULENT_RE_S_MIN,
                "turbulent_Re_s_max": SURROGATE_TURBULENT_RE_S_MAX,
                "constant_edge_velocity_ratio": 1.0,
                "assumptions": [
                    "incompressible attached smooth flow",
                    "prescribed transition",
                    "flat-plate skin-friction correlations",
                    "fixed model and flow condition for any ranking",
                ],
                "physical_validity": "unestablished",
            },
            "closed_form_benchmark": laminar_flat_plate_benchmark(Re_L),
        }
    )
    return result


def volume(params: Params, cfg: Config) -> float:
    """
    Compute nondimensional volume via π∫ Y^2 dx over [0,1].
    
    Args:
        params: Hull parameters
        cfg: Configuration with integration settings
        
    Returns:
        Volume (nondimensional, L=1)
    """
    C = build_coefs(params)
    n = cfg.n_vol_steps
    vol = 0.0
    Y_prev, _ = evaluate_shape(0.0, C)
    dx = 1.0 / n
    
    for i in range(1, n+1):
        X = i * dx
        Y, _ = evaluate_shape(X, C)
        # cross-sectional area A = pi * Y^2
        # integrate volume: V = ∫ A dx 
        vol += 0.5 * math.pi * (Y_prev**2 + Y**2) * dx
        Y_prev = Y
    
    return vol

def compute_drag(params: Params, cfg: Config, medium: Medium) -> Tuple[float, float]:
    """
    Constant-edge-velocity skin-friction screening coefficient and volume.
    
    This is not a source-faithful Parsons/Young solver. It assumes
    ``u_e/U_inf = 1``, prescribed transition, attached incompressible flow, and
    the retained laminar/turbulent skin-friction correlations. Its purpose is
    dimensional and software sensitivity work while the full source model
    remains OPEN.
    
    Args:
        params: Hull parameters
        cfg: Configuration with integration and transition settings
        medium: Working fluid properties
        
    Returns:
        Tuple of ``(C_screen, V_bar)`` where both values are nondimensional.
    """
    if cfg.drag_model != CONSTANT_EDGE_SURROGATE_V1:
        raise ValueError(
            "No validated default drag model exists. Select "
            f"drag_model={CONSTANT_EDGE_SURROGATE_V1!r} explicitly for the "
            "screening surrogate; do not treat its output as vehicle evidence."
        )

    # Volume for normalization 
    V = volume(params, cfg)
    
    # Use provided kinematic viscosity and compute Re_V
    nu = medium.nu  # No longer derived from a hardcoded Re_V
    
    # Resolve the physical flow condition against normalized geometry. The
    # configured Re_V now reaches every local skin-friction calculation.
    _, Re_L = resolve_reynolds(
        U=cfg.speed_U,
        V_bar=V,
        nu=nu,
        ReV=cfg.Re_V,
        reference_length_m=cfg.reference_length_m,
    )
    
    _validate_reynolds_envelope(Re_L)
    X_transition = _transition_point(params, cfg)
    
    # Integrate boundary layer momentum thickness:
    theta = 0.0    # momentum thickness
    s = 0.0        # running surface length
    X_prev = 0.0
    C = build_coefs(params)  # Build coefficients once
    # We will take small steps along X to integrate.
    N_steps = cfg.n_bl_steps
    turbulent_started = False
    for j in range(1, N_steps+1):
        X = j / N_steps  # increment X uniformly
        _, Yp = evaluate_shape(X, C)
        
        # Compute incremental arc length ds (taking into account slope):
        dX = X - X_prev
        ds = math.sqrt(1.0 + Yp*Yp) * dX
        
        # Update running length and previous values:
        s += ds
        X_prev = X
        # Determine if flow is laminar or turbulent at this segment:
        Re_s = Re_L * s
        if X <= X_transition:
            # Laminar segment
            if Re_s <= 0: 
                Cf = 0.0
            else:
                Cf = 1.328 / math.sqrt(Re_s + 1e-16)  # add small number to avoid zero-division
        else:
            # Turbulent segment
            if Re_s <= 0:
                Cf = 0.0
            else:
                if not turbulent_started and Re_s < SURROGATE_TURBULENT_RE_S_MIN:
                    raise ValueError(
                        "First turbulent surrogate station lies below the declared "
                        f"Re_s floor {SURROGATE_TURBULENT_RE_S_MIN:.0e}."
                    )
                if Re_s > SURROGATE_TURBULENT_RE_S_MAX:
                    raise ValueError(
                        "Turbulent surrogate station exceeds the declared "
                        f"Re_s ceiling {SURROGATE_TURBULENT_RE_S_MAX:.0e}."
                    )
                turbulent_started = True
                # use a power-law/Prandtl-Schlichting formula for Cf
                Cf = 0.455 / (math.log10(Re_s + 1e-16)**2.58)
        
        # Update momentum thickness via momentum integral (d theta = Cf/2 * ds for small segment):
        theta += 0.5 * Cf * ds
    
    # Tail-end radius:
    Y_end, _ = evaluate_shape(1.0, C)
    r_TE = Y_end
    
    # Parsons Eq. (5) uses a trailing-edge multiplier
    # (u_e/U_inf)^((H+5)/2). Under this surrogate's explicit constant-edge
    # assumption the ratio is one. The former `(1 + H)` multiplier was not the
    # source equation and has therefore been removed.
    CD = 4 * math.pi * r_TE * theta / (V ** (2.0 / 3.0))
    
    return CD, V
