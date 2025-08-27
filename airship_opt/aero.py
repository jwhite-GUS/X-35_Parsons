"""
Aerodynamics module for airship hull analysis.

This module implements volume calculation and drag coefficient computation
using Young's formula with boundary layer momentum thickness integration.
"""

from __future__ import annotations
import math
from typing import Tuple
from .types import Params, Coefs, Config
from .geometry import build_coefs, evaluate_shape
from .media.base import Medium
from .objectives.utils import compute_ReV

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
    Young-style axisymmetric skin-friction drag (nondimensional) and volume.
    
    Uses boundary layer momentum thickness integration with analytic slopes
    for arc length calculation. Assumes laminar flow until transition point,
    then turbulent flow afterward.
    
    Args:
        params: Hull parameters
        cfg: Configuration with integration and transition settings
        medium: Working fluid properties
        
    Returns:
        Tuple of (CD, V) where CD is drag coefficient and V is volume
    """
    # Volume for normalization 
    V = volume(params, cfg)
    
    # Use provided kinematic viscosity and compute Re_V
    nu = medium.nu  # No longer derived from a hardcoded Re_V
    
    # Compute effective Re_V based on provided U or direct Re_V value
    Re_V = compute_ReV(U=cfg.speed_U, V=V, nu=nu, ReV=cfg.Re_V)
    
    # Determine transition point (approximate) - halfway between X_m and X_i for demonstration:
    if cfg.transition_point is not None:
        X_transition = cfg.transition_point
    else:
        X_transition = params.xm + 0.5 * (params.Xi - params.xm)
        if X_transition > 1.0:
            X_transition = 1.0
    
    # Integrate boundary layer momentum thickness:
    theta = 0.0    # momentum thickness
    s = 0.0        # running surface length
    X_prev = 0.0
    C = build_coefs(params)  # Build coefficients once
    Y_prev, _ = evaluate_shape(X_prev, C)
    
    # We will take small steps along X to integrate.
    N_steps = cfg.n_bl_steps
    for j in range(1, N_steps+1):
        X = j / N_steps  # increment X uniformly
        Y, Yp = evaluate_shape(X, C)
        
        # Compute incremental arc length ds (taking into account slope):
        dX = X - X_prev
        ds = math.sqrt(1.0 + Yp*Yp) * dX
        
        # Update running length and previous values:
        s += ds
        X_prev = X
        Y_prev = Y
        
        # Determine if flow is laminar or turbulent at this segment:
        if X <= X_transition:
            # Laminar segment
            Re_s = s / nu  # local Reynolds number along surface
            if Re_s <= 0: 
                Cf = 0.0
            else:
                Cf = 1.328 / math.sqrt(Re_s + 1e-16)  # add small number to avoid zero-division
        else:
            # Turbulent segment
            Re_s = s / nu
            if Re_s <= 0:
                Cf = 0.0
            else:
                # use a power-law/Prandtl-Schlichting formula for Cf
                Cf = 0.455 / (math.log10(Re_s + 1e-16)**2.58)
        
        # Update momentum thickness via momentum integral (d theta = Cf/2 * ds for small segment):
        theta += 0.5 * Cf * ds
    
    # Tail-end radius:
    Y_end, _ = evaluate_shape(1.0, C)
    r_TE = Y_end
    
    # Assume a shape factor at the trailing edge (turbulent boundary layer ~1.4):
    H_TE = 1.4
    
    # Compute drag coefficient using Young's formula:
    CD = (4 * math.pi * r_TE * theta / (V ** (2.0/3.0))) * (1 + H_TE)
    
    return CD, V
