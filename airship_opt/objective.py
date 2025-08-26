"""
Objective function module for airship hull optimization.

This module implements the objective function with soft constraint penalties
for volume, shape constraints, and parameter bounds.
"""

from __future__ import annotations
from typing import Dict, Tuple
from .types import Params, Config
from .geometry import build_coefs, evaluate_shape
from .aero import compute_drag

def constraint_penalties(params: Params, cfg: Config) -> Dict[str, float]:
    """
    Compute soft penalties for constraint violations.
    
    Args:
        params: Hull parameters
        cfg: Configuration with penalty weights
        
    Returns:
        Dictionary of penalty terms
    """
    p = params
    pen = {}
    
    # Basic parameter bounds (soft penalties)
    if p.fr < 2.5: 
        pen['fr_min'] = cfg.w_bounds * (2.5 - p.fr)
    if not (0.05 < p.xm < 0.95): 
        pen['xm_bounds'] = cfg.w_bounds
    if not (0.02 < p.Xi < 0.98) or p.Xi <= p.xm: 
        pen['Xi_bounds'] = cfg.w_bounds
    if p.rn < 0: 
        pen['rn_positive'] = cfg.w_bounds
    if p.k < 0: 
        pen['k_positive'] = cfg.w_bounds
    if p.t <= 0 or p.t >= 1: 
        pen['t_bounds'] = cfg.w_bounds
    if p.n <= 0 or p.n >= 1: 
        pen['n_bounds'] = cfg.w_bounds
    if p.t >= p.n: 
        pen['t_lt_n'] = cfg.w_bounds  # tail radius must be smaller than radius at inflection
    if p.S < 0: 
        pen['S_positive'] = cfg.w_bounds
    
    # Order constraints
    if p.xm >= p.Xi:
        pen['order'] = cfg.w_bounds * (p.xm - p.Xi + 1e-6)
    
    # Radius ordering
    if not (0.0 <= p.t < p.n <= 1.5):
        pen['radius_order'] = cfg.w_bounds
    
    return pen

def peak_penalty(params: Params, cfg: Config) -> float:
    """
    Compute penalty for peak-at-xm constraint using slope sign changes.
    
    Args:
        params: Hull parameters
        cfg: Configuration with penalty weights
        
    Returns:
        Penalty value for peak constraint violations
    """
    C = build_coefs(params)
    eps = 1.0 / cfg.n_bl_steps  # Use same step size as drag integration
    
    # Check slope conditions at xm (should have positive slope on left, negative on right)
    _, Yp_L = evaluate_shape(params.xm - eps, C)
    _, Yp_R = evaluate_shape(params.xm + eps, C)
    
    # Add shape penalties (soft constraints) - peak should have negative slope on right
    pen = 0.0
    if Yp_L <= 0:  # Should have positive slope approaching peak from left
        pen += cfg.w_shape * (Yp_L ** 2)
    if Yp_R >= 0:  # Should have negative slope leaving peak to right
        pen += cfg.w_shape * (Yp_R ** 2)
    
    return pen

def objective(params: Params, cfg: Config) -> Tuple[float, Dict]:
    """
    Objective function to minimize (drag coefficient) with embedded constraints as penalties.
    
    Args:
        params: Hull parameters
        cfg: Configuration with penalty weights and targets
        
    Returns:
        Tuple of (objective_value, metadata_dict)
    """
    # Compute drag and volume
    CD, V = compute_drag(params, cfg)
    
    # Volume error penalty
    vol_err = (V - cfg.vol_target) / max(cfg.vol_target, 1e-12)
    obj = CD + cfg.w_volume * (vol_err ** 2)
    
    # Constraint penalties
    pens = constraint_penalties(params, cfg)
    obj += sum(pens.values())
    
    # Peak penalty
    obj += peak_penalty(params, cfg)
    
    # Metadata for debugging and analysis
    meta = {
        'CD': CD,
        'V': V,
        'vol_err': vol_err,
        **pens
    }
    
    return obj, meta
