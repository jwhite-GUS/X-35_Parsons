#!/usr/bin/env python3
"""
Debug script to compare modular version with original working version.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

# Import modular version
from airship_opt.types import Params, Config
from airship_opt.geometry import build_coefs, evaluate_shape
from airship_opt.aero import volume

# Import original version functions
from airship_hull_optimization import solve_forebody, solve_midbody, solve_tail, compute_volume

def test_original():
    """Test the original working version."""
    print("=== ORIGINAL VERSION ===")
    
    # Original parameters
    rn, fr, xm, k, Xi, n, S, t = 0.7573, 4.8488, 0.5888, 0.1711, 0.7853, 0.6473, 2.2867, 0.1731
    
    # Solve coefficients
    fore_coefs = solve_forebody(rn, fr, xm)
    mid_coefs = solve_midbody(fr, xm, Xi, n, k, S, t)
    tail_coefs = solve_tail(fr, xm, Xi, n, S, t)
    
    print(f"Original fore coefficients: {fore_coefs}")
    print(f"Original mid coefficients: {mid_coefs}")
    print(f"Original tail coefficients: {tail_coefs}")
    
    # Compute volume
    vol_orig = compute_volume(fore_coefs, mid_coefs, tail_coefs, xm, Xi, fr)
    print(f"Original volume: {vol_orig}")
    
    return vol_orig

def test_modular():
    """Test the modular version."""
    print("\n=== MODULAR VERSION ===")
    
    # Create parameters
    params = Params(rn=0.7573, fr=4.8488, xm=0.5888, k=0.1711,
                   Xi=0.7853, n=0.6473, S=2.2867, t=0.1731)
    cfg = Config(vol_target=0.020257)
    
    # Build coefficients
    coefs = build_coefs(params)
    print(f"Modular fore coefficients: {coefs.fore}")
    print(f"Modular mid coefficients: {coefs.mid}")
    print(f"Modular tail coefficients: {coefs.tail}")
    
    # Compute volume
    vol_mod = volume(params, cfg)
    print(f"Modular volume: {vol_mod}")
    
    return vol_mod

def test_shape_evaluation():
    """Test shape evaluation at key points."""
    print("\n=== SHAPE EVALUATION COMPARISON ===")
    
    # Original parameters
    rn, fr, xm, k, Xi, n, S, t = 0.7573, 4.8488, 0.5888, 0.1711, 0.7853, 0.6473, 2.2867, 0.1731
    
    # Original coefficients
    fore_coefs = solve_forebody(rn, fr, xm)
    mid_coefs = solve_midbody(fr, xm, Xi, n, k, S, t)
    tail_coefs = solve_tail(fr, xm, Xi, n, S, t)
    
    # Modular coefficients
    params = Params(rn=rn, fr=fr, xm=xm, k=k, Xi=Xi, n=n, S=S, t=t)
    coefs = build_coefs(params)
    
    # Test points
    test_points = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    
    for x in test_points:
        # Original evaluation (would need to implement this)
        # orig_y, orig_yp = evaluate_shape_original(x, fore_coefs, mid_coefs, tail_coefs, xm, Xi)
        
        # Modular evaluation
        mod_y, mod_yp = evaluate_shape(x, coefs)
        
        print(f"x={x:.1f}: Y={mod_y:.6f}, Y'={mod_yp:.6f}")

if __name__ == "__main__":
    vol_orig = test_original()
    vol_mod = test_modular()
    test_shape_evaluation()
    
    print(f"\n=== COMPARISON ===")
    print(f"Original volume: {vol_orig}")
    print(f"Modular volume: {vol_mod}")
    print(f"Difference: {abs(vol_orig - vol_mod):.6f}")
    print(f"Ratio: {vol_mod/vol_orig:.6f}")
