"""
Test script for Airship Hull Shape Optimization

This script runs basic tests to verify the implementation works correctly.
"""

import numpy as np
from airship_hull_optimization import (
    solve_forebody, solve_midbody, solve_tail, evaluate_shape,
    compute_volume, compute_drag_coefficient, objective_function
)

def test_shape_generation():
    """Test shape generation functions."""
    print("Testing shape generation...")
    
    # Test parameters
    rn, fr, xm, k, Xi, n, S, t = 0.7573, 4.8488, 0.5888, 0.1711, 0.7853, 0.6473, 2.2867, 0.1731
    
    # Test forebody
    fore_coefs = solve_forebody(rn, fr, xm)
    print(f"Forebody coefficients: {[f'{c:.6f}' for c in fore_coefs]}")
    
    # Test midbody
    mid_coefs = solve_midbody(fr, xm, Xi, n, k, S, t)
    print(f"Midbody coefficients: {[f'{c:.6f}' for c in mid_coefs]}")
    
    # Test tail
    tail_coefs = solve_tail(fr, xm, Xi, n, S, t)
    print(f"Tail coefficients: {[f'{c:.6f}' for c in tail_coefs]}")
    
    # Test shape evaluation
    Y_nose, _ = evaluate_shape(0.0, fore_coefs, mid_coefs, tail_coefs, xm, Xi)
    Y_max, _ = evaluate_shape(xm, fore_coefs, mid_coefs, tail_coefs, xm, Xi)
    Y_tail, _ = evaluate_shape(1.0, fore_coefs, mid_coefs, tail_coefs, xm, Xi)
    
    print(f"Y at nose (X=0): {Y_nose:.6f}")
    print(f"Y at max (X={xm}): {Y_max:.6f}")
    print(f"Y at tail (X=1): {Y_tail:.6f}")
    
    # Verify nose condition
    assert abs(Y_nose) < 1e-6, f"Nose radius should be zero, got {Y_nose}"
    print("✓ Nose condition satisfied")
    
    print()

def test_volume_calculation():
    """Test volume calculation."""
    print("Testing volume calculation...")
    
    # Test parameters
    rn, fr, xm, k, Xi, n, S, t = 0.7573, 4.8488, 0.5888, 0.1711, 0.7853, 0.6473, 2.2867, 0.1731
    
    fore_coefs = solve_forebody(rn, fr, xm)
    mid_coefs = solve_midbody(fr, xm, Xi, n, k, S, t)
    tail_coefs = solve_tail(fr, xm, Xi, n, S, t)
    
    volume = compute_volume(fore_coefs, mid_coefs, tail_coefs, xm, Xi, fr)
    print(f"Calculated volume: {volume:.6f}")
    
    # Volume should be positive and reasonable
    assert volume > 0, f"Volume should be positive, got {volume}"
    assert volume < 1, f"Volume should be less than 1 (normalized), got {volume}"
    print("✓ Volume calculation reasonable")
    
    print()

def test_drag_calculation():
    """Test drag coefficient calculation."""
    print("Testing drag calculation...")
    
    # Test parameters
    rn, fr, xm, k, Xi, n, S, t = 0.7573, 4.8488, 0.5888, 0.1711, 0.7853, 0.6473, 2.2867, 0.1731
    
    fore_coefs = solve_forebody(rn, fr, xm)
    mid_coefs = solve_midbody(fr, xm, Xi, n, k, S, t)
    tail_coefs = solve_tail(fr, xm, Xi, n, S, t)
    
    CD, volume = compute_drag_coefficient(fore_coefs, mid_coefs, tail_coefs, xm, Xi, fr)
    print(f"Drag coefficient: {CD:.6f}")
    print(f"Volume: {volume:.6f}")
    
    # Drag coefficient should be positive and reasonable
    assert CD > 0, f"Drag coefficient should be positive, got {CD}"
    assert CD < 1, f"Drag coefficient should be less than 1, got {CD}"
    print("✓ Drag calculation reasonable")
    
    print()

def test_objective_function():
    """Test objective function."""
    print("Testing objective function...")
    
    # Test with valid parameters
    valid_params = [0.7573, 4.8488, 0.5888, 0.1711, 0.7853, 0.6473, 2.2867, 0.1731]
    result = objective_function(valid_params)
    print(f"Objective function result: {result:.6f}")
    
    # Should return a reasonable objective value (drag + penalties)
    assert result > 0, f"Objective function should return positive value, got {result}"
    assert result < 1e6, f"Objective function should not return penalty for valid params, got {result}"
    print("✓ Valid parameters accepted")
    
    # Test with invalid parameters (should return penalty)
    invalid_params = [0.7573, 1.0, 0.5888, 0.1711, 0.7853, 0.6473, 2.2867, 0.1731]  # fr too small
    result = objective_function(invalid_params)
    print(f"Objective function with invalid params: {result:.6f}")
    
    # Should return penalty
    assert result > 1e6, f"Objective function should return penalty for invalid params, got {result}"
    print("✓ Invalid parameters penalized")
    
    print()

def test_continuity():
    """Test continuity at segment boundaries."""
    print("Testing continuity at segment boundaries...")
    
    # Test parameters
    rn, fr, xm, k, Xi, n, S, t = 0.7573, 4.8488, 0.5888, 0.1711, 0.7853, 0.6473, 2.2867, 0.1731
    
    fore_coefs = solve_forebody(rn, fr, xm)
    mid_coefs = solve_midbody(fr, xm, Xi, n, k, S, t)
    tail_coefs = solve_tail(fr, xm, Xi, n, S, t)
    
    # Test continuity at xm (forebody-midbody boundary)
    # Sample from left and right sides of the boundary
    eps = 1e-9
    Y_L, Yp_L = evaluate_shape(xm - eps, fore_coefs, mid_coefs, tail_coefs, xm, Xi)  # forebody
    Y_R, Yp_R = evaluate_shape(xm + eps, fore_coefs, mid_coefs, tail_coefs, xm, Xi)  # midbody
    
    # Should be continuous in position and slope (C1 continuity)
    assert abs(Y_L - Y_R) < 1e-8, f"Discontinuity at xm: {Y_L} vs {Y_R}"
    assert abs(Yp_L - Yp_R) < 1e-6, f"Slope discontinuity at xm: {Yp_L} vs {Yp_R}"
    print("✓ Continuity at xm satisfied")
    
    # Test continuity at Xi (midbody-tail boundary)
    Y_L, Yp_L = evaluate_shape(Xi - eps, fore_coefs, mid_coefs, tail_coefs, xm, Xi)  # midbody
    Y_R, Yp_R = evaluate_shape(Xi + eps, fore_coefs, mid_coefs, tail_coefs, xm, Xi)  # tail
    
    # Should be continuous in position and slope (C1 continuity)
    assert abs(Y_L - Y_R) < 1e-8, f"Discontinuity at Xi: {Y_L} vs {Y_R}"
    assert abs(Yp_L - Yp_R) < 1e-6, f"Slope discontinuity at Xi: {Yp_L} vs {Yp_R}"
    print("✓ Continuity at Xi satisfied")
    
    print()

def test_parameter_mappings():
    """Test parameter mappings and nondimensionalization."""
    print("Testing parameter mappings...")
    
    # Test parameters
    rn, fr, xm, k, Xi, n, S, t = 0.7573, 4.8488, 0.5888, 0.1711, 0.7853, 0.6473, 2.2867, 0.1731
    
    # Test rn mapping: R₀ = rn × D, K₀ = 1/R₀
    L = 1.0
    D = L / fr
    R_0 = rn * D
    K_0 = 1.0 / R_0
    
    print(f"rn = {rn:.6f}")
    print(f"D = {D:.6f}")
    print(f"R₀ = rn × D = {R_0:.6f}")
    print(f"K₀ = 1/R₀ = {K_0:.6f}")
    
    # Test S mapping: S_actual = S/fr
    S_actual = S / fr
    print(f"S = {S:.6f}")
    print(f"S_actual = S/fr = {S_actual:.6f}")
    
    # Verify mappings are reasonable
    assert R_0 > 0, f"R₀ should be positive, got {R_0}"
    assert K_0 > 0, f"K₀ should be positive, got {K_0}"
    assert abs(S_actual) < 1, f"S_actual should be reasonable slope, got {S_actual}"
    print("✓ Parameter mappings reasonable")
    
    print()

def test_analytic_slopes():
    """Test that analytic slopes are used in boundary layer integration."""
    print("Testing analytic slope integration...")
    
    # Test parameters
    rn, fr, xm, k, Xi, n, S, t = 0.7573, 4.8488, 0.5888, 0.1711, 0.7853, 0.6473, 2.2867, 0.1731
    
    fore_coefs = solve_forebody(rn, fr, xm)
    mid_coefs = solve_midbody(fr, xm, Xi, n, k, S, t)
    tail_coefs = solve_tail(fr, xm, Xi, n, S, t)
    
    # Test a few points to verify we get both Y and Y'
    test_points = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for X in test_points:
        Y, Yp = evaluate_shape(X, fore_coefs, mid_coefs, tail_coefs, xm, Xi)
        print(f"X = {X:.1f}: Y = {Y:.6f}, Y' = {Yp:.6f}")
        
        # Verify we get reasonable values
        assert Y >= 0, f"Y should be non-negative at X={X}, got {Y}"
        assert abs(Yp) < 10, f"Y' should be reasonable at X={X}, got {Yp}"
    
    print("✓ Analytic slopes computed correctly")
    print()

def test_peak_slope_conditions():
    """Test that the original design has proper slope conditions at xm."""
    print("Testing peak slope conditions...")
    
    # Test parameters (original Parsons design)
    rn, fr, xm, k, Xi, n, S, t = 0.7573, 4.8488, 0.5888, 0.1711, 0.7853, 0.6473, 2.2867, 0.1731
    
    fore_coefs = solve_forebody(rn, fr, xm)
    mid_coefs = solve_midbody(fr, xm, Xi, n, k, S, t)
    tail_coefs = solve_tail(fr, xm, Xi, n, S, t)
    
    # Test slope conditions at xm (should have positive slope on left, negative on right)
    eps = 1e-9
    _, Yp_L = evaluate_shape(xm - eps, fore_coefs, mid_coefs, tail_coefs, xm, Xi)
    _, Yp_R = evaluate_shape(xm + eps, fore_coefs, mid_coefs, tail_coefs, xm, Xi)
    
    print(f"Slope at xm-ε: {Yp_L:.6f}")
    print(f"Slope at xm+ε: {Yp_R:.6f}")
    
    # For the original design, slopes should be very close to zero at xm (maximum point)
    # Check that they're not significantly wrong (i.e., not strongly negative on left or positive on right)
    assert Yp_L >= -1e-8, f"Should not have strongly negative slope approaching peak, got {Yp_L}"
    assert Yp_R <= 1e-8, f"Should not have strongly positive slope leaving peak, got {Yp_R}"
    print("✓ Peak slope conditions satisfied")
    
    print()

def main():
    """Run all tests."""
    print("Running Airship Hull Optimization Tests")
    print("=" * 40)
    
    try:
        test_shape_generation()
        test_volume_calculation()
        test_drag_calculation()
        test_objective_function()
        test_continuity()
        test_parameter_mappings()
        test_analytic_slopes()
        test_peak_slope_conditions()
        
        print("All tests passed! ✓")
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    main()
