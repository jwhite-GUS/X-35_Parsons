"""
Airship Hull Shape Optimization - Faithful Replication of Parsons' Work

This script implements an eight-parameter airship hull shape optimization system
that replicates Parsons' original work using piecewise polynomial equations and
Young's drag formula.

Author: Implementation based on Parsons' original work
Date: 2024
"""

import math
import numpy as np
import random
from typing import Tuple, List, Optional

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Target volume for constraint enforcement (will be set from original design)
VOL_TARGET = None  # Will be set dynamically

# Reynolds number for drag calculation
RE_VOL = 1e7

# Shape factor for turbulent boundary layer
H_TE = 1.4

# Integration parameters
N_VOLUME = 1000      # Volume integration slices
N_DRAG = 10000       # Drag integration steps

# Optimization parameters
MAX_ITERATIONS = 200
CONVERGENCE_TOLERANCE = 1e-6

# Penalty weights for objective function
W_VOL = 1e3          # Volume constraint weight
W_SHAPE = 1e4        # Shape constraint weight

# =============================================================================
# SHAPE PARAMETERIZATION MODULE
# =============================================================================

def solve_forebody(rn: float, fr: float, xm: float) -> List[float]:
    """
    Solve coefficients for 4th-degree polynomial forebody (nose to X_m).
    
    Boundary conditions:
      Y(0) = 0 (nose tip radius)
      Y'(0) = 0 (horizontal tangent at nose)
      Y(Xm) = D/2 (maximum radius at X_m, where D = L/fr)
      Y'(Xm) = 0 (slope zero at maximum diameter)
    Uses rn (nose curvature parameter) to set second derivative at nose.
    
    Args:
        rn: Nose radius curvature parameter (nondimensional)
        fr: Fineness ratio (Length/Diameter)
        xm: Location of maximum diameter as fraction of length
        
    Returns:
        List of coefficients [a4, a3, a2, a1, a0] for Y = a4*X^4 + a3*X^3 + a2*X^2 + a1*X + a0
    """
    L = 1.0  # normalize length to 1 for calculation
    D = L / fr  # maximum diameter
    
    # Boundary conditions at X=0:
    a0 = 0.0               # Y(0) = 0
    a1 = 0.0               # Y'(0) = 0
    
    # Nose curvature: map rn to curvature K_0
    # Define: R_0 = rn * D (nondimensional radius of curvature)
    # Then: K_0 = 1/R_0 = 1/(rn * D)
    R_0 = rn * D
    K_0 = 1.0 / R_0
    a2 = K_0 / 2.0        # Y''(0) = 2*a2 = K_0
    
    # At X = X_m: impose Y(Xm) = D/2 and Y'(Xm) = 0
    M = xm  # for readability
    
    # Solve linear system for a3, a4:
    # Y(M) = a4*M^4 + a3*M^3 + a2*M^2 = D/2
    # Y'(M) = 4*a4*M^3 + 3*a3*M^2 + 2*a2*M = 0
    
    # From Y'(M) = 0: 4*a4*M^3 + 3*a3*M^2 = -2*a2*M
    # From Y(M) = D/2: a4*M^4 + a3*M^3 = D/2 - a2*M^2
    
    # Solve explicitly:
    a3 = (2*D - 2*a2*M*M) / (M*M*M)
    a4 = -(3.0/(4*M)) * a3 - a2/(2*M*M)
    
    return [a4, a3, a2, a1, a0]


def solve_midbody(fr: float, xm: float, Xi: float, n: float, k: float, S: float, t: float) -> List[float]:
    """
    Solve coefficients for 5th-degree polynomial midbody (X_m to X_i).
    
    Boundary conditions:
      At X = X_m (u=0): Y = D/2, Y' = 0. We allow a curvature discontinuity at X_m:
                        Y''(X_m)_mid = K_mid, determined by k (nondim curvature param).
      At X = X_i (u = X_i - X_m): Y = R_i, Y' = S_actual, Y'' = 0.
    R_i is radius at inflection (from n param), S_actual is the actual slope at inflection derived from S param.
    
    Args:
        fr: Fineness ratio
        xm: Location of maximum diameter
        Xi: Location of inflection point
        n: Radius at inflection point (fraction of max radius)
        k: Curvature at maximum diameter
        S: Slope at inflection point (nondimensional)
        t: Tail-end radius parameter
        
    Returns:
        List of coefficients [b5, b4, b3, b2, b1, b0] for local polynomial Y(u) on [0, X_i - X_m]
    """
    L = 1.0
    D = L / fr
    
    # Compute target values at segment boundaries:
    Y_m = D / 2.0               # Y at X_m (max radius)
    Yp_m = 0.0                  # slope at X_m
    
    # Curvature at X_m for midbody (could differ from forebody curvature):
    # k is nondimensional curvature: k = (-2 * x_m^2 / D) * K_mid. We solve actual K_mid:
    K_mid = - k * D / (2 * (xm**2))
    
    # Set up local coordinate u = X - X_m, so u=0 at X_m and u = L_mid at X_i.
    L_mid = Xi - xm
    
    # Known coefficients from u=0 conditions:
    b0 = Y_m
    b1 = Yp_m
    b2 = K_mid / 2.0   # since Y''(0) = 2*b2 = K_mid
    
    # Conditions at u = L_mid (X = X_i):
    R_i = n * (D / 2.0)    # actual radius at inflection point (n = 2*R_i/D)
    
    # Convert nondimensional S parameter to actual slope at X_i:
    # Define S as nondimensional slope: S = Y'(X_i) * (L / D)
    # Then: Y'(X_i) = S * (D / L) = S / fr
    S_actual = S / fr
    
    Y_i = R_i
    Yp_i = S_actual
    Ypp_i = 0.0  # inflection: curvature zero at X_i
    
    # Solve linear system for b3, b4, b5 using conditions at u = L_mid.
    u = L_mid
    
    # Form equations:
    # Y(u) = b5*u^5 + b4*u^4 + b3*u^3 + b2*u^2 + b1*u + b0 = Y_i
    # Y'(u) = 5*b5*u^4 + 4*b4*u^3 + 3*b3*u^2 + 2*b2*u + b1 = Yp_i
    # Y''(u) = 20*b5*u^3 + 12*b4*u^2 + 6*b3*u + 2*b2 = Ypp_i = 0
    
    U2, U3, U4, U5 = u**2, u**3, u**4, u**5
    
    # Matrix for left-hand side coefficients:
    A = np.array([
        [ U3,     U4,      U5     ],   # coefficients of [b3, b4, b5] in (I)
        [ 3*U2,   4*U3,    5*U4   ],   # coefficients in (II)
        [ 6*u,    12*U2,   20*U3  ]    # coefficients in (III)
    ], dtype=float)
    
    # Right-hand side constants:
    B = np.array([
        Y_i - (b2*U2 + b1*u + b0),
        Yp_i - (2*b2*u + b1),
        - (2*b2)  # from (III)
    ], dtype=float)
    
    # Solve for b3, b4, b5:
    b3, b4, b5 = np.linalg.solve(A, B)
    
    return [b5, b4, b3, b2, b1, b0]


def solve_tail(fr: float, xm: float, Xi: float, n: float, S: float, t: float) -> List[float]:
    """
    Solve coefficients for 5th-degree polynomial tail (X_i to X=L).
    
    Boundary conditions:
      At X = X_i (v=0): Y = R_i, Y' = S_actual, Y'' = 0  (inflection at X_i).
      At X = L (v = 1 - X_i): Y = T (tail radius), Y' = 0, Y'' = 0 (smooth closed or cylindrical tail end).
    
    Args:
        fr: Fineness ratio
        xm: Location of maximum diameter
        Xi: Location of inflection point
        n: Radius at inflection point
        S: Slope at inflection point (nondimensional)
        t: Tail-end radius parameter
        
    Returns:
        List of coefficients [c5, c4, c3, c2, c1, c0] for local poly Y(v) on [0, L - X_i]
    """
    L = 1.0
    D = L / fr
    
    # Starting values at X_i:
    R_i = n * (D / 2.0)
    
    # Convert nondim S parameter to actual slope at X_i (same as in midbody):
    S_actual = S / fr
    
    Y_i = R_i
    Yp_i = S_actual
    Ypp_i = 0.0
    
    # Local coordinate v = X - X_i, with v=0 at X_i and v_end = L_tail = 1 - X_i at tail end.
    v_end = 1.0 - Xi
    
    # Coeffs from X_i conditions:
    c0 = Y_i
    c1 = Yp_i
    c2 = Ypp_i / 2.0  # = 0 because Y''(X_i) = 0
    
    # Conditions at X = L (v = v_end):
    T = (t * D) / 2.0    # tail radius (tail param t = 2T/D)
    Y_L = T
    Yp_L = 0.0
    Ypp_L = 0.0
    
    # Form equations for c3, c4, c5 at v_end:
    # Y(v_end) = c5*v_end^5 + c4*v_end^4 + c3*v_end^3 + c2*v_end^2 + c1*v_end + c0 = Y_L
    # Y'(v_end) = 5*c5*v_end^4 + 4*c4*v_end^3 + 3*c3*v_end^2 + 2*c2*v_end + c1 = 0
    # Y''(v_end) = 20*c5*v_end^3 + 12*c4*v_end^2 + 6*c3*v_end + 2*c2 = 0
    
    V2, V3, V4, V5 = v_end**2, v_end**3, v_end**4, v_end**5
    
    # Reorder equations for clarity: [Y, Y', Y'']
    A = np.array([
        [ V3,      V4,      V5    ],   # coefficients of [c3, c4, c5] in Y(v_end)
        [ 3*V2,    4*V3,    5*V4  ],   # from Y'(v_end) = 0
        [ 6*v_end, 12*V2,   20*V3 ]    # from Y''(v_end) = 0
    ], dtype=float)
    
    # Right side:
    B = np.array([
        Y_L - (c2*V2 + c1*v_end + c0),   # Y condition
        - (2*c2*v_end + c1),             # Y' condition
        -2*c2                          # Y'' condition (since Y''(v_end) = 0)
    ], dtype=float)
    
    c3, c4, c5 = np.linalg.solve(A, B)
    
    return [c5, c4, c3, c2, c1, c0]


def evaluate_shape(X: float, fore_coefs: List[float], mid_coefs: List[float], 
                  tail_coefs: List[float], xm: float, Xi: float) -> Tuple[float, float]:
    """
    Evaluate the hull radius Y and slope dY/dX at a given axial location X (0 <= X <= 1).
    Uses the piecewise polynomial coefficients for forebody, midbody, and tail.
    
    Args:
        X: Axial position (0 to 1)
        fore_coefs: Forebody polynomial coefficients [a4, a3, a2, a1, a0]
        mid_coefs: Midbody polynomial coefficients [b5, b4, b3, b2, b1, b0]
        tail_coefs: Tail polynomial coefficients [c5, c4, c3, c2, c1, c0]
        xm: Location of maximum diameter
        Xi: Location of inflection point
        
    Returns:
        Tuple (Y, Y') - radius and slope at X
    """
    if X < 0: 
        X = 0
    if X > 1:
        X = 1
        
    a4, a3, a2, a1, a0 = fore_coefs
    b5, b4, b3, b2, b1, b0 = mid_coefs
    c5, c4, c3, c2, c1, c0 = tail_coefs
    
    if X <= xm:
        # Forebody segment
        Y = a4*X**4 + a3*X**3 + a2*X**2 + a1*X + a0
        Yp = 4*a4*X**3 + 3*a3*X**2 + 2*a2*X + a1
        return Y, Yp
    elif X <= Xi:
        # Midbody segment
        u = X - xm
        Y = b5*u**5 + b4*u**4 + b3*u**3 + b2*u**2 + b1*u + b0
        Yp = 5*b5*u**4 + 4*b4*u**3 + 3*b3*u**2 + 2*b2*u + b1
        return Y, Yp
    else:
        # Tail segment
        v = X - Xi
        Y = c5*v**5 + c4*v**4 + c3*v**3 + c2*v**2 + c1*v + c0
        Yp = 5*c5*v**4 + 4*c4*v**3 + 3*c3*v**2 + 2*c2*v + c1
        return Y, Yp


# =============================================================================
# AERODYNAMIC ANALYSIS MODULE
# =============================================================================

def compute_volume(fore_coefs: List[float], mid_coefs: List[float], 
                  tail_coefs: List[float], xm: float, Xi: float, fr: float) -> float:
    """
    Compute the volume of the axisymmetric hull by integrating cross-sectional area.
    Uses trapezoidal integration over fine slices along X from 0 to 1.
    
    Args:
        fore_coefs: Forebody polynomial coefficients
        mid_coefs: Midbody polynomial coefficients
        tail_coefs: Tail polynomial coefficients
        xm: Location of maximum diameter
        Xi: Location of inflection point
        fr: Fineness ratio
        
    Returns:
        Volume in cubic units (normalized)
    """
    vol = 0.0
    Y_prev, _ = evaluate_shape(0.0, fore_coefs, mid_coefs, tail_coefs, xm, Xi)
    dx = 1.0 / N_VOLUME
    
    for i in range(1, N_VOLUME+1):
        X = i * dx
        Y, _ = evaluate_shape(X, fore_coefs, mid_coefs, tail_coefs, xm, Xi)
        # cross-sectional area A = pi * Y^2
        # integrate volume: V = ∫ A dx 
        vol += 0.5 * math.pi * (Y_prev**2 + Y**2) * dx
        Y_prev = Y
        
    return vol


def compute_drag_coefficient(fore_coefs: List[float], mid_coefs: List[float], 
                           tail_coefs: List[float], xm: float, Xi: float, fr: float, 
                           Re_vol: float = RE_VOL, transition_point: Optional[float] = None) -> Tuple[float, float]:
    """
    Compute drag coefficient C_D using a boundary-layer method (Young's formula).
    
    Args:
        fore_coefs: Forebody polynomial coefficients
        mid_coefs: Midbody polynomial coefficients
        tail_coefs: Tail polynomial coefficients
        xm: Location of maximum diameter
        Xi: Location of inflection point
        fr: Fineness ratio
        Re_vol: Reference volume-based Reynolds number
        transition_point: Transition point (if None, use halfway between xm and Xi)
        
    Returns:
        Tuple (CD, volume) - drag coefficient and volume
    """
    # Volume for normalization and to determine fluid viscosity (for constant Re_vol)
    volume = compute_volume(fore_coefs, mid_coefs, tail_coefs, xm, Xi, fr)
    
    # Compute effective kinematic viscosity to achieve desired volume Reynolds number:
    # Rv = (V)^(1/3) * U / nu, take U=1 for simplicity -> nu = (V)^(1/3) / Rv.
    nu = (volume ** (1.0/3.0)) / Re_vol
    
    # Determine transition point
    if transition_point is None:
        X_transition = xm + 0.5 * (Xi - xm)  # halfway between X_m and X_i
    else:
        X_transition = transition_point
    
    if X_transition > 1.0:
        X_transition = 1.0
    
    # Integrate boundary layer momentum thickness:
    theta = 0.0    # momentum thickness
    s = 0.0        # running surface length
    X_prev = 0.0
    Y_prev, Yp_prev = evaluate_shape(X_prev, fore_coefs, mid_coefs, tail_coefs, xm, Xi)
    
    # We will take small steps along X to integrate.
    for j in range(1, N_DRAG+1):
        X = j / N_DRAG  # increment X uniformly
        Y, Yp = evaluate_shape(X, fore_coefs, mid_coefs, tail_coefs, xm, Xi)
        
        # Compute incremental arc length using analytic slope:
        dX = X - X_prev
        ds = math.sqrt(1.0 + Yp*Yp) * dX
        
        # Update running length and previous values:
        s += ds
        X_prev = X
        Y_prev = Y
        Yp_prev = Yp
        
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
    Y_end, _ = evaluate_shape(1.0, fore_coefs, mid_coefs, tail_coefs, xm, Xi)
    r_TE = Y_end
    
    # Assume a shape factor at the trailing edge (turbulent boundary layer ~1.4):
    H_TE = 1.4
    
    # Compute drag coefficient using Young's formula:
    CD = (4 * math.pi * r_TE * theta / (volume ** (2.0/3.0))) * (1 + H_TE)
    
    return CD, volume


# =============================================================================
# OPTIMIZATION MODULE
# =============================================================================

def objective_function(params: List[float]) -> float:
    """
    Objective function to minimize (drag coefficient) with embedded constraints as penalties.
    
    Args:
        params: Array of 8 parameters [rn, fr, xm, k, Xi, n, S, t]
        
    Returns:
        Drag coefficient plus penalty terms
    """
    rn, fr, xm, k, Xi, n, S, t = params
    
    # Initialize penalty
    penalty = 0.0
    
    # Constraint checks with tighter bounds to avoid degeneracy:
    if fr < 2.501:  # Add small margin to avoid hitting exact bound
        penalty += 1000 * (2.501 - fr)
    if not (0.05 < xm < 0.95):  # Keep away from edges
        penalty += 1000
    if not (0.02 < Xi < 0.98) or Xi <= xm + 0.02:  # Ensure minimum segment lengths
        penalty += 1000
    if rn < 0: 
        penalty += 1000
    if k < 0: 
        penalty += 1000
    if t <= 0.05 or t >= n - 0.02:  # Keep tail radius away from bounds
        penalty += 1000
    if n <= 0.05 or n >= 0.95:  # Keep inflection radius away from bounds
        penalty += 1000
    if t >= n: 
        penalty += 1000  # tail radius must be smaller than radius at inflection
    if S < 0: 
        penalty += 1000
    
    # If any basic constraint failed, return a large value immediately:
    if penalty > 0:
        return 1e6 + penalty
    
    # Solve shape polynomials for given parameters
    try:
        fore_coefs = solve_forebody(rn, fr, xm)
        mid_coefs  = solve_midbody(fr, xm, Xi, n, k, S, t)
        tail_coefs = solve_tail(fr, xm, Xi, n, S, t)
    except np.linalg.LinAlgError:
        # If polynomial solving failed (singular matrix, etc.), apply high penalty
        return 1e6
    
    # Compute volume and drag coefficient (reuse volume from drag calculation)
    CD, volume = compute_drag_coefficient(fore_coefs, mid_coefs, tail_coefs, xm, Xi, fr)
    
    # Start with drag coefficient
    obj = CD
    
    # Add volume penalty (soft constraint)
    if VOL_TARGET is not None:
        vol_error = (volume - VOL_TARGET) / VOL_TARGET
        obj += W_VOL * (vol_error ** 2)
    
    # Check if max radius is at X_m using slope sign change (more robust)
    eps = 1.0 / N_DRAG  # Use same step size as drag integration
    _, Yp_L = evaluate_shape(xm - eps, fore_coefs, mid_coefs, tail_coefs, xm, Xi)
    _, Yp_R = evaluate_shape(xm + eps, fore_coefs, mid_coefs, tail_coefs, xm, Xi)
    
    # Add shape penalties (soft constraints) - peak should have negative slope on right
    if Yp_L <= 0:  # Should have positive slope approaching peak from left
        obj += W_SHAPE * (Yp_L ** 2)
    if Yp_R >= 0:  # Should have negative slope leaving peak to right
        obj += W_SHAPE * (Yp_R ** 2)
    
    return obj


def nelder_mead_optimize(initial_guess: List[float]) -> Tuple[List[float], float]:
    """
    Perform optimization using Nelder-Mead simplex method.
    
    Args:
        initial_guess: Initial parameter guess [rn, fr, xm, k, Xi, n, S, t]
        
    Returns:
        Tuple (optimized_params, min_objective) - optimized parameters and minimum objective
    """
    res = np.array(initial_guess, dtype=float)
    
    # Create simplex
    simplex = [res.copy()]
    # Simplex initial step variations (small perturbations for each param)
    for i in range(len(initial_guess)):
        perturbed = res.copy()
        perturbed[i] *= (1.05 if perturbed[i] != 0 else 0.001 + 1.0)  # 5% perturbation or small if zero
        simplex.append(perturbed)
    simplex = np.array(simplex)
    
    # Perform Nelder-Mead loop
    for itr in range(MAX_ITERATIONS):
        # Order simplex by objective values
        vals = [objective_function(x) for x in simplex]
        simplex = simplex[np.argsort(vals)]
        vals.sort()
        best_val = vals[0]
        worst_val = vals[-1]
        
        # Check convergence (simple criterion: relative improvement small)
        if itr > 0 and abs(worst_val - best_val) < CONVERGENCE_TOLERANCE:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        worst = simplex[-1]
        reflected = centroid + 1.0 * (centroid - worst)
        f_ref = objective_function(reflected)
        
        if f_ref < vals[0]:
            # Expansion
            expanded = centroid + 2.0 * (centroid - worst)
            f_exp = objective_function(expanded)
            if f_exp < f_ref:
                simplex[-1] = expanded
            else:
                simplex[-1] = reflected
        elif f_ref < vals[-2]:
            simplex[-1] = reflected
        else:
            # Contraction
            contracted = centroid + 0.5 * (centroid - worst)
            f_con = objective_function(contracted)
            if f_con < worst_val:
                simplex[-1] = contracted
            else:
                # Reduction
                for j in range(1, len(simplex)):
                    simplex[j] = simplex[0] + 0.5 * (simplex[j] - simplex[0])
    
    optimized_params = simplex[0]
    result = objective_function(optimized_params)
    
    return optimized_params.tolist(), result


# =============================================================================
# VALIDATION AND TESTING FUNCTIONS
# =============================================================================

def validate_original_design() -> None:
    """Validate the implementation against the original Parsons design."""
    global VOL_TARGET
    
    print("=== VALIDATING ORIGINAL DESIGN ===")
    
    # Known design parameters (from Parsons' original work)
    original_params = [0.7573, 4.8488, 0.5888, 0.1711, 0.7853, 0.6473, 2.2867, 0.1731]
    
    # Compute drag for the original design
    fore = solve_forebody(*original_params[:3])
    mid = solve_midbody(original_params[1], original_params[2], original_params[4], 
                       original_params[5], original_params[3], original_params[6], original_params[7])
    tail = solve_tail(original_params[1], original_params[2], original_params[4], 
                     original_params[5], original_params[6], original_params[7])
    
    CD_original, vol_original = compute_drag_coefficient(fore, mid, tail, 
                                                       original_params[2], original_params[4], original_params[1])
    
    # Set the volume target from the actual original design
    VOL_TARGET = vol_original
    
    print(f"Original design parameters: {original_params}")
    print(f"Original design volume = {vol_original:.6f}")
    print(f"Original design drag coefficient C_D = {CD_original:.6f}")
    print(f"Volume target = {VOL_TARGET:.6f}")
    print(f"Volume error = {abs(vol_original - VOL_TARGET)/VOL_TARGET*100:.2f}%")
    print()


def run_random_optimizations(num_runs: int = 3) -> None:
    """Run optimization from multiple random starting points."""
    print(f"=== RUNNING {num_runs} RANDOM OPTIMIZATIONS ===")
    
    for run in range(1, num_runs + 1):
        print(f"\n--- Run {run} ---")
        
        # Random initial parameters within reasonable ranges
        init = [
            random.uniform(0.5, 1.0),    # rn
            random.uniform(3.0, 6.0),    # fr
            random.uniform(0.3, 0.7),    # xm
            random.uniform(0.0, 0.3),    # k
            random.uniform(0.6, 0.9),    # Xi
            random.uniform(0.4, 0.8),    # n
            random.uniform(1.0, 3.0),    # S
            random.uniform(0.1, 0.3)     # t
        ]
        
        # Ensure Xi > xm for consistency
        if init[4] <= init[2]:
            init[4] = init[2] + 0.1
            if init[4] > 0.99: 
                init[4] = 0.99
        
        print(f"Initial parameters: {[f'{x:.4f}' for x in init]}")
        
        # Optimize from this random start
        try:
            params_opt, min_objective = nelder_mead_optimize(init)
            print(f"Optimized parameters: {[f'{x:.4f}' for x in params_opt]}")
            print(f"Minimum objective: {min_objective:.6e}")
            
            # Compute true CD and volume from optimized parameters
            fore = solve_forebody(*params_opt[:3])
            mid = solve_midbody(params_opt[1], params_opt[2], params_opt[4], 
                              params_opt[5], params_opt[3], params_opt[6], params_opt[7])
            tail = solve_tail(params_opt[1], params_opt[2], params_opt[4], 
                            params_opt[5], params_opt[6], params_opt[7])
            
            CD_opt, vol_opt = compute_drag_coefficient(fore, mid, tail, params_opt[2], params_opt[4], params_opt[1])
            print(f"Optimized CD: {CD_opt:.6f}")
            print(f"Optimized volume: {vol_opt:.6f}")
            print(f"Volume error: {abs(vol_opt - VOL_TARGET)/VOL_TARGET*100:.2f}%")
            
        except Exception as e:
            print(f"Optimization failed: {e}")
    
    print()


def generate_shape_data(params: List[float], num_points: int = 100) -> Tuple[List[float], List[float]]:
    """Generate shape data for plotting."""
    fore = solve_forebody(*params[:3])
    mid = solve_midbody(params[1], params[2], params[4], params[5], params[3], params[6], params[7])
    tail = solve_tail(params[1], params[2], params[4], params[5], params[6], params[7])
    
    X_points = np.linspace(0, 1, num_points)
    Y_points = []
    
    for X in X_points:
        Y, _ = evaluate_shape(X, fore, mid, tail, params[2], params[4])
        Y_points.append(Y)
    
    return X_points.tolist(), Y_points


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("Airship Hull Shape Optimization - Parsons' Method Implementation")
    print("=" * 60)
    
    # Validate against original design
    validate_original_design()
    
    # Run optimization from known good starting point
    print("=== OPTIMIZATION FROM KNOWN STARTING POINT ===")
    initial_guess = [0.75, 4.8, 0.6, 0.1, 0.78, 0.65, 2.0, 0.17]
    print(f"Initial guess: {initial_guess}")
    
    try:
        optimized_params, min_objective = nelder_mead_optimize(initial_guess)
        print(f"Optimized parameters: {[f'{x:.4f}' for x in optimized_params]}")
        print(f"Minimum objective: {min_objective:.6e}")
        
        # Compute true CD and volume from optimized parameters
        fore = solve_forebody(*optimized_params[:3])
        mid = solve_midbody(optimized_params[1], optimized_params[2], optimized_params[4], 
                          optimized_params[5], optimized_params[3], optimized_params[6], optimized_params[7])
        tail = solve_tail(optimized_params[1], optimized_params[2], optimized_params[4], 
                        optimized_params[5], optimized_params[6], optimized_params[7])
        
        CD_opt, vol_opt = compute_drag_coefficient(fore, mid, tail, optimized_params[2], optimized_params[4], optimized_params[1])
        print(f"Optimized CD: {CD_opt:.6f}")
        print(f"Optimized volume: {vol_opt:.6f}")
        print(f"Volume error: {abs(vol_opt - VOL_TARGET)/VOL_TARGET*100:.2f}%")
        
    except Exception as e:
        print(f"Optimization failed: {e}")
    
    print()
    
    # Run random optimizations
    run_random_optimizations(3)
    
    print("Optimization complete!")


if __name__ == "__main__":
    main()
