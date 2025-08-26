"""
Geometry module for airship hull shape generation.

This module implements Parsons' piecewise polynomial method for generating
airship hull shapes from eight parameters.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple
from .types import Params, Coefs

def solve_forebody(params: Params) -> Tuple[float, ...]:
    """
    Solve coefficients for 4th-degree polynomial forebody (nose to X_m).
    
    Boundary conditions:
      Y(0) = 0 (nose tip radius)
      Y'(0) = 0 (horizontal tangent at nose)
      Y(Xm) = D/2 (maximum radius at X_m, where D = L/fr)
      Y'(Xm) = 0 (slope zero at maximum diameter)
    Uses rn (nose curvature parameter) to set second derivative at nose.
    
    Returns coefficients [a4, a3, a2, a1, a0] for Y = a4*X^4 + a3*X^3 + a2*X^2 + a1*X + a0.
    """
    L = 1.0  # normalize length to 1 for calculation
    D = L / params.fr  # maximum diameter
    
    # Boundary conditions at X=0:
    a0 = 0.0               # Y(0) = 0
    a1 = 0.0               # Y'(0) = 0
    
    # Nose curvature: map rn to curvature K_0
    # Define: R_0 = rn * D (nondimensional radius of curvature)
    # Then: K_0 = 1/R_0 = 1/(rn * D)
    R_0 = params.rn * D
    K_0 = 1.0 / R_0
    a2 = K_0 / 2.0        # Y''(0) = 2*a2 = K_0
    
    # At X = X_m: impose Y(Xm) = D/2 and Y'(Xm) = 0
    M = params.xm  # for readability
    
    # Solve linear system for a3, a4:
    # Y(M) = a4*M^4 + a3*M^3 + a2*M^2 = D/2
    # Y'(M) = 4*a4*M^3 + 3*a3*M^2 + 2*a2*M = 0
    
    # From Y'(M) = 0: 4*a4*M^3 + 3*a3*M^2 = -2*a2*M
    # From Y(M) = D/2: a4*M^4 + a3*M^3 = D/2 - a2*M^2
    
    # Solve explicitly:
    a3 = (2*D - 2*a2*M*M) / (M*M*M)
    a4 = -(3.0/(4*M)) * a3 - a2/(2*M*M)
    
    return (a4, a3, a2, a1, a0)

def solve_midbody(params: Params) -> Tuple[float, ...]:
    """
    Solve coefficients for 5th-degree polynomial midbody (X_m to X_i).
    
    Boundary conditions:
      At X = X_m (u=0): Y = D/2, Y' = 0. We allow a curvature discontinuity at X_m:
                        Y''(X_m)_mid = K_mid, determined by k (nondim curvature param).
      At X = X_i (u = X_i - X_m): Y = R_i, Y' = S_actual, Y'' = 0.
    R_i is radius at inflection (from n param), S_actual is the actual slope at inflection derived from S param.
    
    Returns coefficients [b5, b4, b3, b2, b1, b0] for local polynomial Y(u) on [0, X_i - X_m].
    """
    L = 1.0
    D = L / params.fr
    # Compute target values at segment boundaries:
    Y_m = D / 2.0               # Y at X_m (max radius)
    Yp_m = 0.0                  # slope at X_m
    # Curvature at X_m for midbody (could differ from forebody curvature):
    # k is nondimensional curvature: k = (-2 * x_m^2 / D) * K_mid. We solve actual K_mid:
    K_mid = - params.k * D / (2 * (params.xm**2))
    # Set up local coordinate u = X - X_m, so u=0 at X_m and u = L_mid at X_i.
    L_mid = params.Xi - params.xm
    # Known coefficients from u=0 conditions:
    b0 = Y_m
    b1 = Yp_m
    b2 = K_mid / 2.0   # since Y''(0) = 2*b2 = K_mid
    # Conditions at u = L_mid (X = X_i):
    R_i = params.n * (D / 2.0)    # actual radius at inflection point (n = 2*R_i/D)
    # Convert nondimensional S parameter to actual slope at X_i:
    # From nomenclature: S_i (nondim) = [ -2 * fr * (X_i - X_m) / (1 - r_t) ] * (actual slope at X_i).
    # Solve actual slope: 
    if (1 - params.t) != 0:
        S_actual = - params.S * ((1 - params.t) / (2 * params.fr * (params.Xi - params.xm)))
    else:
        S_actual = - params.S * (1.0)  # if t≈1 (very blunt tail), handle separately (here not expected).
    Y_i = R_i
    Yp_i = S_actual
    Ypp_i = 0.0  # inflection: curvature zero at X_i
    # Solve linear system for b3, b4, b5 using conditions at u = L_mid.
    u = L_mid
    # Form equations:
    # Y(u) = b5*u^5 + b4*u^4 + b3*u^3 + b2*u^2 + b1*u + b0 = Y_i
    # Y'(u) = 5*b5*u^4 + 4*b4*u^3 + 3*b3*u^2 + 2*b2*u + b1 = Yp_i
    # Y''(u) = 20*b5*u^3 + 12*b4*u^2 + 6*b3*u + 2*b2 = Ypp_i = 0
    # Plug in known b0,b1,b2:
    # Simplify system for unknowns b3, b4, b5:
    # (I)   b5*u^5 + b4*u^4 + b3*u^3 = Y_i - (b2*u^2 + b1*u + b0)
    # (II)  5*b5*u^4 + 4*b4*u^3 + 3*b3*u^2 = Yp_i - (2*b2*u + b1)
    # (III) 20*b5*u^3 + 12*b4*u^2 + 6*b3*u = - (2*b2)   [since Ypp_i = 0]
    # Set up and solve 3x3 linear system for [b3, b4, b5]:
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
    return (b5, b4, b3, b2, b1, b0)

def solve_tail(params: Params) -> Tuple[float, ...]:
    """
    Solve coefficients for 5th-degree polynomial tail (X_i to X=L).
    
    Boundary conditions:
      At X = X_i (v=0): Y = R_i, Y' = S_actual, Y'' = 0  (inflection at X_i).
      At X = L (v = 1 - X_i): Y = T (tail radius), Y' = 0, Y'' = 0 (smooth closed or cylindrical tail end).
    
    Returns coefficients [c5, c4, c3, c2, c1, c0] for local poly Y(v) on [0, L - X_i].
    """
    L = 1.0
    D = L / params.fr
    # Starting values at X_i:
    R_i = params.n * (D / 2.0)
    # Convert nondim S parameter to actual slope at X_i (same as in midbody):
    S_actual = 0.0
    if (1 - params.t) != 0:
        S_actual = - params.S * ((1 - params.t) / (2 * params.fr * (params.Xi - params.xm)))
    Y_i = R_i
    Yp_i = S_actual
    Ypp_i = 0.0
    # Local coordinate v = X - X_i, with v=0 at X_i and v_end = L_tail = 1 - X_i at tail end.
    v_end = 1.0 - params.Xi
    # Coeffs from X_i conditions:
    c0 = Y_i
    c1 = Yp_i
    c2 = Ypp_i / 2.0  # = 0 because Y''(X_i) = 0
    # Conditions at X = L (v = v_end):
    T = (params.t * D) / 2.0    # tail radius (tail param t = 2T/D)
    Y_L = T
    Yp_L = 0.0
    Ypp_L = 0.0
    # Form equations for c3, c4, c5 at v_end:
    # Y(v_end) = c5*v_end^5 + c4*v_end^4 + c3*v_end^3 + c2*v_end^2 + c1*v_end + c0 = Y_L
    # Y'(v_end) = 5*c5*v_end^4 + 4*c4*v_end^3 + 3*c3*v_end^2 + 2*c2*v_end + c1 = 0
    # Y''(v_end) = 20*c5*v_end^3 + 12*c4*v_end^2 + 6*c3*v_end + 2*c2 = 0
    # Solve linear system for c3, c4, c5:
    V2, V3, V4, V5 = v_end**2, v_end**3, v_end**4, v_end**5
    A = np.array([
        [ V3,      V4,      V5    ],   # coefficients of [c3, c4, c5] in Y(v_end)
        [ 6*v_end, 12*V2,   20*V3 ],   # from Y''(v_end) = 0   (note: easier to use Y'' and Y')
        [ 3*V2,    4*V3,    5*V4  ]    # from Y'(v_end) = 0
    ], dtype=float)
    # Right side:
    B = np.array([
        Y_L - (c2*V2 + c1*v_end + c0),   # Y condition
        -2*c2,                          # Y'' condition (since Y''(v_end) = 0)
        - (2*c2*v_end + c1)             # Y' condition
    ], dtype=float)
    c3, c4, c5 = np.linalg.solve(A, B)
    return (c5, c4, c3, c2, c1, c0)

def build_coefs(params: Params) -> Coefs:
    """
    Convenience: solve all three segments and return a Coefs bundle.
    """
    return Coefs(
        fore=solve_forebody(params),
        mid=solve_midbody(params),
        tail=solve_tail(params),
        xm=params.xm,
        Xi=params.Xi,
    )

def evaluate_shape(x: float, coefs: Coefs) -> Tuple[float, float]:
    """
    Evaluate the hull radius Y and slope dY/dX at a given axial location X (0 <= X <= 1).
    Uses the piecewise polynomial coefficients for forebody, midbody, and tail.
    
    Returns:
        Tuple of (Y, Y') where Y is the radius and Y' is the slope at position x.
    """
    if x < 0: 
        x = 0
    if x > 1:
        x = 1
        
    a4, a3, a2, a1, a0 = coefs.fore
    b5, b4, b3, b2, b1, b0 = coefs.mid
    c5, c4, c3, c2, c1, c0 = coefs.tail
    
    if x <= coefs.xm:
        # Forebody segment
        Y = a4*x**4 + a3*x**3 + a2*x**2 + a1*x + a0
        Yp = 4*a4*x**3 + 3*a3*x**2 + 2*a2*x + a1
        return Y, Yp
    elif x <= coefs.Xi:
        # Midbody segment
        u = x - coefs.xm
        Y = b5*u**5 + b4*u**4 + b3*u**3 + b2*u**2 + b1*u + b0
        Yp = 5*b5*u**4 + 4*b4*u**3 + 3*b3*u**2 + 2*b2*u + b1
        return Y, Yp
    else:
        # Tail segment
        v = x - coefs.Xi
        Y = c5*v**5 + c4*v**4 + c3*v**3 + c2*v**2 + c1*v + c0
        Yp = 5*c5*v**4 + 4*c4*v**3 + 3*c3*v**2 + 2*c2*v + c1
        return Y, Yp

def sample_profile(params: Params, n: int = 400) -> Tuple[list, list, list]:
    """
    Return arrays X, Y, Yp sampled across [0,1] using the current coefficients.
    
    Args:
        params: Hull parameters
        n: Number of sample points
        
    Returns:
        Tuple of (X, Y, Yp) arrays where X is position, Y is radius, Yp is slope
    """
    C = build_coefs(params)
    X = [i/(n-1) for i in range(n)]
    Y, Yp = [], []
    for x in X:
        y, yp = evaluate_shape(x, C)
        Y.append(y)
        Yp.append(yp)
    return X, Y, Yp
