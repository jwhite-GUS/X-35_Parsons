"""
Optimization module for airship hull shape optimization.

This module implements the Nelder-Mead simplex optimization algorithm
for finding optimal hull parameters.
"""

from __future__ import annotations
from typing import Callable, Tuple, List
from copy import deepcopy
from .types import Params, Config, Result, IterRecord
from .geometry import build_coefs
from .objective import objective

def nelder_mead(
    x0: List[float],
    bounds: List[Tuple[float, float]],
    cfg: Config,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> Result:
    """
    Nelder-Mead simplex optimization driver.
    
    Args:
        x0: Initial parameter vector [rn, fr, xm, k, Xi, n, S, t]
        bounds: Parameter bounds as list of (min, max) tuples
        cfg: Configuration object
        max_iter: Maximum iterations
        tol: Convergence tolerance
        
    Returns:
        Result object with optimized parameters and metadata
    """
    # Simplex init (regular)
    n = len(x0)
    simplex = [x0[:]]
    for i in range(n):
        step = 0.05 * (bounds[i][1] - bounds[i][0])
        xi = x0[:]
        xi[i] = min(max(bounds[i][0], xi[i] + step), bounds[i][1])
        simplex.append(xi)

    history: List[IterRecord] = []
    
    def as_params(x: List[float]) -> Params:
        return Params(*x)

    def eval_obj(x: List[float]) -> Tuple[float, float, float]:
        J, meta = objective(as_params(x), cfg)
        return J, float(meta['CD']), float(meta['V'])

    # Score initial simplex
    scores = [eval_obj(x) for x in simplex]

    it = 0
    while it < max_iter:
        it += 1
        # Order simplex by objective
        paired = sorted(zip(simplex, scores), key=lambda z: z[1][0])
        simplex, scores = [list(p[0]) for p in paired], [p[1] for p in paired]
        bestJ, bestCD, bestV = scores[0]
        vol_err = (bestV - cfg.vol_target) / max(cfg.vol_target, 1e-12)
        history.append(IterRecord(it, simplex[0][:], bestJ, bestCD, bestV, vol_err))

        # Convergence check
        if it > 5:
            if abs(history[-1].obj - history[-5].obj) < tol:
                break

        # Centroid (except worst)
        xbar = [0.0]*n
        for x in simplex[:-1]:
            for i in range(n): 
                xbar[i] += x[i]
        xbar = [v/float(n) for v in xbar]

        # Nelder-Mead parameters
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        
        def project(x):
            return [min(max(v, bounds[i][0]), bounds[i][1]) for i, v in enumerate(x)]

        # Reflect
        xr = project([xbar[i] + alpha*(xbar[i] - simplex[-1][i]) for i in range(n)])
        Jr, CDr, Vr = eval_obj(xr)

        if Jr < scores[0][0]:
            # Expand
            xe = project([xbar[i] + gamma*(xr[i] - xbar[i]) for i in range(n)])
            Je, CDe, Ve = eval_obj(xe)
            if Je < Jr:
                simplex[-1], scores[-1] = xe, (Je, CDe, Ve)
            else:
                simplex[-1], scores[-1] = xr, (Jr, CDr, Vr)
        elif Jr < scores[-2][0]:
            simplex[-1], scores[-1] = xr, (Jr, CDr, Vr)
        else:
            # Contract
            xc = project([xbar[i] + rho*(simplex[-1][i] - xbar[i]) for i in range(n)])
            Jc, CDc, Vc = eval_obj(xc)
            if Jc < scores[-1][0]:
                simplex[-1], scores[-1] = xc, (Jc, CDc, Vc)
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = project([simplex[0][j] + sigma*(simplex[i][j] - simplex[0][j]) for j in range(n)])
                    scores[i] = eval_obj(simplex[i])

    x_best = simplex[0]
    J_best, CD_best, V_best = scores[0]
    res = Result(
        params=Params(*x_best),
        coefs=build_coefs(Params(*x_best)),
        cd=CD_best,
        volume=V_best,
        objective=J_best,
        history=history,
        meta={"iterations": it},
    )
    return res
