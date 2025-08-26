#!/usr/bin/env python3
"""
Visualization script for airship hull optimization results.

This script reads JSON result files and generates publication-quality plots
for radius, slope, and convergence analysis.
"""

import argparse
import math
import json
import sys
import os

# Add the parent directory to the path so we can import airship_opt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for headless operation
except ImportError:
    print("Warning: matplotlib not available. Install with: pip install matplotlib")
    sys.exit(1)

from airship_opt.types import Params
from airship_opt.geometry import build_coefs, evaluate_shape

def plot_radius_and_slope(params: Params, out_prefix: str, n: int = 600):
    """
    Generate radius and slope plots for the hull shape.
    
    Args:
        params: Hull parameters
        out_prefix: Output file prefix
        n: Number of sample points
    """
    C = build_coefs(params)
    X = [i/(n-1) for i in range(n)]
    Y, Yp = [], []
    for x in X:
        y, yp = evaluate_shape(x, C)
        Y.append(y)
        Yp.append(yp)

    # Radius plot
    plt.figure(figsize=(10, 6))
    plt.plot(X, Y, 'b-', linewidth=2, label="Radius Y(x)")
    plt.axvline(C.xm, linestyle="--", color='red', alpha=0.7, label=f"xm = {C.xm:.3f}")
    plt.axvline(C.Xi, linestyle="--", color='green', alpha=0.7, label=f"Xi = {C.Xi:.3f}")
    plt.xlabel("x (L=1)", fontsize=12)
    plt.ylabel("radius Y", fontsize=12)
    plt.title("Airship Hull Shape - Radius Profile", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_radius.png", dpi=200, bbox_inches='tight')
    plt.close()

    # Slope plot
    plt.figure(figsize=(10, 6))
    plt.plot(X, Yp, 'g-', linewidth=2, label="Slope Y'(x)")
    plt.axvline(C.xm, linestyle="--", color='red', alpha=0.7, label=f"xm = {C.xm:.3f}")
    plt.axvline(C.Xi, linestyle="--", color='green', alpha=0.7, label=f"Xi = {C.Xi:.3f}")
    plt.axhline(0, linestyle="-", color='black', alpha=0.3)
    plt.xlabel("x (L=1)", fontsize=12)
    plt.ylabel("slope dY/dx", fontsize=12)
    plt.title("Airship Hull Shape - Slope Profile", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_slope.png", dpi=200, bbox_inches='tight')
    plt.close()

def plot_convergence(history, vol_target: float, out_prefix: str):
    """
    Generate convergence plots for objective and volume error.
    
    Args:
        history: Optimization history from result
        vol_target: Target volume
        out_prefix: Output file prefix
    """
    it = [h["iter"] for h in history]
    obj = [h["obj"] for h in history]
    verr = [(h["volume"] - vol_target) / max(vol_target, 1e-12) for h in history]

    # Objective convergence
    plt.figure(figsize=(10, 6))
    plt.plot(it, obj, 'b-', linewidth=2, label="Objective")
    plt.xlabel("iteration", fontsize=12)
    plt.ylabel("objective", fontsize=12)
    plt.title("Optimization Convergence - Objective Function", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_objective.png", dpi=200, bbox_inches='tight')
    plt.close()

    # Volume error convergence
    plt.figure(figsize=(10, 6))
    plt.plot(it, verr, 'r-', linewidth=2, label="Volume error (fraction)")
    plt.axhline(0, linestyle="-", color='black', alpha=0.3)
    plt.xlabel("iteration", fontsize=12)
    plt.ylabel("volume error", fontsize=12)
    plt.title("Optimization Convergence - Volume Error", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_volerr.png", dpi=200, bbox_inches='tight')
    plt.close()

def create_summary_table(data: dict, out_prefix: str):
    """
    Create a summary table of the optimization results.
    
    Args:
        data: Loaded result data
        out_prefix: Output file prefix
    """
    params = data["params"]
    summary = f"""
Airship Hull Optimization Results
================================

Parameters:
  rn (nose curvature): {params['rn']:.4f}
  fr (fineness ratio): {params['fr']:.4f}
  xm (max diameter):   {params['xm']:.4f}
  k (curvature):       {params['k']:.4f}
  Xi (inflection):     {params['Xi']:.4f}
  n (radius ratio):    {params['n']:.4f}
  S (slope):           {params['S']:.4f}
  t (tail radius):     {params['t']:.4f}

Results:
  Drag Coefficient:    {data['cd']:.6f}
  Volume:              {data['volume']:.6f}
  Objective:           {data['objective']:.6e}
  Iterations:          {data['meta'].get('iterations', 'unknown')}

Geometry:
  Max diameter at:     {params['xm']:.3f}L
  Inflection at:       {params['Xi']:.3f}L
  Tail radius:         {params['t']:.3f}D/2
"""
    
    with open(f"{out_prefix}_summary.txt", "w") as f:
        f.write(summary)
    
    print(summary)

def main():
    ap = argparse.ArgumentParser(description="Generate plots from optimization results")
    ap.add_argument("--result", default="result.json", help="Input JSON result file")
    ap.add_argument("--out-prefix", default="fig", help="Output file prefix")
    ap.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = ap.parse_args()

    if not os.path.exists(args.result):
        print(f"Error: Result file '{args.result}' not found")
        sys.exit(1)

    with open(args.result, "r") as f:
        data = json.load(f)

    p = data["params"]
    params = Params(**p)
    
    # Create summary table
    create_summary_table(data, args.out_prefix)
    
    if not args.no_plots:
        # Generate plots
        plot_radius_and_slope(params, args.out_prefix)
        plot_convergence(data["history"], data.get("meta", {}).get("vol_target", data["volume"]), args.out_prefix)
        
        print(f"Generated plots:")
        print(f"  {args.out_prefix}_radius.png")
        print(f"  {args.out_prefix}_slope.png") 
        print(f"  {args.out_prefix}_objective.png")
        print(f"  {args.out_prefix}_volerr.png")
        print(f"  {args.out_prefix}_summary.txt")

if __name__ == "__main__":
    main()
