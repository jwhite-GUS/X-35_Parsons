#!/usr/bin/env python3
"""
Headless optimization driver for airship hull shape optimization.

This script runs the optimization and saves results to JSON for later analysis.
"""

import argparse
import json
import sys
import os

# Add the parent directory to the path so we can import airship_opt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from airship_opt.types import Params, Config
from airship_opt.optimize import nelder_mead
from airship_opt.io_viz import save_result

def main():
    ap = argparse.ArgumentParser(description="Run airship hull optimization")
    ap.add_argument("--out", default="result.json", help="Output JSON file")
    ap.add_argument("--vol-target", type=float, required=True, help="Target volume")
    ap.add_argument("--max-iter", type=int, default=400, help="Maximum iterations")
    ap.add_argument("--w-volume", type=float, default=1.0, help="Volume penalty weight")
    ap.add_argument("--w-shape", type=float, default=1.0, help="Shape penalty weight")
    ap.add_argument("--seed", type=int, help="Random seed for reproducibility")
    args = ap.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        import random
        random.seed(args.seed)

    # Example: seed from your "original Parsons" or a user config
    p0 = Params(rn=0.7573, fr=4.8488, xm=0.5888, k=0.1711,
                Xi=0.7853, n=0.6473, S=2.2867, t=0.1731)

    cfg = Config(
        vol_target=args.vol_target, 
        w_volume=args.w_volume, 
        w_shape=args.w_shape
    )

    bounds = [
        (0.2,  1.5),   # rn
        (2.5, 15.0),   # fr
        (0.05, 0.95),  # xm
        (0.02, 0.35),  # k
        (0.10, 0.995), # Xi
        (0.30, 1.10),  # n
        (0.50, 3.50),  # S
        (0.00, 0.40),  # t
    ]

    print(f"Starting optimization with volume target: {args.vol_target}")
    print(f"Initial parameters: {list(p0.__dict__.values())}")
    
    res = nelder_mead(list(p0.__dict__.values()), bounds, cfg, max_iter=args.max_iter)
    
    save_result(res, args.out)
    print(f"Saved: {args.out}")
    print(f"CD={res.cd:.6f} | V={res.volume:.6f} | J={res.objective:.6e}")
    print(f"Iterations: {res.meta.get('iterations', 'unknown')}")

if __name__ == "__main__":
    main()
