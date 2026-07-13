#!/usr/bin/env python3
"""
Headless optimization driver for airship hull shape optimization.

This script runs the optimization and saves results to JSON for later analysis.
"""

import argparse
import json
import sys
import os
import logging
import datetime as _dt
from pathlib import Path

# Add the parent directory to the path so we can import airship_opt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from airship_opt.aero import (
    CONSTANT_EDGE_SURROGATE_V1,
    surrogate_static_boundary,
)
from airship_opt.types import Params, Config
from airship_opt.optimize import nelder_mead
from airship_opt.io_viz import save_result, prepare_run_dirs, update_meta, write_manifest

def _get_medium(args):
    """Create medium from command-line arguments."""
    from airship_opt.media.air import make_air
    from airship_opt.media.water import make_water
    from airship_opt.media.custom import make_custom

    if args.medium == "air":
        return make_air(rho=args.rho, nu=args.nu) if args.rho or args.nu else make_air()
    if args.medium == "water":
        return make_water(rho=args.rho, nu=args.nu) if args.rho or args.nu else make_water()
    # Custom medium path
    rho = args.rho if args.rho is not None else 1.0
    nu = args.nu if args.nu is not None else 1.0e-6
    return make_custom(rho=rho, nu=nu)

def main():
    ap = argparse.ArgumentParser(description="Run airship hull optimization")
    ap.add_argument("--results-root", default="results", help="Root directory for results")
    ap.add_argument("--run-name", default="x35", help="Name of this optimization run")
    ap.add_argument("--tag", help="Optional tag/label for this run")
    ap.add_argument(
        "--vol-target",
        type=float,
        required=True,
        help="Target nondimensional volume V_bar = V/L^3",
    )
    ap.add_argument("--max-iter", type=int, default=400, help="Maximum iterations")
    ap.add_argument("--w-volume", type=float, default=1.0, help="Volume penalty weight")
    ap.add_argument("--w-shape", type=float, default=1.0, help="Shape penalty weight")
    ap.add_argument("--seed", type=int, help="Random seed for reproducibility")
    
    # Medium selection arguments
    ap.add_argument("--medium", choices=["air", "water", "custom"], default="water",
                   help="Working fluid (default: water, matching Parsons' regime)")
    ap.add_argument("--rho", type=float, help="Fluid density [kg/m^3] (overrides default)")
    ap.add_argument("--nu", type=float, help="Kinematic viscosity [m^2/s] (overrides default)")
    flow = ap.add_mutually_exclusive_group(required=True)
    flow.add_argument("--U", type=float, help="Free-stream speed [m/s] (requires --reference-length-m)")
    flow.add_argument("--ReV", type=float, help="Direct volume-based Reynolds number Re_V")
    ap.add_argument(
        "--reference-length-m",
        type=float,
        help="Physical reference length L [m], required when deriving Re_V from --U",
    )
    ap.add_argument(
        "--drag-model",
        required=True,
        choices=[CONSTANT_EDGE_SURROGATE_V1],
        help="Explicit screening model; no validated Parsons default exists",
    )
    ap.add_argument(
        "--allow-screening-optimization",
        action="store_true",
        help=(
            "Explicitly allow same-model, same-flow relative surrogate ranking. "
            "The saved coefficient remains machine-flagged as NOT hull evidence."
        ),
    )
    args = ap.parse_args()
    if args.U is not None and args.reference_length_m is None:
        ap.error("--U requires --reference-length-m so normalized volume can be dimensionalized")
    if args.ReV is not None and args.reference_length_m is not None:
        ap.error("--reference-length-m is unnecessary with direct --ReV; choose one flow route")
    if not args.allow_screening_optimization:
        ap.error(
            "--allow-screening-optimization is required: the surrogate may rank "
            "only same-model, same-flow software candidates and is not hull evidence"
        )

    # Set up results directory structure
    results_root = os.getenv("RESULTS_ROOT", args.results_root)
    run_dirs = prepare_run_dirs(results_root, args.run_name)

    # Configure logging with both file and console handlers
    file_handler = logging.FileHandler(run_dirs["logs"] / "run.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler])

    # Log run configuration
    logging.info(f"Run directory: {run_dirs['base'].absolute()}")
    logging.info(f"Starting optimization run: {args.run_name}")
    logging.warning(
        "Using %s: a dimensionally closed screening surrogate, not a "
        "source-faithful Parsons solver or vehicle-validation model.",
        args.drag_model,
    )
    
    # Set random seed if provided
    if args.seed is not None:
        import random
        random.seed(args.seed)
        logging.info(f"Using random seed: {args.seed}")

    # Example: seed from your "original Parsons" or a user config
    p0 = Params(rn=0.7573, fr=4.8488, xm=0.5888, k=0.1711,
                Xi=0.7853, n=0.6473, S=2.2867, t=0.1731)

    # Create medium from arguments
    medium = _get_medium(args)

    # Update config with medium-specific settings 
    cfg = Config(
        vol_target=args.vol_target,
        medium=medium,  # Pass the medium object
        w_volume=args.w_volume,
        w_shape=args.w_shape,
        speed_U=args.U,
        Re_V=args.ReV,
        reference_length_m=args.reference_length_m,
        drag_model=args.drag_model,
        allow_screening_optimization=args.allow_screening_optimization,
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

    logging.info("Starting optimization")
    logging.info(f"Volume target: {args.vol_target}")
    logging.info(f"Initial parameters: {list(p0.__dict__.values())}")
    logging.info(f"Bounds: {bounds}")

    # Update run configuration in meta
    config = {
        "cli": {
            "args": vars(args),
            "bounds": bounds,
            "initial_params": p0.__dict__,
            "tag": args.tag
        },
        "medium": {
            "name": medium.name,
            "rho": medium.rho,
            "nu": medium.nu
        },
        "speed": {
            "U": args.U,
            "reference_length_m": args.reference_length_m,
        },
        "reynolds": {
            "ReV": args.ReV
        },
        "drag_model": {
            "id": args.drag_model,
            "status": "screening-surrogate-open",
        },
        "coefficient_claim": {
            **surrogate_static_boundary(),
            "screening_optimization_acknowledged": True,
        },
    }
    update_meta(run_dirs["base"], config)
    
    # Run optimization
    res = nelder_mead(list(p0.__dict__.values()), bounds, cfg, max_iter=args.max_iter)

    # Compute ReV if needed
    from airship_opt.objectives.utils import resolve_reynolds
    ReV_value, ReL_value = resolve_reynolds(
        U=args.U,
        V_bar=res.volume,
        nu=medium.nu,
        ReV=args.ReV,
        reference_length_m=args.reference_length_m,
    )

    # Update meta with computed ReV
    meta = {
        "medium": {
            "name": medium.name,
            "rho": medium.rho,
            "nu": medium.nu
        },
        "speed": {
            "U": args.U if getattr(args, "U", None) is not None else None,
            "reference_length_m": args.reference_length_m,
        },
        "reynolds": {
            "ReV": float(ReV_value),
            "ReL": float(ReL_value),
        },
        "drag_model": {
            "id": args.drag_model,
            "status": "screening-surrogate-open",
        },
        "coefficient_claim": res.meta["coefficient_claim"],
    }
    update_meta(run_dirs["base"], meta)
    
    # Save results and log completion
    result_path = save_result(res, run_dirs["artifacts"])
    
    # Update final meta information
    end_time = _dt.datetime.utcnow()
    elapsed = (end_time - _dt.datetime.fromisoformat(
        run_dirs["meta"].joinpath("config.json").read_text().split('"start_utc": "')[1].split('"')[0]
    )).total_seconds()
    
    final_meta = {
        "end_utc": end_time.isoformat(),
        "elapsed_sec": elapsed
    }
    update_meta(run_dirs["base"], final_meta)
    
    # Update manifest with final files
    write_manifest(run_dirs["base"])
    
    logging.info(f"Optimization complete")
    logging.info(
        "screening_coefficient=%0.6f | hull_evidence_eligible=false | V=%0.6f | J=%0.6e",
        res.cd,
        res.volume,
        res.objective,
    )
    logging.info(f"Iterations: {res.meta.get('iterations', 'unknown')}")
    logging.info(f"Elapsed time: {elapsed:.2f} seconds")
    
    # Print minimal console output
    print(f"Optimization complete. Outputs saved under: {run_dirs['base']}")
    print(f"Results file: {result_path}")

if __name__ == "__main__":
    main()
