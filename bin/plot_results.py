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
import warnings
import csv
from pathlib import Path

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

def _resolve_result_path(path: Path) -> Path:
    """
    Resolve various input formats to the actual result.json file path.
    
    Args:
        path: Input path (could be directory, pointer file, or direct JSON)
        
    Returns:
        Resolved Path to result.json file
    """
    path = Path(path)
    
    # Case 1: Direct JSON file
    if path.suffix == '.json':
        return path
    
    # Case 2: Pointer file (.txt)
    if path.suffix == '.txt':
        try:
            content = path.read_text().strip()
            content_path = Path(content)
            
            # If content is a directory, append artifacts/result.json
            if content_path.is_dir():
                return content_path / 'artifacts' / 'result.json'
            # If content is already a JSON path, use it
            elif content_path.suffix == '.json':
                return content_path
            else:
                raise ValueError(f"Invalid content in pointer file: {content}")
        except Exception as e:
            raise ValueError(f"Error reading pointer file {path}: {e}")
    
    # Case 3: Directory (assume run directory)
    if path.is_dir():
        return path / 'artifacts' / 'result.json'
    
    # Case 4: Legacy file or other
    return path

def resolve_output_dirs(result_path: Path) -> tuple[Path, Path]:
    """
    Resolve output directories for plots and tables based on result path.
    
    Args:
        result_path: Path to result.json file
        
    Returns:
        Tuple of (figures_dir, tables_dir)
    """
    # Handle legacy case (root-level files)
    if result_path.parent == Path('.'):
        warnings.warn(
            "Using root-level result file. Consider using results/*/artifacts/result.json "
            "for better organization.",
            UserWarning
        )
        return Path('.'), Path('.')
    
    # Handle new directory structure
    if 'artifacts' in str(result_path):
        run_dir = result_path.parent.parent
        figures_dir = run_dir / 'figures'
        tables_dir = run_dir / 'tables'
        figures_dir.mkdir(exist_ok=True)
        tables_dir.mkdir(exist_ok=True)
        return figures_dir, tables_dir
    
    # Default case
    return Path('.'), Path('.')

def plot_radius_and_slope(params: Params, figures_dir: Path, prefix: str = "fig", n: int = 600):
    """
    Generate radius and slope plots for the hull shape.
    
    Args:
        params: Hull parameters
        figures_dir: Directory to save figure files
        prefix: Output file prefix
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
    plt.savefig(figures_dir / f"{prefix}_radius.png", dpi=200, bbox_inches='tight')
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
    plt.savefig(figures_dir / f"{prefix}_slope.png", dpi=200, bbox_inches='tight')
    plt.close()

def plot_convergence(history, vol_target: float, figures_dir: Path, prefix: str = "fig"):
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
    plt.savefig(figures_dir / f"{prefix}_objective.png", dpi=200, bbox_inches='tight')
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
    plt.savefig(figures_dir / f"{prefix}_volerr.png", dpi=200, bbox_inches='tight')
    plt.close()

def create_summary_table(data: dict, tables_dir: Path):
    """
    Create a summary table of the optimization results.
    
    Args:
        data: Loaded result data
        tables_dir: Directory to save the summary file
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
    
    with open(tables_dir / "summary.txt", "w") as f:
        f.write(summary)
    
    # Create CSV summary for machine-readable format
    csv_data = [
        ["Parameter", "Value", "Units"],
        ["cd_te", f"{data['cd']:.6f}", "dimensionless"],
        ["volume", f"{data['volume']:.6f}", "L³"],
        ["objective", f"{data['objective']:.6e}", "dimensionless"],
        ["iterations", str(data['meta'].get('iterations', 'unknown')), "count"],
        ["rn", f"{params['rn']:.4f}", "dimensionless"],
        ["fr", f"{params['fr']:.4f}", "L/D"],
        ["xm", f"{params['xm']:.4f}", "L"],
        ["k", f"{params['k']:.4f}", "dimensionless"],
        ["Xi", f"{params['Xi']:.4f}", "L"],
        ["n", f"{params['n']:.4f}", "dimensionless"],
        ["S", f"{params['S']:.4f}", "dimensionless"],
        ["t", f"{params['t']:.4f}", "D/2"]
    ]
    
    with open(tables_dir / "summary.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)
    
    print(summary)

def main():
    ap = argparse.ArgumentParser(description="Generate plots from optimization results")
    ap.add_argument("--result", default="result.json", help="Input JSON result file, run directory, or latest pointer")
    ap.add_argument("--out-dir", help="Output directory (overrides automatic directory detection)")
    ap.add_argument("--out-prefix", default="fig", help="Output file prefix")
    ap.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = ap.parse_args()

    try:
        result_path = _resolve_result_path(args.result)
        print(f"Resolved result path: {result_path}")
    except Exception as e:
        print(f"Error resolving result path: {e}")
        sys.exit(1)
    
    if not result_path.exists():
        print(f"Error: Result file '{result_path}' not found")
        sys.exit(1)

    # Determine output directories
    if args.out_dir:
        figures_dir = Path(args.out_dir)
        tables_dir = figures_dir.parent / 'tables' if 'figures' in str(figures_dir) else figures_dir
        figures_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
    else:
        figures_dir, tables_dir = resolve_output_dirs(result_path)
    
    # Get run directory for manifest refresh
    run_dir = None
    if 'artifacts' in str(result_path):
        run_dir = result_path.parent.parent

    # Load the result file
    with open(result_path, "r") as f:
        data = json.load(f)

    p = data["params"]
    params = Params(**p)
    
    # Create summary table
    create_summary_table(data, tables_dir)
    
    if not args.no_plots:
        # Generate plots
        plot_radius_and_slope(params, figures_dir, args.out_prefix)
        plot_convergence(
            data["history"], 
            data.get("meta", {}).get("vol_target", data["volume"]), 
            figures_dir,
            args.out_prefix
        )
        
        # Refresh manifest if we're in a run directory
        if run_dir is not None:
            from airship_opt.io_viz import write_manifest
            write_manifest(run_dir)
            print(f"Updated manifest: {run_dir}/manifest.json")
        
        print(f"Generated outputs in:")
        print(f"  Figures: {figures_dir}")
        print(f"  Tables:  {tables_dir}")
        print("\nFiles:")
        print(f"  {figures_dir}/{args.out_prefix}_radius.png")
        print(f"  {figures_dir}/{args.out_prefix}_slope.png") 
        print(f"  {figures_dir}/{args.out_prefix}_objective.png")
        print(f"  {figures_dir}/{args.out_prefix}_volerr.png")
        print(f"  {tables_dir}/summary.txt")
        print(f"  {tables_dir}/summary.csv")

if __name__ == "__main__":
    main()
