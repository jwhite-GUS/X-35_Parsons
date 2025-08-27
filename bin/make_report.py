#!/usr/bin/env python3
"""
Report generator CLI for airship hull optimization.

This script generates HTML, Markdown and optionally PDF reports from
optimization run results.
"""

from pathlib import Path
import argparse
import json
import sys
import os

# Add the parent directory to the path so we can import airship_opt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from airship_opt.reporting.report_builder import write_reports

def _resolve_run_dir(arg: str) -> Path:
    """
    Resolve various input formats to the run directory.
    
    Args:
        arg: Path to result.json, run directory, or pointer file
        
    Returns:
        Path to the run directory
    """
    p = Path(arg)
    if p.is_file() and p.name == "result.json":
        return p.parent.parent
    if (p / "artifacts" / "result.json").exists():
        return p
    # allow latest__*.txt pointer
    if p.suffix == ".txt" and p.exists():
        target = Path(p.read_text(encoding="utf-8").strip())
        return _resolve_run_dir(target)
    raise SystemExit(f"Could not resolve run directory from: {arg}")

def main():
    ap = argparse.ArgumentParser(description="Generate an embedded summary report for a run")
    ap.add_argument("--result", required=True,
                   help="Path to artifacts/result.json, a run dir, or a latest__*.txt pointer")
    ap.add_argument("--no-pdf", action="store_true",
                   help="Skip PDF generation (HTML/MD only)")
    args = ap.parse_args()

    base_dir = _resolve_run_dir(args.result)
    out = write_reports(base_dir, make_pdf=(not args.no_pdf))
    
    print("Report written:")
    for k,v in out.items():
        if v:
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
