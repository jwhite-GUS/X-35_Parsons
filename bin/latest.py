#!/usr/bin/env python3
"""
latest.py — resolve latest results run folder or file paths.

Usage examples:
  # Print absolute path to latest run folder for slug 'x35'
  python bin/latest.py --run-name x35

  # Print absolute path to artifacts/result.json (default mode)
  python bin/latest.py --run-name x35 --artifacts

  # Print figures or tables folder
  python bin/latest.py --run-name x35 --figures
  python bin/latest.py --run-name x35 --tables

  # Print JSON with all canonical paths
  python bin/latest.py --run-name x35 --json
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import sys

RESULTS_ROOT = Path(os.getenv("RESULTS_ROOT") or "results").resolve()
TS_DIR_RE = re.compile(r"^\d{8}-\d{6}(?:[_-]\d+)?__")  # matches our UTC stamp prefix

def _is_run_dir(p: Path) -> bool:
    return (p / "artifacts" / "result.json").exists()

def _resolve_pointer(slug: str) -> Path | None:
    """
    Resolve latest pointer. Returns:
      - Path to run folder, or
      - Path to artifacts/result.json
    or None if not found.
    """
    ptr_dir = RESULTS_ROOT / f"latest__{slug}"
    ptr_txt = RESULTS_ROOT / f"latest__{slug}.txt"

    # Symlink or directory pointer (preferred on Unix)
    if ptr_dir.exists():
        try:
            target = ptr_dir.resolve(strict=True)
        except Exception:
            target = ptr_dir  # non-symlink dir; treat as folder
        if target.is_dir():
            if _is_run_dir(target):
                return target
            # If someone pointed directly at artifacts/result.json's parent incorrectly,
            # fall through to handle as file.
        elif target.is_file():
            # odd case: directory pointer points at file
            return target

    # Windows fallback: text file containing absolute path
    if ptr_txt.exists() and ptr_txt.is_file():
        try:
            target_str = ptr_txt.read_text(encoding="utf-8").strip()
        except Exception:
            target_str = ""
        if not target_str:
            return None
        tpath = Path(target_str)
        if tpath.is_dir():
            return tpath
        if tpath.is_file():
            return tpath

    return None

def _discover_latest_run(slug: str) -> Path | None:
    """
    If no pointer exists, scan results root for timestamped dirs matching
    '*__{slug}' and choose the newest by directory name sort (UTC prefix).
    """
    if not RESULTS_ROOT.exists():
        return None
    candidates = []
    try:
        for d in RESULTS_ROOT.iterdir():
            if d.is_dir() and d.name.endswith(f"__{slug}") and TS_DIR_RE.match(d.name):
                candidates.append(d)
    except FileNotFoundError:
        return None
    if not candidates:
        return None
    # Directory names are UTC stamped; lexicographic sort picks newest last
    candidates.sort()
    return candidates[-1]

def resolve_latest(slug: str) -> dict[str, str] | None:
    """
    Resolve to canonical locations. Returns dict with:
      base, artifacts_json, figures_dir, tables_dir, logs_dir, meta_dir, manifest
    or None if resolution fails.
    """
    target = _resolve_pointer(slug)
    if target is None:
        target = _discover_latest_run(slug)
        if target is None:
            return None

    # If pointer/file points to result.json, normalize to base run directory
    if target.is_file():
        if target.name == "result.json" and target.parent.name == "artifacts":
            base = target.parent.parent
        else:
            # unexpected file; treat parent as base if structure matches
            base = target.parent.parent if target.parent.name == "artifacts" else target.parent
    else:
        base = target

    base = base.resolve()
    artifacts_json = (base / "artifacts" / "result.json").resolve()
    figures_dir    = (base / "figures").resolve()
    tables_dir     = (base / "tables").resolve()
    logs_dir       = (base / "logs").resolve()
    meta_dir       = (base / "meta").resolve()
    manifest       = (base / "manifest.json").resolve()

    return {
        "base": str(base),
        "artifacts_json": str(artifacts_json),
        "figures_dir": str(figures_dir),
        "tables_dir": str(tables_dir),
        "logs_dir": str(logs_dir),
        "meta_dir": str(meta_dir),
        "manifest": str(manifest),
    }

def main():
    ap = argparse.ArgumentParser(description="Resolve latest results run paths for a slug.")
    ap.add_argument("--run-name", required=True, help="Slug (run name), e.g., 'x35'")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--artifacts", action="store_true", help="Print path to artifacts/result.json")
    group.add_argument("--figures", action="store_true", help="Print path to figures directory")
    group.add_argument("--tables", action="store_true", help="Print path to tables directory")
    group.add_argument("--logs", action="store_true", help="Print path to logs directory")
    group.add_argument("--meta", action="store_true", help="Print path to meta directory")
    group.add_argument("--manifest", action="store_true", help="Print path to manifest.json")
    ap.add_argument("--json", action="store_true", help="Print JSON with all canonical paths")
    args = ap.parse_args()

    info = resolve_latest(args.run_name)
    if info is None:
        print(f"ERROR: No latest run found for slug '{args.run_name}' under {RESULTS_ROOT}", file=sys.stderr)
        sys.exit(2)

    if args.json:
        print(json.dumps(info, indent=2))
        return

    if args.artifacts:
        print(info["artifacts_json"])
    elif args.figures:
        print(info["figures_dir"])
    elif args.tables:
        print(info["tables_dir"])
    elif args.logs:
        print(info["logs_dir"])
    elif args.meta:
        print(info["meta_dir"])
    elif args.manifest:
        print(info["manifest"])
    else:
        # default: print base run directory
        print(info["base"])

if __name__ == "__main__":
    main()
