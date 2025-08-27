"""
I/O and visualization module for airship hull optimization.

This module provides functions for saving and loading optimization results
in JSON format for later analysis and visualization.
"""

from __future__ import annotations
import json
import os
import subprocess
import shlex
import datetime as _dt
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Union, Optional, List
from .types import Result, Params, Coefs, IterRecord

def prepare_run_dirs(results_root: str, run_name: str) -> dict:
    """
    Create and prepare directories for a new optimization run.
    
    Args:
        results_root: Root directory for all results
        run_name: Name of this specific run
        
    Returns:
        Dictionary with paths to all subdirectories
    """
    ts = _dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S_%f")[:-3]  # UTC with milliseconds
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_name.strip().lower())
    slug = slug[:40]  # Truncate to 40 chars max
    base = Path(results_root) / f"{ts}__{slug}"
    
    # Handle collisions by appending counter
    counter = 1
    original_base = base
    while base.exists():
        base = Path(str(original_base) + f"-{counter}")
        counter += 1
    sub = {k: base / k for k in ["artifacts", "figures", "logs", "tables", "meta"]}
    for p in sub.values():
        p.mkdir(parents=True, exist_ok=True)
    
    # Capture comprehensive meta
    import sys
    import platform
    
    meta = {
        "timestamp": ts,
        "run_name": slug,
        "start_utc": _dt.datetime.utcnow().isoformat(),
        "git": _git_meta(),
        "env": {
            "RESULTS_ROOT": os.getenv("RESULTS_ROOT"),
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable
        },
        "cli": {},  # Will be filled by run_opt.py
        "packages": _get_package_versions()
    }
    with open(sub["meta"] / "config.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    # Create latest pointer
    latest_path = Path(results_root) / f"latest__{slug}"
    try:
        if latest_path.exists():
            latest_path.unlink()  # Remove existing symlink/file
        latest_path.symlink_to(base.name, target_is_directory=True)
    except (OSError, NotImplementedError):
        # Fallback for Windows or when symlinks aren't allowed
        with open(latest_path.with_suffix('.txt'), 'w') as f:
            f.write(str(base.absolute()))
    
    # Create initial manifest
    write_manifest(base)
    
    return {"base": base, **sub}

def _git_meta() -> Dict[str, Optional[str]]:
    """Get git repository metadata."""
    def _sh(cmd: str) -> Optional[str]:
        try:
            return subprocess.check_output(shlex.split(cmd), stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    return {
        "commit": _sh("git rev-parse HEAD"),
        "branch": _sh("git rev-parse --abbrev-ref HEAD"),
        "dirty": _sh("git status --porcelain")
    }

def write_manifest(base: Path) -> None:
    """Write manifest.json with file information and checksums."""
    manifest = {
        "created_utc": _dt.datetime.utcnow().isoformat(),
        "run_name": base.name.split("__")[-1] if "__" in base.name else "unknown",
        "slug": base.name.split("__")[-1] if "__" in base.name else "unknown",
        "files": []
    }
    
    for file_path in base.rglob("*"):
        if file_path.is_file() and file_path.name != "manifest.json":
            rel_path = file_path.relative_to(base)
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
                    sha256 = hashlib.sha256(content).hexdigest()
                manifest["files"].append({
                    "path": str(rel_path),
                    "size": len(content),
                    "sha256": sha256
                })
            except Exception:
                # Skip files that can't be read
                pass
    
    with open(base / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

def update_meta(base: Path, patch: Dict[str, Any]) -> None:
    """Update meta/config.json with additional information."""
    config_path = base / "meta" / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {}
    
    config.update(patch)
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

def _get_package_versions() -> Dict[str, str]:
    """Get versions of key packages."""
    packages = {}
    try:
        import numpy
        packages["numpy"] = numpy.__version__
    except ImportError:
        pass
    
    try:
        import matplotlib
        packages["matplotlib"] = matplotlib.__version__
    except ImportError:
        pass
    
    try:
        import scipy
        packages["scipy"] = scipy.__version__
    except ImportError:
        pass
    
    return packages

def save_result(res: Result, path: Union[str, Path]) -> Path:
    """
    Save optimization result to JSON file.
    
    Args:
        res: Result object to save
        path: File path or directory for saving
        
    Returns:
        Path to the saved file
    """
    path = Path(path)
    if path.is_dir():
        path = path / "result.json"

    # Load metadata from meta/config.json if it exists
    meta = {}
    meta_path = path.parent.parent / "meta" / "config.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            pass
        
    data: Dict[str, Any] = {
        "params": asdict(res.params),
        "coefs": {
            "fore": list(res.coefs.fore),
            "mid":  list(res.coefs.mid),
            "tail": list(res.coefs.tail),
            "xm":   res.coefs.xm,
            "Xi":   res.coefs.Xi,
        },
        "cd": res.cd,
        "volume": res.volume,
        "objective": res.objective,
        "history": [asdict(h) for h in res.history],
        "meta": {**res.meta, **meta},  # Merge result meta with config meta
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path

def load_result(path: str) -> Dict[str, Any]:
    """
    Load optimization result from JSON file.
    
    Args:
        path: File path to load from
        
    Returns:
        Dictionary containing the loaded data
    """
    with open(path, "r") as f:
        return json.load(f)
