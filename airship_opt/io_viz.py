"""
I/O and visualization module for airship hull optimization.

This module provides functions for saving and loading optimization results
in JSON format for later analysis and visualization.
"""

from __future__ import annotations
import json
from dataclasses import asdict
from typing import Any, Dict
from .types import Result, Params, Coefs, IterRecord

def save_result(res: Result, path: str) -> None:
    """
    Save optimization result to JSON file.
    
    Args:
        res: Result object to save
        path: File path for saving
    """
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
        "meta": res.meta,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

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
