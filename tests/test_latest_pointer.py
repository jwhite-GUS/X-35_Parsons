#!/usr/bin/env python3
"""
Test latest pointer resolution functionality.
"""

import tempfile
import json
import shutil
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bin.plot_results import _resolve_result_path

def test_resolve_result_path():
    """Test _resolve_result_path with various input formats."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a fake run structure
        run_dir = temp_path / "20250826-120000_123__test_run"
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a fake result.json
        result_json = artifacts_dir / "result.json"
        with open(result_json, 'w') as f:
            json.dump({"test": "data"}, f)
        
        # Test 1: Direct JSON file
        resolved = _resolve_result_path(result_json)
        assert resolved == result_json, f"Expected {result_json}, got {resolved}"
        
        # Test 2: Run directory
        resolved = _resolve_result_path(run_dir)
        assert resolved == result_json, f"Expected {result_json}, got {resolved}"
        
        # Test 3: Pointer file pointing to directory
        pointer_file = temp_path / "latest__test_run.txt"
        with open(pointer_file, 'w') as f:
            f.write(str(run_dir.absolute()))
        
        resolved = _resolve_result_path(pointer_file)
        assert resolved == result_json, f"Expected {result_json}, got {resolved}"
        
        # Test 4: Pointer file pointing to JSON
        pointer_file2 = temp_path / "latest__test_run2.txt"
        with open(pointer_file2, 'w') as f:
            f.write(str(result_json.absolute()))
        
        resolved = _resolve_result_path(pointer_file2)
        assert resolved == result_json, f"Expected {result_json}, got {resolved}"
        
        print("✅ All latest pointer resolution tests passed!")

if __name__ == "__main__":
    test_resolve_result_path()
