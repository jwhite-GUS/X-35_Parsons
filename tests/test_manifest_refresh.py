#!/usr/bin/env python3
"""
Test manifest refresh functionality.
"""

import tempfile
import json
import shutil
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from airship_opt.io_viz import write_manifest

def test_manifest_refresh():
    """Test manifest refresh after adding new files."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a fake run structure
        run_dir = temp_path / "20250826-120000_123__test_run"
        artifacts_dir = run_dir / "artifacts"
        figures_dir = run_dir / "figures"
        tables_dir = run_dir / "tables"
        
        for dir_path in [artifacts_dir, figures_dir, tables_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create initial files
        result_json = artifacts_dir / "result.json"
        with open(result_json, 'w') as f:
            json.dump({"test": "data"}, f)
        
        # Write initial manifest
        write_manifest(run_dir)
        
        # Check initial manifest
        manifest_path = run_dir / "manifest.json"
        assert manifest_path.exists(), "Initial manifest should exist"
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Should have result.json (handle Windows path separators)
        file_paths = [f["path"] for f in manifest["files"]]
        print(f"Debug: file_paths = {file_paths}")
        assert any("artifacts" in path and "result.json" in path for path in file_paths), f"result.json should be in manifest, got: {file_paths}"
        
        # Add new files
        test_png = figures_dir / "test.png"
        test_csv = tables_dir / "test.csv"
        
        with open(test_png, 'w') as f:
            f.write("fake png data")
        
        with open(test_csv, 'w') as f:
            f.write("fake csv data")
        
        # Refresh manifest
        write_manifest(run_dir)
        
        # Check updated manifest
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        file_paths = [f["path"] for f in manifest["files"]]
        assert any("artifacts" in path and "result.json" in path for path in file_paths), "result.json should still be in manifest"
        assert any("figures" in path and "test.png" in path for path in file_paths), "test.png should be in manifest"
        assert any("tables" in path and "test.csv" in path for path in file_paths), "test.csv should be in manifest"
        
        # Check that files have non-zero sizes and valid SHA256
        for file_info in manifest["files"]:
            assert file_info["size"] > 0, f"File {file_info['path']} should have non-zero size"
            assert len(file_info["sha256"]) == 64, f"File {file_info['path']} should have valid SHA256"
        
        print("✅ All manifest refresh tests passed!")

if __name__ == "__main__":
    test_manifest_refresh()
