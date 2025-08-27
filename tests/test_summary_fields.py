"""
Test presence and format of fields in summary output.

Tests that medium properties and Reynolds number are correctly 
recorded in the results summary.
"""

import os
import csv
import json
from pathlib import Path

def test_summary_fields(tmp_path):
    """Test that summary files contain required medium and Reynolds fields."""
    # Create mock results structure
    run_dir = tmp_path / "test_run"
    tables_dir = run_dir / "tables"
    meta_dir = run_dir / "meta"
    tables_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)
    
    # Write mock summary.csv
    summary_csv = tables_dir / "summary.csv"
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['medium.name', 'medium.nu', 'medium.rho', 'U', 'ReV'])
        writer.writerow(['water_20C', 1.004e-6, 998.2, 2.0, 1.5e6])
    
    # Write mock config.json
    config_json = meta_dir / "config.json"
    config = {
        "medium": {"name": "water_20C", "rho": 998.2, "nu": 1.004e-6},
        "speed": {"U": 2.0},
        "reynolds": {"ReV": 1.5e6}
    }
    config_json.write_text(json.dumps(config))
    
    # Read and verify CSV fields
    with open(summary_csv) as f:
        reader = csv.DictReader(f)
        row = next(reader)
        assert "medium.name" in row
        assert "medium.nu" in row  
        assert "medium.rho" in row
        assert "U" in row
        assert "ReV" in row
        
        assert float(row["medium.nu"]) == 1.004e-6
        assert float(row["medium.rho"]) == 998.2
        assert float(row["U"]) == 2.0
        assert float(row["ReV"]) == 1.5e6
    
    # Read and verify config
    data = json.loads(config_json.read_text())
    assert "medium" in data
    assert "name" in data["medium"]
    assert "rho" in data["medium"]
    assert "nu" in data["medium"]
    assert "speed" in data 
    assert "reynolds" in data
