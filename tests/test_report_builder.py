"""
Test suite for report builder functionality.

Tests report generation functions with minimal dependencies.
"""

from pathlib import Path
from airship_opt.reporting.report_builder import write_reports

def test_write_reports_smoke(tmp_path: Path):
    """
    Smoke test for report generation.
    
    Creates minimal fake structure and verifies HTML/MD files are created.
    """
    # Create minimal structure
    run_dir = tmp_path / "test_run"
    (run_dir / "artifacts").mkdir(parents=True)
    (run_dir / "meta").mkdir()
    (run_dir / "figures").mkdir()
    
    # Create dummy result.json
    with open(run_dir / "artifacts" / "result.json", "w") as f:
        f.write('{"cd": 0.1, "volume": 0.02, "objective": 0.1}')
        
    # Create dummy config.json
    with open(run_dir / "meta" / "config.json", "w") as f:
        f.write('{"medium": {"name": "test", "rho": 1.0, "nu": 1e-6}}')
        
    # Generate reports
    out = write_reports(run_dir, make_pdf=False)
    
    # Verify files were created
    assert (run_dir / "reports" / "summary.html").exists()
    assert (run_dir / "reports" / "summary.md").exists()
    assert out["html"].endswith("summary.html")
    assert out["md"].endswith("summary.md")
    assert out["pdf"] is None  # no PDF without weasyprint
