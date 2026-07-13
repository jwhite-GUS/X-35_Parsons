"""
Test suite for report builder functionality.

Tests report generation functions with minimal dependencies.
"""

from pathlib import Path
import json
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

    html = (run_dir / "reports" / "summary.html").read_text()
    markdown = (run_dir / "reports" / "summary.md").read_text()
    assert "WITHHELD — missing explicit non-evidence flag" in html
    assert "WITHHELD — missing explicit non-evidence flag" in markdown
    assert "Screening coefficient (NOT hull evidence)</th><td>0.1" not in html


def test_report_exposes_screening_value_only_beside_false_evidence_flag(tmp_path: Path):
    run_dir = tmp_path / "flagged_run"
    (run_dir / "artifacts").mkdir(parents=True)
    (run_dir / "meta").mkdir()
    (run_dir / "figures").mkdir()
    claim = {
        "hull_evidence_eligible": False,
        "physical_magnitude_validated": False,
        "validity_envelope": {
            "assessment": "computational-envelope-pass",
            "Re_L": 1.0e7,
            "Re_L_min": 1.0e5,
            "Re_L_max": 1.0e9,
        },
        "closed_form_benchmark": {
            "passed": True,
            "relative_error": 0.001,
        },
    }
    result = {
        "params": {name: 0.2 for name in ("rn", "fr", "xm", "k", "Xi", "n", "S", "t")},
        "screening_coefficient": 0.0123,
        "cd": 0.0123,
        "coefficient_claim": claim,
        "volume": 0.02,
        "objective": 0.1,
    }
    (run_dir / "artifacts" / "result.json").write_text(json.dumps(result))
    (run_dir / "meta" / "config.json").write_text(
        json.dumps({"medium": {"name": "test", "rho": 1.0, "nu": 1.0e-6}})
    )

    write_reports(run_dir, make_pdf=False)

    html = (run_dir / "reports" / "summary.html").read_text()
    markdown = (run_dir / "reports" / "summary.md").read_text()
    assert "HULL EVIDENCE ELIGIBLE: false" in html
    assert "Screening coefficient (NOT hull evidence)</th><td>0.0123" in html
    assert "Flat-plate arithmetic benchmark passed</th><td>True" in html
    assert "**NOT HULL EVIDENCE.**" in markdown
    assert "- rn: 0.2" in markdown
