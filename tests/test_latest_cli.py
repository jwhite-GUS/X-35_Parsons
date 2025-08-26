import json
import os
from pathlib import Path
import subprocess
import sys
import shutil

PY = sys.executable

def write_txt(p: Path, s: str):
    p.write_text(s, encoding="utf-8")

def test_latest_pointer_txt(tmp_path: Path, monkeypatch):
    # Arrange a fake results tree
    results = tmp_path / "results"
    results.mkdir()
    run = results / "20250101-000000__x35"
    (run / "artifacts").mkdir(parents=True)
    (run / "figures").mkdir()
    (run / "tables").mkdir()
    (run / "logs").mkdir()
    (run / "meta").mkdir()
    (run / "artifacts" / "result.json").write_text("{}", encoding="utf-8")
    # Windows fallback pointer
    write_txt(results / "latest__x35.txt", str(run))

    # Copy script into tmp tree
    repo_root = tmp_path
    bin_dir = repo_root / "bin"
    bin_dir.mkdir()
    script_src = Path(__file__).parent.parent / "bin" / "latest.py"
    shutil.copy(script_src, bin_dir / "latest.py")

    env = os.environ.copy()
    env["RESULTS_ROOT"] = str(results)

    # CLI: JSON mode
    out = subprocess.check_output([PY, str(bin_dir / "latest.py"), "--run-name", "x35", "--json"], env=env).decode()
    info = json.loads(out)
    assert info["base"].endswith("__x35")
    assert info["artifacts_json"].endswith("artifacts/result.json")

def test_latest_dir_resolution(tmp_path: Path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    # two runs: pick newest
    run1 = results / "20250101-000000__x35"
    run2 = results / "20250101-000100__x35"
    for r in (run1, run2):
        (r / "artifacts").mkdir(parents=True)
        (r / "figures").mkdir()
        (r / "tables").mkdir()
        (r / "logs").mkdir()
        (r / "meta").mkdir()
        (r / "artifacts" / "result.json").write_text("{}", encoding="utf-8")

    repo_root = tmp_path
    bin_dir = repo_root / "bin"
    bin_dir.mkdir()
    script_src = Path(__file__).parent.parent / "bin" / "latest.py"
    shutil.copy(script_src, bin_dir / "latest.py")

    env = os.environ.copy()
    env["RESULTS_ROOT"] = str(results)

    # Default prints base dir
    out = subprocess.check_output([sys.executable, str(bin_dir / "latest.py"), "--run-name", "x35"], env=env).decode().strip()
    assert out.endswith("20250101-000100__x35")
