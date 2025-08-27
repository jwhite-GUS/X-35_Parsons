"""
Report builder module for airship hull optimization.

This module provides functions to generate comprehensive HTML, Markdown,
and PDF reports from optimization run results.
"""

from __future__ import annotations
from pathlib import Path
import base64
import json
import csv
import textwrap
from typing import Dict, List

def _b64(path: Path) -> str:
    """Convert a file to base64 encoding."""
    return base64.b64encode(path.read_bytes()).decode("ascii")

def _load_meta(base_dir: Path) -> Dict:
    """Load metadata from meta/config.json."""
    p = base_dir / "meta" / "config.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _load_result(base_dir: Path) -> Dict:
    """Load optimization result from artifacts/result.json."""
    p = base_dir / "artifacts" / "result.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

def _fig_paths(base_dir: Path) -> List[Path]:
    """Get sorted list of figure paths."""
    figs = base_dir / "figures"
    return sorted([p for p in figs.glob("*.png")]) if figs.exists() else []

def _kv(k: str, v: any) -> str:
    """Format a key-value pair as an HTML table row."""
    v = "N/A" if v is None else v
    return f"<tr><th style='text-align:left;padding-right:12px'>{k}</th><td>{v}</td></tr>"

def build_html(base_dir: Path) -> str:
    """
    Build an HTML report with embedded images and metadata.
    
    Args:
        base_dir: Base directory of the optimization run
        
    Returns:
        HTML string with embedded content
    """
    meta = _load_meta(base_dir)
    res = _load_result(base_dir)
    figs = _fig_paths(base_dir)

    med = meta.get("medium", {})
    spd = meta.get("speed", {})
    ren = meta.get("reynolds", {})

    # derive ReV if missing
    ReV = ren.get("ReV")
    if ReV is None and spd.get("U") is not None and med.get("nu") is not None and res.get("volume") is not None:
        ReV = float(spd["U"]) * (res["volume"] ** (1/3)) / float(med["nu"])

    # Parameters table
    params = {k: res.get(k) for k in ("rn","fr","xm","k","Xi","n","S","t")}
    kpis = {
        "cd_te": res.get("cd"),  # note: cd in result.json
        "volume": res.get("volume"),
        "objective": res.get("objective"),
        "iterations": res.get("meta", {}).get("iterations")
    }

    # Embed figures as base64
    fig_html = ""
    for p in figs:
        fig_html += f"<div style='margin:10px 0'><img alt='{p.name}' src='data:image/png;base64,{_b64(p)}' style='max-width:100%;border:1px solid #ddd;border-radius:8px'/><div style='font-size:12px;color:#666'>{p.name}</div></div>"

    # Build HTML
    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Run Summary - {base_dir.name}</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }}
h1,h2 {{ margin: 0.2em 0; }}
table {{ border-collapse: collapse; }}
th,td {{ padding: 4px 6px; border-bottom: 1px solid #eee; }}
.section {{ margin: 18px 0; }}
.kv th {{ font-weight: 600; white-space: nowrap; }}
.meta {{ font-size: 13px; color: #444; }}
.code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }}
.card {{ padding: 12px; border: 1px solid #eee; border-radius: 12px; }}
</style></head><body>
<h1>Run Summary</h1>
<div class="meta">{base_dir}</div>

<div class="section card">
<h2>Flow Conditions</h2>
<table class="kv">
{_kv("Medium", med.get("name"))}
{_kv("Density (kg/m³)", med.get("rho"))}
{_kv("Viscosity ν (m²/s)", med.get("nu"))}
{_kv("Speed U (m/s)", spd.get("U"))}
{_kv("Reynolds (Re_V)", ReV)}
</table>
</div>

<div class="section card">
<h2>KPIs</h2>
<table class="kv">
{_kv("Drag Coefficient (cd_te)", kpis["cd_te"])}
{_kv("Volume (L³)", kpis["volume"])}
{_kv("Objective", kpis["objective"])}
{_kv("Iterations", kpis["iterations"])}
</table>
</div>

<div class="section card">
<h2>Parameters</h2>
<table class="kv">
{''.join(_kv(k, params[k]) for k in params)}
</table>
</div>

<div class="section card">
<h2>Figures</h2>
{fig_html if fig_html else "<em>No figures found.</em>"}
</div>

<div class="section card">
<h2>Artifacts</h2>
<div class="code">
<ul>
<li>result.json — {base_dir / "artifacts" / "result.json"}</li>
<li>summary.txt — {base_dir / "tables" / "summary.txt"}</li>
<li>summary.csv — {base_dir / "tables" / "summary.csv"}</li>
<li>manifest.json — {base_dir / "manifest.json"}</li>
</ul>
</div>
</div>

</body></html>"""
    return html

def build_markdown(base_dir: Path) -> str:
    """
    Build a Markdown report with figure references and metadata.
    
    Args:
        base_dir: Base directory of the optimization run
        
    Returns:
        Markdown string
    """
    meta = _load_meta(base_dir)
    res = _load_result(base_dir)
    med = meta.get("medium", {})
    spd = meta.get("speed", {})
    ren = meta.get("reynolds", {})
    
    ReV = ren.get("ReV")
    if ReV is None and spd.get("U") and med.get("nu") and res.get("volume"):
        ReV = float(spd["U"]) * (res["volume"] ** (1/3)) / float(med["nu"])
        
    lines = ["# Run Summary",
             f"`{base_dir}`",
             "",
             "## Flow Conditions",
             f"- Medium: {med.get('name','unknown')}",
             f"- Density (kg/m³): {med.get('rho','N/A')}",
             f"- Viscosity ν (m²/s): {med.get('nu','N/A')}",
             f"- Speed U (m/s): {spd.get('U','N/A')}",
             f"- Reynolds (Re_V): {ReV if ReV is not None else 'N/A'}",
             "",
             "## KPIs",
             f"- Drag Coefficient (cd_te): {res.get('cd','N/A')}",
             f"- Volume (L³): {res.get('volume','N/A')}",
             f"- Objective: {res.get('objective','N/A')}",
             f"- Iterations: {res.get('meta',{}).get('iterations','N/A')}",
             "",
             "## Parameters"]
             
    for k in ("rn","fr","xm","k","Xi","n","S","t"):
        lines.append(f"- {k}: {res.get(k,'N/A')}")
        
    lines += ["",
              "## Figures"]
              
    figs = _fig_paths(base_dir)
    if figs:
        for p in figs:
            rel = p.relative_to(base_dir)
            lines.append(f"![{p.name}]({rel.as_posix()})")
    else:
        lines.append("_No figures found._")
        
    lines += ["",
              "## Artifacts",
              f"- artifacts/result.json",
              f"- tables/summary.txt",
              f"- tables/summary.csv",
              f"- manifest.json"]
              
    return "\n".join(lines)

def write_reports(base_dir: Path, make_pdf: bool = True) -> Dict[str, str]:
    """
    Generate HTML, Markdown and optionally PDF reports.
    
    Args:
        base_dir: Base directory of the optimization run
        make_pdf: Whether to attempt PDF generation (requires weasyprint)
        
    Returns:
        Dictionary with paths to generated reports
    """
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    html = build_html(base_dir)
    md = build_markdown(base_dir)
    
    html_path = reports_dir / "summary.html"
    md_path = reports_dir / "summary.md"
    
    html_path.write_text(html, encoding="utf-8")
    md_path.write_text(md, encoding="utf-8")

    pdf_path = None
    if make_pdf:
        try:
            import weasyprint  # type: ignore
            pdf_path = reports_dir / "summary.pdf"
            weasyprint.HTML(string=html, base_url=str(base_dir)).write_pdf(str(pdf_path))
        except Exception:
            pdf_path = None  # silently skip if not available

    return {
        "html": str(html_path),
        "md": str(md_path),
        "pdf": str(pdf_path) if pdf_path else None
    }
