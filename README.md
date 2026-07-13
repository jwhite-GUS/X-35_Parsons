# Airship Hull Geometry and Drag-Screening Research

Python tooling for studying Parsons' eight-parameter hull geometry and bounded
optimization workflows. The geometry and search machinery are available. The
current drag path is an explicitly selected, constant-edge-velocity screening
surrogate; it is **not** a source-faithful Young/Parsons solver and must not be
used as vehicle-design evidence. See
[`docs/DRAG-MODEL-DECISION-20260713.md`](docs/DRAG-MODEL-DECISION-20260713.md).

## Key Features
- Eight-parameter hull definition (`rn`, `fr`, `xm`, `k`, `Xi`, `n`, `S`, `t`)
- Piecewise polynomial geometry with continuous slope and curvature
- Explicit volume-based Reynolds and reference-length accounting
- Fail-closed drag-model selection with model identity in run metadata
- Dimensionally closed screening surrogate for software/sensitivity work
- Configurable penalties on volume and shape for robust convergence
- Timestamped results directories with manifests, plots, and reports
- Optional HTML/Markdown/PDF report generation driven by run metadata

## Requirements
- Python 3.8 or newer (3.9+ recommended)
- `pip install -r requirements.txt` provides `numpy`, `scipy`, `matplotlib`
- Optional: `weasyprint` if you want to emit PDF reports

## Setup
```bash
# 1. Create a virtual environment (macOS/Linux shown)
python3 -m venv .venv
source .venv/bin/activate

# On Windows PowerShell
python -m venv .venv
. .venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt
```

## Running an Optimization
`bin/run_opt.py` is the main entry point. Supply a target volume and choose a
working fluid or Reynolds-number specification.

```bash
# Baseline run in water at 20 °C, deriving Re_V from the speed U
python bin/run_opt.py \
  --vol-target 0.0195144 \
  --U 2.0 \
  --reference-length-m 1.0 \
  --drag-model constant-edge-velocity-surrogate-v1 \
  --allow-screening-optimization \
  --medium water \
  --run-name x35_water

# Alternative: specify the Reynolds number directly (air example)
python bin/run_opt.py \
  --vol-target 0.0195144 \
  --ReV 1e7 \
  --drag-model constant-edge-velocity-surrogate-v1 \
  --allow-screening-optimization \
  --medium air \
  --run-name x35_air
```

Important flags:
- `--vol-target` (required): desired nondimensional volume `V/L³`
- `--medium {water,air,custom}`: working fluid; `--rho` / `--nu` override defaults
- `--U` with `--reference-length-m`, or `--ReV`: derive or prescribe the
  volume-based Reynolds number
- `--drag-model constant-edge-velocity-surrogate-v1`: explicit opt-in to the
  OPEN screening model; there is no validated default
- `--allow-screening-optimization`: second, explicit acknowledgement that the
  optimizer output is relative software screening and remains machine-flagged
  `hull_evidence_eligible=false`
- `--max-iter`, `--w-volume`, `--w-shape`: tune the optimization process
- `--results-root`: change the output root (defaults to `results/`)

Each run creates a UTC-stamped directory under `results/` containing:
- `artifacts/result.json` – serialized optimization output
- `figures/` and `tables/` – generated plots and CSV summaries
- `logs/run.log` – detailed console log
- `meta/config.json` – captured CLI arguments, git hash, and environment info
- `manifest.json` and `reports/` – verification hashes and rendered summaries

A pointer file or symlink named `latest__<run-name>` is refreshed after every
run to make automation easier.

## Visualizing Results
```bash
# Plot radius/slope and convergence curves for the most recent x35 run
python bin/plot_results.py --result results/latest__x35.txt

# Export plots and tables to custom locations
python bin/plot_results.py --result results/20250913-175823_962__x35_air/artifacts/result.json \
  --out-dir build/figures --out-prefix x35_air_175823
```

`plot_results.py` resolves pointer files (`latest__*.txt`), run directories, or
explicit `result.json` paths.

## Generating Reports
```bash
# Build HTML and Markdown reports (skip PDF generation)
python bin/make_report.py --result results/latest__x35.txt --no-pdf

# Attempt HTML/Markdown/PDF (requires weasyprint for PDF support)
python bin/make_report.py --result results/latest__x35.txt
```

The reports are written to `<run-dir>/reports/summary.(html|md|pdf)` and include
embedded metadata, KPIs, and plots.

## Programmatic Usage
```python
from airship_opt import CONSTANT_EDGE_SURROGATE_V1, Params, Config, nelder_mead
from airship_opt.media import make_water

params = Params(rn=0.7573, fr=4.8488, xm=0.5888, k=0.1711,
                Xi=0.7853, n=0.6473, S=2.2867, t=0.1731)
medium = make_water()
cfg = Config(
    vol_target=0.0195144,
    medium=medium,
    Re_V=1.0e7,
    drag_model=CONSTANT_EDGE_SURROGATE_V1,
    allow_screening_optimization=True,
)

bounds = [
    (0.2, 1.5),  # rn
    (2.5, 15.0), # fr
    (0.05, 0.95),
    (0.02, 0.35),
    (0.10, 0.995),
    (0.30, 1.10),
    (0.50, 3.50),
    (0.00, 0.40),
]
result = nelder_mead(list(params.__dict__.values()), bounds, cfg)
```

Every saved result carries the model identifier, an explicit non-evidence
flag, the evaluated Reynolds envelope, and a numerical comparison with the
closed-form laminar flat-plate integral. The benchmark checks the retained
integration arithmetic only. It does not validate the magnitude of the hull
screening coefficient.

## Project Layout
```
airship_opt/
  __init__.py            # Package exports
  types.py               # Params, Config, Result dataclasses
  geometry.py            # Hull polynomials and samplers
  aero.py                # Volume and drag calculations
  objective.py           # Objective and penalty terms
  optimize.py            # Nelder–Mead driver
  io_viz.py              # Result I/O, manifests, latest pointer mgmt
  utils.py               # Shared helpers
  media/                 # Fluid property models (air, water, custom)
  objectives/            # Reynolds helpers and penalties
  reporting/             # Report builders (HTML/MD/PDF)

bin/
  run_opt.py             # CLI optimization driver
  plot_results.py        # Plotting and summary tables
  make_report.py         # Report generation CLI
  latest.py              # Resolve latest run directories/files

tests/
  test_medium.py
  test_report_builder.py
  test_summary_fields.py
  test_latest_cli.py
  test_latest_pointer.py
  test_manifest_refresh.py
```

## Running the Test Suite
```bash
pytest
```

## Source status

The repository retains the actual 1974 Parsons, Goodson, and Goldschmied
article and the correct Young R&M 1874 body-of-revolution report. A previously
indexed `NASA-CR-132692.pdf` was found by direct title-page inspection to be an
unrelated Chin/Barbero cable-mount program guide; it has been renamed and
excluded from the drag lineage. The retained Young R&M 1838 is the aerofoil
precursor, not the body-of-revolution source cited by Parsons.

Additional documents in `references/` support future physics reconstruction
and validation:

- Wind-tunnel baselines: NACA TR-138 (1923) and TR-291 (1928) for C-class
  airship hull drag; NACA TR-1271 and TN-3478 for minimum-wave-drag boattails.
- Core lineage: Parsons, Goodson, and Goldschmied (1974); Young (1939) R&M
  1874; R&M 1838 retained only as the cited aerofoil precursor.
- Transition and correlations: MIT boundary-layer notes, MIT empirical
  transition notes (Michel correlation), and the CFD-Online skin-friction
  compendium covering Prandtl, Prandtl–Schlichting, Granville, and ITTC fits.
- Modern CFD: NASA 2012 study on Sears–Haack transition at Mach 2.
- Environment and tooling: U.S. Standard Atmosphere 1976, CoolProp Python
  wrapper docs, SciPy `optimize.minimize` reference, and pymoo algorithm guide.

See `references/README.md` for full provenance and download links.

## References
- Parsons, J. S., Goodson, R. E., & Goldschmied, F. R. (1974). *Shaping of
  Axisymmetric Bodies for Minimum Drag in Incompressible Flow*. Journal of
  Hydronautics, 8(3), 100–107.
- Young, A. D. (1939). *The Calculation of the Total and Skin Friction Drags
  of Bodies of Revolution at Zero Incidence*. ARC R&M 1874.
- Zahm, A. F., Smith, R. H., & Hill, G. C. (1923). *The Drag of C Class
  Airship Hull with Varying Length of Cylindric Midships*. NACA TR-138.
- Zahm, A. F., Smith, R. H., & Louden, F. A. (1928). *Drag of C-Class Airship
  Hulls of Various Fineness Ratios*. NACA TR-291.
- Harder, K. C., & Rennemann, C. (1956). *On Boattail Bodies of Revolution
  Having Minimum Wave Drag*. NACA TR-1271.
- Harder, K. C., & Murman, E. M. (1955). *Minimum-Wave-Drag Bodies with
  Prescribed Base Area and Volume*. NACA TN-3478.
- Choudhari, M. M., Tokugawa, N., Li, F., Chang, C.-L., White, J. A., et al.
  (2012). *Computational Investigation of Supersonic Boundary Layer Transition
  Over Canonical Fuselage Nose Configurations*. NASA TM-2012-217602.
- Minzner, R. A. (1976). *The 1976 Standard Atmosphere and its Relationship to
  Earlier Standards*. NASA TM-X-74335.
