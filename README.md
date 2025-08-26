# Airship Hull Shape Optimization - Parsons' Method

A modular Python implementation of Parsons' eight-parameter airship hull shape optimization method using piecewise polynomial geometry and Young's drag formulation.

## Overview

This package implements Parsons' method for optimizing airship hull shapes using:
- **Eight-parameter hull definition** (rn, fr, xm, k, Xi, n, S, t)
- **Piecewise polynomial geometry** (4th-degree forebody, 5th-degree midbody and tail)
- **Young's drag formulation** with boundary layer momentum thickness integration
- **Nelder-Mead optimization** with soft constraint penalties
- **Modular architecture** for easy integration into larger systems

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Run Optimization
```bash
# Run optimization with target volume
python bin/run_opt.py --vol-target 0.020257 --out result.json

# Run with custom parameters
python bin/run_opt.py --vol-target 0.020257 --max-iter 500 --w-volume 1.0 --w-shape 1.0 --out result.json
```

### Generate Visualizations
```bash
# Create plots from optimization results
python bin/plot_results.py --result result.json --out-prefix figures/

# Generate summary without plots
python bin/plot_results.py --result result.json --out-prefix summary --no-plots
```

### Use as Library
```python
from airship_opt import Params, Config, nelder_mead, sample_profile

# Define parameters
params = Params(rn=0.7573, fr=4.8488, xm=0.5888, k=0.1711,
               Xi=0.7853, n=0.6473, S=2.2867, t=0.1731)

# Sample hull profile
X, Y, Yp = sample_profile(params, n=400)

# Run optimization
cfg = Config(vol_target=0.020257)
result = nelder_mead([0.7573, 4.8488, 0.5888, 0.1711, 0.7853, 0.6473, 2.2867, 0.1731],
                     bounds=[(0.2,1.5), (2.5,15.0), (0.05,0.95), (0.02,0.35), 
                            (0.10,0.995), (0.30,1.10), (0.50,3.50), (0.00,0.40)],
                     cfg=cfg)
```

## Package Structure

```
airship_opt/
├── __init__.py          # Main package interface
├── types.py             # Data structures (Params, Config, Result, etc.)
├── geometry.py          # Shape generation and polynomial solvers
├── aero.py              # Volume and drag calculations
├── objective.py         # Objective function and penalties
├── optimize.py          # Nelder-Mead optimization
└── io_viz.py            # Save/load results

bin/
├── run_opt.py           # Command-line optimization driver
└── plot_results.py      # Visualization script

tests/
├── test_geometry.py     # Geometry tests
├── test_aero.py         # Aerodynamics tests
├── test_objective.py    # Objective function tests
└── test_end2end.py      # End-to-end tests
```

## Parameters

| Parameter | Description | Range | Units |
|-----------|-------------|-------|-------|
| `rn` | Nose curvature ratio | 0.2-1.5 | (rn × D = R₀) |
| `fr` | Fineness ratio (L/D) | 2.5-15.0 | - |
| `xm` | Max diameter station | 0.05-0.95 | L |
| `k` | Curvature at xm | 0.02-0.35 | - |
| `Xi` | Inflection point | 0.10-0.995 | L |
| `n` | Radius ratio at Xi | 0.30-1.10 | - |
| `S` | Slope control | 0.50-3.50 | - |
| `t` | Tail radius | 0.00-0.40 | D/2 |

## Features

### ✅ Implemented
- **Faithful Parsons replication** with corrected parameter mappings
- **Modular architecture** for easy integration and testing
- **Headless optimization** with JSON result output
- **Publication-quality visualization** (radius, slope, convergence plots)
- **Comprehensive test suite** with continuity and parameter validation
- **Soft constraint optimization** with volume and shape penalties
- **Analytic slope integration** for boundary layer calculations

### 🔄 Future Enhancements
- **Bézier curve geometry** (drop-in replacement for polynomials)
- **Advanced optimization algorithms** (SLSQP, genetic algorithms)
- **Multi-objective optimization** (drag vs. volume vs. manufacturability)
- **GUI interface** for interactive design exploration
- **CFD integration** for validation and refinement
- **Manufacturing constraints** (minimum radius of curvature, etc.)

## Critical Implementation Features

### Parameter Mappings
- **Nose curvature**: `R₀ = rn × D`, `K₀ = 1/R₀`
- **Slope at inflection**: `S_actual = S/fr`
- **Volume target**: Dynamic from original design validation

### Boundary Layer Integration
- **Analytic slopes**: `ds = √(1 + Y'²)dx` for accurate arc length
- **Transition model**: Configurable laminar-turbulent transition point
- **Young's formula**: `CD = 4π × r_TE × θ_TE / V^(2/3) × (1 + H_TE)`

### Optimization Strategy
- **Soft constraints**: Volume and shape penalties instead of hard bounds
- **Peak enforcement**: Slope sign changes at maximum diameter point
- **Continuity**: C¹ continuity at segment boundaries with configurable tolerances

## Example Output

```
Airship Hull Optimization Results
================================

Parameters:
  rn (nose curvature): 0.7573
  fr (fineness ratio): 5.4738
  xm (max diameter):   0.5888
  k (curvature):       0.1711
  Xi (inflection):     0.7853
  n (radius ratio):    0.6473
  S (slope):           2.2867
  t (tail radius):     0.1731

Results:
  Drag Coefficient:    0.003979
  Volume:              0.019747
  Objective:           4.614025e-03
  Iterations:          6

Geometry:
  Max diameter at:     0.589L
  Inflection at:       0.785L
  Tail radius:         0.173D/2
```

## Validation

The implementation has been validated against Parsons' original work:
- ✅ **Volume calculation**: Matches within 6% of original design
- ✅ **Drag coefficient**: Realistic values (0.002-0.011 range)
- ✅ **Continuity**: C¹ continuity verified at segment boundaries
- ✅ **Parameter mappings**: Correct nondimensionalization
- ✅ **Optimization**: Converges to reasonable solutions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- Parsons, J. S. (1974). "An Analysis of the Aerodynamic Characteristics of Several Bodies of Revolution Including the Effects of Viscous Flow." NASA CR-132692.
- Young, A. D. (1939). "The Calculation of the Profile Drag of Aerofoils." ARC R&M 1838.
