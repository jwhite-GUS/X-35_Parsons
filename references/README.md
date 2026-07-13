# Airship Optimization Reference Pack

This folder caches primary sources and authoritative documentation for the
X35 geometry and drag-model reconstruction. Hash identity is evidence of
custody, not permission to republish a source or proof that the implemented
model reproduces it.

## Core Aerodynamics

- `Parsons-Goodson-Goldschmied-1974.pdf` — *Shaping of Axisymmetric
  Bodies for Minimum Drag in Incompressible Flow*, Journal of Hydronautics
  8(3), 100–107. SHA-256
  `48ea60d43dc2293e26511773fdc41f42cf9b9f17af6c9e3138f2bcf64eff11f8`.
- `Young1939_RM1874.pdf` — A. D. Young, *The Calculation of the Total and
  Skin Friction Drags of Bodies of Revolution at Zero Incidence*, ARC R&M
  1874. Cranfield witness SHA-256
  `52f4b1326b1ea5316dcc33ac64b61abde1eedf9efc64d69bc7e64405fe11bc73`;
  [public repository locator](https://reports.aerade.cranfield.ac.uk/bitstream/handle/1826.2/1433/arc-rm-1874.pdf?isAllowed=y&sequence=1).
- `Young1939_RM1838.pdf` — *The Calculation of the Profile Drag of
  Aerofoils*. This is the two-dimensional precursor cited by R&M 1874; it is
  not the body-of-revolution source cited by Parsons.
- `NACA-TR-138.pdf` – Zahm, Smith & Hill (1923). Experimental drag for C-class airship hulls with variable cylindrical midsections.
- `NACA-TR-291.pdf` – Zahm, Smith & Louden (1928). Drag measurements for C-class hulls across fineness ratios (baseline for validation curves).
- `NACA-TR-1271.pdf` – Harder & Rennemann (1956). Optimization of boattail bodies for minimum wave drag.
- `NACA-TN-3478.pdf` – Harder & Murman (1955). Higher-order correction to Ward’s slender-body drag theory for optimal boattails.
- `NASA-2012-Sears-Haack-Transition.pdf` – Choudhari et al. (2012). Modern CFD/experiment comparison of boundary-layer transition on Sears–Haack-style noses.

## Excluded from the drag lineage

- `Chin-Barbero-1975-NASA-CR-132692-cable-mount.pdf` — *User's Guide for a
  Revised Computer Program to Analyze the LRC 16-Foot Transonic Dynamics
  Tunnel Active Cable Mount System*, NASA CR-132692. SHA-256
  `65884a4d980491197363849d69157941fceb6e29228a8850f18a0969e40612a0`.
  The old index mislabeled this file as Parsons-era hull-drag methodology.
  Direct title-page inspection disproved that identity; the file is retained
  only to preserve the correction trail.

## Boundary-Layer Correlations & Transition
- `cfdonline_skin_friction.html` – CFD-Online compilation of laminar/turbulent flat-plate Cf correlations (Prandtl, Prandtl–Schlichting, Granville, ITTC, etc.).
- `mit_boundary_layer.html` – MIT Unified Engineering notes covering Blasius solutions and Cf = 0.664/vRe_x derivation.
- `mit_transition.html` – MIT Unified Engineering notes on empirical transition criteria (Michel correlation discussion).
- `ddg_michel_transition.html` – DuckDuckGo capture of Michel correlation search results (for traceability of additional sources).
- `ddg_granville_skin_friction.html` – DuckDuckGo capture of Granville skin-friction correlation search results.

## Environment & Fluid Properties
- `US-Standard-Atmosphere-1976.pdf` – Complete U.S. Standard Atmosphere 1976 (NTRS 19770009539) for altitude-dependent properties.
- `CoolProp-Python-wrapper.html` – Official CoolProp documentation for Python wrapper installation/usage.

## Optimization Tooling
- `scipy-optimize-minimize.html` – SciPy `optimize.minimize` reference (SLSQP/trust-constr details for constrained refinement).
- `pymoo-algorithms.html` – Overview of pymoo’s evolutionary / multi-objective optimizers (NSGA-II, etc.).

## Geometry / Minimum-Drag Theory
- `NACA-TR-1271.pdf` & `NACA-TN-3478.pdf` (see above) – Provide analytic guidance for designing minimum-wave-drag boattails.
- `NASA-2012-Sears-Haack-Transition.pdf` (see above) – Supplies modern data on Sears–Haack bodies at supersonic conditions, informing high-Re regimes.

All HTML captures are static snapshots; if richer context is needed, revisit the listed URLs. These documents should be tracked to ensure long-term reproducibility of the aerodynamic model and future optimization enhancements.
