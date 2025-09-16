# Airship Optimization Reference Pack (Update 2025-09-16)

This folder caches primary source material and authoritative documentation that back the Parsons airship optimization analysis and future enhancements. Files were retrieved on 2025-09-16 (UTC).

## Core Aerodynamics
- `Young1939_RM1838.pdf` – A. D. Young, *The Calculation of the Profile Drag of Aerofoils*, ARC R&M 1838 (1939). Canonical derivation of wake momentum-thickness drag and Blasius-based laminar Cf expressions.
- `NASA-CR-132692.pdf` – Chin & Barbero (1975), NASA Contractor Report 132692. Parsons-era methodology and system-level drag validation data.
- `NACA-TR-138.pdf` – Zahm, Smith & Hill (1923). Experimental drag for C-class airship hulls with variable cylindrical midsections.
- `NACA-TR-291.pdf` – Zahm, Smith & Louden (1928). Drag measurements for C-class hulls across fineness ratios (baseline for validation curves).
- `NACA-TR-1271.pdf` – Harder & Rennemann (1956). Optimization of boattail bodies for minimum wave drag.
- `NACA-TN-3478.pdf` – Harder & Murman (1955). Higher-order correction to Ward’s slender-body drag theory for optimal boattails.
- `NASA-2012-Sears-Haack-Transition.pdf` – Choudhari et al. (2012). Modern CFD/experiment comparison of boundary-layer transition on Sears–Haack-style noses.

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
