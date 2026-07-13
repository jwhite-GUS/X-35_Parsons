# Drag-model validity decision

Date: 2026-07-13

Status: Tier-3 decision; proceed with provenance repair and an explicitly
bounded surrogate. Do not describe the current drag calculation as a faithful
Parsons or Young implementation.

## Decision

I will separate three things that the repository currently blends together:

1. Parsons' eight-parameter geometry;
2. a dimensionally closed, constant-edge-velocity skin-friction surrogate;
3. a future source-faithful Parsons drag stack.

The surrogate will require explicit selection. It may support software and
sensitivity work, but its output will remain OPEN for vehicle design. A
source-faithful implementation remains unavailable until the external-flow,
boundary-layer, transition, separation, and trailing-edge accounts are
reconstructed and benchmarked.

This is not the aerodynamics-domain default of preserving an existing
coefficient function for compatibility. It is selected because the concrete
requirement is stronger: a number exposed to optimization must identify the
model that produced it and must not silently claim source fidelity it does not
possess.

## Extracted source correction

- EXTRACTED: `X-35 Parsons Clean.pdf`, SHA-256
  `48ea60d43dc2293e26511773fdc41f42cf9b9f17af6c9e3138f2bcf64eff11f8`,
  is the eight-page Parsons, Goodson, and Goldschmied 1974 article. It defines
  `Rv = U∞ V^(1/3) / ν`, defines drag coefficient on `V^(2/3)`, gives the
  eight-parameter body class, and states that the drag model includes inviscid
  external flow, laminar and turbulent boundary layers, transition, and
  separation constraints.
- EXTRACTED: `references/NASA-CR-132692.pdf`, SHA-256
  `65884a4d980491197363849d69157941fceb6e29228a8850f18a0969e40612a0`,
  is actually Chin and Barbero's July 1975 guide to a transonic-tunnel active
  cable-mount program. It is not the Parsons source described by the old
  reference index.
- EXTRACTED: `references/Young1939_RM1838.pdf`, SHA-256
  `acf06a3035a87c2639df1a9f516dea73a25c5578bd72b944377cca902247bc2f`,
  is the aerofoil profile-drag report. Parsons cites Young R&M 1874, *The
  Calculation of the Total and Skin Friction Drags of Bodies of Revolution at
  Zero Incidence*, for the body-of-revolution terminal relation.
- EXTRACTED: the acquired R&M 1874 Cranfield witness has SHA-256
  `52f4b1326b1ea5316dcc33ac64b61abde1eedf9efc64d69bc7e64405fe11bc73`.
  Distribution rights beyond linked research custody have not been independently
  adjudicated.

## Constraints-first comparison

| requirement | narrow Reynolds patch | full Parsons implementation now | explicit surrogate plus future faithful solver |
|---|---|---|---|
| correct source identity | no | yes | yes |
| dimensionless local Reynolds path | partly | yes | yes |
| honest terminal equation | no | yes | yes, within the surrogate's stated assumptions |
| no silent vehicle authority | no | possible | yes, by explicit model selection and status output |
| bounded implementation and review | yes | no | yes |
| path to later source fidelity | weak | direct but large | direct through a separate solver contract |

The narrow patch is rejected because it would repair one factor while leaving
the source identity and model claim false. Immediate full reconstruction is
deferred because the repository does not yet contain the required external-flow
and boundary-layer solvers or their validation fixtures. Switching cost for the
selected boundary is DAYS across the CLI, configuration, reports, tests, and
reader evidence; the eventual faithful solver is ARCHITECTURAL within this
research repository.

## Local contracts

### Geometry

- ACCEPTS: eight parameters and integration resolution.
- PRODUCES: a nondimensional profile and `Vbar = V/L³`.
- GUARANTEES: only the implemented polynomial construction and numerical
  integration.
- ASSUMES: `L` is the reference length and the parameter constraints hold.
- DOES NOT KNOW: fluid state, transition, separation, or physical drag.

### Flow condition

- ACCEPTS: either direct `Rv`, or `U∞`, `L`, and `ν`.
- PRODUCES: `Rv = U∞ L Vbar^(1/3)/ν` and
  `Re_L = Rv/Vbar^(1/3)`.
- GUARANTEES: dimensional closure and equivalent direct/derived routes.
- ASSUMES: positive finite inputs and one identified reference length.
- DOES NOT KNOW: whether the flow is incompressible, attached, smooth, or
  physically representative.

### Constant-edge-velocity surrogate

- ACCEPTS: profile, `Re_L`, transition location, and integration resolution.
- PRODUCES: a screening coefficient with an immutable model identifier.
- GUARANTEES: configured Reynolds number reaches every local skin-friction
  calculation; local `Re_s = Re_L sbar`; the terminal coefficient uses the
  declared constant-edge-velocity assumption rather than the old `(1+H)`
  multiplier.
- ASSUMES: incompressible attached flow, `u_e/U∞ = 1`, prescribed transition,
  flat-plate skin-friction correlations, and a nonzero tailboom radius.
- DOES NOT KNOW: the external velocity distribution, pressure gradient,
  axisymmetric boundary-layer development, natural transition, separation,
  roughness, appendages, stability, or realized vehicle performance.

### Future source-faithful solver

- ACCEPTS: physical/nondimensional geometry, flow state, external velocity
  distribution, boundary-layer initial and transition conditions, and
  separation criteria.
- PRODUCES: source-mapped intermediate states and a trailing-edge coefficient.
- GUARANTEES: nothing until source-row tests and independent benchmarks exist.
- ASSUMES: the selected Young/Parsons model's stated valid range.
- DOES NOT KNOW: vehicle authority or operational acceptability.

The locality test passes only after these contracts are separated. The old
`compute_drag` boundary failed because it promised a Parsons/Young coefficient
without accepting or constructing the information that the source model needs.

## Structural check

The repair is HYBRID. Explicit units, reference length, model identity, and
fail-closed selection are structural constraints. Numerical integration is the
irreducible computed work inside the selected model. A warning string alone is
rejected because downstream optimization can ignore it; model identity must
travel in configuration and result metadata.

The phenomena check passes for the surrogate only as a mathematical screening
account. It fails for a realized drag claim because no path supplies pressure
distribution, separation, surface condition, or physical validation.

## Implemented claim boundary and checks

- EXTRACTED: optimization is disabled unless configuration carries
  `allow_screening_optimization=True`; the CLI requires the corresponding
  `--allow-screening-optimization` acknowledgement.
- EXTRACTED: result JSON, run metadata, text/CSV summaries, and HTML/Markdown
  reports carry `hull_evidence_eligible=false` and
  `physical_magnitude_validated=false`. Consumers withhold the numeric value
  when the explicit false evidence flag is absent.
- EXTRACTED: the computational input envelope is
  `1e5 <= Re_L <= 1e9`, transition location in `[0,1]`, first turbulent
  station at `Re_s >= 5e5`, and turbulent stations at `Re_s <= 1e9`.
  These bounds delimit where this software path will run; they do not validate
  its physics.
- EXTRACTED: the retained laminar operation is compared numerically with its
  own closed-form unit-flat-plate integral. At the end-to-end test condition
  `Re_L = 36,997,150.18`, 50,000 right-endpoint steps differ from the closed
  form by `0.00326045`, below the declared `0.005` arithmetic tolerance.
- INFERRED: passing that benchmark establishes only that the numerical
  integration reproduces the stated surrogate operation. It supplies no
  tunnel, CFD, or flight agreement and cannot promote the coefficient to hull
  evidence.

The superseded monolithic script is retained for forensic comparison but its
drag function now raises before the old formula can execute. The former
"faithful replication" specification is marked historical and points here.

## Adversary

- Lazy path: multiplying by speed would make the units look better while
  leaving the wrong sources and missing model states untouched. Rebuttal: the
  source and component boundaries are repaired before arithmetic.
- Fragility path: preserving the old default could let old scripts keep
  producing unlabeled coefficients. Rebuttal: the model must be selected
  explicitly and written to run metadata.
- Alternative path: implement all of Parsons now. Rebuttal: that would require
  unreviewed choices for several absent solvers and correlations. The future
  contract is preserved without pretending those choices are complete.

Verdict: HIGH confidence in the provenance correction, dimensional routing,
arithmetic benchmark, and fail-closed reporting boundary. The surrogate's
usefulness for relative screening remains a HYPOTHESIS; no confidence claim is
made for physical drag prediction.

## What I cannot verify

- I cannot verify that the 1974 article alone contains enough detail to
  reproduce the original E7ET and Douglas-Neumann programs.
- I cannot verify copyright or redistribution permission for every retained
  PDF; public links and hashes establish custody, not publication rights.
- I cannot verify any present X35 drag coefficient, optimized shape, or vehicle
  consequence against water-tunnel, wind-tunnel, CFD, or flight data.
- The independent-hypothesis gate for this review was contaminated because the
  prior dimensional audit was read before that gate was loaded. Claude's
  separate review independently confirmed the source mismatches and approved
  the bounded-surrogate direction, but did not execute the repaired runtime;
  final runtime reconciliation remains required before publication.
