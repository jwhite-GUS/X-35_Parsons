# Airship Hull Shape Optimization - Program Specification
## Faithful Replication of Parsons' Work

### 1. Project Overview

**Objective**: Implement an eight-parameter airship hull shape optimization system that replicates Parsons' original work using piecewise polynomial equations and Young's drag formula.

**Key Features**:
- Eight-parameter hull shape parameterization
- Piecewise polynomial shape generation (forebody, midbody, tail)
- Boundary layer drag calculation using Young's formula
- Nelder-Mead optimization to minimize drag coefficient
- Volume constraint enforcement via soft penalties
- Comprehensive constraint validation

### 2. System Architecture

#### 2.1 Core Components

1. **Shape Parameterization Module**
   - Eight non-dimensional parameters
   - Polynomial coefficient solvers with proper boundary conditions
   - Shape evaluation functions

2. **Aerodynamic Analysis Module**
   - Volume calculation via numerical integration
   - Boundary layer integration using analytic slopes
   - Drag coefficient computation with configurable transition

3. **Optimization Module**
   - Objective function with soft constraint penalties
   - Nelder-Mead algorithm implementation
   - Parameter validation

4. **Visualization Module** (Optional)
   - Hull shape plotting
   - Parameter sensitivity analysis
   - Optimization convergence tracking

#### 2.2 Parameter Definitions

| Parameter | Symbol | Description | Range | Units | Nondimensionalization |
|-----------|--------|-------------|-------|-------|----------------------|
| rn | rn | Nose radius curvature parameter | > 0 | Non-dimensional | R₀ = rn × D |
| fr | fr | Fineness ratio (Length/Diameter) | ≥ 2.5 | Non-dimensional | L/D |
| xm | xm | Location of maximum diameter | 0 < xm < 1 | Fraction of length | X/L |
| k | k | Curvature at maximum diameter | > 0 | Non-dimensional | k = (-2xₘ²/D) × Kₘ |
| Xi | Xi | Location of inflection point | xm < Xi < 1 | Fraction of length | X/L |
| n | n | Radius at inflection point | 0 < n < 1 | Fraction of max radius | 2Rᵢ/D |
| S | S | Slope at inflection point | > 0 | Non-dimensional | S = Y'(Xᵢ) × (L/D) |
| t | t | Tail-end radius parameter | 0 < t < 1 | Non-dimensional | 2T/D |

### 3. Detailed Module Specifications

#### 3.1 Shape Parameterization Module

**Function**: `solve_forebody(rn, fr, xm)`
- **Purpose**: Solve 4th-degree polynomial coefficients for forebody segment
- **Input**: rn, fr, xm parameters
- **Output**: Array [a4, a3, a2, a1, a0] for Y = a4*X^4 + a3*X^3 + a2*X^2 + a1*X + a0
- **Boundary Conditions**:
  - Y(0) = 0 (nose tip)
  - Y'(0) = 0 (horizontal tangent)
  - Y(xm) = D/2 (maximum radius)
  - Y'(xm) = 0 (slope zero at max diameter)
  - Y''(0) = K₀ (nose curvature from rn parameter)
- **Parameter Mapping**: R₀ = rn × D, K₀ = 1/R₀
- **Coefficient Solution**: Explicit linear solve for a₃, a₄ from boundary conditions

**Function**: `solve_midbody(fr, xm, Xi, n, k, S, t)`
- **Purpose**: Solve 5th-degree polynomial coefficients for midbody segment
- **Input**: fr, xm, Xi, n, k, S, t parameters
- **Output**: Array [b5, b4, b3, b2, b1, b0] for local polynomial Y(u)
- **Boundary Conditions**:
  - At u=0 (X=xm): Y = D/2, Y' = 0, Y'' = Kₘ
  - At u = Xi-xm: Y = Rᵢ, Y' = S_actual, Y'' = 0
- **Slope Mapping**: S_actual = S/fr (nondimensional slope conversion)

**Function**: `solve_tail(fr, xm, Xi, n, S, t)`
- **Purpose**: Solve 5th-degree polynomial coefficients for tail segment
- **Input**: fr, xm, Xi, n, S, t parameters
- **Output**: Array [c5, c4, c3, c2, c1, c0] for local polynomial Y(v)
- **Boundary Conditions**:
  - At v=0 (X=Xi): Y = Rᵢ, Y' = S_actual, Y'' = 0
  - At v = 1-Xi (X=L): Y = T, Y' = 0, Y'' = 0
- **Equation Order**: [Y, Y', Y''] for clarity

**Function**: `evaluate_shape(X, fore_coefs, mid_coefs, tail_coefs, xm, Xi)`
- **Purpose**: Evaluate hull radius Y and slope dY/dX at axial location X
- **Input**: X position, polynomial coefficients, segment boundaries
- **Output**: Tuple (Y, Y') - radius and slope at X
- **Logic**: Determine segment (forebody/midbody/tail) and evaluate appropriate polynomial

#### 3.2 Aerodynamic Analysis Module

**Function**: `compute_volume(fore_coefs, mid_coefs, tail_coefs, xm, Xi, fr)`
- **Purpose**: Calculate hull volume by numerical integration
- **Method**: Trapezoidal integration of cross-sectional area A(x) = π[Y(x)]²
- **Integration**: 1000 slices from X=0 to X=1
- **Output**: Volume in cubic units (normalized)

**Function**: `compute_drag_coefficient(fore_coefs, mid_coefs, tail_coefs, xm, Xi, fr, Re_vol=1e7, transition_point=None)`
- **Purpose**: Calculate drag coefficient using Young's formula
- **Method**: Boundary layer momentum thickness integration with analytic slopes
- **Flow Model**: Laminar flow to transition point, turbulent thereafter
- **Arc Length**: ds = √(1 + Y'²)dx using analytic derivatives
- **Friction Coefficients**:
  - Laminar: Cf ≈ 1.328/√Re_x
  - Turbulent: Cf ≈ 0.455/[log₁₀(Re_x)]²·⁵⁸
- **Young's Formula**: CD = (4πr_TE·θ_TE/V^(2/3))·(1 + H_TE)
- **Transition**: Configurable transition point (default: halfway between xm and Xi)
- **Output**: Tuple (CD, volume)

#### 3.3 Optimization Module

**Function**: `objective_function(params)`
- **Purpose**: Objective function for optimization (minimize drag coefficient + penalties)
- **Input**: Array of 8 parameters [rn, fr, xm, k, Xi, n, S, t]
- **Objective**: CD + W_VOL×(ΔV/V_target)² + W_SHAPE×(shape_penalties)
- **Constraints** (enforced via soft penalties):
  - fr ≥ 2.5
  - 0 < xm < 1
  - 0 < Xi < 1 and Xi > xm
  - rn, k, S > 0
  - 0 < t < 1
  - 0 < n < 1
  - t < n
  - Volume ≈ vol_target (soft constraint)
  - Maximum radius at xm (soft constraint)
- **Output**: Drag coefficient plus penalty terms

**Optimization Algorithm**: Nelder-Mead Simplex Method
- **Initialization**: Create simplex from initial guess with small perturbations
- **Operations**: Reflection, expansion, contraction, reduction
- **Convergence**: Relative improvement < 1e-6 or max iterations (200)
- **Output**: Optimized parameters and minimum objective value

### 4. Implementation Requirements

#### 4.1 Dependencies

```python
import math
import numpy as np
import random
from typing import Tuple, List, Optional
```

#### 4.2 Key Constants

```python
# Target volume for constraint enforcement
VOL_TARGET = 0.54188203  # Should match original design

# Reynolds number for drag calculation
RE_VOL = 1e7  # Volume-based Reynolds number

# Shape factor for turbulent boundary layer
H_TE = 1.4

# Integration parameters
N_VOLUME = 1000      # Volume integration slices
N_DRAG = 10000       # Drag integration steps

# Penalty weights for objective function
W_VOL = 1e3          # Volume constraint weight
W_SHAPE = 1e4        # Shape constraint weight
```

#### 4.3 Error Handling

1. **Linear Algebra Errors**: Catch `np.linalg.LinAlgError` in polynomial solvers
2. **Constraint Violations**: Return penalty values for invalid parameter combinations
3. **Numerical Issues**: Add small numbers (1e-16) to avoid division by zero
4. **Convergence**: Check for optimization convergence and handle failed cases

### 5. Validation and Testing

#### 5.1 Known Design Validation

**Test Case**: Original Parsons design parameters
```python
original_params = [0.7573, 4.8488, 0.5888, 0.1711, 0.7853, 0.6473, 2.2867, 0.1731]
```

**Expected Results**:
- Volume should match target within 1%
- Drag coefficient should reproduce original value
- Shape should be monotonic with maximum at xm

#### 5.2 Optimization Validation

**Multiple Random Starts**: Run optimization from 3-5 different random initial guesses
- **Convergence**: All runs should converge to similar drag coefficients
- **Parameter Consistency**: Optimized parameters should be within reasonable ranges
- **Constraint Satisfaction**: All geometric and physical constraints should be satisfied

#### 5.3 Sensitivity Analysis

**Parameter Perturbation**: Test sensitivity of drag coefficient to small parameter changes
- **Expected**: Smooth response to parameter variations
- **Validation**: No discontinuities or numerical instabilities

### 6. Output and Results

#### 6.1 Primary Outputs

1. **Optimized Parameters**: Eight parameter values for minimum drag
2. **Drag Coefficient**: Minimum achievable drag coefficient
3. **Volume**: Calculated hull volume (should match target)
4. **Shape Data**: Arrays of X, Y coordinates for plotting

#### 6.2 Secondary Outputs

1. **Optimization History**: Convergence tracking and iteration count
2. **Constraint Status**: Verification that all constraints are satisfied
3. **Shape Validation**: Confirmation that maximum radius occurs at xm

### 7. Usage Examples

#### 7.1 Basic Optimization Run

```python
# Initial guess (close to known optimal)
initial_guess = [0.75, 4.8, 0.6, 0.1, 0.78, 0.65, 2.0, 0.17]

# Run optimization
optimized_params, min_objective = nelder_mead_optimize(initial_guess)

# Display results
print(f"Optimized parameters: {optimized_params}")
print(f"Minimum objective value: {min_objective:.5f}")
```

#### 7.2 Shape Evaluation

```python
# Generate shape data for plotting
X_points = np.linspace(0, 1, 100)
Y_points = []
for X in X_points:
    Y, _ = evaluate_shape(X, fore_coefs, mid_coefs, tail_coefs, xm, Xi)
    Y_points.append(Y)
```

### 8. Future Enhancements

#### 8.1 Bézier Curve Adaptation

**Motivation**: More direct geometric control and simpler monotonicity enforcement

**Implementation**:
1. Define control points for hull curve
2. Replace polynomial solvers with Bézier interpolation
3. Use control point positions as design variables
4. Maintain same volume and drag calculation methods

#### 8.2 Advanced Optimization

**Potential Improvements**:
1. Use scipy.optimize.minimize with proper constraint handling
2. Implement gradient-based methods for faster convergence
3. Add multi-objective optimization (drag vs. volume vs. stability)
4. Include manufacturing constraints

#### 8.3 Enhanced Analysis

**Additional Capabilities**:
1. Stability analysis (center of buoyancy vs. center of gravity)
2. Structural analysis integration
3. Multi-point optimization for different flight conditions
4. Uncertainty quantification for parameter variations

### 9. Excel Implementation Guide

#### 9.1 Setup Requirements

1. **Parameter Input Section**: 8 cells for shape parameters
2. **Shape Calculation Table**: X positions with corresponding Y values
3. **Volume Integration**: Trapezoidal rule implementation
4. **Drag Calculation**: Boundary layer integration with friction coefficients
5. **Solver Configuration**: Excel Solver with parameter constraints

#### 9.2 Key Formulas

**Shape Calculation**:
- Forebody: `=a4*X^4 + a3*X^3 + a2*X^2 + a1*X + a0`
- Midbody: `=b5*u^5 + b4*u^4 + b3*u^3 + b2*u^2 + b1*u + b0`
- Tail: `=c5*v^5 + c4*v^4 + c3*v^3 + c2*v^2 + c1*v + c0`

**Volume Integration**:
- `=0.5*PI()*(Y_prev^2 + Y_current^2)*delta_X`

**Drag Coefficient**:
- `=4*PI()*r_TE*theta_TE/(V^(2/3))*(1+H_TE)`

**Arc Length**:
- `=SQRT(1 + Y_prime^2)*delta_X`

### 10. Quality Assurance

#### 10.1 Code Standards

1. **Documentation**: Comprehensive comments explaining mathematical derivations
2. **Modularity**: Clear separation of shape generation, analysis, and optimization
3. **Robustness**: Proper error handling and constraint validation
4. **Performance**: Efficient numerical integration and optimization

#### 10.2 Validation Checklist

- [x] Reproduces original Parsons design results
- [x] All constraints properly enforced via soft penalties
- [x] Optimization converges consistently
- [x] Shape is physically realistic (monotonic, smooth)
- [x] Volume constraint satisfied
- [x] Drag calculation matches theoretical expectations
- [x] Analytic slopes used in boundary layer integration
- [x] Configurable transition model
- [x] Proper parameter nondimensionalization

### 11. Critical Implementation Notes

#### 11.1 Parameter Mappings

- **rn → Curvature**: R₀ = rn × D, K₀ = 1/R₀
- **S → Slope**: S_actual = S/fr (nondimensional slope)
- **Volume**: Normalized to unit length (L = 1)
- **Drag**: Based on volume Reynolds number

#### 11.2 Objective Function Structure

The objective function always computes drag and adds penalties:
```python
CD, V = compute_drag_coefficient(...)
obj = CD
obj += W_VOL * ((V - VOL_TARGET)/VOL_TARGET)**2
obj += W_SHAPE * shape_penalties
return obj
```

#### 11.3 Boundary Layer Integration

Uses analytic slopes for accurate arc length calculation:
```python
Y, Yp = evaluate_shape(X, ...)
ds = sqrt(1 + Yp²) * dX
```

This specification provides a complete framework for implementing Parsons' airship hull shape optimization method, ensuring faithful replication of the original work while maintaining code quality and validation standards.
