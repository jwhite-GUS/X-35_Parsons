"""
Test media functionality for airship hull optimization.

Tests medium creation and Reynolds number calculations.
"""

import pytest
import math
from airship_opt.media import make_air, make_water, make_custom
from airship_opt.objectives.utils import compute_ReV

def test_default_media():
    """Test default medium creation and properties."""
    # Test default air at sea level
    air = make_air()
    assert air.name == "air_sl"
    assert math.isclose(air.rho, 1.225)
    assert math.isclose(air.nu, 1.46e-5)
    
    # Test default water at 20C
    water = make_water()
    assert water.name == "water_20C"
    assert math.isclose(water.rho, 998.2)
    assert math.isclose(water.nu, 1.004e-6)

def test_custom_media():
    """Test custom medium creation."""
    medium = make_custom(rho=900.0, nu=8e-7, name="oil")
    assert medium.name == "oil"
    assert medium.rho == 900.0
    assert medium.nu == 8e-7

def test_medium_override():
    """Test medium property override."""
    air = make_air(rho=1.0, nu=1.5e-5, name="air_alt")
    assert air.name == "air_alt"
    assert air.rho == 1.0
    assert air.nu == 1.5e-5

def test_reynolds_calculation():
    """Test volume-based Reynolds number computation."""
    # Case A: Direct ReV provided
    assert compute_ReV(U=None, V=0.1, nu=1e-5, ReV=1e6) == 1e6
    
    # Case B: Computed from U, V, and nu
    V = 0.0195144  # example volume
    U = 2.0  # m/s
    nu = 1.004e-6  # water at 20C
    
    Re_V = compute_ReV(U=U, V=V, nu=nu, ReV=None)
    Re_V_expected = U * (V**(1/3)) / nu
    assert math.isclose(Re_V, Re_V_expected)
    
    # Case C: Missing both U and ReV should raise error
    with pytest.raises(ValueError):
        compute_ReV(U=None, V=0.1, nu=1e-5, ReV=None)
