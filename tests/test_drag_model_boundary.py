"""Regression tests for the explicit X35 drag-surrogate boundary."""

import math
import json

import pytest

from airship_opt.aero import (
    CONSTANT_EDGE_SURROGATE_V1,
    compute_drag,
    laminar_flat_plate_benchmark,
    surrogate_claim_metadata,
    volume,
)
from airship_opt.media import make_water
from airship_opt.io_viz import save_result
from airship_opt.optimize import nelder_mead
from airship_opt.types import Config, Params


PARAMS = Params(
    rn=0.7573,
    fr=4.8488,
    xm=0.5888,
    k=0.1711,
    Xi=0.7853,
    n=0.6473,
    S=2.2867,
    t=0.1731,
)


def config(**patch):
    values = {
        "vol_target": 0.0195144,
        "medium": make_water(),
        "n_vol_steps": 300,
        "n_bl_steps": 500,
        "Re_V": 1.0e7,
        "drag_model": CONSTANT_EDGE_SURROGATE_V1,
    }
    values.update(patch)
    return Config(**values)


def test_drag_has_no_silent_default_model():
    with pytest.raises(ValueError, match="No validated default drag model"):
        compute_drag(PARAMS, config(drag_model=None), make_water())


def test_configured_reynolds_number_reaches_screening_coefficient():
    low_Re_cd, low_volume = compute_drag(
        PARAMS, config(Re_V=1.0e6), make_water()
    )
    high_Re_cd, high_volume = compute_drag(
        PARAMS, config(Re_V=1.0e8), make_water()
    )

    assert math.isclose(low_volume, high_volume)
    assert high_Re_cd < low_Re_cd


def test_speed_and_direct_reynolds_routes_are_equivalent():
    base = config()
    V_bar = volume(PARAMS, base)
    U = 2.0
    L = 3.0
    Re_V = U * L * V_bar ** (1.0 / 3.0) / base.medium.nu

    direct_cd, _ = compute_drag(
        PARAMS,
        config(Re_V=Re_V),
        base.medium,
    )
    derived_cd, _ = compute_drag(
        PARAMS,
        config(Re_V=None, speed_U=U, reference_length_m=L),
        base.medium,
    )

    assert math.isclose(direct_cd, derived_cd, rel_tol=1.0e-12)


def test_laminar_flat_plate_arithmetic_matches_closed_form():
    benchmark = laminar_flat_plate_benchmark(1.0e7)

    assert benchmark["passed"] is True
    assert benchmark["relative_error"] <= benchmark["tolerance"]
    assert "not hull-drag validation" in benchmark["claim"]


def test_screening_model_rejects_reynolds_outside_declared_envelope():
    with pytest.raises(ValueError, match="computational screening envelope"):
        compute_drag(PARAMS, config(Re_V=1.0e12), make_water())


def test_screening_model_rejects_transition_outside_declared_envelope():
    with pytest.raises(ValueError, match="transition point"):
        compute_drag(PARAMS, config(transition_point=1.1), make_water())


def test_optimizer_requires_explicit_screening_acknowledgement():
    bounds = [
        (0.2, 1.5),
        (2.5, 15.0),
        (0.05, 0.95),
        (0.02, 0.35),
        (0.10, 0.995),
        (0.30, 1.10),
        (0.50, 3.50),
        (0.00, 0.40),
    ]

    with pytest.raises(ValueError, match="Screening optimization is disabled"):
        nelder_mead(list(PARAMS.__dict__.values()), bounds, config(), max_iter=0)


def test_optimizer_result_carries_machine_non_evidence_boundary():
    bounds = [
        (0.2, 1.5),
        (2.5, 15.0),
        (0.05, 0.95),
        (0.02, 0.35),
        (0.10, 0.995),
        (0.30, 1.10),
        (0.50, 3.50),
        (0.00, 0.40),
    ]
    cfg = config(
        allow_screening_optimization=True,
        n_vol_steps=80,
        n_bl_steps=120,
    )

    result = nelder_mead(
        list(PARAMS.__dict__.values()), bounds, cfg, max_iter=0
    )
    claim = result.meta["coefficient_claim"]

    assert claim["quantity"] == "screening_coefficient"
    assert claim["hull_evidence_eligible"] is False
    assert claim["physical_magnitude_validated"] is False
    assert claim["screening_optimization_acknowledged"] is True
    assert claim["validity_envelope"]["assessment"] == "computational-envelope-pass"
    assert claim["closed_form_benchmark"]["passed"] is True


def test_serialized_coefficient_and_history_rows_carry_non_evidence_flag(tmp_path):
    bounds = [
        (0.2, 1.5),
        (2.5, 15.0),
        (0.05, 0.95),
        (0.02, 0.35),
        (0.10, 0.995),
        (0.30, 1.10),
        (0.50, 3.50),
        (0.00, 0.40),
    ]
    cfg = config(
        allow_screening_optimization=True,
        n_vol_steps=80,
        n_bl_steps=120,
    )
    result = nelder_mead(
        list(PARAMS.__dict__.values()), bounds, cfg, max_iter=1
    )

    path = save_result(result, tmp_path / "result.json")
    saved = json.loads(path.read_text())

    assert saved["coefficient_claim"]["hull_evidence_eligible"] is False
    assert saved["cd_alias_status"].startswith("deprecated alias")
    assert saved["history"]
    assert all(row["hull_evidence_eligible"] is False for row in saved["history"])
    assert all("screening_coefficient" in row for row in saved["history"])


def test_claim_metadata_does_not_promote_benchmark_to_physical_validation():
    cfg = config(allow_screening_optimization=True)
    claim = surrogate_claim_metadata(PARAMS, cfg, cfg.medium, volume(PARAMS, cfg))

    assert claim["closed_form_benchmark"]["passed"] is True
    assert claim["physical_magnitude_validated"] is False
    assert claim["validity_envelope"]["physical_validity"] == "unestablished"
