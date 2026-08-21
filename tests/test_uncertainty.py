from inverse_gems.uncertainty import solver_status_is_ok, uncertainty_flags


def test_cached_solver_status_is_not_flagged_as_non_success():
    row = {
        "chemistry_status": "complete",
        "solver_status": "cached",
        "solver_rescued": False,
        "water_g": 40.0,
        "xgems_water_g": 40.0,
    }

    assert solver_status_is_ok("cached") is True
    assert "solver_non_success" not in uncertainty_flags(row)
