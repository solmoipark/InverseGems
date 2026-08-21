import numpy as np
import pandas as pd
import pytest

from inverse_gems.kinetics_calibration import calibrate_scm_kinetics
from inverse_gems.reaction_parameters import load_reaction_parameters
from inverse_gems.scm_reaction import (
    SCMLogisticParameters,
    _KINETICS_REGISTRY,
    register_scm_kinetics,
    scm_alpha,
)


TRUE = {"slag": {"A": 0.0, "B": 0.9, "C": 15.0, "D": 0.72, "G": 1.0},
        "fly_ash": {"A": 0.0, "B": 0.6, "C": 45.0, "D": 0.48, "G": 1.0}}


def _synthetic_data(tmp_path, noise=0.015, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    ages = [1, 3, 7, 14, 28, 56, 90, 180, 365]
    for scm, params in TRUE.items():
        for age in ages:
            for _ in range(3):
                alpha = scm_alpha(float(age), SCMLogisticParameters(**params))
                rows.append({"scm": scm, "age_d": age, "dor": float(np.clip(alpha + rng.normal(0, noise), 0, 1))})
    path = tmp_path / "user_dor.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_calibration_recovers_known_parameters(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = _synthetic_data(tmp_path)
    report = calibrate_scm_kinetics(
        data_csv=data, out=tmp_path / "cal", fixed_params={"A": 0.0, "G": 1.0}, config_id="test_cal_v1"
    )
    assert set(report["fits"]) == {"slag", "fly_ash"}
    for scm in TRUE:
        fit = report["fits"][scm]
        assert fit["r2"] > 0.97
        assert fit["parameters"]["D"] == pytest.approx(TRUE[scm]["D"], abs=0.05)
        assert fit["parameters"]["C"] == pytest.approx(TRUE[scm]["C"], rel=0.35)
    assert (tmp_path / "cal" / "calibration_fits.png").exists()

    # the emitted config plugs into the standard reaction-parameter loader
    params = load_reaction_parameters(report["config_path"])
    assert params.id == "test_cal_v1"
    slag = params.scm_parameters["slag"]
    assert isinstance(slag, SCMLogisticParameters)
    assert slag.D == pytest.approx(TRUE["slag"]["D"], abs=0.05)
    # uncalibrated SCMs keep repo defaults
    assert params.scm_parameters["metakaolin"].C == 5.0


def test_calibration_supports_custom_registered_model(tmp_path):
    name = "cal_test_exponential"

    @register_scm_kinetics(name, required=("alpha_max", "k"), asymptote_key="alpha_max")
    def _exponential(t, p):
        return p["alpha_max"] * (1.0 - np.exp(-p["k"] * t))

    try:
        rng = np.random.default_rng(1)
        ages = np.array([1, 3, 7, 14, 28, 56, 90, 180] * 3, dtype=float)
        dor = 0.65 * (1.0 - np.exp(-0.06 * ages)) + rng.normal(0, 0.01, len(ages))
        data = tmp_path / "d.csv"
        pd.DataFrame({"scm": "slag", "age_days": ages, "alpha": np.clip(dor, 0, 1)}).to_csv(data, index=False)

        report = calibrate_scm_kinetics(
            data_csv=data,
            out=tmp_path / "cal",
            model=name,
            param_init={"alpha_max": 0.5, "k": 0.1},
            param_bounds={"alpha_max": (0.05, 1.0), "k": (0.001, 2.0)},
            make_plot=False,
        )
        fit = report["fits"]["slag"]
        assert fit["parameters"]["alpha_max"] == pytest.approx(0.65, abs=0.05)
        assert fit["parameters"]["k"] == pytest.approx(0.06, rel=0.3)
        params = load_reaction_parameters(report["config_path"])
        assert params.scm_parameters["slag"].model == name
    finally:
        _KINETICS_REGISTRY.pop(name, None)


def test_calibration_gates_and_errors(tmp_path):
    few = tmp_path / "few.csv"
    pd.DataFrame({"scm": ["slag"] * 3, "age_d": [1, 7, 28], "dor": [0.1, 0.2, 0.3]}).to_csv(few, index=False)
    with pytest.raises(ValueError, match="No SCM could be calibrated"):
        calibrate_scm_kinetics(data_csv=few, out=tmp_path / "cal", make_plot=False)

    bad_columns = tmp_path / "bad.csv"
    pd.DataFrame({"material_name": ["slag"], "time": [1], "value": [0.1]}).to_csv(bad_columns, index=False)
    with pytest.raises(ValueError, match="column"):
        calibrate_scm_kinetics(data_csv=bad_columns, out=tmp_path / "cal2", make_plot=False)

    with pytest.raises(KeyError, match="Unknown SCM kinetics model"):
        calibrate_scm_kinetics(data_csv=few, out=tmp_path / "cal3", model="no_such_model")


def test_calibration_normalizes_percent_scale(tmp_path):
    rng = np.random.default_rng(5)
    ages = np.array([1, 3, 7, 14, 28, 56, 90, 180, 365] * 2, dtype=float)
    alpha = [scm_alpha(float(a), SCMLogisticParameters(**TRUE["slag"])) for a in ages]
    data = tmp_path / "pct.csv"
    pd.DataFrame({"scm": "slag", "age_d": ages, "dor": np.array(alpha) * 100.0 + rng.normal(0, 1.0, len(ages))}).to_csv(data, index=False)
    report = calibrate_scm_kinetics(
        data_csv=data, out=tmp_path / "cal", fixed_params={"A": 0.0, "G": 1.0}, make_plot=False
    )
    assert report["fits"]["slag"]["parameters"]["D"] == pytest.approx(TRUE["slag"]["D"], abs=0.06)
