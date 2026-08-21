import pandas as pd
import pytest

from inverse_gems.sensitivity import (
    DEFAULT_PARAMETERS,
    _base_structures,
    _lookup_base_value,
    run_reaction_parameter_sensitivity,
    write_sensitivity_tornado,
)


def test_default_parameter_paths_all_resolve():
    base = _base_structures()
    for path in DEFAULT_PARAMETERS:
        value = _lookup_base_value(base, path)
        assert isinstance(value, float)


def test_lookup_falls_back_for_implicit_availability_defaults():
    base = _base_structures()
    assert _lookup_base_value(base, "c3s_c2s_availability.eta.no_such_scm") == 1.0
    with pytest.raises(KeyError):
        _lookup_base_value(base, "pk_model.constants.K1.NoSuchPhase")


def test_sensitivity_mock_sweep_and_tornado(tmp_path):
    report = run_reaction_parameter_sensitivity(
        out=tmp_path / "sens",
        recipes=["OPC 60, slag 40, w/b 0.45, age 28"],
        parameters=["scm_reaction.slag.D", "scm_reaction.slag.C", "pk_model.constants.K1.C3S"],
        use_mock=True,
    )
    assert report["run_count"] > 0
    assert not report["warnings"]

    elasticity = pd.read_csv(tmp_path / "sens" / "sensitivity_elasticity.csv")
    slag_d = elasticity[
        (elasticity.parameter == "scm_reaction.slag.D") & (elasticity.output == "alpha_scm__slag")
    ].iloc[0]
    # D raises the asymptote -> alpha responds positively
    assert slag_d["elasticity"] > 0.1
    # C is a time-scale parameter: raising it slows the reaction at fixed age
    slag_c = elasticity[
        (elasticity.parameter == "scm_reaction.slag.C") & (elasticity.output == "alpha_scm__slag")
    ].iloc[0]
    assert slag_c["elasticity"] < 0.0
    # PK perturbations must not touch SCM kinetics
    pk_alpha = elasticity[
        (elasticity.parameter == "pk_model.constants.K1.C3S") & (elasticity.output == "alpha_scm__slag")
    ].iloc[0]
    assert pk_alpha["elasticity"] == pytest.approx(0.0, abs=1e-12)

    runs = pd.read_csv(tmp_path / "sens" / "sensitivity_runs.csv")
    assert set(runs.direction) == {"minus", "plus"}

    plot = write_sensitivity_tornado(
        elasticity_csv=tmp_path / "sens" / "sensitivity_elasticity.csv",
        out=tmp_path / "sens" / "tornado.png",
        outputs=["porosity", "alpha_scm__slag"],
    )
    assert plot.exists()


def test_sensitivity_records_failures_without_aborting(tmp_path):
    report = run_reaction_parameter_sensitivity(
        out=tmp_path / "sens",
        recipes=["OPC 100, w/b 0.45, age 28"],
        parameters=["scm_reaction.slag.D", "pk_model.constants.K1.NoSuchPhase"],
        use_mock=True,
    )
    assert any("NoSuchPhase" in w for w in report["warnings"])
    assert report["run_count"] > 0
