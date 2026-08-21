import json
from pathlib import Path

import pandas as pd

from inverse_gems.forward_result_summary import summarize_forward_result


def _write_run(tmp_path: Path) -> Path:
    run = tmp_path / "forward_run"
    run.mkdir()
    pd.DataFrame(
        [
            {
                "age_days": 28.0,
                "recipe_id": "recipe_1",
                "chemistry_status": "complete",
                "solver_status": "success",
                "porosity": 0.23,
                "phase_mass__CNASH": 0.045,
                "phase_volume__CNASH": 1.2e-5,
                "phase_volume_reconstructed__CNASH": 1.15e-5,
                "phase_mass__Portlandite": 0.015,
                "phase_volume__Portlandite": 5.0e-6,
                "scalar__pH": 12.6,
                "scalar__system_volume": 8.4e-5,
            },
            {
                "age_days": 90.0,
                "recipe_id": "recipe_1",
                "chemistry_status": "complete",
                "solver_status": "success",
                "porosity": 0.18,
                "phase_mass__CNASH": 0.052,
                "phase_volume__CNASH": 1.3e-5,
                "phase_volume_reconstructed__CNASH": 1.25e-5,
                "phase_mass__Portlandite": 0.0,
                "phase_volume__Portlandite": 0.0,
                "scalar__pH": 12.3,
                "scalar__system_volume": 8.0e-5,
            },
        ]
    ).to_csv(run / "time_series.csv", index=False)
    pd.DataFrame(
        [
            {
                "kind": "phase_mass",
                "raw_name": "CNASH",
                "column": "phase_mass__CNASH",
                "nonzero_count": 2,
                "max_abs_value": 0.052,
            },
            {
                "kind": "phase_mass",
                "raw_name": "Portlandite",
                "column": "phase_mass__Portlandite",
                "nonzero_count": 1,
                "max_abs_value": 0.015,
            },
        ]
    ).to_csv(run / "phase_nonzero_summary.csv", index=False)
    return run


def test_summarize_forward_result_selects_exact_phase_and_scalar(tmp_path):
    run = _write_run(tmp_path)
    out = summarize_forward_result(
        run=run,
        phases=["CNASH"],
        phase_groups=["C-A-S-H", "Portlandite"],
        scalars=["pH", "porosity"],
        out=tmp_path / "summary",
    )

    assert (out / "response_summary.json").exists()
    assert (out / "response_summary.md").exists()
    assert (out / "response_summary.csv").exists()
    payload = json.loads((out / "response_summary.json").read_text(encoding="utf-8"))
    assert payload["mode"] == "time_series"
    assert payload["requested"]["phases"] == ["CNASH"]
    assert payload["requested"]["phase_groups"] == ["C-A-S-H", "Portlandite"]
    assert payload["selected"]["rows"][0]["phases"]["CNASH"]["mass"] == 0.045
    assert payload["selected"]["rows"][0]["phases"]["CNASH"]["volume"] == 1.2e-5
    assert payload["selected"]["rows"][0]["phase_groups"]["C-A-S-H"]["mass"] == 0.045
    assert payload["selected"]["rows"][0]["phase_groups"]["Portlandite"]["mass"] == 0.015
    assert payload["selected_output_policy"]["raw_phase_names_preserved"] is True
    assert payload["selected_output_policy"]["phase_groups_are_configured_sums"] is True
    assert payload["selected"]["rows"][0]["scalars"]["pH"] == 12.6
    assert payload["selected"]["rows"][0]["scalars"]["porosity"] == 0.23

    table = pd.read_csv(out / "response_summary.csv")
    assert "CNASH__mass" in table.columns
    assert "CNASH__volume" in table.columns
    assert "CNASH__volume_reconstructed" in table.columns
    assert "group__C-A-S-H__mass" in table.columns
    assert "group__Portlandite__mass" in table.columns
    assert "scalar__pH" in table.columns
    assert "scalar__porosity" in table.columns


def test_summarize_forward_result_reports_missing_names(tmp_path):
    run = _write_run(tmp_path)
    out = summarize_forward_result(
        run=run,
        phases=["MissingPhase"],
        phase_groups=["MissingGroup"],
        scalars=["MissingScalar"],
        out=tmp_path / "summary",
    )

    payload = json.loads((out / "response_summary.json").read_text(encoding="utf-8"))
    assert payload["missing"]["phases"] == ["MissingPhase"]
    assert payload["missing"]["phase_groups"] == ["MissingGroup"]
    assert payload["missing"]["scalars"] == ["MissingScalar"]
    assert "CNASH" in payload["available"]["phases"]
    assert "C-A-S-H" in payload["available"]["phase_groups"]
    assert "pH" in payload["available"]["scalars"]


def test_summarize_forward_result_uses_top_phase_default(tmp_path):
    run = _write_run(tmp_path)
    out = summarize_forward_result(run=run, out=tmp_path / "summary", top_phases=1)

    payload = json.loads((out / "response_summary.json").read_text(encoding="utf-8"))
    assert payload["requested"]["phase_request_source"] == "top_nonzero_default"
    assert payload["requested"]["phases"] == ["CNASH"]
    assert payload["requested"]["scalar_request_source"] == "default"
