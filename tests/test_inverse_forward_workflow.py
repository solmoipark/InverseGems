import json
from pathlib import Path

import pandas as pd
import yaml

from inverse_gems import cli as cli_module
from inverse_gems.inverse_forward_workflow import run_inverse_forward_workflow


def test_inverse_forward_workflow_writes_forward_query_and_selected_timeseries(tmp_path, monkeypatch):
    def fake_inverse(**kwargs):
        out = Path(kwargs["out"])
        design = out / "task_run" / "design"
        design.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "review_rank": 1,
                    "material_system": "OPC_metakaolin",
                    "validation_status": "complete",
                    "solver_status": "ok",
                    "solver_rescued": False,
                    "xgems_retry_count": 0,
                    "uncertainty_flags": "validated",
                    "OPC": 70.0,
                    "metakaolin": 30.0,
                    "w_b": 0.45,
                    "age_days": 28.0,
                    "validated__amount_C_A_S_H": 0.04,
                }
            ]
        ).to_csv(design / "candidate_review.csv", index=False)
        return out

    def fake_forward(**kwargs):
        out = Path(kwargs["out"])
        db = Path(kwargs["db"])
        out.mkdir(parents=True)
        recipe_dir = db / "recipe_runs" / "r1"
        recipe_dir.mkdir(parents=True)
        (recipe_dir / "unreacted_masses.json").write_text(
            json.dumps({"OPC": 5.0, "metakaolin": 12.0}),
            encoding="utf-8",
        )
        pd.DataFrame(
            [
                {
                    "recipe_id": "r1",
                    "age_days": 28.0,
                    "chemistry_status": "complete",
                    "solver_status": "ok",
                    "solver_rescued": False,
                    "w_b": 0.45,
                    "water_g": 45.0,
                    "xgems_w_b": 0.45,
                    "xgems_water_g": 45.0,
                    "preflight_dir": str(db / "chemistry_runs" / "abc" / "xgems_preflight" / "primary"),
                    "phase_mass__CNASH": 0.04,
                    "phase_volume_reconstructed__CNASH": 0.000015,
                    "phase_mass__Portlandite": 0.001,
                    "scalar__pH": 13.0,
                    "porosity": 0.42,
                }
            ]
        ).to_csv(out / "time_series.csv", index=False)
        (out / "forward_query_summary.json").write_text(
            json.dumps({"age_count": 1, "completed_count": 1, "failed_count": 0}),
            encoding="utf-8",
        )
        return out

    monkeypatch.setattr("inverse_gems.inverse_forward_workflow.run_user_request_with_openai", fake_inverse)
    monkeypatch.setattr("inverse_gems.inverse_forward_workflow.run_forward_query", fake_forward)

    out = run_inverse_forward_workflow(
        inverse_request="OPC and MK only, 28 d, maximize C-A-S-H",
        out=tmp_path / "workflow",
        db=tmp_path / "db",
        use_mock=True,
        forward_age_values=[28.0],
    )

    forward_query = yaml.safe_load((out / "forward_top_candidate_query.yaml").read_text(encoding="utf-8"))
    assert forward_query["recipe"]["binders"] == {"OPC": 70.0, "metakaolin": 30.0}
    assert forward_query["age_grid"]["values"] == [28.0]
    selected = pd.read_csv(out / "workflow_summary" / "forward_top_candidate_selected_timeseries.csv")
    assert selected.loc[0, "unreacted_metakaolin_g"] == 12.0
    assert selected.loc[0, "CNASH_mass_g"] == 40.0
    assert bool(selected.loc[0, "pH_water_reliable"]) is True
    assert selected.loc[0, "xgems_water_delta_g"] == 0.0
    assert pd.isna(selected.loc[0, "uncertainty_flags"]) or selected.loc[0, "uncertainty_flags"] == ""
    summary = json.loads((out / "workflow_summary" / "workflow_summary.json").read_text(encoding="utf-8"))
    assert summary["top_candidate"]["binders"]["metakaolin"] == 30.0
    assert summary["forward"]["completed_count"] == 1
    assert summary["forward"]["pH_water_unreliable_count"] == 0
    assert summary["forward"]["preflight_available_count"] == 1
    markdown = (out / "workflow_summary" / "workflow_summary.md").read_text(encoding="utf-8")
    assert "| 28" in markdown
    assert "| nan |" not in markdown


def test_inverse_forward_workflow_summary_flags_rescued_forward_pH(tmp_path, monkeypatch):
    def fake_inverse(**kwargs):
        out = Path(kwargs["out"])
        design = out / "task_run" / "design"
        design.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "review_rank": 1,
                    "material_system": "OPC_fly_ash",
                    "validation_status": "complete",
                    "solver_status": "ok",
                    "solver_rescued": False,
                    "xgems_retry_count": 0,
                    "OPC": 35.0,
                    "fly_ash": 65.0,
                    "w_b": 0.45,
                    "age_days": 100.0,
                    "validated__amount_Portlandite": 0.001,
                }
            ]
        ).to_csv(design / "candidate_review.csv", index=False)
        return out

    def fake_forward(**kwargs):
        out = Path(kwargs["out"])
        db = Path(kwargs["db"])
        out.mkdir(parents=True)
        recipe_dir = db / "recipe_runs" / "r1"
        recipe_dir.mkdir(parents=True)
        (recipe_dir / "unreacted_masses.json").write_text(
            json.dumps({"OPC": 4.0, "fly_ash": 40.0}),
            encoding="utf-8",
        )
        preflight = db / "chemistry_runs" / "rescued" / "xgems_preflight" / "retry_1"
        preflight.mkdir(parents=True)
        (preflight / "xgems_input_preflight.json").write_text(
            json.dumps({"ok": True, "water": {"matches_recipe_water": False}}),
            encoding="utf-8",
        )
        pd.DataFrame(
            [
                {
                    "recipe_id": "r1",
                    "age_days": 100.0,
                    "chemistry_status": "complete",
                    "solver_status": "ok",
                    "solver_rescued": True,
                    "w_b": 0.45,
                    "water_g": 45.0,
                    "xgems_w_b": 0.30,
                    "xgems_water_g": 30.0,
                    "preflight_dir": str(preflight),
                    "phase_mass__CNASH": 0.03,
                    "phase_mass__Portlandite": 0.0005,
                    "scalar__pH": 12.4,
                    "porosity": 0.39,
                }
            ]
        ).to_csv(out / "time_series.csv", index=False)
        (out / "forward_query_summary.json").write_text(
            json.dumps({"age_count": 1, "completed_count": 1, "failed_count": 0}),
            encoding="utf-8",
        )
        return out

    monkeypatch.setattr("inverse_gems.inverse_forward_workflow.run_user_request_with_openai", fake_inverse)
    monkeypatch.setattr("inverse_gems.inverse_forward_workflow.run_forward_query", fake_forward)

    out = run_inverse_forward_workflow(
        inverse_request="OPC and fly ash only, 100 d, minimize Portlandite and porosity",
        out=tmp_path / "workflow",
        db=tmp_path / "db",
        use_mock=True,
        forward_age_values=[100.0],
    )

    selected = pd.read_csv(out / "workflow_summary" / "forward_top_candidate_selected_timeseries.csv")
    flags = selected.loc[0, "uncertainty_flags"]
    assert "solver_rescued" in flags
    assert "xgems_water_adjusted" in flags
    assert "pH_water_uncertain" in flags
    assert bool(selected.loc[0, "pH_water_reliable"]) is False
    assert selected.loc[0, "xgems_water_delta_g"] == -15.0

    summary = json.loads((out / "workflow_summary" / "workflow_summary.json").read_text(encoding="utf-8"))
    assert summary["forward"]["pH_water_unreliable_count"] == 1
    assert summary["forward"]["uncertainty_flag_counts"]["pH_water_uncertain"] == 1
    assert summary["forward"]["preflight_available_count"] == 1
    markdown = (out / "workflow_summary" / "workflow_summary.md").read_text(encoding="utf-8")
    assert "pH water-unreliable ages" in markdown
    assert "pH_water_uncertain" in markdown


def test_inverse_forward_workflow_cli_forwards_arguments(tmp_path, monkeypatch):
    captured = {}

    def fake_workflow(**kwargs):
        captured.update(kwargs)
        out = Path(kwargs["out"])
        out.mkdir(parents=True)
        return out

    monkeypatch.setattr(cli_module, "run_inverse_forward_workflow", fake_workflow)

    code = cli_module.main(
        [
            "run-inverse-forward-workflow-mock",
            "--inverse-request",
            "OPC and MK only, maximize C-A-S-H at 28 d",
            "--out",
            str(tmp_path / "workflow_cli"),
            "--db",
            str(tmp_path / "db"),
            "--global-db",
            str(tmp_path / "global"),
            "--forward-age-values",
            "28",
            "--strict-materials",
            "--no-plots",
        ]
    )

    assert code == 0
    assert captured["inverse_request"] == "OPC and MK only, maximize C-A-S-H at 28 d"
    assert captured["global_db"] == tmp_path / "global"
    assert captured["forward_age_values"] == [28.0]
    assert captured["strict_materials"] is True
    assert captured["use_mock"] is True
