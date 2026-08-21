import json
from pathlib import Path

import pandas as pd
import yaml

from inverse_gems.forward_query import expand_age_grid, run_forward_query, validate_forward_query_data
from inverse_gems.forward_query_diagnostics import write_forward_query_diagnostics


def _write_query(path: Path) -> Path:
    data = {
        "name": "mock_volume_vs_time",
        "recipe": {
            "binders": {"OPC": 40, "slag": 30, "fly_ash": 30},
            "w_b": 0.4,
        },
        "age_grid": {"start": 0.1, "stop": 10.0, "points": 3, "spacing": "log"},
        "outputs": {
            "phase_masses": "all",
            "phase_volumes": "all",
            "phase_volumes_reconstructed": "all",
            "aqueous_species": "all",
            "scalars": ["pH", "system_volume", "system_mass"],
        },
        "plots": [
            {
                "kind": "phase_volumes",
                "names": "all_nonzero",
                "top_n": 5,
                "filename": "phase_volumes_vs_age.png",
            }
        ],
        "response_summary": {
            "phases": ["Mock C-S-H raw phase", "Mock Portlandite"],
            "scalars": ["pH", "porosity"],
            "table_limit": 10,
        },
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _write_single_query(path: Path) -> Path:
    data = {
        "name": "mock_single_forward",
        "task": "forward_calculation",
        "recipe": {
            "binders": {"OPC": 60, "slag": 40},
            "w_b": 0.45,
        },
        "age_grid": {"values": [28.0]},
        "outputs": {
            "phase_masses": "all",
            "phase_volumes": "all",
            "phase_volumes_reconstructed": "all",
            "aqueous_species": "all",
            "scalars": "all",
        },
        "plots": [],
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_forward_query_validation_and_age_grid():
    data = {
        "recipe": {"binders": {"OPC": 40, "BFS": 30, "FA": 30}, "w_b": 0.4},
        "age_grid": {"values": [0.1, 1.0, 28.0]},
    }
    validated = validate_forward_query_data(data)
    ages = expand_age_grid(validated["age_grid"])
    assert ages == [0.1, 1.0, 28.0]
    assert validated["recipe"]["binders"]["BFS"] == 30


def test_run_forward_query_mock_writes_time_series_and_plot(tmp_path):
    query = _write_query(tmp_path / "forward_query.yaml")
    out = run_forward_query(query=query, out=tmp_path / "out", db=tmp_path / "db", use_mock=True)

    frame = pd.read_csv(out / "time_series.csv")
    summary = json.loads((out / "forward_query_summary.json").read_text(encoding="utf-8"))

    assert len(frame) == 3
    assert summary["completed_count"] == 3
    assert "phase_volume__Mock C-S-H raw phase" in frame.columns
    assert "phase_mass__Mock Ettringite/AFt" in frame.columns
    assert "scalar__pH" in frame.columns
    assert "preflight_dir" in frame.columns
    assert "uncertainty_flags" in frame.columns
    assert frame["pH_water_reliable"].astype(str).str.lower().eq("true").all()
    assert (out / "phase_volumes_vs_age.png").exists()
    assert (out / "forward_query_used.yaml").exists()
    assert (out / "forward_query_manifest.json").exists()
    assert (out / "forward_query_status.csv").exists()
    assert (out / "failed_ages.csv").exists()
    assert (out / "phase_nonzero_summary.csv").exists()
    assert (out / "phase_change_summary.csv").exists()
    assert (out / "scalar_timeseries.csv").exists()
    assert (out / "diagnostics.md").exists()
    assert (out / "response_summary.json").exists()
    assert (out / "response_summary.md").exists()
    assert (out / "response_summary.csv").exists()
    assert (out / "answer.json").exists()
    assert (out / "answer.md").exists()
    assert (out / "narrative_answer.json").exists()
    assert (out / "narrative_answer.md").exists()

    diagnostics = summary["diagnostics"]
    assert diagnostics["row_count"] == 3
    assert diagnostics["completed_count"] == 3
    assert diagnostics["failed_count"] == 0
    nonzero = pd.read_csv(out / "phase_nonzero_summary.csv")
    assert "Mock C-S-H raw phase" in set(nonzero["raw_name"])
    scalar = pd.read_csv(out / "scalar_timeseries.csv")
    assert "scalar__pH" in scalar.columns
    response = json.loads((out / "response_summary.json").read_text(encoding="utf-8"))
    assert response["requested"]["phases"] == ["Mock C-S-H raw phase", "Mock Portlandite"]
    assert response["selected"]["rows"][0]["scalars"]["pH"] == 12.6
    assert response["uncertainty"]["preflight_available_count"] == 3
    assert summary["response_summary"]["json"].endswith("response_summary.json")
    assert summary["response_summary"]["answer"]["markdown"].endswith("answer.md")
    assert summary["response_summary"]["narrative"]["markdown"].endswith("narrative_answer.md")


def test_run_forward_query_single_age_writes_single_result_artifacts(tmp_path):
    query = _write_single_query(tmp_path / "single_forward_query.yaml")
    out = run_forward_query(query=query, out=tmp_path / "single_out", db=tmp_path / "db", use_mock=True)

    summary = json.loads((out / "forward_query_summary.json").read_text(encoding="utf-8"))
    single = json.loads((out / "single_result.json").read_text(encoding="utf-8"))
    masses = pd.read_csv(out / "raw_phase_masses.csv")
    volumes = pd.read_csv(out / "raw_phase_volumes.csv")

    assert summary["age_count"] == 1
    assert summary["single_result"]["chemistry_status"] == "complete"
    assert single["metadata"]["age_days"] == 28.0
    assert single["binder_masses_g"]["OPC"] == 60.0
    assert "Mock C-S-H raw phase" in set(masses["name"])
    assert "Mock Portlandite" in set(volumes["name"])
    assert (out / "single_result.csv").exists()
    assert (out / "raw_phase_volumes_reconstructed.csv").exists()
    assert (out / "raw_aqueous_species.csv").exists()
    assert (out / "raw_scalars.json").exists()
    assert (out / "calculation_summary.md").exists()
    assert (out / "response_summary.json").exists()
    assert (out / "answer.md").exists()
    assert (out / "narrative_answer.md").exists()


def test_forward_query_diagnostics_records_failed_ages(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "query_run_id": "q1",
                "row_index": 1,
                "age_days": 1.0,
                "recipe_id": "r1",
                "chemistry_status": "complete",
                "solver_status": "ok",
                "phase_volume__A raw": 1.0,
                "scalar__pH": 12.5,
            },
            {
                "query_run_id": "q1",
                "row_index": 2,
                "age_days": 28.0,
                "recipe_id": "r2",
                "chemistry_status": "failed",
                "solver_status": "bad",
                "error_message": "solver failed",
                "phase_volume__A raw": 0.0,
                "scalar__pH": 0.0,
            },
        ]
    )

    summary = write_forward_query_diagnostics(frame, tmp_path)

    failed = pd.read_csv(tmp_path / "failed_ages.csv")
    changes = pd.read_csv(tmp_path / "phase_change_summary.csv")
    assert summary["failed_count"] == 1
    assert failed.iloc[0]["age_days"] == 28.0
    assert failed.iloc[0]["error_message"] == "solver failed"
    assert changes.iloc[0]["raw_name"] == "A raw"
