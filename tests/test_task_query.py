import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
import yaml

from inverse_gems.task_query import run_task_query, validate_task_query_data


class ToyTaskQueryEstimator:
    def predict(self, x):
        porosity = 0.20 + 0.002 * x["x__OPC"].to_numpy()
        return pd.DataFrame({"y__porosity": porosity}).to_numpy()


def test_task_query_validation_requires_matching_payload():
    validate_task_query_data(
        {
            "task_type": "forward_time_series",
            "forward_query": {
                "recipe": {"binders": {"OPC": 100}, "w_b": 0.4},
                "age_grid": {"values": [1, 28]},
            },
        }
    )
    with pytest.raises(ValueError, match="forward_query is required"):
        validate_task_query_data({"task_type": "forward_time_series"})


def test_run_task_query_routes_forward_mock(tmp_path):
    query = tmp_path / "task_forward.yaml"
    query.write_text(
        yaml.safe_dump(
            {
                "name": "forward_route",
                "task_type": "forward_time_series",
                "forward_query": {
                    "recipe": {"binders": {"OPC": 40, "slag": 30, "fly_ash": 30}, "w_b": 0.4},
                    "age_grid": {"values": [0.1, 1.0, 28.0]},
                    "plots": [{"kind": "phase_volumes", "names": "all_nonzero", "filename": "volumes.png"}],
                    "response_summary": {
                        "phases": ["Mock C-S-H raw phase"],
                        "scalars": ["pH", "porosity"],
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    out = run_task_query(query=query, out=tmp_path / "out", db=tmp_path / "db", use_mock=True)

    summary = json.loads((out / "task_query_summary.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(out / "forward" / "time_series.csv")
    assert summary["task_type"] == "forward_time_series"
    assert (out / "routed_query" / "forward_query.yaml").exists()
    assert len(frame) == 3
    assert "phase_volume__Mock C-S-H raw phase" in frame.columns
    assert (out / "forward" / "volumes.png").exists()
    assert (out / "forward" / "response_summary.json").exists()
    assert (out / "forward" / "answer.md").exists()
    assert (out / "forward" / "narrative_answer.md").exists()
    assert summary["response_summary"]["json"].endswith("response_summary.json")
    assert summary["response_summary"]["answer"]["markdown"].endswith("answer.md")
    assert summary["response_summary"]["narrative"]["markdown"].endswith("narrative_answer.md")


def test_run_task_query_routes_inverse_design_skip_validation(tmp_path):
    table = tmp_path / "model.csv"
    pd.DataFrame(
        {
            "meta__recipe_id": ["high_opc", "low_opc"],
            "meta__material_system": ["toy", "toy"],
            "meta__age_bin": ["standard", "standard"],
            "x__OPC": [70.0, 30.0],
            "x__slag": [30.0, 70.0],
            "x__w_b": [0.40, 0.40],
            "x__age_days": [28.0, 28.0],
        }
    ).to_csv(table, index=False)
    bundle = tmp_path / "model.joblib"
    joblib.dump(
        {
            "estimator": ToyTaskQueryEstimator(),
            "inputs": ["x__OPC", "x__slag", "x__w_b", "x__age_days"],
            "targets": ["y__porosity"],
        },
        bundle,
    )
    query = tmp_path / "task_inverse.yaml"
    query.write_text(
        yaml.safe_dump(
            {
                "name": "inverse_route",
                "task_type": "inverse_design",
                "design_query": {
                    "name": "toy",
                    "material_system": "toy",
                    "age_days": 28,
                    "age_bin": "standard",
                    "model_table": str(table),
                    "model_bundle": str(bundle),
                    "inputs": {
                        "OPC": {"min": 20, "max": 80},
                        "slag": {"min": 20, "max": 80},
                        "w_b": {"min": 0.30, "max": 0.50},
                    },
                    "targets": {"porosity": {"max": 2.0}},
                    "ranking": {"mode": "lexicographic"},
                    "preferences": [{"input": "OPC", "direction": "minimize"}],
                    "search_top_k": 1,
                    "selection_top_k": 1,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    out = run_task_query(query=query, out=tmp_path / "out_inverse", db=tmp_path / "db", skip_validation=True)

    summary = json.loads((out / "task_query_summary.json").read_text(encoding="utf-8"))
    candidates = pd.read_csv(out / "design" / "final_candidates.csv")
    assert summary["task_type"] == "inverse_design"
    assert (out / "routed_query" / "design_query.yaml").exists()
    assert candidates.iloc[0]["meta__recipe_id"] == "low_opc"


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "data" / "global_chem_db_real_v1_multiage").exists(),
    reason="global chemistry DB artifacts not present (v2 DB rebuild deferred until new SCM kinetics)",
)
def test_run_task_query_global_db_forwards_mock_validation(tmp_path, monkeypatch):
    captured = {}

    def fake_global_design_query(**kwargs):
        captured.update(kwargs)
        out = Path(kwargs["out"])
        out.mkdir(parents=True)
        return out

    monkeypatch.setattr("inverse_gems.global_chemistry_db.run_global_design_query", fake_global_design_query)
    query = tmp_path / "task_inverse_global.yaml"
    query.write_text(
        yaml.safe_dump(
            {
                "name": "inverse_global_route",
                "task_type": "inverse_design",
                "design_query": {
                    "name": "global_toy",
                    "design_space": {
                        "allowed_materials": ["OPC", "metakaolin"],
                        "strict_materials": True,
                        "age_days": 28,
                    },
                    "preferences": [{"target": "C-A-S-H", "direction": "maximize"}],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    out = run_task_query(
        query=query,
        out=tmp_path / "out_inverse_global",
        db=tmp_path / "db",
        global_db=tmp_path / "global_db",
        use_mock=True,
        skip_validation=False,
    )

    assert captured["use_mock_validation"] is True
    assert captured["skip_validation"] is False
    summary = json.loads((out / "task_query_summary.json").read_text(encoding="utf-8"))
    assert summary["use_mock"] is True
