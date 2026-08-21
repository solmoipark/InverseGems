import json

import joblib
import pandas as pd
import yaml

from inverse_gems.design_query_runner import run_design_query


class ToyDesignQueryEstimator:
    def predict(self, x):
        porosity = 0.20 + 0.002 * x["x__OPC"].to_numpy()
        return pd.DataFrame({"y__porosity": porosity}).to_numpy()


def _write_model_bundle(tmp_path):
    table = tmp_path / "model.csv"
    pd.DataFrame(
        {
            "meta__recipe_id": ["high_opc", "low_opc", "mid_opc"],
            "meta__material_system": ["toy", "toy", "toy"],
            "meta__age_bin": ["standard", "standard", "standard"],
            "x__OPC": [70.0, 30.0, 50.0],
            "x__slag": [30.0, 70.0, 50.0],
            "x__w_b": [0.40, 0.40, 0.40],
            "x__age_days": [28.0, 28.0, 28.0],
        }
    ).to_csv(table, index=False)
    bundle = tmp_path / "model.joblib"
    joblib.dump(
        {
            "estimator": ToyDesignQueryEstimator(),
            "inputs": ["x__OPC", "x__slag", "x__w_b", "x__age_days"],
            "targets": ["y__porosity"],
        },
        bundle,
    )
    return table, bundle


def _write_design_query(tmp_path, table=None, bundle=None):
    query = tmp_path / "design_query.yaml"
    data = {
        "name": "toy_query",
        "material_system": "toy",
        "age_days": 28,
        "age_bin": "standard",
        "inputs": {
            "OPC": {"min": 20, "max": 80},
            "slag": {"min": 20, "max": 80},
            "w_b": {"min": 0.30, "max": 0.50},
        },
        "targets": {"porosity": {"max": 2.0}},
        "ranking": {"mode": "lexicographic"},
        "preferences": [{"input": "OPC", "direction": "minimize"}],
        "search_top_k": 2,
        "selection_top_k": 1,
    }
    if table is not None:
        data["model_table"] = str(table)
    if bundle is not None:
        data["model_bundle"] = str(bundle)
    query.write_text(yaml.safe_dump(data), encoding="utf-8")
    return query


def _write_model_registry(tmp_path, table, bundle):
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "id": "toy_age28",
                        "material_system": "toy",
                        "age_days": 28.0,
                        "model_table": str(table),
                        "model_bundle": str(bundle),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return registry


def test_run_design_query_search_only_writes_final_candidates(tmp_path):
    table, bundle = _write_model_bundle(tmp_path)
    query = _write_design_query(tmp_path, table, bundle)

    out = run_design_query(query=query, out=tmp_path / "run", skip_validation=True)

    final_candidates = pd.read_csv(out / "final_candidates.csv")
    summary = json.loads((out / "design_query_run_summary.json").read_text(encoding="utf-8"))
    review = pd.read_csv(out / "candidate_review.csv")
    assert len(final_candidates) == 2
    assert final_candidates.iloc[0]["meta__recipe_id"] == "low_opc"
    assert review.iloc[0]["source_recipe_id"] == "low_opc"
    assert "surrogate_only" in review.iloc[0]["uncertainty_flags"]
    assert summary["final_stage"] == "surrogate_search"
    assert summary["candidate_review"]["outputs"]["csv"].endswith("candidate_review.csv")
    assert summary["inverse_design_flow_summary"]["stages"][-1]["name"] == "candidate_review"
    assert summary["inverse_design_flow_summary"]["stages"][2]["status"] == "skipped"
    assert (out / "inverse_design_flow_summary.json").exists()
    assert (out / "inverse_design_flow_summary.md").exists()
    assert (out / "compiled_query" / "surrogate_candidate_search.yaml").exists()
    assert (out / "surrogate_search" / "candidate_search_summary.json").exists()


def test_run_design_query_resolves_model_from_registry(tmp_path):
    table, bundle = _write_model_bundle(tmp_path)
    query = _write_design_query(tmp_path)
    registry = _write_model_registry(tmp_path, table, bundle)

    out = run_design_query(query=query, out=tmp_path / "run", skip_validation=True, model_registry=registry)

    summary = json.loads((out / "design_query_run_summary.json").read_text(encoding="utf-8"))
    assert summary["compiled_manifest"]["model_registry_entry"]["id"] == "toy_age28"
    assert pd.read_csv(out / "final_candidates.csv").iloc[0]["meta__recipe_id"] == "low_opc"


def test_run_design_query_mock_validation_writes_final_selection(tmp_path):
    table, bundle = _write_model_bundle(tmp_path)
    query = _write_design_query(tmp_path, table, bundle)

    out = run_design_query(
        query=query,
        out=tmp_path / "run",
        db=tmp_path / "db",
        use_mock_validation=True,
    )

    selected = pd.read_csv(out / "final_selected_candidates.csv")
    review = pd.read_csv(out / "candidate_review.csv")
    validation_runs = pd.read_csv(out / "validation" / "validation_runs.csv")
    summary = json.loads((out / "design_query_run_summary.json").read_text(encoding="utf-8"))
    assert len(selected) == 1
    assert selected.iloc[0]["source_recipe_id"] == "low_opc"
    assert selected.iloc[0]["material_system"] == "toy"
    assert review.iloc[0]["source_recipe_id"] == "low_opc"
    assert review.iloc[0]["stage"] == "selection"
    assert validation_runs.iloc[0]["source_material_system"] == "toy"
    assert summary["final_stage"] == "selection"
    assert summary["validation_summary"]["validation_status_counts"]["complete"] == 1
    stages = {stage["name"]: stage for stage in summary["inverse_design_flow_summary"]["stages"]}
    assert stages["xgems_validation"]["status"] == "complete"
    assert stages["final_selection"]["rows_out"] == 1
