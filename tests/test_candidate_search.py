import json

import joblib
import pandas as pd
import pytest
import yaml

from inverse_gems.candidate_search import run_surrogate_candidate_search


class LinearCandidateEstimator:
    def predict(self, x):
        porosity = 0.60 - 0.004 * x["x__OPC"].to_numpy() + 0.10 * x["x__w_b"].to_numpy()
        csh = 0.001 * x["x__OPC"].to_numpy() + 0.0001 * x["x__age_days"].to_numpy()
        return pd.DataFrame({"y__porosity": porosity, "y__amount_C_A_S_H": csh}).to_numpy()


class TieCandidateEstimator:
    def predict(self, x):
        return pd.DataFrame(
            {
                "y__porosity": [0.30, 0.25, 0.20],
                "y__amount_C_A_S_H": [0.0502, 0.0501, 0.040],
            }
        ).to_numpy()


def test_surrogate_candidate_search_filters_and_ranks(tmp_path):
    table = tmp_path / "model.csv"
    pd.DataFrame(
        {
            "meta__recipe_id": ["a", "b", "c", "d"],
            "meta__age_bin": ["standard", "standard", "long_term", "early"],
            "x__OPC": [30.0, 50.0, 70.0, 40.0],
            "x__w_b": [0.40, 0.45, 0.50, 0.35],
            "x__age_days": [28.0, 90.0, 365.0, 7.0],
            "y__porosity": [0.50, 0.44, 0.40, 0.48],
            "y__amount_C_A_S_H": [0.03, 0.05, 0.08, 0.04],
        }
    ).to_csv(table, index=False)
    bundle = tmp_path / "model.joblib"
    joblib.dump(
        {
            "estimator": LinearCandidateEstimator(),
            "inputs": ["x__OPC", "x__w_b", "x__age_days"],
            "targets": ["y__porosity", "y__amount_C_A_S_H"],
        },
        bundle,
    )
    query = tmp_path / "query.yaml"
    query.write_text(
        yaml.safe_dump(
            {
                "model_table": str(table),
                "model_bundle": str(bundle),
                "constraints": {
                    "metadata": {"age_bin": {"include": ["standard", "long_term"]}},
                    "inputs": {"OPC": {"min": 20, "max": 60}},
                    "predicted_targets": {"porosity": {"max": 0.45}},
                },
                "objectives": {"minimize": {"OPC": 0.1, "porosity": 1.0}, "maximize": {"C-A-S-H": 1.0}},
                "top_k": 2,
            }
        ),
        encoding="utf-8",
    )

    out = run_surrogate_candidate_search(query=query, out=tmp_path / "out")

    candidates = pd.read_csv(out / "candidates.csv")
    assert len(candidates) == 1
    assert candidates.iloc[0]["meta__recipe_id"] == "b"
    assert "pred__y__porosity" in candidates.columns
    assert "pred__y__amount_C_A_S_H" in candidates.columns
    summary = json.loads((out / "candidate_search_summary.json").read_text(encoding="utf-8"))
    assert summary["input_rows"] == 4
    assert summary["candidate_rows_after_filters"] == 1
    assert summary["top_k_written"] == 1


def test_surrogate_candidate_search_lexicographic_preferences_use_input_order(tmp_path):
    table = tmp_path / "model.csv"
    pd.DataFrame(
        {
            "meta__recipe_id": ["high_opc", "low_opc", "lower_csh"],
            "x__OPC": [70.0, 40.0, 20.0],
            "x__w_b": [0.40, 0.40, 0.40],
            "x__age_days": [28.0, 28.0, 28.0],
        }
    ).to_csv(table, index=False)
    bundle = tmp_path / "model.joblib"
    joblib.dump(
        {
            "estimator": TieCandidateEstimator(),
            "inputs": ["x__OPC", "x__w_b", "x__age_days"],
            "targets": ["y__porosity", "y__amount_C_A_S_H"],
        },
        bundle,
    )
    query = tmp_path / "query.yaml"
    query.write_text(
        yaml.safe_dump(
            {
                "model_table": str(table),
                "model_bundle": str(bundle),
                "ranking": {"mode": "lexicographic"},
                "preferences": [
                    {"target": "C-A-S-H", "direction": "maximize", "tolerance": 0.001},
                    {"input": "OPC", "direction": "minimize"},
                ],
                "top_k": 3,
            }
        ),
        encoding="utf-8",
    )

    out = run_surrogate_candidate_search(query=query, out=tmp_path / "out")

    candidates = pd.read_csv(out / "candidates.csv")
    assert candidates["meta__recipe_id"].tolist() == ["low_opc", "high_opc", "lower_csh"]
    assert "rank_component__01__maximize__C_A_S_H" in candidates.columns
    assert "rank_component__02__minimize__OPC" in candidates.columns


def test_surrogate_candidate_search_rejects_reaction_signature_mismatch(tmp_path):
    table = tmp_path / "model.csv"
    pd.DataFrame(
        {
            "meta__recipe_id": ["a"],
            "meta__reaction_model_id": ["params_a"],
            "meta__reaction_model_signature": ["sig_a"],
            "x__OPC": [60.0],
            "x__w_b": [0.40],
            "x__age_days": [28.0],
        }
    ).to_csv(table, index=False)
    bundle = tmp_path / "model.joblib"
    joblib.dump(
        {
            "estimator": LinearCandidateEstimator(),
            "inputs": ["x__OPC", "x__w_b", "x__age_days"],
            "targets": ["y__porosity", "y__amount_C_A_S_H"],
            "reaction_provenance": {
                "reaction_model_ids": ["params_b"],
                "reaction_model_signatures": ["sig_b"],
            },
        },
        bundle,
    )
    query = tmp_path / "query.yaml"
    query.write_text(
        yaml.safe_dump({"model_table": str(table), "model_bundle": str(bundle), "top_k": 1}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Reaction-model provenance mismatch"):
        run_surrogate_candidate_search(query=query, out=tmp_path / "out")

    report = json.loads((tmp_path / "out" / "reaction_provenance_report.json").read_text(encoding="utf-8"))
    assert report["compatibility"]["errors"]


def test_surrogate_candidate_search_treats_absent_zero_fixed_input_as_satisfied(tmp_path):
    table = tmp_path / "model.csv"
    pd.DataFrame(
        {
            "meta__recipe_id": ["a", "b"],
            "x__OPC": [40.0, 60.0],
            "x__w_b": [0.40, 0.40],
            "x__age_days": [28.0, 28.0],
        }
    ).to_csv(table, index=False)
    bundle = tmp_path / "model.joblib"
    joblib.dump(
        {
            "estimator": LinearCandidateEstimator(),
            "inputs": ["x__OPC", "x__w_b", "x__age_days"],
            "targets": ["y__porosity", "y__amount_C_A_S_H"],
        },
        bundle,
    )
    query = tmp_path / "query.yaml"
    query.write_text(
        yaml.safe_dump(
            {
                "model_table": str(table),
                "model_bundle": str(bundle),
                "constraints": {"inputs": {"fly_ash": {"max": 0.0}, "OPC": {"max": 50.0}}},
                "top_k": 5,
            }
        ),
        encoding="utf-8",
    )

    out = run_surrogate_candidate_search(query=query, out=tmp_path / "out")

    candidates = pd.read_csv(out / "candidates.csv")
    assert candidates["meta__recipe_id"].tolist() == ["a"]
    summary = json.loads((out / "candidate_search_summary.json").read_text(encoding="utf-8"))
    absent = [item for item in summary["resolved_constraints"] if item["name"] == "fly_ash"][0]
    assert absent["column"] == "<absent input assumed zero>"
    assert absent["rejected"] == 0
