import json

import numpy as np
import pandas as pd
import yaml

from inverse_gems.surrogate import train_baseline_surrogate
from inverse_gems.utils import write_json


def test_train_baseline_surrogate_writes_metrics_and_uses_group_split(tmp_path):
    rows = []
    for base in range(12):
        for age in [1.0, 7.0, 28.0]:
            opc = 100.0 - base
            slag = float(base)
            rows.append(
                {
                    "meta__recipe_id": f"base_{base:06d}_age_{age:g}",
                    "meta__template_name": "synthetic",
                    "meta__prepared_id": f"prepared_{base:06d}_{age:g}",
                    "meta__reaction_model_id": "params_a",
                    "meta__reaction_model_signature": "sig_a",
                    "x__OPC": opc,
                    "x__slag": slag,
                    "x__w_b": 0.4 + base * 0.001,
                    "x__age_days": age,
                    "x__log_age_days": np.log10(age),
                    "y__porosity": 0.3 + 0.001 * slag + 0.0001 * age,
                    "y__amount_C_A_S_H": 0.01 + 0.0002 * slag + 0.00005 * age,
                    "y__amount_sparse_phase": 0.001 if base in {0, 1} and age == 28.0 else 0.0,
                }
            )
    table = tmp_path / "model.csv"
    pd.DataFrame(rows).to_csv(table, index=False)
    write_json(
        table.with_suffix(table.suffix + ".schema.json"),
        {
            "roles": {
                "inputs": [{"column": c} for c in ["x__OPC", "x__slag", "x__w_b", "x__age_days", "x__log_age_days"]],
                "targets": [
                    {"column": c}
                    for c in ["y__porosity", "y__amount_C_A_S_H", "y__amount_sparse_phase"]
                ],
                "metadata": [
                    {"column": "meta__recipe_id"},
                    {"column": "meta__reaction_model_id"},
                    {"column": "meta__reaction_model_signature"},
                ],
            }
        },
    )
    config = tmp_path / "surrogate.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "kind": "ExtraTreesRegressor",
                    "n_estimators": 20,
                    "random_state": 7,
                    "n_jobs": 1,
                    "min_samples_leaf": 1,
                },
                "split": {
                    "strategy": "group_shuffle",
                    "group_column": "meta__recipe_id",
                    "group_regex": r"^(base_\d+)",
                    "test_size": 0.25,
                    "random_state": 7,
                },
                "evaluation": {
                    "nonzero_threshold": 1.0e-12,
                    "sparse_target_fraction_threshold": 0.10,
                    "min_test_nonzero_count": 3,
                    "save_predictions": True,
                    "permutation_importance": {"enabled": False},
                },
            }
        ),
        encoding="utf-8",
    )

    out = train_baseline_surrogate(model_table=table, config=config, out=tmp_path / "surrogate")

    assert (out / "target_metrics.csv").exists()
    assert (out / "test_predictions.csv").exists()
    assert (out / "model.joblib").exists()
    summary = json.loads((out / "surrogate_summary.json").read_text(encoding="utf-8"))
    assert summary["split"]["group_overlap_count"] == 0
    assert summary["train_rows"] + summary["test_rows"] == len(rows)
    assert summary["reaction_provenance"]["reaction_model_signatures"] == ["sig_a"]
    manifest = json.loads((out / "surrogate_model_manifest.json").read_text(encoding="utf-8"))
    assert manifest["reaction_provenance"]["reaction_model_ids"] == ["params_a"]
    metrics = pd.read_csv(out / "target_metrics.csv")
    assert set(metrics["target"]) == {"y__porosity", "y__amount_C_A_S_H", "y__amount_sparse_phase"}
    sparse = metrics.set_index("target").loc["y__amount_sparse_phase"]
    assert sparse["full_nonzero_count"] == 2
    assert bool(sparse["sparse_target"])
    assert bool(sparse["test_nonzero_too_low"])
    assert sparse["evaluation_reliability"] in {"unstable_no_test_nonzero", "unstable_low_test_nonzero"}


def test_train_baseline_surrogate_can_use_sparse_aware_group_split(tmp_path):
    rows = []
    for base in range(20):
        for age in [1.0, 7.0]:
            rows.append(
                {
                    "meta__recipe_id": f"mix_{base:06d}_age_{age:g}",
                    "meta__reaction_model_id": "params_a",
                    "meta__reaction_model_signature": "sig_a",
                    "x__OPC": 100.0 - base,
                    "x__slag": float(base),
                    "x__w_b": 0.40,
                    "x__age_days": age,
                    "x__log_age_days": np.log10(age),
                    "y__porosity": 0.30 + 0.001 * base,
                    "y__amount_sparse_phase": 0.001 if base < 8 else 0.0,
                }
            )
    table = tmp_path / "sparse_model.csv"
    pd.DataFrame(rows).to_csv(table, index=False)
    write_json(
        table.with_suffix(table.suffix + ".schema.json"),
        {
            "roles": {
                "inputs": [{"column": c} for c in ["x__OPC", "x__slag", "x__w_b", "x__age_days", "x__log_age_days"]],
                "targets": [{"column": c} for c in ["y__porosity", "y__amount_sparse_phase"]],
                "metadata": [{"column": "meta__recipe_id"}],
            }
        },
    )
    config = tmp_path / "surrogate_sparse.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "kind": "ExtraTreesRegressor",
                    "n_estimators": 20,
                    "random_state": 11,
                    "n_jobs": 1,
                    "min_samples_leaf": 1,
                },
                "split": {
                    "strategy": "group_shuffle",
                    "group_column": "meta__recipe_id",
                    "group_regex": r"^(mix_\d+)",
                    "test_size": 0.25,
                    "random_state": 11,
                    "sparse_target_support": {
                        "enabled": True,
                        "candidate_splits": 50,
                        "sparse_target_fraction_threshold": 0.50,
                        "min_test_nonzero_count": 4,
                    },
                },
                "evaluation": {
                    "nonzero_threshold": 1.0e-12,
                    "sparse_target_fraction_threshold": 0.50,
                    "min_test_nonzero_count": 4,
                    "save_predictions": True,
                    "permutation_importance": {"enabled": False},
                },
            }
        ),
        encoding="utf-8",
    )

    out = train_baseline_surrogate(model_table=table, config=config, out=tmp_path / "surrogate_sparse")

    summary = json.loads((out / "surrogate_summary.json").read_text(encoding="utf-8"))
    sparse_report = summary["split"]["sparse_target_support"]
    assert sparse_report["enabled"] is True
    by_target = {row["target"]: row for row in sparse_report["targets"]}
    assert by_target["y__amount_sparse_phase"]["test_nonzero_count"] >= 4
    metrics = pd.read_csv(out / "target_metrics.csv").set_index("target")
    assert metrics.loc["y__amount_sparse_phase", "nonzero_true_count"] >= 4
    assert not bool(metrics.loc["y__amount_sparse_phase", "test_nonzero_too_low"])
