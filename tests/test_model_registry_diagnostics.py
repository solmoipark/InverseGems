import json

import pandas as pd

from inverse_gems.model_registry_diagnostics import run_model_registry_diagnostics


def test_model_registry_diagnostics_flags_target_availability(tmp_path):
    model_table = tmp_path / "model_table.csv"
    pd.DataFrame(
        {
            "x__OPC": [30.0, 50.0, 70.0],
            "y__amount_C_A_S_H": [0.02, 0.04, 0.06],
            "y__pH": [12.65, 12.651, 12.652],
            "y__amount_hemicarbonate": [0.0, 0.0, 0.0],
        }
    ).to_csv(model_table, index=False)
    schema = {
        "filters": {"rows_after_status_filter": 3},
        "roles": {
            "targets": [
                {
                    "column": "y__amount_C_A_S_H",
                    "kind": "phase_amount_group",
                    "name": "C_A_S_H",
                    "source": "phase_amount_group__C_A_S_H",
                },
                {"column": "y__pH", "kind": "scalar", "name": "pH", "source": "scalar__pH"},
                {
                    "column": "y__amount_hemicarbonate",
                    "kind": "phase_amount_group",
                    "name": "hemicarbonate",
                    "source": "phase_amount_group__hemicarbonate",
                },
            ]
        },
        "target_summary": {
            "y__amount_C_A_S_H": {
                "count": 3,
                "min": 0.02,
                "max": 0.06,
                "mean": 0.04,
                "median": 0.04,
                "nonzero_count": 3,
                "nonzero_fraction": 1.0,
            },
            "y__pH": {
                "count": 3,
                "min": 12.65,
                "max": 12.652,
                "mean": 12.651,
                "median": 12.651,
                "nonzero_count": 3,
                "nonzero_fraction": 1.0,
            },
            "y__amount_hemicarbonate": {
                "count": 3,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "nonzero_count": 0,
                "nonzero_fraction": 0.0,
            },
        },
    }
    (tmp_path / "model_table.csv.schema.json").write_text(json.dumps(schema), encoding="utf-8")

    bundle_dir = tmp_path / "surrogate"
    bundle_dir.mkdir()
    (bundle_dir / "model.joblib").write_text("placeholder", encoding="utf-8")
    pd.DataFrame(
        {
            "target": ["y__amount_C_A_S_H", "y__pH", "y__amount_hemicarbonate"],
            "n_test": [1, 1, 1],
            "r2": [0.95, 0.99, 1.0],
            "mae": [0.001, 0.0001, 0.0],
            "rmse": [0.001, 0.0001, 0.0],
            "baseline_mean_mae": [0.01, 0.001, 0.0],
            "baseline_mean_rmse": [0.01, 0.001, 0.0],
            "nonzero_true_count": [1, 1, 0],
        }
    ).to_csv(bundle_dir / "target_metrics.csv", index=False)

    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "\n".join(
            [
                "models:",
                "  - id: test_model",
                "    material_system: OPC_test",
                "    age_days: 28.0",
                "    reaction_model_id: current",
                "    reaction_model_signature: abc123",
                f"    model_table: {model_table.as_posix()}",
                f"    model_bundle: {(bundle_dir / 'model.joblib').as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    out = run_model_registry_diagnostics(
        model_registry=registry,
        out=tmp_path / "diagnostics",
        reaction_model_id="current",
        reaction_model_signature="abc123",
    )

    result = pd.read_csv(out / "model_registry_diagnostics.csv")
    by_target = {row["target_label"]: row for _, row in result.iterrows()}
    assert by_target["C_A_S_H"]["status"] == "recommended"
    assert by_target["pH"]["status"] == "not_recommended"
    assert bool(by_target["pH"]["near_constant"])
    assert by_target["hemicarbonate"]["status"] == "not_recommended"
    assert bool(by_target["hemicarbonate"]["all_zero"])
    assert (out / "model_registry_diagnostics.md").exists()


def test_model_registry_diagnostics_filters_reaction_signature(tmp_path):
    table = tmp_path / "table.csv"
    pd.DataFrame({"y__porosity": [0.3, 0.4]}).to_csv(table, index=False)
    (tmp_path / "table.csv.schema.json").write_text(
        json.dumps(
            {
                "roles": {"targets": [{"column": "y__porosity", "kind": "scalar", "name": "porosity"}]},
                "target_summary": {
                    "y__porosity": {
                        "count": 2,
                        "min": 0.3,
                        "max": 0.4,
                        "nonzero_count": 2,
                        "nonzero_fraction": 1.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.joblib").write_text("placeholder", encoding="utf-8")
    pd.DataFrame({"target": ["y__porosity"], "r2": [0.9], "mae": [0.01]}).to_csv(
        bundle / "target_metrics.csv",
        index=False,
    )
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "\n".join(
            [
                "models:",
                "  - id: current_model",
                "    material_system: OPC_only",
                "    age_days: 28.0",
                "    reaction_model_id: current",
                "    reaction_model_signature: keep",
                f"    model_table: {table.as_posix()}",
                f"    model_bundle: {(bundle / 'model.joblib').as_posix()}",
                "  - id: old_model",
                "    material_system: OPC_only",
                "    age_days: 28.0",
                "    reaction_model_id: old",
                "    reaction_model_signature: drop",
                f"    model_table: {table.as_posix()}",
                f"    model_bundle: {(bundle / 'model.joblib').as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    out = run_model_registry_diagnostics(
        model_registry=registry,
        out=tmp_path / "diag",
        reaction_model_signature="keep",
    )

    result = pd.read_csv(out / "model_registry_diagnostics.csv")
    assert set(result["registry_id"]) == {"current_model"}


def test_model_registry_diagnostics_marks_ph_with_water_adjusted_rows_as_caution(tmp_path):
    model_table = tmp_path / "model_table.csv"
    pd.DataFrame({"x__OPC": [40.0, 60.0, 80.0], "y__pH": [12.7, 13.0, 13.3]}).to_csv(model_table, index=False)
    (tmp_path / "model_table.csv.schema.json").write_text(
        json.dumps(
            {
                "filters": {"rows_after_status_filter": 3},
                "roles": {"targets": [{"column": "y__pH", "kind": "scalar", "name": "pH", "source": "scalar__pH"}]},
                "target_summary": {
                    "y__pH": {
                        "count": 3,
                        "min": 12.7,
                        "max": 13.3,
                        "mean": 13.0,
                        "median": 13.0,
                        "nonzero_count": 3,
                        "nonzero_fraction": 1.0,
                        "pH_water_unreliable_count": 1,
                        "pH_water_unreliable_fraction": 1.0 / 3.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    bundle_dir = tmp_path / "surrogate"
    bundle_dir.mkdir()
    (bundle_dir / "model.joblib").write_text("placeholder", encoding="utf-8")
    pd.DataFrame({"target": ["y__pH"], "r2": [0.95], "mae": [0.01]}).to_csv(
        bundle_dir / "target_metrics.csv",
        index=False,
    )
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "\n".join(
            [
                "models:",
                "  - id: test_model",
                "    material_system: OPC_test",
                "    age_days: 28.0",
                "    reaction_model_id: current",
                "    reaction_model_signature: abc123",
                f"    model_table: {model_table.as_posix()}",
                f"    model_bundle: {(bundle_dir / 'model.joblib').as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    out = run_model_registry_diagnostics(
        model_registry=registry,
        out=tmp_path / "diagnostics",
        reaction_model_id="current",
        reaction_model_signature="abc123",
    )

    row = pd.read_csv(out / "model_registry_diagnostics.csv").iloc[0]
    assert row["status"] == "usable_with_caution"
    assert bool(row["pH_water_adjusted"])
    assert "water-adjusted" in row["reasons"]


def test_model_registry_diagnostics_downgrades_unstable_sparse_evaluation(tmp_path):
    model_table = tmp_path / "model_table.csv"
    pd.DataFrame(
        {
            "x__OPC": [40.0, 50.0, 60.0, 70.0],
            "y__amount_rare": [0.0, 0.0, 0.001, 0.0],
        }
    ).to_csv(model_table, index=False)
    (tmp_path / "model_table.csv.schema.json").write_text(
        json.dumps(
            {
                "filters": {"rows_after_status_filter": 4},
                "roles": {
                    "targets": [
                        {
                            "column": "y__amount_rare",
                            "kind": "phase_amount_group",
                            "name": "rare",
                            "source": "phase_amount_group__rare",
                        }
                    ]
                },
                "target_summary": {
                    "y__amount_rare": {
                        "count": 4,
                        "min": 0.0,
                        "max": 0.001,
                        "mean": 0.00025,
                        "median": 0.0,
                        "nonzero_count": 1,
                        "nonzero_fraction": 0.25,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    bundle_dir = tmp_path / "surrogate"
    bundle_dir.mkdir()
    (bundle_dir / "model.joblib").write_text("placeholder", encoding="utf-8")
    pd.DataFrame(
        {
            "target": ["y__amount_rare"],
            "n_total": [4],
            "n_train": [3],
            "n_test": [1],
            "r2": [0.95],
            "mae": [0.0001],
            "full_nonzero_count": [1],
            "full_nonzero_fraction": [0.25],
            "nonzero_true_count": [0],
            "test_nonzero_fraction": [0.0],
            "test_nonzero_missing": [True],
            "test_nonzero_too_low": [True],
            "evaluation_warning": [True],
            "evaluation_reliability": ["unstable_no_test_nonzero"],
            "diagnostic_warning": ["test split contains no nonzero examples"],
        }
    ).to_csv(bundle_dir / "target_metrics.csv", index=False)
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "\n".join(
            [
                "models:",
                "  - id: sparse_eval_model",
                "    material_system: OPC_test",
                "    age_days: 28.0",
                f"    model_table: {model_table.as_posix()}",
                f"    model_bundle: {(bundle_dir / 'model.joblib').as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    out = run_model_registry_diagnostics(model_registry=registry, out=tmp_path / "diagnostics")

    row = pd.read_csv(out / "model_registry_diagnostics.csv").iloc[0]
    assert row["status"] == "usable_with_caution"
    assert bool(row["evaluation_warning"])
    assert bool(row["test_nonzero_missing"])
    assert row["evaluation_reliability"] == "unstable_no_test_nonzero"
    assert "no nonzero examples" in row["reasons"]
