import json

import pandas as pd

from inverse_gems.design_run_report import run_design_run_report


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_run(tmp_path, name, *, issue=False):
    run_dir = tmp_path / name
    manifest = {
        "name": f"{name}_query",
        "query": f"configs/{name}.yaml",
        "model_table": "data/model_table.csv",
        "model_bundle": "reports/model.joblib",
        "model_registry_entry": {
            "id": f"{name}_model",
            "material_system": "OPC_slag",
            "age_days": 28.0,
            "reaction_model_id": "current",
            "reaction_model_signature": "abc123",
        },
        "reaction_model": {"id": "current", "signature": "abc123"},
        "target_constraints": {"porosity": {"max": 0.4}, "C-A-S-H": {"min": 0.04}},
        "target_availability": {
            "checked": True,
            "policy": "warn",
            "requested_targets": ["porosity", "C-A-S-H"],
            "matched_targets": [
                {
                    "requested_target": "porosity",
                    "target_label": "porosity",
                    "target_column": "y__porosity",
                    "status": "recommended",
                    "r2": 0.9,
                    "range": 0.2,
                    "nonzero_fraction": 1.0,
                    "reasons": "ok",
                },
                {
                    "requested_target": "C-A-S-H",
                    "target_label": "C-A-S-H",
                    "target_column": "y__amount_C_A_S_H",
                    "status": "usable_with_caution" if issue else "recommended",
                    "r2": 0.65 if issue else 0.95,
                    "range": 0.01,
                    "nonzero_fraction": 1.0,
                    "reasons": "low R2" if issue else "ok",
                },
            ],
            "issues": [
                {
                    "target": "C-A-S-H",
                    "severity": "warning",
                    "status": "usable_with_caution",
                    "message": "Target 'C-A-S-H' is usable_with_caution: low R2",
                    "diagnostics": {
                        "target_label": "C-A-S-H",
                        "status": "usable_with_caution",
                        "r2": 0.65,
                        "range": 0.01,
                        "nonzero_fraction": 1.0,
                        "reasons": "low R2",
                    },
                }
            ]
            if issue
            else [],
        },
    }
    search_summary = {"candidate_rows_after_filters": 12}
    validation_summary = {
        "requested_top_k": 3,
        "validation_runs": 3,
        "validation_status_counts": {"complete": 3},
        "target_error_summary": {
            "y__porosity": {
                "source_column": "porosity",
                "validated_vs_pred_count": 3,
                "validated_vs_pred_mae": 0.001,
                "validated_vs_pred_max_abs": 0.002,
            },
            "y__amount_C_A_S_H": {
                "source_column": "phase_amount_group__C-A-S-H",
                "validated_vs_pred_count": 3,
                "validated_vs_pred_mae": 0.0005,
                "validated_vs_pred_max_abs": 0.001,
            },
        },
    }
    selection_summary = {
        "input_rows": 3,
        "candidate_rows_after_constraints": 2,
        "top_k_written": 2,
        "pH_uncertainty_policy": {
            "mode": "penalize" if issue else "ignore",
            "unreliable_rows": 1 if issue else 0,
            "rejected_rows": 0,
            "penalized_rows": 1 if issue else 0,
        },
    }
    _write_json(
        run_dir / "design_query_run_summary.json",
        {
            "query": f"configs/{name}.yaml",
            "compiled_manifest": manifest,
            "candidate_search_summary": search_summary,
            "validation_summary": validation_summary,
            "selection_summary": selection_summary,
        },
    )
    pd.DataFrame(
        [
            {
                "selection_rank": 1,
                "material_system": "OPC_slag",
                "recipe_text": "OPC 60, slag 38, gypsum 2, w/b 0.4, age 28",
                "OPC": 60.0,
                "slag": 38.0,
                "gypsum": 2.0,
                "w_b": 0.4,
                "water_g": 40.0,
                "age_days": 28.0,
                "validated__y__porosity": 0.36,
                "validated__y__amount_C_A_S_H": 0.043,
                "pred__y__porosity": 0.361,
                "diff_validated_minus_pred__y__porosity": -0.001,
                "uncertainty_flags": "",
                "preflight_dir": "preflight/run_a/rank1",
            },
            {
                "selection_rank": 2,
                "material_system": "OPC_slag",
                "recipe_text": "OPC 58, slag 40, gypsum 2, w/b 0.4, age 28",
                "OPC": 58.0,
                "slag": 40.0,
                "gypsum": 2.0,
                "w_b": 0.4,
                "water_g": 40.0,
                "age_days": 28.0,
                "validated__y__porosity": 0.37,
                "validated__y__amount_C_A_S_H": 0.042,
                "pred__y__porosity": 0.372,
                "diff_validated_minus_pred__y__porosity": -0.002,
                "uncertainty_flags": "pH_water_uncertain",
                "preflight_dir": "preflight/run_a/rank2",
            },
        ]
    ).to_csv(run_dir / "final_selected_candidates.csv", index=False)
    return run_dir


def test_design_run_report_writes_summary_files(tmp_path):
    run_a = _make_run(tmp_path, "run_a")
    run_b = _make_run(tmp_path, "run_b", issue=True)

    out = run_design_run_report(runs=[run_a, run_b], out=tmp_path / "report")

    assert (out / "design_run_summary.csv").exists()
    assert (out / "design_run_target_errors.csv").exists()
    assert (out / "design_run_target_availability.csv").exists()
    assert (out / "design_run_target_availability_issues.csv").exists()
    assert (out / "design_run_summary.json").exists()
    assert (out / "design_run_summary.md").exists()

    summary = pd.read_csv(out / "design_run_summary.csv")
    assert len(summary) == 2
    assert set(summary["final_selected_count"]) == {2}
    assert "best_validated__y__porosity" in summary.columns
    assert "best_preflight_dir" in summary.columns
    assert "selection_pH_uncertainty_mode" in summary.columns
    assert summary.loc[summary["run_name"] == "run_b", "selection_pH_penalized_rows"].iloc[0] == 1
    assert summary.loc[summary["run_name"] == "run_b", "target_availability_issue_count"].iloc[0] == 1

    errors = pd.read_csv(out / "design_run_target_errors.csv")
    assert set(errors["target_label"]) == {"porosity", "C-A-S-H"}

    issues = pd.read_csv(out / "design_run_target_availability_issues.csv")
    assert list(issues["target"]) == ["C-A-S-H"]

    payload = json.loads((out / "design_run_summary.json").read_text(encoding="utf-8"))
    assert payload["summary"]["run_count"] == 2
    assert payload["summary"]["runs_with_target_availability_issues"] == 1
