import json
from pathlib import Path

import pandas as pd
import yaml

from inverse_gems.global_chemistry_coverage import write_global_chemistry_coverage_report
from inverse_gems.global_chemistry_db import initialize_global_chemistry_db


def test_global_chemistry_coverage_reports_missing_systems_ages_and_target_metrics(tmp_path):
    db = tmp_path / "db"
    manifest = initialize_global_chemistry_db(db=db)
    model_table = Path(manifest["paths"]["model_table"])
    model_table.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "meta__recipe_id": ["r1", "r2"],
            "meta__chem_hash": ["h1", "h2"],
            "meta__material_system": ["OPC_slag", "OPC_slag"],
            "meta__reaction_model_id": ["local_default_parameters", "local_default_parameters"],
            "meta__reaction_model_signature": ["sig", "sig"],
            "meta__chemistry_status": ["complete", "complete"],
            "meta__solver_rescued": [False, True],
            "meta__xgems_retry_count": [0, 1],
            "meta__water_g": [40.0, 45.0],
            "meta__xgems_water_g": [40.0, 30.0],
            "meta__preflight_dir": ["preflight/a", "preflight/b"],
            "x__OPC": [60.0, 50.0],
            "x__slag": [40.0, 50.0],
            "x__w_b": [0.40, 0.45],
            "x__age_days": [28.0, 56.0],
            "x__xgems_water_g": [40.0, 45.0],
            "x__temperature_celsius": [20.0, 20.0],
            "x__chem_oxide_equiv_mol_CaO": [1.0, 1.2],
            "x__chem_oxide_equiv_mol_SiO2": [0.5, 0.7],
            "pH_water_reliable": [True, False],
        }
    ).to_csv(model_table, index=False)
    metrics = Path(manifest["paths"]["model_bundle"])
    metrics.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "target": ["y__porosity", "y__pH"],
            "r2": [0.91, 0.12],
            "mae": [0.01, 0.2],
            "full_nonzero_count": [2, 2],
            "nonzero_true_count": [1, 0],
            "evaluation_warning": [False, True],
            "evaluation_reliability": ["ok", "unstable_no_test_nonzero"],
            "diagnostic_warning": ["", "test split contains no nonzero examples"],
        }
    ).to_csv(metrics.parent / "target_metrics.csv", index=False)
    coverage_config = tmp_path / "coverage.yaml"
    coverage_config.write_text(
        yaml.safe_dump(
            {
                "material_systems": ["OPC_slag", "OPC_fly_ash"],
                "thresholds": {
                    "min_rows_per_material_system": 2,
                    "min_unique_chem_hashes_per_material_system": 2,
                    "min_total_rows": 2,
                    "min_surrogate_r2_recommended": 0.70,
                    "max_pH_unreliable_fraction": 0.20,
                },
                "age_milestones_days": [28, 90],
                "age_tolerance_fraction": 0.05,
                "age_tolerance_min_days": 0.01,
            }
        ),
        encoding="utf-8",
    )

    out = write_global_chemistry_coverage_report(db=db, out=tmp_path / "coverage", coverage_config=coverage_config)

    summary = json.loads((out / "global_chemistry_coverage_summary.json").read_text(encoding="utf-8"))
    material = pd.read_csv(out / "material_system_coverage.csv")
    age = pd.read_csv(out / "age_coverage.csv")
    target_metrics = pd.read_csv(out / "target_metrics_coverage.csv")
    xgems_quality = pd.read_csv(out / "xgems_quality_by_material_system.csv")
    assert summary["row_count"] == 2
    assert summary["unique_chem_hash_count"] == 2
    assert summary["w_b_range"]["column"] == "x__w_b"
    assert summary["w_b_range"]["min"] == 0.40
    assert summary["ready_for_inverse_design"] is False
    assert summary["missing_material_systems"] == ["OPC_fly_ash"]
    assert 90.0 in summary["missing_age_milestones_days"]
    assert summary["xgems_health"]["pH_water_reliability_status"] == "warning"
    assert summary["xgems_health"]["solver_rescued_count"] == 1
    assert summary["xgems_health"]["xgems_water_adjusted_count"] == 1
    assert material.set_index("material_system").loc["OPC_slag", "coverage_status"] == "ok"
    assert material.set_index("material_system").loc["OPC_fly_ash", "coverage_status"] == "missing"
    assert age.set_index("age_days").loc[28.0, "coverage_status"] == "ok"
    assert target_metrics.set_index("target").loc["y__pH", "coverage_status"] == "warning"
    assert "coverage_reasons" in target_metrics.columns
    assert "no nonzero examples" in target_metrics.set_index("target").loc["y__pH", "coverage_reasons"]
    quality_all = xgems_quality.set_index(["group", "group_value"]).loc[("all", "all")]
    assert quality_all["solver_rescued_count"] == 1
    assert quality_all["xgems_water_adjusted_count"] == 1
    assert quality_all["preflight_available_count"] == 2
    assert quality_all["quality_status"] == "warning"
    assert (out / "global_chemistry_coverage.md").exists()
