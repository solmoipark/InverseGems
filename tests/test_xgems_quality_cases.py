import json
from pathlib import Path

import pandas as pd

from inverse_gems import cli as cli_module
from inverse_gems.global_chemistry_db import initialize_global_chemistry_db
from inverse_gems.xgems_quality_cases import write_xgems_quality_case_report


def _write_model_table(db: Path) -> Path:
    manifest = initialize_global_chemistry_db(db=db)
    model_table = Path(manifest["paths"]["model_table"])
    model_table.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "meta__recipe_id": "clean",
                "meta__chem_hash": "h1",
                "meta__material_system": "OPC_slag_fly_ash",
                "meta__age_days": 28.0,
                "meta__OPC": 40.0,
                "meta__slag": 30.0,
                "meta__fly_ash": 30.0,
                "meta__w_b": 0.45,
                "meta__water_g": 45.0,
                "meta__xgems_water_g": 45.0,
                "meta__xgems_w_b": 0.45,
                "meta__chemistry_status": "complete",
                "meta__solver_rescued": False,
                "meta__xgems_retry_count": 0,
                "y__pH": 12.8,
                "y__porosity": 0.42,
            },
            {
                "meta__recipe_id": "rescued",
                "meta__chem_hash": "h2",
                "meta__material_system": "OPC_slag_fly_ash",
                "meta__age_days": 1.0,
                "meta__OPC": 35.0,
                "meta__slag": 25.0,
                "meta__fly_ash": 40.0,
                "meta__w_b": 0.53,
                "meta__water_g": 53.0,
                "meta__xgems_water_g": 45.0,
                "meta__xgems_w_b": 0.45,
                "meta__chemistry_status": "complete",
                "meta__solver_rescued": True,
                "meta__xgems_retry_count": 1,
                "y__pH": 12.6,
                "y__porosity": 0.55,
            },
            {
                "meta__recipe_id": "other_system",
                "meta__chem_hash": "h3",
                "meta__material_system": "OPC_only",
                "meta__age_days": 1.0,
                "meta__OPC": 100.0,
                "meta__w_b": 0.40,
                "meta__water_g": 40.0,
                "meta__xgems_water_g": 35.0,
                "meta__chemistry_status": "complete",
                "meta__solver_rescued": True,
                "meta__xgems_retry_count": 1,
            },
        ]
    ).to_csv(model_table, index=False)
    return model_table


def test_xgems_quality_case_report_extracts_filtered_problem_cases(tmp_path):
    db = tmp_path / "db"
    _write_model_table(db)

    out = write_xgems_quality_case_report(
        db=db,
        out=tmp_path / "quality",
        material_system="OPC_slag_fly_ash",
    )

    summary = json.loads((out / "xgems_quality_case_summary.json").read_text(encoding="utf-8"))
    cases = pd.read_csv(out / "xgems_quality_cases.csv")
    by_age = pd.read_csv(out / "xgems_quality_by_age_band.csv")
    by_w_b = pd.read_csv(out / "xgems_quality_by_w_b_band.csv")
    assert summary["source_row_count"] == 2
    assert summary["problem_row_count"] == 1
    assert cases["meta__recipe_id"].tolist() == ["rescued"]
    assert cases.loc[0, "analysis__water_delta_g"] == -8.0
    assert bool(cases.loc[0, "analysis__pH_water_reliable"]) is False
    assert by_age.set_index("analysis__age_band").loc["<=1d", "problem_count"] == 1
    assert by_w_b.set_index("analysis__w_b_band").loc["<=0.55", "problem_count"] == 1
    assert (out / "xgems_quality_case_analysis.md").exists()


def test_analyze_xgems_quality_cases_cli(tmp_path):
    db = tmp_path / "db"
    _write_model_table(db)

    code = cli_module.main(
        [
            "analyze-xgems-quality-cases",
            "--db",
            str(db),
            "--out",
            str(tmp_path / "cli_quality"),
            "--material-system",
            "OPC_slag_fly_ash",
        ]
    )

    assert code == 0
    cases = pd.read_csv(tmp_path / "cli_quality" / "xgems_quality_cases.csv")
    assert cases["meta__recipe_id"].tolist() == ["rescued"]
