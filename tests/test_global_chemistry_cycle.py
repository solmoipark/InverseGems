import json

import pandas as pd

from inverse_gems.cli import main
from inverse_gems.database import InverseGemsDatabase
from inverse_gems.global_chemistry_cycle import run_global_chemistry_acquisition_cycle


def _candidate_row(recipe_id: str, chem_hash: str, opc: float, slag: float, age: float) -> dict:
    row = {
        "meta__recipe_id": recipe_id,
        "meta__chem_hash": chem_hash,
        "meta__template_name": "cycle_test",
        "meta__material_system": "OPC_slag",
        "meta__target_profile": "cycle",
        "x__OPC": opc,
        "x__slag": slag,
        "x__fly_ash": 0.0,
        "x__metakaolin": 0.0,
        "x__silica_fume": 0.0,
        "x__limestone": 0.0,
        "x__gypsum": 0.0,
        "x__w_b": 0.4,
        "x__water_g": 40.0,
        "x__age_days": age,
        "x__temperature_celsius": 20.0,
        "x__xgems_water_g": 40.0,
    }
    for oxide in ["CaO", "SiO2", "Al2O3", "Fe2O3", "MgO", "SO3", "Na2O", "K2O", "CO2", "H2O"]:
        row[f"x__chem_oxide_equiv_mol_{oxide}"] = 0.0
    row["x__chem_oxide_equiv_mol_CaO"] = opc / 100.0
    row["x__chem_oxide_equiv_mol_SiO2"] = slag / 100.0
    return row


def test_global_acquisition_cycle_runs_mock_batch_and_writes_summary(tmp_path):
    candidate_table = tmp_path / "candidate_table.csv"
    pd.DataFrame(
        [
            _candidate_row("cycle_1", "hash_1", 70.0, 30.0, 28.0),
            _candidate_row("cycle_2", "hash_2", 55.0, 45.0, 90.0),
        ]
    ).to_csv(candidate_table, index=False)

    out = run_global_chemistry_acquisition_cycle(
        db=tmp_path / "global_db",
        out=tmp_path / "cycle",
        candidate_table=candidate_table,
        max_candidates=1,
        use_mock=True,
        train_surrogate=False,
        refresh=False,
        coverage=True,
    )

    summary = json.loads((out / "global_acquisition_cycle_summary.json").read_text(encoding="utf-8"))
    stages = {stage["name"]: stage for stage in summary["stages"]}
    assert summary["status"] == "complete"
    assert stages["acquire_candidates"]["status"] == "complete"
    assert stages["run_batch_cached"]["status"] == "complete"
    assert stages["train_global_chemistry_surrogate"]["status"] == "skipped"
    assert stages["global_chemistry_coverage"]["status"] == "complete"
    assert (out / "acquisition" / "acquisition_candidates.md").exists()
    assert (out / "batch" / "batch_status.csv").exists()
    assert (out / "batch" / "batch_progress.json").exists()
    assert (out / "coverage" / "global_chemistry_coverage_summary.json").exists()
    assert (out / "global_acquisition_cycle_summary.md").exists()

    selected = pd.read_csv(out / "acquisition" / "acquisition_candidates.csv")
    assert selected.iloc[0]["acquisition_bucket"] == "novel_in_domain"
    assert "new reactive chemistry" in selected.iloc[0]["acquisition_reason"]

    db = InverseGemsDatabase(tmp_path / "global_db")
    assert db.get_recipe_run("cycle_1") is not None


def test_global_acquisition_cycle_cli_runs_mock_cycle(tmp_path):
    candidate_table = tmp_path / "candidate_table.csv"
    pd.DataFrame([_candidate_row("cli_cycle_1", "cli_hash_1", 65.0, 35.0, 56.0)]).to_csv(candidate_table, index=False)

    exit_code = main(
        [
            "run-global-acquisition-cycle",
            "--db",
            str(tmp_path / "global_db_cli"),
            "--out",
            str(tmp_path / "cycle_cli"),
            "--candidate-table",
            str(candidate_table),
            "--max-candidates",
            "1",
            "--mock",
            "--skip-refresh",
            "--skip-train",
        ]
    )

    assert exit_code == 0
    summary = json.loads((tmp_path / "cycle_cli" / "global_acquisition_cycle_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    assert (tmp_path / "cycle_cli" / "batch" / "batch_status.csv").exists()
    assert (tmp_path / "cycle_cli" / "batch" / "batch_progress.json").exists()
