import csv
import json

from inverse_gems.batch import run_batch_cached
from inverse_gems.qc import summarize_database


def test_qc_summary_files_created(tmp_path):
    recipes = tmp_path / "recipes.csv"
    with recipes.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["recipe_id", "OPC", "w_b", "age_days", "age_bin"])
        writer.writeheader()
        writer.writerow({"recipe_id": "a", "OPC": 100, "w_b": 0.4, "age_days": 0.0035, "age_bin": "ultra_early"})
        writer.writerow({"recipe_id": "b", "OPC": 100, "w_b": 0.4, "age_days": 1, "age_bin": "around_1d"})
    db = tmp_path / "db"
    run_batch_cached(recipes_csv=recipes, db=db, use_mock=True)
    out = summarize_database(db=db, out=tmp_path / "summary")
    for name in [
        "db_summary.json",
        "recipe_coverage.csv",
        "chemistry_coverage.csv",
        "age_coverage.csv",
        "age_bin_counts.csv",
        "failed_runs.csv",
        "failed_recipe_runs.csv",
        "duplicate_chem_hashes.csv",
        "porosity_outliers.csv",
        "missing_selected_outputs.csv",
    ]:
        assert (out / name).exists()
    summary = json.loads((out / "db_summary.json").read_text(encoding="utf-8"))
    assert summary["number_of_recipe_runs"] == 2
    assert summary["age_bin_counts"]
