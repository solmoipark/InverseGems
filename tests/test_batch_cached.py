import csv
import json

from inverse_gems.batch import run_batch_cached, summarize_batch_status
from inverse_gems.cli import main


def _write_rows(path, rows):
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_batch_cache_diff_age_failure_and_resume(tmp_path):
    csv_path = tmp_path / "recipes.csv"
    rows = [
        {"recipe_id": "a", "OPC": 100, "w_b": 0.4, "age_days": 1, "temperature_celsius": 20},
        {"recipe_id": "b", "OPC": 100, "w_b": 0.4, "age_days": 1, "temperature_celsius": 20},
        {"recipe_id": "c", "OPC": 100, "w_b": 0.4, "age_days": 28, "temperature_celsius": 20},
        {"recipe_id": "bad", "OPC": 100, "age_days": 1, "temperature_celsius": 20},
    ]
    _write_rows(csv_path, rows)
    status = run_batch_cached(recipes_csv=csv_path, db=tmp_path / "db", use_mock=True)
    with status.open(newline="", encoding="utf-8") as handle:
        status_rows = list(csv.DictReader(handle))
    assert len(status_rows) == 4
    assert status_rows[1]["used_cached_xgems"] == "True"
    assert status_rows[0]["chem_hash"] != status_rows[2]["chem_hash"]
    assert status_rows[3]["status"] == "failed"
    run_batch_cached(recipes_csv=csv_path, db=tmp_path / "db", use_mock=True, resume=True)
    with status.open(newline="", encoding="utf-8") as handle:
        resumed = list(csv.DictReader(handle))
    assert len(resumed) >= 4
    progress = json.loads((tmp_path / "db" / "batch_progress.json").read_text(encoding="utf-8"))
    assert progress["recipe_count"] == 4
    assert progress["failed_count"] >= 1
    assert (tmp_path / "db" / "batch_manifest.json").exists()
    assert (tmp_path / "db" / "batch_failures.csv").exists()
    assert (tmp_path / "db" / "batch_summary.md").exists()


def test_batch_failure_records_preflight_directory(tmp_path):
    csv_path = tmp_path / "recipes.csv"
    missing_dat = tmp_path / "missing.dat.lst"
    _write_rows(
        csv_path,
        [{"recipe_id": "dat_missing", "OPC": 100, "w_b": 0.4, "age_days": 28, "temperature_celsius": 20}],
    )

    status = run_batch_cached(recipes_csv=csv_path, db=tmp_path / "db", dat_lst=missing_dat, use_mock=False)
    with status.open(newline="", encoding="utf-8") as handle:
        status_rows = list(csv.DictReader(handle))

    assert status_rows[0]["status"] == "failed"
    assert status_rows[0]["preflight_dir"]
    assert (tmp_path / "db" / "failure_preflight" / "dat_missing" / "xgems_input_preflight.json").exists()


def test_batch_records_non_success_solver_status(tmp_path, monkeypatch):
    csv_path = tmp_path / "recipes.csv"
    _write_rows(
        csv_path,
        [{"recipe_id": "solver_bad", "OPC": 100, "w_b": 0.4, "age_days": 1, "temperature_celsius": 20}],
    )

    def fake_forward(**kwargs):
        return {
            "chem_hash": "abc123",
            "reused_cache": False,
            "chemistry_status": "failed",
            "solver_status": "Failure (no result) in GEM calculation with LPP AIA",
        }

    monkeypatch.setattr("inverse_gems.batch.run_forward_cached", fake_forward)
    status = run_batch_cached(recipes_csv=csv_path, db=tmp_path / "db", use_mock=True)
    with status.open(newline="", encoding="utf-8") as handle:
        status_rows = list(csv.DictReader(handle))

    assert status_rows[0]["status"] == "failed"
    assert "LPP AIA" in status_rows[0]["error_message"]


def test_batch_status_command_writes_summary_to_out(tmp_path):
    csv_path = tmp_path / "recipes.csv"
    _write_rows(
        csv_path,
        [
            {"recipe_id": "a", "OPC": 100, "w_b": 0.4, "age_days": 1, "temperature_celsius": 20},
            {"recipe_id": "b", "OPC": 100, "age_days": 1, "temperature_celsius": 20},
        ],
    )
    db = tmp_path / "db"
    run_batch_cached(recipes_csv=csv_path, db=db, use_mock=True, progress_every=1)

    summary = summarize_batch_status(db=db, out=tmp_path / "batch_report")
    assert summary["has_status_file"] is True
    assert summary["processed_count"] == 2
    assert summary["failed_count"] == 1
    assert (tmp_path / "batch_report" / "batch_progress.json").exists()
    assert (tmp_path / "batch_report" / "batch_failures.csv").exists()
    assert (tmp_path / "batch_report" / "batch_summary.md").exists()

    code = main(["batch-status", "--db", str(db), "--out", str(tmp_path / "batch_report_cli")])
    assert code == 0
    assert (tmp_path / "batch_report_cli" / "batch_progress.json").exists()


def test_batch_resume_summary_counts_only_current_recipe_file(tmp_path):
    first_csv = tmp_path / "first.csv"
    second_csv = tmp_path / "second.csv"
    _write_rows(
        first_csv,
        [
            {"recipe_id": "a", "OPC": 100, "w_b": 0.4, "age_days": 1, "temperature_celsius": 20},
            {"recipe_id": "b", "OPC": 100, "w_b": 0.4, "age_days": 7, "temperature_celsius": 20},
        ],
    )
    _write_rows(
        second_csv,
        [{"recipe_id": "c", "OPC": 100, "w_b": 0.4, "age_days": 28, "temperature_celsius": 20}],
    )
    db = tmp_path / "db"
    run_batch_cached(recipes_csv=first_csv, db=db, use_mock=True)
    run_batch_cached(recipes_csv=second_csv, db=db, use_mock=True, resume=True)

    progress = json.loads((db / "batch_progress.json").read_text(encoding="utf-8"))
    with (db / "batch_status.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert progress["recipe_count"] == 1
    assert progress["processed_count"] == 1
    assert [row["recipe_id"] for row in rows] == ["c"]
