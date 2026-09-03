from __future__ import annotations

import contextlib
import csv
import io
import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from .database import InverseGemsDatabase, copy_raw_outputs_to_chemistry_root, read_name_value_csv, utc_now_iso
from .materials import load_materials
from .phase_volume_reconstruction import write_phase_volume_reconstruction_files
from .porosity import compute_initial_volume_cm3, compute_porosity, load_porosity_config
from .recipe import Recipe
from .utils import write_json
from .xgems_runner import XGEMSRunner


RunnerFactory = Callable[..., Any]

RECONSTRUCTION_FILES = [
    "xgems_phase_volumes_reconstructed.csv",
    "xgems_phase_volume_reconstruction_report.csv",
    "xgems_phase_volume_reconstruction_summary.json",
]

CHEMISTRY_STATUS_FIELDS = [
    "chem_hash",
    "status",
    "message",
    "elapsed_seconds",
    "xgems_run_dir",
    "solver_status",
    "phase_count",
    "reconstructed_nonzero_count",
    "raw_volume_zero_but_reconstructed_nonzero_count",
]

POROSITY_STATUS_FIELDS = [
    "recipe_id",
    "chem_hash",
    "status",
    "message",
    "old_porosity",
    "new_porosity",
    "old_final_solid_volume_cm3",
    "new_final_solid_volume_cm3",
]


def _status_is_complete(status: Any) -> bool:
    text = str(status).lower()
    return bool(text) and "fail" not in text and "error" not in text and "bad" not in text


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _append_status_row(path: Path, row: dict[str, Any], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field) for field in fields})


def _write_text_if_nonempty(path: Path, text: str) -> None:
    if text:
        path.write_text(text, encoding="utf-8")


def _raw_dir_for_chemistry(database: InverseGemsDatabase, chem_row: dict[str, Any]) -> Path:
    stored = chem_row.get("xgems_run_dir")
    candidates: list[Path] = []
    if stored:
        candidate = Path(str(stored))
        candidates.append(candidate)
        if not candidate.is_absolute():
            candidates.append(Path.cwd() / candidate)
    candidates.append(database.chemistry_runs_dir / str(chem_row["chem_hash"]) / "xgems_raw")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _has_complete_reconstruction_files(raw_dir: Path) -> bool:
    if not all((raw_dir / name).exists() for name in RECONSTRUCTION_FILES):
        return False
    summary = _load_json(raw_dir / "xgems_phase_volume_reconstruction_summary.json", None)
    return isinstance(summary, dict) and "reconstructed_nonzero_count" in summary


def _recipe_from_dict(data: dict[str, Any]) -> Recipe:
    return Recipe(
        raw_text=str(data.get("raw_text") or ""),
        binder_masses_g={str(key): float(value) for key, value in (data.get("binder_masses_g") or {}).items()},
        age_days=float(data.get("age_days")),
        water_g=float(data.get("water_g")),
        water_mode=str(data.get("water_mode")),
        w_b=None if data.get("w_b") is None else float(data.get("w_b")),
        basis_g=float(data.get("basis_g", 100.0)),
        warnings=list(data.get("warnings") or []),
        metadata=dict(data.get("metadata") or {}),
    )


def _linked_complete_chemistry_rows(
    database: InverseGemsDatabase,
    *,
    include_unlinked_complete: bool,
) -> list[dict[str, Any]]:
    with database.connect() as conn:
        if include_unlinked_complete:
            rows = conn.execute(
                """
                SELECT *
                FROM chemistry_runs
                WHERE status = 'complete'
                ORDER BY created_at, chem_hash
                """
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT DISTINCT c.*
                FROM recipe_runs r
                JOIN chemistry_runs c ON r.chem_hash = c.chem_hash
                WHERE c.status = 'complete'
                ORDER BY c.created_at, c.chem_hash
                """
            ).fetchall()
    return [dict(row) for row in rows]


def create_sqlite_backup(db: str | Path, *, suffix: str = ".before_reconstructed_volume_backfill.sqlite") -> Path | None:
    database = InverseGemsDatabase(db)
    source = database.sqlite_path
    if not source.exists():
        return None
    backup = source.with_name(source.stem + suffix)
    if not backup.exists():
        shutil.copy2(source, backup)
    return backup


def backfill_reconstructed_phase_volumes(
    *,
    db: str | Path,
    dat_lst: str | Path,
    limit: int | None = None,
    force: bool = False,
    include_unlinked_complete: bool = False,
    status_csv: str | Path | None = None,
    summary_json: str | Path | None = None,
    gems_class_path: str = "xgems:ChemicalEngineDicts",
    xgems_input_mode: str = "formula",
    add_o2: bool = True,
    reuse_runner: bool = True,
    progress_every: int = 100,
    runner_factory: RunnerFactory | None = None,
) -> dict[str, Any]:
    database = InverseGemsDatabase(db)
    rows = _linked_complete_chemistry_rows(database, include_unlinked_complete=include_unlinked_complete)
    if limit is not None:
        rows = rows[: int(limit)]
    status_path = Path(status_csv) if status_csv else database.db_dir / "backfill_reconstructed_volumes_status.csv"
    summary_path = Path(summary_json) if summary_json else database.db_dir / "backfill_reconstructed_volumes_summary.json"

    summary = {
        "started_at": utc_now_iso(),
        "db": str(database.db_dir),
        "dat_lst": str(dat_lst),
        "target_chemistry_runs": len(rows),
        "processed": 0,
        "skipped_existing": 0,
        "complete": 0,
        "failed": 0,
        "missing_input": 0,
        "forced": force,
        "include_unlinked_complete": include_unlinked_complete,
        "reuse_runner": reuse_runner and runner_factory is None,
    }
    runner_cache: dict[float, XGEMSRunner] = {}

    def get_runner(temperature_celsius: float) -> Any:
        if runner_factory is not None:
            return runner_factory(dat_lst_path=dat_lst, temperature_celsius=temperature_celsius)
        if not reuse_runner:
            return XGEMSRunner(
                dat_lst,
                temperature_celsius=temperature_celsius,
                gems_class_path=gems_class_path,
                input_mode=xgems_input_mode,
                add_o2=add_o2,
            )
        key = float(temperature_celsius)
        runner = runner_cache.get(key)
        if runner is None:
            runner = XGEMSRunner(
                dat_lst,
                temperature_celsius=temperature_celsius,
                gems_class_path=gems_class_path,
                input_mode=xgems_input_mode,
                add_o2=add_o2,
            )
            runner_cache[key] = runner
        elif hasattr(runner.gems, "clear"):
            runner.gems.clear()
            runner._composition_cleared = True
        else:
            runner = XGEMSRunner(
                dat_lst,
                temperature_celsius=temperature_celsius,
                gems_class_path=gems_class_path,
                input_mode=xgems_input_mode,
                add_o2=add_o2,
            )
            runner_cache[key] = runner
        return runner

    for index, row in enumerate(rows, start=1):
        start = time.perf_counter()
        chem_hash = str(row["chem_hash"])
        raw_dir = _raw_dir_for_chemistry(database, row)
        reconstructed_path = raw_dir / "xgems_phase_volumes_reconstructed.csv"
        if _has_complete_reconstruction_files(raw_dir) and not force:
            status_row = {
                "chem_hash": chem_hash,
                "status": "skipped_existing",
                "message": "complete reconstructed volume files already exist",
                "elapsed_seconds": 0.0,
                "xgems_run_dir": str(raw_dir),
            }
            _append_status_row(status_path, status_row, CHEMISTRY_STATUS_FIELDS)
            summary["skipped_existing"] += 1
            summary["processed"] += 1
            if progress_every and index % progress_every == 0:
                print(f"backfill chemistry {index}/{len(rows)}: skipped existing")
            continue

        input_path = raw_dir / "input_xgems_species_amounts.json"
        if not input_path.exists():
            input_path = database.chemistry_runs_dir / chem_hash / "input_xgems_species_amounts.json"
        species_amounts = _load_json(input_path, None)
        if not isinstance(species_amounts, dict):
            status_row = {
                "chem_hash": chem_hash,
                "status": "missing_input",
                "message": f"missing or invalid input_xgems_species_amounts.json at {input_path}",
                "elapsed_seconds": time.perf_counter() - start,
                "xgems_run_dir": str(raw_dir),
            }
            _append_status_row(status_path, status_row, CHEMISTRY_STATUS_FIELDS)
            summary["missing_input"] += 1
            summary["processed"] += 1
            continue

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        try:
            runner = get_runner(float(row.get("temperature_celsius") or 20.0))
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                runner.add_species_amounts({str(k): float(v) for k, v in species_amounts.items()}, units="kg")
                solver_status = runner.equilibrate()
                raw_state = runner.capture_raw_state()
            raw_state["solver_status"] = solver_status
            if not _status_is_complete(solver_status):
                raise RuntimeError(f"xGEMS/GEMS reported non-success solver status: {solver_status}")

            raw_dir.mkdir(parents=True, exist_ok=True)
            write_phase_volume_reconstruction_files(raw_dir, raw_state)
            _write_text_if_nonempty(raw_dir / "xgems_reconstruction_backfill_stdout.txt", stdout_buffer.getvalue())
            _write_text_if_nonempty(raw_dir / "xgems_reconstruction_backfill_stderr.txt", stderr_buffer.getvalue())
            copy_raw_outputs_to_chemistry_root(database.chemistry_runs_dir / chem_hash, raw_dir)

            reconstruction_summary = raw_state.get("phase_volume_reconstruction_summary") or {}
            status_row = {
                "chem_hash": chem_hash,
                "status": "complete",
                "message": "",
                "elapsed_seconds": time.perf_counter() - start,
                "xgems_run_dir": str(raw_dir),
                "solver_status": solver_status,
                "phase_count": reconstruction_summary.get("phase_count"),
                "reconstructed_nonzero_count": reconstruction_summary.get("reconstructed_nonzero_count"),
                "raw_volume_zero_but_reconstructed_nonzero_count": reconstruction_summary.get(
                    "raw_volume_zero_but_reconstructed_nonzero_count"
                ),
            }
            summary["complete"] += 1
        except Exception as exc:
            raw_dir.mkdir(parents=True, exist_ok=True)
            _write_text_if_nonempty(raw_dir / "xgems_reconstruction_backfill_stdout.txt", stdout_buffer.getvalue())
            _write_text_if_nonempty(raw_dir / "xgems_reconstruction_backfill_stderr.txt", stderr_buffer.getvalue())
            status_row = {
                "chem_hash": chem_hash,
                "status": "failed",
                "message": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": time.perf_counter() - start,
                "xgems_run_dir": str(raw_dir),
            }
            summary["failed"] += 1
        _append_status_row(status_path, status_row, CHEMISTRY_STATUS_FIELDS)
        summary["processed"] += 1
        if progress_every and (index % progress_every == 0 or status_row["status"] == "failed"):
            print(f"backfill chemistry {index}/{len(rows)}: {status_row['status']} {chem_hash[:10]}")

    summary["finished_at"] = utc_now_iso()
    write_json(summary_path, summary)
    return summary


def recompute_recipe_porosity_from_backfilled_volumes(
    *,
    db: str | Path,
    limit: int | None = None,
    require_reconstructed: bool = True,
    xgems_phase_volume_unit: str = "m3",
    status_csv: str | Path | None = None,
    summary_json: str | Path | None = None,
    progress_every: int = 1000,
    materials_config: str | Path | None = None,
) -> dict[str, Any]:
    database = InverseGemsDatabase(db)
    materials = load_materials(materials_config)
    porosity_config = load_porosity_config()
    porosity_config["xgems_run_mode"] = "reacted_only"
    porosity_config["xgems_phase_volume_unit"] = xgems_phase_volume_unit
    status_path = Path(status_csv) if status_csv else database.db_dir / "backfill_porosity_status.csv"
    summary_path = Path(summary_json) if summary_json else database.db_dir / "backfill_porosity_summary.json"

    with database.connect() as conn:
        rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT r.*, c.status AS chemistry_status, c.xgems_run_dir AS chemistry_xgems_run_dir
                FROM recipe_runs r
                LEFT JOIN chemistry_runs c ON r.chem_hash = c.chem_hash
                ORDER BY r.created_at, r.recipe_id
                """
            ).fetchall()
        ]
    if limit is not None:
        rows = rows[: int(limit)]

    summary = {
        "started_at": utc_now_iso(),
        "db": str(database.db_dir),
        "target_recipe_runs": len(rows),
        "updated": 0,
        "skipped_missing_reconstructed": 0,
        "skipped_noncomplete_chemistry": 0,
        "failed": 0,
        "xgems_phase_volume_unit": xgems_phase_volume_unit,
        "require_reconstructed": require_reconstructed,
    }
    updates: list[tuple[Any, ...]] = []
    warnings_updates: list[tuple[str, str]] = []

    for index, row in enumerate(rows, start=1):
        recipe_id = str(row["recipe_id"])
        chem_hash = str(row["chem_hash"])
        old_porosity = row.get("porosity")
        old_final = row.get("final_solid_volume_cm3")
        try:
            if row.get("chemistry_status") != "complete":
                status_row = {
                    "recipe_id": recipe_id,
                    "chem_hash": chem_hash,
                    "status": "skipped_noncomplete_chemistry",
                    "message": f"chemistry status is {row.get('chemistry_status')}",
                    "old_porosity": old_porosity,
                    "old_final_solid_volume_cm3": old_final,
                }
                summary["skipped_noncomplete_chemistry"] += 1
                _append_status_row(status_path, status_row, POROSITY_STATUS_FIELDS)
                continue
            raw_dir = Path(str(row.get("chemistry_xgems_run_dir") or ""))
            if not raw_dir.exists():
                raw_dir = database.chemistry_runs_dir / chem_hash / "xgems_raw"
            reconstructed_path = raw_dir / "xgems_phase_volumes_reconstructed.csv"
            if require_reconstructed and not reconstructed_path.exists():
                status_row = {
                    "recipe_id": recipe_id,
                    "chem_hash": chem_hash,
                    "status": "skipped_missing_reconstructed",
                    "message": f"missing {reconstructed_path}",
                    "old_porosity": old_porosity,
                    "old_final_solid_volume_cm3": old_final,
                }
                summary["skipped_missing_reconstructed"] += 1
                _append_status_row(status_path, status_row, POROSITY_STATUS_FIELDS)
                continue

            recipe = _recipe_from_dict(_load_json_from_text(row.get("recipe_json"), {}))
            unreacted_masses = _load_json_from_text(row.get("unreacted_masses_json"), {})
            porosity = compute_porosity(
                recipe,
                materials=materials,
                xgems_phase_volumes=read_name_value_csv(raw_dir / "xgems_phase_volumes_raw.csv"),
                xgems_phase_volumes_reconstructed=read_name_value_csv(reconstructed_path),
                unreacted_masses_g={str(k): float(v) for k, v in (unreacted_masses or {}).items()},
                config=porosity_config,
            )
            initial_volume = porosity.get("initial_volume_cm3")
            final_volume = porosity.get("solid_final_volume_cm3")
            new_porosity = porosity.get("porosity_best_effort")
            existing_warnings = _load_json_from_text(row.get("warnings_json"), {})
            if not isinstance(existing_warnings, dict):
                existing_warnings = {}
            existing_warnings["porosity"] = porosity.get("warnings", [])
            existing_warnings["porosity_backfill"] = {
                "updated_at": utc_now_iso(),
                "used_reconstructed_phase_volumes": True,
                "xgems_phase_volume_unit": xgems_phase_volume_unit,
            }
            updates.append((initial_volume, final_volume, new_porosity, json.dumps(existing_warnings, sort_keys=True), recipe_id))
            warnings_updates.append((recipe_id, json.dumps(existing_warnings, sort_keys=True)))

            recipe_dir = Path(str(row.get("run_dir") or ""))
            if recipe_dir.exists():
                initial = compute_initial_volume_cm3(recipe, materials)
                write_json(recipe_dir / "porosity.json", porosity)
                write_json(recipe_dir / "volumes.json", {"initial": initial, "porosity_volumes": porosity})
                write_json(recipe_dir / "warnings.json", existing_warnings)

            status_row = {
                "recipe_id": recipe_id,
                "chem_hash": chem_hash,
                "status": "updated",
                "message": "",
                "old_porosity": old_porosity,
                "new_porosity": new_porosity,
                "old_final_solid_volume_cm3": old_final,
                "new_final_solid_volume_cm3": final_volume,
            }
            summary["updated"] += 1
        except Exception as exc:
            status_row = {
                "recipe_id": recipe_id,
                "chem_hash": chem_hash,
                "status": "failed",
                "message": f"{type(exc).__name__}: {exc}",
                "old_porosity": old_porosity,
                "old_final_solid_volume_cm3": old_final,
            }
            summary["failed"] += 1
        _append_status_row(status_path, status_row, POROSITY_STATUS_FIELDS)
        if progress_every and (index % progress_every == 0 or status_row["status"] == "failed"):
            print(f"recompute porosity {index}/{len(rows)}: {status_row['status']}")

    with database.connect() as conn:
        conn.executemany(
            """
            UPDATE recipe_runs
            SET initial_volume_cm3 = ?,
                final_solid_volume_cm3 = ?,
                porosity = ?,
                warnings_json = ?
            WHERE recipe_id = ?
            """,
            updates,
        )
    summary["finished_at"] = utc_now_iso()
    write_json(summary_path, summary)
    return summary


def _load_json_from_text(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(str(value))
    except Exception:
        return default


def backfill_reconstructed_volumes_and_porosity(
    *,
    db: str | Path,
    dat_lst: str | Path,
    limit: int | None = None,
    force: bool = False,
    include_unlinked_complete: bool = False,
    gems_class_path: str = "xgems:ChemicalEngineDicts",
    xgems_input_mode: str = "formula",
    xgems_phase_volume_unit: str = "m3",
    create_backup: bool = True,
    reuse_runner: bool = True,
    progress_every: int = 100,
    runner_factory: RunnerFactory | None = None,
) -> dict[str, Any]:
    backup = create_sqlite_backup(db) if create_backup else None
    chemistry = backfill_reconstructed_phase_volumes(
        db=db,
        dat_lst=dat_lst,
        limit=limit,
        force=force,
        include_unlinked_complete=include_unlinked_complete,
        gems_class_path=gems_class_path,
        xgems_input_mode=xgems_input_mode,
        reuse_runner=reuse_runner,
        progress_every=progress_every,
        runner_factory=runner_factory,
    )
    porosity = recompute_recipe_porosity_from_backfilled_volumes(
        db=db,
        limit=limit,
        require_reconstructed=True,
        xgems_phase_volume_unit=xgems_phase_volume_unit,
        progress_every=max(progress_every, 1) * 10 if progress_every else 0,
    )
    return {
        "backup": None if backup is None else str(backup),
        "chemistry_backfill": chemistry,
        "porosity_recompute": porosity,
    }
