from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from .cached_forward import run_forward_cached
from .database import InverseGemsDatabase, utc_now_iso
from .sampling import numeric_metadata_from_row, read_recipe_csv, recipe_text_from_row
from .utils import write_json
from .xgems_preflight import run_xgems_input_preflight

BATCH_STATUS_FIELDS = [
    "row_index",
    "recipe_id",
    "chem_hash",
    "prepared_id",
    "reaction_model_id",
    "reaction_model_signature",
    "status",
    "solver_status",
    "used_cached_xgems",
    "solver_rescued",
    "xgems_retry_count",
    "primary_chem_hash",
    "xgems_water_g",
    "xgems_w_b",
    "xgems_water_mode",
    "age_days",
    "temperature_celsius",
    "material_system",
    "recipe_text",
    "preflight_dir",
    "error_message",
    "run_time_seconds",
    "updated_at",
]


def _read_status_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_status_file(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BATCH_STATUS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in BATCH_STATUS_FIELDS})


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_recipe_text_from_row(row: dict[str, Any]) -> str:
    try:
        return recipe_text_from_row(row)
    except Exception:
        return ""


def _write_failure_preflight(
    *,
    row: dict[str, Any],
    db_dir: Path,
    recipe_id: str,
    dat_lst: str | Path | None,
    run_mode: str,
    gems_class_path: str,
    xgems_input_mode: str,
    xgems_water_mode: str,
    xgems_water_factor: float,
    xgems_water_g: float | None,
    xgems_water_w_b: float | None,
    reaction_model_id: str | None,
    reaction_model_config: str | Path | None,
) -> str:
    recipe_text = _safe_recipe_text_from_row(row)
    if not recipe_text:
        return ""
    out = db_dir / "failure_preflight" / recipe_id
    try:
        run_xgems_input_preflight(
            recipe_text=recipe_text,
            out=out,
            dat_lst=dat_lst,
            run_mode=run_mode,
            xgems_input_mode=xgems_input_mode,
            gems_class_path=gems_class_path,
            xgems_water_mode=xgems_water_mode,
            xgems_water_factor=xgems_water_factor,
            xgems_water_g=xgems_water_g,
            xgems_water_w_b=xgems_water_w_b,
            reaction_model_id=reaction_model_id,
            reaction_model_config=reaction_model_config,
            instantiate_runner=False,
        )
    except Exception:
        return ""
    return str(out)


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _batch_summary_from_rows(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    total_rows = int(manifest.get("recipe_count") or len(rows))
    processed = len(rows)
    complete = [row for row in rows if str(row.get("status")) == "complete"]
    failed = [row for row in rows if str(row.get("status")) == "failed"]
    cached = [row for row in rows if str(row.get("used_cached_xgems")).lower() == "true"]
    rescued = [row for row in rows if str(row.get("solver_rescued")).lower() == "true"]
    runtimes = [value for value in (_float_or_none(row.get("run_time_seconds")) for row in rows) if value is not None]
    avg_runtime = sum(runtimes) / len(runtimes) if runtimes else None
    remaining = max(total_rows - processed, 0)
    estimated_remaining = remaining * avg_runtime if avg_runtime is not None else None
    percent_complete = (processed / total_rows * 100.0) if total_rows else 0.0
    return {
        "ok": len(failed) == 0 and processed >= total_rows and total_rows > 0,
        "status": "complete" if processed >= total_rows and len(failed) == 0 and total_rows > 0 else "incomplete_or_failed",
        "recipe_count": total_rows,
        "processed_count": processed,
        "complete_count": len(complete),
        "failed_count": len(failed),
        "remaining_count": remaining,
        "percent_complete": percent_complete,
        "cache_hit_count": len(cached),
        "solver_rescued_count": len(rescued),
        "average_row_runtime_seconds": avg_runtime,
        "estimated_remaining_seconds": estimated_remaining,
        "started_at": manifest.get("started_at"),
        "updated_at": utc_now_iso(),
        "recipes_csv": manifest.get("recipes_csv"),
        "db": manifest.get("db"),
        "dat_lst": manifest.get("dat_lst"),
        "mode": "mock" if manifest.get("use_mock") else "real",
        "workers_effective": manifest.get("workers_effective"),
        "xgems_input_mode": manifest.get("xgems_input_mode"),
        "retry_water_on_failure": manifest.get("retry_water_on_failure"),
    }


def _write_batch_markdown(path: Path, summary: dict[str, Any], failures: list[dict[str, Any]]) -> None:
    lines = [
        "# inverse_gems Batch Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Processed: {summary['processed_count']} / {summary['recipe_count']} ({summary['percent_complete']:.2f}%)",
        f"- Complete: {summary['complete_count']}",
        f"- Failed: {summary['failed_count']}",
        f"- Cache hits: {summary['cache_hit_count']}",
        f"- Solver rescued: {summary['solver_rescued_count']}",
        f"- Mode: `{summary.get('mode')}`",
        f"- Updated: `{summary.get('updated_at')}`",
        "",
    ]
    if failures:
        lines.extend(
            [
                "## Failures",
                "",
                "| row_index | recipe_id | status | preflight_dir | error_message |",
                "| ---: | --- | --- | --- | --- |",
            ]
        )
        for row in failures[:50]:
            error = str(row.get("error_message", "")).replace("|", "/")
            preflight = str(row.get("preflight_dir", "")).replace("|", "/")
            lines.append(
                f"| {row.get('row_index', '')} | {row.get('recipe_id', '')} | "
                f"{row.get('status', '')} | {preflight} | {error} |"
            )
        if len(failures) > 50:
            lines.append(f"| ... | ... | ... | {len(failures) - 50} more failures omitted from markdown preview |")
        lines.append("")
    lines.extend(
        [
            "Files:",
            "- `batch_status.csv`: row-level execution log",
            "- `batch_progress.json`: machine-readable current progress",
            "- `batch_failures.csv`: failed or non-complete rows",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_batch_progress_artifacts(
    *,
    db: str | Path,
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    out: str | Path | None = None,
) -> dict[str, Any]:
    out_dir = Path(out) if out is not None else Path(db)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = _batch_summary_from_rows(rows, manifest)
    failures = [row for row in rows if str(row.get("status")) != "complete"]
    write_json(out_dir / "batch_progress.json", summary)
    with (out_dir / "batch_failures.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BATCH_STATUS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in failures:
            writer.writerow({field: row.get(field, "") for field in BATCH_STATUS_FIELDS})
    _write_batch_markdown(out_dir / "batch_summary.md", summary, failures)
    return summary


def summarize_batch_status(*, db: str | Path, out: str | Path | None = None) -> dict[str, Any]:
    db_dir = Path(db)
    status_path = db_dir / "batch_status.csv"
    manifest_path = db_dir / "batch_manifest.json"
    rows = _read_status_rows(status_path)
    manifest = _load_manifest(manifest_path)
    summary = write_batch_progress_artifacts(db=db_dir, rows=rows, manifest=manifest, out=out or db_dir)
    summary["has_status_file"] = status_path.exists()
    summary["status_path"] = str(status_path)
    summary["manifest_path"] = str(manifest_path)
    return summary


def run_batch_cached(
    *,
    recipes_csv: str | Path,
    db: str | Path,
    dat_lst: str | Path | None = None,
    use_mock: bool = False,
    workers: int = 1,
    resume: bool = False,
    fail_fast: bool = False,
    force_rerun_xgems: bool = False,
    run_mode: str = "reacted_only",
    gems_class_path: str = "xgems:ChemicalEngineDicts",
    xgems_input_mode: str = "formula",
    xgems_water_mode: str = "initial",
    xgems_water_factor: float = 1.0,
    xgems_water_g: float | None = None,
    xgems_water_w_b: float | None = None,
    retry_water_on_failure: bool = False,
    retry_water_cap_w_b_ladder: list[float] | tuple[float, ...] | str = (0.45, 0.40, 0.35, 0.30),
    retry_water_up_w_b_ladder: list[float] | tuple[float, ...] | str = (0.30, 0.35, 0.40, 0.45),
    retry_water_min_w_b: float = 0.30,
    reaction_model_id: str | None = None,
    reaction_model_config: str | Path | None = None,
    progress_every: int = 25,
) -> Path:
    started_at = utc_now_iso()
    db_dir = Path(db)
    if workers != 1:
        # SQLite writes and xGEMS processes are intentionally kept conservative for this milestone.
        workers = 1
    database = InverseGemsDatabase(db)
    rows = read_recipe_csv(recipes_csv)
    current_recipe_ids = {
        str(row.get("recipe_id") or f"row_{index:06d}")
        for index, row in enumerate(rows, 1)
    }
    status_path = db_dir / "batch_status.csv"
    manifest_path = db_dir / "batch_manifest.json"
    manifest = {
        "started_at": started_at,
        "updated_at": started_at,
        "recipes_csv": str(recipes_csv),
        "recipe_count": len(rows),
        "db": str(db_dir),
        "dat_lst": str(dat_lst) if dat_lst else None,
        "use_mock": use_mock,
        "workers_effective": workers,
        "resume": resume,
        "fail_fast": fail_fast,
        "force_rerun_xgems": force_rerun_xgems,
        "run_mode": run_mode,
        "gems_class_path": gems_class_path,
        "xgems_input_mode": xgems_input_mode,
        "xgems_water_mode": xgems_water_mode,
        "xgems_water_factor": xgems_water_factor,
        "xgems_water_g": xgems_water_g,
        "xgems_water_w_b": xgems_water_w_b,
        "retry_water_on_failure": retry_water_on_failure,
        "retry_water_cap_w_b_ladder": retry_water_cap_w_b_ladder,
        "retry_water_up_w_b_ladder": retry_water_up_w_b_ladder,
        "retry_water_min_w_b": retry_water_min_w_b,
        "reaction_model_id": reaction_model_id,
        "reaction_model_config": str(reaction_model_config) if reaction_model_config else None,
        "progress_every": progress_every,
    }
    write_json(manifest_path, manifest)
    existing_completed: set[str] = set()
    status_rows: list[dict[str, Any]] = []
    if resume and status_path.exists():
        prior_status_rows = _read_status_rows(status_path)
        status_rows = [
            row
            for row in prior_status_rows
            if str(row.get("recipe_id") or "") in current_recipe_ids and row.get("status") == "complete"
        ]
        for row in status_rows:
            if row.get("recipe_id"):
                existing_completed.add(str(row["recipe_id"]))
    with status_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BATCH_STATUS_FIELDS)
        writer.writeheader()
        for existing in status_rows:
            writer.writerow({field: existing.get(field, "") for field in BATCH_STATUS_FIELDS})
        for index, row in enumerate(rows, 1):
            metadata = numeric_metadata_from_row(row)
            recipe_id = str(row.get("recipe_id") or f"row_{index:06d}")
            if resume and recipe_id in existing_completed:
                continue
            start = time.perf_counter()
            row_xgems_water_mode = xgems_water_mode
            row_xgems_water_factor = xgems_water_factor
            row_xgems_water_g = xgems_water_g
            row_xgems_water_w_b = xgems_water_w_b
            try:
                row_xgems_water_mode = str(row.get("xgems_water_mode") or xgems_water_mode)
                row_xgems_water_factor = (
                    float(row["xgems_water_factor"]) if row.get("xgems_water_factor") not in (None, "") else xgems_water_factor
                )
                row_xgems_water_g = (
                    float(row["xgems_water_g"]) if row.get("xgems_water_g") not in (None, "") else xgems_water_g
                )
                row_xgems_water_w_b = (
                    float(row["xgems_water_w_b"]) if row.get("xgems_water_w_b") not in (None, "") else xgems_water_w_b
                )
                if resume and database.get_recipe_run(recipe_id):
                    existing_recipe = database.get_recipe_run(recipe_id) or {}
                    existing_chemistry = database.get_chemistry_run(str(existing_recipe.get("chem_hash") or "")) or {}
                    existing_status = str(existing_chemistry.get("status") or "complete")
                    status_row = {
                        "row_index": index,
                        "recipe_id": recipe_id,
                        "chem_hash": existing_recipe.get("chem_hash", ""),
                        "prepared_id": existing_recipe.get("prepared_id", ""),
                        "reaction_model_id": existing_recipe.get("reaction_model_id", ""),
                        "reaction_model_signature": existing_recipe.get("reaction_model_signature", ""),
                        "status": existing_status,
                        "solver_status": existing_recipe.get("primary_solver_status", ""),
                        "used_cached_xgems": True,
                        "solver_rescued": bool(existing_recipe.get("solver_rescued")),
                        "xgems_retry_count": existing_recipe.get("xgems_retry_count", ""),
                        "primary_chem_hash": existing_recipe.get("primary_chem_hash", ""),
                        "xgems_water_g": existing_recipe.get("xgems_water_g", ""),
                        "xgems_w_b": existing_recipe.get("xgems_w_b", ""),
                        "xgems_water_mode": existing_recipe.get("xgems_water_mode", ""),
                        "age_days": metadata.get("age_days", ""),
                        "temperature_celsius": metadata.get("temperature_celsius", ""),
                        "material_system": metadata.get("material_system", ""),
                        "recipe_text": _safe_recipe_text_from_row(row),
                        "preflight_dir": "",
                        "error_message": "skipped existing recipe_id",
                        "run_time_seconds": 0.0,
                        "updated_at": utc_now_iso(),
                    }
                    writer.writerow(status_row)
                    status_rows.append(status_row)
                    continue
                recipe_text = recipe_text_from_row(row)
                result = run_forward_cached(
                    recipe_text=recipe_text,
                    db=db,
                    dat_lst=dat_lst,
                    use_mock=use_mock,
                    force_rerun_xgems=force_rerun_xgems,
                    run_mode=run_mode,
                    normalize=True,
                    temperature_celsius=float(metadata.get("temperature_celsius", 20.0)),
                    gems_class_path=gems_class_path,
                    xgems_input_mode=xgems_input_mode,
                    recipe_id=recipe_id,
                    recipe_metadata=metadata,
                    xgems_water_mode=row_xgems_water_mode,
                    xgems_water_factor=row_xgems_water_factor,
                    xgems_water_g=row_xgems_water_g,
                    xgems_water_w_b=row_xgems_water_w_b,
                    retry_water_on_failure=retry_water_on_failure,
                    retry_water_cap_w_b_ladder=retry_water_cap_w_b_ladder,
                    retry_water_up_w_b_ladder=retry_water_up_w_b_ladder,
                    retry_water_min_w_b=retry_water_min_w_b,
                    reaction_model_id=reaction_model_id,
                    reaction_model_config=reaction_model_config,
                )
                chemistry_status = str(result.get("chemistry_status") or "complete")
                error_message = "" if chemistry_status == "complete" else f"xGEMS/GEMS solver status: {result.get('solver_status')}"
                if result.get("solver_rescued"):
                    error_message = (
                        "solver rescued by adaptive xGEMS-water retry; "
                        f"primary solver status: {result.get('retry_history', [{}])[0].get('solver_status')}"
                    )
                status_row = {
                    "row_index": index,
                    "recipe_id": recipe_id,
                    "chem_hash": result["chem_hash"],
                    "prepared_id": result.get("prepared_id", ""),
                    "reaction_model_id": result.get("reaction_model_id", ""),
                    "reaction_model_signature": result.get("reaction_model_signature", ""),
                    "status": chemistry_status,
                    "solver_status": result.get("solver_status", ""),
                    "used_cached_xgems": bool(result.get("reused_cache")),
                    "solver_rescued": bool(result.get("solver_rescued")),
                    "xgems_retry_count": result.get("xgems_retry_count", ""),
                    "primary_chem_hash": result.get("primary_chem_hash", ""),
                    "xgems_water_g": result.get("xgems_water_g", ""),
                    "xgems_w_b": result.get("xgems_w_b", ""),
                    "xgems_water_mode": result.get("xgems_water_mode", ""),
                    "age_days": metadata.get("age_days", ""),
                    "temperature_celsius": metadata.get("temperature_celsius", ""),
                    "material_system": metadata.get("material_system", ""),
                    "recipe_text": recipe_text,
                    "preflight_dir": result.get("preflight_dir", ""),
                    "error_message": error_message,
                    "run_time_seconds": time.perf_counter() - start,
                    "updated_at": utc_now_iso(),
                }
                writer.writerow(status_row)
                status_rows.append(status_row)
            except Exception as exc:
                failure_preflight_dir = _write_failure_preflight(
                    row=row,
                    db_dir=db_dir,
                    recipe_id=recipe_id,
                    dat_lst=dat_lst,
                    run_mode=run_mode,
                    gems_class_path=gems_class_path,
                    xgems_input_mode=xgems_input_mode,
                    xgems_water_mode=row_xgems_water_mode,
                    xgems_water_factor=row_xgems_water_factor,
                    xgems_water_g=row_xgems_water_g,
                    xgems_water_w_b=row_xgems_water_w_b,
                    reaction_model_id=reaction_model_id,
                    reaction_model_config=reaction_model_config,
                )
                status_row = {
                    "row_index": index,
                    "recipe_id": recipe_id,
                    "chem_hash": "",
                    "prepared_id": "",
                    "reaction_model_id": reaction_model_id or "",
                    "reaction_model_signature": "",
                    "status": "failed",
                    "solver_status": "",
                    "used_cached_xgems": False,
                    "solver_rescued": False,
                    "xgems_retry_count": "",
                    "primary_chem_hash": "",
                    "xgems_water_g": "",
                    "xgems_w_b": "",
                    "xgems_water_mode": "",
                    "age_days": metadata.get("age_days", ""),
                    "temperature_celsius": metadata.get("temperature_celsius", ""),
                    "material_system": metadata.get("material_system", ""),
                    "recipe_text": _safe_recipe_text_from_row(row) if row else "",
                    "preflight_dir": failure_preflight_dir,
                    "error_message": f"{type(exc).__name__}: {exc}",
                    "run_time_seconds": time.perf_counter() - start,
                    "updated_at": utc_now_iso(),
                }
                writer.writerow(status_row)
                status_rows.append(status_row)
                if fail_fast:
                    raise
            handle.flush()
            if progress_every > 0 and len(status_rows) % progress_every == 0:
                manifest["updated_at"] = utc_now_iso()
                write_json(manifest_path, manifest)
                write_batch_progress_artifacts(db=db_dir, rows=status_rows, manifest=manifest)
        manifest["updated_at"] = utc_now_iso()
        write_json(manifest_path, manifest)
        write_batch_progress_artifacts(db=db_dir, rows=status_rows, manifest=manifest)
    return status_path
