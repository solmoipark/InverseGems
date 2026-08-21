from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .batch import run_batch_cached, summarize_batch_status
from .global_chemistry_coverage import write_global_chemistry_coverage_report
from .global_chemistry_db import (
    acquire_global_chemistry_candidates,
    refresh_global_chemistry_db,
    train_global_chemistry_surrogate,
)
from .utils import write_json


def _stage(name: str, status: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "status": status, **extra}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _error_payload(exc: Exception) -> dict[str, str]:
    return {"error_type": type(exc).__name__, "error_message": str(exc)}


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = ["# Global Chemistry Acquisition Cycle", ""]
    lines.append(f"- status: `{summary.get('status')}`")
    lines.append(f"- db: `{summary.get('db')}`")
    lines.append(f"- mode: `{summary.get('mode')}`")
    lines.append("")
    lines.append("| Stage | Status | Artifact | Notes |")
    lines.append("| --- | --- | --- | --- |")
    for stage in summary.get("stages") or []:
        artifact = stage.get("artifact") or stage.get("out") or ""
        notes = stage.get("notes") or stage.get("error_message") or ""
        lines.append(
            "| "
            + " | ".join(
                [
                    str(stage.get("name", "")),
                    str(stage.get("status", "")),
                    str(artifact).replace("|", "/"),
                    str(notes).replace("|", "/"),
                ]
            )
            + " |"
        )
    outputs = summary.get("outputs") or {}
    if outputs:
        lines.extend(["", "## Outputs", ""])
        for key, value in outputs.items():
            lines.append(f"- `{key}`: `{value}`")
    warnings = summary.get("warnings") or []
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_global_chemistry_acquisition_cycle(
    *,
    db: str | Path,
    out: str | Path,
    schema_config: str | Path | None = None,
    recipes_csv: str | Path | None = None,
    candidate_table: str | Path | None = None,
    reference_model_table: str | Path | None = None,
    model_bundle: str | Path | None = None,
    max_candidates: int = 20,
    dat_lst: str | Path | None = None,
    use_mock: bool = False,
    run_batch: bool = True,
    refresh: bool = True,
    train_surrogate: bool = True,
    coverage: bool = True,
    selection: str | Path | None = None,
    model_config: str | Path | None = None,
    surrogate_config: str | Path | None = None,
    coverage_config: str | Path | None = None,
    material_systems_config: str | Path | None = None,
    force_rerun_xgems: bool = False,
    resume_batch: bool = True,
    fail_fast: bool = False,
    run_mode: str = "reacted_only",
    temperature_celsius: float = 20.0,
    pressure: float | None = None,
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
    nearest_distance_warn: float | None = None,
    priority_targets: list[str] | tuple[str, ...] | None = None,
    priority_targets_from_diagnostics: str | Path | None = None,
    priority_target_statuses: list[str] | tuple[str, ...] | None = None,
    priority_target_kinds: list[str] | tuple[str, ...] | None = None,
    priority_target_reliabilities: list[str] | tuple[str, ...] | None = None,
    priority_target_limit: int | None = None,
    target_priority_weight: float | None = None,
    target_region_table: list[str | Path] | tuple[str | Path, ...] | str | Path | None = None,
    target_region_weight: float | None = None,
    target_region_distance_scale: float | None = None,
    target_region_max_reference_rows: int | None = None,
) -> Path:
    """Run one active-learning-like global chemistry DB growth cycle.

    The cycle is deliberately file-first: every stage writes its normal artifacts
    and this function only adds a compact manifest tying them together.
    """

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(db)
    stages: list[dict[str, Any]] = []
    warnings: list[str] = []

    if run_batch and not use_mock and dat_lst is None:
        raise ValueError("dat_lst is required for a real acquisition cycle. Use use_mock=True for a dry/mock cycle.")

    acquisition_dir = out_dir / "acquisition"
    acquisition_path = acquire_global_chemistry_candidates(
        db=db_path,
        out=acquisition_dir,
        schema_config=schema_config,
        recipes_csv=recipes_csv,
        candidate_table=candidate_table,
        reference_model_table=reference_model_table,
        model_bundle=model_bundle,
        max_candidates=max_candidates,
        dat_lst=dat_lst,
        run_mode=run_mode,
        temperature_celsius=temperature_celsius,
        pressure=pressure,
        xgems_water_mode=xgems_water_mode,
        xgems_water_factor=xgems_water_factor,
        xgems_water_g=xgems_water_g,
        xgems_water_w_b=xgems_water_w_b,
        nearest_distance_warn=nearest_distance_warn,
        reaction_model_id=reaction_model_id,
        reaction_model_config=reaction_model_config,
        priority_targets=priority_targets,
        priority_targets_from_diagnostics=priority_targets_from_diagnostics,
        priority_target_statuses=priority_target_statuses,
        priority_target_kinds=priority_target_kinds,
        priority_target_reliabilities=priority_target_reliabilities,
        priority_target_limit=priority_target_limit,
        target_priority_weight=target_priority_weight,
        target_region_table=target_region_table,
        target_region_weight=target_region_weight,
        target_region_distance_scale=target_region_distance_scale,
        target_region_max_reference_rows=target_region_max_reference_rows,
    )
    acquisition_summary = _read_json(Path(acquisition_path) / "acquisition_summary.json")
    stages.append(
        _stage(
            "acquire_candidates",
            "complete",
            out=str(acquisition_path),
            artifact=str(Path(acquisition_path) / "acquisition_candidates.csv"),
            rows=acquisition_summary.get("selected_rows"),
        )
    )

    recipes_path = Path(acquisition_path) / "acquisition_recipes.csv"
    batch_summary: dict[str, Any] = {}
    if run_batch:
        try:
            status_path = run_batch_cached(
                recipes_csv=recipes_path,
                db=db_path,
                dat_lst=dat_lst,
                use_mock=use_mock,
                resume=resume_batch,
                fail_fast=fail_fast,
                force_rerun_xgems=force_rerun_xgems,
                run_mode=run_mode,
                gems_class_path=gems_class_path,
                xgems_input_mode=xgems_input_mode,
                xgems_water_mode=xgems_water_mode,
                xgems_water_factor=xgems_water_factor,
                xgems_water_g=xgems_water_g,
                xgems_water_w_b=xgems_water_w_b,
                retry_water_on_failure=retry_water_on_failure,
                retry_water_cap_w_b_ladder=retry_water_cap_w_b_ladder,
                retry_water_up_w_b_ladder=retry_water_up_w_b_ladder,
                retry_water_min_w_b=retry_water_min_w_b,
                reaction_model_id=reaction_model_id,
                reaction_model_config=reaction_model_config,
            )
            batch_summary = summarize_batch_status(db=db_path, out=out_dir / "batch")
            local_status_path = out_dir / "batch" / "batch_status.csv"
            local_status_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(status_path, local_status_path)
            stages.append(
                _stage(
                    "run_batch_cached",
                    "complete" if batch_summary.get("failed_count", 0) == 0 else "complete_with_failures",
                    artifact=str(local_status_path),
                    source_artifact=str(status_path),
                    out=str(out_dir / "batch"),
                    rows=batch_summary.get("processed_count"),
                    notes=f"failed={batch_summary.get('failed_count', 0)}",
                )
            )
        except Exception as exc:
            payload = _error_payload(exc)
            stages.append(_stage("run_batch_cached", "failed", **payload))
            warnings.append(f"batch failed: {payload['error_type']}: {payload['error_message']}")
            if fail_fast:
                raise
    else:
        stages.append(_stage("run_batch_cached", "skipped", notes="run_batch=False"))

    refresh_manifest: dict[str, Any] = {}
    if refresh and not train_surrogate:
        try:
            refresh_manifest = refresh_global_chemistry_db(
                db=db_path,
                selection=selection,
                model_config=model_config,
                schema_config=schema_config,
            )
            stages.append(
                _stage(
                    "refresh_global_chemistry_db",
                    "complete",
                    artifact=(refresh_manifest.get("paths") or {}).get("model_table"),
                    rows=(refresh_manifest.get("row_counts") or {}).get("model_table"),
                )
            )
        except Exception as exc:
            payload = _error_payload(exc)
            stages.append(_stage("refresh_global_chemistry_db", "failed", **payload))
            warnings.append(f"refresh failed: {payload['error_type']}: {payload['error_message']}")
            if fail_fast:
                raise
    elif refresh and train_surrogate:
        stages.append(_stage("refresh_global_chemistry_db", "merged_into_train", notes="train stage refreshes first"))
    else:
        stages.append(_stage("refresh_global_chemistry_db", "skipped", notes="refresh=False"))

    train_manifest: dict[str, Any] = {}
    if train_surrogate:
        try:
            train_manifest = train_global_chemistry_surrogate(
                db=db_path,
                selection=selection,
                model_config=model_config,
                surrogate_config=surrogate_config,
                schema_config=schema_config,
                refresh=refresh,
            )
            stages.append(
                _stage(
                    "train_global_chemistry_surrogate",
                    "complete",
                    artifact=(train_manifest.get("paths") or {}).get("model_bundle"),
                    rows=(train_manifest.get("row_counts") or {}).get("surrogate_model_table"),
                )
            )
        except Exception as exc:
            payload = _error_payload(exc)
            stages.append(_stage("train_global_chemistry_surrogate", "failed", **payload))
            warnings.append(f"train failed: {payload['error_type']}: {payload['error_message']}")
            if fail_fast:
                raise
    else:
        stages.append(_stage("train_global_chemistry_surrogate", "skipped", notes="train_surrogate=False"))

    coverage_summary: dict[str, Any] = {}
    if coverage:
        try:
            coverage_dir = write_global_chemistry_coverage_report(
                db=db_path,
                out=out_dir / "coverage",
                schema_config=schema_config,
                coverage_config=coverage_config,
                material_systems_config=material_systems_config,
            )
            coverage_summary = _read_json(Path(coverage_dir) / "global_chemistry_coverage_summary.json")
            stages.append(
                _stage(
                    "global_chemistry_coverage",
                    "complete",
                    out=str(coverage_dir),
                    artifact=str(Path(coverage_dir) / "global_chemistry_coverage_summary.json"),
                    rows=coverage_summary.get("row_count"),
                    notes="; ".join(coverage_summary.get("warnings") or []),
                )
            )
        except Exception as exc:
            payload = _error_payload(exc)
            stages.append(_stage("global_chemistry_coverage", "failed", **payload))
            warnings.append(f"coverage failed: {payload['error_type']}: {payload['error_message']}")
            if fail_fast:
                raise
    else:
        stages.append(_stage("global_chemistry_coverage", "skipped", notes="coverage=False"))

    failed_stages = [stage for stage in stages if stage.get("status") == "failed"]
    summary = {
        "status": "complete" if not failed_stages else "complete_with_stage_failures",
        "db": str(db_path),
        "out": str(out_dir),
        "mode": "mock" if use_mock else "real",
        "stages": stages,
        "warnings": warnings,
        "acquisition_summary": acquisition_summary,
        "batch_summary": batch_summary,
        "coverage_summary": coverage_summary,
        "outputs": {
            "acquisition": str(acquisition_path),
            "acquisition_candidates": str(Path(acquisition_path) / "acquisition_candidates.csv"),
            "acquisition_recipes": str(recipes_path),
            "batch": str(out_dir / "batch") if run_batch else None,
            "batch_status": str(out_dir / "batch" / "batch_status.csv") if run_batch else None,
            "coverage": str(out_dir / "coverage") if coverage else None,
            "summary_json": str(out_dir / "global_acquisition_cycle_summary.json"),
            "summary_markdown": str(out_dir / "global_acquisition_cycle_summary.md"),
        },
    }
    write_json(out_dir / "global_acquisition_cycle_summary.json", summary)
    _write_markdown(out_dir / "global_acquisition_cycle_summary.md", summary)
    return out_dir
