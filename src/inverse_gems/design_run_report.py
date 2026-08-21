from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import write_json


BINDER_COLUMNS = ["OPC", "slag", "fly_ash", "metakaolin", "silica_fume", "limestone", "gypsum"]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _finite_or_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _target_label(column: str) -> str:
    label = str(column)
    for prefix in ("y__amount_", "y__volume_", "y__scalar_", "y__"):
        if label.startswith(prefix):
            label = label.removeprefix(prefix)
            break
    return label.replace("C_A_S_H", "C-A-S-H")


def _status_count(status_counts: dict[str, Any], key: str) -> int:
    value = status_counts.get(key, 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _selection_count(selection_summary: dict[str, Any], selected: pd.DataFrame) -> int:
    if not selected.empty:
        return int(len(selected))
    value = selection_summary.get("top_k_written")
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _material_system(manifest: dict[str, Any], best: dict[str, Any]) -> Any:
    registry_entry = manifest.get("model_registry_entry") or {}
    metadata = manifest.get("metadata_constraints") or {}
    material_filter = (metadata.get("material_system") or {}).get("include") or []
    return _first_present(
        best.get("material_system"),
        registry_entry.get("material_system"),
        material_filter[0] if material_filter else None,
    )


def _age_days(manifest: dict[str, Any], best: dict[str, Any]) -> Any:
    input_constraints = manifest.get("input_constraints") or {}
    age_constraint = input_constraints.get("age_days") or {}
    registry_entry = manifest.get("model_registry_entry") or {}
    return _first_present(best.get("age_days"), age_constraint.get("equals"), registry_entry.get("age_days"))


def _run_summary_row(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    run_summary = _read_json(run_dir / "design_query_run_summary.json")
    manifest = run_summary.get("compiled_manifest") or _read_json(run_dir / "compiled_query" / "design_query_manifest.json")
    search_summary = run_summary.get("candidate_search_summary") or _read_json(
        run_dir / "surrogate_search" / "candidate_search_summary.json"
    )
    validation_summary = run_summary.get("validation_summary") or _read_json(
        run_dir / "validation" / "validation_summary.json"
    )
    selection_summary = run_summary.get("selection_summary") or _read_json(run_dir / "selection" / "selection_summary.json")
    pH_policy = selection_summary.get("pH_uncertainty_policy") or {}
    availability = manifest.get("target_availability") or _read_json(
        run_dir / "compiled_query" / "target_availability_report.json"
    )
    selected = _read_csv(run_dir / "final_selected_candidates.csv")
    best = selected.iloc[0].to_dict() if not selected.empty else {}
    registry_entry = manifest.get("model_registry_entry") or {}
    validation_counts = validation_summary.get("validation_status_counts") or {}
    availability_issues = availability.get("issues") or []
    issue_targets = [
        str(issue.get("target") or (issue.get("diagnostics") or {}).get("target_label") or "")
        for issue in availability_issues
    ]
    requested_targets = availability.get("requested_targets") or list((manifest.get("target_constraints") or {}).keys())

    row: dict[str, Any] = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "query_name": manifest.get("name"),
        "query_path": manifest.get("query") or run_summary.get("query"),
        "material_system": _material_system(manifest, best),
        "age_days": _age_days(manifest, best),
        "model_registry_id": registry_entry.get("id"),
        "model_table": manifest.get("model_table") or registry_entry.get("model_table"),
        "model_bundle": manifest.get("model_bundle") or registry_entry.get("model_bundle"),
        "reaction_model_id": (manifest.get("reaction_model") or {}).get("id")
        or registry_entry.get("reaction_model_id"),
        "reaction_model_signature": (manifest.get("reaction_model") or {}).get("signature")
        or registry_entry.get("reaction_model_signature"),
        "target_availability_policy": availability.get("policy"),
        "target_availability_issue_count": len(availability_issues),
        "target_availability_issue_targets": ", ".join([target for target in issue_targets if target]),
        "requested_targets": ", ".join(map(str, requested_targets)),
        "candidate_rows_after_filters": search_summary.get("candidate_rows_after_filters"),
        "validation_requested_top_k": validation_summary.get("requested_top_k"),
        "validation_runs": validation_summary.get("validation_runs"),
        "validation_complete_count": _status_count(validation_counts, "complete"),
        "validation_status_counts": json.dumps(validation_counts, sort_keys=True),
        "final_selected_count": _selection_count(selection_summary, selected),
        "selection_input_rows": selection_summary.get("input_rows"),
        "selection_rows_after_constraints": selection_summary.get("candidate_rows_after_constraints"),
        "selection_top_k_written": selection_summary.get("top_k_written"),
        "selection_pH_uncertainty_mode": pH_policy.get("mode"),
        "selection_pH_unreliable_rows": pH_policy.get("unreliable_rows"),
        "selection_pH_rejected_rows": pH_policy.get("rejected_rows"),
        "selection_pH_penalized_rows": pH_policy.get("penalized_rows"),
        "best_recipe_text": best.get("recipe_text"),
    }
    for column in BINDER_COLUMNS + [
        "w_b",
        "water_g",
        "xgems_water_g",
        "xgems_w_b",
        "xgems_water_mode",
        "xgems_water_delta_g",
        "xgems_water_relative_delta",
        "xgems_water_matches_recipe",
        "pH_water_reliable",
        "pH_unreliable_reason",
        "uncertainty_flags",
        "preflight_dir",
        "age_days",
    ]:
        if column in best:
            row[f"best_{column}"] = _finite_or_none(best.get(column))
    for column, value in best.items():
        if str(column).startswith(("validated__y__", "pred__y__", "diff_validated_minus_pred__y__")):
            row[f"best_{column}"] = _finite_or_none(value)

    target_error_rows = _target_error_rows(run_dir, row, validation_summary)
    target_availability_rows = _target_availability_rows(run_dir, row, availability)
    target_issue_rows = _target_issue_rows(run_dir, row, availability_issues)
    return row, target_error_rows, target_availability_rows, target_issue_rows


def _target_error_rows(
    run_dir: Path,
    summary_row: dict[str, Any],
    validation_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_column, metrics in (validation_summary.get("target_error_summary") or {}).items():
        rows.append(
            {
                "run_name": run_dir.name,
                "run_dir": str(run_dir),
                "material_system": summary_row.get("material_system"),
                "target_column": target_column,
                "target_label": _target_label(str(target_column)),
                "source_column": metrics.get("source_column"),
                "validated_vs_pred_count": metrics.get("validated_vs_pred_count"),
                "validated_vs_pred_mae": metrics.get("validated_vs_pred_mae"),
                "validated_vs_pred_max_abs": metrics.get("validated_vs_pred_max_abs"),
                "validated_vs_source_true_count": metrics.get("validated_vs_source_true_count"),
                "validated_vs_source_true_mae": metrics.get("validated_vs_source_true_mae"),
                "validated_vs_source_true_max_abs": metrics.get("validated_vs_source_true_max_abs"),
            }
        )
    return rows


def _target_availability_rows(
    run_dir: Path,
    summary_row: dict[str, Any],
    availability: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in availability.get("matched_targets") or []:
        rows.append(
            {
                "run_name": run_dir.name,
                "run_dir": str(run_dir),
                "material_system": summary_row.get("material_system"),
                "policy": availability.get("policy"),
                "requested_target": target.get("requested_target"),
                "target_label": target.get("target_label"),
                "target_column": target.get("target_column"),
                "status": target.get("status"),
                "r2": target.get("r2"),
                "range": target.get("range"),
                "nonzero_fraction": target.get("nonzero_fraction"),
                "reasons": target.get("reasons"),
            }
        )
    return rows


def _target_issue_rows(
    run_dir: Path,
    summary_row: dict[str, Any],
    issues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for issue in issues:
        diagnostics = issue.get("diagnostics") or {}
        rows.append(
            {
                "run_name": run_dir.name,
                "run_dir": str(run_dir),
                "material_system": summary_row.get("material_system"),
                "target": issue.get("target") or diagnostics.get("target_label"),
                "severity": issue.get("severity"),
                "status": issue.get("status") or diagnostics.get("status"),
                "message": issue.get("message"),
                "r2": diagnostics.get("r2"),
                "range": diagnostics.get("range"),
                "nonzero_fraction": diagnostics.get("nonzero_fraction"),
                "reasons": diagnostics.get("reasons"),
            }
        )
    return rows


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    existing = [column for column in columns if column in frame.columns]
    if frame.empty or not existing:
        return ""
    text = frame[existing].fillna("").astype(str)
    rows = [existing] + text.values.tolist()
    widths = [max(len(str(row[index])) for row in rows) for index in range(len(existing))]

    def render(row: list[str]) -> str:
        return "| " + " | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([render(existing), separator] + [render(row) for row in text.values.tolist()])


def _write_markdown(
    *,
    out_path: Path,
    summary: dict[str, Any],
    summary_frame: pd.DataFrame,
    error_frame: pd.DataFrame,
    issue_frame: pd.DataFrame,
) -> None:
    preferred_validated_columns = [
        "best_validated__y__porosity",
        "best_validated__y__pH",
        "best_validated__y__amount_C_A_S_H",
        "best_validated__y__amount_ettringite",
        "best_validated__y__amount_Portlandite",
        "best_validated__y__amount_straetlingite",
        "best_validated__y__amount_monosulfate",
        "best_validated__y__amount_monocarbonate",
        "best_validated__y__amount_Calcite",
    ]
    available_validated_columns = [
        column for column in summary_frame.columns if column.startswith("best_validated__y__")
    ]
    best_validated_columns = [
        column for column in preferred_validated_columns if column in available_validated_columns
    ]
    best_validated_columns.extend(
        sorted(column for column in available_validated_columns if column not in set(best_validated_columns))
    )
    run_columns = [
        "run_name",
        "material_system",
        "age_days",
        "candidate_rows_after_filters",
        "validation_complete_count",
        "final_selected_count",
        "target_availability_issue_targets",
        "best_OPC",
        "best_slag",
        "best_fly_ash",
        "best_metakaolin",
        "best_limestone",
        "best_gypsum",
        "best_w_b",
        "best_xgems_water_g",
        "best_xgems_water_delta_g",
        "best_pH_water_reliable",
        "selection_pH_uncertainty_mode",
        "selection_pH_unreliable_rows",
        "selection_pH_rejected_rows",
        "selection_pH_penalized_rows",
        "best_uncertainty_flags",
        "best_preflight_dir",
        *best_validated_columns,
    ]
    error_columns = [
        "run_name",
        "target_label",
        "validated_vs_pred_count",
        "validated_vs_pred_mae",
        "validated_vs_pred_max_abs",
    ]
    issue_columns = ["run_name", "target", "severity", "status", "message", "r2", "reasons"]
    lines = [
        "# Design Run Summary",
        "",
        f"- runs summarized: {summary['run_count']}",
        f"- total final selected candidates: {summary['total_final_selected_count']}",
        f"- runs with target-availability issues: {summary['runs_with_target_availability_issues']}",
        "",
        "## Runs",
        "",
        _markdown_table(summary_frame, run_columns),
        "",
        "## Validation Error By Target",
        "",
        _markdown_table(error_frame, error_columns),
        "",
    ]
    if not issue_frame.empty:
        lines.extend(["## Target Availability Issues", "", _markdown_table(issue_frame, issue_columns), ""])
    out_path.write_text("\n".join(line for line in lines if line is not None).rstrip() + "\n", encoding="utf-8")


def run_design_run_report(*, runs: list[str | Path], out: str | Path) -> Path:
    """Summarize completed design-query run directories without rerunning chemistry."""
    run_dirs = [Path(run) for run in runs]
    if not run_dirs:
        raise ValueError("At least one --runs path is required.")

    summary_rows: list[dict[str, Any]] = []
    target_error_rows: list[dict[str, Any]] = []
    target_availability_rows: list[dict[str, Any]] = []
    target_issue_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for run_dir in run_dirs:
        if not run_dir.exists():
            warnings.append(f"Run directory does not exist: {run_dir}")
            continue
        if not (run_dir / "design_query_run_summary.json").exists() and not (
            run_dir / "compiled_query" / "design_query_manifest.json"
        ).exists():
            warnings.append(f"Run directory does not look like a design-query run: {run_dir}")
            continue
        row, error_rows, availability_rows, issue_rows = _run_summary_row(run_dir)
        summary_rows.append(row)
        target_error_rows.extend(error_rows)
        target_availability_rows.extend(availability_rows)
        target_issue_rows.extend(issue_rows)

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(summary_rows)
    error_frame = pd.DataFrame(target_error_rows)
    availability_frame = pd.DataFrame(target_availability_rows)
    issue_frame = pd.DataFrame(target_issue_rows)
    if not summary_frame.empty:
        summary_frame = summary_frame.sort_values(["material_system", "run_name"], na_position="last").reset_index(
            drop=True
        )
    if not error_frame.empty:
        error_frame = error_frame.sort_values(["material_system", "run_name", "target_label"]).reset_index(drop=True)
    if not availability_frame.empty:
        availability_frame = availability_frame.sort_values(
            ["material_system", "run_name", "target_label"]
        ).reset_index(drop=True)
    if not issue_frame.empty:
        issue_frame = issue_frame.sort_values(["material_system", "run_name", "target"]).reset_index(drop=True)

    summary_csv = out_dir / "design_run_summary.csv"
    error_csv = out_dir / "design_run_target_errors.csv"
    availability_csv = out_dir / "design_run_target_availability.csv"
    issues_csv = out_dir / "design_run_target_availability_issues.csv"
    summary_frame.to_csv(summary_csv, index=False)
    error_frame.to_csv(error_csv, index=False)
    availability_frame.to_csv(availability_csv, index=False)
    issue_frame.to_csv(issues_csv, index=False)

    summary = {
        "run_count": int(len(summary_frame)),
        "requested_run_count": int(len(run_dirs)),
        "total_final_selected_count": int(summary_frame.get("final_selected_count", pd.Series(dtype=int)).sum())
        if not summary_frame.empty
        else 0,
        "runs_with_target_availability_issues": int(
            (summary_frame.get("target_availability_issue_count", pd.Series(dtype=int)) > 0).sum()
        )
        if not summary_frame.empty
        else 0,
        "warnings": warnings,
        "files": {
            "summary_csv": str(summary_csv),
            "target_errors_csv": str(error_csv),
            "target_availability_csv": str(availability_csv),
            "target_availability_issues_csv": str(issues_csv),
            "json": str(out_dir / "design_run_summary.json"),
            "markdown": str(out_dir / "design_run_summary.md"),
        },
    }
    payload = {
        "summary": summary,
        "runs": summary_frame.to_dict(orient="records"),
        "target_errors": error_frame.to_dict(orient="records"),
        "target_availability": availability_frame.to_dict(orient="records"),
        "target_availability_issues": issue_frame.to_dict(orient="records"),
    }
    write_json(out_dir / "design_run_summary.json", payload)
    _write_markdown(
        out_path=out_dir / "design_run_summary.md",
        summary=summary,
        summary_frame=summary_frame,
        error_frame=error_frame,
        issue_frame=issue_frame,
    )
    return out_dir
