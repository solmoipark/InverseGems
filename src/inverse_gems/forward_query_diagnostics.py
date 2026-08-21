from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .utils import write_json


STATUS_COLUMNS = [
    "query_run_id",
    "row_index",
    "age_days",
    "recipe_id",
    "chem_hash",
    "chemistry_status",
    "solver_status",
    "solver_rescued",
    "xgems_retry_count",
    "reused_cache",
    "error_message",
    "w_b",
    "water_g",
    "xgems_w_b",
    "xgems_water_g",
    "xgems_water_mode",
    "xgems_water_delta_g",
    "xgems_water_matches_recipe",
    "pH_water_reliable",
    "pH_unreliable_reason",
    "uncertainty_flags",
    "preflight_dir",
    "porosity",
]

SCALAR_BASE_COLUMNS = [
    "query_run_id",
    "row_index",
    "age_days",
    "recipe_id",
    "chemistry_status",
    "solver_status",
    "porosity",
    "initial_volume_cm3",
    "final_solid_volume_cm3",
    "w_b",
    "water_g",
    "xgems_w_b",
    "xgems_water_g",
    "xgems_water_delta_g",
    "pH_water_reliable",
]

PHASE_PREFIXES = {
    "phase_mass": "phase_mass__",
    "phase_volume": "phase_volume__",
    "phase_volume_reconstructed": "phase_volume_reconstructed__",
}

PHASE_NONZERO_COLUMNS = [
    "kind",
    "raw_name",
    "column",
    "nonzero_count",
    "nonzero_fraction",
    "first_nonzero_age_days",
    "last_nonzero_age_days",
    "max_abs_value",
    "max_abs_age_days",
    "final_value",
]

PHASE_CHANGE_COLUMNS = [
    "kind",
    "raw_name",
    "column",
    "first_value",
    "final_value",
    "absolute_change",
    "relative_change_from_first_nonzero",
    "min_value",
    "max_value",
    "range",
    "max_abs_step_change",
    "max_abs_step_age_from",
    "max_abs_step_age_to",
]


def _existing_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _raw_name(column: str) -> str:
    return column.split("__", 1)[1] if "__" in column else column


def _phase_columns(frame: pd.DataFrame) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for kind, prefix in PHASE_PREFIXES.items():
        for column in frame.columns:
            if column.startswith(prefix):
                out.append((kind, column, _raw_name(column)))
    return out


def _first_last_age(ages: pd.Series, mask: pd.Series) -> tuple[float | None, float | None]:
    selected = ages[mask.fillna(False)]
    if selected.empty:
        return None, None
    return float(selected.iloc[0]), float(selected.iloc[-1])


def build_phase_nonzero_summary(frame: pd.DataFrame) -> pd.DataFrame:
    ages = _numeric(frame, "age_days") if "age_days" in frame else pd.Series(dtype=float)
    rows: list[dict[str, Any]] = []
    denominator = max(len(frame), 1)
    for kind, column, raw_name in _phase_columns(frame):
        values = _numeric(frame, column).fillna(0.0)
        abs_values = values.abs()
        nonzero = abs_values > 0.0
        first_age, last_age = _first_last_age(ages, nonzero) if not ages.empty else (None, None)
        max_idx = abs_values.idxmax() if not abs_values.empty else None
        rows.append(
            {
                "kind": kind,
                "raw_name": raw_name,
                "column": column,
                "nonzero_count": int(nonzero.sum()),
                "nonzero_fraction": float(nonzero.sum()) / float(denominator),
                "first_nonzero_age_days": first_age,
                "last_nonzero_age_days": last_age,
                "max_abs_value": float(abs_values.loc[max_idx]) if max_idx is not None else 0.0,
                "max_abs_age_days": float(ages.loc[max_idx]) if max_idx is not None and not ages.empty else None,
                "final_value": float(values.iloc[-1]) if len(values) else 0.0,
            }
        )
    if not rows:
        return pd.DataFrame(columns=PHASE_NONZERO_COLUMNS)
    return pd.DataFrame(rows, columns=PHASE_NONZERO_COLUMNS).sort_values(
        ["kind", "max_abs_value", "raw_name"], ascending=[True, False, True]
    )


def build_phase_change_summary(frame: pd.DataFrame) -> pd.DataFrame:
    ages = _numeric(frame, "age_days") if "age_days" in frame else pd.Series(dtype=float)
    rows: list[dict[str, Any]] = []
    for kind, column, raw_name in _phase_columns(frame):
        values = _numeric(frame, column).fillna(0.0)
        if values.empty:
            continue
        first_value = float(values.iloc[0])
        final_value = float(values.iloc[-1])
        abs_values = values.abs()
        nonzero = abs_values > 0.0
        first_nonzero_value = float(values[nonzero].iloc[0]) if nonzero.any() else 0.0
        denominator = abs(first_nonzero_value)
        relative_change = None if denominator == 0.0 else (final_value - first_nonzero_value) / denominator
        diffs = values.diff()
        if len(diffs.dropna()):
            step_idx = diffs.abs().idxmax()
            max_abs_step_change = float(diffs.loc[step_idx])
            step_age_to = float(ages.loc[step_idx]) if not ages.empty else None
            step_age_from = float(ages.shift(1).loc[step_idx]) if not ages.empty else None
        else:
            max_abs_step_change = 0.0
            step_age_to = None
            step_age_from = None
        rows.append(
            {
                "kind": kind,
                "raw_name": raw_name,
                "column": column,
                "first_value": first_value,
                "final_value": final_value,
                "absolute_change": final_value - first_value,
                "relative_change_from_first_nonzero": relative_change,
                "min_value": float(values.min()),
                "max_value": float(values.max()),
                "range": float(values.max() - values.min()),
                "max_abs_step_change": max_abs_step_change,
                "max_abs_step_age_from": step_age_from,
                "max_abs_step_age_to": step_age_to,
            }
        )
    if not rows:
        return pd.DataFrame(columns=PHASE_CHANGE_COLUMNS)
    return pd.DataFrame(rows, columns=PHASE_CHANGE_COLUMNS).sort_values(
        ["kind", "range", "raw_name"], ascending=[True, False, True]
    )


def build_status_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = _existing_columns(frame, STATUS_COLUMNS)
    return frame[columns].copy() if columns else pd.DataFrame()


def build_failed_ages(frame: pd.DataFrame) -> pd.DataFrame:
    if "chemistry_status" not in frame:
        return pd.DataFrame()
    status = frame["chemistry_status"].astype(str).str.lower()
    failed = frame[status.ne("complete")].copy()
    columns = _existing_columns(failed, STATUS_COLUMNS)
    return failed[columns].copy() if columns else failed


def build_scalar_timeseries(frame: pd.DataFrame) -> pd.DataFrame:
    scalar_columns = [column for column in frame.columns if column.startswith("scalar__")]
    columns = _existing_columns(frame, SCALAR_BASE_COLUMNS) + scalar_columns
    return frame[columns].copy() if columns else pd.DataFrame()


def _write_csv(frame: pd.DataFrame, path: Path) -> str:
    frame.to_csv(path, index=False)
    return str(path)


def _markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 10) -> str:
    if frame.empty:
        return "_None._"
    cols = [column for column in columns if column in frame.columns]
    if not cols:
        return "_None._"
    subset = frame[cols].head(limit).fillna("")
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in subset.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in cols) + " |")
    return "\n".join([header, separator, *rows])


def _write_diagnostics_md(
    *,
    out_dir: Path,
    frame: pd.DataFrame,
    status: pd.DataFrame,
    failed: pd.DataFrame,
    nonzero: pd.DataFrame,
    changes: pd.DataFrame,
    scalar: pd.DataFrame,
    paths: dict[str, str],
) -> str:
    completed = 0
    failed_count = 0
    if "chemistry_status" in status:
        counts = status["chemistry_status"].astype(str).str.lower().value_counts()
        completed = int(counts.get("complete", 0))
        failed_count = int(len(status) - completed)
    lines = [
        "# Forward Query Diagnostics",
        "",
        f"- rows: {len(frame)}",
        f"- completed rows: {completed}",
        f"- failed rows: {failed_count}",
        f"- phase columns: {len(_phase_columns(frame))}",
        f"- scalar columns: {len([column for column in frame.columns if column.startswith('scalar__')])}",
        "",
        "## Output Files",
        "",
    ]
    for label, path in sorted(paths.items()):
        lines.append(f"- {label}: `{path}`")
    lines.extend(
        [
            "",
            "## Failed Ages",
            "",
            _markdown_table(failed, ["age_days", "chemistry_status", "solver_status", "error_message"], limit=20),
            "",
            "## Largest Nonzero Phase Values",
            "",
            _markdown_table(
                nonzero.sort_values("max_abs_value", ascending=False) if not nonzero.empty else nonzero,
                ["kind", "raw_name", "nonzero_count", "max_abs_value", "max_abs_age_days", "final_value"],
                limit=15,
            ),
            "",
            "## Largest Phase Ranges",
            "",
            _markdown_table(
                changes.sort_values("range", ascending=False) if not changes.empty else changes,
                ["kind", "raw_name", "first_value", "final_value", "range", "max_abs_step_change"],
                limit=15,
            ),
            "",
            "## Scalar Columns",
            "",
            ", ".join([column for column in scalar.columns if column.startswith("scalar__")]) or "_None._",
            "",
        ]
    )
    path = out_dir / "diagnostics.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def write_forward_query_diagnostics(frame: pd.DataFrame, out_dir: str | Path) -> dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    status = build_status_table(frame)
    failed = build_failed_ages(frame)
    nonzero = build_phase_nonzero_summary(frame)
    changes = build_phase_change_summary(frame)
    scalar = build_scalar_timeseries(frame)
    paths = {
        "status": _write_csv(status, out_path / "forward_query_status.csv"),
        "failed_ages": _write_csv(failed, out_path / "failed_ages.csv"),
        "phase_nonzero_summary": _write_csv(nonzero, out_path / "phase_nonzero_summary.csv"),
        "phase_change_summary": _write_csv(changes, out_path / "phase_change_summary.csv"),
        "scalar_timeseries": _write_csv(scalar, out_path / "scalar_timeseries.csv"),
    }
    paths["diagnostics_markdown"] = _write_diagnostics_md(
        out_dir=out_path,
        frame=frame,
        status=status,
        failed=failed,
        nonzero=nonzero,
        changes=changes,
        scalar=scalar,
        paths=paths,
    )
    summary = {
        "files": paths,
        "row_count": int(len(frame)),
        "completed_count": int((status.get("chemistry_status", pd.Series(dtype=str)).astype(str).str.lower() == "complete").sum())
        if not status.empty and "chemistry_status" in status
        else 0,
        "failed_count": int(len(failed)),
        "phase_column_count": int(len(_phase_columns(frame))),
        "scalar_column_count": int(len([column for column in frame.columns if column.startswith("scalar__")])),
        "nonzero_phase_count": int((nonzero.get("nonzero_count", pd.Series(dtype=float)) > 0).sum()) if not nonzero.empty else 0,
    }
    write_json(out_path / "forward_query_diagnostics_summary.json", summary)
    return summary
