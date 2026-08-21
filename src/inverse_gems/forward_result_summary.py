from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .uncertainty import flag_counts, is_missing, read_preflight_summary, uncertainty_flags
from .utils import config_path, load_yaml, write_json


PHASE_KINDS = {
    "mass": "phase_mass__",
    "volume": "phase_volume__",
    "volume_reconstructed": "phase_volume_reconstructed__",
}

DEFAULT_SCALARS = ["porosity", "pH", "system_volume", "system_mass"]

BASE_SCALAR_COLUMNS = {
    "age_days": "age_days",
    "porosity": "porosity",
    "initial_volume_cm3": "initial_volume_cm3",
    "final_solid_volume_cm3": "final_solid_volume_cm3",
    "water_g": "water_g",
    "w_b": "w_b",
    "xgems_water_g": "xgems_water_g",
    "xgems_w_b": "xgems_w_b",
    "chemistry_status": "chemistry_status",
    "solver_status": "solver_status",
}


def _clean_value(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _numeric_value(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if pd.isna(number):
        return 0.0
    return number


def _load_time_series(run: Path) -> pd.DataFrame:
    path = run / "time_series.csv"
    if not path.exists():
        raise ValueError(f"Forward run directory does not contain time_series.csv: {run}")
    return pd.read_csv(path)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _selection_path(path: str | Path | None) -> Path:
    return Path(path) if path is not None else config_path("output_selection.yaml")


def _load_selection(path: str | Path | None) -> dict[str, Any]:
    selection_path = _selection_path(path)
    if not selection_path.exists():
        return {}
    return load_yaml(selection_path)


def _phase_names(frame: pd.DataFrame) -> list[str]:
    names: set[str] = set()
    for prefix in PHASE_KINDS.values():
        for column in frame.columns:
            if column.startswith(prefix):
                names.add(column[len(prefix) :])
    return sorted(names)


def _group_items(group_definition: Any) -> list[str]:
    if isinstance(group_definition, dict):
        group_definition = group_definition.get("include", [])
    if isinstance(group_definition, str):
        return [group_definition]
    return [str(item) for item in list(group_definition or [])]


def _phase_group_definitions(selection: dict[str, Any]) -> dict[str, dict[str, Any]]:
    groups = selection.get("phase_groups") or {}
    return {
        "amounts": dict(groups.get("amounts") or groups.get("phase_amounts") or {}),
        "volumes": dict(groups.get("volumes") or groups.get("phase_volumes") or {}),
    }


def _available_phase_groups(selection: dict[str, Any]) -> list[str]:
    definitions = _phase_group_definitions(selection)
    names = set(definitions["amounts"]) | set(definitions["volumes"])
    return sorted(str(name) for name in names)


def _requested_phase_groups(groups: list[str] | None, selection: dict[str, Any]) -> tuple[list[str], str]:
    requested = list(groups or [])
    if not requested:
        return [], "none"
    available = _available_phase_groups(selection)
    if len(requested) == 1 and str(requested[0]).lower() in {"all", "*"}:
        return available, "all_configured"
    return requested, "explicit"


def _phase_columns(frame: pd.DataFrame, phase: str) -> dict[str, str]:
    columns: dict[str, str] = {}
    for kind, prefix in PHASE_KINDS.items():
        column = f"{prefix}{phase}"
        if column in frame.columns:
            columns[kind] = column
    return columns


def _phase_group_columns(frame: pd.DataFrame, group: str, selection: dict[str, Any]) -> dict[str, list[str]]:
    definitions = _phase_group_definitions(selection)
    columns: dict[str, list[str]] = {}
    amount_definition = definitions["amounts"].get(group)
    if amount_definition is not None:
        mass_columns = [f"phase_mass__{name}" for name in _group_items(amount_definition)]
        columns["mass"] = [column for column in mass_columns if column in frame.columns]
    volume_definition = definitions["volumes"].get(group)
    if volume_definition is not None:
        for kind, prefix in [("volume", "phase_volume__"), ("volume_reconstructed", "phase_volume_reconstructed__")]:
            candidates = [f"{prefix}{name}" for name in _group_items(volume_definition)]
            columns[kind] = [column for column in candidates if column in frame.columns]
    return {kind: found for kind, found in columns.items() if found}


def _scalar_column(frame: pd.DataFrame, scalar: str) -> str | None:
    if scalar in BASE_SCALAR_COLUMNS and BASE_SCALAR_COLUMNS[scalar] in frame.columns:
        return BASE_SCALAR_COLUMNS[scalar]
    scalar_column = f"scalar__{scalar}"
    if scalar_column in frame.columns:
        return scalar_column
    if scalar in frame.columns:
        return scalar
    return None


def _available_scalars(frame: pd.DataFrame) -> list[str]:
    names = {name for name, column in BASE_SCALAR_COLUMNS.items() if column in frame.columns}
    for column in frame.columns:
        if column.startswith("scalar__"):
            names.add(column.split("__", 1)[1])
    return sorted(names)


def _top_phase_names(run: Path, frame: pd.DataFrame, limit: int) -> list[str]:
    if limit <= 0:
        return []
    path = run / "phase_nonzero_summary.csv"
    if path.exists():
        try:
            summary = pd.read_csv(path)
            summary = summary[summary.get("nonzero_count", 0).fillna(0).astype(float) > 0]
            if "max_abs_value" in summary:
                summary = summary.sort_values("max_abs_value", ascending=False)
            names: list[str] = []
            for name in summary.get("raw_name", []):
                text = str(name)
                if text not in names:
                    names.append(text)
                if len(names) >= limit:
                    return names
        except Exception:
            pass
    names = _phase_names(frame)
    return names[:limit]


def _selected_rows(
    frame: pd.DataFrame,
    *,
    phases: list[str],
    phase_groups: list[str],
    scalars: list[str],
    selection: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scalar_columns = {name: _scalar_column(frame, name) for name in scalars}
    phase_columns = {name: _phase_columns(frame, name) for name in phases}
    phase_group_columns = {name: _phase_group_columns(frame, name, selection) for name in phase_groups}
    for _, row in frame.iterrows():
        item: dict[str, Any] = {
            "age_days": _clean_value(row.get("age_days")),
            "recipe_id": _clean_value(row.get("recipe_id")),
            "chemistry_status": _clean_value(row.get("chemistry_status")),
            "solver_status": _clean_value(row.get("solver_status")),
            "uncertainty_flags": _clean_value(row.get("uncertainty_flags")),
            "preflight_dir": _clean_value(row.get("preflight_dir")),
            "phases": {},
            "phase_groups": {},
            "scalars": {},
        }
        for phase, columns in phase_columns.items():
            item["phases"][phase] = {kind: _clean_value(row[column]) for kind, column in columns.items()}
        for group, grouped_columns in phase_group_columns.items():
            item["phase_groups"][group] = {
                kind: _clean_value(sum(_numeric_value(row[column]) for column in columns))
                for kind, columns in grouped_columns.items()
            }
        for scalar, column in scalar_columns.items():
            if column:
                item["scalars"][scalar] = _clean_value(row[column])
        rows.append(item)
    return rows


def _wide_selected_table(
    frame: pd.DataFrame,
    *,
    phases: list[str],
    phase_groups: list[str],
    scalars: list[str],
    selection: dict[str, Any],
) -> pd.DataFrame:
    out = pd.DataFrame()
    for column in [
        "age_days",
        "chemistry_status",
        "solver_status",
        "uncertainty_flags",
        "preflight_dir",
        "recipe_id",
    ]:
        if column in frame.columns:
            out[column] = frame[column]
    for phase in phases:
        for kind, column in _phase_columns(frame, phase).items():
            out[f"{phase}__{kind}"] = frame[column]
    for group in phase_groups:
        for kind, columns in _phase_group_columns(frame, group, selection).items():
            out[f"group__{group}__{kind}"] = frame[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
    for scalar in scalars:
        column = _scalar_column(frame, scalar)
        if column:
            out[f"scalar__{scalar}"] = frame[column]
    return out


def _uncertainty_summary(frame: pd.DataFrame) -> dict[str, Any]:
    rows = frame.to_dict(orient="records")
    row_summaries: list[dict[str, Any]] = []
    for row in rows:
        raw_flags = row.get("uncertainty_flags")
        flags = [item for item in str(raw_flags).split(";") if item] if not is_missing(raw_flags) else uncertainty_flags(row)
        preflight = read_preflight_summary(row.get("preflight_dir"))
        row_summaries.append(
            {
                "age_days": _clean_value(row.get("age_days")),
                "recipe_id": _clean_value(row.get("recipe_id")),
                "chemistry_status": _clean_value(row.get("chemistry_status")),
                "solver_status": _clean_value(row.get("solver_status")),
                "solver_rescued": _clean_value(row.get("solver_rescued")),
                "xgems_water_g": _clean_value(row.get("xgems_water_g")),
                "xgems_water_delta_g": _clean_value(row.get("xgems_water_delta_g")),
                "xgems_water_matches_recipe": _clean_value(row.get("xgems_water_matches_recipe")),
                "pH_water_reliable": _clean_value(row.get("pH_water_reliable")),
                "pH_unreliable_reason": _clean_value(row.get("pH_unreliable_reason")),
                "uncertainty_flags": ";".join(flags),
                "preflight_dir": _clean_value(row.get("preflight_dir")),
                "preflight": preflight,
            }
        )
    return {
        "flag_counts": flag_counts(row_summaries),
        "rows": row_summaries,
        "preflight_available_count": sum(1 for row in row_summaries if (row.get("preflight") or {}).get("exists")),
    }


def _markdown_table(frame: pd.DataFrame, limit: int = 20) -> str:
    if frame.empty:
        return "_None._"
    subset = frame.head(limit).fillna("")
    columns = list(subset.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in subset.iterrows():
        lines.append("| " + " | ".join(_markdown_value(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _markdown_value(value: Any) -> str:
    value = _clean_value(value)
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def _write_markdown(
    *,
    path: Path,
    run: Path,
    payload: dict[str, Any],
    selected_table: pd.DataFrame,
    table_limit: int,
) -> None:
    lines = [
        "# Forward Result Summary",
        "",
        f"- run: `{run}`",
        f"- mode: {payload['mode']}",
        f"- rows: {payload['row_count']}",
        f"- completed rows: {payload['completed_count']}",
        f"- failed rows: {payload['failed_count']}",
        "",
        "Raw phase names are preserved exactly in raw outputs. Selected phase groups, when present, are reported separately from raw phases using the configured grouping file.",
        "",
        "## Requested Outputs",
        "",
        f"- phases: {', '.join(payload['requested']['phases']) or '_none_'}",
        f"- phase groups: {', '.join(payload['requested']['phase_groups']) or '_none_'}",
        f"- scalars: {', '.join(payload['requested']['scalars']) or '_none_'}",
        "",
        "## Selected Values",
        "",
        _markdown_table(selected_table, limit=table_limit),
    ]
    missing_phases = payload["missing"]["phases"]
    missing_phase_groups = payload["missing"]["phase_groups"]
    missing_scalars = payload["missing"]["scalars"]
    if missing_phases or missing_phase_groups or missing_scalars:
        lines.extend(["", "## Missing Requests", ""])
        if missing_phases:
            lines.append(f"- missing phases: {', '.join(missing_phases)}")
        if missing_phase_groups:
            lines.append(f"- missing phase groups: {', '.join(missing_phase_groups)}")
        if missing_scalars:
            lines.append(f"- missing scalars: {', '.join(missing_scalars)}")
        lines.extend(
            [
                "",
                "Available phase names:",
                ", ".join(payload["available"]["phases"][:80]) or "_none_",
                "",
                "Available scalar names:",
                ", ".join(payload["available"]["scalars"][:80]) or "_none_",
                "",
                "Available phase group names:",
                ", ".join(payload["available"]["phase_groups"][:80]) or "_none_",
            ]
        )
    failed = payload.get("failed_ages", [])
    if failed:
        lines.extend(["", "## Failed Ages", ""])
        for item in failed:
            lines.append(
                f"- age_days={item.get('age_days')}: {item.get('chemistry_status')} / {item.get('solver_status')}"
            )
    uncertainty = payload.get("uncertainty") or {}
    if uncertainty:
        lines.extend(["", "## Uncertainty And Preflight", ""])
        counts = uncertainty.get("flag_counts") or {}
        lines.append(f"- flag counts: `{counts or {}}`")
        lines.append(f"- preflight reports available: `{uncertainty.get('preflight_available_count', 0)}`")
        rows = pd.DataFrame(uncertainty.get("rows") or [])
        if not rows.empty:
            lines.extend(
                [
                    "",
                    _markdown_table(
                        rows[
                            [
                                column
                                for column in [
                                    "age_days",
                                    "uncertainty_flags",
                                    "pH_water_reliable",
                                    "xgems_water_delta_g",
                                    "preflight_dir",
                                ]
                                if column in rows.columns
                            ]
                        ],
                        limit=table_limit,
                    ),
                ]
            )
    lines.extend(["", "## Files", ""])
    for label, file_path in sorted(payload["files"].items()):
        lines.append(f"- {label}: `{file_path}`")
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize_forward_result(
    *,
    run: str | Path,
    phases: list[str] | None = None,
    phase_groups: list[str] | None = None,
    scalars: list[str] | None = None,
    selection: str | Path | None = None,
    out: str | Path | None = None,
    top_phases: int = 8,
    table_limit: int = 20,
) -> Path:
    run_path = Path(run)
    out_dir = Path(out) if out is not None else run_path
    out_dir.mkdir(parents=True, exist_ok=True)
    selection_path = _selection_path(selection)
    selection_data = _load_selection(selection)
    frame = _load_time_series(run_path)
    if "age_days" in frame.columns:
        frame = frame.copy()
        frame["_sort_age_days"] = pd.to_numeric(frame["age_days"], errors="coerce")
        sort_columns = ["_sort_age_days"]
        if "row_index" in frame.columns:
            sort_columns.append("row_index")
        frame = frame.sort_values(sort_columns, kind="mergesort").drop(columns=["_sort_age_days"])
    else:
        frame = frame.reset_index(drop=True)

    requested_phases = list(phases or [])
    phase_request_source = "explicit"
    if not requested_phases:
        requested_phases = _top_phase_names(run_path, frame, top_phases)
        phase_request_source = "top_nonzero_default"

    requested_scalars = list(scalars or DEFAULT_SCALARS)
    scalar_request_source = "explicit" if scalars else "default"
    requested_phase_groups, phase_group_request_source = _requested_phase_groups(phase_groups, selection_data)

    available_phases = _phase_names(frame)
    available_phase_groups = _available_phase_groups(selection_data)
    available_scalars = _available_scalars(frame)
    missing_phases = [phase for phase in requested_phases if not _phase_columns(frame, phase)]
    missing_phase_groups = [group for group in requested_phase_groups if not _phase_group_columns(frame, group, selection_data)]
    missing_scalars = [scalar for scalar in requested_scalars if _scalar_column(frame, scalar) is None]

    selected_table = _wide_selected_table(
        frame,
        phases=requested_phases,
        phase_groups=requested_phase_groups,
        scalars=requested_scalars,
        selection=selection_data,
    )
    selected_rows = _selected_rows(
        frame,
        phases=requested_phases,
        phase_groups=requested_phase_groups,
        scalars=requested_scalars,
        selection=selection_data,
    )

    chemistry_status = frame.get("chemistry_status", pd.Series([""] * len(frame))).astype(str).str.lower()
    solver_status = frame.get("solver_status", pd.Series([""] * len(frame))).astype(str).str.lower()
    failed_mask = (
        chemistry_status.str.contains("fail|error", regex=True, na=False)
        | solver_status.str.contains("fail|error", regex=True, na=False)
    )
    completed_mask = (
        chemistry_status.str.contains("complete|success", regex=True, na=False)
        | solver_status.str.contains("complete|success|solved|ok", regex=True, na=False)
    ) & ~failed_mask

    failed_ages: list[dict[str, Any]] = []
    for _, row in frame[failed_mask].iterrows():
        failed_ages.append(
            {
                "age_days": _clean_value(row.get("age_days")),
                "recipe_id": _clean_value(row.get("recipe_id")),
                "chemistry_status": _clean_value(row.get("chemistry_status")),
                "solver_status": _clean_value(row.get("solver_status")),
            }
        )

    json_path = out_dir / "response_summary.json"
    md_path = out_dir / "response_summary.md"
    csv_path = out_dir / "response_summary.csv"
    payload: dict[str, Any] = {
        "run": str(run_path),
        "mode": "single" if len(frame) == 1 else "time_series",
        "row_count": int(len(frame)),
        "completed_count": int(completed_mask.sum()),
        "failed_count": int(failed_mask.sum()),
        "requested": {
            "phases": requested_phases,
            "phase_request_source": phase_request_source,
            "phase_groups": requested_phase_groups,
            "phase_group_request_source": phase_group_request_source,
            "scalars": requested_scalars,
            "scalar_request_source": scalar_request_source,
        },
        "selected": {"rows": selected_rows},
        "missing": {"phases": missing_phases, "phase_groups": missing_phase_groups, "scalars": missing_scalars},
        "available": {"phases": available_phases, "phase_groups": available_phase_groups, "scalars": available_scalars},
        "selected_output_policy": {
            "raw_phase_names_preserved": True,
            "phase_groups_are_configured_sums": True,
            "phase_groups_do_not_rename_raw_outputs": True,
            "selection_config": str(selection_path),
        },
        "failed_ages": failed_ages,
        "uncertainty": _uncertainty_summary(frame),
        "source_metadata": {
            "manifest": _load_json(run_path / "manifest.json"),
            "query_summary": _load_json(run_path / "query_summary.json"),
            "output_selection": str(selection_path),
        },
        "files": {
            "source_time_series": str(run_path / "time_series.csv"),
            "response_summary_csv": str(csv_path),
            "response_summary_json": str(json_path),
            "response_summary_md": str(md_path),
        },
    }

    selected_table.to_csv(csv_path, index=False)
    write_json(json_path, payload)
    _write_markdown(path=md_path, run=run_path, payload=payload, selected_table=selected_table, table_limit=table_limit)
    return out_dir
