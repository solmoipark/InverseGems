from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .utils import write_json


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Response summary JSON does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if _is_number(value):
        return f"{float(value):.6g}"
    return str(value).replace("|", "\\|")


def _flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {
        "age_days": row.get("age_days"),
        "chemistry_status": row.get("chemistry_status"),
        "solver_status": row.get("solver_status"),
    }
    for phase, values in (row.get("phases") or {}).items():
        for metric, value in (values or {}).items():
            flat[f"{phase}.{metric}"] = value
    for group, values in (row.get("phase_groups") or {}).items():
        for metric, value in (values or {}).items():
            flat[f"phase_group.{group}.{metric}"] = value
    for scalar, value in (row.get("scalars") or {}).items():
        flat[f"scalar.{scalar}"] = value
    return flat


def _numeric_series(rows: list[dict[str, Any]]) -> dict[str, list[tuple[Any, float]]]:
    series: dict[str, list[tuple[Any, float]]] = {}
    for row in rows:
        age = row.get("age_days")
        for name, value in _flatten_row(row).items():
            if name in {"age_days", "chemistry_status", "solver_status"}:
                continue
            if _is_number(value):
                series.setdefault(name, []).append((age, float(value)))
    return series


def _series_highlights(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    highlights: list[dict[str, Any]] = []
    for name, values in sorted(_numeric_series(rows).items()):
        if not values:
            continue
        first_age, first = values[0]
        final_age, final = values[-1]
        min_age, min_value = min(values, key=lambda item: item[1])
        max_age, max_value = max(values, key=lambda item: item[1])
        highlights.append(
            {
                "output": name,
                "first": first,
                "first_age_days": first_age,
                "final": final,
                "final_age_days": final_age,
                "min": min_value,
                "min_age_days": min_age,
                "max": max_value,
                "max_age_days": max_age,
            }
        )
    return highlights


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    columns: list[str] = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def _build_markdown(answer: dict[str, Any]) -> str:
    requested = answer["requested"]
    missing = answer["missing"]
    lines = [
        "# Forward Answer",
        "",
        answer["headline"],
        "",
        "Raw phase names are preserved exactly in raw outputs. Selected phase groups are reported separately when configured.",
        "",
        "## Requested",
        "",
        f"- phases: {', '.join(requested['phases']) or '_none_'}",
        f"- phase groups: {', '.join(requested.get('phase_groups') or []) or '_none_'}",
        f"- scalars: {', '.join(requested['scalars']) or '_none_'}",
        "",
        "## Results Preview",
        "",
        _markdown_table(answer["rows_preview"]),
    ]
    if answer["series_highlights"]:
        lines.extend(
            [
                "",
                "## Numeric Summary",
                "",
                _markdown_table(answer["series_highlights"]),
            ]
        )
    if missing["phases"] or missing.get("phase_groups") or missing["scalars"]:
        lines.extend(["", "## Missing Requests", ""])
        if missing["phases"]:
            lines.append(f"- missing phases: {', '.join(missing['phases'])}")
        if missing.get("phase_groups"):
            lines.append(f"- missing phase groups: {', '.join(missing['phase_groups'])}")
        if missing["scalars"]:
            lines.append(f"- missing scalars: {', '.join(missing['scalars'])}")
    if answer["failed_ages"]:
        lines.extend(["", "## Failed Ages", ""])
        for item in answer["failed_ages"]:
            lines.append(
                f"- age_days={item.get('age_days')}: {item.get('chemistry_status')} / {item.get('solver_status')}"
            )
    uncertainty = answer.get("uncertainty") or {}
    if uncertainty:
        lines.extend(
            [
                "",
                "## Uncertainty And Preflight",
                "",
                f"- flag counts: `{uncertainty.get('flag_counts') or {}}`",
                f"- preflight reports available: `{uncertainty.get('preflight_available_count', 0)}`",
            ]
        )
        rows = uncertainty.get("rows_preview") or []
        if rows:
            lines.extend(["", _markdown_table(rows)])
    lines.extend(["", "## Files", ""])
    for label, file_path in sorted(answer["files"].items()):
        lines.append(f"- {label}: `{file_path}`")
    return "\n".join(lines) + "\n"


def build_forward_answer(
    response_summary: dict[str, Any],
    *,
    summary_path: str | Path | None = None,
    table_limit: int = 10,
) -> dict[str, Any]:
    rows = list((response_summary.get("selected") or {}).get("rows") or [])
    preview = [_flatten_row(row) for row in rows[: max(int(table_limit), 1)]]
    row_count = int(response_summary.get("row_count") or len(rows))
    completed_count = int(response_summary.get("completed_count") or 0)
    failed_count = int(response_summary.get("failed_count") or 0)
    mode = str(response_summary.get("mode") or ("single" if row_count == 1 else "time_series"))
    if failed_count:
        headline = f"Forward {mode} completed {completed_count} of {row_count} rows; {failed_count} row(s) failed."
    else:
        headline = f"Forward {mode} completed successfully for {row_count} row(s)."
    files = dict(response_summary.get("files") or {})
    if summary_path is not None:
        files["source_response_summary_json"] = str(summary_path)
    return {
        "answer_type": "forward_result_answer",
        "mode": mode,
        "headline": headline,
        "row_count": row_count,
        "completed_count": completed_count,
        "failed_count": failed_count,
        "requested": response_summary.get("requested") or {"phases": [], "phase_groups": [], "scalars": []},
        "missing": response_summary.get("missing") or {"phases": [], "phase_groups": [], "scalars": []},
        "rows_preview": preview,
        "series_highlights": _series_highlights(rows),
        "failed_ages": response_summary.get("failed_ages") or [],
        "uncertainty": {
            "flag_counts": (response_summary.get("uncertainty") or {}).get("flag_counts") or {},
            "preflight_available_count": (response_summary.get("uncertainty") or {}).get("preflight_available_count", 0),
            "rows_preview": [
                {
                    "age_days": row.get("age_days"),
                    "uncertainty_flags": row.get("uncertainty_flags"),
                    "pH_water_reliable": row.get("pH_water_reliable"),
                    "xgems_water_delta_g": row.get("xgems_water_delta_g"),
                    "preflight_dir": row.get("preflight_dir"),
                }
                for row in ((response_summary.get("uncertainty") or {}).get("rows") or [])[: max(int(table_limit), 1)]
            ],
        },
        "raw_phase_policy": "Raw xGEMS/GEMS names are preserved exactly; selected phase groups are reported separately from raw phases.",
        "files": files,
    }


def write_forward_answer(
    *,
    summary: str | Path,
    out: str | Path | None = None,
    table_limit: int = 10,
) -> Path:
    summary_path = Path(summary)
    out_dir = Path(out) if out is not None else summary_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = _load_json(summary_path)
    answer = build_forward_answer(payload, summary_path=summary_path, table_limit=table_limit)
    json_path = out_dir / "answer.json"
    md_path = out_dir / "answer.md"
    answer["files"]["answer_json"] = str(json_path)
    answer["files"]["answer_markdown"] = str(md_path)
    write_json(json_path, answer)
    md_path.write_text(_build_markdown(answer), encoding="utf-8")
    return out_dir
