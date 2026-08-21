from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import config_path, load_yaml, project_root, write_json


EPS = 1.0e-30


def _load_model_registry(path: str | Path | None = None) -> dict[str, Any]:
    registry_path = Path(path) if path is not None else config_path("design_query_model_registry.global_v1.yaml")
    if not registry_path.exists():
        return {"models": [], "path": str(registry_path)}
    data = load_yaml(registry_path)
    data["path"] = str(registry_path)
    return data


def _resolve_relative(path_value: Any, *, registry_dir: Path) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    candidates = [
        registry_dir / path,
        project_root() / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return project_root() / path


def _schema_path(model_table: Path) -> Path:
    return Path(str(model_table) + ".schema.json")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_csv(path)
    return pd.read_csv(path)


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip().lower() in {"true", "1", "1.0", "yes", "y"}


def _summary_from_series(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric[numeric.notna() & numeric.apply(lambda value: math.isfinite(float(value)))]
    if finite.empty:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "nonzero_count": 0,
            "nonzero_fraction": 0.0,
        }
    nonzero = finite.abs() > EPS
    return {
        "count": int(finite.count()),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "median": float(finite.median()),
        "nonzero_count": int(nonzero.sum()),
        "nonzero_fraction": float(nonzero.mean()),
    }


def _target_label(column: str, spec: dict[str, Any] | None) -> str:
    if spec and spec.get("name"):
        return str(spec["name"])
    label = str(column)
    for prefix in ("y__amount_", "y__volume_", "y__scalar_", "y__"):
        if label.startswith(prefix):
            return label.removeprefix(prefix)
    return label


def _target_kind(column: str, spec: dict[str, Any] | None) -> str:
    if spec and spec.get("kind"):
        return str(spec["kind"])
    if column == "y__porosity" or column == "y__pH":
        return "scalar"
    if column.startswith("y__amount_"):
        return "phase_amount_group"
    if column.startswith("y__volume_"):
        return "phase_volume_group"
    return ""


def _metrics_by_target(model_bundle: Path) -> tuple[dict[str, dict[str, Any]], Path | None]:
    metrics_path = model_bundle.parent / "target_metrics.csv"
    if not metrics_path.exists():
        return {}, None
    metrics = pd.read_csv(metrics_path)
    if "target" not in metrics.columns:
        return {}, metrics_path
    return {str(row["target"]): row.to_dict() for _, row in metrics.iterrows()}, metrics_path


def _manifest_path(model_bundle: Path) -> Path | None:
    for name in ("surrogate_model_manifest.json", "surrogate_summary.json"):
        path = model_bundle.parent / name
        if path.exists():
            return path
    return None


def _target_columns(
    *,
    schema: dict[str, Any],
    metrics: dict[str, dict[str, Any]],
    model_table: Path,
    warnings: list[str],
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, Any]]:
    specs = {
        str(spec.get("column")): dict(spec)
        for spec in (schema.get("roles", {}) or {}).get("targets", [])
        if spec.get("column")
    }
    target_summary = dict(schema.get("target_summary") or {})
    columns = list(dict.fromkeys(list(specs) + list(target_summary) + list(metrics)))
    fallback_summary: dict[str, Any] = {}

    if not columns and model_table.exists():
        frame = _read_table(model_table)
        columns = [column for column in frame.columns if str(column).startswith("y__")]
        fallback_summary = {column: _summary_from_series(frame[column]) for column in columns}
    elif any(column not in target_summary for column in columns) and model_table.exists():
        missing = [column for column in columns if column not in target_summary]
        if missing:
            frame = _read_table(model_table)
            for column in missing:
                if column in frame.columns:
                    fallback_summary[column] = _summary_from_series(frame[column])

    if not columns:
        warnings.append(f"No target columns found for model table {model_table}.")
    return columns, specs, {**fallback_summary, **target_summary}


def _range_threshold(label: str, column: str, *, min_range: float, ph_min_range: float) -> float:
    normalized = str(label or column).lower()
    if normalized == "ph" or column == "y__pH":
        return ph_min_range
    return min_range


def _classify_target(
    *,
    label: str,
    column: str,
    count: int | None,
    minimum: float | None,
    maximum: float | None,
    value_range: float | None,
    nonzero_fraction: float | None,
    r2: float | None,
    min_r2: float,
    sparse_threshold: float,
    min_range: float,
    ph_min_range: float,
    metrics_present: bool,
) -> tuple[str, list[str], dict[str, bool]]:
    flags = {
        "all_zero": False,
        "sparse": False,
        "near_constant": False,
        "low_r2": False,
        "missing_metrics": False,
    }
    reasons: list[str] = []

    if count is not None and count <= 0:
        flags["all_zero"] = True
        reasons.append("no finite values")

    if nonzero_fraction is not None:
        if nonzero_fraction <= 0.0:
            flags["all_zero"] = True
            reasons.append("all zero")
        elif nonzero_fraction < sparse_threshold:
            flags["sparse"] = True
            reasons.append(f"sparse nonzero fraction {nonzero_fraction:.4g}")

    threshold = _range_threshold(label, column, min_range=min_range, ph_min_range=ph_min_range)
    if value_range is not None and value_range < threshold:
        flags["near_constant"] = True
        reasons.append(f"range {value_range:.4g} below threshold {threshold:.4g}")

    if minimum is not None and maximum is not None and abs(minimum) <= EPS and abs(maximum) <= EPS:
        flags["all_zero"] = True
        if "all zero" not in reasons:
            reasons.append("all zero")

    if not metrics_present:
        flags["missing_metrics"] = True
        reasons.append("missing surrogate metrics")
    elif r2 is None:
        flags["low_r2"] = True
        reasons.append("missing R2")
    elif r2 < min_r2 and not flags["near_constant"] and not flags["all_zero"]:
        flags["low_r2"] = True
        reasons.append(f"R2 {r2:.4g} below threshold {min_r2:.4g}")

    if flags["all_zero"] or flags["sparse"] or flags["near_constant"] or flags["missing_metrics"]:
        status = "not_recommended"
    elif flags["low_r2"]:
        status = "usable_with_caution"
    else:
        status = "recommended"

    if not reasons:
        reasons.append("target has variation and surrogate metrics pass thresholds")
    return status, reasons, flags


def _diagnose_entry(
    entry: dict[str, Any],
    *,
    registry_dir: Path,
    min_r2: float,
    sparse_threshold: float,
    min_range: float,
    ph_min_range: float,
    warnings: list[str],
) -> list[dict[str, Any]]:
    model_table_value = entry.get("model_table")
    model_bundle_value = entry.get("model_bundle")
    if not model_table_value or not model_bundle_value:
        warnings.append(f"Registry entry {entry.get('id', '<unknown>')} has no model_table/model_bundle.")
        return []

    model_table = _resolve_relative(model_table_value, registry_dir=registry_dir)
    model_bundle = _resolve_relative(model_bundle_value, registry_dir=registry_dir)
    schema_file = _schema_path(model_table)
    schema: dict[str, Any] = {}
    if schema_file.exists():
        schema = _read_json(schema_file)
    else:
        warnings.append(f"Missing schema file for {entry.get('id')}: {schema_file}")

    metrics, metrics_path = _metrics_by_target(model_bundle)
    if metrics_path is None:
        warnings.append(f"Missing target_metrics.csv for {entry.get('id')}: {model_bundle.parent}")

    manifest_file = _manifest_path(model_bundle)
    manifest = _read_json(manifest_file) if manifest_file else {}
    target_columns, target_specs, target_summary = _target_columns(
        schema=schema,
        metrics=metrics,
        model_table=model_table,
        warnings=warnings,
    )

    row_count = (
        _as_int((schema.get("filters") or {}).get("rows_after_status_filter"))
        or _as_int(schema.get("row_count"))
        or _as_int((manifest.get("data") or {}).get("row_count"))
    )

    rows: list[dict[str, Any]] = []
    for column in target_columns:
        spec = target_specs.get(column, {})
        summary = target_summary.get(column, {}) or {}
        metric = metrics.get(column, {})
        label = _target_label(column, spec)
        kind = _target_kind(column, spec)
        minimum = _as_float(summary.get("min", metric.get("true_min")))
        maximum = _as_float(summary.get("max", metric.get("true_max")))
        value_range = None if minimum is None or maximum is None else maximum - minimum
        nonzero_count = _as_int(summary.get("nonzero_count", metric.get("nonzero_true_count")))
        count = _as_int(summary.get("count")) or _as_int(metric.get("n_test"))
        nonzero_fraction = _as_float(summary.get("nonzero_fraction"))
        if nonzero_fraction is None and count:
            nonzero_fraction = None if nonzero_count is None else nonzero_count / count
        r2 = _as_float(metric.get("r2"))
        full_nonzero_count = _as_int(metric.get("full_nonzero_count"))
        full_nonzero_fraction = _as_float(metric.get("full_nonzero_fraction"))
        train_nonzero_count = _as_int(metric.get("train_nonzero_count"))
        train_nonzero_fraction = _as_float(metric.get("train_nonzero_fraction"))
        test_nonzero_count = _as_int(metric.get("nonzero_true_count"))
        test_nonzero_fraction = _as_float(metric.get("test_nonzero_fraction"))
        test_nonzero_coverage_fraction = _as_float(metric.get("test_nonzero_coverage_fraction"))
        sparse_target_metric = _as_bool(metric.get("sparse_target"))
        test_nonzero_missing = _as_bool(metric.get("test_nonzero_missing"))
        test_nonzero_too_low = _as_bool(metric.get("test_nonzero_too_low"))
        evaluation_warning = _as_bool(metric.get("evaluation_warning"))
        evaluation_reliability = str(metric.get("evaluation_reliability", "") or "")
        diagnostic_warning = str(metric.get("diagnostic_warning", "") or "")
        pH_water_unreliable_count = _as_int(summary.get("pH_water_unreliable_count"))
        pH_water_unreliable_fraction = _as_float(summary.get("pH_water_unreliable_fraction"))
        pH_xgems_water_changed_count = _as_int(summary.get("pH_xgems_water_changed_count"))
        pH_xgems_water_delta_g_max_abs = _as_float(summary.get("pH_xgems_water_delta_g_max_abs"))
        status, reasons, flags = _classify_target(
            label=label,
            column=column,
            count=count,
            minimum=minimum,
            maximum=maximum,
            value_range=value_range,
            nonzero_fraction=nonzero_fraction,
            r2=r2,
            min_r2=min_r2,
            sparse_threshold=sparse_threshold,
            min_range=min_range,
            ph_min_range=ph_min_range,
            metrics_present=bool(metric),
        )
        if evaluation_warning:
            if diagnostic_warning and diagnostic_warning.lower() != "nan":
                reasons.append(diagnostic_warning)
            else:
                reasons.append("surrogate evaluation has limited nonzero support")
            if status == "recommended":
                status = "usable_with_caution"
        is_ph = str(label).lower() == "ph" or column == "y__pH"
        pH_water_adjusted = bool(is_ph and pH_water_unreliable_fraction is not None and pH_water_unreliable_fraction > 0.0)
        if pH_water_adjusted:
            reasons.append(
                "pH includes water-adjusted/rescued rows "
                f"({pH_water_unreliable_fraction:.4g} of model rows)"
            )
            if status == "recommended":
                status = "usable_with_caution"
        rows.append(
            {
                "registry_id": entry.get("id", ""),
                "material_system": entry.get("material_system", ""),
                "age_days": entry.get("age_days", ""),
                "reaction_model_id": entry.get("reaction_model_id", ""),
                "reaction_model_signature": entry.get("reaction_model_signature", ""),
                "model_table": str(model_table),
                "model_bundle": str(model_bundle),
                "schema_path": str(schema_file) if schema_file.exists() else "",
                "target_metrics_path": str(metrics_path) if metrics_path else "",
                "manifest_path": str(manifest_file) if manifest_file else "",
                "row_count": row_count,
                "target_column": column,
                "target_label": label,
                "target_kind": kind,
                "count": count,
                "min": minimum,
                "max": maximum,
                "range": value_range,
                "mean": _as_float(summary.get("mean", metric.get("true_mean"))),
                "median": _as_float(summary.get("median", metric.get("true_median"))),
                "nonzero_count": nonzero_count,
                "nonzero_fraction": nonzero_fraction,
                "full_nonzero_count": full_nonzero_count,
                "full_nonzero_fraction": full_nonzero_fraction,
                "train_nonzero_count": train_nonzero_count,
                "train_nonzero_fraction": train_nonzero_fraction,
                "test_nonzero_count": test_nonzero_count,
                "test_nonzero_fraction": test_nonzero_fraction,
                "test_nonzero_coverage_fraction": test_nonzero_coverage_fraction,
                "r2": r2,
                "mae": _as_float(metric.get("mae")),
                "rmse": _as_float(metric.get("rmse")),
                "baseline_mean_mae": _as_float(metric.get("baseline_mean_mae")),
                "baseline_mean_rmse": _as_float(metric.get("baseline_mean_rmse")),
                "precision_nonzero": _as_float(metric.get("precision_nonzero")),
                "recall_nonzero": _as_float(metric.get("recall_nonzero")),
                "f1_nonzero": _as_float(metric.get("f1_nonzero")),
                "pH_water_unreliable_count": pH_water_unreliable_count,
                "pH_water_unreliable_fraction": pH_water_unreliable_fraction,
                "pH_xgems_water_changed_count": pH_xgems_water_changed_count,
                "pH_xgems_water_delta_g_max_abs": pH_xgems_water_delta_g_max_abs,
                "sparse_target_metric": sparse_target_metric,
                "test_nonzero_missing": test_nonzero_missing,
                "test_nonzero_too_low": test_nonzero_too_low,
                "evaluation_warning": evaluation_warning,
                "evaluation_reliability": evaluation_reliability,
                "diagnostic_warning": diagnostic_warning,
                "all_zero": flags["all_zero"],
                "sparse": flags["sparse"],
                "near_constant": flags["near_constant"],
                "low_r2": flags["low_r2"],
                "missing_metrics": flags["missing_metrics"],
                "pH_water_adjusted": pH_water_adjusted,
                "status": status,
                "reasons": "; ".join(dict.fromkeys(reasons)),
            }
        )
    return rows


def diagnose_model_target_availability(
    *,
    model_table: str | Path,
    model_bundle: str | Path,
    registry_entry: dict[str, Any] | None = None,
    registry_dir: str | Path | None = None,
    min_r2: float = 0.70,
    sparse_threshold: float = 0.01,
    min_range: float = 1.0e-8,
    ph_min_range: float = 0.01,
) -> tuple[pd.DataFrame, list[str]]:
    """Return target diagnostics for one resolved model without writing files."""
    entry = dict(registry_entry or {})
    entry["model_table"] = str(model_table)
    entry["model_bundle"] = str(model_bundle)
    base_dir: Path
    if registry_dir is not None:
        base_dir = Path(registry_dir)
    elif entry.get("_registry_path"):
        base_dir = Path(str(entry["_registry_path"])).parent
    else:
        base_dir = Path.cwd()
    warnings: list[str] = []
    rows = _diagnose_entry(
        entry,
        registry_dir=base_dir,
        min_r2=min_r2,
        sparse_threshold=sparse_threshold,
        min_range=min_range,
        ph_min_range=ph_min_range,
        warnings=warnings,
    )
    return pd.DataFrame(rows), warnings


def _write_markdown(frame: pd.DataFrame, out_path: Path, summary: dict[str, Any]) -> None:
    def markdown_table(table: pd.DataFrame) -> str:
        if table.empty:
            return ""
        text = table.fillna("").astype(str)
        headers = list(text.columns)
        rows = [headers] + text.values.tolist()
        widths = [max(len(str(row[index])) for row in rows) for index in range(len(headers))]

        def render(row: list[str]) -> str:
            return "| " + " | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)) + " |"

        separator = "| " + " | ".join("-" * width for width in widths) + " |"
        return "\n".join([render(headers), separator] + [render(row) for row in text.values.tolist()])

    lines = [
        "# Model Registry Diagnostics",
        "",
        f"- registry models checked: {summary['model_count']}",
        f"- target rows: {summary['target_count']}",
        f"- recommended: {summary['status_counts'].get('recommended', 0)}",
        f"- usable_with_caution: {summary['status_counts'].get('usable_with_caution', 0)}",
        f"- not_recommended: {summary['status_counts'].get('not_recommended', 0)}",
        "",
    ]
    if not frame.empty:
        status = frame.groupby(["material_system", "status"], dropna=False).size().reset_index(name="count")
        lines.extend(["## Status By Material System", ""])
        lines.append(markdown_table(status))
        lines.append("")
        flagged = frame[frame["status"] != "recommended"].copy()
        if not flagged.empty:
            columns = [
                "registry_id",
                "target_label",
                "status",
                "range",
                "nonzero_fraction",
                "test_nonzero_count",
                "evaluation_reliability",
                "r2",
                "reasons",
            ]
            lines.extend(["## Targets To Treat Carefully", ""])
            lines.append(markdown_table(flagged[columns]))
            lines.append("")
    if summary.get("warnings"):
        lines.extend(["## Warnings", ""])
        for warning in summary["warnings"]:
            lines.append(f"- {warning}")
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_model_registry_diagnostics(
    *,
    model_registry: str | Path | None = None,
    out: str | Path,
    reaction_model_id: str | None = None,
    reaction_model_signature: str | None = None,
    min_r2: float = 0.70,
    sparse_threshold: float = 0.01,
    min_range: float = 1.0e-8,
    ph_min_range: float = 0.01,
) -> Path:
    registry = _load_model_registry(model_registry)
    registry_path = Path(str(registry.get("path") or model_registry or "configs/design_query_model_registry.global_v1.yaml"))
    registry_dir = registry_path.parent if registry_path.parent != Path("") else Path.cwd()
    entries = list(registry.get("models") or registry.get("entries") or [])
    if reaction_model_id is not None:
        entries = [entry for entry in entries if str(entry.get("reaction_model_id", "")) == str(reaction_model_id)]
    if reaction_model_signature is not None:
        entries = [
            entry
            for entry in entries
            if str(entry.get("reaction_model_signature", "")) == str(reaction_model_signature)
        ]

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    for entry in entries:
        rows.extend(
            _diagnose_entry(
                entry,
                registry_dir=registry_dir,
                min_r2=min_r2,
                sparse_threshold=sparse_threshold,
                min_range=min_range,
                ph_min_range=ph_min_range,
                warnings=warnings,
            )
        )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["material_system", "registry_id", "status", "target_label"]).reset_index(drop=True)
    csv_path = out_dir / "model_registry_diagnostics.csv"
    frame.to_csv(csv_path, index=False)
    frame.to_json(out_dir / "model_registry_diagnostics.json", orient="records", indent=2)

    status_counts = frame["status"].value_counts(dropna=False).to_dict() if not frame.empty else {}
    summary = {
        "registry": str(registry_path),
        "model_count": len(entries),
        "target_count": int(len(frame)),
        "filters": {
            "reaction_model_id": reaction_model_id,
            "reaction_model_signature": reaction_model_signature,
        },
        "thresholds": {
            "min_r2": min_r2,
            "sparse_threshold": sparse_threshold,
            "min_range": min_range,
            "ph_min_range": ph_min_range,
        },
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "warnings": warnings,
        "files": {
            "csv": str(csv_path),
            "json": str(out_dir / "model_registry_diagnostics.json"),
            "markdown": str(out_dir / "model_registry_diagnostics.md"),
        },
    }
    write_json(out_dir / "model_registry_diagnostics_summary.json", summary)
    _write_markdown(frame, out_dir / "model_registry_diagnostics.md", summary)
    return out_dir
