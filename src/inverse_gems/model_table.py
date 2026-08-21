from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .reaction_provenance import ensure_reaction_metadata_columns, reaction_provenance_from_frame
from .utils import load_yaml, write_json


SAFE_NAME_RE = re.compile(r"[^0-9A-Za-z_]+")


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_csv(path)
    return pd.read_csv(path)


def _write_table(frame: pd.DataFrame, path: Path, output_format: str | None) -> tuple[Path, list[str]]:
    warnings: list[str] = []
    fmt = (output_format or path.suffix.lower().lstrip(".") or "csv").lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        try:
            frame.to_parquet(path, index=False)
            return path, warnings
        except Exception as exc:
            fallback = path.with_suffix(".csv")
            frame.to_csv(fallback, index=False)
            warnings.append(
                f"Could not write parquet to {path}; wrote CSV fallback {fallback}. "
                f"Original error: {type(exc).__name__}: {exc}"
            )
            return fallback, warnings
    frame.to_csv(path, index=False)
    return path, warnings


def _safe_name(name: str, *, sanitize: bool = True) -> str:
    if not sanitize:
        return str(name)
    cleaned = SAFE_NAME_RE.sub("_", str(name)).strip("_")
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned or "unnamed"


def _target_source_for_scalar(name: str, source: Any) -> str:
    if source:
        return str(source)
    if name == "pH":
        return "scalar__pH"
    if name == "porosity":
        return "porosity"
    return f"scalar__{name}"


def _phase_group_targets(config: dict[str, Any], section: str, prefix: str) -> list[dict[str, str]]:
    raw = config.get("targets", {}).get(section, {}) or {}
    include = raw.get("include", []) if isinstance(raw, dict) else raw
    if isinstance(include, str):
        include = [include]
    return [
        {
            "name": str(name),
            "source": f"{prefix}{name}",
            "kind": section,
        }
        for name in include
    ]


def _target_specs(config: dict[str, Any]) -> list[dict[str, str]]:
    targets = config.get("targets", {}) or {}
    specs: list[dict[str, str]] = []
    scalars = targets.get("scalars", {}) or {}
    if isinstance(scalars, list):
        scalars = {str(name): None for name in scalars}
    for name, source in scalars.items():
        specs.append({"name": str(name), "source": _target_source_for_scalar(str(name), source), "kind": "scalar"})
    specs.extend(_phase_group_targets(config, "phase_amount_groups", "phase_amount_group__"))
    specs.extend(_phase_group_targets(config, "phase_volume_groups", "phase_volume_group__"))
    return specs


def _derived_input(frame: pd.DataFrame, name: str, spec: dict[str, Any]) -> pd.Series:
    source = str(spec.get("source") or "")
    if source not in frame.columns:
        raise ValueError(f"Derived input '{name}' refers to missing source column '{source}'.")
    values = pd.to_numeric(frame[source], errors="coerce")
    transform = str(spec.get("transform") or "identity")
    if transform == "identity":
        return values
    if transform == "log10":
        if (values <= 0).any():
            raise ValueError(f"Derived input '{name}' cannot apply log10 to non-positive values.")
        return np.log10(values)
    if transform == "ln":
        if (values <= 0).any():
            raise ValueError(f"Derived input '{name}' cannot apply ln to non-positive values.")
        return np.log(values)
    if transform == "sqrt":
        if (values < 0).any():
            raise ValueError(f"Derived input '{name}' cannot apply sqrt to negative values.")
        return np.sqrt(values)
    raise ValueError(f"Unsupported derived input transform '{transform}' for '{name}'.")


def _ensure_columns(frame: pd.DataFrame, columns: list[str], role: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing {role} columns in feature table: {missing}")


def _numeric_or_original(frame: pd.DataFrame, column: str) -> pd.Series:
    converted = pd.to_numeric(frame[column], errors="coerce")
    if converted.notna().any():
        return converted
    return frame[column]


def _as_bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower()
    if text in {"", "0", "0.0", "false", "no", "none", "nan"}:
        return False
    return True


def _drop_missing(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, int]:
    before = len(frame)
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    keep = numeric.notna().all(axis=1) & np.isfinite(numeric).all(axis=1)
    return frame[keep].copy(), before - int(keep.sum())


def _normalize_include_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def _apply_column_filter(frame: pd.DataFrame, column: str, spec: Any) -> tuple[pd.DataFrame, int]:
    if column not in frame.columns:
        raise ValueError(f"Filter requested missing feature column '{column}'.")
    if not isinstance(spec, dict):
        spec = {"include": spec}
    before = len(frame)
    values = frame[column]
    if spec.get("include") is not None:
        include = _normalize_include_values(spec["include"])
        frame = frame[values.astype(str).isin(include)].copy()
        values = frame[column]
    if spec.get("exclude") is not None:
        exclude = _normalize_include_values(spec["exclude"])
        frame = frame[~values.astype(str).isin(exclude)].copy()
        values = frame[column]
    numeric = pd.to_numeric(values, errors="coerce")
    if spec.get("min") is not None:
        frame = frame[numeric >= float(spec["min"])].copy()
        values = frame[column]
        numeric = pd.to_numeric(values, errors="coerce")
    if spec.get("max") is not None:
        frame = frame[numeric <= float(spec["max"])].copy()
        values = frame[column]
        numeric = pd.to_numeric(values, errors="coerce")
    if spec.get("equals") is not None:
        try:
            equals = float(spec["equals"])
            frame = frame[(pd.to_numeric(values, errors="coerce") - equals).abs() <= float(spec.get("tolerance", 1.0e-12))].copy()
        except (TypeError, ValueError):
            frame = frame[values.astype(str) == str(spec["equals"])].copy()
    return frame, before - len(frame)


def _apply_view_filters(frame: pd.DataFrame, filters: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int], list[dict[str, Any]]]:
    rejected: dict[str, int] = {}
    resolved: list[dict[str, Any]] = []
    view_filters = filters.get("columns") or filters.get("view") or {}
    for column, spec in (view_filters or {}).items():
        frame, count = _apply_column_filter(frame, str(column), spec)
        rejected[str(column)] = int(count)
        resolved.append({"column": str(column), "spec": spec, "rejected": int(count)})
    return frame, rejected, resolved


def _series_summary(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values)]
    if values.empty:
        return {"count": 0}
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "max": float(values.max()),
        "nonzero_count": int((values.abs() > 1.0e-30).sum()),
        "nonzero_fraction": float((values.abs() > 1.0e-30).mean()),
    }


def _add_target_quality_summary(
    *,
    target_summary: dict[str, dict[str, Any]],
    target_output_specs: list[tuple[str, dict[str, str]]],
    frame: pd.DataFrame,
) -> None:
    if frame.empty:
        return
    for output, spec in target_output_specs:
        if not (output == "y__pH" or str(spec.get("name", "")).lower() == "ph"):
            continue
        summary = target_summary.setdefault(output, {})
        reliable = None
        if "pH_water_reliable" in frame.columns:
            reliable = frame["pH_water_reliable"].map(_as_bool_value)
        elif "solver_rescued" in frame.columns:
            reliable = ~frame["solver_rescued"].map(_as_bool_value)
        if reliable is None:
            continue
        count = int(len(reliable))
        unreliable_count = int((~reliable).sum())
        summary["pH_water_reliable_count"] = int(reliable.sum())
        summary["pH_water_unreliable_count"] = unreliable_count
        summary["pH_water_unreliable_fraction"] = None if count <= 0 else float(unreliable_count / count)
        if "xgems_water_delta_g" in frame.columns:
            delta = pd.to_numeric(frame["xgems_water_delta_g"], errors="coerce").abs()
            finite_delta = delta[delta.notna() & np.isfinite(delta)]
            summary["pH_xgems_water_changed_count"] = int((finite_delta > 1.0e-9).sum())
            summary["pH_xgems_water_delta_g_max_abs"] = None if finite_delta.empty else float(finite_delta.max())


def build_model_table(
    *,
    feature_table: str | Path,
    config: str | Path,
    out: str | Path,
    output_format: str | None = None,
    reaction_model_id: str | None = None,
    reaction_model_signature: str | None = None,
) -> Path:
    config_data = load_yaml(config)
    frame = _read_table(feature_table)
    original_rows = len(frame)
    warnings: list[str] = []

    filters = config_data.get("filters", {}) or {}
    chemistry_status = filters.get("chemistry_status")
    if chemistry_status is not None:
        if "chemistry_status" not in frame.columns:
            raise ValueError("Filter requested chemistry_status but feature table has no chemistry_status column.")
        frame = frame[frame["chemistry_status"].astype(str) == str(chemistry_status)].copy()
    rows_after_status_filter = len(frame)
    reaction_filter = {
        "reaction_model_id": reaction_model_id if reaction_model_id is not None else filters.get("reaction_model_id"),
        "reaction_model_signature": reaction_model_signature
        if reaction_model_signature is not None
        else filters.get("reaction_model_signature"),
    }
    rows_before_reaction_filter = len(frame)
    for column, value in reaction_filter.items():
        if value is None:
            continue
        if column not in frame.columns:
            raise ValueError(f"Reaction-model filter requested missing feature column '{column}'.")
        include = _normalize_include_values(value)
        frame = frame[frame[column].astype(str).isin(include)].copy()
    rows_after_reaction_filter = len(frame)
    frame, view_rejected, resolved_view_filters = _apply_view_filters(frame, filters)
    rows_after_view_filters = len(frame)

    output_config = config_data.get("output", {}) or {}
    sanitize = bool(output_config.get("sanitize_column_names", True))
    metadata_prefix = str(output_config.get("metadata_prefix", "meta__"))
    input_prefix = str(output_config.get("input_prefix", "x__"))
    target_prefix = str(output_config.get("target_prefix", "y__"))

    metadata_columns = ensure_reaction_metadata_columns(list((config_data.get("metadata", {}) or {}).get("include", []) or []))
    input_columns = list((config_data.get("inputs", {}) or {}).get("include", []) or [])
    target_specs = _target_specs(config_data)
    target_sources = [spec["source"] for spec in target_specs]
    for column in metadata_columns:
        if column not in frame.columns:
            frame[column] = ""
    _ensure_columns(frame, input_columns, "input")
    _ensure_columns(frame, target_sources, "target")

    model = pd.DataFrame(index=frame.index)
    schema: dict[str, Any] = {
        "feature_table": str(feature_table),
        "config": str(config),
        "output_table": str(out),
        "roles": {"metadata": [], "inputs": [], "targets": []},
        "filters": {
            "input_rows": int(original_rows),
            "rows_after_status_filter": int(rows_after_status_filter),
            "rows_before_reaction_model_filter": int(rows_before_reaction_filter),
            "rows_after_reaction_model_filter": int(rows_after_reaction_filter),
            "rows_after_view_filters": int(rows_after_view_filters),
            "rows_dropped_missing_inputs": 0,
            "rows_dropped_missing_targets": 0,
            "rejected_by_view_filters": view_rejected,
            "resolved_view_filters": resolved_view_filters,
            "reaction_model_filter": {key: value for key, value in reaction_filter.items() if value is not None},
        },
        "reaction_provenance_input": reaction_provenance_from_frame(frame),
    }

    for source in metadata_columns:
        output = f"{metadata_prefix}{_safe_name(source, sanitize=sanitize)}"
        model[output] = frame[source]
        schema["roles"]["metadata"].append({"column": output, "source": source})

    input_output_columns: list[str] = []
    for source in input_columns:
        output = f"{input_prefix}{_safe_name(source, sanitize=sanitize)}"
        model[output] = pd.to_numeric(frame[source], errors="coerce")
        input_output_columns.append(output)
        schema["roles"]["inputs"].append({"column": output, "source": source, "derived": False})

    derived = (config_data.get("inputs", {}) or {}).get("derived", {}) or {}
    for name, spec in derived.items():
        output = f"{input_prefix}{_safe_name(str(name), sanitize=sanitize)}"
        model[output] = _derived_input(frame, str(name), spec or {})
        input_output_columns.append(output)
        schema["roles"]["inputs"].append({"column": output, "source": spec.get("source"), "derived": True, "transform": spec.get("transform")})

    target_output_columns: list[str] = []
    target_output_specs: list[tuple[str, dict[str, str]]] = []
    for spec in target_specs:
        if spec["kind"] == "scalar":
            output_name = spec["name"]
        elif spec["kind"] == "phase_amount_groups":
            output_name = f"amount_{spec['name']}"
        elif spec["kind"] == "phase_volume_groups":
            output_name = f"volume_{spec['name']}"
        else:
            output_name = spec["name"]
        output = f"{target_prefix}{_safe_name(output_name, sanitize=sanitize)}"
        model[output] = pd.to_numeric(frame[spec["source"]], errors="coerce")
        target_output_columns.append(output)
        target_output_specs.append((output, spec))
        schema["roles"]["targets"].append({"column": output, "source": spec["source"], "name": spec["name"], "kind": spec["kind"]})

    if filters.get("drop_missing_inputs", True):
        model, dropped = _drop_missing(model, input_output_columns)
        schema["filters"]["rows_dropped_missing_inputs"] = int(dropped)
    if filters.get("drop_missing_targets", True):
        model, dropped = _drop_missing(model, target_output_columns)
        schema["filters"]["rows_dropped_missing_targets"] = int(dropped)

    schema["row_count"] = int(len(model))
    schema["column_count"] = int(len(model.columns))
    schema["reaction_provenance"] = reaction_provenance_from_frame(model, prefix=metadata_prefix)
    final_feature_frame = frame.loc[model.index] if len(model) else frame.iloc[0:0]
    schema["target_summary"] = {column: _series_summary(model[column]) for column in target_output_columns}
    _add_target_quality_summary(
        target_summary=schema["target_summary"],
        target_output_specs=target_output_specs,
        frame=final_feature_frame,
    )
    schema["input_summary"] = {column: _series_summary(model[column]) for column in input_output_columns}
    if len(model) == 0:
        warnings.append("Model table has zero rows after filters.")
    for column in target_output_columns:
        summary = schema["target_summary"][column]
        if summary.get("nonzero_count") == 0:
            warnings.append(f"Target column '{column}' is all zero.")
        if summary.get("count") and summary.get("nonzero_fraction", 1.0) < 0.01:
            warnings.append(f"Target column '{column}' is sparse: nonzero_fraction={summary['nonzero_fraction']:.4g}.")

    out_path = Path(out)
    written_path, write_warnings = _write_table(model.reset_index(drop=True), out_path, output_format)
    warnings.extend(write_warnings)
    schema["output_table"] = str(written_path)
    schema["warnings"] = warnings
    write_json(written_path.with_suffix(written_path.suffix + ".schema.json"), schema)
    write_json(written_path.with_suffix(written_path.suffix + ".warnings.json"), {"warnings": warnings})
    return written_path
