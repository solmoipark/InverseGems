from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .global_chemistry_db import load_global_manifest
from .utils import config_path, load_yaml, write_json


CHEM_INPUT_PREFIX = "x__chem_oxide_equiv_mol_"
TRUE_VALUES = {"true", "1", "1.0", "yes", "y"}
COMPLETE_STATUS_VALUES = {"complete", "success", "ok", "cached"}


def _read_csv_if_exists(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    value = Path(path)
    if not value.exists():
        return pd.DataFrame()
    return pd.read_csv(value)


def _manifest_path(manifest: dict[str, Any], key: str) -> Path | None:
    value = (manifest.get("paths") or {}).get(key)
    return None if not value else Path(str(value))


def _load_coverage_config(path: str | Path | None) -> dict[str, Any]:
    return load_yaml(path or config_path("global_chemistry_coverage.yaml"))


def _material_systems_from_config(config: dict[str, Any], material_systems_config: str | Path | None) -> list[str]:
    explicit = [str(item) for item in (config.get("material_systems") or [])]
    if explicit:
        return explicit
    systems = load_yaml(material_systems_config or config_path("material_systems.yaml"))
    return sorted(systems)


def _column(frame: pd.DataFrame, *names: str) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def _status(ok: bool, warning: bool = False) -> str:
    if ok:
        return "ok"
    if warning:
        return "warning"
    return "missing"


def _bool_series(frame: pd.DataFrame, *names: str, default: bool = False) -> pd.Series:
    column = _column(frame, *names)
    if column is None or frame.empty:
        return pd.Series([default] * len(frame), index=frame.index, dtype=bool)
    return frame[column].fillna(default).astype(str).str.strip().str.lower().isin(TRUE_VALUES)


def _numeric_series(frame: pd.DataFrame, *names: str) -> pd.Series:
    column = _column(frame, *names)
    if column is None or frame.empty:
        return pd.Series([float("nan")] * len(frame), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _water_delta_series(frame: pd.DataFrame) -> pd.Series:
    water = _numeric_series(frame, "water_g", "meta__water_g", "x__water_g")
    xgems_water = _numeric_series(frame, "xgems_water_g", "meta__xgems_water_g", "x__xgems_water_g")
    return xgems_water - water


def _pH_water_reliable_series(frame: pd.DataFrame, *, water_tolerance_g: float = 1.0e-9) -> pd.Series:
    explicit = _column(frame, "pH_water_reliable", "meta__pH_water_reliable")
    if explicit is not None:
        return frame[explicit].fillna(True).astype(str).str.strip().str.lower().isin(TRUE_VALUES)
    solver_rescued = _bool_series(frame, "solver_rescued", "meta__solver_rescued")
    water_delta = _water_delta_series(frame)
    water_adjusted = water_delta.abs() > water_tolerance_g
    water_adjusted = water_adjusted.fillna(False)
    return ~(solver_rescued | water_adjusted)


def _material_system_coverage(frame: pd.DataFrame, config: dict[str, Any], material_systems_config: str | Path | None) -> pd.DataFrame:
    systems = _material_systems_from_config(config, material_systems_config)
    thresholds = config.get("thresholds") or {}
    min_rows = int(thresholds.get("min_rows_per_material_system", 20))
    min_hashes = int(thresholds.get("min_unique_chem_hashes_per_material_system", min_rows))
    system_col = _column(frame, "meta__material_system", "material_system")
    hash_col = _column(frame, "meta__chem_hash", "chem_hash")
    rows: list[dict[str, Any]] = []
    for system in systems:
        subset = frame[frame[system_col].astype(str) == system] if system_col and not frame.empty else pd.DataFrame()
        unique_hashes = int(subset[hash_col].dropna().astype(str).nunique()) if hash_col and not subset.empty else 0
        row_count = int(len(subset))
        rows.append(
            {
                "material_system": system,
                "row_count": row_count,
                "unique_chem_hash_count": unique_hashes,
                "min_rows_expected": min_rows,
                "min_unique_chem_hashes_expected": min_hashes,
                "coverage_status": _status(row_count >= min_rows and unique_hashes >= min_hashes, warning=row_count > 0),
            }
        )
    return pd.DataFrame(rows)


def _age_coverage(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    milestones = [float(value) for value in (config.get("age_milestones_days") or [])]
    age_col = _column(frame, "x__age_days", "meta__age_days", "age_days")
    if not age_col or frame.empty:
        return pd.DataFrame(
            {
                "age_days": milestones,
                "row_count_within_tolerance": [0] * len(milestones),
                "nearest_age_days": [None] * len(milestones),
                "coverage_status": ["missing"] * len(milestones),
            }
        )
    ages = pd.to_numeric(frame[age_col], errors="coerce").dropna()
    fraction = float(config.get("age_tolerance_fraction", 0.10))
    min_days = float(config.get("age_tolerance_min_days", 0.01))
    rows: list[dict[str, Any]] = []
    for age in milestones:
        tolerance = max(abs(age) * fraction, min_days)
        within = ages[(ages - age).abs() <= tolerance]
        nearest = None if ages.empty else float(ages.iloc[(ages - age).abs().argmin()])
        count = int(len(within))
        rows.append(
            {
                "age_days": age,
                "tolerance_days": tolerance,
                "row_count_within_tolerance": count,
                "nearest_age_days": nearest,
                "coverage_status": "ok" if count > 0 else "missing",
            }
        )
    return pd.DataFrame(rows)


def _range_rows(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        rows.append(
            {
                "column": column,
                "finite_count": int(len(values)),
                "min": None if values.empty else float(values.min()),
                "p05": None if values.empty else float(values.quantile(0.05)),
                "median": None if values.empty else float(values.median()),
                "p95": None if values.empty else float(values.quantile(0.95)),
                "max": None if values.empty else float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def _chemistry_input_ranges(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in frame.columns if column.startswith(CHEM_INPUT_PREFIX)]
    extra = [
        column
        for column in [
            "x__xgems_water_g",
            "meta__xgems_water_g",
            "x__temperature_celsius",
            "meta__temperature_celsius",
            "x__w_b",
            "meta__w_b",
        ]
        if column in frame.columns
    ]
    return _range_rows(frame, columns + extra)


def _xgems_quality_row(frame: pd.DataFrame, *, group: str, group_value: Any, config: dict[str, Any]) -> dict[str, Any]:
    thresholds = config.get("thresholds") or {}
    row_count = int(len(frame))
    status_col = _column(frame, "meta__chemistry_status", "chemistry_status")
    statuses = frame[status_col].fillna("<missing>").astype(str).str.strip().str.lower() if status_col and row_count else pd.Series(dtype=str)
    complete_count = int(statuses.isin(COMPLETE_STATUS_VALUES).sum()) if row_count else 0
    solver_rescued = _bool_series(frame, "solver_rescued", "meta__solver_rescued")
    retry_count = _numeric_series(frame, "xgems_retry_count", "meta__xgems_retry_count").fillna(0.0)
    water_delta = _water_delta_series(frame)
    water_adjusted = water_delta.abs() > float(thresholds.get("xgems_water_adjusted_tolerance_g", 1.0e-9))
    water_adjusted = water_adjusted.fillna(False)
    pH_reliable = _pH_water_reliable_series(
        frame,
        water_tolerance_g=float(thresholds.get("xgems_water_adjusted_tolerance_g", 1.0e-9)),
    )
    preflight_col = _column(frame, "preflight_dir", "meta__preflight_dir")
    preflight_available = (
        frame[preflight_col].fillna("").astype(str).str.strip().ne("")
        if preflight_col and row_count
        else pd.Series([False] * row_count, index=frame.index, dtype=bool)
    )
    complete_fraction = complete_count / row_count if row_count else 0.0
    solver_rescued_count = int(solver_rescued.sum()) if row_count else 0
    solver_rescued_fraction = solver_rescued_count / row_count if row_count else 0.0
    retry_count_nonzero = int((retry_count > 0).sum()) if row_count else 0
    water_adjusted_count = int(water_adjusted.sum()) if row_count else 0
    water_adjusted_fraction = water_adjusted_count / row_count if row_count else 0.0
    pH_unreliable_count = int((~pH_reliable).sum()) if row_count else 0
    pH_unreliable_fraction = pH_unreliable_count / row_count if row_count else 0.0
    warnings: list[str] = []
    if complete_fraction < float(thresholds.get("min_xgems_complete_fraction", 0.95)):
        warnings.append("low_complete_fraction")
    if solver_rescued_fraction > float(thresholds.get("max_solver_rescued_fraction", 0.10)):
        warnings.append("high_solver_rescued_fraction")
    if water_adjusted_fraction > float(thresholds.get("max_xgems_water_adjusted_fraction", 0.10)):
        warnings.append("high_xgems_water_adjusted_fraction")
    if pH_unreliable_fraction > float(thresholds.get("max_pH_unreliable_fraction", 0.05)):
        warnings.append("high_pH_water_unreliable_fraction")
    return {
        "group": group,
        "group_value": group_value,
        "row_count": row_count,
        "complete_count": complete_count,
        "complete_fraction": complete_fraction,
        "solver_rescued_count": solver_rescued_count,
        "solver_rescued_fraction": solver_rescued_fraction,
        "xgems_retry_nonzero_count": retry_count_nonzero,
        "xgems_water_adjusted_count": water_adjusted_count,
        "xgems_water_adjusted_fraction": water_adjusted_fraction,
        "pH_water_unreliable_count": pH_unreliable_count,
        "pH_water_unreliable_fraction": pH_unreliable_fraction,
        "preflight_available_count": int(preflight_available.sum()) if row_count else 0,
        "preflight_available_fraction": float(preflight_available.sum() / row_count) if row_count else 0.0,
        "max_abs_xgems_water_delta_g": None if water_delta.dropna().empty else float(water_delta.abs().max()),
        "quality_status": "ok" if not warnings else "warning",
        "quality_warnings": "; ".join(warnings),
    }


def _xgems_quality_by_material_system(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    system_col = _column(frame, "meta__material_system", "material_system")
    rows = [_xgems_quality_row(frame, group="all", group_value="all", config=config)]
    if system_col and not frame.empty:
        for system, subset in frame.groupby(system_col, dropna=False):
            rows.append(_xgems_quality_row(subset, group="material_system", group_value=system, config=config))
    return pd.DataFrame(rows)


def _target_metrics(path: Path | None, config: dict[str, Any]) -> pd.DataFrame:
    metrics = _read_csv_if_exists(path)
    if metrics.empty:
        return metrics
    thresholds = config.get("thresholds") or {}
    min_r2 = float(thresholds.get("min_surrogate_r2_recommended", 0.70))
    sparse_eval_columns = ["evaluation_warning", "test_nonzero_missing", "test_nonzero_too_low"]
    statuses: list[str] = []
    reasons: list[str] = []
    for _, row in metrics.iterrows():
        row_reasons: list[str] = []
        if "r2" not in metrics.columns:
            row_reasons.append("missing R2")
        else:
            r2 = pd.to_numeric(pd.Series([row.get("r2")]), errors="coerce").iloc[0]
            if pd.isna(r2) or float(r2) < min_r2:
                row_reasons.append(f"R2 below recommended {min_r2:g}")
        for column in sparse_eval_columns:
            if column in metrics.columns and str(row.get(column)).strip().lower() in {"true", "1", "1.0", "yes"}:
                row_reasons.append(column)
        diagnostic_warning = str(row.get("diagnostic_warning", "") or "")
        if diagnostic_warning and diagnostic_warning.lower() != "nan":
            row_reasons.append(diagnostic_warning)
        statuses.append("warning" if row_reasons else "ok")
        reasons.append("; ".join(dict.fromkeys(row_reasons)))
    metrics["coverage_status"] = statuses
    metrics["coverage_reasons"] = reasons
    metrics["min_r2_recommended"] = min_r2
    return metrics


def _xgems_health(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    thresholds = config.get("thresholds") or {}
    row_count = int(len(frame))
    health: dict[str, Any] = {"row_count": row_count}
    for column in ["meta__chemistry_status", "chemistry_status"]:
        if column in frame.columns:
            health["chemistry_status_counts"] = frame[column].fillna("<missing>").astype(str).value_counts().to_dict()
            break
    quality = _xgems_quality_row(frame, group="all", group_value="all", config=config)
    health.update(
        {
            "complete_count": quality["complete_count"],
            "complete_fraction": quality["complete_fraction"],
            "solver_rescued_count": quality["solver_rescued_count"],
            "solver_rescued_fraction": quality["solver_rescued_fraction"],
            "xgems_water_adjusted_count": quality["xgems_water_adjusted_count"],
            "xgems_water_adjusted_fraction": quality["xgems_water_adjusted_fraction"],
            "pH_water_unreliable_count": quality["pH_water_unreliable_count"],
            "pH_water_unreliable_fraction": quality["pH_water_unreliable_fraction"],
            "pH_water_reliability_status": "ok"
            if float(quality["pH_water_unreliable_fraction"]) <= float(thresholds.get("max_pH_unreliable_fraction", 0.05))
            else "warning",
            "quality_status": quality["quality_status"],
            "quality_warnings": quality["quality_warnings"],
        }
    )
    for logical_name, candidates in {
        "solver_rescued": ["solver_rescued", "meta__solver_rescued"],
        "xgems_water_matches_recipe": ["xgems_water_matches_recipe", "meta__xgems_water_matches_recipe"],
    }.items():
        column = _column(frame, *candidates)
        if column:
            health[f"{logical_name}_counts"] = frame[column].fillna("<missing>").astype(str).value_counts().to_dict()
    return health


def _overall_summary(
    *,
    manifest: dict[str, Any],
    model_table: pd.DataFrame,
    material_coverage: pd.DataFrame,
    age_coverage: pd.DataFrame,
    target_metrics: pd.DataFrame,
    health: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    thresholds = config.get("thresholds") or {}
    hash_col = _column(model_table, "meta__chem_hash", "chem_hash")
    reaction_id_col = _column(model_table, "meta__reaction_model_id", "reaction_model_id")
    reaction_sig_col = _column(model_table, "meta__reaction_model_signature", "reaction_model_signature")
    w_b_col = _column(model_table, "x__w_b", "meta__w_b", "w_b")
    row_count = int(len(model_table))
    unique_hashes = int(model_table[hash_col].dropna().astype(str).nunique()) if hash_col and not model_table.empty else 0
    warnings: list[str] = []
    if row_count < int(thresholds.get("min_total_rows", 200)):
        warnings.append(f"model table has {row_count} rows; target minimum is {thresholds.get('min_total_rows', 200)}")
    missing_systems = material_coverage[material_coverage["coverage_status"] == "missing"]["material_system"].tolist() if not material_coverage.empty else []
    if missing_systems:
        warnings.append("missing material systems: " + ", ".join(map(str, missing_systems)))
    missing_ages = age_coverage[age_coverage["coverage_status"] == "missing"]["age_days"].tolist() if not age_coverage.empty else []
    if missing_ages:
        warnings.append("missing age milestones: " + ", ".join(f"{float(age):g}" for age in missing_ages))
    if target_metrics.empty:
        warnings.append("target_metrics.csv is missing; train the surrogate before inverse routing confidence checks")
    else:
        low_targets = target_metrics[target_metrics.get("coverage_status", "") == "warning"].get("target", pd.Series(dtype=str)).tolist()
        if low_targets:
            warnings.append("targets with surrogate metric warnings: " + ", ".join(map(str, low_targets)))
    if health.get("pH_water_reliability_status") == "warning":
        warnings.append("pH includes too many water-adjusted/rescued rows for reliable target routing")
    quality_warnings = str(health.get("quality_warnings") or "")
    for warning in [item.strip() for item in quality_warnings.split(";") if item.strip()]:
        if warning == "high_solver_rescued_fraction":
            warnings.append("xGEMS/GEMS validation contains many solver-rescued rows")
        elif warning == "high_xgems_water_adjusted_fraction":
            warnings.append("xGEMS/GEMS validation contains many water-adjusted rows")
        elif warning == "low_complete_fraction":
            warnings.append("xGEMS/GEMS validation complete fraction is below the configured threshold")
    w_b_range = None
    if w_b_col:
        w_b_values = pd.to_numeric(model_table[w_b_col], errors="coerce").dropna()
        if not w_b_values.empty:
            w_b_range = {"column": w_b_col, "min": float(w_b_values.min()), "max": float(w_b_values.max())}
            expected = config.get("w_b_expected_range")
            if expected and len(expected) == 2:
                expected_min, expected_max = float(expected[0]), float(expected[1])
                if w_b_range["min"] < expected_min or w_b_range["max"] > expected_max:
                    warnings.append(
                        f"w/b range {w_b_range['min']:.4g}-{w_b_range['max']:.4g} is outside expected {expected_min:.4g}-{expected_max:.4g}"
                    )
    return {
        "db": manifest.get("db"),
        "model_table": (manifest.get("paths") or {}).get("model_table"),
        "model_bundle": (manifest.get("paths") or {}).get("model_bundle"),
        "row_count": row_count,
        "unique_chem_hash_count": unique_hashes,
        "duplicate_chem_hash_rows": max(0, row_count - unique_hashes),
        "reaction_model_ids": sorted(model_table[reaction_id_col].dropna().astype(str).unique().tolist()) if reaction_id_col else [],
        "reaction_model_signatures": sorted(model_table[reaction_sig_col].dropna().astype(str).unique().tolist()) if reaction_sig_col else [],
        "w_b_range": w_b_range,
        "material_system_count": int((material_coverage["row_count"] > 0).sum()) if not material_coverage.empty else 0,
        "missing_material_systems": missing_systems,
        "missing_age_milestones_days": [float(age) for age in missing_ages],
        "xgems_health": health,
        "warnings": warnings,
        "ready_for_inverse_design": not warnings,
    }


def _write_markdown(
    summary: dict[str, Any],
    material: pd.DataFrame,
    age: pd.DataFrame,
    target_metrics: pd.DataFrame,
    xgems_quality: pd.DataFrame,
    path: Path,
) -> None:
    lines = ["# Global Chemistry Coverage Report", ""]
    lines.append(f"- model table rows: `{summary.get('row_count')}`")
    lines.append(f"- unique chem hashes: `{summary.get('unique_chem_hash_count')}`")
    lines.append(f"- ready for inverse design: `{summary.get('ready_for_inverse_design')}`")
    if summary.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in summary["warnings"]:
            lines.append(f"- {warning}")
    for title, frame, columns in [
        ("Material Systems", material, ["material_system", "row_count", "unique_chem_hash_count", "coverage_status"]),
        ("Age Milestones", age, ["age_days", "row_count_within_tolerance", "nearest_age_days", "coverage_status"]),
        (
            "xGEMS Quality",
            xgems_quality,
            [
                "group",
                "group_value",
                "row_count",
                "complete_fraction",
                "solver_rescued_fraction",
                "xgems_water_adjusted_fraction",
                "pH_water_unreliable_fraction",
                "quality_status",
                "quality_warnings",
            ],
        ),
        ("Target Metrics", target_metrics, ["target", "r2", "mae", "nonzero_true_count", "full_nonzero_count", "evaluation_reliability", "coverage_status", "coverage_reasons"]),
    ]:
        lines.extend(["", f"## {title}", ""])
        if frame.empty:
            lines.append("No data.")
            continue
        use_columns = [column for column in columns if column in frame.columns]
        lines.append("| " + " | ".join(use_columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(use_columns)) + " |")
        for row in frame[use_columns].to_dict(orient="records"):
            lines.append("| " + " | ".join(str(row.get(column, "")).replace("|", "\\|") for column in use_columns) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_global_chemistry_coverage_report(
    *,
    db: str | Path,
    out: str | Path,
    schema_config: str | Path | None = None,
    coverage_config: str | Path | None = None,
    material_systems_config: str | Path | None = None,
    target_metrics: str | Path | None = None,
) -> Path:
    manifest = load_global_manifest(db, schema_config=schema_config)
    config = _load_coverage_config(coverage_config)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_table_path = _manifest_path(manifest, "model_table")
    model_table = _read_csv_if_exists(model_table_path)
    material = _material_system_coverage(model_table, config, material_systems_config)
    age = _age_coverage(model_table, config)
    chemistry_ranges = _chemistry_input_ranges(model_table)
    xgems_quality = _xgems_quality_by_material_system(model_table, config)
    metrics_path = Path(target_metrics) if target_metrics else None
    if metrics_path is None:
        bundle = _manifest_path(manifest, "model_bundle")
        metrics_path = bundle.parent / "target_metrics.csv" if bundle else None
    metrics = _target_metrics(metrics_path, config)
    health = _xgems_health(model_table, config)
    summary = _overall_summary(
        manifest=manifest,
        model_table=model_table,
        material_coverage=material,
        age_coverage=age,
        target_metrics=metrics,
        health=health,
        config=config,
    )
    summary["coverage_config"] = str(coverage_config or config_path("global_chemistry_coverage.yaml"))
    summary["schema_config"] = str(schema_config or config_path("global_chemistry_db.yaml"))
    summary["target_metrics"] = None if metrics_path is None else str(metrics_path)
    summary["outputs"] = {
        "summary_json": str(out_dir / "global_chemistry_coverage_summary.json"),
        "material_system_coverage": str(out_dir / "material_system_coverage.csv"),
        "age_coverage": str(out_dir / "age_coverage.csv"),
        "chemistry_input_ranges": str(out_dir / "chemistry_input_ranges.csv"),
        "xgems_quality_by_material_system": str(out_dir / "xgems_quality_by_material_system.csv"),
        "target_metrics_coverage": str(out_dir / "target_metrics_coverage.csv"),
        "markdown": str(out_dir / "global_chemistry_coverage.md"),
    }
    material.to_csv(out_dir / "material_system_coverage.csv", index=False)
    age.to_csv(out_dir / "age_coverage.csv", index=False)
    chemistry_ranges.to_csv(out_dir / "chemistry_input_ranges.csv", index=False)
    xgems_quality.to_csv(out_dir / "xgems_quality_by_material_system.csv", index=False)
    metrics.to_csv(out_dir / "target_metrics_coverage.csv", index=False)
    write_json(out_dir / "global_chemistry_coverage_summary.json", summary)
    _write_markdown(summary, material, age, metrics, xgems_quality, out_dir / "global_chemistry_coverage.md")
    (out_dir / "global_chemistry_manifest_snapshot.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return out_dir
