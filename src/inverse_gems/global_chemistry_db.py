from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .age_grids import add_age_metadata
from .chem_hash import round_sig
from .chemistry_candidate_table import build_chemistry_candidate_table
from .chemistry_design_query_runner import run_chemistry_design_query
from .chemistry_domain import write_chemistry_domain_report
from .database import InverseGemsDatabase, utc_now_iso
from .feature_table import build_feature_table
from .forward_query import run_forward_query
from .model_table import build_model_table
from .sampling import COMPONENTS, write_recipe_csv
from .surrogate import train_baseline_surrogate
from .utils import config_path, load_yaml, short_hash, write_json


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_csv(path)
    return pd.read_csv(path)


def _near_exact_signatures(frame: pd.DataFrame, columns: list[str], *, digits: int) -> pd.Series:
    if not columns:
        return pd.Series([""] * len(frame), index=frame.index)

    def value_signature(value: Any) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        if pd.isna(numeric):
            return ""
        return f"{round_sig(numeric, digits):.12g}"

    signature = frame[columns[0]].map(value_signature).astype(str)
    for column in columns[1:]:
        signature = signature + "|" + frame[column].map(value_signature).astype(str)
    return signature


def _write_table(frame: pd.DataFrame, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return out


def _read_json_if_exists(path: str | Path) -> dict[str, Any]:
    value = Path(path)
    if not value.exists():
        return {}
    return json.loads(value.read_text(encoding="utf-8"))


def _load_schema(schema_config: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(schema_config or config_path("global_chemistry_db.yaml"))


def _artifact_dir(db: str | Path, schema: dict[str, Any]) -> Path:
    return Path(db) / str((schema.get("paths") or {}).get("artifact_dir", "global_chemistry"))


def _manifest_path(db: str | Path, schema: dict[str, Any]) -> Path:
    return _artifact_dir(db, schema) / "global_chemistry_manifest.json"


def _path_from_manifest(manifest: dict[str, Any], key: str) -> Path | None:
    value = (manifest.get("paths") or {}).get(key)
    return None if not value else Path(str(value))


def load_global_manifest(db: str | Path, schema_config: str | Path | None = None) -> dict[str, Any]:
    schema = _load_schema(schema_config)
    path = _manifest_path(db, schema)
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        return _rebase_manifest_if_moved(db, manifest, schema)
    return initialize_global_chemistry_db(db=db, schema_config=schema_config)


def _rebase_manifest_if_moved(db: str | Path, manifest: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Re-anchor manifest paths when the DB directory was copied or moved.

    The manifest stores concrete artifact paths. A copied DB would otherwise
    keep pointing at the original directory, silently writing refresh/train
    outputs into the source DB. On mismatch we recompute every path against
    the requested db root (same layout initialize uses) and persist the fix.
    """
    recorded = str(manifest.get("db") or "")
    db_path = Path(db)
    try:
        if recorded and Path(recorded).resolve() == db_path.resolve():
            return manifest
    except OSError:
        pass
    database = InverseGemsDatabase(db)
    artifact_dir = _artifact_dir(db, schema)
    paths_cfg = schema.get("paths") or {}
    surrogate_dir = artifact_dir / str(paths_cfg.get("surrogate_dir", "global_surrogate"))
    manifest["db"] = str(db_path)
    manifest["sqlite_path"] = str(database.sqlite_path)
    manifest["paths"] = {
        "artifact_dir": str(artifact_dir),
        "schema": str(artifact_dir / "global_chemistry_schema.json"),
        "feature_table": str(artifact_dir / str(paths_cfg.get("feature_table", "global_feature_table.csv"))),
        "model_table": str(artifact_dir / str(paths_cfg.get("model_table", "global_model_table.csv"))),
        "surrogate_dir": str(surrogate_dir),
        "model_bundle": str(surrogate_dir / "model.joblib"),
        "lookup_dir": str(artifact_dir / str(paths_cfg.get("lookup_dir", "lookups"))),
        "acquisition_dir": str(artifact_dir / str(paths_cfg.get("acquisition_dir", "acquisitions"))),
    }
    manifest["rebased_from"] = recorded or None
    manifest["updated_at"] = utc_now_iso()
    write_json(_manifest_path(db, schema), manifest)
    return manifest


def initialize_global_chemistry_db(*, db: str | Path, schema_config: str | Path | None = None) -> dict[str, Any]:
    schema = _load_schema(schema_config)
    database = InverseGemsDatabase(db)
    artifact_dir = _artifact_dir(db, schema)
    paths_cfg = schema.get("paths") or {}
    artifact_dir.mkdir(parents=True, exist_ok=True)
    schema_path = artifact_dir / "global_chemistry_schema.json"
    write_json(schema_path, schema)
    manifest = {
        "schema_version": int(schema.get("schema_version", 1)),
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "db": str(Path(db)),
        "sqlite_path": str(database.sqlite_path),
        "reactive_chemistry_centered": True,
        "recipe_age_materials_are_provenance": True,
        "paths": {
            "artifact_dir": str(artifact_dir),
            "schema": str(schema_path),
            "feature_table": str(artifact_dir / str(paths_cfg.get("feature_table", "global_feature_table.csv"))),
            "model_table": str(artifact_dir / str(paths_cfg.get("model_table", "global_model_table.csv"))),
            "surrogate_dir": str(artifact_dir / str(paths_cfg.get("surrogate_dir", "global_surrogate"))),
            "model_bundle": str(artifact_dir / str(paths_cfg.get("surrogate_dir", "global_surrogate")) / "model.joblib"),
            "lookup_dir": str(artifact_dir / str(paths_cfg.get("lookup_dir", "lookups"))),
            "acquisition_dir": str(artifact_dir / str(paths_cfg.get("acquisition_dir", "acquisitions"))),
        },
        "chemistry_inputs": list(schema.get("chemistry_inputs") or []),
        "defaults": schema.get("defaults") or {},
        "row_counts": {},
    }
    write_json(_manifest_path(db, schema), manifest)
    return manifest


def _update_manifest(db: str | Path, updates: dict[str, Any], schema_config: str | Path | None = None) -> dict[str, Any]:
    schema = _load_schema(schema_config)
    manifest = load_global_manifest(db, schema_config=schema_config)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(manifest.get(key), dict):
            manifest[key].update(value)
        else:
            manifest[key] = value
    manifest["updated_at"] = utc_now_iso()
    write_json(_manifest_path(db, schema), manifest)
    return manifest


def refresh_global_chemistry_db(
    *,
    db: str | Path,
    selection: str | Path | None = None,
    model_config: str | Path | None = None,
    schema_config: str | Path | None = None,
) -> dict[str, Any]:
    schema = _load_schema(schema_config)
    defaults = schema.get("defaults") or {}
    manifest = load_global_manifest(db, schema_config=schema_config)
    feature_table = _path_from_manifest(manifest, "feature_table")
    model_table = _path_from_manifest(manifest, "model_table")
    if feature_table is None or model_table is None:
        raise ValueError("Global DB manifest is missing feature/model table paths.")
    feature_path = build_feature_table(
        db=db,
        selection=selection or defaults.get("selection_config") or config_path("output_selection.yaml"),
        out=feature_table,
    )
    model_path = build_model_table(
        feature_table=feature_path,
        config=model_config or defaults.get("model_dataset_config") or config_path("model_dataset_chemistry_stable_targets.yaml"),
        out=model_table,
    )
    feature = _read_table(feature_path)
    model = _read_table(model_path)
    model_schema_path = Path(model_path).with_suffix(Path(model_path).suffix + ".schema.json")
    model_schema = _read_json_if_exists(model_schema_path)
    return _update_manifest(
        db,
        {
            "paths": {"feature_table": str(feature_path), "model_table": str(model_path)},
            "row_counts": {"feature_table": int(len(feature)), "model_table": int(len(model))},
            "last_refresh": {"selection": str(selection or defaults.get("selection_config")), "model_config": str(model_config or defaults.get("model_dataset_config"))},
            "reaction_provenance": model_schema.get("reaction_provenance") or {},
            "reaction_provenance_input": model_schema.get("reaction_provenance_input") or {},
            "model_table_schema": str(model_schema_path),
        },
        schema_config=schema_config,
    )


def train_global_chemistry_surrogate(
    *,
    db: str | Path,
    selection: str | Path | None = None,
    model_config: str | Path | None = None,
    surrogate_config: str | Path | None = None,
    schema_config: str | Path | None = None,
    refresh: bool = True,
    no_save_model: bool = False,
) -> dict[str, Any]:
    schema = _load_schema(schema_config)
    defaults = schema.get("defaults") or {}
    manifest = refresh_global_chemistry_db(
        db=db,
        selection=selection,
        model_config=model_config,
        schema_config=schema_config,
    ) if refresh else load_global_manifest(db, schema_config=schema_config)
    model_table = _path_from_manifest(manifest, "model_table")
    surrogate_dir = _path_from_manifest(manifest, "surrogate_dir")
    if model_table is None or surrogate_dir is None:
        raise ValueError("Global DB manifest is missing model_table or surrogate_dir.")
    frame = _read_table(model_table)
    if frame.empty:
        raise ValueError(f"Global model table has zero rows: {model_table}")
    surrogate_path = train_baseline_surrogate(
        model_table=model_table,
        config=surrogate_config or defaults.get("surrogate_config") or config_path("surrogate_baseline.yaml"),
        out=surrogate_dir,
        save_model=not no_save_model,
    )
    bundle = surrogate_path / "model.joblib"
    return _update_manifest(
        db,
        {
            "paths": {"surrogate_dir": str(surrogate_path), "model_bundle": str(bundle)},
            "row_counts": {"surrogate_model_table": int(len(frame))},
            "last_surrogate_training": {
                "surrogate_config": str(surrogate_config or defaults.get("surrogate_config")),
                "model_saved": not no_save_model,
            },
        },
        schema_config=schema_config,
    )


def _candidate_table_from_inputs(
    *,
    out_dir: Path,
    recipes_csv: str | Path | None,
    candidate_table: str | Path | None,
    dat_lst: str | Path | None,
    run_mode: str,
    temperature_celsius: float,
    pressure: float | None,
    xgems_water_mode: str,
    xgems_water_factor: float,
    xgems_water_g: float | None,
    xgems_water_w_b: float | None,
    reaction_model_id: str | None,
    reaction_model_config: str | Path | None,
) -> Path:
    if candidate_table is not None:
        return Path(candidate_table)
    if recipes_csv is None:
        raise ValueError("Provide either recipes_csv or candidate_table.")
    return build_chemistry_candidate_table(
        recipes_csv=recipes_csv,
        out=out_dir / "chemistry_candidate_table.csv",
        dat_lst=dat_lst,
        run_mode=run_mode,
        temperature_celsius=temperature_celsius,
        pressure=pressure,
        xgems_water_mode=xgems_water_mode,
        xgems_water_factor=xgems_water_factor,
        xgems_water_g=xgems_water_g,
        xgems_water_w_b=xgems_water_w_b,
        reaction_model_id=reaction_model_id,
        reaction_model_config=reaction_model_config,
    )


def lookup_global_chemistry(
    *,
    db: str | Path,
    out: str | Path,
    recipes_csv: str | Path | None = None,
    candidate_table: str | Path | None = None,
    reference_model_table: str | Path | None = None,
    model_bundle: str | Path | None = None,
    dat_lst: str | Path | None = None,
    run_mode: str = "reacted_only",
    temperature_celsius: float = 20.0,
    pressure: float | None = None,
    xgems_water_mode: str = "initial",
    xgems_water_factor: float = 1.0,
    xgems_water_g: float | None = None,
    xgems_water_w_b: float | None = None,
    reaction_model_id: str | None = None,
    reaction_model_config: str | Path | None = None,
    nearest_distance_warn: float | None = None,
    schema_config: str | Path | None = None,
) -> Path:
    schema = _load_schema(schema_config)
    defaults = schema.get("defaults") or {}
    manifest = load_global_manifest(db, schema_config=schema_config)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = _candidate_table_from_inputs(
        out_dir=out_dir,
        recipes_csv=recipes_csv,
        candidate_table=candidate_table,
        dat_lst=dat_lst,
        run_mode=run_mode,
        temperature_celsius=temperature_celsius,
        pressure=pressure,
        xgems_water_mode=xgems_water_mode,
        xgems_water_factor=xgems_water_factor,
        xgems_water_g=xgems_water_g,
        xgems_water_w_b=xgems_water_w_b,
        reaction_model_id=reaction_model_id,
        reaction_model_config=reaction_model_config,
    )
    reference_path = Path(reference_model_table) if reference_model_table else _path_from_manifest(manifest, "model_table")
    bundle_path = Path(model_bundle) if model_bundle else _path_from_manifest(manifest, "model_bundle")
    candidates = _read_table(candidate_path)
    report = candidates.copy()
    exact_column = str((schema.get("identity") or {}).get("exact_hash_column", "meta__chem_hash"))
    if reference_path and reference_path.exists():
        reference = _read_table(reference_path)
        reference_hashes = reference.get(exact_column, pd.Series(dtype=str)).dropna().astype(str)
        hit_counts = reference_hashes.value_counts()
        report["exact_chem_hash_hit"] = report[exact_column].astype(str).isin(set(reference_hashes))
        report["exact_chem_hash_hit_count"] = report[exact_column].astype(str).map(hit_counts).fillna(0).astype(int)
        near_digits = int((schema.get("acquisition") or {}).get("near_exact_sig_digits", 10))
        chemistry_inputs = [
            str(column)
            for column in (schema.get("chemistry_inputs") or [])
            if str(column) in report.columns and str(column) in reference.columns
        ]
        if chemistry_inputs:
            reference_signatures = _near_exact_signatures(reference, chemistry_inputs, digits=near_digits)
            signature_counts = reference_signatures.value_counts()
            report_signatures = _near_exact_signatures(report, chemistry_inputs, digits=near_digits)
            report["near_exact_chemistry_hit"] = report_signatures.isin(set(reference_signatures))
            report["near_exact_chemistry_hit_count"] = report_signatures.map(signature_counts).fillna(0).astype(int)
        else:
            report["near_exact_chemistry_hit"] = False
            report["near_exact_chemistry_hit_count"] = 0
        report["near_exact_sig_digits"] = near_digits
        domain_dir = write_chemistry_domain_report(
            reference_model_table=reference_path,
            candidate_table=candidate_path,
            model_bundle=bundle_path if bundle_path and bundle_path.exists() else None,
            out=out_dir / "domain_report",
            nearest_distance_warn=float(nearest_distance_warn if nearest_distance_warn is not None else defaults.get("nearest_distance_warn", 0.25)),
        )
        domain = pd.read_csv(Path(domain_dir) / "chemistry_domain_report.csv")
        keys = ["meta__recipe_id", "meta__chem_hash"]
        domain = domain.rename(columns={"recipe_id": "meta__recipe_id", "chem_hash": "meta__chem_hash"})
        merge_keys = [key for key in keys if key in report.columns and key in domain.columns]
        if merge_keys:
            keep = [column for column in domain.columns if column in merge_keys or column not in report.columns]
            report = report.merge(domain[keep], on=merge_keys, how="left")
    else:
        report["exact_chem_hash_hit"] = False
        report["exact_chem_hash_hit_count"] = 0
        report["near_exact_chemistry_hit"] = False
        report["near_exact_chemistry_hit_count"] = 0
    report_path = _write_table(report, out_dir / "global_chemistry_lookup.csv")
    warnings: list[str] = []
    if recipes_csv is not None and dat_lst is None:
        warnings.append(
            "dat_lst was not provided for recipe projection; exact chem_hash hits may not match DB rows "
            "whose hashes include a real dat.lst file."
        )
    summary = {
        "db": str(db),
        "candidate_table": str(candidate_path),
        "reference_model_table": None if reference_path is None else str(reference_path),
        "model_bundle": None if bundle_path is None else str(bundle_path),
        "candidate_rows": int(len(report)),
        "exact_hit_count": int(report["exact_chem_hash_hit"].sum()) if "exact_chem_hash_hit" in report else 0,
        "near_exact_chemistry_hit_count": (
            int(report["near_exact_chemistry_hit"].sum()) if "near_exact_chemistry_hit" in report else 0
        ),
        "out_of_domain_count": int(report.get("out_of_domain", pd.Series(dtype=bool)).fillna(False).sum()),
        "dat_lst": None if dat_lst is None else str(dat_lst),
        "pressure": pressure,
        "xgems_water": {
            "mode": xgems_water_mode,
            "factor": xgems_water_factor,
            "water_g": xgems_water_g,
            "water_w_b": xgems_water_w_b,
        },
        "warnings": warnings,
        "outputs": {"lookup_report": str(report_path)},
    }
    write_json(out_dir / "global_chemistry_lookup_summary.json", summary)
    return out_dir


def _recipe_rows_from_candidate_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in frame.reset_index(drop=True).iterrows():
        recipe_id = str(row.get("meta__recipe_id") or row.get("recipe_id") or f"acquired_{index + 1:06d}")
        out: dict[str, Any] = {
            "recipe_id": recipe_id,
            "template_name": row.get("meta__template_name", ""),
            "material_system": row.get("meta__material_system", ""),
            "target_profile": row.get("meta__target_profile", row.get("target_profile", "")),
            "target_profile_description": row.get("meta__target_profile_description", row.get("target_profile_description", "")),
            "w_b": float(row.get("x__w_b") or 0.0),
            "water_g": float(row.get("x__water_g") or 0.0),
            "water_mode": "wb_total",
            "temperature_celsius": float(row.get("x__temperature_celsius") or 20.0),
        }
        for component in COMPONENTS:
            out[component] = float(row.get(f"x__{component}") or 0.0)
        out = add_age_metadata(out, float(row.get("x__age_days") or 28.0))
        rows.append(out)
    return rows


def _safe_column_suffix(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")


def _target_label(target: str) -> str:
    label = str(target)
    for prefix in ("y__amount_", "y__volume_", "y__scalar_", "y__"):
        if label.startswith(prefix):
            label = label.removeprefix(prefix)
            break
    return label.replace("_", "-") if label == "C_A_S_H" else label


def _normalize_target_name(value: str) -> str:
    text = str(value).lower()
    return "".join(ch for ch in text if ch.isalnum())


def _priority_targets_from_diagnostics(
    diagnostics: str | Path | None,
    *,
    statuses: list[str] | tuple[str, ...] = ("usable_with_caution", "not_recommended"),
    target_kinds: list[str] | tuple[str, ...] | None = None,
    reliabilities: list[str] | tuple[str, ...] | None = None,
    limit: int | None = None,
) -> list[str]:
    if diagnostics is None:
        return []
    path = Path(diagnostics)
    if path.is_dir():
        path = path / "model_registry_diagnostics.csv"
    if not path.exists():
        raise ValueError(f"Diagnostics file does not exist: {path}")
    frame = pd.read_csv(path)
    if frame.empty or "target_column" not in frame.columns:
        return []
    wanted = {str(status) for status in statuses}
    if "status" in frame.columns:
        frame = frame[frame["status"].astype(str).isin(wanted)]
    wanted_reliabilities = {str(item) for item in (reliabilities or []) if str(item).strip()}
    if wanted_reliabilities and "evaluation_reliability" in frame.columns:
        frame = frame[frame["evaluation_reliability"].fillna("").astype(str).isin(wanted_reliabilities)]
    kinds = {str(kind).strip().lower() for kind in (target_kinds or []) if str(kind).strip()}
    if kinds:
        if "phase" in kinds:
            kinds.update({"amount", "volume"})
        columns_for_filter = frame["target_column"].fillna("").astype(str)

        def allowed_by_kind(value: str) -> bool:
            is_amount = value.startswith("y__amount_")
            is_volume = value.startswith("y__volume_")
            return (
                ("amount" in kinds and is_amount)
                or ("volume" in kinds and is_volume)
                or ("scalar" in kinds and not is_amount and not is_volume)
            )

        frame = frame[columns_for_filter.apply(allowed_by_kind)]
    if "active_learning_priority_score" in frame.columns:
        frame = frame.sort_values("active_learning_priority_score", ascending=False)
    elif "r2" in frame.columns:
        frame = frame.sort_values("r2", ascending=True, na_position="last")
    columns = [str(value) for value in frame["target_column"].dropna().tolist()]
    unique = list(dict.fromkeys(columns))
    if limit is not None:
        unique = unique[: int(limit)]
    return unique


def _resolve_priority_targets(requested: list[str], available: list[str]) -> tuple[list[str], list[str]]:
    if not requested:
        return [], []
    by_exact = {target: target for target in available}
    by_norm: dict[str, list[str]] = {}
    for target in available:
        by_norm.setdefault(_normalize_target_name(target), []).append(target)
        by_norm.setdefault(_normalize_target_name(_target_label(target)), []).append(target)
    resolved: list[str] = []
    missing: list[str] = []
    for item in requested:
        if item in by_exact:
            resolved.append(by_exact[item])
            continue
        matches = by_norm.get(_normalize_target_name(item), [])
        if matches:
            resolved.extend(matches)
        else:
            missing.append(item)
    return list(dict.fromkeys(resolved)), list(dict.fromkeys(missing))


def _add_surrogate_priority_scores(
    report: pd.DataFrame,
    *,
    model_bundle: Path | None,
    priority_targets: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    priority_report: dict[str, Any] = {
        "requested_targets": priority_targets,
        "resolved_targets": [],
        "missing_targets": [],
        "input_columns_missing": [],
        "enabled": False,
    }
    if not priority_targets:
        report["priority_target_score"] = 0.0
        return report, priority_report
    if model_bundle is None or not model_bundle.exists():
        priority_report["missing_targets"] = priority_targets
        priority_report["reason"] = "model_bundle_missing"
        report["priority_target_score"] = 0.0
        return report, priority_report
    bundle = joblib.load(model_bundle)
    inputs = [str(item) for item in (bundle.get("inputs") or [])]
    targets = [str(item) for item in (bundle.get("targets") or [])]
    resolved, missing = _resolve_priority_targets(priority_targets, targets)
    priority_report["resolved_targets"] = resolved
    priority_report["missing_targets"] = missing
    missing_inputs = [column for column in inputs if column not in report.columns]
    priority_report["input_columns_missing"] = missing_inputs
    if not resolved or missing_inputs:
        report["priority_target_score"] = 0.0
        return report, priority_report

    pred = bundle["estimator"].predict(report[inputs])
    pred_array = np.asarray(pred)
    if pred_array.ndim == 1:
        pred_array = pred_array[:, None]
    target_index = {target: index for index, target in enumerate(targets)}
    total = pd.Series(0.0, index=report.index)
    for target in resolved:
        values = pd.Series(pred_array[:, target_index[target]], index=report.index).abs()
        raw_column = f"priority_pred_abs__{_safe_column_suffix(target)}"
        score_column = f"priority_score__{_safe_column_suffix(target)}"
        report[raw_column] = values
        lo = float(values.min()) if len(values) else 0.0
        hi = float(values.max()) if len(values) else 0.0
        if hi > lo:
            normalized = (values - lo) / (hi - lo)
        else:
            normalized = pd.Series(0.0, index=report.index)
        report[score_column] = normalized
        total += normalized
    report["priority_target_score"] = total / max(len(resolved), 1)
    priority_report["enabled"] = True
    return report, priority_report


def _target_region_paths(paths: list[str | Path] | tuple[str | Path, ...] | str | Path | None) -> list[Path]:
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        paths = [paths]
    out: list[Path] = []
    for path in paths:
        value = Path(path)
        if value.is_dir():
            value = value / "target_region_nonzero_rows.csv"
        out.append(value)
    return out


def _add_target_region_scores(
    report: pd.DataFrame,
    *,
    target_region_tables: list[str | Path] | tuple[str | Path, ...] | str | Path | None,
    schema: dict[str, Any],
    distance_scale: float = 0.25,
    max_reference_rows: int | None = 1000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = _target_region_paths(target_region_tables)
    target_region_report: dict[str, Any] = {
        "enabled": False,
        "tables": [str(path) for path in paths],
        "reference_rows": 0,
        "input_columns": [],
        "missing_tables": [],
        "missing_input_columns": [],
        "distance_scale": float(distance_scale),
    }
    if not paths:
        report["target_region_score"] = 0.0
        return report, target_region_report

    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            target_region_report["missing_tables"].append(str(path))
            continue
        frame = _read_table(path)
        if not frame.empty:
            frame = frame.copy()
            frame["target_region_source"] = str(path)
            frames.append(frame)
    if not frames:
        report["target_region_score"] = 0.0
        target_region_report["reason"] = "no target-region reference rows"
        return report, target_region_report

    reference = pd.concat(frames, ignore_index=True)
    if max_reference_rows is not None and len(reference) > int(max_reference_rows):
        reference = reference.sample(n=int(max_reference_rows), random_state=42).reset_index(drop=True)

    schema_inputs = [str(column) for column in (schema.get("chemistry_inputs") or [])]
    columns = [column for column in schema_inputs if column in report.columns and column in reference.columns]
    if not columns:
        columns = [
            column
            for column in report.columns
            if str(column).startswith("x__chem_") and column in reference.columns
        ]
    if not columns:
        report["target_region_score"] = 0.0
        target_region_report["missing_input_columns"] = schema_inputs
        target_region_report["reason"] = "no overlapping target-region input columns"
        return report, target_region_report

    ref_numeric = reference[columns].apply(pd.to_numeric, errors="coerce")
    cand_numeric = report[columns].apply(pd.to_numeric, errors="coerce")
    ref_min = ref_numeric.min(axis=0)
    ref_max = ref_numeric.max(axis=0)
    cand_min = cand_numeric.min(axis=0)
    cand_max = cand_numeric.max(axis=0)
    lo = pd.concat([ref_min, cand_min], axis=1).min(axis=1)
    hi = pd.concat([ref_max, cand_max], axis=1).max(axis=1)
    scale = (hi - lo).replace(0.0, 1.0)
    ref_scaled = ((ref_numeric - lo) / scale).fillna(0.0).to_numpy(dtype=float)
    cand_scaled = ((cand_numeric - lo) / scale).fillna(0.0).to_numpy(dtype=float)
    ref_records = reference.reset_index(drop=True)

    distances: list[float] = []
    nearest_indices: list[int] = []
    for row in cand_scaled:
        diff = ref_scaled - row
        squared = (diff * diff).sum(axis=1)
        nearest = int(squared.argmin())
        nearest_indices.append(nearest)
        distances.append(float(squared[nearest] ** 0.5))
    distance_series = pd.Series(distances, index=report.index)
    report["target_region_nearest_distance"] = distance_series
    report["target_region_score"] = distance_series.map(
        lambda value: float(math.exp(-float(value) / max(float(distance_scale), 1.0e-12)))
    )
    report["target_region_nearest_recipe_id"] = [
        ref_records.iloc[index].get("meta__recipe_id", ref_records.iloc[index].get("recipe_id", ""))
        for index in nearest_indices
    ]
    report["target_region_nearest_source"] = [
        ref_records.iloc[index].get("target_region_source", "")
        for index in nearest_indices
    ]
    target_region_report.update(
        {
            "enabled": True,
            "reference_rows": int(len(reference)),
            "input_columns": columns,
            "max_reference_rows": max_reference_rows,
            "nearest_distance_min": None if distance_series.empty else float(distance_series.min()),
            "nearest_distance_median": None if distance_series.empty else float(distance_series.median()),
            "nearest_distance_max": None if distance_series.empty else float(distance_series.max()),
        }
    )
    return report, target_region_report


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "1.0", "yes", "y"}


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(out):
        return default
    return out


def _acquisition_bucket(row: pd.Series) -> str:
    exact = _bool_value(row.get("exact_chem_hash_hit"))
    near_exact = _bool_value(row.get("near_exact_chemistry_hit"))
    out_of_domain = _bool_value(row.get("out_of_domain"))
    priority = _float_value(row.get("priority_target_score"))
    if exact:
        return "known_exact_hash"
    if near_exact:
        return "known_near_exact_chemistry"
    if out_of_domain and priority > 0.0:
        return "novel_out_of_domain_priority_target"
    if out_of_domain:
        return "novel_out_of_domain"
    if priority > 0.0:
        return "novel_priority_target"
    return "novel_in_domain"


def _acquisition_reasons(row: pd.Series) -> str:
    reasons: list[str] = []
    if _bool_value(row.get("exact_chem_hash_hit")):
        reasons.append("exact chem_hash already exists")
    elif _bool_value(row.get("near_exact_chemistry_hit")):
        reasons.append("near-exact reactive chemistry already exists")
    else:
        reasons.append("new reactive chemistry")
    if _bool_value(row.get("out_of_domain")):
        reasons.append("outside current surrogate/reference domain")
    outside_count = _float_value(row.get("outside_input_range_count"))
    if outside_count > 0:
        reasons.append(f"{int(outside_count)} chemistry input(s) outside reference range")
    distance = _float_value(row.get("nearest_scaled_distance"))
    if distance > 0:
        reasons.append(f"nearest scaled distance {distance:.4g}")
    priority = _float_value(row.get("priority_target_score"))
    if priority > 0:
        reasons.append(f"target-priority score {priority:.4g}")
    region_score = _float_value(row.get("target_region_score"))
    if region_score > 0:
        reasons.append(f"target-region score {region_score:.4g}")
    region_distance = _float_value(row.get("target_region_nearest_distance"))
    if region_distance > 0:
        reasons.append(f"target-region nearest distance {region_distance:.4g}")
    score = _float_value(row.get("acquisition_score"))
    reasons.append(f"acquisition score {score:.4g}")
    return "; ".join(reasons)


def _compact_candidate_rows(frame: pd.DataFrame, *, limit: int = 10) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    columns = [
        "meta__recipe_id",
        "meta__material_system",
        "x__age_days",
        "x__w_b",
        "acquisition_bucket",
        "acquisition_score",
        "nearest_scaled_distance",
        "priority_target_score",
        "target_region_score",
        "target_region_nearest_distance",
        "target_region_nearest_recipe_id",
        "acquisition_reason",
    ]
    for row in frame.head(limit).to_dict(orient="records"):
        rows.append({column: row.get(column) for column in columns if column in row})
    return rows


def _write_acquisition_markdown(path: Path, *, summary: dict[str, Any], selected: pd.DataFrame) -> None:
    lines = ["# Global Chemistry Acquisition Candidates", ""]
    lines.append(f"- selected rows: `{summary.get('selected_rows')}`")
    lines.append(f"- max candidates: `{summary.get('max_candidates')}`")
    lines.append(f"- target-priority enabled: `{(summary.get('priority_targets') or {}).get('enabled')}`")
    bucket_counts = summary.get("selection_bucket_counts") or {}
    if bucket_counts:
        lines.extend(["", "## Selection Buckets", ""])
        for bucket, count in bucket_counts.items():
            lines.append(f"- `{bucket}`: `{count}`")
    lines.extend(["", "## Selected Candidates", ""])
    if selected.empty:
        lines.append("No candidates selected.")
    else:
        columns = [
            "meta__recipe_id",
            "meta__material_system",
            "x__age_days",
            "acquisition_bucket",
            "acquisition_score",
            "nearest_scaled_distance",
            "priority_target_score",
            "target_region_score",
            "target_region_nearest_distance",
            "acquisition_reason",
        ]
        use_columns = [column for column in columns if column in selected.columns]
        lines.append("| " + " | ".join(use_columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(use_columns)) + " |")
        for row in selected[use_columns].to_dict(orient="records"):
            lines.append("| " + " | ".join(str(row.get(column, "")).replace("|", "\\|") for column in use_columns) + " |")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def acquire_global_chemistry_candidates(
    *,
    db: str | Path,
    out: str | Path,
    recipes_csv: str | Path | None = None,
    candidate_table: str | Path | None = None,
    reference_model_table: str | Path | None = None,
    model_bundle: str | Path | None = None,
    max_candidates: int = 20,
    dat_lst: str | Path | None = None,
    run_mode: str = "reacted_only",
    temperature_celsius: float = 20.0,
    pressure: float | None = None,
    xgems_water_mode: str = "initial",
    xgems_water_factor: float = 1.0,
    xgems_water_g: float | None = None,
    xgems_water_w_b: float | None = None,
    reaction_model_id: str | None = None,
    reaction_model_config: str | Path | None = None,
    nearest_distance_warn: float | None = None,
    schema_config: str | Path | None = None,
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
    schema = _load_schema(schema_config)
    acq_cfg = schema.get("acquisition") or {}
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    lookup_dir = lookup_global_chemistry(
        db=db,
        out=out_dir / "lookup",
        recipes_csv=recipes_csv,
        candidate_table=candidate_table,
        reference_model_table=reference_model_table,
        model_bundle=model_bundle,
        dat_lst=dat_lst,
        run_mode=run_mode,
        temperature_celsius=temperature_celsius,
        pressure=pressure,
        xgems_water_mode=xgems_water_mode,
        xgems_water_factor=xgems_water_factor,
        xgems_water_g=xgems_water_g,
        xgems_water_w_b=xgems_water_w_b,
        reaction_model_id=reaction_model_id,
        reaction_model_config=reaction_model_config,
        nearest_distance_warn=nearest_distance_warn,
        schema_config=schema_config,
    )
    report = pd.read_csv(Path(lookup_dir) / "global_chemistry_lookup.csv")
    bundle_path = Path(model_bundle) if model_bundle else _path_from_manifest(load_global_manifest(db, schema_config=schema_config), "model_bundle")
    requested_priority_targets = list(priority_targets or [])
    diagnostic_statuses = tuple(priority_target_statuses or ("usable_with_caution", "not_recommended"))
    requested_priority_targets.extend(
        _priority_targets_from_diagnostics(
            priority_targets_from_diagnostics,
            statuses=diagnostic_statuses,
            target_kinds=priority_target_kinds,
            reliabilities=priority_target_reliabilities,
            limit=priority_target_limit,
        )
    )
    requested_priority_targets = list(dict.fromkeys([str(target) for target in requested_priority_targets if str(target).strip()]))
    report, priority_report = _add_surrogate_priority_scores(
        report,
        model_bundle=bundle_path,
        priority_targets=requested_priority_targets,
    )
    report, target_region_report = _add_target_region_scores(
        report,
        target_region_tables=target_region_table,
        schema=schema,
        distance_scale=float(
            target_region_distance_scale
            if target_region_distance_scale is not None
            else acq_cfg.get("target_region_distance_scale", 0.25)
        ),
        max_reference_rows=(
            target_region_max_reference_rows
            if target_region_max_reference_rows is not None
            else acq_cfg.get("target_region_max_reference_rows", 1000)
        ),
    )
    exact_hit = report.get("exact_chem_hash_hit", pd.Series(False, index=report.index)).fillna(False).astype(bool)
    near_exact_hit = report.get("near_exact_chemistry_hit", pd.Series(False, index=report.index)).fillna(False).astype(bool)
    already_known = exact_hit | near_exact_hit
    out_of_domain = report.get("out_of_domain", pd.Series(False, index=report.index)).fillna(False).astype(bool)
    outside_count = pd.to_numeric(report.get("outside_input_range_count", pd.Series(0, index=report.index)), errors="coerce").fillna(0.0)
    distance = pd.to_numeric(report.get("nearest_scaled_distance", pd.Series(0, index=report.index)), errors="coerce").fillna(0.0)
    score = (
        (~already_known).astype(float) * float(acq_cfg.get("novelty_bonus", 10.0))
        - already_known.astype(float) * float(acq_cfg.get("exact_hit_penalty", 100.0))
        + out_of_domain.astype(float) * float(acq_cfg.get("out_of_domain_bonus", 100.0))
        + outside_count * float(acq_cfg.get("outside_range_bonus", 1000.0))
        + distance * float(acq_cfg.get("nearest_distance_weight", 10.0))
        + pd.to_numeric(report.get("priority_target_score", pd.Series(0.0, index=report.index)), errors="coerce").fillna(0.0)
        * float(target_priority_weight if target_priority_weight is not None else acq_cfg.get("target_priority_weight", 25.0))
        + pd.to_numeric(report.get("target_region_score", pd.Series(0.0, index=report.index)), errors="coerce").fillna(0.0)
        * float(target_region_weight if target_region_weight is not None else acq_cfg.get("target_region_weight", 50.0))
    )
    report["acquisition_score"] = score
    if "nearest_scaled_distance" not in report.columns:
        report["nearest_scaled_distance"] = 0.0
    report["acquisition_bucket"] = report.apply(_acquisition_bucket, axis=1)
    report["acquisition_reason"] = report.apply(_acquisition_reasons, axis=1)
    report = report.sort_values(["acquisition_score", "nearest_scaled_distance"], ascending=[False, False])
    hash_column = str((schema.get("identity") or {}).get("exact_hash_column", "meta__chem_hash"))
    if hash_column in report.columns:
        report = report.drop_duplicates(subset=[hash_column], keep="first")
    selected = report.head(int(max_candidates)).copy()
    selected_path = _write_table(selected, out_dir / "acquisition_candidates.csv")
    recipe_rows = _recipe_rows_from_candidate_frame(selected)
    recipes_path = out_dir / "acquisition_recipes.csv"
    write_recipe_csv(recipes_path, recipe_rows)
    summary = {
        "db": str(db),
        "candidate_rows": int(len(report)),
        "selected_rows": int(len(selected)),
        "max_candidates": int(max_candidates),
        "priority_targets": priority_report,
        "priority_target_diagnostics_filter": {
            "statuses": list(diagnostic_statuses),
            "kinds": list(priority_target_kinds or []),
            "reliabilities": list(priority_target_reliabilities or []),
            "limit": priority_target_limit,
        },
        "target_priority_weight": float(target_priority_weight if target_priority_weight is not None else acq_cfg.get("target_priority_weight", 25.0)),
        "target_region": target_region_report,
        "target_region_weight": float(target_region_weight if target_region_weight is not None else acq_cfg.get("target_region_weight", 50.0)),
        "selection_bucket_counts": selected.get("acquisition_bucket", pd.Series(dtype=str)).value_counts().to_dict(),
        "top_candidates": _compact_candidate_rows(selected),
        "outputs": {
            "lookup": str(lookup_dir),
            "acquisition_candidates": str(selected_path),
            "acquisition_recipes": str(recipes_path),
            "markdown": str(out_dir / "acquisition_candidates.md"),
        },
    }
    write_json(out_dir / "acquisition_summary.json", summary)
    _write_acquisition_markdown(out_dir / "acquisition_candidates.md", summary=summary, selected=selected)
    return out_dir


def run_global_forward_query(
    *,
    db: str | Path,
    query: str | Path,
    out: str | Path,
    **kwargs: Any,
) -> Path:
    load_global_manifest(db)
    return run_forward_query(query=query, out=out, db=db, **kwargs)


def run_global_design_query(
    *,
    db: str | Path,
    query: str | Path,
    out: str | Path,
    model_bundle: str | Path | None = None,
    reference_model_table: str | Path | None = None,
    **kwargs: Any,
) -> Path:
    manifest = load_global_manifest(db)
    bundle = Path(model_bundle) if model_bundle else _path_from_manifest(manifest, "model_bundle")
    model_table = Path(reference_model_table) if reference_model_table else _path_from_manifest(manifest, "model_table")
    if bundle is None or not bundle.exists():
        raise ValueError("Global DB has no trained model bundle yet. Run train-global-chem-surrogate first.")
    if model_table is None or not model_table.exists():
        raise ValueError("Global DB has no model table yet. Run refresh-global-chem-db first.")
    return run_chemistry_design_query(
        query=query,
        out=out,
        db=db,
        model_bundle=bundle,
        reference_model_table=model_table,
        **kwargs,
    )


def copy_existing_artifacts_into_global_db(
    *,
    db: str | Path,
    feature_table: str | Path | None = None,
    model_table: str | Path | None = None,
    model_bundle: str | Path | None = None,
    schema_config: str | Path | None = None,
) -> dict[str, Any]:
    manifest = load_global_manifest(db, schema_config=schema_config)
    copied: dict[str, str] = {}
    for key, source in [("feature_table", feature_table), ("model_table", model_table), ("model_bundle", model_bundle)]:
        if source is None:
            continue
        target = _path_from_manifest(manifest, key)
        if target is None:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied[key] = str(target)
    row_counts = {}
    for key in ["feature_table", "model_table"]:
        path = _path_from_manifest(manifest, key)
        if path and path.exists():
            row_counts[key] = int(len(_read_table(path)))
    return _update_manifest(db, {"paths": copied, "row_counts": row_counts}, schema_config=schema_config)


def _table_columns(database: InverseGemsDatabase, table: str) -> list[str]:
    with database.connect() as conn:
        return [str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def _copy_run_directory(source: Path, target: Path) -> None:
    if not source.exists() or not source.is_dir():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, dirs_exist_ok=True)


def import_cached_db_into_global_db(
    *,
    db: str | Path,
    source_db: str | Path,
    schema_config: str | Path | None = None,
    copy_run_dirs: bool = True,
) -> dict[str, Any]:
    """Merge an existing cached inverse_gems DB into a global chemistry DB.

    Chemistry and prepared-chemistry rows are upserted by their natural keys.
    Recipe rows are replaced by recipe_id. Source-contribution rows for replaced
    recipe_ids are deleted first to avoid duplicate provenance rows.
    """

    initialize_global_chemistry_db(db=db, schema_config=schema_config)
    target = InverseGemsDatabase(db)
    source = InverseGemsDatabase(source_db)
    imported: dict[str, int] = {}

    for table in ["chemistry_runs", "prepared_chemistry_runs", "recipe_runs"]:
        target_columns = _table_columns(target, table)
        source_columns = _table_columns(source, table)
        columns = [column for column in target_columns if column in source_columns]
        if not columns:
            imported[table] = 0
            continue
        with source.connect() as source_conn:
            rows = [dict(row) for row in source_conn.execute(f"SELECT {','.join(columns)} FROM {table}").fetchall()]
        if copy_run_dirs:
            for row in rows:
                if table == "chemistry_runs":
                    chem_hash = str(row.get("chem_hash") or "")
                    if chem_hash:
                        src = Path(str(row.get("xgems_run_dir") or source.chemistry_runs_dir / chem_hash))
                        dst = target.chemistry_dir(chem_hash)
                        _copy_run_directory(src, dst)
                        row["xgems_run_dir"] = str(dst)
                elif table == "recipe_runs":
                    recipe_id = str(row.get("recipe_id") or "")
                    if recipe_id:
                        src = Path(str(row.get("run_dir") or source.recipe_runs_dir / recipe_id))
                        dst = target.recipe_dir(recipe_id)
                        _copy_run_directory(src, dst)
                        row["run_dir"] = str(dst)
                elif table == "prepared_chemistry_runs":
                    prepared_id = str(row.get("prepared_id") or "")
                    if prepared_id:
                        _copy_run_directory(source.prepared_chemistry_dir(prepared_id), target.prepared_chemistry_dir(prepared_id))
        placeholders = ",".join("?" for _ in columns)
        updates = ",".join(f"{column}=excluded.{column}" for column in columns)
        with target.connect() as target_conn:
            target_conn.executemany(
                f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders}) "
                f"ON CONFLICT DO UPDATE SET {updates}",
                [[row.get(column) for column in columns] for row in rows],
            )
        imported[table] = len(rows)

    source_contribution_columns = [column for column in _table_columns(target, "source_contributions") if column in _table_columns(source, "source_contributions")]
    with source.connect() as source_conn:
        source_rows = [
            dict(row)
            for row in source_conn.execute(
                f"SELECT {','.join(source_contribution_columns)} FROM source_contributions"
            ).fetchall()
        ] if source_contribution_columns else []
    recipe_ids = sorted({str(row.get("recipe_id")) for row in source_rows if row.get("recipe_id") not in (None, "")})
    if source_rows:
        with target.connect() as target_conn:
            if recipe_ids:
                target_conn.executemany("DELETE FROM source_contributions WHERE recipe_id = ?", [(recipe_id,) for recipe_id in recipe_ids])
            target_conn.executemany(
                f"INSERT INTO source_contributions ({','.join(source_contribution_columns)}) "
                f"VALUES ({','.join('?' for _ in source_contribution_columns)})",
                [[row.get(column) for column in source_contribution_columns] for row in source_rows],
            )
    imported["source_contributions"] = len(source_rows)

    manifest = _update_manifest(
        db,
        {
            "last_import": {
                "source_db": str(source_db),
                "copy_run_dirs": bool(copy_run_dirs),
                "imported_rows": imported,
            }
        },
        schema_config=schema_config,
    )
    manifest["imported_rows"] = imported
    return manifest
