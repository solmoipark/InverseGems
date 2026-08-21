from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .cached_forward import run_forward_cached
from .feature_table import build_feature_table
from .utils import config_path, load_yaml, write_json


BINDER_COLUMNS: tuple[tuple[str, str], ...] = (
    ("x__OPC", "OPC"),
    ("x__slag", "slag"),
    ("x__fly_ash", "fly ash"),
    ("x__metakaolin", "metakaolin"),
    ("x__silica_fume", "silica fume"),
    ("x__limestone", "limestone"),
    ("x__gypsum", "gypsum"),
)

DEFAULT_TARGET_SOURCES: dict[str, str] = {
    "y__porosity": "porosity",
    "y__pH": "scalar__pH",
    "y__amount_C_A_S_H": "phase_amount_group__C-A-S-H",
    "y__amount_ettringite": "phase_amount_group__ettringite",
    "y__amount_monocarbonate": "phase_amount_group__monocarbonate",
    "y__amount_Calcite": "phase_amount_group__Calcite",
    "y__amount_Portlandite": "phase_amount_group__Portlandite",
}

SAFE_NAME_RE = re.compile(r"[^0-9A-Za-z_]+")


def _safe_name(name: str) -> str:
    cleaned = SAFE_NAME_RE.sub("_", str(name)).strip("_")
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned or "unnamed"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _as_float(value: Any, *, default: float | None = None) -> float | None:
    if _is_missing(value):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _format_number(value: float) -> str:
    return f"{float(value):.12g}"


def candidate_recipe_text(row: Mapping[str, Any]) -> str:
    """Rebuild a deterministic recipe string from surrogate candidate columns.

    Prefers x__ binder feature columns. Model bundles whose surrogate features
    are chemistry vectors (e.g. registry-selected global chemistry models expose
    x__chem_oxide_equiv_mol_* features) carry the recipe composition in meta__
    columns instead, so meta__ binder columns are used as a fallback.
    """
    parts: list[str] = []
    for prefix in ("x__", "meta__"):
        for column, label in BINDER_COLUMNS:
            value = _as_float(row.get(prefix + column.removeprefix("x__")), default=0.0)
            if value is not None and abs(value) > 1.0e-12:
                parts.append(f"{label} {_format_number(value)}")
        if parts:
            break
    if not parts:
        raise ValueError("Candidate row has no nonzero binder x__ or meta__ columns.")

    w_b = _as_float(_source_value(row, "x__w_b", "w_b", "meta__w_b"))
    if w_b is not None:
        parts.append(f"w/b {_format_number(w_b)}")
    else:
        water_g = _as_float(_source_value(row, "x__water_g", "water_g", "meta__water_g"))
        if water_g is None:
            raise ValueError("Candidate row must include x__w_b/meta__w_b or water_g.")
        parts.append(f"water {_format_number(water_g)}")

    age = None
    for name in ("x__age_days", "age_days", "age", "meta__age_days"):
        value = _as_float(row.get(name))
        if value is not None and value > 0:
            age = value
            break
    if age is None:
        raise ValueError("Candidate row must include positive x__age_days or meta__age_days.")
    parts.append(f"age {_format_number(age)}")
    return ", ".join(parts)


def _read_candidates(path: str | Path, top_k: int | None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if top_k is not None and top_k > 0:
        frame = frame.head(top_k)
    return frame.reset_index(drop=True)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _source_value(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = row.get(name)
        if not _is_missing(value):
            return value
    return None


def _rank_value(row: Mapping[str, Any], index: int) -> int:
    value = _as_float(row.get("rank"), default=None)
    if value is None:
        return index + 1
    return int(value)


def _target_sources_from_selection_config(selection_config: dict[str, Any]) -> dict[str, str]:
    phase_groups = (selection_config.get("phase_groups") or {}) if selection_config else {}
    sources: dict[str, str] = {}
    for section, prefix in [
        ("amounts", "phase_amount_group__"),
        ("volumes", "phase_volume_group__"),
    ]:
        for group_name in ((phase_groups.get(section) or {}) if isinstance(phase_groups, dict) else {}):
            safe = _safe_name(str(group_name))
            kind = "amount" if section == "amounts" else "volume"
            sources[f"y__{kind}_{safe}"] = f"{prefix}{group_name}"
    return sources


def _fallback_target_source(target: str) -> str | None:
    if target.startswith("y__amount_"):
        return f"phase_amount_group__{target.removeprefix('y__amount_')}"
    if target.startswith("y__volume_"):
        return f"phase_volume_group__{target.removeprefix('y__volume_')}"
    if target.startswith("y__") and target not in {"y__porosity", "y__pH"}:
        return f"scalar__{target.removeprefix('y__')}"
    return None


def _target_sources_from_candidates(frame: pd.DataFrame, selection_config: dict[str, Any] | None = None) -> dict[str, str]:
    configured_sources = _target_sources_from_selection_config(selection_config or {})
    targets: dict[str, str] = {}
    for column in frame.columns:
        if not column.startswith("pred__"):
            continue
        target = column.removeprefix("pred__")
        if target in DEFAULT_TARGET_SOURCES:
            targets[target] = DEFAULT_TARGET_SOURCES[target]
        elif target in configured_sources:
            targets[target] = configured_sources[target]
        else:
            fallback = _fallback_target_source(target)
            if fallback is not None:
                targets[target] = fallback
    return targets


def _comparison_rows(
    *,
    candidates: pd.DataFrame,
    run_records: list[dict[str, Any]],
    validation_features: pd.DataFrame,
    target_sources: dict[str, str],
) -> list[dict[str, Any]]:
    feature_by_recipe = {
        str(row["recipe_id"]): row
        for row in validation_features.to_dict(orient="records")
        if not _is_missing(row.get("recipe_id"))
    }
    rows: list[dict[str, Any]] = []
    for record in run_records:
        index = int(record["candidate_index"])
        candidate = candidates.iloc[index].to_dict()
        recipe_id = str(record.get("recipe_id") or "")
        feature = feature_by_recipe.get(recipe_id, {})
        row: dict[str, Any] = {
            "candidate_rank": record.get("candidate_rank"),
            "candidate_index": index,
            "source_recipe_id": record.get("source_recipe_id"),
            "validation_recipe_id": recipe_id or None,
            "source_chem_hash": record.get("source_chem_hash"),
            "validation_chem_hash": record.get("chem_hash"),
            "chemistry_status": record.get("chemistry_status"),
            "solver_status": record.get("solver_status"),
            "solver_rescued": record.get("solver_rescued"),
            "xgems_retry_count": record.get("xgems_retry_count"),
            "preflight_dir": record.get("preflight_dir"),
            "recipe_text": record.get("recipe_text"),
            "error": record.get("error"),
            "out_of_domain": candidate.get("out_of_domain"),
            "nearest_scaled_distance": candidate.get("nearest_scaled_distance"),
            "outside_input_range_count": candidate.get("outside_input_range_count"),
            "outside_input_range_columns": candidate.get("outside_input_range_columns"),
            "nearest_reference_recipe_id": candidate.get("nearest_reference_recipe_id"),
            "nearest_reference_chem_hash": candidate.get("nearest_reference_chem_hash"),
        }
        for target, feature_column in target_sources.items():
            pred = _as_float(candidate.get(f"pred__{target}"))
            source_true = _as_float(candidate.get(f"true__{target}"))
            validated = _as_float(feature.get(feature_column))
            row[f"pred__{target}"] = pred
            row[f"source_true__{target}"] = source_true
            row[f"validated__{target}"] = validated
            if pred is not None and validated is not None:
                row[f"diff_validated_minus_pred__{target}"] = validated - pred
                row[f"abs_diff_validated_minus_pred__{target}"] = abs(validated - pred)
            else:
                row[f"diff_validated_minus_pred__{target}"] = None
                row[f"abs_diff_validated_minus_pred__{target}"] = None
            if source_true is not None and validated is not None:
                row[f"diff_validated_minus_source_true__{target}"] = validated - source_true
                row[f"abs_diff_validated_minus_source_true__{target}"] = abs(validated - source_true)
            else:
                row[f"diff_validated_minus_source_true__{target}"] = None
                row[f"abs_diff_validated_minus_source_true__{target}"] = None
        rows.append(row)
    return rows


def _target_error_summary(comparison: pd.DataFrame, target_sources: dict[str, str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for target in target_sources:
        pred_column = comparison.get(f"abs_diff_validated_minus_pred__{target}", pd.Series(dtype=float))
        true_column = comparison.get(f"abs_diff_validated_minus_source_true__{target}", pd.Series(dtype=float))
        pred_abs = pd.to_numeric(pred_column, errors="coerce")
        true_abs = pd.to_numeric(true_column, errors="coerce")
        pred_abs = pred_abs[pred_abs.notna()]
        true_abs = true_abs[true_abs.notna()]
        summary[target] = {
            "source_column": target_sources[target],
            "validated_vs_pred_count": int(len(pred_abs)),
            "validated_vs_pred_mae": None if pred_abs.empty else float(pred_abs.mean()),
            "validated_vs_pred_max_abs": None if pred_abs.empty else float(pred_abs.max()),
            "validated_vs_source_true_count": int(len(true_abs)),
            "validated_vs_source_true_mae": None if true_abs.empty else float(true_abs.mean()),
            "validated_vs_source_true_max_abs": None if true_abs.empty else float(true_abs.max()),
        }
    return summary


def _pH_reliability_summary(validation_features: pd.DataFrame) -> dict[str, Any]:
    if validation_features.empty or "scalar__pH" not in validation_features.columns:
        return {"pH_rows": 0, "pH_water_unreliable_count": 0, "pH_water_unreliable_fraction": 0.0}
    pH_values = pd.to_numeric(validation_features["scalar__pH"], errors="coerce")
    pH_rows = int(pH_values.notna().sum())
    if "pH_water_reliable" in validation_features.columns:
        reliable = validation_features["pH_water_reliable"].map(lambda value: str(value).strip().lower() in {"true", "1", "1.0", "yes"})
    elif "solver_rescued" in validation_features.columns:
        reliable = ~validation_features["solver_rescued"].map(lambda value: str(value).strip().lower() in {"true", "1", "1.0", "yes"})
    else:
        reliable = pd.Series([True] * len(validation_features), index=validation_features.index)
    unreliable = (~reliable) & pH_values.notna()
    unreliable_count = int(unreliable.sum())
    return {
        "pH_rows": pH_rows,
        "pH_water_unreliable_count": unreliable_count,
        "pH_water_unreliable_fraction": 0.0 if pH_rows <= 0 else float(unreliable_count / pH_rows),
    }


def validate_candidates(
    *,
    candidates: str | Path,
    out: str | Path,
    db: str | Path,
    dat_lst: str | Path | None = None,
    top_k: int | None = 10,
    use_mock: bool = False,
    selection: str | Path | None = None,
    run_mode: str = "reacted_only",
    normalize: bool = True,
    allow_non_100: bool = False,
    temperature_celsius: float = 20.0,
    pressure: float | None = None,
    force_rerun_xgems: bool = False,
    gems_class_path: str | None = "xgems:ChemicalEngineDicts",
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
    fail_fast: bool = False,
) -> Path:
    if not use_mock and dat_lst is None:
        raise ValueError("--dat-lst is required for real candidate validation.")

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    selection_path = Path(selection) if selection is not None else config_path("output_selection.yaml")
    candidates_path = Path(candidates)
    frame = _read_candidates(candidates_path, top_k)
    _write_csv(frame, out_dir / "query_candidates_used.csv")

    selection_config = load_yaml(selection_path) if selection_path.exists() else {}
    target_sources = _target_sources_from_candidates(frame, selection_config)
    run_records: list[dict[str, Any]] = []
    warnings: list[str] = []

    for index, row_series in frame.iterrows():
        row = row_series.to_dict()
        rank = _rank_value(row, index)
        recipe_text = candidate_recipe_text(row)
        source_recipe_id = _source_value(row, "meta__recipe_id", "recipe_id")
        source_chem_hash = _source_value(row, "meta__chem_hash", "chem_hash")
        source_material_system = _source_value(row, "meta__material_system", "material_system")
        record: dict[str, Any] = {
            "candidate_index": int(index),
            "candidate_rank": rank,
            "source_recipe_id": source_recipe_id,
            "source_chem_hash": source_chem_hash,
            "source_material_system": source_material_system,
            "candidate_score": _source_value(row, "score"),
            "recipe_text": recipe_text,
        }
        recipe_metadata = {
            "template_name": "candidate_validation",
            "validation_candidate_rank": rank,
            "validation_source_recipe_id": source_recipe_id,
            "validation_source_chem_hash": source_chem_hash,
        }
        if not _is_missing(source_material_system):
            recipe_metadata["material_system"] = source_material_system
        try:
            result = run_forward_cached(
                recipe_text=recipe_text,
                db=db,
                dat_lst=dat_lst,
                use_mock=use_mock,
                force_rerun_xgems=force_rerun_xgems,
                run_mode=run_mode,
                normalize=normalize,
                allow_non_100=allow_non_100,
                temperature_celsius=temperature_celsius,
                pressure=pressure,
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
                recipe_metadata=recipe_metadata,
            )
            record.update(result)
            record["error"] = None
        except Exception as exc:
            message = f"Candidate rank {rank} failed validation: {type(exc).__name__}: {exc}"
            warnings.append(message)
            record.update(
                {
                    "recipe_id": None,
                    "chem_hash": None,
                    "chemistry_status": "error",
                    "solver_status": None,
                    "solver_rescued": None,
                    "xgems_retry_count": None,
                    "error": message,
                }
            )
            if fail_fast:
                run_records.append(record)
                _write_csv(pd.DataFrame(run_records), out_dir / "validation_runs.csv")
                raise
        run_records.append(record)
        _write_csv(pd.DataFrame(run_records), out_dir / "validation_runs.csv")

    all_features_path = out_dir / "validated_feature_table_all.csv"
    build_feature_table(db=db, selection=selection_path, out=all_features_path, output_format="csv")
    try:
        all_features = pd.read_csv(all_features_path, low_memory=False)
    except pd.errors.EmptyDataError:
        all_features = pd.DataFrame()
    recipe_ids = [str(record["recipe_id"]) for record in run_records if record.get("recipe_id")]
    if "recipe_id" in all_features.columns:
        validation_features = all_features[all_features["recipe_id"].astype(str).isin(recipe_ids)].copy()
    else:
        validation_features = pd.DataFrame()
    if recipe_ids and not validation_features.empty:
        order = pd.DataFrame({"recipe_id": recipe_ids, "_validation_order": range(len(recipe_ids))})
        validation_features = validation_features.merge(order, on="recipe_id", how="left")
        validation_features = validation_features.sort_values("_validation_order").drop(columns=["_validation_order"])
    _write_csv(validation_features, out_dir / "validated_feature_table.csv")

    comparison = pd.DataFrame(
        _comparison_rows(
            candidates=frame,
            run_records=run_records,
            validation_features=validation_features,
            target_sources=target_sources,
        )
    )
    _write_csv(comparison, out_dir / "validation_comparison.csv")
    pH_reliability = _pH_reliability_summary(validation_features)
    if pH_reliability.get("pH_water_unreliable_count", 0) > 0:
        warnings.append(
            "Validated pH includes water-adjusted/rescued xGEMS rows; pH should be treated as conditional on the adjusted xGEMS water."
        )

    status_counts = {}
    if run_records:
        status_counts = (
            pd.Series([str(record.get("chemistry_status")) for record in run_records])
            .value_counts(dropna=False)
            .to_dict()
        )
    summary = {
        "candidates_path": str(candidates_path),
        "selection_path": str(selection_path),
        "db": str(db),
        "out": str(out_dir),
        "requested_top_k": top_k,
        "candidate_rows_used": int(len(frame)),
        "validation_runs": int(len(run_records)),
        "validation_status_counts": status_counts,
        "target_sources": target_sources,
        "target_error_summary": _target_error_summary(comparison, target_sources),
        "pH_reliability": pH_reliability,
        "warnings": warnings,
        "outputs": {
            "query_candidates_used": str(out_dir / "query_candidates_used.csv"),
            "validation_runs": str(out_dir / "validation_runs.csv"),
            "validated_feature_table": str(out_dir / "validated_feature_table.csv"),
            "validated_feature_table_all": str(all_features_path),
            "validation_comparison": str(out_dir / "validation_comparison.csv"),
        },
    }
    write_json(out_dir / "validation_summary.json", summary)
    return out_dir
