from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import load_yaml, write_json


SAFE_NAME_RE = re.compile(r"[^0-9A-Za-z_]+")
KNOWN_CONSTRAINT_SECTIONS = {
    "columns",
    "metadata",
    "inputs",
    "validated_targets",
    "predicted_targets",
    "source_true_targets",
    "prediction_errors",
}


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_csv(path)
    return pd.read_csv(path)


def _safe_name(name: str) -> str:
    cleaned = SAFE_NAME_RE.sub("_", str(name)).strip("_")
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned or "unnamed"


def _load_features(validation_path: Path, feature_table: str | Path | None) -> pd.DataFrame | None:
    if feature_table is None:
        candidate = validation_path.parent / "validated_feature_table.csv"
        if not candidate.exists():
            return None
        feature_table = candidate
    path = Path(feature_table)
    if not path.exists():
        raise ValueError(f"Feature table does not exist: {path}")
    return _read_table(path)


def _merge_feature_table(validation: pd.DataFrame, features: pd.DataFrame | None) -> pd.DataFrame:
    if features is None or features.empty:
        return validation.copy()
    if "validation_recipe_id" not in validation.columns:
        raise ValueError("Validation comparison table must include validation_recipe_id to merge feature data.")
    if "recipe_id" not in features.columns:
        raise ValueError("Feature table must include recipe_id to merge with validation comparison.")
    merged = validation.merge(
        features.rename(columns={"recipe_id": "validation_recipe_id"}),
        on="validation_recipe_id",
        how="left",
        suffixes=("", "__feature"),
    )
    return merged


def _target_candidates(name: str, *, prefix: str = "validated") -> list[str]:
    safe = _safe_name(name)
    return [
        f"{prefix}__{name}",
        f"{prefix}__y__{safe}",
        f"{prefix}__y__amount_{safe}",
        f"{prefix}__y__volume_{safe}",
        name,
        f"phase_amount_group__{name}",
        f"phase_volume_group__{name}",
        f"phase_amount__{name}",
        f"phase_volume__{name}",
        f"scalar__{name}",
        "porosity" if name == "porosity" else "",
    ]


def _prediction_error_candidates(name: str) -> list[str]:
    safe = _safe_name(name)
    stems = [f"y__{safe}", f"y__amount_{safe}", f"y__volume_{safe}"]
    columns: list[str] = []
    for stem in stems:
        columns.extend(
            [
                f"abs_diff_validated_minus_pred__{stem}",
                f"diff_validated_minus_pred__{stem}",
            ]
        )
    columns.extend([name, f"abs_{name}"])
    return columns


def _resolve_column(frame: pd.DataFrame, name: str, *, role: str) -> str:
    safe = _safe_name(name)
    if role in {"columns", "metadata"}:
        candidates = [name, f"meta__{safe}", f"{name}__feature"]
    elif role == "inputs":
        candidates = [name, f"x__{safe}", f"{name}__feature"]
    elif role == "validated_targets":
        candidates = _target_candidates(name, prefix="validated")
    elif role == "predicted_targets":
        candidates = _target_candidates(name, prefix="pred")
    elif role == "source_true_targets":
        candidates = _target_candidates(name, prefix="source_true") + _target_candidates(name, prefix="true")
    elif role == "prediction_errors":
        candidates = _prediction_error_candidates(name)
    else:
        candidates = (
            _target_candidates(name, prefix="validated")
            + [name, f"x__{safe}", f"meta__{safe}", f"{name}__feature"]
            + _prediction_error_candidates(name)
        )
    candidates = [candidate for candidate in dict.fromkeys(candidates) if candidate]
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"Could not resolve {role} column for '{name}'. Tried: {candidates}")


def _is_absent_zero_constraint(spec: dict[str, Any]) -> bool:
    """Return true when an absent input column can safely be treated as fixed zero."""
    spec = spec or {}
    has_zero_upper = spec.get("max") is not None and float(spec["max"]) <= 0.0
    has_zero_equals = False
    if spec.get("equals") is not None:
        try:
            tolerance = float(spec.get("tolerance", 1.0e-12))
            has_zero_equals = abs(float(spec["equals"])) <= tolerance
        except (TypeError, ValueError):
            has_zero_equals = str(spec["equals"]).strip() in {"0", "0.0"}
    if not has_zero_upper and not has_zero_equals:
        return False
    if spec.get("min") is not None and float(spec["min"]) > 0.0:
        return False
    if spec.get("min_abs") is not None and float(spec["min_abs"]) > 0.0:
        return False
    if spec.get("include") is not None:
        include = spec["include"]
        include = include if isinstance(include, list) else [include]
        try:
            return all(float(value) == 0.0 for value in include)
        except (TypeError, ValueError):
            return False
    return True


def _as_constraint_spec(spec: Any) -> dict[str, Any]:
    if isinstance(spec, dict):
        return spec
    if isinstance(spec, list):
        return {"include": spec}
    return {"include": [spec]}


def _apply_one_constraint(frame: pd.DataFrame, column: str, spec: dict[str, Any]) -> tuple[pd.DataFrame, int]:
    before = len(frame)
    values = frame[column]
    if spec.get("include") is not None:
        include = spec["include"]
        if not isinstance(include, list):
            include = [include]
        frame = frame[values.astype(str).isin([str(item) for item in include])]
        values = frame[column]
    if spec.get("exclude") is not None:
        exclude = spec["exclude"]
        if not isinstance(exclude, list):
            exclude = [exclude]
        frame = frame[~values.astype(str).isin([str(item) for item in exclude])]
        values = frame[column]
    numeric = pd.to_numeric(values, errors="coerce")
    if spec.get("min") is not None:
        frame = frame[numeric >= float(spec["min"])]
        values = frame[column]
        numeric = pd.to_numeric(values, errors="coerce")
    if spec.get("max") is not None:
        frame = frame[numeric <= float(spec["max"])]
        values = frame[column]
        numeric = pd.to_numeric(values, errors="coerce")
    if spec.get("max_abs") is not None:
        frame = frame[numeric.abs() <= float(spec["max_abs"])]
        values = frame[column]
        numeric = pd.to_numeric(values, errors="coerce")
    if spec.get("min_abs") is not None:
        frame = frame[numeric.abs() >= float(spec["min_abs"])]
        values = frame[column]
        numeric = pd.to_numeric(values, errors="coerce")
    if spec.get("equals") is not None:
        try:
            equals = float(spec["equals"])
            tolerance = float(spec.get("tolerance", 1.0e-12))
            frame = frame[(numeric - equals).abs() <= tolerance]
        except (TypeError, ValueError):
            frame = frame[values.astype(str) == str(spec["equals"])]
    return frame, before - len(frame)


def _constraint_items(config: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    constraints = config.get("constraints", {}) or {}
    items: list[tuple[str, str, dict[str, Any]]] = []
    for section, values in constraints.items():
        if section in KNOWN_CONSTRAINT_SECTIONS:
            for name, spec in (values or {}).items():
                items.append((section, str(name), _as_constraint_spec(spec)))
        else:
            items.append(("columns", str(section), _as_constraint_spec(values)))
    return items


def _apply_constraints(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int], list[dict[str, Any]]]:
    rejected: dict[str, int] = {}
    resolved: list[dict[str, Any]] = []
    for section, name, spec in _constraint_items(config):
        try:
            column = _resolve_column(frame, name, role=section)
        except ValueError:
            if section == "inputs" and _is_absent_zero_constraint(spec):
                key = f"{section}:{name}"
                rejected[key] = 0
                resolved.append(
                    {
                        "section": section,
                        "name": name,
                        "column": "<absent input assumed zero>",
                        "spec": spec,
                        "rejected": 0,
                    }
                )
                continue
            raise
        frame, count = _apply_one_constraint(frame, column, spec)
        key = f"{section}:{name}"
        rejected[key] = int(count)
        resolved.append({"section": section, "name": name, "column": column, "spec": spec, "rejected": int(count)})
    return frame, rejected, resolved


def _normalize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    lo = float(values.min())
    hi = float(values.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) <= 0.0:
        return pd.Series(0.0, index=series.index)
    return (values - lo) / (hi - lo)


def _name_is_pH(name: str) -> bool:
    text = str(name).strip().lower().replace("-", "_")
    safe = _safe_name(name).lower()
    return text in {"ph", "p_h", "scalar__ph", "y__ph"} or safe in {"ph", "p_h", "scalar__ph", "y__ph"} or safe.endswith("__ph")


def _objective_items(objectives: dict[str, Any], direction: str) -> list[tuple[str, str, float]]:
    raw = objectives.get(direction, {}) or {}
    items: list[tuple[str, str, float]] = []
    for section, values in raw.items():
        if section in KNOWN_CONSTRAINT_SECTIONS:
            for name, weight in (values or {}).items():
                items.append((section, str(name), float(weight)))
        else:
            items.append(("objective", str(section), float(values)))
    return items


def _score_candidates(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.Series, list[dict[str, Any]], pd.DataFrame]:
    score_config = config.get("score", {}) or {}
    use_minmax = bool(score_config.get("normalize", "minmax") == "minmax")
    objectives = config.get("objectives", {}) or {}
    score = pd.Series(0.0, index=frame.index)
    resolved: list[dict[str, Any]] = []
    frame = frame.copy()
    for direction, sign in [("minimize", 1.0), ("maximize", -1.0)]:
        for section, name, weight in _objective_items(objectives, direction):
            column = _resolve_column(frame, name, role=section)
            raw = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
            contribution_values = _normalize(raw) if use_minmax else raw
            contribution = sign * weight * contribution_values
            score += contribution
            component = f"selection_score_component__{_safe_name(direction)}__{_safe_name(name)}"
            frame[component] = contribution
            resolved.append(
                {
                    "direction": direction,
                    "section": section,
                    "name": name,
                    "column": column,
                    "weight": weight,
                    "normalized": use_minmax,
                }
            )
    return score, resolved, frame


def _preference_items(config: dict[str, Any], *, default_target_section: str) -> list[dict[str, str]]:
    preferences = config.get("preferences")
    if preferences is None:
        preferences = (config.get("ranking", {}) or {}).get("preferences")
    if not preferences:
        return []
    if not isinstance(preferences, list):
        raise ValueError("preferences must be a list when provided.")

    items: list[dict[str, str]] = []
    for index, preference in enumerate(preferences):
        if not isinstance(preference, dict):
            raise ValueError(f"Preference {index + 1} must be a mapping.")
        data = dict(preference)
        if "maximize" in data or "minimize" in data:
            direction = "maximize" if "maximize" in data else "minimize"
            name_value = data[direction]
            if isinstance(name_value, dict):
                if len(name_value) != 1:
                    raise ValueError(f"Preference {index + 1} can name only one section/objective.")
                section, name = next(iter(name_value.items()))
            else:
                section = data.get("section") or data.get("kind") or "target"
                name = name_value
        else:
            direction = str(data.get("direction") or "").lower()
            section = data.get("section") or data.get("kind")
            name = data.get("name") or data.get("target") or data.get("input")
            if section is None:
                section = "inputs" if data.get("input") is not None else "target"
        if direction not in {"minimize", "maximize"}:
            raise ValueError(f"Preference {index + 1} must set direction to minimize or maximize.")
        if not name:
            raise ValueError(f"Preference {index + 1} must include a name.")

        section = str(section)
        if section in {"target", "targets", "predicted_targets", "source_true_targets"}:
            section = default_target_section
        elif section in {"input", "inputs"}:
            section = "inputs"
        elif section in {"metadata", "meta"}:
            section = "metadata"
        elif section in {"prediction_error", "prediction_errors"}:
            section = "prediction_errors"
        elif section != default_target_section:
            section = default_target_section
        item = {"section": section, "name": str(name), "direction": direction}
        if data.get("tolerance") is not None:
            item["tolerance"] = data["tolerance"]
        items.append(item)
    return items


def _pH_usage(config: dict[str, Any]) -> dict[str, bool]:
    constraint = False
    ranking = False
    for section, name, _spec in _constraint_items(config):
        if section in {"validated_targets", "predicted_targets", "source_true_targets", "columns"} and _name_is_pH(name):
            constraint = True
    objectives = config.get("objectives", {}) or {}
    for direction in ["minimize", "maximize"]:
        for section, name, _weight in _objective_items(objectives, direction):
            if section in {"validated_targets", "predicted_targets", "source_true_targets", "objective"} and _name_is_pH(name):
                ranking = True
    for preference in _preference_items(config, default_target_section="validated_targets"):
        if preference["section"] in {"validated_targets", "predicted_targets", "source_true_targets"} and _name_is_pH(preference["name"]):
            ranking = True
    return {"constraint": constraint, "ranking": ranking, "used": constraint or ranking}


def _truthy_series(frame: pd.DataFrame, *columns: str, default: bool = False) -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return frame[column].fillna(default).astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes", "y"})
    return pd.Series([default] * len(frame), index=frame.index, dtype=bool)


def _pH_water_reliable_series(frame: pd.DataFrame) -> pd.Series:
    for column in ["pH_water_reliable", "pH_water_reliable__feature"]:
        if column in frame.columns:
            return frame[column].fillna(True).astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes", "y"})
    flags = pd.Series([""] * len(frame), index=frame.index)
    for column in ["uncertainty_flags", "uncertainty_flags__feature"]:
        if column in frame.columns:
            flags = frame[column].fillna("").astype(str)
            break
    flagged = flags.str.contains("pH_water_uncertain", case=False, regex=False)
    rescued = _truthy_series(frame, "solver_rescued", "solver_rescued__feature")
    water_changed = pd.Series([False] * len(frame), index=frame.index, dtype=bool)
    for column in ["xgems_water_matches_recipe", "xgems_water_matches_recipe__feature"]:
        if column in frame.columns:
            water_changed = ~frame[column].fillna(True).astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes", "y"})
            break
    return ~(flagged | rescued | water_changed)


def _pH_uncertainty_policy(config: dict[str, Any]) -> dict[str, Any]:
    raw = dict(((config.get("uncertainty_policy") or {}).get("pH") or {}))
    raw.update(config.get("pH_uncertainty_policy") or {})
    usage = _pH_usage(config)
    enabled = bool(raw.get("enabled", True))
    apply_when_used = bool(raw.get("apply_when_pH_used", True))
    requested_mode = str(raw.get("mode") or "auto").lower()
    if not enabled or (apply_when_used and not usage["used"]):
        mode = "ignore"
    elif requested_mode == "auto":
        mode = "exclude" if usage["constraint"] else "penalize" if usage["ranking"] else "ignore"
    else:
        mode = requested_mode
    if mode not in {"ignore", "penalize", "exclude"}:
        raise ValueError("pH_uncertainty_policy.mode must be one of auto, ignore, penalize, exclude.")
    return {
        "enabled": enabled,
        "requested_mode": requested_mode,
        "mode": mode,
        "apply_when_pH_used": apply_when_used,
        "pH_used_as_constraint": usage["constraint"],
        "pH_used_in_ranking": usage["ranking"],
        "pH_used": usage["used"],
        "penalty": float(raw.get("penalty", 1.0e6)),
    }


def _apply_pH_uncertainty_policy(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    policy = _pH_uncertainty_policy(config)
    policy["input_rows"] = int(len(frame))
    if frame.empty or policy["mode"] == "ignore":
        policy["unreliable_rows"] = 0
        policy["rejected_rows"] = 0
        policy["penalized_rows"] = 0
        return frame, policy
    frame = frame.copy()
    reliable = _pH_water_reliable_series(frame)
    unreliable = ~reliable
    frame["selection_pH_water_unreliable"] = unreliable
    policy["unreliable_rows"] = int(unreliable.sum())
    if policy["mode"] == "exclude":
        filtered = frame[reliable].copy()
        policy["rejected_rows"] = int(len(frame) - len(filtered))
        policy["penalized_rows"] = 0
        return filtered, policy
    frame["selection_pH_uncertainty_penalty"] = unreliable.astype(float) * float(policy["penalty"])
    policy["rejected_rows"] = 0
    policy["penalized_rows"] = int(unreliable.sum())
    return frame, policy


def _rank_candidates(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    ranking = config.get("ranking", {}) or {}
    preferences = _preference_items(config, default_target_section="validated_targets")
    mode = str(ranking.get("mode") or ("lexicographic" if preferences else "weighted")).lower()

    if mode not in {"weighted", "score", "lexicographic", "ordered"}:
        raise ValueError(f"Unsupported ranking mode: {mode}")
    if mode in {"weighted", "score"}:
        score, resolved_objectives, frame = _score_candidates(frame, config)
        if "selection_pH_uncertainty_penalty" in frame.columns:
            penalty = pd.to_numeric(frame["selection_pH_uncertainty_penalty"], errors="coerce").fillna(0.0)
            score += penalty
            resolved_objectives.append(
                {
                    "mode": "pH_uncertainty_policy",
                    "direction": "penalize",
                    "section": "uncertainty",
                    "name": "pH_water_unreliable",
                    "column": "selection_pH_uncertainty_penalty",
                    "weight": float(penalty.max()) if len(penalty) else 0.0,
                }
            )
        frame["selection_score"] = score
        sort_columns = ["selection_score"] + [column for column in ["candidate_rank", "source_recipe_id"] if column in frame.columns]
        frame = frame.sort_values(sort_columns, kind="mergesort")
        return frame, resolved_objectives

    if not preferences:
        raise ValueError("Lexicographic ranking requires at least one preference.")
    frame = frame.copy()
    resolved: list[dict[str, Any]] = []
    sort_columns: list[str] = []
    if "selection_pH_uncertainty_penalty" in frame.columns:
        component = "selection_rank_component__00__pH_water_reliability"
        penalty = pd.to_numeric(frame["selection_pH_uncertainty_penalty"], errors="coerce").fillna(0.0)
        frame[component] = (penalty > 0.0).astype(int)
        sort_columns.append(component)
        resolved.append(
            {
                "mode": "pH_uncertainty_policy",
                "direction": "penalize",
                "section": "uncertainty",
                "name": "pH_water_unreliable",
                "column": "selection_pH_uncertainty_penalty",
                "component_column": component,
            }
        )
    for index, preference in enumerate(preferences, start=1):
        column = _resolve_column(frame, preference["name"], role=preference["section"])
        numeric = pd.to_numeric(frame[column], errors="coerce")
        tolerance = float(preference.get("tolerance") or 0.0)
        if tolerance > 0.0:
            numeric_for_sort = np.floor(numeric / tolerance)
            sort_value = -numeric_for_sort if preference["direction"] == "maximize" else numeric_for_sort
        else:
            sort_value = -numeric if preference["direction"] == "maximize" else numeric
        component = f"selection_rank_component__{index:02d}__{_safe_name(preference['direction'])}__{_safe_name(preference['name'])}"
        frame[component] = sort_value.fillna(np.inf)
        sort_columns.append(component)
        resolved.append({**preference, "column": column, "component_column": component, "mode": "lexicographic"})

    tie_breakers = [column for column in ["candidate_rank", "source_recipe_id"] if column in frame.columns]
    frame = frame.sort_values(sort_columns + tie_breakers, kind="mergesort")
    frame["selection_score"] = np.arange(len(frame), dtype=float)
    return frame, resolved


def _preferred_output_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "selection_rank",
        "selection_score",
        "candidate_rank",
        "source_recipe_id",
        "validation_recipe_id",
        "source_chem_hash",
        "validation_chem_hash",
        "chemistry_status",
        "solver_status",
        "solver_rescued",
        "xgems_retry_count",
        "preflight_dir",
        "uncertainty_flags",
        "out_of_domain",
        "nearest_scaled_distance",
        "outside_input_range_count",
        "outside_input_range_columns",
        "nearest_reference_recipe_id",
        "nearest_reference_chem_hash",
        "template_name",
        "material_system",
        "recipe_text",
        "OPC",
        "slag",
        "fly_ash",
        "metakaolin",
        "silica_fume",
        "limestone",
        "gypsum",
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
        "selection_pH_water_unreliable",
        "selection_pH_uncertainty_penalty",
        "age_days",
        "age_label",
        "age_bin",
        "validated__y__porosity",
        "validated__y__pH",
        "validated__y__amount_C_A_S_H",
        "validated__y__amount_ettringite",
        "validated__y__amount_monocarbonate",
        "validated__y__amount_Calcite",
        "validated__y__amount_Portlandite",
        "pred__y__porosity",
        "pred__y__pH",
        "pred__y__amount_C_A_S_H",
        "pred__y__amount_ettringite",
        "pred__y__amount_monocarbonate",
        "pred__y__amount_Calcite",
        "pred__y__amount_Portlandite",
        "diff_validated_minus_pred__y__porosity",
        "diff_validated_minus_pred__y__pH",
        "diff_validated_minus_pred__y__amount_C_A_S_H",
        "diff_validated_minus_pred__y__amount_ettringite",
        "xgems_run_dir",
        "error",
    ]
    score_components = [column for column in frame.columns if column.startswith("selection_score_component__")]
    rank_components = [column for column in frame.columns if column.startswith("selection_rank_component__")]
    dynamic_target_columns = [
        column
        for column in frame.columns
        if column.startswith(
            (
                "validated__y__",
                "pred__y__",
                "source_true__y__",
                "diff_validated_minus_pred__y__",
                "abs_diff_validated_minus_pred__y__",
            )
        )
    ]
    existing = [column for column in preferred if column in frame.columns]
    return list(dict.fromkeys(existing + dynamic_target_columns + score_components + rank_components))


def _write_markdown(selected: pd.DataFrame, path: Path) -> None:
    columns = [
        "selection_rank",
        "source_recipe_id",
        "validated__y__porosity",
        "validated__y__amount_C_A_S_H",
        "validated__y__amount_ettringite",
        "validated__y__amount_Portlandite",
        "validated__y__pH",
        "pH_water_reliable",
        "selection_pH_water_unreliable",
        "selection_pH_uncertainty_penalty",
        "selection_score",
        "recipe_text",
    ]
    for column in selected.columns:
        if column.startswith("validated__y__") and column not in columns:
            columns.insert(max(len(columns) - 2, 0), column)
    columns = [column for column in columns if column in selected.columns]
    display = selected[columns].copy()
    for column in display.columns:
        if column in {"selection_rank", "source_recipe_id", "recipe_text"}:
            continue
        display[column] = pd.to_numeric(display[column], errors="coerce").map(
            lambda value: "" if pd.isna(value) else f"{value:.6g}"
        )
    lines = ["# Selected Candidates", ""]
    if display.empty:
        lines.append("No candidates satisfied the selection constraints.")
    else:
        headers = list(display.columns)
        rows = [[str(value) for value in row] for row in display.astype(str).to_numpy().tolist()]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def select_candidates(
    *,
    validation: str | Path,
    config: str | Path,
    out: str | Path,
    feature_table: str | Path | None = None,
) -> Path:
    validation_path = Path(validation)
    config_path = Path(config)
    config_data = load_yaml(config_path)
    validation_frame = _read_table(validation_path)
    features = _load_features(validation_path, feature_table)
    frame = _merge_feature_table(validation_frame, features)
    input_rows = len(frame)

    filtered, rejected, resolved_constraints = _apply_constraints(frame, config_data)
    filtered, pH_policy = _apply_pH_uncertainty_policy(filtered, config_data)
    filtered = filtered.copy()
    if len(filtered):
        filtered, resolved_objectives = _rank_candidates(filtered, config_data)
    else:
        resolved_objectives = []
        filtered["selection_score"] = []

    top_k = int(config_data.get("top_k") or (config_data.get("output", {}) or {}).get("top_k") or 20)
    selected = filtered.head(top_k).copy()
    selected.insert(0, "selection_rank", range(1, len(selected) + 1))
    output_columns = _preferred_output_columns(selected)

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected[output_columns].to_csv(out_dir / "selected_candidates.csv", index=False)
    (out_dir / "selected_candidates.json").write_text(
        selected[output_columns].to_json(orient="records", indent=2),
        encoding="utf-8",
    )
    _write_markdown(selected[output_columns], out_dir / "selected_candidates.md")
    shutil.copy2(config_path, out_dir / "selection_config_used.yaml")
    write_json(out_dir / "rejected_by_selection_constraints.json", rejected)
    write_json(
        out_dir / "selection_summary.json",
        {
            "validation": str(validation_path),
            "feature_table": str(feature_table or (validation_path.parent / "validated_feature_table.csv")),
            "config": str(config_path),
            "out": str(out_dir),
            "input_rows": int(input_rows),
            "candidate_rows_after_constraints": int(len(filtered)),
            "top_k_requested": top_k,
            "top_k_written": int(len(selected)),
            "resolved_constraints": resolved_constraints,
            "resolved_objectives": resolved_objectives,
            "pH_uncertainty_policy": pH_policy,
            "output_columns": output_columns,
        },
    )
    return out_dir
