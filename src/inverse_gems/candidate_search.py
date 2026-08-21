from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .reaction_provenance import (
    check_reaction_provenance_compatibility,
    expected_reaction_provenance_from_query,
    reaction_provenance_from_frame,
)
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


def _safe_name(name: str) -> str:
    cleaned = SAFE_NAME_RE.sub("_", str(name)).strip("_")
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned or "unnamed"


def _load_bundle(path: str | Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "estimator" not in bundle:
        raise ValueError(f"Surrogate bundle must be a dict with an estimator: {path}")
    return bundle


def _predict_with_bundle(frame: pd.DataFrame, bundle: dict[str, Any]) -> tuple[pd.DataFrame, list[str], list[str]]:
    inputs = [str(column) for column in bundle.get("inputs", [])]
    targets = [str(column) for column in bundle.get("targets", [])]
    missing = [column for column in inputs if column not in frame.columns]
    if missing:
        raise ValueError(f"Candidate model table is missing surrogate input columns: {missing}")
    predictions = bundle["estimator"].predict(frame[inputs])
    if predictions.ndim == 1:
        predictions = predictions[:, None]
    out = frame.copy()
    for index, target in enumerate(targets):
        out[f"pred__{target}"] = predictions[:, index]
        if target in out.columns:
            out[f"true__{target}"] = out[target]
            out[f"residual__{target}"] = out[target] - predictions[:, index]
    return out, inputs, targets


def _candidate_columns_for_name(name: str, *, predicted: bool = True) -> list[str]:
    safe = _safe_name(name)
    prefixes = ["pred__"] if predicted else [""]
    target_stems = [
        f"y__{safe}",
        f"y__amount_{safe}",
        f"y__volume_{safe}",
    ]
    columns: list[str] = []
    for prefix in prefixes:
        columns.extend([f"{prefix}{stem}" for stem in target_stems])
    columns.extend(
        [
            name,
            f"x__{safe}",
            f"meta__{safe}",
            f"pred__{name}",
            f"true__{name}",
        ]
    )
    return list(dict.fromkeys(columns))


def _resolve_column(frame: pd.DataFrame, name: str, *, role: str) -> str:
    if name in frame.columns:
        return name
    safe = _safe_name(name)
    if role == "input":
        candidates = [f"x__{safe}", f"meta__{safe}"]
    elif role == "metadata":
        candidates = [f"meta__{safe}", f"x__{safe}"]
    elif role == "target":
        candidates = _candidate_columns_for_name(name, predicted=True)
    else:
        candidates = _candidate_columns_for_name(name, predicted=True) + [f"x__{safe}", f"meta__{safe}"]
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
    if spec.get("include") is not None:
        include = spec["include"]
        include = include if isinstance(include, list) else [include]
        try:
            return all(float(value) == 0.0 for value in include)
        except (TypeError, ValueError):
            return False
    return True


def _apply_range_or_include(frame: pd.DataFrame, column: str, spec: dict[str, Any]) -> tuple[pd.DataFrame, int]:
    before = len(frame)
    if spec.get("include") is not None:
        include = spec["include"]
        if isinstance(include, str):
            include = [include]
        frame = frame[frame[column].isin(include)]
    if spec.get("exclude") is not None:
        exclude = spec["exclude"]
        if isinstance(exclude, str):
            exclude = [exclude]
        frame = frame[~frame[column].isin(exclude)]
    if spec.get("min") is not None:
        frame = frame[pd.to_numeric(frame[column], errors="coerce") >= float(spec["min"])]
    if spec.get("max") is not None:
        frame = frame[pd.to_numeric(frame[column], errors="coerce") <= float(spec["max"])]
    if spec.get("equals") is not None:
        values = frame[column]
        numeric = pd.to_numeric(values, errors="coerce")
        try:
            equals = float(spec["equals"])
            tolerance = float(spec.get("tolerance", 1.0e-12))
            frame = frame[(numeric - equals).abs() <= tolerance]
        except (TypeError, ValueError):
            frame = frame[values.astype(str) == str(spec["equals"])]
    return frame, before - len(frame)


def _apply_constraints(frame: pd.DataFrame, query: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int], list[dict[str, Any]]]:
    rejected: dict[str, int] = {}
    resolved: list[dict[str, Any]] = []
    constraints = query.get("constraints", {}) or {}
    for section, role in [("metadata", "metadata"), ("inputs", "input"), ("predicted_targets", "target")]:
        for name, spec in (constraints.get(section, {}) or {}).items():
            try:
                column = _resolve_column(frame, str(name), role=role)
            except ValueError:
                if role == "input" and _is_absent_zero_constraint(spec or {}):
                    key = f"{section}:{name}"
                    rejected[key] = 0
                    resolved.append(
                        {
                            "section": section,
                            "name": str(name),
                            "column": "<absent input assumed zero>",
                            "rejected": 0,
                            "spec": spec,
                        }
                    )
                    continue
                raise
            frame, count = _apply_range_or_include(frame, column, spec or {})
            key = f"{section}:{name}"
            rejected[key] = int(count)
            resolved.append({"section": section, "name": str(name), "column": column, "rejected": int(count), "spec": spec})
    return frame, rejected, resolved


def _normalize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    lo = float(values.min())
    hi = float(values.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) <= 0.0:
        return pd.Series(0.0, index=series.index)
    return (values - lo) / (hi - lo)


def _score_candidates(frame: pd.DataFrame, query: dict[str, Any]) -> tuple[pd.Series, list[dict[str, Any]]]:
    score_config = query.get("score", {}) or {}
    normalize = bool(score_config.get("normalize", "minmax") == "minmax")
    objectives = query.get("objectives", {}) or {}
    score = pd.Series(0.0, index=frame.index)
    resolved: list[dict[str, Any]] = []
    for direction, sign in [("minimize", 1.0), ("maximize", -1.0)]:
        for name, weight in (objectives.get(direction, {}) or {}).items():
            column = _resolve_column(frame, str(name), role="objective")
            raw = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
            contribution_values = _normalize(raw) if normalize else raw
            contribution = sign * float(weight) * contribution_values
            score += contribution
            frame[f"score_component__{_safe_name(direction)}__{_safe_name(str(name))}"] = contribution
            resolved.append(
                {
                    "direction": direction,
                    "name": str(name),
                    "column": column,
                    "weight": float(weight),
                    "normalized": normalize,
                }
            )
    return score, resolved


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
        if section in {"target", "targets", "validated_targets", "source_true_targets"}:
            section = default_target_section
        elif section in {"input", "inputs"}:
            section = "inputs"
        elif section in {"metadata", "meta"}:
            section = "metadata"
        elif section != default_target_section:
            section = default_target_section
        item = {"section": section, "name": str(name), "direction": direction}
        if data.get("tolerance") is not None:
            item["tolerance"] = data["tolerance"]
        items.append(item)
    return items


def _role_for_preference(section: str) -> str:
    if section == "inputs":
        return "input"
    if section == "metadata":
        return "metadata"
    return "target"


def _rank_candidates(frame: pd.DataFrame, query: dict[str, Any], target_columns: list[str]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    ranking = query.get("ranking", {}) or {}
    preferences = _preference_items(query, default_target_section="predicted_targets")
    mode = str(ranking.get("mode") or ("lexicographic" if preferences else "weighted")).lower()

    if mode not in {"weighted", "score", "lexicographic", "ordered"}:
        raise ValueError(f"Unsupported ranking mode: {mode}")
    if mode in {"weighted", "score"}:
        score, resolved_objectives = _score_candidates(frame, query)
        frame["score"] = score
        frame = frame.sort_values(
            ["score"] + [column for column in ["meta__recipe_id", "meta__chem_hash"] if column in frame.columns],
            kind="mergesort",
        )
        return frame, resolved_objectives

    if not preferences:
        preferences = [
            {"section": "predicted_targets", "name": target.removeprefix("y__amount_").removeprefix("y__"), "direction": "maximize"}
            for target in target_columns
        ]
    resolved: list[dict[str, Any]] = []
    sort_columns: list[str] = []
    for index, preference in enumerate(preferences, start=1):
        column = _resolve_column(frame, preference["name"], role=_role_for_preference(preference["section"]))
        numeric = pd.to_numeric(frame[column], errors="coerce")
        tolerance = float(preference.get("tolerance") or 0.0)
        if tolerance > 0.0:
            numeric_for_sort = np.floor(numeric / tolerance)
            sort_value = -numeric_for_sort if preference["direction"] == "maximize" else numeric_for_sort
        else:
            sort_value = -numeric if preference["direction"] == "maximize" else numeric
        component = f"rank_component__{index:02d}__{_safe_name(preference['direction'])}__{_safe_name(preference['name'])}"
        frame[component] = sort_value.fillna(np.inf)
        sort_columns.append(component)
        resolved.append({**preference, "column": column, "component_column": component, "mode": "lexicographic"})

    tie_breakers = [column for column in ["meta__recipe_id", "meta__chem_hash"] if column in frame.columns]
    frame = frame.sort_values(sort_columns + tie_breakers, kind="mergesort")
    frame["score"] = np.arange(len(frame), dtype=float)
    return frame, resolved


def _output_columns(frame: pd.DataFrame, target_columns: list[str]) -> list[str]:
    metadata = [column for column in frame.columns if column.startswith("meta__")]
    inputs = [column for column in frame.columns if column.startswith("x__")]
    predicted = [f"pred__{target}" for target in target_columns if f"pred__{target}" in frame.columns]
    true = [f"true__{target}" for target in target_columns if f"true__{target}" in frame.columns]
    residual = [f"residual__{target}" for target in target_columns if f"residual__{target}" in frame.columns]
    score_components = [column for column in frame.columns if column.startswith("score_component__")]
    rank_components = [column for column in frame.columns if column.startswith("rank_component__")]
    return list(dict.fromkeys(["rank", "score"] + metadata + inputs + predicted + true + residual + score_components + rank_components))


def run_surrogate_candidate_search(
    *,
    query: str | Path,
    out: str | Path,
    model_table: str | Path | None = None,
    model_bundle: str | Path | None = None,
    reaction_model_id: str | None = None,
    reaction_model_signature: str | None = None,
    reaction_model_mismatch_policy: str | None = None,
) -> Path:
    query_path = Path(query)
    query_data = load_yaml(query_path)
    model_table_path = Path(model_table or query_data.get("model_table") or "")
    model_bundle_path = Path(model_bundle or query_data.get("model_bundle") or "")
    if not model_table_path.exists():
        raise ValueError(f"Model table does not exist: {model_table_path}")
    if not model_bundle_path.exists():
        raise ValueError(f"Model bundle does not exist: {model_bundle_path}")

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = _read_table(model_table_path)
    bundle = _load_bundle(model_bundle_path)
    expected = expected_reaction_provenance_from_query(query_data)
    if reaction_model_id:
        expected["reaction_model_ids"] = sorted(set(expected.get("reaction_model_ids", []) + [str(reaction_model_id)]))
    if reaction_model_signature:
        expected["reaction_model_signatures"] = sorted(
            set(expected.get("reaction_model_signatures", []) + [str(reaction_model_signature)])
        )
    if reaction_model_mismatch_policy is not None:
        expected["mismatch_policy"] = reaction_model_mismatch_policy
    table_provenance = reaction_provenance_from_frame(frame, prefix="meta__")
    bundle_provenance = dict(bundle.get("reaction_provenance") or {})
    compatibility = check_reaction_provenance_compatibility(
        table=table_provenance,
        bundle=bundle_provenance,
        expected=expected,
    )
    provenance_report = {
        "table": table_provenance,
        "bundle": bundle_provenance,
        "expected": expected,
        "compatibility": compatibility,
    }
    write_json(out_dir / "reaction_provenance_report.json", provenance_report)
    if compatibility["errors"]:
        raise ValueError("Reaction-model provenance mismatch: " + "; ".join(compatibility["errors"]))

    predicted, input_columns, target_columns = _predict_with_bundle(frame, bundle)
    original_count = len(predicted)
    filtered, rejected, resolved_constraints = _apply_constraints(predicted, query_data)
    filtered = filtered.copy()
    if len(filtered):
        filtered, resolved_objectives = _rank_candidates(filtered, query_data, target_columns)
    else:
        resolved_objectives = []
        filtered["score"] = []

    top_k = int(query_data.get("top_k", 50))
    candidates = filtered.head(top_k).copy()
    candidates.insert(0, "rank", range(1, len(candidates) + 1))
    columns = _output_columns(candidates, target_columns)

    candidates[columns].to_csv(out_dir / "candidates.csv", index=False)
    (out_dir / "candidates.json").write_text(candidates[columns].to_json(orient="records", indent=2), encoding="utf-8")
    shutil.copy2(query_path, out_dir / "query_used.yaml")
    write_json(out_dir / "rejected_by_constraint_counts.json", rejected)
    summary = {
        "query": str(query_path),
        "model_table": str(model_table_path),
        "model_bundle": str(model_bundle_path),
        "input_rows": int(original_count),
        "candidate_rows_after_filters": int(len(filtered)),
        "top_k_requested": top_k,
        "top_k_written": int(len(candidates)),
        "input_columns": input_columns,
        "target_columns": target_columns,
        "reaction_provenance": provenance_report,
        "resolved_constraints": resolved_constraints,
        "resolved_objectives": resolved_objectives,
        "output_columns": columns,
    }
    write_json(out_dir / "candidate_search_summary.json", summary)
    return out_dir
