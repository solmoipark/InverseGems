from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .utils import load_yaml, write_json


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_csv(path)
    return pd.read_csv(path)


def _feature_column(section: str, name: str) -> str:
    if section == "phase_amounts":
        return f"phase_amount__{name}"
    if section == "phase_volumes":
        return f"phase_volume__{name}"
    if section == "aqueous_species":
        return f"aqueous__{name}"
    if section == "scalars":
        return "porosity" if name == "porosity" else f"scalar__{name}"
    return name


def _ensure_column(frame: pd.DataFrame, column: str, missing_policy: str) -> pd.DataFrame:
    if column not in frame.columns:
        if missing_policy == "zero":
            frame[column] = 0.0
        else:
            raise ValueError(f"Missing required feature column '{column}'.")
    return frame


def _apply_range_filter(frame: pd.DataFrame, column: str, bounds: dict[str, Any]) -> tuple[pd.DataFrame, int]:
    before = len(frame)
    if bounds.get("min") is not None:
        frame = frame[pd.to_numeric(frame[column], errors="coerce") >= float(bounds["min"])]
    if bounds.get("max") is not None:
        frame = frame[pd.to_numeric(frame[column], errors="coerce") <= float(bounds["max"])]
    return frame, before - len(frame)


def run_inverse_query(
    *,
    feature_table: str | Path,
    query: str | Path,
    out: str | Path,
) -> Path:
    query_data = load_yaml(query)
    frame = _read_table(feature_table)
    missing_policy = query_data.get("missing_phase_policy", "zero")
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rejected: dict[str, int] = {}
    original_count = len(frame)

    constraints = query_data.get("recipe_constraints", {}) or {}
    for name, bounds in constraints.items():
        if "include" in (bounds or {}):
            before = len(frame)
            frame = frame[frame[name].isin(bounds["include"])] if name in frame else frame.iloc[0:0]
            rejected[f"constraint:{name}"] = before - len(frame)
        else:
            if name not in frame:
                rejected[f"constraint:{name}"] = len(frame)
                frame = frame.iloc[0:0]
            else:
                frame, count = _apply_range_filter(frame, name, bounds or {})
                rejected[f"constraint:{name}"] = count

    targets = query_data.get("targets", {}) or {}
    target_columns: list[str] = []
    for section in ["phase_amounts", "phase_volumes", "aqueous_species", "scalars"]:
        for name, bounds in (targets.get(section, {}) or {}).items():
            column = _feature_column(section, name)
            frame = _ensure_column(frame, column, missing_policy)
            target_columns.append(column)
            frame, count = _apply_range_filter(frame, column, bounds or {})
            rejected[f"target:{section}:{name}"] = count

    objectives = query_data.get("objectives", {}) or {}
    score = pd.Series(0.0, index=frame.index)
    for name, weight in (objectives.get("minimize", {}) or {}).items():
        column = _feature_column("phase_amounts", name) if f"phase_amount__{name}" in frame.columns else name
        column = "porosity" if name == "porosity" else column
        frame = _ensure_column(frame, column, missing_policy)
        score += float(weight) * pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    for name, weight in (objectives.get("maximize", {}) or {}).items():
        column = _feature_column("phase_amounts", name) if f"phase_amount__{name}" in frame.columns else name
        frame = _ensure_column(frame, column, missing_policy)
        score -= float(weight) * pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame = frame.copy()
    frame["score"] = score
    frame = frame.sort_values(["score", "recipe_id"], kind="mergesort")
    top_k = int(query_data.get("top_k", 50))
    candidates = frame.head(top_k).copy()
    candidates.insert(0, "rank", range(1, len(candidates) + 1))

    required = [
        "rank",
        "score",
        "recipe_id",
        "chem_hash",
        "template_name",
        "OPC",
        "slag",
        "fly_ash",
        "metakaolin",
        "silica_fume",
        "limestone",
        "gypsum",
        "water_g",
        "w_b",
        "water_mode",
        "age_days",
        "age_hours",
        "age_minutes",
        "age_label",
        "age_bin",
        "temperature_celsius",
        "porosity",
    ]
    for col in required + target_columns:
        if col not in candidates.columns:
            candidates[col] = 0.0 if col not in {"template_name", "water_mode", "age_label", "age_bin"} else ""
    output_cols = list(dict.fromkeys(required + target_columns + [c for c in candidates.columns if c.endswith("_D_eff") or c.startswith("alpha_")]))
    candidates[output_cols].to_csv(out_dir / "candidates.csv", index=False)
    (out_dir / "candidates.json").write_text(candidates[output_cols].to_json(orient="records", indent=2), encoding="utf-8")
    shutil.copy2(query, out_dir / "query_used.yaml")
    write_json(
        out_dir / "constraint_summary.json",
        {"input_rows": original_count, "candidate_rows_after_filters": int(len(frame)), "top_k_written": int(len(candidates))},
    )
    write_json(out_dir / "rejected_by_constraint_counts.json", rejected)
    return out_dir
