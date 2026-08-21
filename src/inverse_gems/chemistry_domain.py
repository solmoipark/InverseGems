from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .utils import write_json


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_csv(path)
    return pd.read_csv(path)


def _input_columns_from_bundle_or_table(model_bundle: str | Path | None, reference: pd.DataFrame, candidates: pd.DataFrame) -> list[str]:
    if model_bundle is not None:
        bundle = joblib.load(model_bundle)
        inputs = [str(column) for column in bundle.get("inputs", [])] if isinstance(bundle, dict) else []
        if inputs:
            return inputs
    return sorted(column for column in reference.columns if column.startswith("x__") and column in candidates.columns)


def _scale_reference(reference: pd.DataFrame, candidates: pd.DataFrame, inputs: list[str]) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    ref = reference[inputs].apply(pd.to_numeric, errors="coerce")
    cand = candidates[inputs].apply(pd.to_numeric, errors="coerce")
    ref_min = ref.min(axis=0)
    ref_max = ref.max(axis=0)
    scale = (ref_max - ref_min).replace(0.0, 1.0)
    ref_scaled = ((ref - ref_min) / scale).fillna(0.0).to_numpy(dtype=float)
    cand_scaled = ((cand - ref_min) / scale).fillna(0.0).to_numpy(dtype=float)
    return ref_scaled, cand_scaled, ref_min, ref_max


def _nearest_distances(reference_scaled: np.ndarray, candidate_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances: list[float] = []
    indices: list[int] = []
    for row in candidate_scaled:
        diff = reference_scaled - row
        squared = np.einsum("ij,ij->i", diff, diff)
        nearest = int(np.argmin(squared))
        distances.append(float(np.sqrt(squared[nearest])))
        indices.append(nearest)
    return np.asarray(distances), np.asarray(indices, dtype=int)


def write_chemistry_domain_report(
    *,
    reference_model_table: str | Path,
    candidate_table: str | Path,
    out: str | Path,
    model_bundle: str | Path | None = None,
    nearest_distance_warn: float = 0.25,
) -> Path:
    """Compare chemistry candidate rows against the surrogate training/input domain."""

    reference_path = Path(reference_model_table)
    candidate_path = Path(candidate_table)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    reference = _read_table(reference_path)
    candidates = _read_table(candidate_path)
    inputs = _input_columns_from_bundle_or_table(model_bundle, reference, candidates)
    missing_reference = [column for column in inputs if column not in reference.columns]
    missing_candidates = [column for column in inputs if column not in candidates.columns]
    if missing_reference or missing_candidates:
        raise ValueError(
            "Domain report input columns are missing: "
            f"reference={missing_reference}, candidates={missing_candidates}"
        )

    ref_scaled, cand_scaled, ref_min, ref_max = _scale_reference(reference, candidates, inputs)
    distances, nearest_indices = _nearest_distances(ref_scaled, cand_scaled)
    cand_numeric = candidates[inputs].apply(pd.to_numeric, errors="coerce")
    below = cand_numeric.lt(ref_min, axis=1)
    above = cand_numeric.gt(ref_max, axis=1)
    outside = below | above

    rows: list[dict[str, Any]] = []
    reference_records = reference.reset_index(drop=True)
    for idx, candidate in candidates.reset_index(drop=True).iterrows():
        outside_columns = [column for column in inputs if bool(outside.iloc[idx][column])]
        nearest_row = reference_records.iloc[int(nearest_indices[idx])]
        rows.append(
            {
                "candidate_index": int(idx),
                "recipe_id": candidate.get("meta__recipe_id", candidate.get("recipe_id", "")),
                "chem_hash": candidate.get("meta__chem_hash", candidate.get("chem_hash", "")),
                "outside_input_range_count": int(len(outside_columns)),
                "outside_input_range_columns": ";".join(outside_columns),
                "nearest_scaled_distance": float(distances[idx]),
                "nearest_reference_recipe_id": nearest_row.get("meta__recipe_id", nearest_row.get("recipe_id", "")),
                "nearest_reference_chem_hash": nearest_row.get("meta__chem_hash", nearest_row.get("chem_hash", "")),
                "out_of_domain": bool(len(outside_columns) > 0 or distances[idx] > float(nearest_distance_warn)),
            }
        )
    report = pd.DataFrame(rows)
    report_path = out_dir / "chemistry_domain_report.csv"
    report.to_csv(report_path, index=False)

    input_summary = []
    for column in inputs:
        candidate_values = pd.to_numeric(candidates[column], errors="coerce")
        input_summary.append(
            {
                "input": column,
                "reference_min": float(ref_min[column]),
                "reference_max": float(ref_max[column]),
                "candidate_min": None if candidate_values.dropna().empty else float(candidate_values.min()),
                "candidate_max": None if candidate_values.dropna().empty else float(candidate_values.max()),
                "candidate_below_reference_count": int(below[column].sum()),
                "candidate_above_reference_count": int(above[column].sum()),
            }
        )
    pd.DataFrame(input_summary).to_csv(out_dir / "chemistry_domain_input_ranges.csv", index=False)
    summary = {
        "reference_model_table": str(reference_path),
        "candidate_table": str(candidate_path),
        "model_bundle": None if model_bundle is None else str(model_bundle),
        "reference_rows": int(len(reference)),
        "candidate_rows": int(len(candidates)),
        "input_columns": inputs,
        "nearest_distance_warn": float(nearest_distance_warn),
        "out_of_domain_count": int(report["out_of_domain"].sum()) if not report.empty else 0,
        "outside_input_range_count": int((report["outside_input_range_count"] > 0).sum()) if not report.empty else 0,
        "nearest_scaled_distance_max": None if report.empty else float(report["nearest_scaled_distance"].max()),
        "outputs": {
            "candidate_report": str(report_path),
            "input_ranges": str(out_dir / "chemistry_domain_input_ranges.csv"),
        },
    }
    write_json(out_dir / "chemistry_domain_summary.json", summary)
    return out_dir
