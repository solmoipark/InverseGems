from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .age_grids import detect_age_preset
from .database import InverseGemsDatabase, read_name_value_csv
from .utils import load_yaml, write_json


def _json(value: str | None) -> dict[str, Any]:
    return json.loads(value) if value else {}


def _stats(values: pd.Series) -> dict[str, float | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"min": None, "max": None, "mean": None}
    return {"min": float(numeric.min()), "max": float(numeric.max()), "mean": float(numeric.mean())}


def summarize_database(
    *,
    db: str | Path,
    out: str | Path,
    selection: str | Path | None = None,
    check_selected_outputs: bool = False,
) -> Path:
    database = InverseGemsDatabase(db)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    recipe_rows = database.recipe_rows()
    recipes = pd.DataFrame(recipe_rows)
    chem_rows = []
    with database.connect() as conn:
        chem_rows = [dict(row) for row in conn.execute("SELECT * FROM chemistry_runs").fetchall()]
    chemistry = pd.DataFrame(chem_rows)
    if recipes.empty:
        recipes = pd.DataFrame(columns=["recipe_id", "chem_hash", "porosity"])
    recipes.to_csv(out_dir / "recipe_coverage.csv", index=False)
    chemistry.to_csv(out_dir / "chemistry_coverage.csv", index=False)

    age_coverage = recipes[["recipe_id", "age_days", "age_hours", "age_minutes", "age_label", "age_bin"]].copy() if "age_days" in recipes else pd.DataFrame()
    age_coverage.to_csv(out_dir / "age_coverage.csv", index=False)
    age_counts = recipes["age_bin"].value_counts().rename_axis("age_bin").reset_index(name="count") if "age_bin" in recipes else pd.DataFrame(columns=["age_bin", "count"])
    age_counts.to_csv(out_dir / "age_bin_counts.csv", index=False)

    chemistry_status_by_hash = {}
    if not chemistry.empty and {"chem_hash", "status"}.issubset(chemistry.columns):
        chemistry_status_by_hash = dict(zip(chemistry["chem_hash"].astype(str), chemistry["status"].astype(str)))
    recipe_final_status = (
        recipes.get("chem_hash", pd.Series(dtype=str)).astype(str).map(chemistry_status_by_hash)
        if "chem_hash" in recipes
        else pd.Series(dtype=str)
    )
    failed_recipe_runs = recipes[recipe_final_status != "complete"].copy() if len(recipe_final_status) else pd.DataFrame()
    failed_recipe_runs.to_csv(out_dir / "failed_recipe_runs.csv", index=False)

    failed = chemistry[chemistry.get("status", pd.Series(dtype=str)) != "complete"] if not chemistry.empty and "status" in chemistry else pd.DataFrame()
    batch_status = Path(db) / "batch_status.csv"
    if batch_status.exists():
        batch = pd.read_csv(batch_status)
        failed_batch = batch[batch["status"] != "complete"] if "status" in batch else pd.DataFrame()
        if not failed_batch.empty and "chem_hash" in failed_batch:
            existing_hashes = set(failed.get("chem_hash", pd.Series(dtype=str)).dropna().astype(str))
            failed_batch = failed_batch[~failed_batch["chem_hash"].astype(str).isin(existing_hashes)]
        failed = pd.concat([failed, failed_batch], ignore_index=True, sort=False)
    failed.to_csv(out_dir / "failed_runs.csv", index=False)

    duplicates = (
        recipes.groupby("chem_hash")
        .agg(n_recipe_runs=("recipe_id", "count"), recipe_ids=("recipe_id", lambda x: ";".join(map(str, x))))
        .reset_index()
    )
    duplicates = duplicates[duplicates["n_recipe_runs"] > 1]
    duplicates.to_csv(out_dir / "duplicate_chem_hashes.csv", index=False)

    porosity = pd.to_numeric(recipes.get("porosity", pd.Series(dtype=float)), errors="coerce")
    porosity_outliers = recipes[(porosity < 0) | (porosity > 1)].copy()
    porosity_outliers.to_csv(out_dir / "porosity_outliers.csv", index=False)

    missing_rows: list[dict[str, Any]] = []
    ph_by_hash: dict[str, float | None] = {}
    selection_data = load_yaml(selection) if selection else {}
    if check_selected_outputs:
        for _, chem in chemistry.iterrows() if not chemistry.empty else []:
            raw_dir = Path(str(chem.get("xgems_run_dir", "")))
            available = {
                "phase_amounts": read_name_value_csv(raw_dir / "xgems_phase_amounts_raw.csv"),
                "phase_volumes": read_name_value_csv(raw_dir / "xgems_phase_volumes_raw.csv"),
                "aqueous_species": read_name_value_csv(raw_dir / "xgems_aqueous_species_raw.csv"),
            }
            scalars_path = raw_dir / "xgems_scalars_raw.json"
            scalars = json.loads(scalars_path.read_text(encoding="utf-8")) if scalars_path.exists() else {}
            if chem.get("chem_hash") is not None:
                ph_by_hash[str(chem.get("chem_hash"))] = scalars.get("pH")
            for section in ["phase_amounts", "phase_volumes", "aqueous_species"]:
                for name in (selection_data.get(section, {}) or {}).get("include", []) or []:
                    if name not in available[section]:
                        missing_rows.append({"chem_hash": chem.get("chem_hash"), "section": section, "name": name})
            for name in (selection_data.get("scalars", {}) or {}).get("include", []) or []:
                if name in {"porosity", "density"}:
                    continue
                if name not in scalars:
                    missing_rows.append({"chem_hash": chem.get("chem_hash"), "section": "scalars", "name": name})
    pd.DataFrame(missing_rows, columns=["chem_hash", "section", "name"]).to_csv(out_dir / "missing_selected_outputs.csv", index=False)

    recipe_jsons = [_json(value) for value in recipes.get("recipe_json", [])]
    binders = pd.DataFrame([r.get("binder_masses_g", {}) for r in recipe_jsons])
    template_counts = recipes.get("template_name", pd.Series(dtype=str)).fillna("").value_counts().to_dict()
    unique_ages = sorted(pd.to_numeric(recipes.get("age_days", pd.Series(dtype=float)), errors="coerce").dropna().unique().tolist())
    summary = {
        "number_of_recipe_runs": int(len(recipes)),
        "number_of_unique_chem_hashes": int(recipes["chem_hash"].nunique()) if "chem_hash" in recipes else 0,
        "cache_hit_ratio": float(len(recipes) - recipes["chem_hash"].nunique()) / float(len(recipes)) if len(recipes) else 0.0,
        "number_of_failed_recipe_runs": int(len(failed_recipe_runs)),
        "number_of_failed_chemistry_attempts": int(len(failed)),
        "number_of_failed_xgems_runs": int(len(failed)),
        "recipes_per_template": template_counts,
        "binder_component_stats": {col: _stats(binders.get(col, pd.Series(dtype=float))) for col in binders.columns},
        "w_b": _stats(recipes.get("w_b", pd.Series(dtype=float))),
        "water_g": _stats(recipes.get("water_g", pd.Series(dtype=float))),
        "xgems_w_b": _stats(recipes.get("xgems_w_b", pd.Series(dtype=float))),
        "xgems_water_g": _stats(recipes.get("xgems_water_g", pd.Series(dtype=float))),
        "solver_rescued_count": int(pd.to_numeric(recipes.get("solver_rescued", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
        "age_days": _stats(recipes.get("age_days", pd.Series(dtype=float))),
        "n_unique_ages": len(unique_ages),
        "age_preset_detected": detect_age_preset(unique_ages),
        "age_bin_counts": dict(zip(age_counts.get("age_bin", []), age_counts.get("count", []))),
        "temperature_celsius": _stats(recipes.get("temperature_celsius", pd.Series(dtype=float))),
        "porosity": _stats(recipes.get("porosity", pd.Series(dtype=float))),
        "porosity_outside_0_1_count": int(len(porosity_outliers)),
        "missing_pH_count": int(recipes["chem_hash"].astype(str).map(ph_by_hash).isna().sum()) if check_selected_outputs and "chem_hash" in recipes else None,
        "missing_selected_outputs_check_performed": bool(check_selected_outputs),
        "missing_selected_outputs_count": int(len(missing_rows)) if check_selected_outputs else None,
    }
    write_json(out_dir / "db_summary.json", summary)
    return out_dir


def _ph_for_hash(database: InverseGemsDatabase):
    cache: dict[str, float | None] = {}

    def inner(chem_hash: str) -> float | None:
        if chem_hash in cache:
            return cache[chem_hash]
        chem = database.get_chemistry_run(str(chem_hash)) or {}
        raw_dir = Path(str(chem.get("xgems_run_dir", "")))
        scalars_path = raw_dir / "xgems_scalars_raw.json"
        if not scalars_path.exists():
            cache[chem_hash] = None
        else:
            cache[chem_hash] = json.loads(scalars_path.read_text(encoding="utf-8")).get("pH")
        return cache[chem_hash]

    return inner
