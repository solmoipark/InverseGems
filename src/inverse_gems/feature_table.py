from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .database import InverseGemsDatabase, read_name_value_csv
from .materials import BINDER_COMPONENTS
from .utils import config_path, load_yaml, write_json

MAX_WARNING_MESSAGES = 1000


def _json_loads(value: str | None) -> dict[str, Any]:
    return json.loads(value) if value else {}


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(result):
        return None
    return result


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"", "0", "0.0", "false", "no", "none", "nan"}:
        return False
    return True


def _group_items(group_definition: Any) -> list[str]:
    if isinstance(group_definition, dict):
        group_definition = group_definition.get("include", [])
    if isinstance(group_definition, str):
        return [group_definition]
    return list(group_definition or [])


def _append_warning(warnings: list[str], message: str) -> None:
    if len(warnings) < MAX_WARNING_MESSAGES:
        warnings.append(message)
    elif len(warnings) == MAX_WARNING_MESSAGES:
        warnings.append(f"Warning list truncated after {MAX_WARNING_MESSAGES} messages.")


def _add_grouped_values(
    *,
    row: dict[str, Any],
    warnings: list[str],
    recipe_id: str,
    values: dict[str, float],
    groups: dict[str, Any],
    column_prefix: str,
    value_label: str,
) -> None:
    for group_name, definition in (groups or {}).items():
        total = 0.0
        for raw_name in _group_items(definition):
            if raw_name not in values:
                _append_warning(
                    warnings,
                    f"Missing grouped {value_label} '{raw_name}' for group '{group_name}' "
                    f"and recipe {recipe_id}; filled with 0.0.",
                )
            total += float(values.get(raw_name, 0.0) or 0.0)
        row[f"{column_prefix}__{group_name}"] = total


def inspect_available_outputs(db: str | Path, chem_hash: str) -> dict[str, Any]:
    database = InverseGemsDatabase(db)
    chem = database.get_chemistry_run(chem_hash)
    if not chem:
        raise ValueError(f"Unknown chem_hash: {chem_hash}")
    raw_dir = Path(chem["xgems_run_dir"])
    scalars_path = raw_dir / "xgems_scalars_raw.json"
    return {
        "phase_amount_names": sorted(read_name_value_csv(raw_dir / "xgems_phase_amounts_raw.csv")),
        "phase_volume_names": sorted(read_name_value_csv(raw_dir / "xgems_phase_volumes_raw.csv")),
        "aqueous_species_names": sorted(read_name_value_csv(raw_dir / "xgems_aqueous_species_raw.csv")),
        "scalar_keys": sorted(json.loads(scalars_path.read_text(encoding="utf-8")) if scalars_path.exists() else {}),
    }


def build_feature_table(
    *,
    db: str | Path,
    selection: str | Path | None = None,
    out: str | Path,
    output_format: str | None = None,
    recipe_id_prefix: str | None = None,
    material_system: str | None = None,
    age_days: float | None = None,
    age_tolerance: float = 1.0e-12,
) -> Path:
    database = InverseGemsDatabase(db)
    selection_data = load_yaml(selection or config_path("output_selection.yaml"))
    recipe_rows = database.recipe_rows()
    if recipe_id_prefix:
        recipe_rows = [row for row in recipe_rows if str(row.get("recipe_id") or "").startswith(str(recipe_id_prefix))]
    if material_system:
        recipe_rows = [row for row in recipe_rows if str(row.get("material_system") or "") == str(material_system)]
    if age_days is not None:
        target_age = float(age_days)
        tolerance = float(age_tolerance)
        recipe_rows = [
            row
            for row in recipe_rows
            if row.get("age_days") is not None and abs(float(row.get("age_days")) - target_age) <= tolerance
        ]
    with database.connect() as conn:
        chemistry_by_hash = {
            str(row["chem_hash"]): dict(row)
            for row in conn.execute("SELECT * FROM chemistry_runs").fetchall()
        }
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for recipe_row in recipe_rows:
        recipe = _json_loads(recipe_row.get("recipe_json"))
        reaction = _json_loads(recipe_row.get("reaction_degrees_json"))
        chem = chemistry_by_hash.get(str(recipe_row["chem_hash"]), {})
        raw_dir = Path(chem.get("xgems_run_dir") or "")
        phase_amounts = read_name_value_csv(raw_dir / "xgems_phase_amounts_raw.csv")
        phase_volumes = read_name_value_csv(raw_dir / "xgems_phase_volumes_raw.csv")
        phase_volumes_reconstructed = read_name_value_csv(raw_dir / "xgems_phase_volumes_reconstructed.csv")
        phase_volumes_for_features = dict(phase_volumes)
        phase_volumes_for_features.update(
            {name: value for name, value in phase_volumes_reconstructed.items() if abs(float(value or 0.0)) > 0.0}
        )
        aqueous = read_name_value_csv(raw_dir / "xgems_aqueous_species_raw.csv")
        scalars_path = raw_dir / "xgems_scalars_raw.json"
        scalars = json.loads(scalars_path.read_text(encoding="utf-8")) if scalars_path.exists() else {}
        canonical = _json_loads(chem.get("canonical_vector_json"))
        oxide = _json_loads(chem.get("oxide_equivalent_vector_json"))
        binder = recipe.get("binder_masses_g", {})
        initial_water_g = _as_float(recipe_row.get("water_g", recipe.get("water_g")))
        xgems_water_g = _as_float(recipe_row.get("xgems_water_g") or (reaction.get("xgems_water") or {}).get("equilibrium_water_g"))
        xgems_w_b = _as_float(recipe_row.get("xgems_w_b") or (reaction.get("xgems_water") or {}).get("equilibrium_w_b"))
        solver_rescued = _as_bool(recipe_row.get("solver_rescued"))
        water_delta = None
        water_relative_delta = None
        water_matches_recipe: bool | None = None
        if initial_water_g is not None and xgems_water_g is not None:
            water_delta = xgems_water_g - initial_water_g
            water_relative_delta = None if abs(initial_water_g) <= 0.0 else water_delta / initial_water_g
            water_matches_recipe = abs(water_delta) <= 1.0e-9
        pH_unreliable_reasons: list[str] = []
        if solver_rescued:
            pH_unreliable_reasons.append("solver_rescued")
        if water_matches_recipe is False:
            pH_unreliable_reasons.append("xgems_water_changed")
        if water_matches_recipe is None:
            pH_unreliable_reasons.append("xgems_water_unknown")
        pH_water_reliable = len(pH_unreliable_reasons) == 0
        row: dict[str, Any] = {
            "recipe_id": recipe_row["recipe_id"],
            "chem_hash": recipe_row["chem_hash"],
            "prepared_id": recipe_row.get("prepared_id") or recipe.get("metadata", {}).get("prepared_id"),
            "reaction_model_id": recipe_row.get("reaction_model_id") or recipe.get("metadata", {}).get("reaction_model_id"),
            "reaction_model_signature": recipe_row.get("reaction_model_signature")
            or recipe.get("metadata", {}).get("reaction_model_signature"),
            "chemistry_status": chem.get("status"),
            "xgems_run_dir": chem.get("xgems_run_dir"),
            "template_name": recipe_row.get("template_name") or recipe.get("metadata", {}).get("template_name", ""),
            "material_system": recipe_row.get("material_system") or recipe.get("metadata", {}).get("material_system", ""),
            "target_profile": recipe_row.get("target_profile") or recipe.get("metadata", {}).get("target_profile", ""),
            "target_profile_description": recipe_row.get("target_profile_description")
            or recipe.get("metadata", {}).get("target_profile_description", ""),
            "water_g": initial_water_g,
            "w_b": recipe_row.get("w_b", recipe.get("w_b")),
            "water_mode": recipe_row.get("water_mode", recipe.get("water_mode")),
            "xgems_water_g": xgems_water_g,
            "xgems_w_b": xgems_w_b,
            "xgems_water_mode": recipe_row.get("xgems_water_mode") or (reaction.get("xgems_water") or {}).get("mode"),
            "xgems_water_delta_g": water_delta,
            "xgems_water_relative_delta": water_relative_delta,
            "xgems_water_matches_recipe": water_matches_recipe,
            "pH_water_reliable": pH_water_reliable,
            "pH_unreliable_reason": ";".join(pH_unreliable_reasons),
            "solver_rescued": solver_rescued,
            "xgems_retry_count": recipe_row.get("xgems_retry_count") or 0,
            "primary_chem_hash": recipe_row.get("primary_chem_hash"),
            "primary_solver_status": recipe_row.get("primary_solver_status"),
            "age_days": recipe_row.get("age_days", recipe.get("age_days")),
            "age_hours": recipe_row.get("age_hours", recipe.get("metadata", {}).get("age_hours")),
            "age_minutes": recipe_row.get("age_minutes", recipe.get("metadata", {}).get("age_minutes")),
            "age_label": recipe_row.get("age_label", recipe.get("metadata", {}).get("age_label")),
            "age_bin": recipe_row.get("age_bin", recipe.get("metadata", {}).get("age_bin")),
            "temperature_celsius": chem.get("temperature_celsius"),
            "initial_volume_cm3": recipe_row.get("initial_volume_cm3"),
            "final_solid_volume_cm3": recipe_row.get("final_solid_volume_cm3"),
            "porosity": recipe_row.get("porosity"),
        }
        for component in sorted(BINDER_COMPONENTS):
            row[component] = float(binder.get(component, 0.0))
        for phase in ("C3S", "C2S", "C3A", "C4AF"):
            row[f"opc_phase_fraction_{phase}"] = float((reaction.get("opc_phase_mass_percent") or {}).get(phase, 0.0)) / 100.0
            row[f"alpha_{phase}"] = float((reaction.get("opc") or {}).get(phase, 0.0))
        for scm, alpha in (reaction.get("scm") or {}).items():
            row[f"{scm}_alpha_eff"] = alpha
            row[f"{scm}_D_eff"] = ((reaction.get("availability_modifier") or {}).get("per_scm") or {}).get(scm, {}).get("D_eff")
            row[f"{scm}_alpha_intrinsic"] = alpha
        for key, value in canonical.items():
            row[f"chem_element_mol_{key}"] = value
        for key, value in oxide.items():
            row[f"chem_oxide_equiv_mol_{key}"] = value
        for name in selection_data.get("phase_amounts", {}).get("include", []) or []:
            if name not in phase_amounts:
                _append_warning(warnings, f"Missing selected phase amount '{name}' for recipe {recipe_row['recipe_id']}; filled with 0.0.")
            row[f"phase_amount__{name}"] = phase_amounts.get(name, 0.0)
        for name in selection_data.get("phase_volumes", {}).get("include", []) or []:
            if name not in phase_volumes_for_features:
                _append_warning(warnings, f"Missing selected phase volume '{name}' for recipe {recipe_row['recipe_id']}; filled with 0.0.")
            row[f"phase_volume__{name}"] = phase_volumes_for_features.get(name, 0.0)
        for name in selection_data.get("aqueous_species", {}).get("include", []) or []:
            if name not in aqueous:
                _append_warning(warnings, f"Missing selected aqueous species '{name}' for recipe {recipe_row['recipe_id']}; filled with 0.0.")
            row[f"aqueous__{name}"] = aqueous.get(name, 0.0)
        phase_groups = selection_data.get("phase_groups", {}) or {}
        _add_grouped_values(
            row=row,
            warnings=warnings,
            recipe_id=recipe_row["recipe_id"],
            values=phase_amounts,
            groups=phase_groups.get("amounts", {}) or phase_groups.get("phase_amounts", {}),
            column_prefix="phase_amount_group",
            value_label="phase amount",
        )
        _add_grouped_values(
            row=row,
            warnings=warnings,
            recipe_id=recipe_row["recipe_id"],
            values=phase_volumes_for_features,
            groups=phase_groups.get("volumes", {}) or phase_groups.get("phase_volumes", {}),
            column_prefix="phase_volume_group",
            value_label="phase volume",
        )
        for name in selection_data.get("scalars", {}).get("include", []) or []:
            if name == "porosity":
                row["scalar__porosity"] = recipe_row.get("porosity")
            elif name == "density":
                mass = scalars.get("system_mass")
                volume = scalars.get("system_volume")
                row["scalar__density"] = None if not mass or not volume else mass / volume
            else:
                if name not in scalars:
                    _append_warning(warnings, f"Missing selected scalar '{name}' for recipe {recipe_row['recipe_id']}; filled with 0.0.")
                row[f"scalar__{name}"] = scalars.get(name, 0.0)
        rows.append(row)
    frame = pd.DataFrame(rows)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = (output_format or out_path.suffix.lower().lstrip(".") or "csv").lower()
    if fmt == "parquet":
        try:
            frame.to_parquet(out_path, index=False)
        except Exception:
            frame.to_csv(out_path, index=False)
    else:
        frame.to_csv(out_path, index=False)
    write_json(out_path.with_suffix(out_path.suffix + ".warnings.json"), {"warnings": warnings})
    return out_path
