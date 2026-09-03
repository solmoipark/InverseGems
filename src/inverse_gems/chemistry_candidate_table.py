from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

from .age_grids import age_metadata
from .cached_forward import _recipe_projection_payload
from .chem_hash import compute_chem_hash
from .chemistry_vector import chemistry_vector_from_source_ledger
from .constants import ELEMENT_BASIS, MOLAR_MASS_G_MOL, OXIDE_EQUIVALENT_BASIS
from .materials import load_materials
from .recipe import parse_recipe
from .reaction_model import current_reaction_model_metadata
from .reaction_parameters import load_reaction_parameters
from .sampling import COMPONENTS, numeric_metadata_from_row, recipe_text_from_row
from .source_ledger import build_source_ledger
from .utils import config_path, short_hash, write_json
from .xgems_input_builder import build_xgems_input


def _read_recipe_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_table(frame: pd.DataFrame, path: Path, output_format: str | None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = (output_format or path.suffix.lower().lstrip(".") or "csv").lower()
    if fmt == "parquet":
        try:
            frame.to_parquet(path, index=False)
            return path
        except Exception:
            path = path.with_suffix(".csv")
    frame.to_csv(path, index=False)
    return path


def _meta_column(name: str) -> str:
    return f"meta__{name}"


def _input_column(name: str) -> str:
    return f"x__{name}"


def _candidate_prepared_id(payload: dict[str, Any]) -> str:
    return f"prepared_candidate_{short_hash(payload, 20)}"


def _row_temperature(row: dict[str, Any], default: float) -> float:
    value = row.get("temperature_celsius")
    return float(default if value in (None, "") else value)


def chemistry_candidate_row_from_recipe_row(
    row: dict[str, Any],
    *,
    dat_lst: str | Path | None = None,
    run_mode: str = "reacted_only",
    normalize: bool = True,
    allow_non_100: bool = False,
    temperature_celsius: float = 20.0,
    pressure: float | None = None,
    xgems_water_mode: str = "initial",
    xgems_water_factor: float = 1.0,
    xgems_water_g: float | None = None,
    xgems_water_w_b: float | None = None,
    reaction_model_id: str | None = None,
    reaction_model_config: str | Path | None = None,
    materials_config: str | Path | None = None,
) -> dict[str, Any]:
    """Project a recipe candidate to the same chemistry feature space used by xGEMS.

    This intentionally stops before solver execution. The result can be passed to
    a chemistry-level surrogate and later validated with the normal xGEMS runner.
    """

    materials = load_materials(materials_config)
    recipe_text = recipe_text_from_row(row)
    recipe = parse_recipe(recipe_text, materials=materials, normalize=normalize, allow_non_100=allow_non_100)
    recipe.metadata.update(age_metadata(recipe.age_days).to_dict())
    recipe.metadata.update(numeric_metadata_from_row(row))
    if row.get("recipe_id") not in (None, ""):
        recipe.metadata["recipe_id"] = str(row["recipe_id"])
    temperature = _row_temperature(row, temperature_celsius)

    reaction_parameters = load_reaction_parameters(reaction_model_config, reaction_model_id=reaction_model_id)
    reaction_model = current_reaction_model_metadata(
        reaction_model_id=reaction_parameters.id,
        reaction_model_config=reaction_model_config,
        reaction_parameters=reaction_parameters,
    )
    xgems_input = build_xgems_input(
        recipe,
        materials=materials,
        run_mode=run_mode,
        temperature_celsius=temperature,
        reaction_parameters=reaction_parameters,
        xgems_water_mode=xgems_water_mode,
        xgems_water_factor=xgems_water_factor,
        xgems_water_g=xgems_water_g,
        xgems_water_w_b=xgems_water_w_b,
    )
    recipe_id = str(recipe.metadata.get("recipe_id") or f"candidate_{short_hash(recipe.to_dict(), 12)}")
    ledger_rows = build_source_ledger(
        recipe_id=recipe_id,
        chem_hash="",
        recipe=recipe,
        xgems_input=xgems_input,
        materials=materials,
    )
    canonical_vector = chemistry_vector_from_source_ledger(ledger_rows)
    water_mol = xgems_input.equilibrium_water_g / MOLAR_MASS_G_MOL["H2O"]
    hash_result = compute_chem_hash(
        canonical_vector,
        water_mol=water_mol,
        temperature_celsius=temperature,
        pressure=pressure,
        dat_lst_path=dat_lst,
        species_map_path=config_path("species_map.yaml"),
        xgems_run_mode=run_mode,
    )
    prepared_payload = {
        "recipe_projection": _recipe_projection_payload(recipe),
        "reaction_model_signature": reaction_model["reaction_model_signature"],
        "chem_hash": hash_result.chem_hash,
        "dat_lst_hash": hash_result.dat_lst_hash,
        "species_map_hash": hash_result.species_map_hash,
        "run_mode": run_mode,
        "temperature_celsius": temperature,
        "pressure": pressure,
        "xgems_water_policy": xgems_input.water_policy,
    }
    out: dict[str, Any] = {
        _meta_column("recipe_id"): recipe_id,
        _meta_column("chem_hash"): hash_result.chem_hash,
        _meta_column("prepared_id"): _candidate_prepared_id(prepared_payload),
        _meta_column("reaction_model_id"): reaction_model["reaction_model_id"],
        _meta_column("reaction_model_signature"): reaction_model["reaction_model_signature"],
        _meta_column("chem_hash_version"): hash_result.chem_hash_version,
        _meta_column("dat_lst_hash"): hash_result.dat_lst_hash,
        _meta_column("species_map_hash"): hash_result.species_map_hash,
        _meta_column("template_name"): recipe.metadata.get("template_name", ""),
        _meta_column("material_system"): recipe.metadata.get("material_system", ""),
        _meta_column("target_profile"): recipe.metadata.get("target_profile", ""),
        _meta_column("target_profile_description"): recipe.metadata.get("target_profile_description", ""),
        _meta_column("age_label"): recipe.metadata.get("age_label", ""),
        _meta_column("age_bin"): recipe.metadata.get("age_bin", ""),
        _meta_column("xgems_run_mode"): run_mode,
        _input_column("w_b"): recipe.w_b,
        _input_column("water_g"): recipe.water_g,
        _input_column("age_days"): recipe.age_days,
        _input_column("temperature_celsius"): temperature,
        _input_column("xgems_water_g"): xgems_input.equilibrium_water_g,
        _input_column("xgems_w_b"): xgems_input.equilibrium_w_b,
        _input_column("water_mol"): water_mol,
    }
    for component in COMPONENTS:
        out[_input_column(component)] = float(recipe.binder_masses_g.get(component, 0.0))
    for element in ELEMENT_BASIS:
        out[_input_column(f"chem_element_mol_{element}")] = float(canonical_vector.element_mol.get(element, 0.0))
    for oxide in OXIDE_EQUIVALENT_BASIS:
        out[_input_column(f"chem_oxide_equiv_mol_{oxide}")] = float(
            canonical_vector.oxide_equivalent_mol.get(oxide, 0.0)
        )
    return out


def build_chemistry_candidate_table(
    *,
    recipes_csv: str | Path,
    out: str | Path,
    dat_lst: str | Path | None = None,
    run_mode: str = "reacted_only",
    normalize: bool = True,
    allow_non_100: bool = False,
    temperature_celsius: float = 20.0,
    pressure: float | None = None,
    xgems_water_mode: str = "initial",
    xgems_water_factor: float = 1.0,
    xgems_water_g: float | None = None,
    xgems_water_w_b: float | None = None,
    reaction_model_id: str | None = None,
    reaction_model_config: str | Path | None = None,
    materials_config: str | Path | None = None,
    output_format: str | None = None,
) -> Path:
    rows = [
        chemistry_candidate_row_from_recipe_row(
            row,
            dat_lst=dat_lst,
            run_mode=run_mode,
            normalize=normalize,
            allow_non_100=allow_non_100,
            temperature_celsius=temperature_celsius,
            pressure=pressure,
            xgems_water_mode=xgems_water_mode,
            xgems_water_factor=xgems_water_factor,
            xgems_water_g=xgems_water_g,
            xgems_water_w_b=xgems_water_w_b,
            reaction_model_id=reaction_model_id,
            reaction_model_config=reaction_model_config,
            materials_config=materials_config,
        )
        for row in _read_recipe_rows(recipes_csv)
    ]
    frame = pd.DataFrame(rows)
    written = _write_table(frame, Path(out), output_format)
    write_json(
        written.with_suffix(written.suffix + ".schema.json"),
        {
            "recipes_csv": str(recipes_csv),
            "out": str(written),
            "row_count": int(len(frame)),
            "input_space": "reactive_chemistry",
            "run_mode": run_mode,
            "dat_lst": None if dat_lst is None else str(dat_lst),
            "reaction_model_id": reaction_model_id,
            "reaction_model_config": None if reaction_model_config is None else str(reaction_model_config),
            "metadata_columns": [column for column in frame.columns if column.startswith("meta__")],
            "input_columns": [column for column in frame.columns if column.startswith("x__")],
        },
    )
    return written
