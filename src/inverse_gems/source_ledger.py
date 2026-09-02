from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .chemistry_vector import chemistry_vector_from_source_ledger
from .constants import ELEMENT_BASIS, FORMULAS, MOLAR_MASS_G_MOL, OXIDE_EQUIVALENTS
from .materials import SCM_NAMES, Material, load_materials
from .recipe import Recipe
from .xgems_input_builder import XGEMSInput


LEDGER_FIELDS = [
    "recipe_id",
    "chem_hash",
    "source_material",
    "source_phase_or_oxide",
    "source_mass_g_initial",
    "reaction_degree",
    "reacted_mass_g",
    "unreacted_mass_g",
    "component",
    "component_mol",
    "oxide_equivalent",
    "oxide_equivalent_mol",
]


def _element_oxide_pair(component_name: str, element: str) -> tuple[str, float]:
    oxide_map = OXIDE_EQUIVALENTS.get(component_name, {})
    if component_name in {"CaO", "C3S", "C2S", "C3A", "C4AF", "CaCO3", "Cal", "gypsum", "Gp"} and element == "Ca":
        return "CaO", oxide_map.get("CaO", 0.0)
    if component_name in {"SiO2", "C3S", "C2S"} and element == "Si":
        return "SiO2", oxide_map.get("SiO2", 0.0)
    if component_name in {"Al2O3", "C3A", "C4AF"} and element == "Al":
        return "Al2O3", oxide_map.get("Al2O3", 0.0)
    if component_name in {"Fe2O3", "C4AF"} and element == "Fe":
        return "Fe2O3", oxide_map.get("Fe2O3", 0.0)
    if component_name == "MgO" and element == "Mg":
        return "MgO", oxide_map.get("MgO", 0.0)
    if component_name in {"SO3", "gypsum", "Gp"} and element == "S":
        return "SO3", oxide_map.get("SO3", 0.0)
    if component_name == "Na2O" and element == "Na":
        return "Na2O", oxide_map.get("Na2O", 0.0)
    if component_name == "K2O" and element == "K":
        return "K2O", oxide_map.get("K2O", 0.0)
    if component_name in {"CO2", "CaCO3", "Cal"} and element == "C":
        return "CO2", oxide_map.get("CO2", 0.0)
    if component_name in {"H2O", "H2O@", "gypsum", "Gp"} and element == "H":
        return "H2O", oxide_map.get("H2O", 0.0)
    return "", 0.0


def rows_for_component_mass(
    *,
    recipe_id: str,
    chem_hash: str,
    source_material: str,
    source_phase_or_oxide: str,
    source_mass_g_initial: float,
    reaction_degree: float,
    reacted_mass_g: float,
    unreacted_mass_g: float,
    component_name: str,
) -> list[dict[str, Any]]:
    if component_name not in FORMULAS:
        raise ValueError(f"No formula is defined for component '{component_name}'.")
    formula_mol = float(reacted_mass_g) / MOLAR_MASS_G_MOL[component_name] if reacted_mass_g else 0.0
    rows: list[dict[str, Any]] = []
    for element in ELEMENT_BASIS:
        coefficient = FORMULAS[component_name].get(element, 0.0)
        if coefficient == 0:
            continue
        oxide, oxide_coeff = _element_oxide_pair(component_name, element)
        rows.append(
            {
                "recipe_id": recipe_id,
                "chem_hash": chem_hash,
                "source_material": source_material,
                "source_phase_or_oxide": source_phase_or_oxide,
                "source_mass_g_initial": float(source_mass_g_initial),
                "reaction_degree": float(reaction_degree),
                "reacted_mass_g": float(reacted_mass_g),
                "unreacted_mass_g": float(unreacted_mass_g),
                "component": element,
                "component_mol": formula_mol * float(coefficient),
                "oxide_equivalent": oxide,
                "oxide_equivalent_mol": formula_mol * float(oxide_coeff),
            }
        )
    return rows


def build_source_ledger(
    *,
    recipe_id: str,
    chem_hash: str,
    recipe: Recipe,
    xgems_input: XGEMSInput,
    materials: dict[str, Material] | None = None,
) -> list[dict[str, Any]]:
    materials = materials or load_materials()
    rows: list[dict[str, Any]] = []
    reaction = xgems_input.reaction_degrees
    opc_phase_pct = reaction.get("opc_phase_mass_percent", {})
    opc_alpha = reaction.get("opc", {})
    opc_mass = recipe.binder_masses_g.get("OPC", 0.0)
    if opc_mass > 0:
        for phase, pct in opc_phase_pct.items():
            initial = opc_mass * float(pct) / 100.0
            alpha = float(opc_alpha.get(phase, 0.0))
            rows.extend(
                rows_for_component_mass(
                    recipe_id=recipe_id,
                    chem_hash=chem_hash,
                    source_material="OPC",
                    source_phase_or_oxide=phase,
                    source_mass_g_initial=initial,
                    reaction_degree=alpha,
                    reacted_mass_g=initial * alpha,
                    unreacted_mass_g=initial * (1.0 - alpha),
                    component_name=phase,
                )
            )

    # OPC minor oxides (SO3 as CaSO4 with its companion CaO, MgO, Na2O, K2O) — see
    # xgems_input_builder._add_opc_minor_oxides; dropped oxides carry no component.
    minor = reaction.get("opc_minor_oxides") or {}
    if opc_mass > 0 and minor.get("enabled"):
        for oxide, entry in (minor.get("per_oxide") or {}).items():
            if entry.get("dropped") or entry.get("species") is None:
                continue
            initial = float(entry["mass_g"])
            alpha = float(entry["degree"])
            rows.extend(
                rows_for_component_mass(
                    recipe_id=recipe_id,
                    chem_hash=chem_hash,
                    source_material="OPC",
                    source_phase_or_oxide=oxide,
                    source_mass_g_initial=initial,
                    reaction_degree=alpha,
                    reacted_mass_g=initial * alpha,
                    unreacted_mass_g=initial * (1.0 - alpha),
                    component_name=oxide,
                )
            )
            cao_g = entry.get("companion_CaO_g")
            if cao_g:
                rows.extend(
                    rows_for_component_mass(
                        recipe_id=recipe_id,
                        chem_hash=chem_hash,
                        source_material="OPC",
                        source_phase_or_oxide="CaO(CaSO4)",
                        source_mass_g_initial=float(cao_g),
                        reaction_degree=alpha,
                        reacted_mass_g=float(cao_g) * alpha,
                        unreacted_mass_g=float(cao_g) * (1.0 - alpha),
                        component_name="CaO",
                    )
                )

    scm_alpha = reaction.get("scm", {})
    for material_name, mass in recipe.binder_masses_g.items():
        if material_name not in SCM_NAMES or mass <= 0:
            continue
        alpha = float(scm_alpha.get(material_name, 0.0))
        oxides = materials[material_name].oxide_mass_percent or {}
        for oxide, pct in oxides.items():
            initial = mass * float(pct) / 100.0
            rows.extend(
                rows_for_component_mass(
                    recipe_id=recipe_id,
                    chem_hash=chem_hash,
                    source_material=material_name,
                    source_phase_or_oxide=oxide,
                    source_mass_g_initial=initial,
                    reaction_degree=alpha,
                    reacted_mass_g=initial * alpha,
                    unreacted_mass_g=initial * (1.0 - alpha),
                    component_name=oxide,
                )
            )

    if recipe.binder_masses_g.get("limestone", 0.0) > 0:
        mass = recipe.binder_masses_g["limestone"]
        rows.extend(
            rows_for_component_mass(
                recipe_id=recipe_id,
                chem_hash=chem_hash,
                source_material="limestone",
                source_phase_or_oxide="CaCO3",
                source_mass_g_initial=mass,
                reaction_degree=1.0,
                reacted_mass_g=mass,
                unreacted_mass_g=0.0,
                component_name="CaCO3",
            )
        )
    if recipe.binder_masses_g.get("gypsum", 0.0) > 0:
        mass = recipe.binder_masses_g["gypsum"]
        rows.extend(
            rows_for_component_mass(
                recipe_id=recipe_id,
                chem_hash=chem_hash,
                source_material="gypsum",
                source_phase_or_oxide="Gp",
                source_mass_g_initial=mass,
                reaction_degree=1.0,
                reacted_mass_g=mass,
                unreacted_mass_g=0.0,
                component_name="Gp",
            )
        )
    equilibrium_water_g = float(getattr(xgems_input, "equilibrium_water_g", recipe.water_g))
    if equilibrium_water_g > 0:
        rows.extend(
            rows_for_component_mass(
                recipe_id=recipe_id,
                chem_hash=chem_hash,
                source_material="water",
                source_phase_or_oxide="H2O",
                source_mass_g_initial=equilibrium_water_g,
                reaction_degree=1.0,
                reacted_mass_g=equilibrium_water_g,
                unreacted_mass_g=0.0,
                component_name="H2O",
            )
        )
    return rows


def update_ledger_hash(rows: list[dict[str, Any]], chem_hash: str) -> list[dict[str, Any]]:
    for row in rows:
        row["chem_hash"] = chem_hash
    return rows


def write_source_ledger_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in LEDGER_FIELDS})


def ledger_component_vector(rows: list[dict[str, Any]]) -> dict[str, float]:
    return chemistry_vector_from_source_ledger(rows).element_mol
