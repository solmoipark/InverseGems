from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .availability_modifier import C3S_C2SAvailabilityModifier
from .bogue import bogue_phases
from .materials import SCM_NAMES, Material, load_materials
from .pk_model import parrot_killoh
from .recipe import Recipe
from .reaction_parameters import ReactionParameterSet, load_reaction_parameters
from .scm_reaction import parameters_to_dict, scm_alpha
from .utils import add_amount, config_path, load_yaml


@dataclass
class XGEMSInput:
    species_amounts_kg: dict[str, float]
    lower_bounds_kg: dict[str, float]
    reaction_degrees: dict[str, Any]
    unreacted_masses_g: dict[str, float]
    warnings: list[str]
    mode: str
    equilibrium_water_g: float
    equilibrium_w_b: float
    water_policy: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "species_amounts_kg": dict(self.species_amounts_kg),
            "lower_bounds_kg": dict(self.lower_bounds_kg),
            "reaction_degrees": self.reaction_degrees,
            "unreacted_masses_g": dict(self.unreacted_masses_g),
            "warnings": list(self.warnings),
            "mode": self.mode,
            "equilibrium_water_g": self.equilibrium_water_g,
            "equilibrium_w_b": self.equilibrium_w_b,
            "water_policy": dict(self.water_policy),
        }


def load_species_map(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(path or config_path("species_map.yaml"))


def _species(species_map: dict[str, Any], section: str, key: str, default: str) -> str:
    return str((species_map.get(section) or {}).get(key, default))


def _water_to_binder_for_pk(recipe: Recipe) -> float:
    if recipe.w_b is not None:
        return float(recipe.w_b)
    total_binder = sum(recipe.binder_masses_g.values())
    return float(recipe.water_g / total_binder)


def resolve_xgems_water(
    recipe: Recipe,
    *,
    mode: str = "initial",
    factor: float = 1.0,
    water_g: float | None = None,
    water_w_b: float | None = None,
) -> dict[str, Any]:
    total_binder = sum(recipe.binder_masses_g.values())
    if total_binder <= 0:
        raise ValueError("Recipe must contain positive binder mass.")
    initial_water_g = float(recipe.water_g)
    mode = str(mode or "initial")
    factor = float(factor)
    if mode == "initial":
        equilibrium_water_g = initial_water_g
    elif mode == "fraction_of_initial":
        equilibrium_water_g = initial_water_g * factor
    elif mode == "cap_w_b":
        if water_w_b is None:
            raise ValueError("xGEMS water mode 'cap_w_b' requires xgems_water_w_b.")
        equilibrium_water_g = min(initial_water_g, total_binder * float(water_w_b))
    elif mode == "fixed_w_b":
        if water_w_b is None:
            raise ValueError("xGEMS water mode 'fixed_w_b' requires xgems_water_w_b.")
        equilibrium_water_g = total_binder * float(water_w_b)
    elif mode == "direct_water_g":
        if water_g is None:
            raise ValueError("xGEMS water mode 'direct_water_g' requires xgems_water_g.")
        equilibrium_water_g = float(water_g)
    else:
        raise ValueError(
            "xgems_water_mode must be one of: initial, fraction_of_initial, cap_w_b, fixed_w_b, direct_water_g."
        )
    if equilibrium_water_g < 0:
        raise ValueError("Resolved xGEMS equilibrium water must be non-negative.")
    return {
        "mode": mode,
        "initial_water_g": initial_water_g,
        "initial_w_b": initial_water_g / total_binder,
        "equilibrium_water_g": equilibrium_water_g,
        "equilibrium_w_b": equilibrium_water_g / total_binder,
        "factor": None if initial_water_g == 0 else equilibrium_water_g / initial_water_g,
        "requested_factor": factor,
        "requested_water_g": water_g,
        "requested_w_b": water_w_b,
    }


BOGUE_OXIDES = {"CaO", "SiO2", "Al2O3", "Fe2O3"}
_M_CAO = 56.0774
_M_SO3 = 80.0632


def _add_opc_minor_oxides(
    species_amounts: dict[str, float],
    lower_bounds: dict[str, float],
    *,
    species_map: dict[str, Any],
    opc_mass_g: float,
    oxide_mass_percent: dict[str, float],
    opc_phase_mass_percent: dict[str, float],
    pk_degrees: dict[str, float],
    config: dict[str, Any],
    run_mode: str,
    warnings: list[str],
) -> dict[str, Any]:
    """Add the OPC oxides that Bogue does not represent (SO3 as CaSO4, MgO, Na2O, K2O, …).

    Returns metadata {enabled, per_oxide, reacted_g, unreacted_g}. Oxides without an entry in
    ``species_map['oxides']`` are reported as ``dropped`` and a warning is added.
    """
    total_pct = sum(float(v) for v in opc_phase_mass_percent.values())
    clinker_mean = (
        sum(float(opc_phase_mass_percent[p]) * float(pk_degrees.get(p, 0.0)) for p in opc_phase_mass_percent) / total_pct
        if total_pct > 0
        else 0.0
    )
    degrees_cfg = dict(config.get("degrees") or {})
    sulfate_as_caso4 = bool(config.get("sulfate_as_calcium_sulfate", True))
    oxide_species = species_map.get("oxides") or {}
    per_oxide: dict[str, Any] = {}
    reacted_total = 0.0
    unreacted_total = 0.0
    for oxide, pct in oxide_mass_percent.items():
        if oxide in BOGUE_OXIDES or float(pct) <= 0:
            continue
        mass_g = opc_mass_g * float(pct) / 100.0
        rule = degrees_cfg.get(oxide, "clinker_mean")
        degree = clinker_mean if rule == "clinker_mean" else float(rule)
        degree = min(max(degree, 0.0), 1.0)
        if oxide not in oxide_species:
            warnings.append(f"OPC minor oxide '{oxide}' has no xGEMS species mapping; dropped ({mass_g:.3f} g).")
            per_oxide[oxide] = {"mass_g": mass_g, "degree": None, "species": None, "dropped": True}
            unreacted_total += mass_g
            continue
        reacted_g = mass_g * degree
        unreacted_g = mass_g - reacted_g
        species = str(oxide_species[oxide])
        if run_mode == "reacted_only":
            add_amount(species_amounts, species, reacted_g * 1.0e-3)
        else:
            add_amount(species_amounts, species, mass_g * 1.0e-3)
            lower_bounds[species] = lower_bounds.get(species, 0.0) + unreacted_g * 1.0e-3
        entry: dict[str, Any] = {"mass_g": mass_g, "degree": degree, "degree_rule": rule, "reacted_g": reacted_g, "unreacted_g": unreacted_g, "species": species}
        if oxide == "SO3" and sulfate_as_caso4:
            # calcium of the calcium sulfate (Bogue removed it from C3S; it must re-enter here)
            cao_species = str(oxide_species.get("CaO", "CaO"))
            cao_g = mass_g * _M_CAO / _M_SO3
            cao_reacted = cao_g * degree
            if run_mode == "reacted_only":
                add_amount(species_amounts, cao_species, cao_reacted * 1.0e-3)
            else:
                add_amount(species_amounts, cao_species, cao_g * 1.0e-3)
                lower_bounds[cao_species] = lower_bounds.get(cao_species, 0.0) + (cao_g - cao_reacted) * 1.0e-3
            entry["companion_CaO_g"] = cao_g
            entry["companion_CaO_reacted_g"] = cao_reacted
            reacted_g += cao_reacted
            unreacted_g += cao_g - cao_reacted
        per_oxide[oxide] = entry
        reacted_total += reacted_g
        unreacted_total += unreacted_g
    return {"enabled": True, "clinker_mean_degree": clinker_mean, "per_oxide": per_oxide, "reacted_g": reacted_total, "unreacted_g": unreacted_total}


def build_xgems_input(
    recipe: Recipe,
    *,
    materials: dict[str, Material] | None = None,
    species_map: dict[str, Any] | None = None,
    run_mode: str = "reacted_only",
    opc_phase_mass_percent: dict[str, float] | None = None,
    temperature_celsius: float = 20.0,
    relative_humidity: float | None = None,
    fineness_m2_kg: float | None = None,
    apply_availability_modifier: bool | None = None,
    reaction_parameters: ReactionParameterSet | None = None,
    xgems_water_mode: str = "initial",
    xgems_water_factor: float = 1.0,
    xgems_water_g: float | None = None,
    xgems_water_w_b: float | None = None,
) -> XGEMSInput:
    if run_mode not in {"reacted_only", "lower_bound_legacy"}:
        raise ValueError("run_mode must be 'reacted_only' or 'lower_bound_legacy'.")

    materials = materials or load_materials()
    species_map = species_map or load_species_map()
    reaction_parameters = reaction_parameters or load_reaction_parameters()
    relative_humidity = (
        reaction_parameters.relative_humidity if relative_humidity is None else float(relative_humidity)
    )
    fineness_m2_kg = reaction_parameters.fineness_m2_kg if fineness_m2_kg is None else float(fineness_m2_kg)
    apply_availability = (
        reaction_parameters.apply_availability_modifier
        if apply_availability_modifier is None
        else bool(apply_availability_modifier)
    )
    warnings: list[str] = list(recipe.warnings)
    warnings.extend(reaction_parameters.warnings)
    species_amounts: dict[str, float] = {}
    lower_bounds: dict[str, float] = {}
    unreacted_masses: dict[str, float] = {}
    water_policy = resolve_xgems_water(
        recipe,
        mode=xgems_water_mode,
        factor=xgems_water_factor,
        water_g=xgems_water_g,
        water_w_b=xgems_water_w_b,
    )
    equilibrium_water_g = float(water_policy["equilibrium_water_g"])

    opc_material = materials["OPC"]
    bogue_metadata: dict[str, Any] | None = None
    if opc_phase_mass_percent is None:
        if opc_material.oxide_mass_percent is None:
            raise ValueError("OPC phase composition not provided and OPC material lacks oxide composition.")
        bogue = bogue_phases(opc_material.oxide_mass_percent)
        opc_phase_mass_percent = bogue.phase_mass_percent
        bogue_metadata = bogue.to_dict()
        warnings.extend(bogue.warnings)
    else:
        opc_phase_mass_percent = {key: float(value) for key, value in opc_phase_mass_percent.items()}

    wc_for_pk = _water_to_binder_for_pk(recipe)
    pk_degrees = parrot_killoh(
        recipe.age_days,
        wc=wc_for_pk,
        RH=relative_humidity,
        T_celsius=temperature_celsius,
        fineness_m2_kg=fineness_m2_kg,
        parameters=reaction_parameters.pk_parameters,
    )
    pk_degrees_float = {phase: float(value) for phase, value in pk_degrees.items()}

    scm_params = reaction_parameters.scm_parameters
    availability_metadata: dict[str, Any] = {"applied": False}
    effective_scm_params = scm_params
    if apply_availability:
        modifier = C3S_C2SAvailabilityModifier(config=reaction_parameters.availability_config, materials=materials)
        modification = modifier.modify_parameters(recipe.binder_masses_g, opc_phase_mass_percent, scm_params)
        effective_scm_params = modification.parameters
        availability_metadata = modification.metadata

    scm_degrees: dict[str, float] = {}
    for name, mass in recipe.binder_masses_g.items():
        if name in SCM_NAMES and mass > 0:
            scm_degrees[name] = float(scm_alpha(recipe.age_days, effective_scm_params[name]))

    water_species = _species(species_map, "water", "H2O", "H2O@")
    add_amount(species_amounts, water_species, equilibrium_water_g * 1.0e-3)

    opc_mass = recipe.binder_masses_g.get("OPC", 0.0)
    if opc_mass > 0:
        unreacted_opc = 0.0
        for phase, phase_percent in opc_phase_mass_percent.items():
            phase_mass_g = opc_mass * float(phase_percent) / 100.0
            alpha = pk_degrees_float.get(phase, 0.0)
            reacted_g = phase_mass_g * alpha
            unreacted_g = phase_mass_g * (1.0 - alpha)
            xgems_species = _species(species_map, "clinker_phases", phase, phase)
            if run_mode == "reacted_only":
                add_amount(species_amounts, xgems_species, reacted_g * 1.0e-3)
            else:
                add_amount(species_amounts, xgems_species, phase_mass_g * 1.0e-3)
                lower_bounds[xgems_species] = lower_bounds.get(xgems_species, 0.0) + unreacted_g * 1.0e-3
            unreacted_opc += unreacted_g
        unreacted_masses["OPC"] = unreacted_opc

    opc_minor_metadata: dict[str, Any] = {"enabled": False}
    minor_cfg = dict(getattr(reaction_parameters, "opc_minor_oxides", None) or {})
    if opc_mass > 0 and opc_material.oxide_mass_percent and bool(minor_cfg.get("enabled", True)):
        opc_minor_metadata = _add_opc_minor_oxides(
            species_amounts,
            lower_bounds,
            species_map=species_map,
            opc_mass_g=opc_mass,
            oxide_mass_percent=opc_material.oxide_mass_percent,
            opc_phase_mass_percent=opc_phase_mass_percent,
            pk_degrees=pk_degrees_float,
            config=minor_cfg,
            run_mode=run_mode,
            warnings=warnings,
        )
        unreacted_masses["OPC"] = unreacted_masses.get("OPC", 0.0) + float(opc_minor_metadata["unreacted_g"])

    for name, mass in recipe.binder_masses_g.items():
        if mass <= 0:
            continue
        material = materials[name]
        if name in SCM_NAMES:
            alpha = scm_degrees[name]
            unreacted_masses[name] = mass * (1.0 - alpha)
            if material.oxide_mass_percent is None:
                warnings.append(f"SCM '{name}' has no oxide composition; no SCM oxides added to xGEMS.")
                continue
            for oxide, pct in material.oxide_mass_percent.items():
                xgems_species = _species(species_map, "oxides", oxide, oxide)
                add_amount(species_amounts, xgems_species, mass * alpha * float(pct) / 100.0 * 1.0e-3)
        elif name == "limestone":
            compounds = material.compound_mass_percent or {"CaCO3": 100.0}
            for compound, pct in compounds.items():
                xgems_species = _species(species_map, "limestone", compound, compound)
                add_amount(species_amounts, xgems_species, mass * float(pct) / 100.0 * 1.0e-3)
        elif name == "gypsum":
            xgems_species = _species(species_map, "gypsum", "gypsum", "Gp")
            add_amount(species_amounts, xgems_species, mass * 1.0e-3)

    reaction_degrees = {
        "age_days": recipe.age_days,
        "water_to_binder_for_pk": wc_for_pk,
        "xgems_water": water_policy,
        "opc": pk_degrees_float,
        "opc_phase_mass_percent": opc_phase_mass_percent,
        "opc_bogue_metadata": bogue_metadata,
        "opc_minor_oxides": opc_minor_metadata,
        "scm": scm_degrees,
        "scm_intrinsic_parameters": parameters_to_dict(scm_params),
        "scm_effective_parameters": parameters_to_dict(effective_scm_params),
        "availability_modifier": availability_metadata,
        "reaction_parameter_set": {
            "id": reaction_parameters.id,
            "config_path": reaction_parameters.config_path,
            "config_hash": reaction_parameters.config_hash,
            "relative_humidity": relative_humidity,
            "fineness_m2_kg": fineness_m2_kg,
            "apply_availability_modifier": apply_availability,
            "warnings": reaction_parameters.warnings,
        },
        "pk_parameters": reaction_parameters.pk_parameters.to_dict(),
    }

    return XGEMSInput(
        species_amounts_kg=dict(sorted(species_amounts.items())),
        lower_bounds_kg=dict(sorted(lower_bounds.items())),
        reaction_degrees=reaction_degrees,
        unreacted_masses_g=dict(sorted(unreacted_masses.items())),
        warnings=warnings,
        mode=run_mode,
        equilibrium_water_g=equilibrium_water_g,
        equilibrium_w_b=float(water_policy["equilibrium_w_b"]),
        water_policy=water_policy,
    )
