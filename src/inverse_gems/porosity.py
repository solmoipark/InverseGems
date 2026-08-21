from __future__ import annotations

from pathlib import Path
from typing import Any

from .materials import Material, load_materials
from .recipe import Recipe
from .utils import as_float, config_path, load_yaml


def load_porosity_config(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(path or config_path("porosity.yaml"))


def compute_initial_volume_cm3(recipe: Recipe, materials: dict[str, Material] | None = None) -> dict[str, Any]:
    materials = materials or load_materials()
    component_volumes: dict[str, float] = {}
    warnings: list[str] = []
    for name, mass in recipe.binder_masses_g.items():
        material = materials.get(name)
        if material is None:
            warnings.append(f"No material density for '{name}'; omitted from initial volume.")
            continue
        component_volumes[name] = float(mass) / material.density_g_cm3
    water_density = materials.get("water").density_g_cm3 if materials.get("water") else 1.0
    water_volume = recipe.water_g / water_density
    total = water_volume + sum(component_volumes.values())
    return {
        "water_volume_cm3": water_volume,
        "binder_component_volumes_cm3": component_volumes,
        "initial_volume_cm3": total,
        "warnings": warnings,
    }


def _is_non_solid_phase(name: str, patterns: list[str]) -> bool:
    lowered = name.lower()
    return any(pattern.lower() in lowered for pattern in patterns)


def _volume_to_cm3(value: float, unit: str) -> float:
    normalized = unit.strip().lower()
    if normalized in {"cm3", "cm^3", "cc", "ml"}:
        return value
    if normalized in {"m3", "m^3"}:
        return value * 1.0e6
    if normalized in {"dm3", "l", "liter", "litre"}:
        return value * 1000.0
    raise ValueError(f"Unsupported xGEMS phase volume unit '{unit}'.")


def compute_porosity(
    recipe: Recipe,
    *,
    materials: dict[str, Material] | None = None,
    xgems_phase_volumes: dict[str, Any] | None = None,
    xgems_phase_volumes_reconstructed: dict[str, Any] | None = None,
    unreacted_masses_g: dict[str, float] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    materials = materials or load_materials()
    config = config or load_porosity_config()
    xgems_phase_volumes = xgems_phase_volumes or {}
    xgems_phase_volumes_reconstructed = xgems_phase_volumes_reconstructed or {}
    unreacted_masses_g = unreacted_masses_g or {}
    warnings: list[str] = []

    initial = compute_initial_volume_cm3(recipe, materials)
    warnings.extend(initial["warnings"])
    patterns = list(config.get("non_solid_phase_name_patterns", []))
    xgems_volume_unit = str(config.get("xgems_phase_volume_unit", "cm3"))
    included_phases: dict[str, float] = {}
    excluded_phases: dict[str, Any] = {}
    skipped_phases: dict[str, Any] = {}
    phase_volume_sources: dict[str, str] = {}
    use_reconstructed = bool(config.get("prefer_reconstructed_phase_volumes", True))
    effective_phase_volumes = dict(xgems_phase_volumes)
    if use_reconstructed and xgems_phase_volumes_reconstructed:
        for name, value in xgems_phase_volumes_reconstructed.items():
            numeric = as_float(value)
            if numeric is not None and abs(numeric) > 0.0:
                effective_phase_volumes[name] = value

    for name, value in effective_phase_volumes.items():
        if _is_non_solid_phase(name, patterns):
            excluded_phases[name] = value
            continue
        numeric = as_float(value)
        if numeric is None:
            skipped_phases[name] = value
            warnings.append(f"Could not interpret xGEMS phase volume for '{name}' as a number; skipped.")
            continue
        included_phases[name] = _volume_to_cm3(numeric, xgems_volume_unit)
        if name in xgems_phase_volumes_reconstructed and abs(as_float(xgems_phase_volumes_reconstructed.get(name), 0.0) or 0.0) > 0.0:
            phase_volume_sources[name] = "reconstructed_from_phase_species"
        else:
            phase_volume_sources[name] = "raw_phase_volume"

    xgems_solid_volume = sum(included_phases.values())
    unreacted_volumes: dict[str, float] = {}
    if config.get("include_unreacted_binders", True) and config.get("xgems_run_mode", "reacted_only") in (
        "reacted_only",
    ):
        for name, mass in unreacted_masses_g.items():
            material = materials.get(name)
            if material is None:
                warnings.append(f"No material density for unreacted '{name}'; omitted from porosity.")
                continue
            unreacted_volumes[name] = float(mass) / material.density_g_cm3

    solid_final = xgems_solid_volume + sum(unreacted_volumes.values())
    initial_volume = float(initial["initial_volume_cm3"])
    porosity = 1.0 - solid_final / initial_volume if initial_volume > 0 else float("nan")
    if porosity < 0.0 or porosity > 1.0:
        warnings.append(f"Best-effort porosity {porosity:g} is outside [0, 1].")
        if config.get("clip_porosity_to_0_1", False):
            porosity = max(0.0, min(1.0, porosity))

    return {
        "porosity_best_effort": porosity,
        "porosity_inputs_raw": {
            "xgems_phase_volumes": xgems_phase_volumes,
            "xgems_phase_volumes_reconstructed": xgems_phase_volumes_reconstructed,
            "prefer_reconstructed_phase_volumes": use_reconstructed,
            "unreacted_masses_g": unreacted_masses_g,
            "initial": initial,
            "config": config,
        },
        "initial_volume_cm3": initial_volume,
        "xgems_solid_phase_volume_cm3": xgems_solid_volume,
        "unreacted_binder_volumes_cm3": unreacted_volumes,
        "solid_final_volume_cm3": solid_final,
        "included_phase_volumes_cm3": included_phases,
        "included_phase_volume_sources": phase_volume_sources,
        "excluded_non_solid_phase_volumes_raw": excluded_phases,
        "skipped_phase_volumes_raw": skipped_phases,
        "warnings": warnings,
    }
