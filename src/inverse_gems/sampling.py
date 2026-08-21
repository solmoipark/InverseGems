from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Any

from .age_grids import DEFAULT_AGE_PRESET, add_age_metadata, get_age_values, parse_age_list
from .materials import BINDER_COMPONENTS
from .utils import config_path as default_config_path
from .utils import load_yaml

COMPONENTS = ["OPC", "slag", "fly_ash", "metakaolin", "silica_fume", "limestone", "gypsum"]

RECIPE_COLUMNS = [
    "recipe_id",
    "template_name",
    "material_system",
    "target_profile",
    *COMPONENTS,
    "w_b",
    "water_g",
    "water_mode",
    "age_days",
    "age_hours",
    "age_minutes",
    "age_label",
    "age_bin",
    "temperature_celsius",
]


def _bounds_from_config(config: dict[str, Any]) -> dict[str, tuple[float, float]]:
    return {
        component: (
            float((config.get(component) or {}).get("min", 0.0)),
            float((config.get(component) or {}).get("max", 100.0)),
        )
        for component in COMPONENTS
    }


def _bounds_pair(raw: Any, *, default: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
    if raw is None:
        return default
    if isinstance(raw, dict):
        return float(raw.get("min", default[0])), float(raw.get("max", default[1]))
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return float(raw[0]), float(raw[1])
    raise ValueError(f"Bounds must be [min, max] or {{min, max}}, got {raw!r}.")


def _material_system_sampling_config(
    base_config: dict[str, Any],
    *,
    material_system: str,
    material_systems_path: str | Path | None = None,
    strict_materials: bool = False,
    strict_allowed_materials: list[str] | tuple[str, ...] | None = None,
    optional_materials: list[str] | tuple[str, ...] = ("gypsum",),
) -> dict[str, Any]:
    systems = load_yaml(material_systems_path or default_config_path("material_systems.yaml"))
    if material_system not in systems:
        raise ValueError(f"Unknown material system '{material_system}'. Available: {sorted(systems)}")
    profile = systems[material_system] or {}
    allowed = {str(item) for item in profile.get("allowed", [])}
    if strict_materials:
        if strict_allowed_materials:
            allowed = {str(item) for item in strict_allowed_materials}
        else:
            allowed = allowed - {str(item) for item in optional_materials}
    fixed_zero = {str(item) for item in profile.get("fixed_zero", [])}
    unknown = sorted((allowed | fixed_zero) - set(COMPONENTS))
    if unknown:
        raise ValueError(f"Material system '{material_system}' contains unknown components: {unknown}")

    bounds_config = profile.get("bounds", {}) or {}
    template: dict[str, Any] = {}
    for component in COMPONENTS:
        if component in fixed_zero or (allowed and component not in allowed):
            template[component] = [0.0, 0.0]
        else:
            lo, hi = _bounds_pair(bounds_config.get(component), default=(0.0, 0.0))
            template[component] = [lo, hi]
    if profile.get("constraints"):
        template["constraints"] = profile["constraints"]

    config = dict(base_config)
    config["templates"] = {material_system: template}
    config["mixed"] = {"general_simplex_fraction": 0.0, "recipe_templates_fraction": 1.0}

    water = dict(config.get("water") or {})
    if profile.get("default_w_b") is not None:
        lo, hi = _bounds_pair(profile.get("default_w_b"), default=(0.30, 0.55))
        water["w_b"] = {"min": lo, "max": hi}
        water["mode"] = water.get("mode", "wb_total")
    config["water"] = water
    config["_active_material_system"] = material_system
    return config


def _load_target_sampling_profile(
    *,
    target_profile: str | None,
    target_profiles_path: str | Path | None = None,
) -> dict[str, Any]:
    if not target_profile:
        return {}
    profiles = load_yaml(target_profiles_path or default_config_path("targeted_sampling.yaml"))
    raw_profiles = profiles.get("profiles", profiles) or {}
    if target_profile not in raw_profiles:
        raise ValueError(f"Unknown target sampling profile '{target_profile}'. Available: {sorted(raw_profiles)}")
    profile = dict(raw_profiles[target_profile] or {})
    profile["name"] = target_profile
    return profile


def _apply_target_profile_to_config(config: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    if not profile:
        return config
    config = dict(config)
    water = dict(config.get("water") or {})
    if profile.get("water"):
        profile_water = profile.get("water") or {}
        if profile_water.get("mode") is not None:
            water["mode"] = profile_water["mode"]
        if profile_water.get("w_b") is not None:
            lo, hi = _bounds_pair(profile_water.get("w_b"), default=(0.30, 0.55))
            water["w_b"] = {"min": lo, "max": hi}
        if profile_water.get("water_g") is not None:
            lo, hi = _bounds_pair(profile_water.get("water_g"), default=(30.0, 55.0))
            water["water_g"] = {"min": lo, "max": hi}
    config["water"] = water
    config["_target_profile"] = profile.get("name", "")
    config["_target_profile_description"] = profile.get("description", "")
    return config


def _apply_target_profile_to_material_system_config(
    config: dict[str, Any],
    *,
    profile: dict[str, Any],
    material_system: str,
) -> dict[str, Any]:
    if not profile:
        return config
    config = _apply_target_profile_to_config(config, profile)
    overrides = profile.get("bounds_overrides") or {}
    system_overrides: dict[str, Any] = {}
    if overrides.get("default"):
        system_overrides.update(overrides.get("default") or {})
    if overrides.get(material_system):
        system_overrides.update(overrides.get(material_system) or {})
    if system_overrides:
        config = dict(config)
        templates = {name: dict(template or {}) for name, template in (config.get("templates") or {}).items()}
        template = dict(templates.get(material_system) or {})
        for component, raw_bounds in system_overrides.items():
            if component not in COMPONENTS:
                raise ValueError(f"Target sampling profile '{profile.get('name')}' has unknown component '{component}'.")
            lo, hi = _bounds_pair(raw_bounds, default=(0.0, 0.0))
            template[component] = [lo, hi]
        templates[material_system] = template
        config["templates"] = templates
    if profile.get("constraints_overrides"):
        system_constraints = (profile.get("constraints_overrides") or {}).get(material_system)
        if system_constraints:
            config = dict(config)
            templates = {name: dict(template or {}) for name, template in (config.get("templates") or {}).items()}
            template = dict(templates.get(material_system) or {})
            constraints = dict(template.get("constraints") or {})
            constraints.update(system_constraints)
            template["constraints"] = constraints
            templates[material_system] = template
            config["templates"] = templates
    return config


def _sample_bounded_simplex(
    rng: random.Random,
    bounds: dict[str, tuple[float, float]],
    *,
    total: float = 100.0,
    constraints: dict[str, Any] | None = None,
) -> dict[str, float]:
    mins = {key: bounds.get(key, (0.0, 100.0))[0] for key in COMPONENTS}
    maxs = {key: bounds.get(key, (0.0, 100.0))[1] for key in COMPONENTS}
    residual = total - sum(mins.values())
    if residual < -1.0e-9:
        raise ValueError("Component lower bounds exceed total binder mass.")
    caps = {key: maxs[key] - mins[key] for key in COMPONENTS}
    active = [key for key in COMPONENTS if caps[key] > 1.0e-12]
    for _ in range(10000):
        weights = [rng.expovariate(1.0) for _ in active]
        denom = sum(weights) or 1.0
        row = dict(mins)
        for key, weight in zip(active, weights):
            row[key] += residual * weight / denom
        if any(row[key] > maxs[key] + 1.0e-9 for key in COMPONENTS):
            continue
        if not _constraints_ok(row, constraints or {}):
            continue
        # Correct tiny numerical drift exactly on OPC.
        drift = total - sum(row.values())
        row["OPC"] += drift
        return {key: max(0.0, float(row[key])) for key in COMPONENTS}
    raise RuntimeError("Could not sample a bounded simplex row after many attempts.")


def _constraints_ok(row: dict[str, float], constraints: dict[str, Any]) -> bool:
    ratio = constraints.get("metakaolin_to_limestone_ratio")
    if ratio:
        limestone = row.get("limestone", 0.0)
        if limestone <= 0:
            return False
        value = row.get("metakaolin", 0.0) / limestone
        return float(ratio[0]) <= value <= float(ratio[1])
    return True


def _sample_water(rng: random.Random, config: dict[str, Any]) -> tuple[float, float, str]:
    mode = (config.get("water") or {}).get("mode", "wb_total")
    water = config.get("water") or {}
    if mode == "mixed":
        p = float(water.get("wb_probability", 0.7))
        mode = "wb_total" if rng.random() < p else "direct_h2o"
    if mode == "direct_h2o":
        bounds = water.get("water_g") or {"min": 5, "max": 70}
        water_g = rng.uniform(float(bounds["min"]), float(bounds["max"]))
        return water_g / 100.0, water_g, "direct_h2o"
    bounds = water.get("w_b") or {"min": 0.25, "max": 0.70}
    w_b = rng.uniform(float(bounds["min"]), float(bounds["max"]))
    return w_b, 100.0 * w_b, "wb_total"


def _sample_temperature(rng: random.Random, config: dict[str, Any]) -> float:
    temp = config.get("temperature_celsius") or {"mode": "fixed", "value": 20}
    if temp.get("mode", "fixed") == "uniform":
        return rng.uniform(float(temp["min"]), float(temp["max"]))
    return float(temp.get("value", 20.0))


def _sample_continuous_ages(
    rng: random.Random,
    *,
    age_sampling: str,
    age_min: float,
    age_max: float,
    age_count: int,
) -> list[float]:
    if age_count <= 0:
        raise ValueError("age_count must be positive.")
    if age_min <= 0.0 or age_max <= 0.0:
        raise ValueError("Continuous age sampling requires positive age_min and age_max.")
    if age_max < age_min:
        raise ValueError("age_max cannot be smaller than age_min.")
    mode = str(age_sampling).lower()
    values: list[float] = []
    for _ in range(age_count):
        if mode == "uniform":
            values.append(rng.uniform(age_min, age_max))
        elif mode == "log_uniform":
            lo = math.log(age_min)
            hi = math.log(age_max)
            values.append(math.exp(rng.uniform(lo, hi)))
        else:
            raise ValueError("age_sampling must be preset, uniform, or log_uniform.")
    return values


def _base_recipe_row(
    *,
    rng: random.Random,
    config: dict[str, Any],
    mode: str,
    index: int,
    recipe_id_prefix: str = "base",
) -> dict[str, Any]:
    if mode == "mixed":
        mixed = config.get("mixed") or {}
        p = float(mixed.get("general_simplex_fraction", 0.5))
        mode = "general_simplex" if rng.random() < p else "recipe_templates"
    if mode == "recipe_templates":
        templates = config.get("templates") or {}
        name = rng.choice(sorted(templates))
        template = templates[name]
        bounds = {
            component: (float((template.get(component) or [0, 0])[0]), float((template.get(component) or [0, 0])[1]))
            for component in COMPONENTS
        }
        binders = _sample_bounded_simplex(rng, bounds, constraints=template.get("constraints"))
        template_name = name
    elif mode == "general_simplex":
        binders = _sample_bounded_simplex(
            rng,
            _bounds_from_config(config.get("general_simplex") or {}),
            total=float((config.get("general_simplex") or {}).get("total_binder_g", 100.0)),
        )
        template_name = "general_simplex"
    else:
        raise ValueError("mode must be general_simplex, recipe_templates, or mixed.")
    w_b, water_g, water_mode = _sample_water(rng, config)
    prefix = str(recipe_id_prefix or "base")
    row: dict[str, Any] = {
        "recipe_id": f"{prefix}_{index:06d}",
        "template_name": template_name,
        "material_system": config.get("_active_material_system", ""),
        "target_profile": config.get("_target_profile", ""),
        "target_profile_description": config.get("_target_profile_description", ""),
        **binders,
        "w_b": w_b,
        "water_g": water_g,
        "water_mode": water_mode,
        "temperature_celsius": _sample_temperature(rng, config),
    }
    return row


def generate_recipe_rows(
    *,
    config_path: str | Path,
    n: int,
    mode: str,
    age_preset: str = DEFAULT_AGE_PRESET,
    ages: str | None = None,
    seed: int = 42,
    material_system: str | None = None,
    material_systems: list[str] | tuple[str, ...] | None = None,
    material_systems_sampling: str = "random",
    material_systems_path: str | Path | None = None,
    target_profile: str | None = None,
    target_profiles_path: str | Path | None = None,
    recipe_id_prefix: str = "base",
    strict_materials: bool = False,
    strict_allowed_materials: list[str] | tuple[str, ...] | None = None,
    age_sampling: str = "preset",
    age_min: float = 0.1,
    age_max: float = 365.0,
    age_count: int = 1,
) -> list[dict[str, Any]]:
    config = load_yaml(config_path)
    profile = _load_target_sampling_profile(target_profile=target_profile, target_profiles_path=target_profiles_path)
    config = _apply_target_profile_to_config(config, profile)
    requested_systems = [str(item) for item in (material_systems or []) if str(item).strip()]
    if profile and not material_system and not requested_systems:
        requested_systems = [str(item) for item in profile.get("material_systems", []) if str(item).strip()]
    if material_system and requested_systems:
        raise ValueError("Use either material_system or material_systems, not both.")
    material_systems_sampling = str(material_systems_sampling or "random").lower()
    if profile and profile.get("material_systems_sampling") and material_systems_sampling == "random":
        material_systems_sampling = str(profile.get("material_systems_sampling")).lower()
    if material_systems_sampling not in {"random", "balanced"}:
        raise ValueError("material_systems_sampling must be random or balanced.")
    if material_system:
        config = _material_system_sampling_config(
            config,
            material_system=material_system,
            material_systems_path=material_systems_path,
            strict_materials=strict_materials,
            strict_allowed_materials=strict_allowed_materials,
        )
        config = _apply_target_profile_to_material_system_config(config, profile=profile, material_system=material_system)
        mode = "recipe_templates"
    rng = random.Random(seed)
    age_sampling = str(age_sampling or "preset").lower()
    profile_age = profile.get("age_sampling") or {}
    if profile_age and not ages and age_sampling == "preset":
        if profile_age.get("ages") is not None:
            ages = str(profile_age.get("ages"))
        else:
            age_sampling = str(profile_age.get("mode", age_sampling)).lower()
            age_min = float(profile_age.get("min", age_min))
            age_max = float(profile_age.get("max", age_max))
            age_count = int(profile_age.get("count", age_count))
    if ages:
        age_values: list[float] | None = parse_age_list(ages)
    elif age_sampling == "preset":
        age_values = get_age_values(age_preset)
    else:
        age_values = None
    rows: list[dict[str, Any]] = []
    for i in range(1, n + 1):
        row_config = config
        row_mode = mode
        sampled_material_system = ""
        if requested_systems:
            sorted_systems = sorted(requested_systems)
            if material_systems_sampling == "balanced":
                sampled_material_system = sorted_systems[(i - 1) % len(sorted_systems)]
            else:
                sampled_material_system = rng.choice(sorted_systems)
            row_config = _material_system_sampling_config(
                config,
                material_system=sampled_material_system,
                material_systems_path=material_systems_path,
                strict_materials=strict_materials,
                strict_allowed_materials=strict_allowed_materials,
            )
            row_config = _apply_target_profile_to_material_system_config(
                row_config,
                profile=profile,
                material_system=sampled_material_system,
            )
            row_mode = "recipe_templates"
        base = _base_recipe_row(rng=rng, config=row_config, mode=row_mode, index=i, recipe_id_prefix=recipe_id_prefix)
        if requested_systems:
            base["material_system_sampling_mode"] = "mixed"
            base["material_systems_sampling"] = material_systems_sampling
            base["sampled_material_system"] = sampled_material_system
        else:
            base["material_system_sampling_mode"] = "single" if material_system else "config"
        base_age_values = (
            list(age_values)
            if age_values is not None
            else _sample_continuous_ages(
                rng,
                age_sampling=age_sampling,
                age_min=float(age_min),
                age_max=float(age_max),
                age_count=int(age_count),
            )
        )
        for age_index, age in enumerate(base_age_values, 1):
            row = add_age_metadata(base, age)
            row["age_sampling_mode"] = "explicit" if ages else age_sampling
            row["age_sample_index"] = age_index
            row["recipe_id"] = f"{base['recipe_id']}_age_{age:.8g}".replace(".", "p")
            rows.append(row)
    return rows


def expand_age_rows(
    base_rows: list[dict[str, Any]],
    *,
    age_preset: str | None = None,
    ages: str | None = None,
) -> list[dict[str, Any]]:
    age_values = parse_age_list(ages) if ages else get_age_values(age_preset or DEFAULT_AGE_PRESET)
    rows: list[dict[str, Any]] = []
    for i, base in enumerate(base_rows, 1):
        base_id = base.get("recipe_id") or f"base_{i:06d}"
        for age in age_values:
            row = dict(base)
            row = add_age_metadata(row, age)
            row["recipe_id"] = f"{base_id}_age_{age:g}".replace(".", "p")
            rows.append(row)
    return rows


def write_recipe_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = sorted(set().union(*(row.keys() for row in rows)) - set(RECIPE_COLUMNS)) if rows else []
    fields = RECIPE_COLUMNS + extra
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_recipe_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def recipe_text_from_row(row: dict[str, Any]) -> str:
    parts: list[str] = []
    labels = {
        "OPC": "OPC",
        "slag": "slag",
        "fly_ash": "fly ash",
        "metakaolin": "metakaolin",
        "silica_fume": "silica fume",
        "limestone": "limestone",
        "gypsum": "gypsum",
    }
    for key, label in labels.items():
        value = row.get(key)
        if value not in (None, "") and float(value) != 0.0:
            parts.append(f"{label} {float(value):.12g}")
    water_mode = row.get("water_mode")
    if water_mode == "direct_h2o" and row.get("water_g") not in (None, ""):
        parts.append(f"water {float(row['water_g']):.12g}")
    elif row.get("w_b") not in (None, ""):
        parts.append(f"w/b {float(row['w_b']):.12g}")
    elif row.get("water_g") not in (None, ""):
        parts.append(f"water {float(row['water_g']):.12g}")
    else:
        raise ValueError("Recipe row must include w_b or water_g.")
    age = row.get("age_days") or row.get("age")
    if age in (None, ""):
        raise ValueError("Recipe row must include age_days.")
    parts.append(f"age {float(age):.12g}")
    return ", ".join(parts)


def numeric_metadata_from_row(row: dict[str, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for key in [
        "recipe_id",
        "template_name",
        "material_system",
        "target_profile",
        "target_profile_description",
        "water_mode",
        "xgems_water_mode",
        "age_label",
        "age_bin",
    ]:
        if row.get(key) not in (None, ""):
            meta[key] = row[key]
    for key in [
        "age_days",
        "age_hours",
        "age_minutes",
        "temperature_celsius",
        "water_g",
        "w_b",
        "xgems_water_g",
        "xgems_w_b",
        "xgems_water_factor",
    ]:
        if row.get(key) not in (None, ""):
            meta[key] = float(row[key])
    return meta
