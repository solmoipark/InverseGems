from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .materials import BINDER_COMPONENTS, canonicalize_material_name
from .model_registry_diagnostics import diagnose_model_target_availability
from .reaction_model import current_reaction_model_metadata
from .reaction_parameters import load_reaction_parameters
from .utils import config_path, load_yaml, write_json


BINDER_INPUTS = {
    "OPC",
    "slag",
    "fly_ash",
    "metakaolin",
    "silica_fume",
    "limestone",
    "gypsum",
    "water_g",
    "w_b",
    "age_days",
    "age_hours",
    "age_minutes",
    "temperature_celsius",
}

ScalarValue = str | int | float | bool


class ConstraintSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: float | None = None
    max: float | None = None
    equals: ScalarValue | None = None
    tolerance: float | None = None
    min_abs: float | None = None
    max_abs: float | None = None
    include: list[ScalarValue] | None = None
    exclude: list[ScalarValue] | None = None

    @model_validator(mode="after")
    def validate_bounds(self) -> "ConstraintSpec":
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("min cannot be greater than max.")
        for name in ("tolerance", "min_abs", "max_abs"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative.")
        if self.min_abs is not None and self.max_abs is not None and self.min_abs > self.max_abs:
            raise ValueError("min_abs cannot be greater than max_abs.")
        return self


class ConstraintGroups(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: dict[str, ConstraintSpec] | None = None
    inputs: dict[str, ConstraintSpec] | None = None
    targets: dict[str, Any] | None = None
    predicted_targets: dict[str, ConstraintSpec] | None = None
    validated_targets: dict[str, ConstraintSpec] | None = None
    source_true_targets: dict[str, ConstraintSpec] | None = None
    prediction_errors: dict[str, ConstraintSpec] | None = None


class PreferenceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str | None = None
    input: str | None = None
    name: str | None = None
    section: str | None = None
    kind: str | None = None
    direction: Literal["maximize", "minimize"] | None = None
    maximize: str | dict[str, str] | None = None
    minimize: str | dict[str, str] | None = None
    tolerance: float | None = None

    @model_validator(mode="after")
    def validate_preference(self) -> "PreferenceSpec":
        shorthand_count = int(self.maximize is not None) + int(self.minimize is not None)
        if shorthand_count > 1:
            raise ValueError("Preference cannot set both maximize and minimize.")
        if shorthand_count == 0:
            if self.direction is None:
                raise ValueError("Preference must include direction when maximize/minimize shorthand is not used.")
            if self.name is None and self.target is None and self.input is None:
                raise ValueError("Preference must include one of name, target, or input.")
        if self.tolerance is not None and self.tolerance < 0:
            raise ValueError("Preference tolerance must be non-negative.")
        return self


class RankingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["weighted", "score", "lexicographic", "ordered"] = "weighted"
    preferences: list[PreferenceSpec] | None = None


class DesignSpaceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    material_systems: str | list[str] | None = None
    allowed_materials: list[str] | None = None
    strict_materials: bool | None = None
    input_constraints: dict[str, ConstraintSpec] | None = None
    age_days: float | ConstraintSpec | None = None
    age_bin: str | list[str] | None = None

    @model_validator(mode="after")
    def validate_design_space(self) -> "DesignSpaceSpec":
        if isinstance(self.age_days, (int, float)) and float(self.age_days) <= 0:
            raise ValueError("design_space.age_days must be positive.")
        return self


class ValidationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_k_xgems: int | None = Field(default=None, ge=1)
    search_top_k: int | None = Field(default=None, ge=1)
    use_thermo_cache: bool = True
    use_nearest_neighbors: bool = True
    prediction_errors: dict[str, ConstraintSpec] | None = None


class ReactionModelRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    signature: str | None = None
    config: str | None = None
    mismatch_policy: Literal["error", "warn", "warning", "ignore", "off", "false"] | None = None


class DesignQuerySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    design_space: DesignSpaceSpec | None = None
    reaction_model: ReactionModelRef | None = None
    output_constraints: dict[str, Any] | None = None
    validation: ValidationSpec | None = None
    material_system: str | list[str] | None = None
    age_days: float | None = None
    age_bin: str | list[str] | None = None
    model_id: str | None = None
    model_table: str | None = None
    model_bundle: str | None = None
    inputs: dict[str, ConstraintSpec] | None = None
    metadata: dict[str, ConstraintSpec] | None = None
    targets: dict[str, Any] | None = None
    prediction_errors: dict[str, ConstraintSpec] | None = None
    hard_constraints: ConstraintGroups | None = None
    constraints: ConstraintGroups | None = None
    ranking: RankingSpec | None = None
    preferences: list[PreferenceSpec] | None = None
    objectives: dict[str, Any] | list[PreferenceSpec] | None = None
    chemistry_status: str = "complete"
    top_k: int | None = Field(default=None, ge=1)
    search_top_k: int | None = Field(default=None, ge=1)
    selection_top_k: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_query(self) -> "DesignQuerySpec":
        if self.age_days is not None and self.age_days <= 0:
            raise ValueError("age_days must be positive.")
        return self


def validate_design_query_data(
    data: dict[str, Any],
    *,
    model_table_override: str | Path | None = None,
    model_bundle_override: str | Path | None = None,
    model_registry: str | Path | None = None,
    reaction_model_id: str | None = None,
    reaction_model_signature: str | None = None,
    reaction_model_config: str | Path | None = None,
    require_model_paths: bool = False,
) -> dict[str, Any]:
    """Validate API/LLM-facing design query data and return a plain dict."""
    model = DesignQuerySpec.model_validate(data)
    payload = model.model_dump(exclude_none=True)
    if require_model_paths:
        resolved = resolve_model_paths(
            payload,
            model_table_override=model_table_override,
            model_bundle_override=model_bundle_override,
            model_registry=model_registry,
            reaction_model_id=reaction_model_id,
            reaction_model_signature=reaction_model_signature,
            reaction_model_config=reaction_model_config,
        )
        if not resolved["model_table"]:
            raise ValueError("Design query must include model_table, --model-table, or match the model registry.")
        if not resolved["model_bundle"]:
            raise ValueError("Design query must include model_bundle, --model-bundle, or match the model registry.")
    return payload


def validate_design_query_file(
    query: str | Path,
    *,
    model_registry: str | Path | None = None,
    reaction_model_id: str | None = None,
    reaction_model_signature: str | None = None,
    reaction_model_config: str | Path | None = None,
    require_model_paths: bool = False,
) -> dict[str, Any]:
    return validate_design_query_data(
        load_yaml(query),
        model_registry=model_registry,
        reaction_model_id=reaction_model_id,
        reaction_model_signature=reaction_model_signature,
        reaction_model_config=reaction_model_config,
        require_model_paths=require_model_paths,
    )


def design_query_json_schema() -> dict[str, Any]:
    return DesignQuerySpec.model_json_schema()


def save_design_query_schema(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, design_query_json_schema())
    return out


def load_model_registry(path: str | Path | None = None) -> dict[str, Any]:
    registry_path = Path(path) if path is not None else config_path("design_query_model_registry.global_v1.yaml")
    if not registry_path.exists():
        return {"models": [], "path": str(registry_path)}
    data = load_yaml(registry_path)
    data["path"] = str(registry_path)
    return data


def _single_material_system(value: Any) -> str | None:
    systems = _as_list(value)
    systems = [str(item) for item in systems if str(item)]
    return systems[0] if len(systems) == 1 else None


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _require_same(label: str, first: str | None, second: str | None) -> str | None:
    first = _clean_str(first)
    second = _clean_str(second)
    if first and second and first != second:
        raise ValueError(f"Conflicting reaction-model {label}: query={first!r}, override={second!r}.")
    return second or first


def _require_same_path(label: str, first: str | None, second: str | None) -> str | None:
    first = _clean_str(first)
    second = _clean_str(second)
    if first and second:
        first_path = Path(first).resolve(strict=False)
        second_path = Path(second).resolve(strict=False)
        if first_path != second_path:
            raise ValueError(f"Conflicting reaction-model {label}: query={first!r}, override={second!r}.")
    return second or first


def _first_entry_reaction_value(entry: dict[str, Any], singular: str, plural: str) -> str | None:
    values: list[Any] = []
    for value in [entry.get(singular), entry.get(plural)]:
        values.extend(_as_list(value))
    cleaned = [_clean_str(value) for value in values]
    cleaned = [value for value in cleaned if value]
    unique = list(dict.fromkeys(cleaned))
    return unique[0] if len(unique) == 1 else None


def _entry_reaction_model_id(entry: dict[str, Any]) -> str | None:
    nested = entry.get("reaction_model") if isinstance(entry.get("reaction_model"), dict) else {}
    direct = _first_entry_reaction_value(entry, "reaction_model_id", "reaction_model_ids")
    nested_id = _first_entry_reaction_value(nested, "id", "ids")
    return direct or nested_id


def _entry_reaction_model_signature(entry: dict[str, Any]) -> str | None:
    nested = entry.get("reaction_model") if isinstance(entry.get("reaction_model"), dict) else {}
    direct = _first_entry_reaction_value(entry, "reaction_model_signature", "reaction_model_signatures")
    nested_signature = _first_entry_reaction_value(nested, "signature", "signatures")
    return direct or nested_signature


def _entry_matches_reaction_request(entry: dict[str, Any], request: dict[str, Any]) -> bool:
    request_id = _clean_str(request.get("id"))
    request_signature = _clean_str(request.get("signature"))
    if request_id:
        entry_id = _entry_reaction_model_id(entry)
        if entry_id != request_id:
            return False
    if request_signature:
        entry_signature = _entry_reaction_model_signature(entry)
        if entry_signature != request_signature:
            return False
    return True


def _format_registry_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": entry.get("id"),
            "material_system": entry.get("material_system"),
            "age_days": entry.get("age_days"),
            "age_min_days": entry.get("age_min_days"),
            "age_max_days": entry.get("age_max_days"),
            "reaction_model_id": _entry_reaction_model_id(entry),
            "reaction_model_signature": _entry_reaction_model_signature(entry),
        }
        for entry in entries
    ]


def _entry_age_delta(entry: dict[str, Any], requested_age: float, default_tolerance: float) -> float | None:
    age_min = entry.get("age_min_days")
    age_max = entry.get("age_max_days")
    if age_min is not None or age_max is not None:
        lower = float(age_min if age_min is not None else "-inf")
        upper = float(age_max if age_max is not None else "inf")
        if lower <= requested_age <= upper:
            return 0.0
        delta = min(abs(requested_age - lower), abs(requested_age - upper))
        tolerance = float(entry.get("age_tolerance", default_tolerance))
        return delta if delta <= tolerance else None
    if entry.get("age_days") is None:
        return None
    tolerance = float(entry.get("age_tolerance", default_tolerance))
    delta = abs(float(entry["age_days"]) - requested_age)
    return delta if delta <= tolerance else None


def resolve_reaction_model_request(
    query_data: dict[str, Any],
    *,
    reaction_model_id: str | None = None,
    reaction_model_signature: str | None = None,
    reaction_model_config: str | Path | None = None,
) -> dict[str, Any]:
    reaction = dict(query_data.get("reaction_model") or {})
    requested_id = _require_same("id", reaction.get("id"), reaction_model_id)
    requested_signature = _require_same("signature", reaction.get("signature"), reaction_model_signature)
    requested_config = _require_same_path("config", reaction.get("config"), str(reaction_model_config) if reaction_model_config else None)

    if requested_config and not requested_signature:
        parameters = load_reaction_parameters(requested_config, reaction_model_id=requested_id)
        metadata = current_reaction_model_metadata(
            reaction_model_id=parameters.id,
            reaction_model_config=requested_config,
            reaction_parameters=parameters,
        )
        requested_id = requested_id or metadata["reaction_model_id"]
        requested_signature = metadata["reaction_model_signature"]

    return {
        "id": requested_id,
        "signature": requested_signature,
        "config": requested_config,
        "mismatch_policy": reaction.get("mismatch_policy"),
    }


def _reaction_model_block(query_data: dict[str, Any], registry_entry: dict[str, Any] | None, request: dict[str, Any]) -> dict[str, Any] | None:
    block = dict(query_data.get("reaction_model") or {})
    if request.get("id"):
        block["id"] = request["id"]
    elif registry_entry and _entry_reaction_model_id(registry_entry):
        block["id"] = _entry_reaction_model_id(registry_entry)
    if request.get("signature"):
        block["signature"] = request["signature"]
    elif registry_entry and _entry_reaction_model_signature(registry_entry):
        block["signature"] = _entry_reaction_model_signature(registry_entry)
    if request.get("config"):
        block["config"] = request["config"]
    if request.get("mismatch_policy"):
        block["mismatch_policy"] = request["mismatch_policy"]
    return block or None


def _match_model_registry(
    query_data: dict[str, Any],
    model_registry: str | Path | None = None,
    *,
    reaction_model_id: str | None = None,
    reaction_model_signature: str | None = None,
    reaction_model_config: str | Path | None = None,
) -> dict[str, Any] | None:
    registry = load_model_registry(model_registry)
    entries = list(registry.get("models") or registry.get("entries") or [])
    reaction_request = resolve_reaction_model_request(
        query_data,
        reaction_model_id=reaction_model_id,
        reaction_model_signature=reaction_model_signature,
        reaction_model_config=reaction_model_config,
    )
    model_id = query_data.get("model_id")
    if model_id:
        for entry in entries:
            if str(entry.get("id")) == str(model_id):
                if not _entry_matches_reaction_request(entry, reaction_request):
                    raise ValueError(
                        f"Registry entry {model_id!r} does not match requested reaction model. "
                        f"Requested id={reaction_request.get('id')!r}, signature={reaction_request.get('signature')!r}; "
                        f"entry id={_entry_reaction_model_id(entry)!r}, signature={_entry_reaction_model_signature(entry)!r}."
                    )
                return {**entry, "_registry_path": registry.get("path")}
        raise ValueError(f"No design-query model registry entry has id '{model_id}'.")

    material_system = _single_material_system(_query_material_system(query_data))
    age_days = _query_age_days(query_data)
    if not material_system or age_days is None:
        return None
    default_tolerance = float(registry.get("default_age_tolerance", 1.0e-9))
    matches: list[dict[str, Any]] = []
    base_matches: list[dict[str, Any]] = []
    for entry in entries:
        if str(entry.get("material_system")) != material_system:
            continue
        age_delta = _entry_age_delta(entry, float(age_days), default_tolerance)
        if age_delta is not None:
            base_matches.append({**entry, "_registry_path": registry.get("path")})
    if reaction_request.get("id") or reaction_request.get("signature"):
        for entry in base_matches:
            if _entry_matches_reaction_request(entry, reaction_request):
                matches.append(entry)
    else:
        unversioned_matches = [
            entry for entry in base_matches if not _entry_reaction_model_id(entry) and not _entry_reaction_model_signature(entry)
        ]
        matches = unversioned_matches or base_matches
    if base_matches and not matches and (reaction_request.get("id") or reaction_request.get("signature")):
        raise ValueError(
            f"No registry models match requested reaction model for material_system={material_system!r}, age_days={age_days!r}. "
            f"Requested id={reaction_request.get('id')!r}, signature={reaction_request.get('signature')!r}. "
            f"Available matches: {_format_registry_entries(base_matches)}"
        )
    if len(matches) > 1:
        ids = [str(entry.get("id")) for entry in matches]
        raise ValueError(
            f"Multiple registry models match material_system={material_system!r}, age_days={age_days!r}. "
            f"Set model_id explicitly or provide reaction_model.id/signature. Matches: {ids}"
        )
    return matches[0] if matches else None


def _query_material_system(query_data: dict[str, Any]) -> Any:
    return query_data.get("material_system") or _design_space(query_data).get("material_systems")


def _query_age_days(query_data: dict[str, Any]) -> Any:
    if query_data.get("age_days") is not None:
        return query_data.get("age_days")
    value = _design_space(query_data).get("age_days")
    if isinstance(value, dict):
        return value.get("equals")
    return value


def resolve_model_paths(
    query_data: dict[str, Any],
    *,
    model_table_override: str | Path | None = None,
    model_bundle_override: str | Path | None = None,
    model_registry: str | Path | None = None,
    reaction_model_id: str | None = None,
    reaction_model_signature: str | None = None,
    reaction_model_config: str | Path | None = None,
) -> dict[str, Any]:
    registry_entry = None
    if not (model_table_override and model_bundle_override):
        registry_entry = _match_model_registry(
            query_data,
            model_registry,
            reaction_model_id=reaction_model_id,
            reaction_model_signature=reaction_model_signature,
            reaction_model_config=reaction_model_config,
        )
    model_table = model_table_override or query_data.get("model_table") or (registry_entry or {}).get("model_table")
    model_bundle = model_bundle_override or query_data.get("model_bundle") or (registry_entry or {}).get("model_bundle")
    return {
        "model_table": str(model_table) if model_table else "",
        "model_bundle": str(model_bundle) if model_bundle else "",
        "registry_entry": registry_entry,
    }


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _is_auto_material_system(value: Any) -> bool:
    values = [str(item).strip().lower() for item in _as_list(value)]
    return any(item in {"auto", "any", "all"} for item in values)


def _design_space(query_data: dict[str, Any]) -> dict[str, Any]:
    return dict(query_data.get("design_space") or {})


def _as_constraint_spec(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return {"include": value}
    return {"equals": value}


def _merge_constraints(*parts: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for part in parts:
        for name, spec in (part or {}).items():
            existing = merged.setdefault(str(name), {})
            existing.update(_as_constraint_spec(spec))
    return merged


def _normalise_target_constraints(raw: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not raw:
        return {}
    section_names = {"phase_amounts", "phase_volumes", "aqueous_species", "scalars"}
    if any(section in raw for section in section_names):
        flat: dict[str, dict[str, Any]] = {}
        for values in raw.values():
            if isinstance(values, dict):
                flat.update(_merge_constraints(values))
        return flat
    return _merge_constraints(raw)


def _normalise_metadata(query_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    hard = query_data.get("hard_constraints", {}) or {}
    constraints = query_data.get("constraints", {}) or {}
    design_space = _design_space(query_data)
    metadata = _merge_constraints(
        query_data.get("metadata"),
        hard.get("metadata"),
        constraints.get("metadata"),
    )
    material_system = query_data.get("material_system") or design_space.get("material_systems")
    if material_system and not _is_auto_material_system(material_system):
        metadata.setdefault("material_system", {})["include"] = _as_list(material_system)
    age_bin = query_data.get("age_bin") or design_space.get("age_bin")
    if age_bin:
        metadata.setdefault("age_bin", {})["include"] = _as_list(age_bin)
    return metadata


def _allowed_material_constraints(allowed_materials: list[str] | None) -> dict[str, dict[str, Any]]:
    if not allowed_materials:
        return {}
    allowed = {canonicalize_material_name(str(name)) for name in allowed_materials}
    if "water" in allowed:
        allowed.remove("water")
    unknown = sorted(allowed - BINDER_COMPONENTS)
    if unknown:
        raise ValueError(f"allowed_materials contains unsupported binder component(s): {unknown}")
    return {name: {"max": 0.0} for name in sorted(BINDER_COMPONENTS - allowed)}


def _normalise_input_constraints(query_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    hard = query_data.get("hard_constraints", {}) or {}
    constraints = query_data.get("constraints", {}) or {}
    design_space = _design_space(query_data)
    inputs = _merge_constraints(
        _allowed_material_constraints(design_space.get("allowed_materials")),
        query_data.get("inputs"),
        design_space.get("input_constraints"),
        hard.get("inputs"),
        constraints.get("inputs"),
    )
    if query_data.get("age_days") is not None:
        inputs.setdefault("age_days", {})["equals"] = float(query_data["age_days"])
    if design_space.get("age_days") is not None:
        age_value = design_space["age_days"]
        if isinstance(age_value, dict):
            inputs.setdefault("age_days", {}).update(_as_constraint_spec(age_value))
        else:
            inputs.setdefault("age_days", {})["equals"] = float(age_value)
    return inputs


def _normalise_prediction_errors(query_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    hard = query_data.get("hard_constraints", {}) or {}
    constraints = query_data.get("constraints", {}) or {}
    validation = dict(query_data.get("validation") or {})
    return _merge_constraints(
        query_data.get("prediction_errors"),
        validation.get("prediction_errors"),
        hard.get("prediction_errors"),
        constraints.get("prediction_errors"),
    )


def _normalise_hard_targets(query_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    hard = query_data.get("hard_constraints", {}) or {}
    constraints = query_data.get("constraints", {}) or {}
    return _merge_constraints(
        _normalise_target_constraints(query_data.get("output_constraints")),
        _normalise_target_constraints(query_data.get("targets")),
        _normalise_target_constraints(hard.get("targets")),
        _normalise_target_constraints(constraints.get("targets")),
        _normalise_target_constraints(constraints.get("validated_targets")),
    )


def _infer_preference_section(name: str, raw_section: Any, *, target_section: str) -> str:
    if raw_section is None:
        return "inputs" if name in BINDER_INPUTS else target_section
    section = str(raw_section)
    if section in {"input", "inputs"}:
        return "inputs"
    if section in {"target", "targets", "predicted_targets", "validated_targets", "source_true_targets"}:
        return target_section
    if section in {"metadata", "meta"}:
        return "metadata"
    if section in {"prediction_error", "prediction_errors"}:
        return "prediction_errors"
    return target_section


def _normalise_one_preference(preference: dict[str, Any], *, target_section: str) -> dict[str, str]:
    if "maximize" in preference or "minimize" in preference:
        direction = "maximize" if "maximize" in preference else "minimize"
        raw_name = preference[direction]
        if isinstance(raw_name, dict):
            if len(raw_name) != 1:
                raise ValueError("Preference mappings must contain exactly one item.")
            raw_section, name = next(iter(raw_name.items()))
        else:
            raw_section = preference.get("section") or preference.get("kind")
            name = raw_name
    else:
        direction = str(preference.get("direction") or "").lower()
        raw_section = preference.get("section") or preference.get("kind")
        name = preference.get("name") or preference.get("target") or preference.get("input")
    if direction not in {"maximize", "minimize"}:
        raise ValueError("Each preference must set direction to maximize or minimize.")
    if not name:
        raise ValueError("Each preference must include a name.")
    name = str(name)
    item = {
        "section": _infer_preference_section(name, raw_section, target_section=target_section),
        "name": name,
        "direction": direction,
    }
    if preference.get("tolerance") is not None:
        item["tolerance"] = preference["tolerance"]
    return item


def _normalise_preferences(query_data: dict[str, Any], *, target_section: str) -> list[dict[str, str]]:
    preferences = query_data.get("preferences")
    if preferences is None:
        preferences = (query_data.get("ranking", {}) or {}).get("preferences")
    objectives = query_data.get("objectives")
    if preferences is None and isinstance(objectives, list):
        preferences = objectives
    if not preferences:
        return []
    if not isinstance(preferences, list):
        raise ValueError("preferences must be a list.")
    return [_normalise_one_preference(dict(item), target_section=target_section) for item in preferences]


def _target_name_keys(value: Any) -> set[str]:
    text = str(value)
    variants = {text}
    for prefix in ("pred__", "true__", "residual__"):
        if text.startswith(prefix):
            variants.add(text.removeprefix(prefix))
    for candidate in list(variants):
        for prefix in ("y__amount_", "y__volume_", "y__scalar_", "y__"):
            if candidate.startswith(prefix):
                variants.add(candidate.removeprefix(prefix))
    for candidate in list(variants):
        variants.add(candidate.replace("_", "-"))
        variants.add(candidate.replace("-", "_"))
        variants.add(candidate.replace("_", " "))
    return {re.sub(r"[^0-9a-z]+", "", candidate.lower()) for candidate in variants if str(candidate).strip()}


def _requested_target_names(
    *,
    target_constraints: dict[str, dict[str, Any]],
    prediction_errors: dict[str, dict[str, Any]],
    search_preferences: list[dict[str, Any]],
    selection_preferences: list[dict[str, Any]],
) -> list[str]:
    names: list[str] = []
    for name in list(target_constraints) + list(prediction_errors):
        names.append(str(name))
    for preference in list(search_preferences) + list(selection_preferences):
        if preference.get("section") in {"predicted_targets", "validated_targets", "source_true_targets"}:
            names.append(str(preference.get("name")))
    return list(dict.fromkeys(name for name in names if name and name != "None"))


def _check_target_availability(
    *,
    requested_targets: list[str],
    model_table: str | Path,
    model_bundle: str | Path,
    model_resolution: dict[str, Any],
    out_dir: Path,
    policy: str,
) -> dict[str, Any]:
    if policy not in {"ignore", "warn", "error"}:
        raise ValueError("target_availability_policy must be one of: ignore, warn, error.")
    report: dict[str, Any] = {
        "policy": policy,
        "checked": policy != "ignore",
        "requested_targets": requested_targets,
        "matched_targets": [],
        "issues": [],
        "warnings": [],
    }
    if policy == "ignore":
        return report
    frame, warnings = diagnose_model_target_availability(
        model_table=model_table,
        model_bundle=model_bundle,
        registry_entry=model_resolution.get("registry_entry"),
    )
    report["warnings"] = warnings
    rows = frame.to_dict(orient="records") if not frame.empty else []
    rows_by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        keys = set()
        keys.update(_target_name_keys(row.get("target_label", "")))
        keys.update(_target_name_keys(row.get("target_column", "")))
        for key in keys:
            rows_by_key.setdefault(key, row)

    for target in requested_targets:
        keys = _target_name_keys(target)
        row = next((rows_by_key[key] for key in keys if key in rows_by_key), None)
        if row is None:
            report["issues"].append(
                {
                    "target": target,
                    "status": "missing_diagnostics",
                    "severity": "warning",
                    "message": f"Target {target!r} was requested but was not found in the model target diagnostics.",
                }
            )
            continue
        status = str(row.get("status") or "")
        matched = {
            "requested_target": target,
            "target_label": row.get("target_label"),
            "target_column": row.get("target_column"),
            "status": status,
            "range": row.get("range"),
            "nonzero_fraction": row.get("nonzero_fraction"),
            "r2": row.get("r2"),
            "reasons": row.get("reasons"),
        }
        report["matched_targets"].append(matched)
        if status != "recommended":
            report["issues"].append(
                {
                    "target": target,
                    "status": status,
                    "severity": "warning",
                    "message": f"Target {target!r} is {status}: {row.get('reasons')}",
                    "diagnostics": matched,
                }
            )

    report_path = out_dir / "target_availability_report.json"
    report["report_path"] = str(report_path)
    write_json(report_path, report)
    if policy == "error" and report["issues"]:
        messages = [str(issue["message"]) for issue in report["issues"]]
        raise ValueError("Target availability check failed: " + " | ".join(messages))
    return report


def _has_ordered_objectives(query_data: dict[str, Any]) -> bool:
    return bool(query_data.get("preferences")) or bool((query_data.get("ranking", {}) or {}).get("preferences")) or isinstance(
        query_data.get("objectives"), list
    )


def _search_top_k(query_data: dict[str, Any]) -> int:
    validation = dict(query_data.get("validation") or {})
    return int(validation.get("search_top_k") or query_data.get("search_top_k") or query_data.get("top_k") or 50)


def _selection_top_k(query_data: dict[str, Any]) -> int:
    validation = dict(query_data.get("validation") or {})
    return int(validation.get("top_k_xgems") or query_data.get("selection_top_k") or query_data.get("top_k") or 10)


def compile_design_query(
    *,
    query: str | Path,
    out: str | Path,
    model_table: str | Path | None = None,
    model_bundle: str | Path | None = None,
    model_registry: str | Path | None = None,
    reaction_model_id: str | None = None,
    reaction_model_signature: str | None = None,
    reaction_model_config: str | Path | None = None,
    target_availability_policy: Literal["ignore", "warn", "error"] = "warn",
) -> Path:
    """Compile an API/LLM-facing design query into executable search/selection YAML files."""
    query_path = Path(query)
    query_data = validate_design_query_data(
        load_yaml(query_path),
        model_table_override=model_table,
        model_bundle_override=model_bundle,
        model_registry=model_registry,
        reaction_model_id=reaction_model_id,
        reaction_model_signature=reaction_model_signature,
        reaction_model_config=reaction_model_config,
        require_model_paths=False,
    )
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    name = str(query_data.get("name") or query_path.stem)
    ranking = dict(query_data.get("ranking") or {})
    if _has_ordered_objectives(query_data) and not ranking.get("mode"):
        ranking["mode"] = "lexicographic"
    ranking.setdefault("mode", "weighted")

    reaction_request = resolve_reaction_model_request(
        query_data,
        reaction_model_id=reaction_model_id,
        reaction_model_signature=reaction_model_signature,
        reaction_model_config=reaction_model_config,
    )
    model_resolution = resolve_model_paths(
        query_data,
        model_table_override=model_table,
        model_bundle_override=model_bundle,
        model_registry=model_registry,
        reaction_model_id=reaction_request.get("id"),
        reaction_model_signature=reaction_request.get("signature"),
    )
    model_table_value = model_resolution["model_table"]
    model_bundle_value = model_resolution["model_bundle"]
    if not model_table_value or not model_bundle_value:
        raise ValueError(
            "Design query must include model_table/model_bundle, pass --model-table/--model-bundle, "
            "or match configs/design_query_model_registry.global_v1.yaml."
        )
    input_constraints = _normalise_input_constraints(query_data)
    target_constraints = _normalise_hard_targets(query_data)
    metadata_constraints = _normalise_metadata(query_data)
    prediction_errors = _normalise_prediction_errors(query_data)

    search_preferences = _normalise_preferences(query_data, target_section="predicted_targets")
    selection_preferences = _normalise_preferences(query_data, target_section="validated_targets")
    requested_targets = _requested_target_names(
        target_constraints=target_constraints,
        prediction_errors=prediction_errors,
        search_preferences=search_preferences,
        selection_preferences=selection_preferences,
    )
    target_availability = _check_target_availability(
        requested_targets=requested_targets,
        model_table=model_table_value,
        model_bundle=model_bundle_value,
        model_resolution=model_resolution,
        out_dir=out_dir,
        policy=target_availability_policy,
    )

    search_query = {
        "name": f"{name}_surrogate_search",
        "model_table": model_table_value,
        "model_bundle": model_bundle_value,
        "constraints": {
            "metadata": metadata_constraints,
            "inputs": input_constraints,
            "predicted_targets": target_constraints,
        },
        "ranking": ranking,
        "preferences": search_preferences,
        "top_k": _search_top_k(query_data),
    }
    reaction_block = _reaction_model_block(query_data, model_resolution.get("registry_entry"), reaction_request)
    if reaction_block:
        search_query["reaction_model"] = reaction_block
    selection_query = {
        "top_k": _selection_top_k(query_data),
        "constraints": {
            "chemistry_status": query_data.get("chemistry_status", "complete"),
            "metadata": metadata_constraints,
            "inputs": input_constraints,
            "validated_targets": target_constraints,
            "prediction_errors": prediction_errors,
        },
        "ranking": ranking,
        "preferences": selection_preferences,
    }

    if isinstance(query_data.get("objectives"), dict):
        search_query["objectives"] = query_data["objectives"]
        selection_query["objectives"] = query_data["objectives"]

    if not metadata_constraints:
        search_query["constraints"].pop("metadata")
        selection_query["constraints"].pop("metadata")
    if not input_constraints:
        search_query["constraints"].pop("inputs")
        selection_query["constraints"].pop("inputs")
    if not target_constraints:
        search_query["constraints"].pop("predicted_targets")
        selection_query["constraints"].pop("validated_targets")
    if not prediction_errors:
        selection_query["constraints"].pop("prediction_errors")
    if not search_preferences:
        search_query.pop("preferences")
    if not selection_preferences:
        selection_query.pop("preferences")

    _write_yaml(out_dir / "surrogate_candidate_search.yaml", search_query)
    _write_yaml(out_dir / "candidate_selection.yaml", selection_query)
    shutil.copy2(query_path, out_dir / "design_query_used.yaml")
    write_json(
        out_dir / "design_query_manifest.json",
        {
            "query": str(query_path),
            "out": str(out_dir),
            "name": name,
            "model_table": model_table_value,
            "model_bundle": model_bundle_value,
            "model_registry_entry": model_resolution.get("registry_entry"),
            "requested_reaction_model": reaction_request,
            "ranking": ranking,
            "design_space": query_data.get("design_space"),
            "output_constraints": query_data.get("output_constraints"),
            "validation": query_data.get("validation"),
            "reaction_model": reaction_block,
            "input_constraints": input_constraints,
            "target_constraints": target_constraints,
            "metadata_constraints": metadata_constraints,
            "prediction_errors": prediction_errors,
            "target_availability": target_availability,
            "search_query": str(out_dir / "surrogate_candidate_search.yaml"),
            "selection_query": str(out_dir / "candidate_selection.yaml"),
        },
    )
    return out_dir
