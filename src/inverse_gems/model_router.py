from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Literal

import yaml

from .design_query import (
    _is_auto_material_system,
    _normalise_hard_targets,
    _normalise_prediction_errors,
    _normalise_preferences,
    _query_age_days,
    _query_material_system,
    _requested_target_names,
    _target_name_keys,
    load_model_registry,
    resolve_reaction_model_request,
)
from .materials import BINDER_COMPONENTS, canonicalize_material_name
from .model_registry_diagnostics import diagnose_model_target_availability
from .utils import config_path, load_yaml, project_root, write_json


OPTIONAL_MATERIALS = {"gypsum"}
DEFAULT_AGE_DAYS = 28.0
RouteTargetPolicy = Literal["recommended", "allow_caution"]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _clean_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_")
    return re.sub(r"_+", "_", cleaned) or "unnamed"


def _registry_dir(registry: dict[str, Any]) -> Path:
    path = Path(str(registry.get("path") or config_path("design_query_model_registry.global_v1.yaml")))
    return path.parent if path.parent != Path("") else Path.cwd()


def _resolve_path(path_value: Any, *, base_dir: Path) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    for candidate in [base_dir / path, project_root() / path, Path.cwd() / path]:
        if candidate.exists():
            return candidate
    return project_root() / path


def _canonical_materials(values: list[Any] | None) -> list[str]:
    canonical: list[str] = []
    for value in values or []:
        name = canonicalize_material_name(str(value))
        if name != "water":
            canonical.append(name)
    return list(dict.fromkeys(canonical))


def _profile_allowed_materials(material_system: str, profiles: dict[str, Any]) -> list[str]:
    profile = profiles.get(material_system) or {}
    return _canonical_materials(profile.get("allowed") or [])


def _profile_bounds(material_system: str, profiles: dict[str, Any]) -> dict[str, Any]:
    profile = profiles.get(material_system) or {}
    return dict(profile.get("bounds") or {})


def _query_design_space(query_data: dict[str, Any]) -> dict[str, Any]:
    return dict(query_data.get("design_space") or {})


def _query_allowed_materials(query_data: dict[str, Any]) -> list[str]:
    design_space = _query_design_space(query_data)
    return _canonical_materials(design_space.get("allowed_materials") or [])


def _query_explicit_systems(query_data: dict[str, Any]) -> list[str]:
    systems = _as_list(_query_material_system(query_data))
    systems = [str(system) for system in systems if str(system)]
    if any(_is_auto_material_system(system) for system in systems):
        return []
    return systems


def _target_status_map(
    *,
    entry: dict[str, Any],
    registry_dir: Path,
    warnings: list[str],
) -> dict[str, dict[str, Any]]:
    model_table = entry.get("model_table")
    model_bundle = entry.get("model_bundle")
    if not model_table or not model_bundle:
        return {}
    frame, diag_warnings = diagnose_model_target_availability(
        model_table=_resolve_path(model_table, base_dir=registry_dir),
        model_bundle=_resolve_path(model_bundle, base_dir=registry_dir),
        registry_entry=entry,
        registry_dir=registry_dir,
    )
    warnings.extend(diag_warnings)
    by_key: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict(orient="records") if not frame.empty else []:
        keys = set()
        keys.update(_target_name_keys(row.get("target_label", "")))
        keys.update(_target_name_keys(row.get("target_column", "")))
        for key in keys:
            by_key.setdefault(key, row)
    return by_key


def _requested_targets(query_data: dict[str, Any]) -> list[str]:
    target_constraints = _normalise_hard_targets(query_data)
    prediction_errors = _normalise_prediction_errors(query_data)
    search_preferences = _normalise_preferences(query_data, target_section="predicted_targets")
    selection_preferences = _normalise_preferences(query_data, target_section="validated_targets")
    return _requested_target_names(
        target_constraints=target_constraints,
        prediction_errors=prediction_errors,
        search_preferences=search_preferences,
        selection_preferences=selection_preferences,
    )


def _target_support(
    *,
    requested_targets: list[str],
    status_by_key: dict[str, dict[str, Any]],
    target_policy: RouteTargetPolicy,
) -> tuple[int, list[str], list[dict[str, Any]]]:
    score = 0
    blockers: list[str] = []
    matched: list[dict[str, Any]] = []
    for target in requested_targets:
        row = next((status_by_key[key] for key in _target_name_keys(target) if key in status_by_key), None)
        if row is None:
            blockers.append(f"target {target!r} missing from diagnostics")
            matched.append({"target": target, "status": "missing"})
            continue
        status = str(row.get("status") or "")
        matched.append(
            {
                "target": target,
                "target_label": row.get("target_label"),
                "target_column": row.get("target_column"),
                "status": status,
                "r2": row.get("r2"),
                "range": row.get("range"),
                "reasons": row.get("reasons"),
            }
        )
        if status == "recommended":
            score += 3
        elif status == "usable_with_caution" and target_policy == "allow_caution":
            score += 1
        else:
            blockers.append(f"target {target!r} is {status or 'unknown'}: {row.get('reasons')}")
    return score, blockers, matched


def _material_match(
    *,
    system: str,
    profile_allowed: list[str],
    requested_allowed: list[str],
) -> tuple[int, list[str], list[str]]:
    if not requested_allowed:
        return (0, [], [])
    requested = set(requested_allowed)
    system_allowed = set(profile_allowed)
    required_by_system = system_allowed - OPTIONAL_MATERIALS
    effective_requested = set(requested)
    if "OPC" in required_by_system:
        effective_requested.add("OPC")
    extra_required = sorted(required_by_system - effective_requested)
    if extra_required:
        return (-10_000, [f"material system {system!r} requires unallowed material(s): {extra_required}"], extra_required)
    covered = sorted((required_by_system & effective_requested) - {"OPC"})
    unused_requested = sorted((effective_requested - OPTIONAL_MATERIALS) - system_allowed)
    # Prefer systems that use the non-OPC materials the user named, but keep
    # smaller systems eligible when the user named broad allowed materials.
    score = 20 * len(covered) - 3 * len(unused_requested) - len(system_allowed)
    if requested and (requested - OPTIONAL_MATERIALS) == (required_by_system - OPTIONAL_MATERIALS):
        score += 25
    return (score, [], unused_requested)


def _entry_matches_reaction(entry: dict[str, Any], request: dict[str, Any]) -> bool:
    request_id = request.get("id")
    request_signature = request.get("signature")
    if request_id and str(entry.get("reaction_model_id", "")) != str(request_id):
        return False
    if request_signature and str(entry.get("reaction_model_signature", "")) != str(request_signature):
        return False
    return True


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
    age = _clean_float(entry.get("age_days"))
    if age is None:
        return None
    tolerance = float(entry.get("age_tolerance", default_tolerance))
    delta = abs(age - requested_age)
    return delta if delta <= tolerance else None


def _candidate_rows(
    *,
    query_data: dict[str, Any],
    registry: dict[str, Any],
    profiles: dict[str, Any],
    target_policy: RouteTargetPolicy,
    default_age_days: float,
    reaction_model_id: str | None,
    reaction_model_signature: str | None,
    reaction_model_config: str | Path | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    registry_dir = _registry_dir(registry)
    entries = [dict(entry) for entry in registry.get("models") or registry.get("entries") or []]
    reaction_request = resolve_reaction_model_request(
        query_data,
        reaction_model_id=reaction_model_id,
        reaction_model_signature=reaction_model_signature,
        reaction_model_config=reaction_model_config,
    )
    requested_age = _clean_float(_query_age_days(query_data))
    if requested_age is None:
        requested_age = float(default_age_days)
    requested_allowed = _query_allowed_materials(query_data)
    explicit_systems = set(_query_explicit_systems(query_data))
    requested_targets = _requested_targets(query_data)
    default_tolerance = float(registry.get("default_age_tolerance", 1.0e-9))
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []

    for entry in entries:
        system = str(entry.get("material_system") or "")
        if not system:
            continue
        if explicit_systems and system not in explicit_systems:
            continue
        age_delta = _entry_age_delta(entry, requested_age, default_tolerance)
        if age_delta is None:
            continue
        if (reaction_request.get("id") or reaction_request.get("signature")) and not _entry_matches_reaction(entry, reaction_request):
            continue

        profile_allowed = _profile_allowed_materials(system, profiles)
        material_score, material_blockers, unused_requested = _material_match(
            system=system,
            profile_allowed=profile_allowed,
            requested_allowed=requested_allowed,
        )
        status_by_key = _target_status_map(entry=entry, registry_dir=registry_dir, warnings=warnings)
        target_score, target_blockers, matched_targets = _target_support(
            requested_targets=requested_targets,
            status_by_key=status_by_key,
            target_policy=target_policy,
        )
        blockers = material_blockers + target_blockers
        display_age = requested_age if entry.get("age_min_days") is not None or entry.get("age_max_days") is not None else _clean_float(entry.get("age_days"))
        row = {
            "id": entry.get("id"),
            "material_system": system,
            "age_days": display_age,
            "model_table": entry.get("model_table"),
            "model_bundle": entry.get("model_bundle"),
            "reaction_model_id": entry.get("reaction_model_id"),
            "reaction_model_signature": entry.get("reaction_model_signature"),
            "profile_allowed_materials": profile_allowed,
            "requested_allowed_materials": requested_allowed,
            "requested_targets": requested_targets,
            "matched_targets": matched_targets,
            "unused_requested_materials": unused_requested,
            "material_score": material_score,
            "target_score": target_score,
            "provenance_score": 5 if entry.get("reaction_model_signature") else 0,
            "age_score": -age_delta,
            "score": material_score + target_score + (5 if entry.get("reaction_model_signature") else 0) - age_delta,
            "eligible": not blockers,
            "blockers": blockers,
        }
        rows.append(row)
    rows.sort(key=lambda row: (bool(row["eligible"]), float(row["score"]), str(row.get("id") or "")), reverse=True)
    return rows, warnings


def _merge_profile_defaults(query_data: dict[str, Any], *, selected_system: str, profiles: dict[str, Any]) -> dict[str, Any]:
    routed = copy.deepcopy(query_data)
    design_space = dict(routed.get("design_space") or {})
    profile_allowed = _profile_allowed_materials(selected_system, profiles)
    requested_allowed = _query_allowed_materials(routed)
    if requested_allowed:
        strict_materials = bool(design_space.get("strict_materials", False))
        allowed = requested_allowed if strict_materials else list(dict.fromkeys(requested_allowed + [name for name in profile_allowed if name in OPTIONAL_MATERIALS]))
    else:
        allowed = profile_allowed
    if allowed:
        design_space["allowed_materials"] = allowed

    bounds = _profile_bounds(selected_system, profiles)
    input_constraints = dict(design_space.get("input_constraints") or {})
    allowed_set = set(allowed)
    strict_materials = bool(design_space.get("strict_materials", False))
    for name, pair in bounds.items():
        canonical = canonicalize_material_name(str(name)) if str(name) in BINDER_COMPONENTS else str(name)
        if strict_materials and allowed_set and canonical not in allowed_set:
            continue
        if canonical not in input_constraints and isinstance(pair, list) and len(pair) == 2:
            input_constraints[canonical] = {"min": float(pair[0]), "max": float(pair[1])}
    if "w_b" not in input_constraints:
        default_w_b = (profiles.get(selected_system) or {}).get("default_w_b")
        if isinstance(default_w_b, list) and len(default_w_b) == 2:
            input_constraints["w_b"] = {"min": float(default_w_b[0]), "max": float(default_w_b[1])}
    if input_constraints:
        design_space["input_constraints"] = input_constraints
    if design_space.get("age_days") is None and routed.get("age_days") is None:
        design_space["age_days"] = float((profiles.get(selected_system) or {}).get("default_age_days") or DEFAULT_AGE_DAYS)

    design_space["material_systems"] = selected_system
    routed["design_space"] = design_space
    routed["material_system"] = selected_system
    if routed.get("age_days") is None:
        age_value = design_space.get("age_days")
        if not isinstance(age_value, dict):
            routed["age_days"] = float(age_value)
    return routed


def _route_selection_explanation(
    *,
    selected: dict[str, Any],
    candidates: list[dict[str, Any]],
    target_policy: RouteTargetPolicy,
    default_age_days: float,
) -> dict[str, Any]:
    requested_targets = list(selected.get("requested_targets") or [])
    requested_materials = list(selected.get("requested_allowed_materials") or [])
    top_alternatives = []
    for row in candidates[:5]:
        top_alternatives.append(
            {
                "id": row.get("id"),
                "material_system": row.get("material_system"),
                "eligible": row.get("eligible"),
                "score": row.get("score"),
                "material_score": row.get("material_score"),
                "target_score": row.get("target_score"),
                "provenance_score": row.get("provenance_score"),
                "age_score": row.get("age_score"),
                "blockers": row.get("blockers"),
                "unused_requested_materials": row.get("unused_requested_materials"),
            }
        )
    target_support = []
    for target in selected.get("matched_targets") or []:
        target_support.append(
            {
                "target": target.get("target"),
                "status": target.get("status"),
                "target_label": target.get("target_label"),
                "target_column": target.get("target_column"),
                "r2": target.get("r2"),
                "range": target.get("range"),
                "reasons": target.get("reasons"),
            }
        )
    reason_parts = [
        f"Selected {selected.get('id') or selected.get('material_system')} because it was the highest-ranked eligible model",
        f"for material system {selected.get('material_system')!r}",
        f"at age {selected.get('age_days')} days.",
    ]
    if requested_materials:
        reason_parts.append(f"Requested materials were {requested_materials}.")
    if requested_targets:
        reason_parts.append(f"Requested targets were {requested_targets}.")
    return {
        "summary": " ".join(reason_parts),
        "selection_rules": [
            "Only eligible registry entries can be selected.",
            "Candidate score is material_score + target_score + provenance_score + age_score.",
            "material_score favors material systems that use the requested non-OPC materials without requiring unallowed materials.",
            "target_score favors requested targets whose diagnostics are recommended, or usable with caution only when target_policy=allow_caution.",
            "provenance_score favors entries with an explicit reaction-model signature.",
            f"If age_days is omitted, routing uses default_age_days={default_age_days:g}.",
        ],
        "target_policy": target_policy,
        "requested_materials": requested_materials,
        "requested_targets": requested_targets,
        "selected": {
            "id": selected.get("id"),
            "material_system": selected.get("material_system"),
            "age_days": selected.get("age_days"),
            "score": selected.get("score"),
            "material_score": selected.get("material_score"),
            "target_score": selected.get("target_score"),
            "provenance_score": selected.get("provenance_score"),
            "age_score": selected.get("age_score"),
            "profile_allowed_materials": selected.get("profile_allowed_materials"),
            "unused_requested_materials": selected.get("unused_requested_materials"),
        },
        "target_support": target_support,
        "top_alternatives": top_alternatives,
    }


def route_design_query(
    query_data: dict[str, Any],
    *,
    model_registry: str | Path | None = None,
    material_systems_config: str | Path | None = None,
    reaction_model_id: str | None = None,
    reaction_model_signature: str | None = None,
    reaction_model_config: str | Path | None = None,
    target_policy: RouteTargetPolicy = "recommended",
    default_age_days: float = DEFAULT_AGE_DAYS,
) -> dict[str, Any]:
    """Select a registry model/material system for an API-facing design query."""
    registry = load_model_registry(model_registry)
    profiles = load_yaml(material_systems_config or config_path("material_systems.yaml"))
    candidates, warnings = _candidate_rows(
        query_data=query_data,
        registry=registry,
        profiles=profiles,
        target_policy=target_policy,
        default_age_days=default_age_days,
        reaction_model_id=reaction_model_id,
        reaction_model_signature=reaction_model_signature,
        reaction_model_config=reaction_model_config,
    )
    eligible = [row for row in candidates if row["eligible"]]
    if not eligible:
        raise ValueError(
            "No eligible material-system model found for the design query. "
            f"Candidates checked: {candidates[:10]}"
        )
    selected = eligible[0]
    routed_query = _merge_profile_defaults(query_data, selected_system=str(selected["material_system"]), profiles=profiles)
    if selected.get("id") and not routed_query.get("model_id"):
        routed_query["model_id"] = str(selected["id"])
    explanation = _route_selection_explanation(
        selected=selected,
        candidates=candidates,
        target_policy=target_policy,
        default_age_days=default_age_days,
    )
    return {
        "selected": selected,
        "routed_query": routed_query,
        "candidates": candidates,
        "warnings": warnings,
        "target_policy": target_policy,
        "default_age_days": default_age_days,
        "selection_explanation": explanation,
    }


def route_design_query_file(
    *,
    query: str | Path,
    out: str | Path,
    model_registry: str | Path | None = None,
    material_systems_config: str | Path | None = None,
    reaction_model_id: str | None = None,
    reaction_model_signature: str | None = None,
    reaction_model_config: str | Path | None = None,
    target_policy: RouteTargetPolicy = "recommended",
    default_age_days: float = DEFAULT_AGE_DAYS,
) -> Path:
    query_path = Path(query)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = route_design_query(
        load_yaml(query_path),
        model_registry=model_registry,
        material_systems_config=material_systems_config,
        reaction_model_id=reaction_model_id,
        reaction_model_signature=reaction_model_signature,
        reaction_model_config=reaction_model_config,
        target_policy=target_policy,
        default_age_days=default_age_days,
    )
    routed_path = out_dir / "routed_design_query.yaml"
    with routed_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(result["routed_query"], handle, sort_keys=False)
    write_json(out_dir / "model_route_report.json", result)
    return out_dir


# ---------------------------------------------------------------------------
# Public aliases for external orchestration layers (e.g. GemsPilot).
# The underscore-prefixed names remain for internal use; these aliases are
# the supported cross-package surface for feasibility diagnosis.
# ---------------------------------------------------------------------------
candidate_rows = _candidate_rows
query_age_days = _query_age_days
query_allowed_materials = _query_allowed_materials
query_design_space = _query_design_space
query_explicit_systems = _query_explicit_systems
