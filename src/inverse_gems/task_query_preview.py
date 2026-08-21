from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import yaml

from .forward_query import expand_age_grid
from .model_router import route_design_query
from .task_query import TaskQuerySpec, run_task_query
from .utils import load_yaml, write_json


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _risk(level: str, code: str, message: str) -> dict[str, str]:
    return {"level": level, "code": code, "message": message}


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compact_values(values: list[float], limit: int = 12) -> dict[str, Any]:
    if len(values) <= limit:
        return {"values": values, "count": len(values)}
    return {
        "count": len(values),
        "first": values[0],
        "last": values[-1],
        "sample": values[: min(5, len(values))] + values[-min(5, len(values)) :],
    }


def _binder_masses(recipe: dict[str, Any]) -> dict[str, float]:
    binders: dict[str, float] = {}
    for name, value in (recipe.get("binders") or {}).items():
        binders[str(name)] = binders.get(str(name), 0.0) + float(value)
    for name in ["OPC", "slag", "fly_ash", "metakaolin", "silica_fume", "limestone", "gypsum"]:
        if name in recipe:
            binders[name] = binders.get(name, 0.0) + float(recipe[name])
    return {name: value for name, value in binders.items() if abs(value) > 0.0}


def _output_names(value: Any) -> str | list[str]:
    if value is None:
        return "all"
    return value


def _forward_preview(forward_query: dict[str, Any], risks: list[dict[str, str]]) -> dict[str, Any]:
    recipe = dict(forward_query.get("recipe") or {})
    binders = _binder_masses(recipe)
    total_binder = sum(binders.values())
    if total_binder and abs(total_binder - 100.0) > 1.0e-6:
        risks.append(_risk("warning", "binder_sum_not_100", f"Binder masses sum to {total_binder:g} g, not 100 g."))
    water_mode = "w_b" if recipe.get("w_b") is not None else "water_g" if recipe.get("water_g") is not None else "missing"
    if water_mode == "missing":
        risks.append(_risk("error", "water_missing", "Forward query has no w_b or water_g."))

    age_grid = dict(forward_query.get("age_grid") or {})
    try:
        age_values = expand_age_grid(age_grid)
        age_summary = _compact_values(age_values)
    except Exception as exc:
        risks.append(_risk("error", "age_grid_invalid", str(exc)))
        age_summary = {"error": str(exc)}

    outputs = dict(forward_query.get("outputs") or {})
    response_summary = dict(forward_query.get("response_summary") or {})
    plots = list(forward_query.get("plots") or [])
    return {
        "name": forward_query.get("name"),
        "task": forward_query.get("task"),
        "material_system": forward_query.get("material_system"),
        "recipe": {
            "binders": binders,
            "binder_total_g": total_binder,
            "water_mode": water_mode,
            "water_value": recipe.get(water_mode) if water_mode != "missing" else None,
        },
        "age_grid": age_summary,
        "temperature_celsius": forward_query.get("temperature_celsius", 20.0),
        "outputs": {
            "phase_masses": _output_names(outputs.get("phase_masses")),
            "phase_volumes": _output_names(outputs.get("phase_volumes")),
            "phase_volumes_reconstructed": _output_names(outputs.get("phase_volumes_reconstructed")),
            "aqueous_species": _output_names(outputs.get("aqueous_species")),
            "scalars": _output_names(outputs.get("scalars")),
        },
        "plots": plots,
        "response_summary": {
            "phases": list(response_summary.get("phases") or []),
            "scalars": list(response_summary.get("scalars") or []),
            "top_phases": response_summary.get("top_phases"),
            "table_limit": response_summary.get("table_limit"),
            "narrative_enabled": response_summary.get("narrative_enabled", True),
            "narrative_language": response_summary.get("narrative_language", "ko"),
        },
    }


def _design_space_age(design_query: dict[str, Any]) -> Any:
    if design_query.get("age_days") is not None:
        return design_query.get("age_days")
    design_space = dict(design_query.get("design_space") or {})
    return design_space.get("age_days")


def _design_material_system(design_query: dict[str, Any]) -> Any:
    if design_query.get("material_system") is not None:
        return design_query.get("material_system")
    design_space = dict(design_query.get("design_space") or {})
    return design_space.get("material_systems")


def _design_targets(design_query: dict[str, Any]) -> dict[str, Any]:
    targets: dict[str, Any] = {}
    for key in ["output_constraints", "targets", "prediction_errors"]:
        if design_query.get(key):
            targets[key] = design_query[key]
    validation = dict(design_query.get("validation") or {})
    if validation.get("prediction_errors"):
        targets["validation.prediction_errors"] = validation["prediction_errors"]
    constraints = dict(design_query.get("constraints") or {})
    hard_constraints = dict(design_query.get("hard_constraints") or {})
    for group_name, group in [("constraints", constraints), ("hard_constraints", hard_constraints)]:
        if group.get("targets"):
            targets[f"{group_name}.targets"] = group["targets"]
        if group.get("predicted_targets"):
            targets[f"{group_name}.predicted_targets"] = group["predicted_targets"]
        if group.get("validated_targets"):
            targets[f"{group_name}.validated_targets"] = group["validated_targets"]
    return targets


def _requested_target_names(design_query: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for section in _design_targets(design_query).values():
        if isinstance(section, dict):
            names.extend(str(name) for name in section)
    for item in _as_list(design_query.get("preferences")) + _as_list(design_query.get("objectives")):
        if isinstance(item, dict) and item.get("target"):
            names.append(str(item["target"]))
    ranking = dict(design_query.get("ranking") or {})
    for item in _as_list(ranking.get("preferences")):
        if isinstance(item, dict) and item.get("target"):
            names.append(str(item["target"]))
    return list(dict.fromkeys(names))


def _target_is_pH(name: Any) -> bool:
    text = str(name).strip().lower().replace("-", "_")
    safe = re.sub(r"[^0-9a-z_]+", "_", text).strip("_")
    return text in {"ph", "p_h", "scalar__ph", "y__ph"} or safe in {"ph", "p_h", "scalar__ph", "y__ph"} or safe.endswith("__ph")


def _pH_uncertainty_policy_preview(design_query: dict[str, Any]) -> dict[str, Any]:
    constraint_sections = _design_targets(design_query)
    pH_as_constraint = any(
        _target_is_pH(name)
        for section in constraint_sections.values()
        if isinstance(section, dict)
        for name in section
    )
    pH_in_ranking = False
    for item in _as_list(design_query.get("preferences")) + _as_list(design_query.get("objectives")):
        if isinstance(item, dict) and item.get("target") is not None and _target_is_pH(item.get("target")):
            pH_in_ranking = True
    ranking = dict(design_query.get("ranking") or {})
    for item in _as_list(ranking.get("preferences")):
        if isinstance(item, dict) and item.get("target") is not None and _target_is_pH(item.get("target")):
            pH_in_ranking = True

    requested = dict(design_query.get("pH_uncertainty_policy") or {})
    requested_mode = str(requested.get("mode") or "auto").lower()
    if requested_mode == "auto":
        resolved_mode = "exclude" if pH_as_constraint else "penalize" if pH_in_ranking else "ignore"
    else:
        resolved_mode = requested_mode
    pH_used = pH_as_constraint or pH_in_ranking
    return {
        "enabled": bool(requested.get("enabled", True)),
        "requested_mode": requested_mode,
        "resolved_mode": resolved_mode if pH_used else "ignore",
        "pH_used": pH_used,
        "pH_used_as_constraint": pH_as_constraint,
        "pH_used_in_ranking": pH_in_ranking,
        "effect": (
            "pH-water-unreliable candidates will be excluded during final selection"
            if pH_used and resolved_mode == "exclude"
            else "pH-water-unreliable candidates will receive a ranking penalty during final selection"
            if pH_used and resolved_mode == "penalize"
            else "no pH-specific uncertainty adjustment is expected"
        ),
    }


def _routing_preview(
    design_query: dict[str, Any],
    *,
    risks: list[dict[str, str]],
    model_registry: str | Path | None,
    material_systems_config: str | Path | None,
    route_target_policy: str,
    reaction_model_id: str | None,
    reaction_model_signature: str | None,
    reaction_model_config: str | Path | None,
    default_age_days: float,
) -> dict[str, Any]:
    if design_query.get("model_id"):
        return {"attempted": False, "status": "explicit_model_id", "model_id": design_query.get("model_id")}
    if design_query.get("model_table") and design_query.get("model_bundle"):
        return {
            "attempted": False,
            "status": "explicit_model_paths",
            "model_table": design_query.get("model_table"),
            "model_bundle": design_query.get("model_bundle"),
        }
    try:
        route = route_design_query(
            design_query,
            model_registry=model_registry,
            material_systems_config=material_systems_config,
            reaction_model_id=reaction_model_id,
            reaction_model_signature=reaction_model_signature,
            reaction_model_config=reaction_model_config,
            target_policy=route_target_policy,  # type: ignore[arg-type]
            default_age_days=default_age_days,
        )
    except Exception as exc:
        risks.append(_risk("error", "model_route_failed", str(exc)))
        return {"attempted": True, "status": "failed", "error": str(exc)}

    selected = dict(route.get("selected") or {})
    for target in selected.get("matched_targets") or []:
        status = str(target.get("status") or "")
        if status == "usable_with_caution":
            risks.append(
                _risk(
                    "warning",
                    "target_usable_with_caution",
                    f"Target {target.get('target')!r} is usable with caution: {target.get('reasons')}",
                )
            )
        elif status and status != "recommended":
            risks.append(
                _risk(
                    "error",
                    "target_not_recommended",
                    f"Target {target.get('target')!r} is {status}: {target.get('reasons')}",
                )
            )
    top_candidates = []
    for row in (route.get("candidates") or [])[:5]:
        top_candidates.append(
            {
                "id": row.get("id"),
                "material_system": row.get("material_system"),
                "age_days": row.get("age_days"),
                "eligible": row.get("eligible"),
                "score": row.get("score"),
                "material_score": row.get("material_score"),
                "target_score": row.get("target_score"),
                "provenance_score": row.get("provenance_score"),
                "age_score": row.get("age_score"),
                "blockers": row.get("blockers"),
            }
        )
    return {
        "attempted": True,
        "status": "selected",
        "target_policy": route.get("target_policy"),
        "selected": {
            "id": selected.get("id"),
            "material_system": selected.get("material_system"),
            "age_days": selected.get("age_days"),
            "model_table": selected.get("model_table"),
            "model_bundle": selected.get("model_bundle"),
            "reaction_model_id": selected.get("reaction_model_id"),
            "reaction_model_signature": selected.get("reaction_model_signature"),
            "matched_targets": selected.get("matched_targets"),
        },
        "candidate_count": len(route.get("candidates") or []),
        "top_candidates": top_candidates,
        "selection_explanation": route.get("selection_explanation") or {},
        "warnings": route.get("warnings") or [],
        "routed_query": route.get("routed_query"),
    }


def _inverse_preview(
    design_query: dict[str, Any],
    *,
    risks: list[dict[str, str]],
    model_registry: str | Path | None,
    material_systems_config: str | Path | None,
    route_target_policy: str,
    reaction_model_id: str | None,
    reaction_model_signature: str | None,
    reaction_model_config: str | Path | None,
    default_age_days: float,
) -> dict[str, Any]:
    design_space = dict(design_query.get("design_space") or {})
    age = _design_space_age(design_query)
    if age is None:
        risks.append(_risk("warning", "age_defaulted", f"No age_days supplied; routing preview uses {default_age_days:g} days."))
    material_system = _design_material_system(design_query)
    if not material_system:
        risks.append(_risk("info", "material_system_auto", "No material_system supplied; local routing will choose one."))
    targets = _design_targets(design_query)
    target_names = _requested_target_names(design_query)
    pH_policy = _pH_uncertainty_policy_preview(design_query)
    if pH_policy["pH_used"]:
        risks.append(
            _risk(
                "warning",
                "ph_uncertainty_policy",
                f"pH was requested; {pH_policy['effect']}.",
            )
        )
    routing = _routing_preview(
        design_query,
        risks=risks,
        model_registry=model_registry,
        material_systems_config=material_systems_config,
        route_target_policy=route_target_policy,
        reaction_model_id=reaction_model_id,
        reaction_model_signature=reaction_model_signature,
        reaction_model_config=reaction_model_config,
        default_age_days=default_age_days,
    )
    return {
        "name": design_query.get("name"),
        "material_system": material_system,
        "allowed_materials": list(design_space.get("allowed_materials") or []),
        "age_days": age,
        "age_bin": design_query.get("age_bin") or design_space.get("age_bin"),
        "input_constraints": design_query.get("inputs") or design_space.get("input_constraints") or {},
        "target_constraints": targets,
        "requested_targets": target_names,
        "preferences": design_query.get("preferences") or design_query.get("objectives") or [],
        "ranking": design_query.get("ranking") or {},
        "validation": design_query.get("validation") or {},
        "pH_uncertainty_policy": pH_policy,
        "model_routing": routing,
    }


def preview_task_query_data(
    data: dict[str, Any],
    *,
    model_registry: str | Path | None = None,
    material_systems_config: str | Path | None = None,
    route_target_policy: str = "recommended",
    reaction_model_id: str | None = None,
    reaction_model_signature: str | None = None,
    reaction_model_config: str | Path | None = None,
    default_age_days: float = 28.0,
) -> dict[str, Any]:
    risks: list[dict[str, str]] = []
    try:
        spec = TaskQuerySpec.model_validate(data)
    except Exception as exc:
        return {
            "valid": False,
            "status": "invalid",
            "error": str(exc),
            "risks": [_risk("error", "schema_invalid", str(exc))],
        }

    payload = spec.model_dump(mode="json", exclude_none=True)
    result: dict[str, Any] = {
        "valid": True,
        "status": "preview_complete",
        "task_type": spec.task_type,
        "name": spec.name,
        "user_request": spec.user_request,
        "will_execute": False,
        "risks": risks,
    }
    if spec.task_type in {"forward_calculation", "forward_time_series"}:
        result["execution_mode"] = "forward"
        result["requires_dat_lst_for_real_run"] = True
        result["forward_query"] = _forward_preview(payload["forward_query"], risks)
    else:
        result["execution_mode"] = "inverse_design"
        result["requires_dat_lst_for_validation"] = True
        result["inverse_design"] = _inverse_preview(
            payload["design_query"],
            risks=risks,
            model_registry=model_registry,
            material_systems_config=material_systems_config,
            route_target_policy=route_target_policy,
            reaction_model_id=reaction_model_id,
            reaction_model_signature=reaction_model_signature,
            reaction_model_config=reaction_model_config,
            default_age_days=default_age_days,
        )
    if any(item["level"] == "error" for item in risks):
        result["status"] = "preview_with_errors"
    elif risks:
        result["status"] = "preview_with_warnings"
    return result


def _render_dict_items(values: dict[str, Any]) -> list[str]:
    lines = []
    for key, value in values.items():
        lines.append(f"- `{key}`: `{value}`")
    return lines


def _render_target_support(targets: list[dict[str, Any]]) -> list[str]:
    if not targets:
        return ["- None"]
    lines = []
    for target in targets:
        parts = [
            f"`{target.get('target')}`",
            f"status `{target.get('status')}`",
        ]
        if target.get("r2") is not None:
            parts.append(f"R2 `{target.get('r2')}`")
        if target.get("target_column"):
            parts.append(f"column `{target.get('target_column')}`")
        lines.append("- " + ", ".join(parts))
        if target.get("reasons"):
            lines.append(f"  - Reasons: `{target.get('reasons')}`")
    return lines


def _render_candidate_rows(candidates: list[dict[str, Any]]) -> list[str]:
    if not candidates:
        return ["- None"]
    lines = []
    for candidate in candidates:
        lines.append(
            "- "
            f"`{candidate.get('id')}` / `{candidate.get('material_system')}`: "
            f"eligible `{candidate.get('eligible')}`, "
            f"score `{candidate.get('score')}`, "
            f"material `{candidate.get('material_score')}`, "
            f"target `{candidate.get('target_score')}`, "
            f"provenance `{candidate.get('provenance_score')}`, "
            f"age `{candidate.get('age_score')}`"
        )
        if candidate.get("blockers"):
            lines.append(f"  - Blockers: `{candidate.get('blockers')}`")
    return lines


def render_task_query_preview_markdown(preview: dict[str, Any]) -> str:
    lines = ["# inverse_gems task query preview", ""]
    lines.append(f"- Status: `{preview.get('status')}`")
    lines.append(f"- Valid schema: `{preview.get('valid')}`")
    lines.append(f"- Task type: `{preview.get('task_type')}`")
    if preview.get("name"):
        lines.append(f"- Name: `{preview.get('name')}`")
    if preview.get("user_request"):
        lines.append(f"- User request: {preview.get('user_request')}")
    lines.append("")

    if preview.get("forward_query"):
        forward = preview["forward_query"]
        lines.extend(["## Forward query", ""])
        recipe = forward.get("recipe") or {}
        lines.append(f"- Material system: `{forward.get('material_system')}`")
        lines.append(f"- Binder total: `{recipe.get('binder_total_g')}` g")
        lines.append(f"- Water: `{recipe.get('water_mode')}` = `{recipe.get('water_value')}`")
        lines.append(f"- Age grid: `{forward.get('age_grid')}`")
        lines.append(f"- Temperature: `{forward.get('temperature_celsius')}` C")
        lines.append("")
        lines.extend(["### Binders", ""])
        lines.extend(_render_dict_items(recipe.get("binders") or {}) or ["- None"])
        lines.extend(["", "### Requested response", ""])
        response = forward.get("response_summary") or {}
        lines.append(f"- Phases: `{response.get('phases')}`")
        lines.append(f"- Scalars: `{response.get('scalars')}`")
        lines.append(f"- Narrative: `{response.get('narrative_enabled')}` / `{response.get('narrative_language')}`")
        lines.append("")

    if preview.get("inverse_design"):
        inverse = preview["inverse_design"]
        routing = inverse.get("model_routing") or {}
        selected = routing.get("selected") or {}
        lines.extend(["## Inverse design", ""])
        lines.append(f"- Material system: `{inverse.get('material_system')}`")
        lines.append(f"- Allowed materials: `{inverse.get('allowed_materials')}`")
        lines.append(f"- Age days: `{inverse.get('age_days')}`")
        lines.append(f"- Requested targets: `{inverse.get('requested_targets')}`")
        pH_policy = inverse.get("pH_uncertainty_policy") or {}
        if pH_policy:
            lines.append(f"- pH uncertainty policy: `{pH_policy.get('resolved_mode')}`")
            lines.append(f"- pH policy effect: {pH_policy.get('effect')}")
        lines.append("")
        lines.extend(["### Input constraints", ""])
        lines.extend(_render_dict_items(inverse.get("input_constraints") or {}) or ["- None"])
        lines.extend(["", "### Model routing", ""])
        lines.append(f"- Status: `{routing.get('status')}`")
        lines.append(f"- Selected model: `{selected.get('id')}`")
        lines.append(f"- Selected material system: `{selected.get('material_system')}`")
        lines.append(f"- Candidate count: `{routing.get('candidate_count')}`")
        explanation = routing.get("selection_explanation") or {}
        explanation_selected = explanation.get("selected") or {}
        if explanation:
            lines.append(f"- Selection summary: {explanation.get('summary')}")
            lines.append(
                "- Score breakdown: "
                f"total `{explanation_selected.get('score')}`, "
                f"material `{explanation_selected.get('material_score')}`, "
                f"target `{explanation_selected.get('target_score')}`, "
                f"provenance `{explanation_selected.get('provenance_score')}`, "
                f"age `{explanation_selected.get('age_score')}`"
            )
            lines.extend(["", "### Target support", ""])
            lines.extend(_render_target_support(list(explanation.get("target_support") or [])))
            lines.extend(["", "### Top routing candidates", ""])
            lines.extend(_render_candidate_rows(list(explanation.get("top_alternatives") or routing.get("top_candidates") or [])))
        lines.append("")

    lines.extend(["## Risks and notes", ""])
    risks = preview.get("risks") or []
    if risks:
        for item in risks:
            lines.append(f"- `{item.get('level')}` `{item.get('code')}`: {item.get('message')}")
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def write_task_query_preview(preview: dict[str, Any], out: str | Path) -> Path:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "parsed_query_preview.json", preview)
    (out_dir / "parsed_query_preview.md").write_text(render_task_query_preview_markdown(preview), encoding="utf-8")
    return out_dir


def preview_task_query_file(
    *,
    query: str | Path,
    out: str | Path,
    model_registry: str | Path | None = None,
    material_systems_config: str | Path | None = None,
    route_target_policy: str = "recommended",
    reaction_model_id: str | None = None,
    reaction_model_signature: str | None = None,
    reaction_model_config: str | Path | None = None,
    default_age_days: float = 28.0,
) -> Path:
    query_path = Path(query)
    preview = preview_task_query_data(
        load_yaml(query_path),
        model_registry=model_registry,
        material_systems_config=material_systems_config,
        route_target_policy=route_target_policy,
        reaction_model_id=reaction_model_id,
        reaction_model_signature=reaction_model_signature,
        reaction_model_config=reaction_model_config,
        default_age_days=default_age_days,
    )
    preview["source"] = {
        "task_query": str(query_path),
        "task_query_sha256": _file_sha256(query_path),
    }
    out_dir = write_task_query_preview(preview, out)
    target_query = out_dir / "task_query.yaml"
    if query_path.resolve(strict=False) != target_query.resolve(strict=False):
        shutil.copy2(query_path, target_query)
    return out_dir


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _preview_risk_counts(preview: dict[str, Any]) -> dict[str, int]:
    counts = {"error": 0, "warning": 0, "info": 0}
    for item in preview.get("risks") or []:
        level = str(item.get("level") or "info")
        counts[level] = counts.get(level, 0) + 1
    return counts


def _validate_confirmed_preview(
    *,
    preview_dir: Path,
    confirmed: bool,
    allow_preview_errors: bool,
    fail_on_preview_warnings: bool,
) -> tuple[Path, dict[str, Any], dict[str, int], list[str]]:
    if not confirmed:
        raise ValueError("Confirmed execution requires confirmed=True or --confirm-preview.")
    query_path = preview_dir / "task_query.yaml"
    preview_path = preview_dir / "parsed_query_preview.json"
    if not query_path.exists():
        raise FileNotFoundError(f"Missing confirmed task query: {query_path}")
    if not preview_path.exists():
        raise FileNotFoundError(f"Missing parsed query preview: {preview_path}")

    preview = _load_json(preview_path)
    counts = _preview_risk_counts(preview)
    warnings: list[str] = []
    if not preview.get("valid"):
        raise ValueError("Preview is not schema-valid; refusing confirmed execution.")
    if counts.get("error", 0) and not allow_preview_errors:
        raise ValueError("Preview contains error-level risks; rerun with allow_preview_errors=True only if intentional.")
    if counts.get("warning", 0) and fail_on_preview_warnings:
        raise ValueError("Preview contains warning-level risks and fail_on_preview_warnings=True.")

    source = dict(preview.get("source") or {})
    expected_hash = source.get("task_query_sha256")
    if expected_hash:
        current_hash = _file_sha256(query_path)
        if current_hash != expected_hash:
            raise ValueError("task_query.yaml does not match the reviewed preview hash.")
    else:
        warnings.append("Preview has no task_query hash; cannot verify whether task_query.yaml changed after preview.")
    return query_path, preview, counts, warnings


def run_confirmed_task_query(
    *,
    preview_dir: str | Path,
    out: str | Path,
    db: str | Path,
    confirmed: bool = False,
    allow_preview_errors: bool = False,
    fail_on_preview_warnings: bool = False,
    dat_lst: str | Path | None = None,
    model_table: str | Path | None = None,
    model_bundle: str | Path | None = None,
    global_db: str | Path | None = None,
    global_n_candidates: int = 500,
    strict_materials: bool | None = None,
    model_registry: str | Path | None = None,
    material_systems_config: str | Path | None = None,
    route_target_policy: str = "recommended",
    use_mock: bool = False,
    skip_validation: bool = False,
    validation_top_k: int | None = None,
    output_selection: str | Path | None = None,
    run_mode: str = "reacted_only",
    normalize: bool = True,
    allow_non_100: bool = False,
    temperature_celsius: float | None = None,
    pressure: float | None = None,
    force_rerun_xgems: bool = False,
    gems_class_path: str | None = "xgems:ChemicalEngineDicts",
    xgems_input_mode: str = "formula",
    xgems_water_mode: str = "initial",
    xgems_water_factor: float = 1.0,
    xgems_water_g: float | None = None,
    xgems_water_w_b: float | None = None,
    retry_water_on_failure: bool = False,
    retry_water_cap_w_b_ladder: list[float] | tuple[float, ...] | str = (0.45, 0.40, 0.35, 0.30),
    retry_water_up_w_b_ladder: list[float] | tuple[float, ...] | str = (0.30, 0.35, 0.40, 0.45),
    retry_water_min_w_b: float = 0.30,
    reaction_model_id: str | None = None,
    reaction_model_signature: str | None = None,
    reaction_model_config: str | Path | None = None,
    disable_plots: bool = False,
    fail_fast: bool = False,
) -> Path:
    preview_path = Path(preview_dir)
    query_path, preview, risk_counts, preview_warnings = _validate_confirmed_preview(
        preview_dir=preview_path,
        confirmed=confirmed,
        allow_preview_errors=allow_preview_errors,
        fail_on_preview_warnings=fail_on_preview_warnings,
    )
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = out_dir / "confirmed_preview"
    audit_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(query_path, audit_dir / "task_query.yaml")
    shutil.copy2(preview_path / "parsed_query_preview.json", audit_dir / "parsed_query_preview.json")
    markdown = preview_path / "parsed_query_preview.md"
    if markdown.exists():
        shutil.copy2(markdown, audit_dir / "parsed_query_preview.md")

    run_dir = run_task_query(
        query=query_path,
        out=out_dir,
        db=db,
        dat_lst=dat_lst,
        model_table=model_table,
        model_bundle=model_bundle,
        global_db=global_db,
        global_n_candidates=global_n_candidates,
        strict_materials=strict_materials,
        model_registry=model_registry,
        material_systems_config=material_systems_config,
        route_target_policy=route_target_policy,
        use_mock=use_mock,
        skip_validation=skip_validation,
        validation_top_k=validation_top_k,
        output_selection=output_selection,
        run_mode=run_mode,
        normalize=normalize,
        allow_non_100=allow_non_100,
        temperature_celsius=temperature_celsius,
        pressure=pressure,
        force_rerun_xgems=force_rerun_xgems,
        gems_class_path=gems_class_path,
        xgems_input_mode=xgems_input_mode,
        xgems_water_mode=xgems_water_mode,
        xgems_water_factor=xgems_water_factor,
        xgems_water_g=xgems_water_g,
        xgems_water_w_b=xgems_water_w_b,
        retry_water_on_failure=retry_water_on_failure,
        retry_water_cap_w_b_ladder=retry_water_cap_w_b_ladder,
        retry_water_up_w_b_ladder=retry_water_up_w_b_ladder,
        retry_water_min_w_b=retry_water_min_w_b,
        reaction_model_id=reaction_model_id,
        reaction_model_signature=reaction_model_signature,
        reaction_model_config=reaction_model_config,
        disable_plots=disable_plots,
        fail_fast=fail_fast,
    )
    write_json(
        out_dir / "confirmed_run_manifest.json",
        {
            "preview_dir": str(preview_path),
            "task_query": str(query_path),
            "confirmed": confirmed,
            "allow_preview_errors": allow_preview_errors,
            "fail_on_preview_warnings": fail_on_preview_warnings,
            "preview_status": preview.get("status"),
            "preview_risk_counts": risk_counts,
            "preview_warnings": preview_warnings,
            "task_type": preview.get("task_type"),
            "run_dir": str(run_dir),
            "audit_dir": str(audit_dir),
            "global_db": str(global_db) if global_db is not None else None,
        },
    )
    return run_dir
