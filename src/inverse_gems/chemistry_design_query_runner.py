from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .candidate_review import write_candidate_review
from .candidate_search import run_surrogate_candidate_search
from .candidate_selection import select_candidates
from .candidate_validation import validate_candidates
from .chemistry_candidate_table import build_chemistry_candidate_table
from .chemistry_domain import write_chemistry_domain_report
from .design_query import compile_design_query, validate_design_query_data
from .inverse_design_flow import write_inverse_design_flow_summary
from .materials import BINDER_COMPONENTS, canonicalize_material_name
from .sampling import generate_recipe_rows, write_recipe_csv
from .utils import config_path, load_yaml, write_json


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_if_exists(source: Path, target: Path) -> str | None:
    if not source.exists():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return str(target)


def _annotate_search_candidates_with_domain(search_dir: Path, domain_dir: Path) -> None:
    candidates_path = search_dir / "candidates.csv"
    domain_path = domain_dir / "chemistry_domain_report.csv"
    if not candidates_path.exists() or not domain_path.exists():
        return
    candidates = pd.read_csv(candidates_path)
    domain = pd.read_csv(domain_path)
    rename = {"recipe_id": "meta__recipe_id", "chem_hash": "meta__chem_hash"}
    domain = domain.rename(columns=rename)
    keys = [key for key in ["meta__recipe_id", "meta__chem_hash"] if key in candidates.columns and key in domain.columns]
    if not keys:
        return
    annotation_columns = [
        column
        for column in [
            *keys,
            "outside_input_range_count",
            "outside_input_range_columns",
            "nearest_scaled_distance",
            "nearest_reference_recipe_id",
            "nearest_reference_chem_hash",
            "out_of_domain",
        ]
        if column in domain.columns
    ]
    annotated = candidates.merge(domain[annotation_columns], on=keys, how="left")
    annotated.to_csv(candidates_path, index=False)
    (search_dir / "candidates.json").write_text(annotated.to_json(orient="records", indent=2), encoding="utf-8")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _design_space(query_data: dict[str, Any]) -> dict[str, Any]:
    return dict(query_data.get("design_space") or {})


def _query_age(query_data: dict[str, Any], default_age_days: float) -> float:
    if query_data.get("age_days") is not None:
        return float(query_data["age_days"])
    value = _design_space(query_data).get("age_days")
    if isinstance(value, dict):
        if value.get("equals") is None:
            raise ValueError("Chemistry design query needs a single age; use age_days or design_space.age_days.equals.")
        return float(value["equals"])
    if value is not None:
        return float(value)
    return float(default_age_days)


def _constraint_spec(query_data: dict[str, Any], name: str) -> dict[str, Any]:
    design_space = _design_space(query_data)
    hard = dict(query_data.get("hard_constraints") or {})
    constraints = dict(query_data.get("constraints") or {})
    sources = [
        query_data.get("inputs") or {},
        design_space.get("input_constraints") or {},
        hard.get("inputs") or {},
        constraints.get("inputs") or {},
    ]
    merged: dict[str, Any] = {}
    for source in sources:
        if name in source:
            value = source[name]
            if isinstance(value, dict):
                merged.update(value)
            else:
                merged["equals"] = value
    return merged


def _exact_constraint_value(spec: dict[str, Any]) -> float | None:
    if spec.get("equals") is not None:
        return float(spec["equals"])
    if spec.get("min") is not None and spec.get("max") is not None and float(spec["min"]) == float(spec["max"]):
        return float(spec["min"])
    return None


def _apply_generation_exact_constraints(rows: list[dict[str, Any]], query_data: dict[str, Any]) -> None:
    w_b = _exact_constraint_value(_constraint_spec(query_data, "w_b"))
    water_g = _exact_constraint_value(_constraint_spec(query_data, "water_g"))
    if w_b is None and water_g is None:
        return
    for row in rows:
        total_binder = sum(float(row.get(component) or 0.0) for component in ["OPC", "slag", "fly_ash", "metakaolin", "silica_fume", "limestone", "gypsum"])
        if water_g is not None:
            row["water_g"] = float(water_g)
            row["w_b"] = 0.0 if total_binder <= 0.0 else float(water_g) / total_binder
            row["water_mode"] = "direct_h2o"
        elif w_b is not None:
            row["w_b"] = float(w_b)
            row["water_g"] = total_binder * float(w_b)
            row["water_mode"] = "wb_total"


def _is_auto(value: Any) -> bool:
    return str(value).strip().lower() in {"", "auto", "any", "all", "none"}


def _explicit_material_system(query_data: dict[str, Any]) -> str | None:
    value = query_data.get("material_system") or _design_space(query_data).get("material_systems")
    systems = [str(item) for item in _as_list(value) if not _is_auto(item)]
    return systems[0] if len(systems) == 1 else None


def _requested_materials(query_data: dict[str, Any]) -> list[str]:
    values = _design_space(query_data).get("allowed_materials") or []
    return [canonicalize_material_name(str(value)) for value in values if canonicalize_material_name(str(value)) != "water"]


def _profile_allowed_materials(material_system: str, material_systems_config: str | Path) -> set[str]:
    profiles = load_yaml(material_systems_config)
    profile = profiles.get(material_system) or {}
    return {canonicalize_material_name(str(item)) for item in profile.get("allowed", [])}


def _strict_materials_enabled(query_data: dict[str, Any], override: bool | None = None) -> bool:
    if override is not None:
        return bool(override)
    return bool(_design_space(query_data).get("strict_materials", False))


def _strict_allowed_materials(
    query_data: dict[str, Any],
    *,
    material_system: str,
    material_systems_config: str | Path,
    strict_materials: bool,
) -> list[str] | None:
    if not strict_materials:
        return None
    requested = _requested_materials(query_data)
    if requested:
        return requested
    profile_allowed = _profile_allowed_materials(material_system, material_systems_config)
    return sorted((profile_allowed - {"gypsum"}) & BINDER_COMPONENTS)


def _query_with_strict_material_constraints(
    query_data: dict[str, Any],
    *,
    strict_allowed_materials: list[str] | None,
) -> dict[str, Any]:
    if not strict_allowed_materials:
        return query_data
    strict_allowed = {canonicalize_material_name(str(name)) for name in strict_allowed_materials}
    out = dict(query_data)
    inputs = dict(out.get("inputs") or {})
    for component in sorted(BINDER_COMPONENTS - strict_allowed):
        spec = dict(inputs.get(component) or {})
        spec["max"] = 0.0
        inputs[component] = spec
    out["inputs"] = inputs
    design_space = dict(out.get("design_space") or {})
    design_space["strict_materials"] = True
    design_space["allowed_materials"] = sorted(strict_allowed)
    out["design_space"] = design_space
    return out


def _material_system_from_allowed_materials(query_data: dict[str, Any], material_systems_config: str | Path) -> str:
    explicit = _explicit_material_system(query_data)
    if explicit:
        return explicit
    requested = set(_requested_materials(query_data))
    if not requested:
        raise ValueError("Chemistry design query needs material_system or design_space.allowed_materials.")
    profiles = load_yaml(material_systems_config)
    scored: list[tuple[int, str]] = []
    for name, profile in profiles.items():
        allowed = {canonicalize_material_name(str(item)) for item in profile.get("allowed", [])}
        fixed_zero = {canonicalize_material_name(str(item)) for item in profile.get("fixed_zero", [])}
        required = (allowed - {"gypsum"}) - fixed_zero
        if "OPC" in required:
            requested_effective = set(requested) | {"OPC"}
        else:
            requested_effective = set(requested)
        if required.issubset(requested_effective) and requested_effective.issubset(allowed | {"gypsum"}):
            scored.append((len(required), str(name)))
    if not scored:
        raise ValueError(f"Could not infer a material system from allowed_materials={sorted(requested)}.")
    scored.sort(reverse=True)
    return scored[0][1]


def run_chemistry_design_query(
    *,
    query: str | Path,
    out: str | Path,
    model_bundle: str | Path,
    reference_model_table: str | Path | None = None,
    n_candidates: int = 500,
    seed: int = 42,
    sampling_config: str | Path | None = None,
    material_systems_config: str | Path | None = None,
    dat_lst: str | Path | None = None,
    skip_validation: bool = True,
    use_mock_validation: bool = False,
    db: str | Path | None = None,
    validation_top_k: int | None = None,
    output_selection: str | Path | None = None,
    run_mode: str = "reacted_only",
    normalize: bool = True,
    allow_non_100: bool = False,
    temperature_celsius: float = 20.0,
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
    reaction_model_config: str | Path | None = None,
    materials_config: str | Path | None = None,
    strict_materials: bool | None = None,
    target_availability_policy: str = "warn",
    fail_fast: bool = False,
) -> Path:
    query_path = Path(query)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    material_systems_path = material_systems_config or config_path("material_systems.yaml")
    query_data = validate_design_query_data(load_yaml(query_path), require_model_paths=False)
    material_system = _material_system_from_allowed_materials(query_data, material_systems_path)
    age_days = _query_age(query_data, default_age_days=28.0)
    strict_enabled = _strict_materials_enabled(query_data, strict_materials)
    strict_allowed = _strict_allowed_materials(
        query_data,
        material_system=material_system,
        material_systems_config=material_systems_path,
        strict_materials=strict_enabled,
    )
    effective_query_data = _query_with_strict_material_constraints(
        query_data,
        strict_allowed_materials=strict_allowed,
    )
    effective_query_path = out_dir / "effective_design_query.yaml"
    import yaml

    with effective_query_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(effective_query_data, handle, sort_keys=False)

    recipes_dir = out_dir / "generated_candidates"
    recipes_dir.mkdir(parents=True, exist_ok=True)
    rows = generate_recipe_rows(
        config_path=sampling_config or config_path("sampling.yaml"),
        n=int(n_candidates),
        mode="mixed",
        ages=f"{age_days:g}",
        seed=int(seed),
        material_system=material_system,
        material_systems_path=material_systems_path,
        recipe_id_prefix=f"chem_{material_system}",
        strict_materials=strict_enabled,
        strict_allowed_materials=strict_allowed,
    )
    _apply_generation_exact_constraints(rows, effective_query_data)
    recipe_csv = recipes_dir / "candidate_recipes.csv"
    write_recipe_csv(recipe_csv, rows)

    chemistry_table = out_dir / "chemistry_candidate_table.csv"
    build_chemistry_candidate_table(
        recipes_csv=recipe_csv,
        out=chemistry_table,
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

    compiled_dir = out_dir / "compiled_query"
    compile_design_query(
        query=effective_query_path,
        out=compiled_dir,
        model_table=chemistry_table,
        model_bundle=model_bundle,
        reaction_model_id=reaction_model_id,
        reaction_model_config=reaction_model_config,
        target_availability_policy=target_availability_policy,
    )
    search_dir = out_dir / "surrogate_search"
    run_surrogate_candidate_search(query=compiled_dir / "surrogate_candidate_search.yaml", out=search_dir)

    domain_report = None
    if reference_model_table is not None:
        domain_report = write_chemistry_domain_report(
            reference_model_table=reference_model_table,
            candidate_table=chemistry_table,
            model_bundle=model_bundle,
            out=out_dir / "domain_report",
        )
        _annotate_search_candidates_with_domain(search_dir, Path(domain_report))

    final_csv: str | None = None
    final_json: str | None = None
    stage = "surrogate_search"
    validation_dir = out_dir / "validation"
    selection_dir = out_dir / "selection"
    if skip_validation:
        final_csv = _copy_if_exists(search_dir / "candidates.csv", out_dir / "final_candidates.csv")
        final_json = _copy_if_exists(search_dir / "candidates.json", out_dir / "final_candidates.json")
    else:
        if db is None:
            raise ValueError("--db is required unless --skip-validation is used.")
        validation = validate_candidates(
            candidates=search_dir / "candidates.csv",
            out=validation_dir,
            db=db,
            dat_lst=dat_lst,
            top_k=validation_top_k,
            use_mock=use_mock_validation,
            selection=output_selection,
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
            reaction_model_config=reaction_model_config,
            materials_config=materials_config,
            fail_fast=fail_fast,
        )
        selection = select_candidates(
            validation=validation / "validation_comparison.csv",
            config=compiled_dir / "candidate_selection.yaml",
            out=selection_dir,
            feature_table=validation / "validated_feature_table.csv",
        )
        final_csv = _copy_if_exists(selection / "selected_candidates.csv", out_dir / "final_selected_candidates.csv")
        final_json = _copy_if_exists(selection / "selected_candidates.json", out_dir / "final_selected_candidates.json")
        _copy_if_exists(selection / "selected_candidates.md", out_dir / "final_selected_candidates.md")
        stage = "selection"

    candidate_review = None
    if final_csv is not None:
        candidate_review = write_candidate_review(
            candidates=final_csv,
            out=out_dir,
            stage=stage,
            title=f"Chemistry Candidate Review ({stage})",
        )
    summary = {
        "query": str(query_path),
        "out": str(out_dir),
        "model_bundle": str(model_bundle),
        "reference_model_table": None if reference_model_table is None else str(reference_model_table),
        "material_system": material_system,
        "age_days": age_days,
        "strict_materials": strict_enabled,
        "strict_allowed_materials": strict_allowed,
        "n_candidates": int(n_candidates),
        "seed": int(seed),
        "skip_validation": skip_validation,
        "use_mock_validation": use_mock_validation,
        "final_stage": stage,
        "final_csv": final_csv,
        "final_json": final_json,
        "candidate_review": candidate_review,
        "paths": {
            "candidate_recipes": str(recipe_csv),
            "effective_design_query": str(effective_query_path),
            "chemistry_candidate_table": str(chemistry_table),
            "compiled_query": str(compiled_dir),
            "surrogate_search": str(search_dir),
            "domain_report": None if domain_report is None else str(domain_report),
            "validation": None if skip_validation else str(validation_dir),
            "selection": None if skip_validation else str(selection_dir),
        },
        "compiled_manifest": _read_json_if_exists(compiled_dir / "design_query_manifest.json"),
        "candidate_search_summary": _read_json_if_exists(search_dir / "candidate_search_summary.json"),
        "domain_summary": {} if domain_report is None else _read_json_if_exists(Path(domain_report) / "chemistry_domain_summary.json"),
        "validation_summary": {} if skip_validation else _read_json_if_exists(validation_dir / "validation_summary.json"),
        "selection_summary": {} if skip_validation else _read_json_if_exists(selection_dir / "selection_summary.json"),
    }
    summary["inverse_design_flow_summary"] = write_inverse_design_flow_summary(run_summary=summary, out=out_dir)
    write_json(out_dir / "chemistry_design_query_summary.json", summary)
    return out_dir
