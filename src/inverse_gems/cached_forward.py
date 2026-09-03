from __future__ import annotations

import contextlib
import io
import json
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

from .chem_hash import compute_chem_hash
from .chemistry_vector import chemistry_vector_from_source_ledger
from .constants import MOLAR_MASS_G_MOL
from .age_grids import age_metadata
from .database import (
    InverseGemsDatabase,
    copy_raw_outputs_to_chemistry_root,
    read_name_value_csv,
    save_raw_xgems_state,
    utc_now_iso,
)
from .materials import load_materials
from .porosity import compute_initial_volume_cm3, compute_porosity, load_porosity_config
from .recipe import Recipe, parse_recipe
from .reaction_model import current_reaction_model_metadata
from .reaction_parameters import ReactionParameterSet, load_reaction_parameters
from .solver_recovery import resolve_water_recovery_policy
from .source_ledger import build_source_ledger, update_ledger_hash, write_source_ledger_csv
from .utils import config_path, short_hash, write_json
from .xgems_input_builder import build_xgems_input
from .xgems_preflight import write_xgems_input_preflight_artifacts
from .xgems_runner import MockXGEMSRunner, XGEMSRunner


RunnerFactory = Callable[..., Any]


def _new_recipe_id(recipe: Recipe, chem_hash: str) -> str:
    return f"recipe_{short_hash({'recipe': recipe.to_dict(), 'chem_hash': chem_hash}, 10)}_{uuid.uuid4().hex[:8]}"


def _package_version() -> str:
    try:
        return version("inverse-gems")
    except PackageNotFoundError:
        return "0.1.0"


def _recipe_projection_payload(recipe: Recipe) -> dict[str, Any]:
    return {
        "binder_masses_g": {key: float(recipe.binder_masses_g[key]) for key in sorted(recipe.binder_masses_g)},
        "age_days": float(recipe.age_days),
        "water_g": float(recipe.water_g),
        "water_mode": recipe.water_mode,
        "w_b": None if recipe.w_b is None else float(recipe.w_b),
        "basis_g": float(recipe.basis_g),
    }


def _status_is_complete(status: Any) -> bool:
    text = str(status).lower()
    return bool(text) and "fail" not in text and "error" not in text and "bad" not in text


def _raw_dir_for_run(chem_dir: Path, force_rerun: bool) -> Path:
    base = chem_dir / "xgems_raw"
    if not force_rerun or not base.exists():
        return base
    return chem_dir / f"xgems_raw_{uuid.uuid4().hex[:8]}"


def _unique_source_mass_summary(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    seen: set[tuple[Any, ...]] = set()
    summary: dict[str, float] = {}
    for row in rows:
        key = (
            row.get("source_material"),
            row.get("source_phase_or_oxide"),
            row.get("source_mass_g_initial"),
            row.get("reaction_degree"),
        )
        if key in seen:
            continue
        seen.add(key)
        material = str(row.get("source_material"))
        summary[material] = summary.get(material, 0.0) + float(row.get(field) or 0.0)
    return summary


def _retry_cap_ladder(values: list[float] | tuple[float, ...] | str, minimum: float) -> list[float]:
    if isinstance(values, str):
        parsed = [float(item.strip()) for item in values.split(",") if item.strip()]
    else:
        parsed = [float(value) for value in values]
    return [value for value in parsed if value >= float(minimum)]


def _retry_fixed_ladder(values: list[float] | tuple[float, ...] | str) -> list[float]:
    if isinstance(values, str):
        parsed = [float(item.strip()) for item in values.split(",") if item.strip()]
    else:
        parsed = [float(value) for value in values]
    return [value for value in parsed if value > 0.0]


def _attempt_label(spec: dict[str, Any], index: int) -> str:
    if spec.get("label"):
        return str(spec["label"])
    if index == 0:
        return "primary"
    if spec.get("mode") == "cap_w_b":
        return f"retry_cap_w_b_{float(spec.get('water_w_b')):g}"
    if spec.get("mode") == "fixed_w_b":
        return f"retry_fixed_w_b_{float(spec.get('water_w_b')):g}"
    return f"retry_{index}"


def _chemistry_provenance_payload(
    *,
    hash_result: Any,
    water_mol: float,
    run_mode: str,
    temperature_celsius: float,
    pressure: float | None,
    xgems_water_policy: dict[str, Any],
) -> dict[str, Any]:
    return {
        "provenance_schema_version": 1,
        "inverse_gems_version": _package_version(),
        "chem_hash": hash_result.chem_hash,
        "chem_hash_version": hash_result.chem_hash_version,
        "dat_lst_hash": hash_result.dat_lst_hash,
        "species_map_hash": hash_result.species_map_hash,
        "xgems_run_mode": run_mode,
        "temperature_celsius": temperature_celsius,
        "pressure": pressure,
        "water_mol": water_mol,
        "xgems_water_policy": xgems_water_policy,
        "hash_payload": hash_result.hash_payload,
    }


def _prepare_attempt(
    *,
    recipe: Recipe,
    materials: dict[str, Any],
    run_mode: str,
    temperature_celsius: float,
    pressure: float | None,
    dat_lst: str | Path | None,
    use_mock: bool,
    reaction_parameters: ReactionParameterSet,
    water_mode: str,
    water_factor: float,
    water_g: float | None,
    water_w_b: float | None,
) -> dict[str, Any]:
    xgems_input = build_xgems_input(
        recipe,
        materials=materials,
        run_mode=run_mode,
        temperature_celsius=temperature_celsius,
        reaction_parameters=reaction_parameters,
        xgems_water_mode=water_mode,
        xgems_water_factor=water_factor,
        xgems_water_g=water_g,
        xgems_water_w_b=water_w_b,
    )
    ledger_rows = build_source_ledger(
        recipe_id="pending",
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
        temperature_celsius=temperature_celsius,
        pressure=pressure,
        dat_lst_path=dat_lst,
        dat_lst_hash="mock-dat" if use_mock else None,
        species_map_path=config_path("species_map.yaml"),
        xgems_run_mode=run_mode,
    )
    return {
        "xgems_input": xgems_input,
        "ledger_rows": ledger_rows,
        "canonical_vector": canonical_vector,
        "water_mol": water_mol,
        "hash_result": hash_result,
    }


def run_forward_cached(
    *,
    recipe_text: str,
    db: str | Path = "data/inverse_gems_db",
    dat_lst: str | Path | None = None,
    use_mock: bool = False,
    force_rerun_xgems: bool = False,
    run_mode: str = "reacted_only",
    normalize: bool = True,
    allow_non_100: bool = False,
    temperature_celsius: float = 20.0,
    pressure: float | None = None,
    gems_class_path: str | None = None,
    xgems_input_mode: str = "formula",
    xgems_water_mode: str = "initial",
    xgems_water_factor: float = 1.0,
    xgems_water_g: float | None = None,
    xgems_water_w_b: float | None = None,
    retry_water_on_failure: bool = False,
    retry_water_cap_w_b_ladder: list[float] | tuple[float, ...] | str = (0.45, 0.40, 0.35, 0.30),
    retry_water_up_w_b_ladder: list[float] | tuple[float, ...] | str = (0.30, 0.35, 0.40, 0.45),
    retry_water_min_w_b: float = 0.30,
    retry_water_policy: str | Any = "ladder",
    retry_water_max_retries: int = 6,
    xgems_call_budget: Any = None,
    runner_factory: RunnerFactory | None = None,
    recipe_id: str | None = None,
    recipe_metadata: dict[str, Any] | None = None,
    reaction_model_id: str | None = None,
    reaction_model_config: str | Path | None = None,
    materials_config: str | Path | None = None,
) -> dict[str, Any]:
    database = InverseGemsDatabase(db)
    materials = load_materials(materials_config)
    recipe = parse_recipe(recipe_text, materials=materials, normalize=normalize, allow_non_100=allow_non_100)
    age_meta = age_metadata(recipe.age_days).to_dict()
    recipe.metadata.update(age_meta)
    recipe.metadata.update(
        {
            "template_name": "",
            "temperature_celsius": temperature_celsius,
            "water_g": recipe.water_g,
            "w_b": recipe.w_b,
            "water_mode": recipe.water_mode,
            "materials_config": str(materials_config) if materials_config else None,
        }
    )
    if recipe_metadata:
        recipe.metadata.update(recipe_metadata)
    reaction_parameters = load_reaction_parameters(
        reaction_model_config,
        reaction_model_id=reaction_model_id or recipe.metadata.get("reaction_model_id"),
    )
    reaction_model = current_reaction_model_metadata(
        reaction_model_id=reaction_parameters.id,
        reaction_model_config=reaction_model_config,
        reaction_parameters=reaction_parameters,
    )
    recipe.metadata.update(
        {
            "reaction_model_id": reaction_model["reaction_model_id"],
            "reaction_model_signature": reaction_model["reaction_model_signature"],
            "reaction_model_signature_version": reaction_model["reaction_model_signature_version"],
            "reaction_parameter_set_id": reaction_parameters.id,
            "reaction_parameter_config_path": reaction_parameters.config_path,
            "reaction_parameter_config_hash": reaction_parameters.config_hash,
        }
    )
    if not use_mock and dat_lst is None:
        raise ValueError("--dat-lst is required for real cached xGEMS/GEMS runs.")

    primary_spec: dict[str, Any] = {
        "label": "primary",
        "mode": xgems_water_mode,
        "factor": xgems_water_factor,
        "water_g": xgems_water_g,
        "water_w_b": xgems_water_w_b,
    }
    recovery_policy = None
    if retry_water_on_failure:
        ladder_specs: list[dict[str, Any]] = []
        for cap_w_b in _retry_cap_ladder(retry_water_cap_w_b_ladder, retry_water_min_w_b):
            ladder_specs.append(
                {
                    "label": f"retry_cap_w_b_{cap_w_b:g}",
                    "mode": "cap_w_b",
                    "factor": 1.0,
                    "water_g": None,
                    "water_w_b": cap_w_b,
                }
            )
        for fixed_w_b in _retry_fixed_ladder(retry_water_up_w_b_ladder):
            ladder_specs.append(
                {
                    "label": f"retry_fixed_w_b_{fixed_w_b:g}",
                    "mode": "fixed_w_b",
                    "factor": 1.0,
                    "water_g": None,
                    "water_w_b": fixed_w_b,
                }
            )
        recovery_policy = resolve_water_recovery_policy(
            retry_water_policy,
            ladder_specs=ladder_specs,
            min_w_b=retry_water_min_w_b,
            max_retries=retry_water_max_retries,
        )

    seen_hashes: set[str] = set()
    attempt_history: list[dict[str, Any]] = []
    final_attempt: dict[str, Any] | None = None
    primary_record: dict[str, Any] | None = None

    attempt_index = 0
    spec: dict[str, Any] | None = primary_spec
    while spec is not None:
        prepared = _prepare_attempt(
            recipe=recipe,
            materials=materials,
            run_mode=run_mode,
            temperature_celsius=temperature_celsius,
            pressure=pressure,
            dat_lst=dat_lst,
            use_mock=use_mock,
            reaction_parameters=reaction_parameters,
            water_mode=str(spec["mode"]),
            water_factor=float(spec["factor"]),
            water_g=spec.get("water_g"),
            water_w_b=spec.get("water_w_b"),
        )
        xgems_input = prepared["xgems_input"]
        hash_result = prepared["hash_result"]
        label = _attempt_label(spec, attempt_index)
        if hash_result.chem_hash in seen_hashes:
            attempt_history.append(
                {
                    "attempt_index": attempt_index,
                    "label": label,
                    "status": "skipped_duplicate",
                    "chem_hash": hash_result.chem_hash,
                    "xgems_water_mode": xgems_input.water_policy.get("mode"),
                    "xgems_water_g": xgems_input.equilibrium_water_g,
                    "xgems_w_b": xgems_input.equilibrium_w_b,
                }
            )
            spec = recovery_policy.next_attempt(attempt_history) if recovery_policy else None
            attempt_index += 1
            continue
        seen_hashes.add(hash_result.chem_hash)

        chem_dir = database.chemistry_dir(hash_result.chem_hash)
        write_json(chem_dir / "canonical_component_vector.json", hash_result.exact_vector["element_mol"])
        write_json(chem_dir / "oxide_equivalent_vector.json", hash_result.exact_vector["oxide_equivalent_mol"])
        write_json(chem_dir / "chem_hash_payload.json", hash_result.to_dict())
        preflight_dir = chem_dir / "xgems_preflight" / label
        write_xgems_input_preflight_artifacts(
            recipe=recipe,
            xgems_input=xgems_input,
            out=preflight_dir,
            dat_lst=dat_lst,
            run_mode=run_mode,
            temperature_celsius=temperature_celsius,
            xgems_input_mode=xgems_input_mode,
            gems_class_path=gems_class_path or "xgems:ChemicalEngineDicts",
            reaction_parameter_set=reaction_parameters.to_dict(),
            context={
                "source": "run_forward_cached",
                "attempt_index": attempt_index,
                "attempt_label": label,
                "chem_hash": hash_result.chem_hash,
                "use_mock": use_mock,
                "force_rerun_xgems": force_rerun_xgems,
            },
        )

        reused_cache = database.chemistry_complete(hash_result.chem_hash) and not force_rerun_xgems
        raw_state: dict[str, Any] = {}
        solver_status: Any = "cached"
        raw_dir = chem_dir / "xgems_raw"
        attempt_warnings: dict[str, Any] = {
            "recipe": recipe.warnings,
            "xgems_input": xgems_input.warnings,
            "solver": [],
            "cache": [],
        }
        if reused_cache:
            chem_row = database.get_chemistry_run(hash_result.chem_hash) or {}
            raw_dir = Path(chem_row.get("xgems_run_dir") or raw_dir)
            raw_state["phase_volumes"] = read_name_value_csv(raw_dir / "xgems_phase_volumes_raw.csv")
            raw_state["phase_volumes_reconstructed"] = read_name_value_csv(raw_dir / "xgems_phase_volumes_reconstructed.csv")
            raw_state["phase_masses"] = read_name_value_csv(raw_dir / "xgems_phase_amounts_raw.csv")
            raw_state["aqueous_species"] = read_name_value_csv(raw_dir / "xgems_aqueous_species_raw.csv")
            scalars_path = raw_dir / "xgems_scalars_raw.json"
            raw_state["scalars"] = json.loads(scalars_path.read_text(encoding="utf-8")) if scalars_path.exists() else {}
            attempt_warnings["cache"].append(f"Reused complete chemistry run {hash_result.chem_hash}.")
        else:
            if xgems_call_budget is not None:
                xgems_call_budget.consume(f"{recipe.metadata.get('recipe_id') or recipe_id or 'recipe'}:{label}")
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            if runner_factory is not None:
                runner = runner_factory(dat_lst_path=dat_lst, temperature_celsius=temperature_celsius)
            elif use_mock:
                runner = MockXGEMSRunner(dat_lst_path=dat_lst, temperature_celsius=temperature_celsius)
            else:
                runner = XGEMSRunner(
                    dat_lst,
                    temperature_celsius=temperature_celsius,
                    gems_class_path=gems_class_path or "xgems:ChemicalEngineDicts",
                    input_mode=xgems_input_mode,
                )
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                runner.add_species_amounts(xgems_input.species_amounts_kg, units="kg")
                runner.apply_lower_bounds(xgems_input.lower_bounds_kg, units="kg")
                solver_status = runner.equilibrate()
                raw_state = runner.capture_raw_state()
            raw_state["solver_status"] = solver_status
            if not _status_is_complete(solver_status):
                attempt_warnings["solver"].append(f"xGEMS/GEMS reported non-success solver status: {solver_status}")
            raw_dir = _raw_dir_for_run(chem_dir, force_rerun_xgems)
            save_raw_xgems_state(raw_dir, raw_state, xgems_input.species_amounts_kg)
            (raw_dir / "xgems_stdout.txt").write_text(stdout_buffer.getvalue(), encoding="utf-8")
            (raw_dir / "xgems_stderr.txt").write_text(stderr_buffer.getvalue(), encoding="utf-8")
            copy_raw_outputs_to_chemistry_root(chem_dir, raw_dir)
            database.upsert_chemistry_run(
                {
                    "chem_hash": hash_result.chem_hash,
                    "chem_hash_version": hash_result.chem_hash_version,
                    "created_at": utc_now_iso(),
                    "status": "complete" if _status_is_complete(solver_status) else "failed",
                    "dat_lst_hash": hash_result.dat_lst_hash,
                    "species_map_hash": hash_result.species_map_hash,
                    "temperature_celsius": temperature_celsius,
                    "pressure": pressure,
                    "water_mol": prepared["water_mol"],
                    "canonical_vector_json": json.dumps(hash_result.exact_vector["element_mol"], sort_keys=True),
                    "oxide_equivalent_vector_json": json.dumps(hash_result.exact_vector["oxide_equivalent_mol"], sort_keys=True),
                    "xgems_run_dir": str(raw_dir),
                    "warnings_json": json.dumps(attempt_warnings, sort_keys=True),
                }
            )

        chemistry_status = "complete" if _status_is_complete(solver_status) else "failed"
        record = {
            "attempt_index": attempt_index,
            "label": label,
            "chem_hash": hash_result.chem_hash,
            "status": chemistry_status,
            "solver_status": solver_status,
            "reused_cache": reused_cache,
            "xgems_water_mode": xgems_input.water_policy.get("mode"),
            "xgems_water_g": xgems_input.equilibrium_water_g,
            "xgems_w_b": xgems_input.equilibrium_w_b,
            "preflight_dir": str(preflight_dir),
        }
        if spec.get("diagnosis"):
            record["diagnosis"] = spec["diagnosis"]
        attempt_history.append(record)
        if primary_record is None:
            primary_record = record
        current_attempt = {
            **prepared,
            "chem_dir": chem_dir,
            "raw_dir": raw_dir,
            "raw_state": raw_state,
            "solver_status": solver_status,
            "chemistry_status": chemistry_status,
            "reused_cache": reused_cache,
            "warnings": attempt_warnings,
            "history_record": record,
            "preflight_dir": preflight_dir,
        }
        final_attempt = current_attempt
        if chemistry_status == "complete":
            break
        if recovery_policy is None:
            break
        spec = recovery_policy.next_attempt(attempt_history)
        attempt_index += 1

    if final_attempt is None:
        raise RuntimeError("No xGEMS/GEMS attempt was executed.")

    xgems_input = final_attempt["xgems_input"]
    ledger_rows = final_attempt["ledger_rows"]
    hash_result = final_attempt["hash_result"]
    raw_state = final_attempt["raw_state"]
    solver_status = final_attempt["solver_status"]
    chem_dir = final_attempt["chem_dir"]
    raw_dir = final_attempt["raw_dir"]
    reused_cache = bool(final_attempt["reused_cache"])
    preflight_dir = final_attempt.get("preflight_dir")
    warnings = final_attempt["warnings"]
    primary_record = primary_record or final_attempt["history_record"]
    solver_rescued = (
        primary_record.get("status") != "complete"
        and final_attempt["history_record"].get("status") == "complete"
        and primary_record.get("chem_hash") != final_attempt["history_record"].get("chem_hash")
    )
    retry_count = sum(1 for record in attempt_history[1:] if record.get("status") != "skipped_duplicate")
    warnings["retry"] = {
        "enabled": retry_water_on_failure,
        "policy": retry_water_policy if isinstance(retry_water_policy, str) else type(retry_water_policy).__name__,
        "solver_rescued": solver_rescued,
        "retry_count": retry_count,
        "history": attempt_history,
    }
    if solver_rescued:
        warnings.setdefault("solver", []).append(
            "Primary xGEMS/GEMS attempt failed; final result uses an adaptive xGEMS-water retry."
        )

    recipe_id = recipe_id or str(recipe.metadata.get("recipe_id") or "") or _new_recipe_id(recipe, hash_result.chem_hash)
    for row in ledger_rows:
        row["recipe_id"] = recipe_id
    update_ledger_hash(ledger_rows, hash_result.chem_hash)
    prepared_identity_payload = {
        "recipe_projection": _recipe_projection_payload(recipe),
        "reaction_model_signature": reaction_model["reaction_model_signature"],
        "chem_hash": hash_result.chem_hash,
        "dat_lst_hash": hash_result.dat_lst_hash,
        "species_map_hash": hash_result.species_map_hash,
        "run_mode": run_mode,
        "temperature_celsius": temperature_celsius,
        "pressure": pressure,
        "xgems_water_policy": xgems_input.water_policy,
    }
    prepared_id = f"prepared_{short_hash(prepared_identity_payload, 20)}"
    chemistry_provenance = _chemistry_provenance_payload(
        hash_result=hash_result,
        water_mol=final_attempt["water_mol"],
        run_mode=run_mode,
        temperature_celsius=temperature_celsius,
        pressure=pressure,
        xgems_water_policy=xgems_input.water_policy,
    )
    recipe.metadata.update(
        {
            "prepared_id": prepared_id,
            "xgems_water_mode": xgems_input.water_policy.get("mode"),
            "xgems_water_g": xgems_input.equilibrium_water_g,
            "xgems_w_b": xgems_input.equilibrium_w_b,
            "xgems_water_factor": xgems_input.water_policy.get("factor"),
            "solver_rescued": solver_rescued,
            "xgems_retry_count": retry_count,
            "primary_chem_hash": primary_record.get("chem_hash"),
            "primary_solver_status": str(primary_record.get("solver_status")),
        }
    )
    run_provenance = {
        "provenance_schema_version": 1,
        "inverse_gems_version": _package_version(),
        "recipe_id": recipe_id,
        "prepared_id": prepared_id,
        "chem_hash": hash_result.chem_hash,
        "primary_chem_hash": primary_record.get("chem_hash"),
        "prepared_identity_payload": prepared_identity_payload,
        "reaction_model": reaction_model,
        "reaction_parameter_set": reaction_parameters.to_dict(),
        "chemistry_provenance": chemistry_provenance,
        "solver": {
            "chemistry_status": final_attempt["chemistry_status"],
            "solver_status": str(solver_status),
            "solver_rescued": solver_rescued,
            "xgems_retry_count": retry_count,
            "retry_history": attempt_history,
        },
    }
    write_json(chem_dir / "chemistry_provenance.json", chemistry_provenance)
    if Path(raw_dir).exists():
        write_json(Path(raw_dir) / "chemistry_provenance.json", chemistry_provenance)

    porosity_config = load_porosity_config()
    porosity_config["xgems_run_mode"] = run_mode
    porosity_config["xgems_phase_volume_unit"] = "cm3" if use_mock else "m3"
    porosity = compute_porosity(
        recipe,
        materials=materials,
        xgems_phase_volumes=raw_state.get("phase_volumes") or {},
        xgems_phase_volumes_reconstructed=raw_state.get("phase_volumes_reconstructed") or {},
        unreacted_masses_g=xgems_input.unreacted_masses_g,
        config=porosity_config,
    )
    warnings["porosity"] = porosity.get("warnings", [])
    initial_volume = compute_initial_volume_cm3(recipe, materials)
    source_ledger_hash = short_hash(ledger_rows, 20)
    prepared_record = {
        "prepared_id": prepared_id,
        "recipe_id": recipe_id,
        "chem_hash": hash_result.chem_hash,
        "created_at": utc_now_iso(),
        "reaction_model_id": reaction_model["reaction_model_id"],
        "reaction_model_signature": reaction_model["reaction_model_signature"],
        "reaction_model_signature_version": reaction_model["reaction_model_signature_version"],
        "reaction_model_json": json.dumps(reaction_model["reaction_model_payload"], sort_keys=True),
        "recipe_json": json.dumps(recipe.to_dict(), sort_keys=True),
        "reaction_degrees_json": json.dumps(xgems_input.reaction_degrees, sort_keys=True),
        "xgems_species_amounts_json": json.dumps(xgems_input.species_amounts_kg, sort_keys=True),
        "unreacted_masses_json": json.dumps(xgems_input.unreacted_masses_g, sort_keys=True),
        "canonical_vector_json": json.dumps(hash_result.exact_vector["element_mol"], sort_keys=True),
        "oxide_equivalent_vector_json": json.dumps(hash_result.exact_vector["oxide_equivalent_mol"], sort_keys=True),
        "water_mol": final_attempt["water_mol"],
        "temperature_celsius": temperature_celsius,
        "pressure": pressure,
        "xgems_water_policy_json": json.dumps(xgems_input.water_policy, sort_keys=True),
        "source_ledger_hash": source_ledger_hash,
        "warnings_json": json.dumps(warnings, sort_keys=True),
    }
    database.upsert_prepared_chemistry_run(prepared_record)
    prepared_dir = database.prepared_chemistry_dir(prepared_id)
    write_json(
        prepared_dir / "prepared_chemistry.json",
        {
            "prepared_id": prepared_id,
            "recipe_id": recipe_id,
            "chem_hash": hash_result.chem_hash,
            "prepared_identity_payload": prepared_identity_payload,
            "reaction_model": reaction_model,
            "reaction_parameter_set": reaction_parameters.to_dict(),
            "run_provenance": run_provenance,
            "recipe": recipe.to_dict(),
            "reaction_degrees": xgems_input.reaction_degrees,
            "xgems_species_amounts_kg": xgems_input.species_amounts_kg,
            "unreacted_masses_g": xgems_input.unreacted_masses_g,
            "canonical_vector": hash_result.exact_vector["element_mol"],
            "oxide_equivalent_vector": hash_result.exact_vector["oxide_equivalent_mol"],
            "water_mol": final_attempt["water_mol"],
            "xgems_water_policy": xgems_input.water_policy,
            "source_ledger_hash": source_ledger_hash,
            "warnings": warnings,
        },
    )
    write_json(prepared_dir / "run_provenance.json", run_provenance)
    (prepared_dir / "linked_chem_hash.txt").write_text(hash_result.chem_hash, encoding="utf-8")
    recipe_dir = database.recipe_dir(recipe_id)
    write_json(recipe_dir / "recipe.json", recipe.to_dict())
    write_json(recipe_dir / "run_provenance.json", run_provenance)
    write_json(recipe_dir / "reaction_degrees.json", xgems_input.reaction_degrees)
    write_source_ledger_csv(recipe_dir / "source_contribution_ledger.csv", ledger_rows)
    write_json(recipe_dir / "unreacted_masses.json", xgems_input.unreacted_masses_g)
    write_json(
        recipe_dir / "volumes.json",
        {
            "initial": initial_volume,
            "porosity_volumes": porosity,
        },
    )
    write_json(recipe_dir / "porosity.json", porosity)
    (recipe_dir / "linked_chem_hash.txt").write_text(hash_result.chem_hash, encoding="utf-8")
    write_json(recipe_dir / "warnings.json", warnings)
    database.insert_recipe_run(
        {
            "recipe_id": recipe_id,
            "chem_hash": hash_result.chem_hash,
            "prepared_id": prepared_id,
            "created_at": utc_now_iso(),
            "reaction_model_id": reaction_model["reaction_model_id"],
            "reaction_model_signature": reaction_model["reaction_model_signature"],
            "reaction_model_signature_version": reaction_model["reaction_model_signature_version"],
            "reaction_parameter_set_id": reaction_parameters.id,
            "reaction_parameter_config_path": reaction_parameters.config_path,
            "reaction_parameter_config_hash": reaction_parameters.config_hash,
            "reaction_model_json": json.dumps(reaction_model["reaction_model_payload"], sort_keys=True),
            "template_name": recipe.metadata.get("template_name"),
            "material_system": recipe.metadata.get("material_system"),
            "target_profile": recipe.metadata.get("target_profile"),
            "target_profile_description": recipe.metadata.get("target_profile_description"),
            "recipe_json": json.dumps(recipe.to_dict(), sort_keys=True),
            "reaction_degrees_json": json.dumps(xgems_input.reaction_degrees, sort_keys=True),
            "initial_masses_json": json.dumps(recipe.binder_masses_g, sort_keys=True),
            "reacted_masses_json": json.dumps(_unique_source_mass_summary(ledger_rows, "reacted_mass_g"), sort_keys=True),
            "unreacted_masses_json": json.dumps(xgems_input.unreacted_masses_g, sort_keys=True),
            "water_g": recipe.metadata.get("water_g", recipe.water_g),
            "w_b": recipe.metadata.get("w_b", recipe.w_b),
            "water_mode": recipe.metadata.get("water_mode", recipe.water_mode),
            "xgems_water_g": xgems_input.equilibrium_water_g,
            "xgems_w_b": xgems_input.equilibrium_w_b,
            "xgems_water_mode": xgems_input.water_policy.get("mode"),
            "xgems_water_policy_json": json.dumps(xgems_input.water_policy, sort_keys=True),
            "solver_rescued": int(bool(solver_rescued)),
            "xgems_retry_count": retry_count,
            "primary_chem_hash": primary_record.get("chem_hash"),
            "primary_solver_status": str(primary_record.get("solver_status")),
            "retry_history_json": json.dumps(attempt_history, sort_keys=True),
            "age_days": recipe.metadata.get("age_days", recipe.age_days),
            "age_hours": recipe.metadata.get("age_hours"),
            "age_minutes": recipe.metadata.get("age_minutes"),
            "age_label": recipe.metadata.get("age_label"),
            "age_bin": recipe.metadata.get("age_bin"),
            "temperature_celsius": recipe.metadata.get("temperature_celsius", temperature_celsius),
            "initial_volume_cm3": porosity.get("initial_volume_cm3"),
            "final_solid_volume_cm3": porosity.get("solid_final_volume_cm3"),
            "porosity": porosity.get("porosity_best_effort"),
            "run_dir": str(recipe_dir),
            "warnings_json": json.dumps(warnings, sort_keys=True),
        }
    )
    database.insert_source_contributions(ledger_rows)
    chemistry_status = str(final_attempt["chemistry_status"])
    return {
        "recipe_id": recipe_id,
        "chem_hash": hash_result.chem_hash,
        "prepared_id": prepared_id,
        "reused_cache": reused_cache,
        "chemistry_dir": str(chem_dir),
        "prepared_chemistry_dir": str(prepared_dir),
        "recipe_dir": str(recipe_dir),
        "preflight_dir": str(preflight_dir) if preflight_dir else None,
        "reaction_model_id": reaction_model["reaction_model_id"],
        "reaction_model_signature": reaction_model["reaction_model_signature"],
        "reaction_model_signature_version": reaction_model["reaction_model_signature_version"],
        "reaction_parameter_set_id": reaction_parameters.id,
        "reaction_parameter_config_hash": reaction_parameters.config_hash,
        "solver_status": solver_status,
        "chemistry_status": chemistry_status,
        "xgems_water_g": xgems_input.equilibrium_water_g,
        "xgems_w_b": xgems_input.equilibrium_w_b,
        "xgems_water_mode": xgems_input.water_policy.get("mode"),
        "solver_rescued": solver_rescued,
        "xgems_retry_count": retry_count,
        "primary_chem_hash": primary_record.get("chem_hash"),
        "retry_history": attempt_history,
        "porosity": porosity.get("porosity_best_effort"),
    }
