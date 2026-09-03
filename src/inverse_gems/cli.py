from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

from .acceptance import run_acceptance_suite
from .active_learning_priority import write_active_learning_target_priorities
from .api import run_request
from .backfill import backfill_reconstructed_volumes_and_porosity
from .candidate_search import run_surrogate_candidate_search
from .candidate_selection import select_candidates
from .candidate_validation import validate_candidates
from .chemistry_candidate_table import build_chemistry_candidate_table
from .chemistry_design_query_runner import run_chemistry_design_query
from .chemistry_domain import write_chemistry_domain_report
from .materials import load_materials, material_subset
from .cached_forward import run_forward_cached
from .batch import run_batch_cached, summarize_batch_status
from .database import InverseGemsDatabase
from .design_query import compile_design_query, design_query_json_schema, save_design_query_schema, validate_design_query_file
from .design_run_report import run_design_run_report
from .design_query_runner import run_design_query
from .env_check import check_environment
from .feature_diagnostics import run_feature_diagnostics
from .feature_table import build_feature_table, inspect_available_outputs
from .forward_answer import write_forward_answer
from .forward_narrative import write_forward_narrative
from .forward_query import (
    forward_query_json_schema,
    run_forward_query,
    save_forward_query_schema,
    validate_forward_query_file,
)
from .forward_result_summary import summarize_forward_result
from .global_chemistry_db import (
    acquire_global_chemistry_candidates,
    copy_existing_artifacts_into_global_db,
    initialize_global_chemistry_db,
    import_cached_db_into_global_db,
    lookup_global_chemistry,
    refresh_global_chemistry_db,
    run_global_design_query,
    run_global_forward_query,
    train_global_chemistry_surrogate,
)
from .global_chemistry_cycle import run_global_chemistry_acquisition_cycle
from .global_chemistry_coverage import write_global_chemistry_coverage_report
from .inverse_query import run_inverse_query
from .inverse_forward_workflow import run_inverse_forward_workflow
from .llm_task_router import (
    save_rendered_task_router_prompt,
    validate_llm_task_query_file,
    write_llm_task_validation_report,
)
from .model_registry_diagnostics import run_model_registry_diagnostics
from .model_router import route_design_query_file
from .model_table import build_model_table
from .openai_task_router import parse_user_request_with_openai, run_user_request_with_openai
from .porosity import compute_porosity, load_porosity_config
from .recipe import parse_recipe
from .sampling import expand_age_rows, generate_recipe_rows, read_recipe_csv, write_recipe_csv
from .qc import summarize_database
from .reaction_model import current_reaction_model_metadata
from .reaction_parameters import load_reaction_parameters
from .surrogate import train_baseline_surrogate
from .task_query import run_task_query, save_task_query_schema, task_query_json_schema, validate_task_query_file
from .task_query_preview import preview_task_query_file, run_confirmed_task_query
from .target_region_analysis import write_target_region_analysis
from .xgems_input_builder import build_xgems_input
from .xgems_output_capture import save_run_outputs
from .xgems_preflight import run_xgems_input_preflight
from .xgems_quality_cases import write_xgems_quality_case_report
from .xgems_runner import MockXGEMSRunner, XGEMSRunner


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_str_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in str(value).split(",") if item.strip()]


def run_forward_recipe(
    *,
    recipe_text: str,
    out: str | Path = "runs",
    dat_lst: str | Path | None = None,
    use_mock: bool = False,
    run_mode: str = "reacted_only",
    normalize: bool = True,
    allow_non_100: bool = False,
    temperature_celsius: float = 20.0,
    command_name: str = "forward",
    gems_class_path: str | None = None,
    xgems_input_mode: str = "species",
    xgems_water_mode: str = "initial",
    xgems_water_factor: float = 1.0,
    xgems_water_g: float | None = None,
    xgems_water_w_b: float | None = None,
    reaction_model_id: str | None = None,
    reaction_model_config: str | Path | None = None,
    materials_config: str | Path | None = None,
) -> Path:
    materials = load_materials(materials_config)
    recipe = parse_recipe(recipe_text, materials=materials, normalize=normalize, allow_non_100=allow_non_100)
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
        temperature_celsius=temperature_celsius,
        reaction_parameters=reaction_parameters,
        xgems_water_mode=xgems_water_mode,
        xgems_water_factor=xgems_water_factor,
        xgems_water_g=xgems_water_g,
        xgems_water_w_b=xgems_water_w_b,
    )
    recipe.metadata.update(
        {
            "xgems_water_mode": xgems_input.water_policy.get("mode"),
            "xgems_water_g": xgems_input.equilibrium_water_g,
            "xgems_w_b": xgems_input.equilibrium_w_b,
            "xgems_water_factor": xgems_input.water_policy.get("factor"),
            "reaction_model_id": reaction_model["reaction_model_id"],
            "reaction_model_signature": reaction_model["reaction_model_signature"],
            "reaction_model_signature_version": reaction_model["reaction_model_signature_version"],
            "reaction_parameter_set_id": reaction_parameters.id,
            "reaction_parameter_config_path": reaction_parameters.config_path,
            "reaction_parameter_config_hash": reaction_parameters.config_hash,
        }
    )

    runner: MockXGEMSRunner | XGEMSRunner
    if use_mock:
        runner = MockXGEMSRunner(dat_lst_path=dat_lst, temperature_celsius=temperature_celsius)
    else:
        if dat_lst is None:
            raise ValueError("--dat-lst is required for real xGEMS/GEMS runs.")
        runner = XGEMSRunner(
            dat_lst,
            temperature_celsius=temperature_celsius,
            gems_class_path=gems_class_path or os.environ.get("INVERSE_GEMS_GEMS_CLASS_PATH", "run.GEMSCalc:GEMS"),
            input_mode=xgems_input_mode,
        )

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        runner.add_species_amounts(xgems_input.species_amounts_kg, units="kg")
        runner.apply_lower_bounds(xgems_input.lower_bounds_kg, units="kg")
        solver_status = runner.equilibrate()
        raw_state = runner.capture_raw_state()
    raw_state["solver_status"] = solver_status

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

    warning_payload: dict[str, Any] = {
        "recipe": recipe.warnings,
        "xgems_input": xgems_input.warnings,
        "porosity": porosity.get("warnings", []),
        "solver": [],
    }
    solver_text = str(solver_status)
    if "fail" in solver_text.lower() or "error" in solver_text.lower():
        warning_payload["solver"].append(f"xGEMS/GEMS reported non-success solver status: {solver_text}")
    manifest = {
        "command": command_name,
        "runner": "mock" if use_mock else "xgems",
        "dat_lst_path": str(dat_lst) if dat_lst else None,
        "run_mode": run_mode,
        "temperature_celsius": temperature_celsius,
        "gems_class_path": gems_class_path or os.environ.get("INVERSE_GEMS_GEMS_CLASS_PATH", "run.GEMSCalc:GEMS"),
        "xgems_input_mode": xgems_input_mode,
        "xgems_water": xgems_input.water_policy,
        "reaction_model_id": reaction_model["reaction_model_id"],
        "reaction_model_signature": reaction_model["reaction_model_signature"],
        "reaction_model_signature_version": reaction_model["reaction_model_signature_version"],
        "reaction_parameter_set_id": reaction_parameters.id,
        "reaction_parameter_config_path": reaction_parameters.config_path,
        "reaction_parameter_config_hash": reaction_parameters.config_hash,
        "raw_phase_names_preserved": True,
        "solver_status": solver_status,
    }
    run_provenance = {
        "provenance_schema_version": 1,
        "command": command_name,
        "recipe": recipe.to_dict(),
        "reaction_model": reaction_model,
        "reaction_parameter_set": reaction_parameters.to_dict(),
        "xgems": {
            "runner": "mock" if use_mock else "xgems",
            "dat_lst_path": str(dat_lst) if dat_lst else None,
            "run_mode": run_mode,
            "temperature_celsius": temperature_celsius,
            "gems_class_path": gems_class_path or os.environ.get("INVERSE_GEMS_GEMS_CLASS_PATH", "run.GEMSCalc:GEMS"),
            "input_mode": xgems_input_mode,
            "water_policy": xgems_input.water_policy,
            "solver_status": str(solver_status),
        },
    }
    return save_run_outputs(
        out_base=out,
        manifest=manifest,
        user_request=recipe_text,
        recipe=recipe.to_dict(),
        materials_used=material_subset(materials, recipe.binder_masses_g),
        reaction_degrees=xgems_input.reaction_degrees,
        xgems_species_amounts=xgems_input.species_amounts_kg,
        raw_state=raw_state,
        porosity=porosity,
        warnings=warning_payload,
        stdout_text=stdout_buffer.getvalue(),
        stderr_text=stderr_buffer.getvalue(),
        run_provenance=run_provenance,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inverse-gems")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_xgems_water_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--xgems-water-mode",
            choices=["initial", "fraction_of_initial", "cap_w_b", "fixed_w_b", "direct_water_g"],
            default="initial",
            help="Water sent to xGEMS. Initial recipe water is still used for PK hydration and porosity.",
        )
        subparser.add_argument("--xgems-water-factor", type=float, default=1.0)
        subparser.add_argument("--xgems-water-g", type=float, default=None)
        subparser.add_argument("--xgems-water-w-b", type=float, default=None)

    def add_retry_water_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--retry-water-on-failure",
            action="store_true",
            help="After a non-success solver status, retry with capped xGEMS water. Initial recipe water still controls PK and porosity.",
        )
        subparser.add_argument("--retry-water-cap-w-b-ladder", type=_parse_float_list, default=[0.45, 0.40, 0.35, 0.30])
        subparser.add_argument("--retry-water-up-w-b-ladder", type=_parse_float_list, default=[0.30, 0.35, 0.40, 0.45])
        subparser.add_argument("--retry-water-min-w-b", type=float, default=0.30)

    def add_reaction_model_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--reaction-model-id",
            default=None,
            help="Optional name for the PK/SCM/availability parameter set used to project recipes to chemistry.",
        )
        subparser.add_argument(
            "--reaction-model-config",
            default=None,
            type=Path,
            help="YAML file with actual PK/SCM/availability parameter overrides for recipe-to-chemistry projection.",
        )

    def add_common(subparser: argparse.ArgumentParser, *, needs_dat: bool) -> None:
        if needs_dat:
            subparser.add_argument("--dat-lst", required=True, type=Path)
        else:
            subparser.add_argument("--dat-lst", required=False, type=Path)
        subparser.add_argument("--recipe", required=True)
        subparser.add_argument("--out", default=Path("runs"), type=Path)
        subparser.add_argument("--run-mode", choices=["reacted_only", "lower_bound_legacy"], default="reacted_only")
        subparser.add_argument("--temperature-celsius", type=float, default=20.0)
        subparser.add_argument(
            "--xgems-input-mode",
            choices=["species", "formula"],
            default="species",
            help="Use 'species' for direct xGEMS species amounts, or 'formula' to convert input names to element bulk composition.",
        )
        subparser.add_argument(
            "--gems-class-path",
            default=None,
            help="Import path for the real GEMS class, for example 'run.GEMSCalc:GEMS'. "
            "Can also be set with INVERSE_GEMS_GEMS_CLASS_PATH.",
        )
        add_xgems_water_args(subparser)
        add_reaction_model_args(subparser)
        subparser.add_argument("--no-normalize", action="store_true")

    add_common(subparsers.add_parser("forward"), needs_dat=True)
    add_common(subparsers.add_parser("inspect-xgems-output"), needs_dat=True)
    add_common(subparsers.add_parser("forward-mock"), needs_dat=False)

    check_env = subparsers.add_parser("check-env")
    check_env.add_argument("--dat-lst", default=None, type=Path)
    check_env.add_argument("--gems-class-path", default="xgems:ChemicalEngineDicts")
    check_env.add_argument("--xgems-input-mode", choices=["species", "formula"], default="formula")
    check_env.add_argument("--require-xgems", action="store_true")
    check_env.add_argument("--instantiate-runner", action="store_true")
    check_env.add_argument("--out", default=None, type=Path)

    preflight = subparsers.add_parser("preflight-xgems-input")
    preflight.add_argument("--dat-lst", required=True, type=Path)
    preflight.add_argument("--recipe", required=True)
    preflight.add_argument("--out", required=True, type=Path)
    preflight.add_argument("--run-mode", choices=["reacted_only", "lower_bound_legacy"], default="reacted_only")
    preflight.add_argument("--temperature-celsius", type=float, default=20.0)
    preflight.add_argument("--xgems-input-mode", choices=["species", "formula"], default="formula")
    preflight.add_argument("--gems-class-path", default="xgems:ChemicalEngineDicts")
    preflight.add_argument("--instantiate-runner", action="store_true")
    preflight.add_argument("--table-limit", type=int, default=40)
    add_xgems_water_args(preflight)
    add_reaction_model_args(preflight)
    preflight.add_argument("--no-normalize", action="store_true")

    def add_acceptance_args(subparser: argparse.ArgumentParser, *, needs_dat: bool) -> None:
        if needs_dat:
            subparser.add_argument("--dat-lst", required=True, type=Path)
        else:
            subparser.add_argument("--dat-lst", required=False, type=Path)
        subparser.add_argument("--out", default=Path("results/acceptance"), type=Path)
        subparser.add_argument("--db", default=None, type=Path)
        subparser.add_argument("--case", action="append", default=[], help="Run only a named acceptance case.")
        subparser.add_argument("--run-mode", choices=["reacted_only", "lower_bound_legacy"], default="reacted_only")
        subparser.add_argument("--temperature-celsius", type=float, default=20.0)
        subparser.add_argument("--pressure", type=float, default=None)
        subparser.add_argument("--force-rerun-xgems", action="store_true")
        subparser.add_argument("--xgems-input-mode", choices=["species", "formula"], default="formula")
        subparser.add_argument("--gems-class-path", default="xgems:ChemicalEngineDicts")
        add_xgems_water_args(subparser)
        subparser.add_argument(
            "--retry-water-on-failure",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Retry failed real xGEMS calculations with the configured water ladder.",
        )
        subparser.add_argument("--retry-water-cap-w-b-ladder", type=_parse_float_list, default=[0.45, 0.40, 0.35, 0.30])
        subparser.add_argument("--retry-water-up-w-b-ladder", type=_parse_float_list, default=[0.30, 0.35, 0.40, 0.45])
        subparser.add_argument("--retry-water-min-w-b", type=float, default=0.30)
        subparser.add_argument("--with-plots", action="store_true", help="Generate acceptance forward-query plots.")
        subparser.add_argument("--top-n-phases", type=int, default=8)
        subparser.add_argument("--no-normalize", action="store_true")
        subparser.add_argument("--fail-fast", action="store_true")

    add_acceptance_args(subparsers.add_parser("acceptance-real"), needs_dat=True)
    add_acceptance_args(subparsers.add_parser("acceptance-mock"), needs_dat=False)

    result_summary = subparsers.add_parser("summarize-forward-result")
    result_summary.add_argument("--run", required=True, type=Path)
    result_summary.add_argument("--phase", action="append", default=[])
    result_summary.add_argument("--phase-group", action="append", default=[])
    result_summary.add_argument("--scalar", action="append", default=[])
    result_summary.add_argument("--out", default=None, type=Path)
    result_summary.add_argument("--top-phases", type=int, default=8)
    result_summary.add_argument("--table-limit", type=int, default=20)

    forward_answer = subparsers.add_parser("write-forward-answer")
    forward_answer.add_argument("--summary", default=None, type=Path)
    forward_answer.add_argument("--run", default=None, type=Path)
    forward_answer.add_argument("--out", default=None, type=Path)
    forward_answer.add_argument("--table-limit", type=int, default=10)

    forward_narrative = subparsers.add_parser("write-forward-narrative")
    forward_narrative.add_argument("--answer", default=None, type=Path)
    forward_narrative.add_argument("--run", default=None, type=Path)
    forward_narrative.add_argument("--out", default=None, type=Path)
    forward_narrative.add_argument("--language", choices=["ko", "en"], default="ko")
    forward_narrative.add_argument("--use-openai", action="store_true")
    forward_narrative.add_argument("--model", default=None)
    forward_narrative.add_argument("--max-output-tokens", type=int, default=1200)
    forward_narrative.add_argument("--temperature", type=float, default=0.0)

    def add_request_facade_args(subparser: argparse.ArgumentParser, *, needs_dat: bool) -> None:
        subparser.add_argument("--request", default=None)
        subparser.add_argument("--request-file", default=None, type=Path)
        subparser.add_argument("--task-query", default=None, type=Path)
        subparser.add_argument("--forward-query", default=None, type=Path)
        subparser.add_argument("--confirmed-preview", default=None, type=Path)
        subparser.add_argument("--confirm-preview", action="store_true")
        subparser.add_argument("--allow-preview-errors", action="store_true")
        subparser.add_argument("--fail-on-preview-warnings", action="store_true")
        if needs_dat:
            subparser.add_argument("--dat-lst", required=False, type=Path)
        else:
            subparser.add_argument("--dat-lst", required=False, type=Path)
        subparser.add_argument("--out", required=True, type=Path)
        subparser.add_argument("--db", required=True, type=Path)
        subparser.add_argument("--use-openai", action="store_true")
        subparser.add_argument("--llm-config", default=None, type=Path)
        subparser.add_argument("--model", default=None)
        subparser.add_argument("--max-repairs", type=int, default=None)
        subparser.add_argument("--model-table", default=None, type=Path)
        subparser.add_argument("--model-bundle", default=None, type=Path)
        subparser.add_argument("--model-registry", default=None, type=Path)
        subparser.add_argument("--material-systems-config", default=Path("configs/material_systems.yaml"), type=Path)
        subparser.add_argument("--route-target-policy", choices=["recommended", "allow_caution"], default="recommended")
        subparser.add_argument("--skip-validation", action="store_true")
        subparser.add_argument("--validation-top-k", type=int, default=None)
        subparser.add_argument("--selection", default=None, type=Path)
        subparser.add_argument("--run-mode", choices=["reacted_only", "lower_bound_legacy"], default="reacted_only")
        subparser.add_argument("--temperature-celsius", type=float, default=None)
        subparser.add_argument("--pressure", type=float, default=None)
        subparser.add_argument("--force-rerun-xgems", action="store_true")
        subparser.add_argument("--xgems-input-mode", choices=["species", "formula"], default="formula")
        subparser.add_argument("--gems-class-path", default="xgems:ChemicalEngineDicts")
        add_xgems_water_args(subparser)
        add_retry_water_args(subparser)
        add_reaction_model_args(subparser)
        subparser.add_argument("--reaction-model-signature", default=None)
        subparser.add_argument("--no-normalize", action="store_true")
        subparser.add_argument("--no-plots", action="store_true")
        subparser.add_argument("--fail-fast", action="store_true")

    add_request_facade_args(subparsers.add_parser("run-request"), needs_dat=True)
    add_request_facade_args(subparsers.add_parser("run-request-mock"), needs_dat=False)

    def add_cached_common(subparser: argparse.ArgumentParser, *, needs_dat: bool, needs_recipe: bool = True) -> None:
        if needs_dat:
            subparser.add_argument("--dat-lst", required=True, type=Path)
        else:
            subparser.add_argument("--dat-lst", required=False, type=Path)
        if needs_recipe:
            subparser.add_argument("--recipe", required=True)
        subparser.add_argument("--db", default=Path("data/inverse_gems_db"), type=Path)
        subparser.add_argument("--run-mode", choices=["reacted_only", "lower_bound_legacy"], default="reacted_only")
        subparser.add_argument("--temperature-celsius", type=float, default=20.0)
        subparser.add_argument("--pressure", type=float, default=None)
        subparser.add_argument("--force-rerun-xgems", action="store_true")
        subparser.add_argument("--xgems-input-mode", choices=["species", "formula"], default="formula")
        subparser.add_argument("--gems-class-path", default="xgems:ChemicalEngineDicts")
        add_xgems_water_args(subparser)
        add_retry_water_args(subparser)
        add_reaction_model_args(subparser)
        subparser.add_argument("--no-normalize", action="store_true")

    add_cached_common(subparsers.add_parser("forward-cached"), needs_dat=True)
    add_cached_common(subparsers.add_parser("forward-cached-mock"), needs_dat=False)
    batch = subparsers.add_parser("run-recipes-cached")
    add_cached_common(batch, needs_dat=True, needs_recipe=False)
    batch.add_argument("--recipes-csv", required=True, type=Path)

    inspect_chem = subparsers.add_parser("inspect-chemistry")
    inspect_chem.add_argument("--db", default=Path("data/inverse_gems_db"), type=Path)
    inspect_chem.add_argument("--chem-hash", required=True)

    inspect_recipe = subparsers.add_parser("inspect-recipe")
    inspect_recipe.add_argument("--db", default=Path("data/inverse_gems_db"), type=Path)
    inspect_recipe.add_argument("--recipe-id", required=True)

    compare = subparsers.add_parser("compare-recipes")
    compare.add_argument("--db", default=Path("data/inverse_gems_db"), type=Path)
    compare.add_argument("--recipe-id-a", required=True)
    compare.add_argument("--recipe-id-b", required=True)

    available = subparsers.add_parser("inspect-available-outputs")
    available.add_argument("--db", default=Path("data/inverse_gems_db"), type=Path)
    available.add_argument("--chem-hash", required=True)

    features = subparsers.add_parser("build-feature-table")
    features.add_argument("--db", default=Path("data/inverse_gems_db"), type=Path)
    features.add_argument("--selection", default=Path("configs/output_selection.yaml"), type=Path)
    features.add_argument("--out", required=True, type=Path)
    features.add_argument("--format", choices=["csv", "parquet"], default=None)
    features.add_argument("--recipe-id-prefix", default=None)
    features.add_argument("--material-system", default=None)
    features.add_argument("--age-days", type=float, default=None)
    features.add_argument("--age-tolerance", type=float, default=1.0e-12)

    diagnostics = subparsers.add_parser("feature-diagnostics")
    diagnostics.add_argument("--feature-table", required=True, type=Path)
    diagnostics.add_argument("--out", required=True, type=Path)
    diagnostics.add_argument("--correlation-threshold", type=float, default=0.98)
    diagnostics.add_argument("--sparse-threshold", type=float, default=0.01)
    diagnostics.add_argument("--no-plots", action="store_true")

    registry_diagnostics = subparsers.add_parser("model-registry-diagnostics")
    registry_diagnostics.add_argument(
        "--model-registry",
        default=Path("configs/design_query_model_registry.global_v1.yaml"),
        type=Path,
    )
    registry_diagnostics.add_argument("--out", required=True, type=Path)
    registry_diagnostics.add_argument("--reaction-model-id", default=None)
    registry_diagnostics.add_argument("--reaction-model-signature", default=None)
    registry_diagnostics.add_argument("--min-r2", type=float, default=0.70)
    registry_diagnostics.add_argument("--sparse-threshold", type=float, default=0.01)
    registry_diagnostics.add_argument("--min-range", type=float, default=1.0e-8)
    registry_diagnostics.add_argument("--ph-min-range", type=float, default=0.01)

    active_learning = subparsers.add_parser("recommend-active-learning-targets")
    active_learning.add_argument("--diagnostics", required=True, type=Path)
    active_learning.add_argument("--out", required=True, type=Path)
    active_learning.add_argument("--status", action="append", default=[])
    active_learning.add_argument("--target-kind", choices=["phase", "amount", "volume", "scalar"], action="append", default=[])
    active_learning.add_argument("--exclude-target", action="append", default=[])
    active_learning.add_argument("--min-r2", type=float, default=0.70)
    active_learning.add_argument("--max-targets", type=int, default=None)

    design_run_summary = subparsers.add_parser("summarize-design-runs")
    design_run_summary.add_argument(
        "--runs",
        required=True,
        nargs="+",
        type=Path,
        help="Completed run-design-query output directories to summarize.",
    )
    design_run_summary.add_argument("--out", required=True, type=Path)

    model_table = subparsers.add_parser("build-model-table")
    model_table.add_argument("--feature-table", required=True, type=Path)
    model_table.add_argument("--config", default=Path("configs/model_dataset.yaml"), type=Path)
    model_table.add_argument("--out", required=True, type=Path)
    model_table.add_argument("--format", choices=["csv", "parquet"], default=None)
    model_table.add_argument("--reaction-model-id", default=None)
    model_table.add_argument("--reaction-model-signature", default=None)

    chemistry_candidates = subparsers.add_parser("build-chemistry-candidate-table")
    chemistry_candidates.add_argument("--recipes-csv", required=True, type=Path)
    chemistry_candidates.add_argument("--out", required=True, type=Path)
    chemistry_candidates.add_argument("--dat-lst", default=None, type=Path)
    chemistry_candidates.add_argument("--run-mode", choices=["reacted_only", "lower_bound_legacy"], default="reacted_only")
    chemistry_candidates.add_argument("--temperature-celsius", type=float, default=20.0)
    chemistry_candidates.add_argument("--pressure", type=float, default=None)
    chemistry_candidates.add_argument("--format", choices=["csv", "parquet"], default=None)
    add_xgems_water_args(chemistry_candidates)
    add_reaction_model_args(chemistry_candidates)
    chemistry_candidates.add_argument("--no-normalize", action="store_true")

    domain = subparsers.add_parser("chemistry-domain-report")
    domain.add_argument("--reference-model-table", required=True, type=Path)
    domain.add_argument("--candidate-table", required=True, type=Path)
    domain.add_argument("--model-bundle", default=None, type=Path)
    domain.add_argument("--out", required=True, type=Path)
    domain.add_argument("--nearest-distance-warn", type=float, default=0.25)

    chemistry_design = subparsers.add_parser("run-chemistry-design-query")
    chemistry_design.add_argument("--query", required=True, type=Path)
    chemistry_design.add_argument("--out", required=True, type=Path)
    chemistry_design.add_argument("--model-bundle", required=True, type=Path)
    chemistry_design.add_argument("--reference-model-table", default=None, type=Path)
    chemistry_design.add_argument("--n-candidates", type=int, default=500)
    chemistry_design.add_argument("--seed", type=int, default=42)
    chemistry_design.add_argument("--sampling-config", default=Path("configs/sampling.yaml"), type=Path)
    chemistry_design.add_argument("--material-systems-config", default=Path("configs/material_systems.yaml"), type=Path)
    chemistry_design.add_argument("--strict-materials", action="store_true")
    chemistry_design.add_argument("--dat-lst", default=None, type=Path)
    chemistry_design.add_argument("--db", default=Path("data/inverse_gems_chemistry_design_db"), type=Path)
    chemistry_design.add_argument("--validate", action="store_true")
    chemistry_design.add_argument("--validation-top-k", type=int, default=None)
    chemistry_design.add_argument("--selection", default=Path("configs/output_selection.yaml"), type=Path)
    chemistry_design.add_argument("--run-mode", choices=["reacted_only", "lower_bound_legacy"], default="reacted_only")
    chemistry_design.add_argument("--temperature-celsius", type=float, default=20.0)
    chemistry_design.add_argument("--pressure", type=float, default=None)
    chemistry_design.add_argument("--force-rerun-xgems", action="store_true")
    chemistry_design.add_argument("--xgems-input-mode", choices=["species", "formula"], default="formula")
    chemistry_design.add_argument("--gems-class-path", default="xgems:ChemicalEngineDicts")
    add_xgems_water_args(chemistry_design)
    add_retry_water_args(chemistry_design)
    add_reaction_model_args(chemistry_design)
    chemistry_design.add_argument(
        "--target-availability-policy",
        choices=["ignore", "warn", "error"],
        default="warn",
    )
    chemistry_design.add_argument("--no-normalize", action="store_true")
    chemistry_design.add_argument("--fail-fast", action="store_true")

    init_global = subparsers.add_parser("init-global-chem-db")
    init_global.add_argument("--db", required=True, type=Path)
    init_global.add_argument("--schema", default=Path("configs/global_chemistry_db.yaml"), type=Path)

    import_global = subparsers.add_parser("import-global-chem-artifacts")
    import_global.add_argument("--db", required=True, type=Path)
    import_global.add_argument("--schema", default=Path("configs/global_chemistry_db.yaml"), type=Path)
    import_global.add_argument("--feature-table", default=None, type=Path)
    import_global.add_argument("--model-table", default=None, type=Path)
    import_global.add_argument("--model-bundle", default=None, type=Path)

    import_global_db = subparsers.add_parser("import-global-chem-db")
    import_global_db.add_argument("--db", required=True, type=Path)
    import_global_db.add_argument("--source-db", required=True, type=Path)
    import_global_db.add_argument("--schema", default=Path("configs/global_chemistry_db.yaml"), type=Path)
    import_global_db.add_argument("--no-copy-run-dirs", action="store_true")

    refresh_global = subparsers.add_parser("refresh-global-chem-db")
    refresh_global.add_argument("--db", required=True, type=Path)
    refresh_global.add_argument("--schema", default=Path("configs/global_chemistry_db.yaml"), type=Path)
    refresh_global.add_argument("--selection", default=None, type=Path)
    refresh_global.add_argument("--model-config", default=None, type=Path)

    train_global = subparsers.add_parser("train-global-chem-surrogate")
    train_global.add_argument("--db", required=True, type=Path)
    train_global.add_argument("--schema", default=Path("configs/global_chemistry_db.yaml"), type=Path)
    train_global.add_argument("--selection", default=None, type=Path)
    train_global.add_argument("--model-config", default=None, type=Path)
    train_global.add_argument("--surrogate-config", default=None, type=Path)
    train_global.add_argument("--no-refresh", action="store_true")
    train_global.add_argument("--no-save-model", action="store_true")

    global_coverage = subparsers.add_parser("global-chemistry-coverage")
    global_coverage.add_argument("--db", required=True, type=Path)
    global_coverage.add_argument("--out", required=True, type=Path)
    global_coverage.add_argument("--schema", default=Path("configs/global_chemistry_db.yaml"), type=Path)
    global_coverage.add_argument("--coverage-config", default=Path("configs/global_chemistry_coverage.yaml"), type=Path)
    global_coverage.add_argument("--material-systems-config", default=Path("configs/material_systems.yaml"), type=Path)
    global_coverage.add_argument("--target-metrics", default=None, type=Path)

    target_region = subparsers.add_parser("analyze-target-region")
    target_region.add_argument("--model-table", required=True, type=Path)
    target_region.add_argument("--target", required=True)
    target_region.add_argument("--out", required=True, type=Path)
    target_region.add_argument("--threshold", type=float, default=1.0e-30)
    target_region.add_argument("--top-n", type=int, default=50)
    target_region.add_argument("--prefer", choices=["amount", "volume"], default="amount")

    quality_cases = subparsers.add_parser("analyze-xgems-quality-cases")
    quality_cases.add_argument("--db", default=None, type=Path)
    quality_cases.add_argument("--model-table", default=None, type=Path)
    quality_cases.add_argument("--out", required=True, type=Path)
    quality_cases.add_argument("--material-system", default=None)
    quality_cases.add_argument("--all-cases", action="store_true", help="Export all filtered rows instead of only problem cases.")
    quality_cases.add_argument("--top-n", type=int, default=50)
    quality_cases.add_argument("--water-tolerance-g", type=float, default=1.0e-9)

    lookup_global = subparsers.add_parser("lookup-global-chem-db")
    lookup_global.add_argument("--db", required=True, type=Path)
    lookup_global.add_argument("--out", required=True, type=Path)
    lookup_global.add_argument("--schema", default=Path("configs/global_chemistry_db.yaml"), type=Path)
    lookup_global.add_argument("--recipes-csv", default=None, type=Path)
    lookup_global.add_argument("--candidate-table", default=None, type=Path)
    lookup_global.add_argument("--reference-model-table", default=None, type=Path)
    lookup_global.add_argument("--model-bundle", default=None, type=Path)
    lookup_global.add_argument("--dat-lst", default=None, type=Path)
    lookup_global.add_argument("--run-mode", choices=["reacted_only", "lower_bound_legacy"], default="reacted_only")
    lookup_global.add_argument("--temperature-celsius", type=float, default=20.0)
    lookup_global.add_argument("--pressure", type=float, default=None)
    lookup_global.add_argument("--nearest-distance-warn", type=float, default=None)
    add_xgems_water_args(lookup_global)
    add_reaction_model_args(lookup_global)

    acquire_global = subparsers.add_parser("acquire-global-chemistry")
    acquire_global.add_argument("--db", required=True, type=Path)
    acquire_global.add_argument("--out", required=True, type=Path)
    acquire_global.add_argument("--schema", default=Path("configs/global_chemistry_db.yaml"), type=Path)
    acquire_global.add_argument("--recipes-csv", default=None, type=Path)
    acquire_global.add_argument("--candidate-table", default=None, type=Path)
    acquire_global.add_argument("--reference-model-table", default=None, type=Path)
    acquire_global.add_argument("--model-bundle", default=None, type=Path)
    acquire_global.add_argument("--max-candidates", type=int, default=20)
    acquire_global.add_argument("--dat-lst", default=None, type=Path)
    acquire_global.add_argument("--run-mode", choices=["reacted_only", "lower_bound_legacy"], default="reacted_only")
    acquire_global.add_argument("--temperature-celsius", type=float, default=20.0)
    acquire_global.add_argument("--pressure", type=float, default=None)
    acquire_global.add_argument("--nearest-distance-warn", type=float, default=None)
    acquire_global.add_argument("--priority-target", action="append", default=[])
    acquire_global.add_argument("--priority-targets-from-diagnostics", default=None, type=Path)
    acquire_global.add_argument("--priority-target-status", action="append", default=[])
    acquire_global.add_argument("--priority-target-kind", choices=["phase", "amount", "volume", "scalar"], action="append", default=[])
    acquire_global.add_argument("--priority-target-reliability", action="append", default=[])
    acquire_global.add_argument("--priority-target-limit", type=int, default=None)
    acquire_global.add_argument("--target-priority-weight", type=float, default=None)
    acquire_global.add_argument("--target-region-table", action="append", default=[])
    acquire_global.add_argument("--target-region-weight", type=float, default=None)
    acquire_global.add_argument("--target-region-distance-scale", type=float, default=None)
    acquire_global.add_argument("--target-region-max-reference-rows", type=int, default=None)
    add_xgems_water_args(acquire_global)
    add_reaction_model_args(acquire_global)

    acquisition_cycle = subparsers.add_parser("run-global-acquisition-cycle")
    acquisition_cycle.add_argument("--db", required=True, type=Path)
    acquisition_cycle.add_argument("--out", required=True, type=Path)
    acquisition_cycle.add_argument("--schema", default=Path("configs/global_chemistry_db.yaml"), type=Path)
    acquisition_cycle.add_argument("--recipes-csv", default=None, type=Path)
    acquisition_cycle.add_argument("--candidate-table", default=None, type=Path)
    acquisition_cycle.add_argument("--reference-model-table", default=None, type=Path)
    acquisition_cycle.add_argument("--model-bundle", default=None, type=Path)
    acquisition_cycle.add_argument("--max-candidates", type=int, default=20)
    acquisition_cycle.add_argument("--dat-lst", default=None, type=Path)
    acquisition_cycle.add_argument("--mock", action="store_true")
    acquisition_cycle.add_argument("--skip-batch", action="store_true")
    acquisition_cycle.add_argument("--skip-refresh", action="store_true")
    acquisition_cycle.add_argument("--skip-train", action="store_true")
    acquisition_cycle.add_argument("--skip-coverage", action="store_true")
    acquisition_cycle.add_argument("--selection", default=None, type=Path)
    acquisition_cycle.add_argument("--model-config", default=None, type=Path)
    acquisition_cycle.add_argument("--surrogate-config", default=None, type=Path)
    acquisition_cycle.add_argument("--coverage-config", default=Path("configs/global_chemistry_coverage.yaml"), type=Path)
    acquisition_cycle.add_argument("--material-systems-config", default=Path("configs/material_systems.yaml"), type=Path)
    acquisition_cycle.add_argument("--force-rerun-xgems", action="store_true")
    acquisition_cycle.add_argument("--no-resume-batch", action="store_true")
    acquisition_cycle.add_argument("--fail-fast", action="store_true")
    acquisition_cycle.add_argument("--run-mode", choices=["reacted_only", "lower_bound_legacy"], default="reacted_only")
    acquisition_cycle.add_argument("--temperature-celsius", type=float, default=20.0)
    acquisition_cycle.add_argument("--pressure", type=float, default=None)
    acquisition_cycle.add_argument("--xgems-input-mode", choices=["species", "formula"], default="formula")
    acquisition_cycle.add_argument("--gems-class-path", default="xgems:ChemicalEngineDicts")
    acquisition_cycle.add_argument("--nearest-distance-warn", type=float, default=None)
    acquisition_cycle.add_argument("--priority-target", action="append", default=[])
    acquisition_cycle.add_argument("--priority-targets-from-diagnostics", default=None, type=Path)
    acquisition_cycle.add_argument("--priority-target-status", action="append", default=[])
    acquisition_cycle.add_argument("--priority-target-kind", choices=["phase", "amount", "volume", "scalar"], action="append", default=[])
    acquisition_cycle.add_argument("--priority-target-reliability", action="append", default=[])
    acquisition_cycle.add_argument("--priority-target-limit", type=int, default=None)
    acquisition_cycle.add_argument("--target-priority-weight", type=float, default=None)
    acquisition_cycle.add_argument("--target-region-table", action="append", default=[])
    acquisition_cycle.add_argument("--target-region-weight", type=float, default=None)
    acquisition_cycle.add_argument("--target-region-distance-scale", type=float, default=None)
    acquisition_cycle.add_argument("--target-region-max-reference-rows", type=int, default=None)
    add_xgems_water_args(acquisition_cycle)
    add_retry_water_args(acquisition_cycle)
    add_reaction_model_args(acquisition_cycle)

    global_forward = subparsers.add_parser("run-global-forward-query")
    global_forward.add_argument("--global-db", required=True, type=Path)
    global_forward.add_argument("--query", required=True, type=Path)
    global_forward.add_argument("--out", required=True, type=Path)
    global_forward.add_argument("--dat-lst", required=True, type=Path)
    global_forward.add_argument("--run-mode", choices=["reacted_only", "lower_bound_legacy"], default="reacted_only")
    global_forward.add_argument("--pressure", type=float, default=None)
    global_forward.add_argument("--force-rerun-xgems", action="store_true")
    global_forward.add_argument("--xgems-input-mode", choices=["species", "formula"], default="formula")
    global_forward.add_argument("--gems-class-path", default="xgems:ChemicalEngineDicts")
    add_xgems_water_args(global_forward)
    add_retry_water_args(global_forward)
    add_reaction_model_args(global_forward)
    global_forward.add_argument("--no-normalize", action="store_true")
    global_forward.add_argument("--no-plots", action="store_true")
    global_forward.add_argument("--fail-fast", action="store_true")

    global_forward_mock = subparsers.add_parser("run-global-forward-query-mock")
    global_forward_mock.add_argument("--global-db", required=True, type=Path)
    global_forward_mock.add_argument("--query", required=True, type=Path)
    global_forward_mock.add_argument("--out", required=True, type=Path)
    global_forward_mock.add_argument("--dat-lst", required=False, type=Path)
    global_forward_mock.add_argument("--run-mode", choices=["reacted_only", "lower_bound_legacy"], default="reacted_only")
    global_forward_mock.add_argument("--pressure", type=float, default=None)
    global_forward_mock.add_argument("--force-rerun-xgems", action="store_true")
    global_forward_mock.add_argument("--xgems-input-mode", choices=["species", "formula"], default="formula")
    global_forward_mock.add_argument("--gems-class-path", default="xgems:ChemicalEngineDicts")
    add_xgems_water_args(global_forward_mock)
    add_retry_water_args(global_forward_mock)
    add_reaction_model_args(global_forward_mock)
    global_forward_mock.add_argument("--no-normalize", action="store_true")
    global_forward_mock.add_argument("--no-plots", action="store_true")
    global_forward_mock.add_argument("--fail-fast", action="store_true")

    global_design = subparsers.add_parser("run-global-design-query")
    global_design.add_argument("--global-db", required=True, type=Path)
    global_design.add_argument("--query", required=True, type=Path)
    global_design.add_argument("--out", required=True, type=Path)
    global_design.add_argument("--model-bundle", default=None, type=Path)
    global_design.add_argument("--reference-model-table", default=None, type=Path)
    global_design.add_argument("--n-candidates", type=int, default=500)
    global_design.add_argument("--seed", type=int, default=42)
    global_design.add_argument("--sampling-config", default=Path("configs/sampling.yaml"), type=Path)
    global_design.add_argument("--material-systems-config", default=Path("configs/material_systems.yaml"), type=Path)
    global_design.add_argument("--strict-materials", action="store_true")
    global_design.add_argument("--dat-lst", default=None, type=Path)
    global_design.add_argument("--validate", action="store_true")
    global_design.add_argument(
        "--mock-validation",
        action="store_true",
        help="Use the mock xGEMS runner for the validation stage when --validate is set.",
    )
    global_design.add_argument("--validation-top-k", type=int, default=None)
    global_design.add_argument("--selection", default=Path("configs/output_selection.yaml"), type=Path)
    global_design.add_argument("--run-mode", choices=["reacted_only", "lower_bound_legacy"], default="reacted_only")
    global_design.add_argument("--temperature-celsius", type=float, default=20.0)
    global_design.add_argument("--pressure", type=float, default=None)
    global_design.add_argument("--force-rerun-xgems", action="store_true")
    global_design.add_argument("--xgems-input-mode", choices=["species", "formula"], default="formula")
    global_design.add_argument("--gems-class-path", default="xgems:ChemicalEngineDicts")
    add_xgems_water_args(global_design)
    add_retry_water_args(global_design)
    add_reaction_model_args(global_design)
    global_design.add_argument("--target-availability-policy", choices=["ignore", "warn", "error"], default="warn")
    global_design.add_argument("--no-normalize", action="store_true")
    global_design.add_argument("--fail-fast", action="store_true")

    surrogate = subparsers.add_parser("train-baseline-surrogate")
    surrogate.add_argument("--model-table", required=True, type=Path)
    surrogate.add_argument("--config", default=Path("configs/surrogate_baseline.yaml"), type=Path)
    surrogate.add_argument("--out", required=True, type=Path)
    surrogate.add_argument("--no-save-model", action="store_true")

    candidate = subparsers.add_parser("surrogate-candidate-search")
    candidate.add_argument("--query", required=True, type=Path)
    candidate.add_argument("--out", required=True, type=Path)
    candidate.add_argument("--model-table", default=None, type=Path)
    candidate.add_argument("--model-bundle", default=None, type=Path)
    candidate.add_argument("--reaction-model-id", default=None)
    candidate.add_argument("--reaction-model-signature", default=None)
    candidate.add_argument("--reaction-model-mismatch-policy", choices=["error", "warn", "ignore"], default=None)

    def add_validate_candidate_args(subparser: argparse.ArgumentParser, *, needs_dat: bool) -> None:
        subparser.add_argument("--candidates", required=True, type=Path)
        if needs_dat:
            subparser.add_argument("--dat-lst", required=True, type=Path)
        else:
            subparser.add_argument("--dat-lst", required=False, type=Path)
        subparser.add_argument("--db", required=True, type=Path)
        subparser.add_argument("--out", required=True, type=Path)
        subparser.add_argument("--top-k", type=int, default=10)
        subparser.add_argument("--selection", default=Path("configs/output_selection.yaml"), type=Path)
        subparser.add_argument("--run-mode", choices=["reacted_only", "lower_bound_legacy"], default="reacted_only")
        subparser.add_argument("--temperature-celsius", type=float, default=20.0)
        subparser.add_argument("--pressure", type=float, default=None)
        subparser.add_argument("--force-rerun-xgems", action="store_true")
        subparser.add_argument("--xgems-input-mode", choices=["species", "formula"], default="formula")
        subparser.add_argument("--gems-class-path", default="xgems:ChemicalEngineDicts")
        add_xgems_water_args(subparser)
        add_retry_water_args(subparser)
        add_reaction_model_args(subparser)
        subparser.add_argument("--no-normalize", action="store_true")
        subparser.add_argument("--fail-fast", action="store_true")

    add_validate_candidate_args(subparsers.add_parser("validate-candidates"), needs_dat=True)
    add_validate_candidate_args(subparsers.add_parser("validate-candidates-mock"), needs_dat=False)

    select = subparsers.add_parser("select-candidates")
    select.add_argument("--validation", required=True, type=Path)
    select.add_argument("--config", default=Path("configs/candidate_selection.example.yaml"), type=Path)
    select.add_argument("--out", required=True, type=Path)
    select.add_argument(
        "--feature-table",
        default=None,
        type=Path,
        help="Optional validated feature table. Defaults to validated_feature_table.csv beside --validation.",
    )

    gen = subparsers.add_parser("generate-recipes")
    gen.add_argument("--config", default=Path("configs/sampling.yaml"), type=Path)
    gen.add_argument("--n", required=True, type=int)
    gen.add_argument("--mode", choices=["general_simplex", "recipe_templates", "mixed"], default="mixed")
    gen.add_argument("--age-preset", default="early_dense_v1")
    gen.add_argument("--ages", default=None, help="Comma-separated explicit age_days list. Overrides --age-preset.")
    gen.add_argument("--age-sampling", choices=["preset", "uniform", "log_uniform"], default="preset")
    gen.add_argument("--age-min", type=float, default=0.1)
    gen.add_argument("--age-max", type=float, default=365.0)
    gen.add_argument("--age-count", type=int, default=1)
    gen.add_argument("--material-system", default=None, help="Material-system profile from configs/material_systems.yaml.")
    gen.add_argument("--material-systems", default=None, help="Comma-separated material-system profiles for mixed global sampling.")
    gen.add_argument("--material-systems-sampling", choices=["random", "balanced"], default="random")
    gen.add_argument("--material-systems-config", default=Path("configs/material_systems.yaml"), type=Path)
    gen.add_argument("--target-profile", default=None, help="Targeted sampling profile from configs/targeted_sampling.yaml.")
    gen.add_argument("--target-profiles-config", default=Path("configs/targeted_sampling.yaml"), type=Path)
    gen.add_argument("--strict-materials", action="store_true")
    gen.add_argument("--strict-allowed-materials", default=None, help="Comma-separated material names for strict sampling.")
    gen.add_argument("--recipe-id-prefix", default="base")
    gen.add_argument("--out", required=True, type=Path)
    gen.add_argument("--seed", type=int, default=42)

    expand = subparsers.add_parser("expand-ages")
    expand.add_argument("--recipes", required=True, type=Path)
    expand.add_argument("--age-preset", default="early_dense_v1")
    expand.add_argument("--ages", default=None)
    expand.add_argument("--out", required=True, type=Path)

    rb = subparsers.add_parser("run-batch-cached")
    rb.add_argument("--dat-lst", required=True, type=Path)
    rb.add_argument("--recipes", required=True, type=Path)
    rb.add_argument("--db", required=True, type=Path)
    rb.add_argument("--workers", type=int, default=1)
    rb.add_argument("--resume", action="store_true")
    rb.add_argument("--fail-fast", action="store_true")
    rb.add_argument("--force-rerun-xgems", action="store_true")
    rb.add_argument("--run-mode", choices=["reacted_only", "lower_bound_legacy"], default="reacted_only")
    rb.add_argument("--gems-class-path", default="xgems:ChemicalEngineDicts")
    rb.add_argument("--xgems-input-mode", choices=["species", "formula"], default="formula")
    add_xgems_water_args(rb)
    add_retry_water_args(rb)
    add_reaction_model_args(rb)
    rb.add_argument("--progress-every", type=int, default=25)

    rbm = subparsers.add_parser("run-batch-cached-mock")
    rbm.add_argument("--recipes", required=True, type=Path)
    rbm.add_argument("--db", required=True, type=Path)
    rbm.add_argument("--workers", type=int, default=1)
    rbm.add_argument("--resume", action="store_true")
    rbm.add_argument("--fail-fast", action="store_true")
    rbm.add_argument("--force-rerun-xgems", action="store_true")
    rbm.add_argument("--run-mode", choices=["reacted_only", "lower_bound_legacy"], default="reacted_only")
    add_xgems_water_args(rbm)
    add_retry_water_args(rbm)
    add_reaction_model_args(rbm)
    rbm.add_argument("--progress-every", type=int, default=25)

    batch_status = subparsers.add_parser("batch-status")
    batch_status.add_argument("--db", required=True, type=Path)
    batch_status.add_argument("--out", default=None, type=Path)

    summ = subparsers.add_parser("summarize-db")
    summ.add_argument("--db", required=True, type=Path)
    summ.add_argument("--selection", default=None, type=Path)
    summ.add_argument("--out", required=True, type=Path)
    summ.add_argument("--check-selected-outputs", action="store_true")

    backfill = subparsers.add_parser("backfill-reconstructed-volumes")
    backfill.add_argument("--db", required=True, type=Path)
    backfill.add_argument("--dat-lst", required=True, type=Path)
    backfill.add_argument("--limit", type=int, default=None)
    backfill.add_argument("--force", action="store_true")
    backfill.add_argument("--include-unlinked-complete", action="store_true")
    backfill.add_argument("--gems-class-path", default="xgems:ChemicalEngineDicts")
    backfill.add_argument("--xgems-input-mode", choices=["species", "formula"], default="formula")
    backfill.add_argument("--xgems-phase-volume-unit", choices=["m3", "cm3", "cm^3", "cc", "ml"], default="m3")
    backfill.add_argument("--no-backup", action="store_true")
    backfill.add_argument("--no-reuse-runner", action="store_true")
    backfill.add_argument("--progress-every", type=int, default=100)

    inv = subparsers.add_parser("inverse-query")
    inv.add_argument("--feature-table", required=True, type=Path)
    inv.add_argument("--query", required=True, type=Path)
    inv.add_argument("--out", required=True, type=Path)

    compile_query = subparsers.add_parser("compile-design-query")
    compile_query.add_argument("--query", required=True, type=Path)
    compile_query.add_argument("--out", required=True, type=Path)
    compile_query.add_argument("--model-table", default=None, type=Path)
    compile_query.add_argument("--model-bundle", default=None, type=Path)
    compile_query.add_argument("--model-registry", default=Path("configs/design_query_model_registry.global_v1.yaml"), type=Path)
    compile_query.add_argument("--reaction-model-id", default=None)
    compile_query.add_argument("--reaction-model-signature", default=None)
    compile_query.add_argument("--reaction-model-config", default=None, type=Path)
    compile_query.add_argument(
        "--target-availability-policy",
        choices=["ignore", "warn", "error"],
        default="warn",
    )

    route_query = subparsers.add_parser("route-design-query")
    route_query.add_argument("--query", required=True, type=Path)
    route_query.add_argument("--out", required=True, type=Path)
    route_query.add_argument("--model-registry", default=Path("configs/design_query_model_registry.global_v1.yaml"), type=Path)
    route_query.add_argument("--material-systems-config", default=Path("configs/material_systems.yaml"), type=Path)
    route_query.add_argument("--reaction-model-id", default=None)
    route_query.add_argument("--reaction-model-signature", default=None)
    route_query.add_argument("--reaction-model-config", default=None, type=Path)
    route_query.add_argument("--target-policy", choices=["recommended", "allow_caution"], default="recommended")
    route_query.add_argument("--default-age-days", type=float, default=28.0)

    validate_query = subparsers.add_parser("validate-design-query")
    validate_query.add_argument("--query", required=True, type=Path)
    validate_query.add_argument("--model-registry", default=Path("configs/design_query_model_registry.global_v1.yaml"), type=Path)
    validate_query.add_argument("--reaction-model-id", default=None)
    validate_query.add_argument("--reaction-model-signature", default=None)
    validate_query.add_argument("--reaction-model-config", default=None, type=Path)
    validate_query.add_argument("--require-model-paths", action="store_true")

    schema = subparsers.add_parser("design-query-schema")
    schema.add_argument("--out", default=None, type=Path)

    run_query = subparsers.add_parser("run-design-query")
    run_query.add_argument("--query", required=True, type=Path)
    run_query.add_argument("--out", required=True, type=Path)
    run_query.add_argument("--db", default=Path("data/inverse_gems_design_query_db"), type=Path)
    run_query.add_argument("--dat-lst", default=None, type=Path)
    run_query.add_argument("--model-table", default=None, type=Path)
    run_query.add_argument("--model-bundle", default=None, type=Path)
    run_query.add_argument("--model-registry", default=Path("configs/design_query_model_registry.global_v1.yaml"), type=Path)
    run_query.add_argument("--skip-validation", action="store_true")
    run_query.add_argument("--mock-validation", action="store_true")
    run_query.add_argument("--validation-top-k", type=int, default=None)
    run_query.add_argument("--selection", default=Path("configs/output_selection.yaml"), type=Path)
    run_query.add_argument("--run-mode", choices=["reacted_only", "lower_bound_legacy"], default="reacted_only")
    run_query.add_argument("--temperature-celsius", type=float, default=20.0)
    run_query.add_argument("--pressure", type=float, default=None)
    run_query.add_argument("--force-rerun-xgems", action="store_true")
    run_query.add_argument("--xgems-input-mode", choices=["species", "formula"], default="formula")
    run_query.add_argument("--gems-class-path", default="xgems:ChemicalEngineDicts")
    add_xgems_water_args(run_query)
    add_retry_water_args(run_query)
    add_reaction_model_args(run_query)
    run_query.add_argument("--reaction-model-signature", default=None)
    run_query.add_argument(
        "--target-availability-policy",
        choices=["ignore", "warn", "error"],
        default="warn",
    )
    run_query.add_argument("--no-normalize", action="store_true")
    run_query.add_argument("--fail-fast", action="store_true")

    forward_query_schema = subparsers.add_parser("forward-query-schema")
    forward_query_schema.add_argument("--out", default=None, type=Path)

    validate_forward = subparsers.add_parser("validate-forward-query")
    validate_forward.add_argument("--query", required=True, type=Path)

    def add_forward_query_args(subparser: argparse.ArgumentParser, *, needs_dat: bool) -> None:
        subparser.add_argument("--query", required=True, type=Path)
        subparser.add_argument("--out", required=True, type=Path)
        subparser.add_argument("--db", default=Path("data/inverse_gems_forward_query_db"), type=Path)
        if needs_dat:
            subparser.add_argument("--dat-lst", required=True, type=Path)
        else:
            subparser.add_argument("--dat-lst", default=None, type=Path)
        subparser.add_argument("--run-mode", choices=["reacted_only", "lower_bound_legacy"], default="reacted_only")
        subparser.add_argument("--pressure", type=float, default=None)
        subparser.add_argument("--force-rerun-xgems", action="store_true")
        subparser.add_argument("--xgems-input-mode", choices=["species", "formula"], default="formula")
        subparser.add_argument("--gems-class-path", default="xgems:ChemicalEngineDicts")
        add_xgems_water_args(subparser)
        add_retry_water_args(subparser)
        add_reaction_model_args(subparser)
        subparser.add_argument("--no-plots", action="store_true")
        subparser.add_argument("--no-normalize", action="store_true")
        subparser.add_argument("--fail-fast", action="store_true")

    add_forward_query_args(subparsers.add_parser("run-forward-query"), needs_dat=True)
    add_forward_query_args(subparsers.add_parser("run-forward-query-mock"), needs_dat=False)

    task_query_schema = subparsers.add_parser("task-query-schema")
    task_query_schema.add_argument("--out", default=None, type=Path)

    validate_task = subparsers.add_parser("validate-task-query")
    validate_task.add_argument("--query", required=True, type=Path)

    preview_task = subparsers.add_parser("preview-task-query")
    preview_task.add_argument("--query", required=True, type=Path)
    preview_task.add_argument("--out", required=True, type=Path)
    preview_task.add_argument("--model-registry", default=Path("configs/design_query_model_registry.global_v1.yaml"), type=Path)
    preview_task.add_argument("--material-systems-config", default=Path("configs/material_systems.yaml"), type=Path)
    preview_task.add_argument("--route-target-policy", choices=["recommended", "allow_caution"], default="recommended")
    preview_task.add_argument("--reaction-model-id", default=None)
    preview_task.add_argument("--reaction-model-signature", default=None)
    preview_task.add_argument("--reaction-model-config", default=None, type=Path)
    preview_task.add_argument("--default-age-days", type=float, default=28.0)

    router_prompt = subparsers.add_parser("render-task-router-prompt")
    router_prompt.add_argument("--out", required=True, type=Path)
    router_prompt.add_argument("--prompt", default=Path("configs/llm_task_router.prompt.md"), type=Path)
    router_prompt.add_argument("--schema", default=Path("configs/task_query.schema.json"), type=Path)
    router_prompt.add_argument("--examples", default=Path("configs/task_query.examples.yaml"), type=Path)

    validate_llm = subparsers.add_parser("validate-llm-task-output")
    validate_llm.add_argument("--input", required=True, type=Path)
    validate_llm.add_argument("--out", default=None, type=Path)
    validate_llm.add_argument("--user-request", default=None)
    validate_llm.add_argument("--base-prompt", default=Path("configs/llm_task_router.prompt.md"), type=Path)
    validate_llm.add_argument("--fail-on-invalid", action="store_true")

    parse_openai = subparsers.add_parser("parse-task-query-openai")
    parse_openai.add_argument("--request", default=None)
    parse_openai.add_argument("--request-file", default=None, type=Path)
    parse_openai.add_argument("--out", required=True, type=Path)
    parse_openai.add_argument("--llm-config", default=Path("configs/openai_task_router.yaml"), type=Path)
    parse_openai.add_argument("--model", default=None)
    parse_openai.add_argument("--max-repairs", default=None, type=int)
    parse_openai.add_argument("--model-registry", default=Path("configs/design_query_model_registry.global_v1.yaml"), type=Path)
    parse_openai.add_argument("--material-systems-config", default=Path("configs/material_systems.yaml"), type=Path)
    parse_openai.add_argument("--route-target-policy", choices=["recommended", "allow_caution"], default="recommended")
    parse_openai.add_argument("--reaction-model-id", default=None)
    parse_openai.add_argument("--reaction-model-signature", default=None)
    parse_openai.add_argument("--reaction-model-config", default=None, type=Path)

    def add_task_query_args(subparser: argparse.ArgumentParser, *, needs_dat: bool) -> None:
        subparser.add_argument("--query", required=True, type=Path)
        subparser.add_argument("--out", required=True, type=Path)
        subparser.add_argument("--db", default=Path("data/inverse_gems_task_query_db"), type=Path)
        subparser.add_argument("--dat-lst", default=None, type=Path)
        subparser.add_argument("--model-table", default=None, type=Path)
        subparser.add_argument("--model-bundle", default=None, type=Path)
        subparser.add_argument("--global-db", default=None, type=Path)
        subparser.add_argument("--global-n-candidates", type=int, default=500)
        subparser.add_argument("--strict-materials", action="store_true")
        subparser.add_argument("--model-registry", default=Path("configs/design_query_model_registry.global_v1.yaml"), type=Path)
        subparser.add_argument("--material-systems-config", default=Path("configs/material_systems.yaml"), type=Path)
        subparser.add_argument("--route-target-policy", choices=["recommended", "allow_caution"], default="recommended")
        subparser.add_argument("--skip-validation", action="store_true")
        subparser.add_argument("--validation-top-k", type=int, default=None)
        subparser.add_argument("--selection", default=Path("configs/output_selection.yaml"), type=Path)
        subparser.add_argument("--run-mode", choices=["reacted_only", "lower_bound_legacy"], default="reacted_only")
        subparser.add_argument("--temperature-celsius", type=float, default=None)
        subparser.add_argument("--pressure", type=float, default=None)
        subparser.add_argument("--force-rerun-xgems", action="store_true")
        subparser.add_argument("--xgems-input-mode", choices=["species", "formula"], default="formula")
        subparser.add_argument("--gems-class-path", default="xgems:ChemicalEngineDicts")
        add_xgems_water_args(subparser)
        add_retry_water_args(subparser)
        add_reaction_model_args(subparser)
        subparser.add_argument("--reaction-model-signature", default=None)
        subparser.add_argument("--no-plots", action="store_true")
        subparser.add_argument("--no-normalize", action="store_true")
        subparser.add_argument("--fail-fast", action="store_true")

    add_task_query_args(subparsers.add_parser("run-task-query"), needs_dat=True)
    add_task_query_args(subparsers.add_parser("run-task-query-mock"), needs_dat=False)

    def add_confirmed_task_query_args(subparser: argparse.ArgumentParser, *, needs_dat: bool) -> None:
        subparser.add_argument("--preview", required=True, type=Path, help="Directory containing task_query.yaml and parsed_query_preview.json.")
        subparser.add_argument("--out", required=True, type=Path)
        subparser.add_argument("--db", default=Path("data/inverse_gems_task_query_db"), type=Path)
        subparser.add_argument("--dat-lst", default=None, type=Path)
        subparser.add_argument("--confirm-preview", action="store_true", help="Required acknowledgement that the preview has been reviewed.")
        subparser.add_argument("--allow-preview-errors", action="store_true")
        subparser.add_argument("--fail-on-preview-warnings", action="store_true")
        subparser.add_argument("--model-table", default=None, type=Path)
        subparser.add_argument("--model-bundle", default=None, type=Path)
        subparser.add_argument("--global-db", default=None, type=Path)
        subparser.add_argument("--global-n-candidates", type=int, default=500)
        subparser.add_argument("--strict-materials", action="store_true")
        subparser.add_argument("--model-registry", default=Path("configs/design_query_model_registry.global_v1.yaml"), type=Path)
        subparser.add_argument("--material-systems-config", default=Path("configs/material_systems.yaml"), type=Path)
        subparser.add_argument("--route-target-policy", choices=["recommended", "allow_caution"], default="recommended")
        subparser.add_argument("--skip-validation", action="store_true")
        subparser.add_argument("--validation-top-k", type=int, default=None)
        subparser.add_argument("--selection", default=Path("configs/output_selection.yaml"), type=Path)
        subparser.add_argument("--run-mode", choices=["reacted_only", "lower_bound_legacy"], default="reacted_only")
        subparser.add_argument("--temperature-celsius", type=float, default=None)
        subparser.add_argument("--pressure", type=float, default=None)
        subparser.add_argument("--force-rerun-xgems", action="store_true")
        subparser.add_argument("--xgems-input-mode", choices=["species", "formula"], default="formula")
        subparser.add_argument("--gems-class-path", default="xgems:ChemicalEngineDicts")
        add_xgems_water_args(subparser)
        add_retry_water_args(subparser)
        add_reaction_model_args(subparser)
        subparser.add_argument("--reaction-model-signature", default=None)
        subparser.add_argument("--no-plots", action="store_true")
        subparser.add_argument("--no-normalize", action="store_true")
        subparser.add_argument("--fail-fast", action="store_true")

    add_confirmed_task_query_args(subparsers.add_parser("run-confirmed-task-query"), needs_dat=True)
    add_confirmed_task_query_args(subparsers.add_parser("run-confirmed-task-query-mock"), needs_dat=False)

    def add_openai_run_args(subparser: argparse.ArgumentParser, *, needs_dat: bool) -> None:
        subparser.add_argument("--request", default=None)
        subparser.add_argument("--request-file", default=None, type=Path)
        subparser.add_argument("--out", required=True, type=Path)
        subparser.add_argument("--db", default=Path("data/inverse_gems_openai_task_db"), type=Path)
        subparser.add_argument("--dat-lst", default=None, type=Path)
        subparser.add_argument("--llm-config", default=Path("configs/openai_task_router.yaml"), type=Path)
        subparser.add_argument("--model", default=None)
        subparser.add_argument("--max-repairs", default=None, type=int)
        subparser.add_argument("--model-table", default=None, type=Path)
        subparser.add_argument("--model-bundle", default=None, type=Path)
        subparser.add_argument("--global-db", default=None, type=Path)
        subparser.add_argument("--global-n-candidates", type=int, default=500)
        subparser.add_argument("--strict-materials", action="store_true")
        subparser.add_argument("--model-registry", default=Path("configs/design_query_model_registry.global_v1.yaml"), type=Path)
        subparser.add_argument("--material-systems-config", default=Path("configs/material_systems.yaml"), type=Path)
        subparser.add_argument("--route-target-policy", choices=["recommended", "allow_caution"], default="recommended")
        subparser.add_argument("--skip-validation", action="store_true")
        subparser.add_argument("--validation-top-k", type=int, default=None)
        subparser.add_argument("--selection", default=Path("configs/output_selection.yaml"), type=Path)
        subparser.add_argument("--run-mode", choices=["reacted_only", "lower_bound_legacy"], default="reacted_only")
        subparser.add_argument("--temperature-celsius", type=float, default=None)
        subparser.add_argument("--pressure", type=float, default=None)
        subparser.add_argument("--force-rerun-xgems", action="store_true")
        subparser.add_argument("--xgems-input-mode", choices=["species", "formula"], default="formula")
        subparser.add_argument("--gems-class-path", default="xgems:ChemicalEngineDicts")
        add_xgems_water_args(subparser)
        add_retry_water_args(subparser)
        add_reaction_model_args(subparser)
        subparser.add_argument("--reaction-model-signature", default=None)
        subparser.add_argument("--no-plots", action="store_true")
        subparser.add_argument("--no-normalize", action="store_true")
        subparser.add_argument("--fail-fast", action="store_true")

    add_openai_run_args(subparsers.add_parser("run-openai-task-query"), needs_dat=True)
    add_openai_run_args(subparsers.add_parser("run-openai-task-query-mock"), needs_dat=False)

    def add_inverse_forward_workflow_args(subparser: argparse.ArgumentParser, *, needs_dat: bool) -> None:
        subparser.add_argument("--inverse-request", default=None)
        subparser.add_argument("--inverse-request-file", default=None, type=Path)
        subparser.add_argument("--out", required=True, type=Path)
        subparser.add_argument("--db", default=Path("data/inverse_gems_inverse_forward_workflow_db"), type=Path)
        if needs_dat:
            subparser.add_argument("--dat-lst", default=None, type=Path)
        else:
            subparser.add_argument("--dat-lst", default=None, type=Path)
        subparser.add_argument("--llm-config", default=Path("configs/openai_task_router.yaml"), type=Path)
        subparser.add_argument("--model", default=None)
        subparser.add_argument("--max-repairs", default=None, type=int)
        subparser.add_argument("--global-db", default=None, type=Path)
        subparser.add_argument("--global-n-candidates", type=int, default=500)
        subparser.add_argument("--strict-materials", action="store_true")
        subparser.add_argument("--model-registry", default=Path("configs/design_query_model_registry.global_v1.yaml"), type=Path)
        subparser.add_argument("--material-systems-config", default=Path("configs/material_systems.yaml"), type=Path)
        subparser.add_argument("--route-target-policy", choices=["recommended", "allow_caution"], default="recommended")
        subparser.add_argument("--validation-top-k", type=int, default=5)
        subparser.add_argument("--selection", default=Path("configs/output_selection.yaml"), type=Path)
        subparser.add_argument("--candidate-rank", type=int, default=1)
        subparser.add_argument("--forward-age-values", type=_parse_float_list, default=None)
        subparser.add_argument("--forward-age-start", type=float, default=None)
        subparser.add_argument("--forward-age-stop", type=float, default=None)
        subparser.add_argument("--forward-age-points", type=int, default=24)
        subparser.add_argument("--forward-age-spacing", choices=["linear", "log"], default="log")
        subparser.add_argument("--forward-phase", action="append", default=[])
        subparser.add_argument("--forward-phase-group", action="append", default=[])
        subparser.add_argument("--forward-scalar", action="append", default=[])
        subparser.add_argument("--forward-table-limit", type=int, default=20)
        subparser.add_argument("--forward-top-phases", type=int, default=8)
        subparser.add_argument("--forward-narrative-language", choices=["ko", "en"], default="ko")
        subparser.add_argument("--run-mode", choices=["reacted_only", "lower_bound_legacy"], default="reacted_only")
        subparser.add_argument("--temperature-celsius", type=float, default=None)
        subparser.add_argument("--pressure", type=float, default=None)
        subparser.add_argument("--force-rerun-xgems", action="store_true")
        subparser.add_argument("--xgems-input-mode", choices=["species", "formula"], default="formula")
        subparser.add_argument("--gems-class-path", default="xgems:ChemicalEngineDicts")
        add_xgems_water_args(subparser)
        add_retry_water_args(subparser)
        add_reaction_model_args(subparser)
        subparser.add_argument("--reaction-model-signature", default=None)
        subparser.add_argument("--no-plots", action="store_true")
        subparser.add_argument("--no-normalize", action="store_true")
        subparser.add_argument("--fail-fast", action="store_true")

    add_inverse_forward_workflow_args(subparsers.add_parser("run-inverse-forward-workflow"), needs_dat=True)
    add_inverse_forward_workflow_args(subparsers.add_parser("run-inverse-forward-workflow-mock"), needs_dat=False)


    sensitivity = subparsers.add_parser("sensitivity-analysis")
    sensitivity.add_argument("--out", required=True, type=Path)
    sensitivity.add_argument("--recipe", action="append", required=True, dest="recipes",
                             help="Recipe text; repeat for multiple recipes.")
    sensitivity.add_argument("--rel-delta", type=float, default=0.2)
    sensitivity.add_argument("--use-mock", action="store_true")
    sensitivity.add_argument("--dat-lst", default=None, type=Path)
    sensitivity.add_argument("--parameters", default=None,
                             help="Comma-separated parameter paths (default: built-in set).")
    sensitivity.add_argument("--track-phases", default=None, help="Comma-separated raw phase names.")

    calibrate = subparsers.add_parser("calibrate-scm-kinetics")
    calibrate.add_argument("--data", required=True, type=Path, help="CSV with scm, age_d, dor columns.")
    calibrate.add_argument("--out", required=True, type=Path)
    calibrate.add_argument("--model", default="five_param_logistic")
    calibrate.add_argument("--id", dest="config_id", default=None)
    calibrate.add_argument("--scms", default=None, help="Comma-separated SCM names (default: all in data).")
    calibrate.add_argument("--fix", default=None, help="Comma-separated fixed params, e.g. A=0,G=1.")
    calibrate.add_argument("--min-points", type=int, default=8)

    cv = subparsers.add_parser("evaluate-surrogate-cv")
    cv.add_argument("--model-table", required=True, type=Path)
    cv.add_argument("--out", required=True, type=Path)
    cv.add_argument("--config", default=None, type=Path)
    cv.add_argument("--repeats", type=int, default=5)
    cv.add_argument("--targets", default=None, help="Comma-separated target names (short names allowed).")
    return parser


def _recipe_text_from_csv_row(row: dict[str, str]) -> str:
    parts: list[str] = []
    name_map = {
        "OPC": "OPC",
        "slag": "slag",
        "fly_ash": "fly ash",
        "metakaolin": "metakaolin",
        "silica_fume": "silica fume",
        "limestone": "limestone",
        "gypsum": "gypsum",
    }
    for key, label in name_map.items():
        value = row.get(key)
        if value not in (None, "") and float(value) != 0.0:
            parts.append(f"{label} {float(value):g}")
    if row.get("w_b") not in (None, ""):
        parts.append(f"w/b {float(row['w_b']):g}")
    elif row.get("water_g") not in (None, ""):
        parts.append(f"water {float(row['water_g']):g}")
    else:
        raise ValueError("Each recipes CSV row must include either w_b or water_g.")
    age = row.get("age_days") or row.get("age")
    if age in (None, ""):
        raise ValueError("Each recipes CSV row must include age_days.")
    parts.append(f"age {float(age):g}")
    return ", ".join(parts)


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _request_text_from_args(args: argparse.Namespace) -> str:
    if getattr(args, "request", None) and getattr(args, "request_file", None):
        raise ValueError("Use either --request or --request-file, not both.")
    if getattr(args, "request", None):
        return str(args.request)
    if getattr(args, "request_file", None):
        return Path(args.request_file).read_text(encoding="utf-8")
    raise ValueError("Provide a natural-language request with --request or --request-file.")


def _inverse_request_text_from_args(args: argparse.Namespace) -> str:
    if getattr(args, "inverse_request", None) and getattr(args, "inverse_request_file", None):
        raise ValueError("Use either --inverse-request or --inverse-request-file, not both.")
    if getattr(args, "inverse_request", None):
        return str(args.inverse_request)
    if getattr(args, "inverse_request_file", None):
        return Path(args.inverse_request_file).read_text(encoding="utf-8")
    raise ValueError("Provide a natural-language inverse request with --inverse-request or --inverse-request-file.")


def _inspect_chemistry(args: argparse.Namespace) -> int:
    db = InverseGemsDatabase(args.db)
    chem = db.get_chemistry_run(args.chem_hash)
    if not chem:
        raise ValueError(f"Unknown chem_hash: {args.chem_hash}")
    available = inspect_available_outputs(args.db, args.chem_hash)
    _print_json(
        {
            "chemistry_run": chem,
            "canonical_vector": json.loads(chem.get("canonical_vector_json") or "{}"),
            "oxide_equivalent_vector": json.loads(chem.get("oxide_equivalent_vector_json") or "{}"),
            "linked_recipe_ids": db.linked_recipe_ids(args.chem_hash),
            "available_outputs": available,
        }
    )
    return 0


def _inspect_recipe(args: argparse.Namespace) -> int:
    db = InverseGemsDatabase(args.db)
    recipe = db.get_recipe_run(args.recipe_id)
    if not recipe:
        raise ValueError(f"Unknown recipe_id: {args.recipe_id}")
    source_rows = db.source_rows_for_recipe(args.recipe_id)
    summary: dict[str, dict[str, float]] = {}
    for row in source_rows:
        material = str(row["source_material"])
        summary.setdefault(material, {"reacted_mass_g_row_sum": 0.0, "unreacted_mass_g_row_sum": 0.0})
        summary[material]["reacted_mass_g_row_sum"] += float(row["reacted_mass_g"])
        summary[material]["unreacted_mass_g_row_sum"] += float(row["unreacted_mass_g"])
    _print_json({"recipe_run": recipe, "source_contribution_summary": summary, "source_rows_count": len(source_rows)})
    return 0


def _compare_recipes(args: argparse.Namespace) -> int:
    db = InverseGemsDatabase(args.db)
    a = db.get_recipe_run(args.recipe_id_a)
    b = db.get_recipe_run(args.recipe_id_b)
    if not a or not b:
        raise ValueError("Both recipe IDs must exist.")
    _print_json(
        {
            "same_chem_hash": a["chem_hash"] == b["chem_hash"],
            "recipe_a": json.loads(a["recipe_json"]),
            "recipe_b": json.loads(b["recipe_json"]),
            "reaction_degrees_a": json.loads(a["reaction_degrees_json"]),
            "reaction_degrees_b": json.loads(b["reaction_degrees_json"]),
            "unreacted_masses_a": json.loads(a["unreacted_masses_json"]),
            "unreacted_masses_b": json.loads(b["unreacted_masses_json"]),
            "porosity_a": a["porosity"],
            "porosity_b": b["porosity"],
        }
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "sensitivity-analysis":
            from .sensitivity import run_reaction_parameter_sensitivity

            report = run_reaction_parameter_sensitivity(
                out=args.out,
                recipes=args.recipes,
                parameters=_parse_str_list(args.parameters),
                rel_delta=args.rel_delta,
                use_mock=args.use_mock,
                dat_lst=args.dat_lst,
                track_phases=_parse_str_list(args.track_phases),
            )
            _print_json(
                {
                    "run_count": report["run_count"],
                    "warnings": report["warnings"],
                    "outputs": report["outputs"],
                }
            )
            return 0
        if args.command == "calibrate-scm-kinetics":
            from .kinetics_calibration import calibrate_scm_kinetics

            fixed = {}
            for pair in (args.fix or "").split(","):
                pair = pair.strip()
                if pair:
                    key, _, value = pair.partition("=")
                    fixed[key.strip()] = float(value)
            report = calibrate_scm_kinetics(
                data_csv=args.data,
                out=args.out,
                model=args.model,
                config_id=args.config_id,
                scms=_parse_str_list(args.scms),
                fixed_params=fixed or None,
                min_points=args.min_points,
            )
            _print_json(
                {
                    "config_path": report["config_path"],
                    "id": report["id"],
                    "fits": {name: {"r2": fit["r2"], "rmse": fit["rmse"], "n": fit["n_points"]} for name, fit in report["fits"].items()},
                    "warnings": report["warnings"],
                }
            )
            return 0
        if args.command == "evaluate-surrogate-cv":
            from .surrogate_cv import evaluate_surrogate_repeated_cv

            out_dir = evaluate_surrogate_repeated_cv(
                model_table=args.model_table,
                out=args.out,
                config=args.config,
                n_repeats=args.repeats,
                targets=_parse_str_list(args.targets),
            )
            print(out_dir)
            return 0
        if args.command == "check-env":
            report = check_environment(
                dat_lst=args.dat_lst,
                gems_class_path=args.gems_class_path,
                xgems_input_mode=args.xgems_input_mode,
                require_xgems=args.require_xgems,
                instantiate_runner=args.instantiate_runner,
                out=args.out,
            )
            _print_json(report)
            return 0 if report.get("ok") else 1
        if args.command == "preflight-xgems-input":
            report = run_xgems_input_preflight(
                recipe_text=args.recipe,
                out=args.out,
                dat_lst=args.dat_lst,
                run_mode=args.run_mode,
                normalize=not args.no_normalize,
                allow_non_100=args.no_normalize,
                temperature_celsius=args.temperature_celsius,
                xgems_input_mode=args.xgems_input_mode,
                gems_class_path=args.gems_class_path,
                xgems_water_mode=args.xgems_water_mode,
                xgems_water_factor=args.xgems_water_factor,
                xgems_water_g=args.xgems_water_g,
                xgems_water_w_b=args.xgems_water_w_b,
                reaction_model_id=args.reaction_model_id,
                reaction_model_config=args.reaction_model_config,
                instantiate_runner=args.instantiate_runner,
                table_limit=args.table_limit,
            )
            _print_json({"ok": report["ok"], "flags": report["flags"], "out": str(args.out)})
            return 0 if report.get("ok") else 1
        if args.command in {"acceptance-real", "acceptance-mock"}:
            report = run_acceptance_suite(
                out=args.out,
                dat_lst=getattr(args, "dat_lst", None),
                use_mock=args.command == "acceptance-mock",
                db=args.db,
                case_ids=args.case or None,
                run_mode=args.run_mode,
                normalize=not args.no_normalize,
                allow_non_100=args.no_normalize,
                temperature_celsius=args.temperature_celsius,
                pressure=args.pressure,
                force_rerun_xgems=args.force_rerun_xgems,
                gems_class_path=args.gems_class_path,
                xgems_input_mode=args.xgems_input_mode,
                xgems_water_mode=args.xgems_water_mode,
                xgems_water_factor=args.xgems_water_factor,
                xgems_water_g=args.xgems_water_g,
                xgems_water_w_b=args.xgems_water_w_b,
                retry_water_on_failure=args.retry_water_on_failure,
                retry_water_cap_w_b_ladder=args.retry_water_cap_w_b_ladder,
                retry_water_up_w_b_ladder=args.retry_water_up_w_b_ladder,
                retry_water_min_w_b=args.retry_water_min_w_b,
                disable_plots=not args.with_plots,
                top_n_phases=args.top_n_phases,
                fail_fast=args.fail_fast,
            )
            _print_json(report["summary"])
            return 0 if report["summary"].get("ok") else 1
        if args.command == "summarize-forward-result":
            print(
                summarize_forward_result(
                    run=args.run,
                    phases=args.phase,
                    phase_groups=args.phase_group,
                    scalars=args.scalar,
                    out=args.out,
                    top_phases=args.top_phases,
                    table_limit=args.table_limit,
                )
            )
            return 0
        if args.command == "write-forward-answer":
            summary_path = args.summary or (args.run / "response_summary.json" if args.run else None)
            if summary_path is None:
                raise ValueError("write-forward-answer requires either --summary or --run.")
            print(write_forward_answer(summary=summary_path, out=args.out, table_limit=args.table_limit))
            return 0
        if args.command == "write-forward-narrative":
            answer_path = args.answer or (args.run / "answer.json" if args.run else None)
            if answer_path is None:
                raise ValueError("write-forward-narrative requires either --answer or --run.")
            print(
                write_forward_narrative(
                    answer=answer_path,
                    out=args.out,
                    language=args.language,
                    use_openai=args.use_openai,
                    model=args.model,
                    max_output_tokens=args.max_output_tokens,
                    temperature=args.temperature,
                )
            )
            return 0
        if args.command in {"run-request", "run-request-mock"}:
            request_text = None
            if getattr(args, "request", None) or getattr(args, "request_file", None):
                request_text = _request_text_from_args(args)
            result = run_request(
                request=request_text,
                task_query=args.task_query,
                forward_query=args.forward_query,
                confirmed_preview=args.confirmed_preview,
                confirm_preview=args.confirm_preview,
                allow_preview_errors=args.allow_preview_errors,
                fail_on_preview_warnings=args.fail_on_preview_warnings,
                out=args.out,
                db=args.db,
                dat_lst=getattr(args, "dat_lst", None),
                use_mock=args.command == "run-request-mock",
                use_openai=args.use_openai,
                llm_config=args.llm_config,
                model=args.model,
                max_repairs=args.max_repairs,
                model_table=args.model_table,
                model_bundle=args.model_bundle,
                model_registry=args.model_registry,
                material_systems_config=args.material_systems_config,
                route_target_policy=args.route_target_policy,
                skip_validation=args.skip_validation,
                validation_top_k=args.validation_top_k,
                output_selection=args.selection,
                run_mode=args.run_mode,
                normalize=not args.no_normalize,
                allow_non_100=args.no_normalize,
                temperature_celsius=args.temperature_celsius,
                pressure=args.pressure,
                force_rerun_xgems=args.force_rerun_xgems,
                gems_class_path=args.gems_class_path,
                xgems_input_mode=args.xgems_input_mode,
                xgems_water_mode=args.xgems_water_mode,
                xgems_water_factor=args.xgems_water_factor,
                xgems_water_g=args.xgems_water_g,
                xgems_water_w_b=args.xgems_water_w_b,
                retry_water_on_failure=args.retry_water_on_failure,
                retry_water_cap_w_b_ladder=args.retry_water_cap_w_b_ladder,
                retry_water_up_w_b_ladder=args.retry_water_up_w_b_ladder,
                retry_water_min_w_b=args.retry_water_min_w_b,
                reaction_model_id=args.reaction_model_id,
                reaction_model_signature=args.reaction_model_signature,
                reaction_model_config=args.reaction_model_config,
                disable_plots=args.no_plots,
                fail_fast=args.fail_fast,
            )
            _print_json(result.to_dict())
            return 0
        if args.command in {"forward-cached", "forward-cached-mock"}:
            result = run_forward_cached(
                recipe_text=args.recipe,
                db=args.db,
                dat_lst=getattr(args, "dat_lst", None),
                use_mock=args.command == "forward-cached-mock",
                force_rerun_xgems=args.force_rerun_xgems,
                run_mode=args.run_mode,
                normalize=not args.no_normalize,
                allow_non_100=args.no_normalize,
                temperature_celsius=args.temperature_celsius,
                pressure=args.pressure,
                gems_class_path=args.gems_class_path,
                xgems_input_mode=args.xgems_input_mode,
                xgems_water_mode=args.xgems_water_mode,
                xgems_water_factor=args.xgems_water_factor,
                xgems_water_g=args.xgems_water_g,
                xgems_water_w_b=args.xgems_water_w_b,
                retry_water_on_failure=args.retry_water_on_failure,
                retry_water_cap_w_b_ladder=args.retry_water_cap_w_b_ladder,
                retry_water_up_w_b_ladder=args.retry_water_up_w_b_ladder,
                retry_water_min_w_b=args.retry_water_min_w_b,
                reaction_model_id=args.reaction_model_id,
                reaction_model_config=args.reaction_model_config,
            )
            _print_json(result)
            return 0
        if args.command == "run-recipes-cached":
            results = []
            with args.recipes_csv.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    recipe_text = _recipe_text_from_csv_row(row)
                    temperature = float(row.get("temperature_celsius") or args.temperature_celsius)
                    results.append(
                        run_forward_cached(
                            recipe_text=recipe_text,
                            db=args.db,
                            dat_lst=args.dat_lst,
                            use_mock=False,
                            force_rerun_xgems=args.force_rerun_xgems,
                            run_mode=args.run_mode,
                            normalize=not args.no_normalize,
                            allow_non_100=args.no_normalize,
                            temperature_celsius=temperature,
                            pressure=args.pressure,
                            gems_class_path=args.gems_class_path,
                            xgems_input_mode=args.xgems_input_mode,
                            xgems_water_mode=args.xgems_water_mode,
                            xgems_water_factor=args.xgems_water_factor,
                            xgems_water_g=args.xgems_water_g,
                            xgems_water_w_b=args.xgems_water_w_b,
                            retry_water_on_failure=args.retry_water_on_failure,
                            retry_water_cap_w_b_ladder=args.retry_water_cap_w_b_ladder,
                            retry_water_up_w_b_ladder=args.retry_water_up_w_b_ladder,
                            retry_water_min_w_b=args.retry_water_min_w_b,
                            reaction_model_id=args.reaction_model_id,
                            reaction_model_config=args.reaction_model_config,
                        )
                    )
            _print_json({"runs": results})
            return 0
        if args.command == "inspect-chemistry":
            return _inspect_chemistry(args)
        if args.command == "inspect-recipe":
            return _inspect_recipe(args)
        if args.command == "compare-recipes":
            return _compare_recipes(args)
        if args.command == "inspect-available-outputs":
            _print_json(inspect_available_outputs(args.db, args.chem_hash))
            return 0
        if args.command == "build-feature-table":
            out = build_feature_table(
                db=args.db,
                selection=args.selection,
                out=args.out,
                output_format=args.format,
                recipe_id_prefix=args.recipe_id_prefix,
                material_system=args.material_system,
                age_days=args.age_days,
                age_tolerance=args.age_tolerance,
            )
            print(out)
            return 0
        if args.command == "feature-diagnostics":
            print(
                run_feature_diagnostics(
                    feature_table=args.feature_table,
                    out=args.out,
                    correlation_threshold=args.correlation_threshold,
                    sparse_threshold=args.sparse_threshold,
                    plot=not args.no_plots,
                )
            )
            return 0
        if args.command == "model-registry-diagnostics":
            print(
                run_model_registry_diagnostics(
                    model_registry=args.model_registry,
                    out=args.out,
                    reaction_model_id=args.reaction_model_id,
                    reaction_model_signature=args.reaction_model_signature,
                    min_r2=args.min_r2,
                    sparse_threshold=args.sparse_threshold,
                    min_range=args.min_range,
                    ph_min_range=args.ph_min_range,
                )
            )
            return 0
        if args.command == "recommend-active-learning-targets":
            print(
                write_active_learning_target_priorities(
                    diagnostics=args.diagnostics,
                    out=args.out,
                    statuses=args.status or None,
                    target_kinds=args.target_kind or None,
                    exclude_targets=args.exclude_target or None,
                    min_r2=args.min_r2,
                    max_targets=args.max_targets,
                )
            )
            return 0
        if args.command == "summarize-design-runs":
            print(run_design_run_report(runs=args.runs, out=args.out))
            return 0
        if args.command == "build-model-table":
            print(
                build_model_table(
                    feature_table=args.feature_table,
                    config=args.config,
                    out=args.out,
                    output_format=args.format,
                    reaction_model_id=args.reaction_model_id,
                    reaction_model_signature=args.reaction_model_signature,
                )
            )
            return 0
        if args.command == "build-chemistry-candidate-table":
            print(
                build_chemistry_candidate_table(
                    recipes_csv=args.recipes_csv,
                    out=args.out,
                    dat_lst=args.dat_lst,
                    run_mode=args.run_mode,
                    normalize=not args.no_normalize,
                    allow_non_100=args.no_normalize,
                    temperature_celsius=args.temperature_celsius,
                    pressure=args.pressure,
                    xgems_water_mode=args.xgems_water_mode,
                    xgems_water_factor=args.xgems_water_factor,
                    xgems_water_g=args.xgems_water_g,
                    xgems_water_w_b=args.xgems_water_w_b,
                    reaction_model_id=args.reaction_model_id,
                    reaction_model_config=args.reaction_model_config,
                    output_format=args.format,
                )
            )
            return 0
        if args.command == "chemistry-domain-report":
            print(
                write_chemistry_domain_report(
                    reference_model_table=args.reference_model_table,
                    candidate_table=args.candidate_table,
                    model_bundle=args.model_bundle,
                    out=args.out,
                    nearest_distance_warn=args.nearest_distance_warn,
                )
            )
            return 0
        if args.command == "run-chemistry-design-query":
            print(
                run_chemistry_design_query(
                    query=args.query,
                    out=args.out,
                    model_bundle=args.model_bundle,
                    reference_model_table=args.reference_model_table,
                    n_candidates=args.n_candidates,
                    seed=args.seed,
                    sampling_config=args.sampling_config,
                    material_systems_config=args.material_systems_config,
                    dat_lst=args.dat_lst,
                    skip_validation=not args.validate,
                    db=args.db,
                    validation_top_k=args.validation_top_k,
                    output_selection=args.selection,
                    run_mode=args.run_mode,
                    normalize=not args.no_normalize,
                    allow_non_100=args.no_normalize,
                    temperature_celsius=args.temperature_celsius,
                    pressure=args.pressure,
                    force_rerun_xgems=args.force_rerun_xgems,
                    gems_class_path=args.gems_class_path,
                    xgems_input_mode=args.xgems_input_mode,
                    xgems_water_mode=args.xgems_water_mode,
                    xgems_water_factor=args.xgems_water_factor,
                    xgems_water_g=args.xgems_water_g,
                    xgems_water_w_b=args.xgems_water_w_b,
                    retry_water_on_failure=args.retry_water_on_failure,
                    retry_water_cap_w_b_ladder=args.retry_water_cap_w_b_ladder,
                    retry_water_up_w_b_ladder=args.retry_water_up_w_b_ladder,
                    retry_water_min_w_b=args.retry_water_min_w_b,
                    reaction_model_id=args.reaction_model_id,
                    reaction_model_config=args.reaction_model_config,
                    strict_materials=args.strict_materials if args.strict_materials else None,
                    target_availability_policy=args.target_availability_policy,
                    fail_fast=args.fail_fast,
                )
            )
            return 0
        if args.command == "init-global-chem-db":
            _print_json(initialize_global_chemistry_db(db=args.db, schema_config=args.schema))
            return 0
        if args.command == "import-global-chem-artifacts":
            _print_json(
                copy_existing_artifacts_into_global_db(
                    db=args.db,
                    schema_config=args.schema,
                    feature_table=args.feature_table,
                    model_table=args.model_table,
                    model_bundle=args.model_bundle,
                )
            )
            return 0
        if args.command == "import-global-chem-db":
            _print_json(
                import_cached_db_into_global_db(
                    db=args.db,
                    source_db=args.source_db,
                    schema_config=args.schema,
                    copy_run_dirs=not args.no_copy_run_dirs,
                )
            )
            return 0
        if args.command == "refresh-global-chem-db":
            _print_json(
                refresh_global_chemistry_db(
                    db=args.db,
                    schema_config=args.schema,
                    selection=args.selection,
                    model_config=args.model_config,
                )
            )
            return 0
        if args.command == "train-global-chem-surrogate":
            _print_json(
                train_global_chemistry_surrogate(
                    db=args.db,
                    schema_config=args.schema,
                    selection=args.selection,
                    model_config=args.model_config,
                    surrogate_config=args.surrogate_config,
                    refresh=not args.no_refresh,
                    no_save_model=args.no_save_model,
                )
            )
            return 0
        if args.command == "global-chemistry-coverage":
            print(
                write_global_chemistry_coverage_report(
                    db=args.db,
                    out=args.out,
                    schema_config=args.schema,
                    coverage_config=args.coverage_config,
                    material_systems_config=args.material_systems_config,
                    target_metrics=args.target_metrics,
                )
            )
            return 0
        if args.command == "analyze-target-region":
            print(
                write_target_region_analysis(
                    model_table=args.model_table,
                    target=args.target,
                    out=args.out,
                    threshold=args.threshold,
                    top_n=args.top_n,
                    prefer=args.prefer,
                )
            )
            return 0
        if args.command == "analyze-xgems-quality-cases":
            print(
                write_xgems_quality_case_report(
                    db=args.db,
                    model_table=args.model_table,
                    out=args.out,
                    material_system=args.material_system,
                    only_problem_cases=not args.all_cases,
                    top_n=args.top_n,
                    water_tolerance_g=args.water_tolerance_g,
                )
            )
            return 0
        if args.command == "lookup-global-chem-db":
            print(
                lookup_global_chemistry(
                    db=args.db,
                    out=args.out,
                    schema_config=args.schema,
                    recipes_csv=args.recipes_csv,
                    candidate_table=args.candidate_table,
                    reference_model_table=args.reference_model_table,
                    model_bundle=args.model_bundle,
                    dat_lst=args.dat_lst,
                    run_mode=args.run_mode,
                    temperature_celsius=args.temperature_celsius,
                    pressure=args.pressure,
                    xgems_water_mode=args.xgems_water_mode,
                    xgems_water_factor=args.xgems_water_factor,
                    xgems_water_g=args.xgems_water_g,
                    xgems_water_w_b=args.xgems_water_w_b,
                    nearest_distance_warn=args.nearest_distance_warn,
                    reaction_model_id=args.reaction_model_id,
                    reaction_model_config=args.reaction_model_config,
                )
            )
            return 0
        if args.command == "acquire-global-chemistry":
            print(
                acquire_global_chemistry_candidates(
                    db=args.db,
                    out=args.out,
                    schema_config=args.schema,
                    recipes_csv=args.recipes_csv,
                    candidate_table=args.candidate_table,
                    reference_model_table=args.reference_model_table,
                    model_bundle=args.model_bundle,
                    max_candidates=args.max_candidates,
                    dat_lst=args.dat_lst,
                    run_mode=args.run_mode,
                    temperature_celsius=args.temperature_celsius,
                    pressure=args.pressure,
                    xgems_water_mode=args.xgems_water_mode,
                    xgems_water_factor=args.xgems_water_factor,
                    xgems_water_g=args.xgems_water_g,
                    xgems_water_w_b=args.xgems_water_w_b,
                    nearest_distance_warn=args.nearest_distance_warn,
                    reaction_model_id=args.reaction_model_id,
                    reaction_model_config=args.reaction_model_config,
                    priority_targets=args.priority_target,
                    priority_targets_from_diagnostics=args.priority_targets_from_diagnostics,
                    priority_target_statuses=args.priority_target_status,
                    priority_target_kinds=args.priority_target_kind,
                    priority_target_reliabilities=args.priority_target_reliability,
                    priority_target_limit=args.priority_target_limit,
                    target_priority_weight=args.target_priority_weight,
                    target_region_table=args.target_region_table or None,
                    target_region_weight=args.target_region_weight,
                    target_region_distance_scale=args.target_region_distance_scale,
                    target_region_max_reference_rows=args.target_region_max_reference_rows,
                )
            )
            return 0
        if args.command == "run-global-acquisition-cycle":
            print(
                run_global_chemistry_acquisition_cycle(
                    db=args.db,
                    out=args.out,
                    schema_config=args.schema,
                    recipes_csv=args.recipes_csv,
                    candidate_table=args.candidate_table,
                    reference_model_table=args.reference_model_table,
                    model_bundle=args.model_bundle,
                    max_candidates=args.max_candidates,
                    dat_lst=args.dat_lst,
                    use_mock=args.mock,
                    run_batch=not args.skip_batch,
                    refresh=not args.skip_refresh,
                    train_surrogate=not args.skip_train,
                    coverage=not args.skip_coverage,
                    selection=args.selection,
                    model_config=args.model_config,
                    surrogate_config=args.surrogate_config,
                    coverage_config=args.coverage_config,
                    material_systems_config=args.material_systems_config,
                    force_rerun_xgems=args.force_rerun_xgems,
                    resume_batch=not args.no_resume_batch,
                    fail_fast=args.fail_fast,
                    run_mode=args.run_mode,
                    temperature_celsius=args.temperature_celsius,
                    pressure=args.pressure,
                    gems_class_path=args.gems_class_path,
                    xgems_input_mode=args.xgems_input_mode,
                    xgems_water_mode=args.xgems_water_mode,
                    xgems_water_factor=args.xgems_water_factor,
                    xgems_water_g=args.xgems_water_g,
                    xgems_water_w_b=args.xgems_water_w_b,
                    retry_water_on_failure=args.retry_water_on_failure,
                    retry_water_cap_w_b_ladder=args.retry_water_cap_w_b_ladder,
                    retry_water_up_w_b_ladder=args.retry_water_up_w_b_ladder,
                    retry_water_min_w_b=args.retry_water_min_w_b,
                    reaction_model_id=args.reaction_model_id,
                    reaction_model_config=args.reaction_model_config,
                    nearest_distance_warn=args.nearest_distance_warn,
                    priority_targets=args.priority_target,
                    priority_targets_from_diagnostics=args.priority_targets_from_diagnostics,
                    priority_target_statuses=args.priority_target_status,
                    priority_target_kinds=args.priority_target_kind,
                    priority_target_reliabilities=args.priority_target_reliability,
                    priority_target_limit=args.priority_target_limit,
                    target_priority_weight=args.target_priority_weight,
                    target_region_table=args.target_region_table or None,
                    target_region_weight=args.target_region_weight,
                    target_region_distance_scale=args.target_region_distance_scale,
                    target_region_max_reference_rows=args.target_region_max_reference_rows,
                )
            )
            return 0
        if args.command in {"run-global-forward-query", "run-global-forward-query-mock"}:
            print(
                run_global_forward_query(
                    db=args.global_db,
                    query=args.query,
                    out=args.out,
                    dat_lst=args.dat_lst,
                    use_mock=args.command == "run-global-forward-query-mock",
                    run_mode=args.run_mode,
                    normalize=not args.no_normalize,
                    allow_non_100=args.no_normalize,
                    pressure=args.pressure,
                    force_rerun_xgems=args.force_rerun_xgems,
                    gems_class_path=args.gems_class_path,
                    xgems_input_mode=args.xgems_input_mode,
                    xgems_water_mode=args.xgems_water_mode,
                    xgems_water_factor=args.xgems_water_factor,
                    xgems_water_g=args.xgems_water_g,
                    xgems_water_w_b=args.xgems_water_w_b,
                    retry_water_on_failure=args.retry_water_on_failure,
                    retry_water_cap_w_b_ladder=args.retry_water_cap_w_b_ladder,
                    retry_water_up_w_b_ladder=args.retry_water_up_w_b_ladder,
                    retry_water_min_w_b=args.retry_water_min_w_b,
                    reaction_model_id=args.reaction_model_id,
                    reaction_model_config=args.reaction_model_config,
                    disable_plots=args.no_plots,
                    fail_fast=args.fail_fast,
                )
            )
            return 0
        if args.command == "run-global-design-query":
            print(
                run_global_design_query(
                    db=args.global_db,
                    query=args.query,
                    out=args.out,
                    model_bundle=args.model_bundle,
                    reference_model_table=args.reference_model_table,
                    n_candidates=args.n_candidates,
                    seed=args.seed,
                    sampling_config=args.sampling_config,
                    material_systems_config=args.material_systems_config,
                    dat_lst=args.dat_lst,
                    skip_validation=not args.validate,
                    use_mock_validation=args.mock_validation,
                    validation_top_k=args.validation_top_k,
                    output_selection=args.selection,
                    run_mode=args.run_mode,
                    normalize=not args.no_normalize,
                    allow_non_100=args.no_normalize,
                    temperature_celsius=args.temperature_celsius,
                    pressure=args.pressure,
                    force_rerun_xgems=args.force_rerun_xgems,
                    gems_class_path=args.gems_class_path,
                    xgems_input_mode=args.xgems_input_mode,
                    xgems_water_mode=args.xgems_water_mode,
                    xgems_water_factor=args.xgems_water_factor,
                    xgems_water_g=args.xgems_water_g,
                    xgems_water_w_b=args.xgems_water_w_b,
                    retry_water_on_failure=args.retry_water_on_failure,
                    retry_water_cap_w_b_ladder=args.retry_water_cap_w_b_ladder,
                    retry_water_up_w_b_ladder=args.retry_water_up_w_b_ladder,
                    retry_water_min_w_b=args.retry_water_min_w_b,
                    reaction_model_id=args.reaction_model_id,
                    reaction_model_config=args.reaction_model_config,
                    strict_materials=args.strict_materials if args.strict_materials else None,
                    target_availability_policy=args.target_availability_policy,
                    fail_fast=args.fail_fast,
                )
            )
            return 0
        if args.command == "train-baseline-surrogate":
            print(
                train_baseline_surrogate(
                    model_table=args.model_table,
                    config=args.config,
                    out=args.out,
                    save_model=not args.no_save_model,
                )
            )
            return 0
        if args.command == "surrogate-candidate-search":
            print(
                run_surrogate_candidate_search(
                    query=args.query,
                    out=args.out,
                    model_table=args.model_table,
                    model_bundle=args.model_bundle,
                    reaction_model_id=args.reaction_model_id,
                    reaction_model_signature=args.reaction_model_signature,
                    reaction_model_mismatch_policy=args.reaction_model_mismatch_policy,
                )
            )
            return 0
        if args.command in {"validate-candidates", "validate-candidates-mock"}:
            print(
                validate_candidates(
                    candidates=args.candidates,
                    dat_lst=getattr(args, "dat_lst", None),
                    db=args.db,
                    out=args.out,
                    top_k=args.top_k,
                    use_mock=args.command == "validate-candidates-mock",
                    selection=args.selection,
                    run_mode=args.run_mode,
                    normalize=not args.no_normalize,
                    allow_non_100=args.no_normalize,
                    temperature_celsius=args.temperature_celsius,
                    pressure=args.pressure,
                    force_rerun_xgems=args.force_rerun_xgems,
                    gems_class_path=args.gems_class_path,
                    xgems_input_mode=args.xgems_input_mode,
                    xgems_water_mode=args.xgems_water_mode,
                    xgems_water_factor=args.xgems_water_factor,
                    xgems_water_g=args.xgems_water_g,
                    xgems_water_w_b=args.xgems_water_w_b,
                    retry_water_on_failure=args.retry_water_on_failure,
                    retry_water_cap_w_b_ladder=args.retry_water_cap_w_b_ladder,
                    retry_water_up_w_b_ladder=args.retry_water_up_w_b_ladder,
                    retry_water_min_w_b=args.retry_water_min_w_b,
                    reaction_model_id=args.reaction_model_id,
                    reaction_model_config=args.reaction_model_config,
                    fail_fast=args.fail_fast,
                )
            )
            return 0
        if args.command == "select-candidates":
            print(
                select_candidates(
                    validation=args.validation,
                    config=args.config,
                    out=args.out,
                    feature_table=args.feature_table,
                )
            )
            return 0
        if args.command == "generate-recipes":
            rows = generate_recipe_rows(
                config_path=args.config,
                n=args.n,
                mode=args.mode,
                age_preset=args.age_preset,
                ages=args.ages,
                seed=args.seed,
                material_system=args.material_system,
                material_systems=_parse_str_list(args.material_systems),
                material_systems_sampling=args.material_systems_sampling,
                material_systems_path=args.material_systems_config,
                target_profile=args.target_profile,
                target_profiles_path=args.target_profiles_config,
                recipe_id_prefix=args.recipe_id_prefix,
                strict_materials=args.strict_materials,
                strict_allowed_materials=_parse_str_list(args.strict_allowed_materials),
                age_sampling=args.age_sampling,
                age_min=args.age_min,
                age_max=args.age_max,
                age_count=args.age_count,
            )
            write_recipe_csv(args.out, rows)
            print(args.out)
            return 0
        if args.command == "expand-ages":
            rows = expand_age_rows(read_recipe_csv(args.recipes), age_preset=args.age_preset, ages=args.ages)
            write_recipe_csv(args.out, rows)
            print(args.out)
            return 0
        if args.command in {"run-batch-cached", "run-batch-cached-mock"}:
            status = run_batch_cached(
                recipes_csv=args.recipes,
                db=args.db,
                dat_lst=getattr(args, "dat_lst", None),
                use_mock=args.command == "run-batch-cached-mock",
                workers=args.workers,
                resume=args.resume,
                fail_fast=args.fail_fast,
                force_rerun_xgems=args.force_rerun_xgems,
                run_mode=args.run_mode,
                gems_class_path=getattr(args, "gems_class_path", "xgems:ChemicalEngineDicts"),
                xgems_input_mode=getattr(args, "xgems_input_mode", "formula"),
                xgems_water_mode=args.xgems_water_mode,
                xgems_water_factor=args.xgems_water_factor,
                xgems_water_g=args.xgems_water_g,
                xgems_water_w_b=args.xgems_water_w_b,
                retry_water_on_failure=args.retry_water_on_failure,
                retry_water_cap_w_b_ladder=args.retry_water_cap_w_b_ladder,
                retry_water_up_w_b_ladder=args.retry_water_up_w_b_ladder,
                retry_water_min_w_b=args.retry_water_min_w_b,
                reaction_model_id=args.reaction_model_id,
                reaction_model_config=args.reaction_model_config,
                progress_every=args.progress_every,
            )
            print(status)
            return 0
        if args.command == "batch-status":
            summary = summarize_batch_status(db=args.db, out=args.out)
            _print_json(summary)
            return 0 if summary.get("has_status_file") else 1
        if args.command == "summarize-db":
            print(
                summarize_database(
                    db=args.db,
                    selection=args.selection,
                    out=args.out,
                    check_selected_outputs=args.check_selected_outputs,
                )
            )
            return 0
        if args.command == "backfill-reconstructed-volumes":
            _print_json(
                backfill_reconstructed_volumes_and_porosity(
                    db=args.db,
                    dat_lst=args.dat_lst,
                    limit=args.limit,
                    force=args.force,
                    include_unlinked_complete=args.include_unlinked_complete,
                    gems_class_path=args.gems_class_path,
                    xgems_input_mode=args.xgems_input_mode,
                    xgems_phase_volume_unit=args.xgems_phase_volume_unit,
                    create_backup=not args.no_backup,
                    reuse_runner=not args.no_reuse_runner,
                    progress_every=args.progress_every,
                )
            )
            return 0
        if args.command == "inverse-query":
            print(run_inverse_query(feature_table=args.feature_table, query=args.query, out=args.out))
            return 0
        if args.command == "compile-design-query":
            print(
                compile_design_query(
                    query=args.query,
                    out=args.out,
                    model_table=args.model_table,
                    model_bundle=args.model_bundle,
                    model_registry=args.model_registry,
                    reaction_model_id=args.reaction_model_id,
                    reaction_model_signature=args.reaction_model_signature,
                    reaction_model_config=args.reaction_model_config,
                    target_availability_policy=args.target_availability_policy,
                )
            )
            return 0
        if args.command == "route-design-query":
            print(
                route_design_query_file(
                    query=args.query,
                    out=args.out,
                    model_registry=args.model_registry,
                    material_systems_config=args.material_systems_config,
                    reaction_model_id=args.reaction_model_id,
                    reaction_model_signature=args.reaction_model_signature,
                    reaction_model_config=args.reaction_model_config,
                    target_policy=args.target_policy,
                    default_age_days=args.default_age_days,
                )
            )
            return 0
        if args.command == "validate-design-query":
            _print_json(
                validate_design_query_file(
                    args.query,
                    model_registry=args.model_registry,
                    reaction_model_id=args.reaction_model_id,
                    reaction_model_signature=args.reaction_model_signature,
                    reaction_model_config=args.reaction_model_config,
                    require_model_paths=args.require_model_paths,
                )
            )
            return 0
        if args.command == "design-query-schema":
            if args.out:
                print(save_design_query_schema(args.out))
            else:
                _print_json(design_query_json_schema())
            return 0
        if args.command == "forward-query-schema":
            if args.out:
                print(save_forward_query_schema(args.out))
            else:
                _print_json(forward_query_json_schema())
            return 0
        if args.command == "validate-forward-query":
            _print_json(validate_forward_query_file(args.query))
            return 0
        if args.command == "task-query-schema":
            if args.out:
                print(save_task_query_schema(args.out))
            else:
                _print_json(task_query_json_schema())
            return 0
        if args.command == "validate-task-query":
            _print_json(validate_task_query_file(args.query))
            return 0
        if args.command == "preview-task-query":
            print(
                preview_task_query_file(
                    query=args.query,
                    out=args.out,
                    model_registry=args.model_registry,
                    material_systems_config=args.material_systems_config,
                    route_target_policy=args.route_target_policy,
                    reaction_model_id=args.reaction_model_id,
                    reaction_model_signature=args.reaction_model_signature,
                    reaction_model_config=args.reaction_model_config,
                    default_age_days=args.default_age_days,
                )
            )
            return 0
        if args.command == "render-task-router-prompt":
            print(
                save_rendered_task_router_prompt(
                    args.out,
                    prompt=args.prompt,
                    schema=args.schema,
                    examples=args.examples,
                )
            )
            return 0
        if args.command == "validate-llm-task-output":
            base_prompt = None
            if args.base_prompt and Path(args.base_prompt).exists():
                base_prompt = Path(args.base_prompt).read_text(encoding="utf-8")
            report = validate_llm_task_query_file(
                args.input,
                original_user_request=args.user_request,
                base_prompt=base_prompt,
            )
            if args.out:
                write_llm_task_validation_report(report, args.out)
                print(args.out)
            else:
                _print_json(report)
            if args.fail_on_invalid and not report.get("valid"):
                return 1
            return 0
        if args.command == "parse-task-query-openai":
            print(
                parse_user_request_with_openai(
                    user_request=_request_text_from_args(args),
                    out=args.out,
                    config=args.llm_config,
                    model=args.model,
                    max_repairs=args.max_repairs,
                    model_registry=args.model_registry,
                    material_systems_config=args.material_systems_config,
                    route_target_policy=args.route_target_policy,
                    reaction_model_id=args.reaction_model_id,
                    reaction_model_signature=args.reaction_model_signature,
                    reaction_model_config=args.reaction_model_config,
                )
            )
            return 0
        if args.command == "run-design-query":
            print(
                run_design_query(
                    query=args.query,
                    out=args.out,
                    db=args.db,
                    dat_lst=args.dat_lst,
                    model_table=args.model_table,
                    model_bundle=args.model_bundle,
                    model_registry=args.model_registry,
                    skip_validation=args.skip_validation,
                    use_mock_validation=args.mock_validation,
                    validation_top_k=args.validation_top_k,
                    output_selection=args.selection,
                    run_mode=args.run_mode,
                    normalize=not args.no_normalize,
                    allow_non_100=args.no_normalize,
                    temperature_celsius=args.temperature_celsius,
                    pressure=args.pressure,
                    force_rerun_xgems=args.force_rerun_xgems,
                    gems_class_path=args.gems_class_path,
                    xgems_input_mode=args.xgems_input_mode,
                    xgems_water_mode=args.xgems_water_mode,
                    xgems_water_factor=args.xgems_water_factor,
                    xgems_water_g=args.xgems_water_g,
                    xgems_water_w_b=args.xgems_water_w_b,
                    retry_water_on_failure=args.retry_water_on_failure,
                    retry_water_cap_w_b_ladder=args.retry_water_cap_w_b_ladder,
                    retry_water_up_w_b_ladder=args.retry_water_up_w_b_ladder,
                    retry_water_min_w_b=args.retry_water_min_w_b,
                    reaction_model_id=args.reaction_model_id,
                    reaction_model_signature=args.reaction_model_signature,
                    reaction_model_config=args.reaction_model_config,
                    target_availability_policy=args.target_availability_policy,
                    fail_fast=args.fail_fast,
                )
            )
            return 0
        if args.command in {"run-forward-query", "run-forward-query-mock"}:
            print(
                run_forward_query(
                    query=args.query,
                    out=args.out,
                    db=args.db,
                    dat_lst=getattr(args, "dat_lst", None),
                    use_mock=args.command == "run-forward-query-mock",
                    run_mode=args.run_mode,
                    normalize=not args.no_normalize,
                    allow_non_100=args.no_normalize,
                    pressure=args.pressure,
                    force_rerun_xgems=args.force_rerun_xgems,
                    gems_class_path=args.gems_class_path,
                    xgems_input_mode=args.xgems_input_mode,
                    xgems_water_mode=args.xgems_water_mode,
                    xgems_water_factor=args.xgems_water_factor,
                    xgems_water_g=args.xgems_water_g,
                    xgems_water_w_b=args.xgems_water_w_b,
                    retry_water_on_failure=args.retry_water_on_failure,
                    retry_water_cap_w_b_ladder=args.retry_water_cap_w_b_ladder,
                    retry_water_up_w_b_ladder=args.retry_water_up_w_b_ladder,
                    retry_water_min_w_b=args.retry_water_min_w_b,
                    reaction_model_id=args.reaction_model_id,
                    reaction_model_config=args.reaction_model_config,
                    disable_plots=args.no_plots,
                    fail_fast=args.fail_fast,
                )
            )
            return 0
        if args.command in {"run-task-query", "run-task-query-mock"}:
            print(
                run_task_query(
                    query=args.query,
                    out=args.out,
                    db=args.db,
                    dat_lst=getattr(args, "dat_lst", None),
                    model_table=args.model_table,
                    model_bundle=args.model_bundle,
                    global_db=args.global_db,
                    global_n_candidates=args.global_n_candidates,
                    strict_materials=args.strict_materials if args.strict_materials else None,
                    model_registry=args.model_registry,
                    material_systems_config=args.material_systems_config,
                    route_target_policy=args.route_target_policy,
                    use_mock=args.command == "run-task-query-mock",
                    skip_validation=args.skip_validation,
                    validation_top_k=args.validation_top_k,
                    output_selection=args.selection,
                    run_mode=args.run_mode,
                    normalize=not args.no_normalize,
                    allow_non_100=args.no_normalize,
                    temperature_celsius=args.temperature_celsius,
                    pressure=args.pressure,
                    force_rerun_xgems=args.force_rerun_xgems,
                    gems_class_path=args.gems_class_path,
                    xgems_input_mode=args.xgems_input_mode,
                    xgems_water_mode=args.xgems_water_mode,
                    xgems_water_factor=args.xgems_water_factor,
                    xgems_water_g=args.xgems_water_g,
                    xgems_water_w_b=args.xgems_water_w_b,
                    retry_water_on_failure=args.retry_water_on_failure,
                    retry_water_cap_w_b_ladder=args.retry_water_cap_w_b_ladder,
                    retry_water_up_w_b_ladder=args.retry_water_up_w_b_ladder,
                    retry_water_min_w_b=args.retry_water_min_w_b,
                    reaction_model_id=args.reaction_model_id,
                    reaction_model_signature=args.reaction_model_signature,
                    reaction_model_config=args.reaction_model_config,
                    disable_plots=args.no_plots,
                    fail_fast=args.fail_fast,
                )
            )
            return 0
        if args.command in {"run-confirmed-task-query", "run-confirmed-task-query-mock"}:
            print(
                run_confirmed_task_query(
                    preview_dir=args.preview,
                    out=args.out,
                    db=args.db,
                    dat_lst=getattr(args, "dat_lst", None),
                    confirmed=args.confirm_preview,
                    allow_preview_errors=args.allow_preview_errors,
                    fail_on_preview_warnings=args.fail_on_preview_warnings,
                    model_table=args.model_table,
                    model_bundle=args.model_bundle,
                    global_db=args.global_db,
                    global_n_candidates=args.global_n_candidates,
                    strict_materials=args.strict_materials if args.strict_materials else None,
                    model_registry=args.model_registry,
                    material_systems_config=args.material_systems_config,
                    route_target_policy=args.route_target_policy,
                    use_mock=args.command == "run-confirmed-task-query-mock",
                    skip_validation=args.skip_validation,
                    validation_top_k=args.validation_top_k,
                    output_selection=args.selection,
                    run_mode=args.run_mode,
                    normalize=not args.no_normalize,
                    allow_non_100=args.no_normalize,
                    temperature_celsius=args.temperature_celsius,
                    pressure=args.pressure,
                    force_rerun_xgems=args.force_rerun_xgems,
                    gems_class_path=args.gems_class_path,
                    xgems_input_mode=args.xgems_input_mode,
                    xgems_water_mode=args.xgems_water_mode,
                    xgems_water_factor=args.xgems_water_factor,
                    xgems_water_g=args.xgems_water_g,
                    xgems_water_w_b=args.xgems_water_w_b,
                    retry_water_on_failure=args.retry_water_on_failure,
                    retry_water_cap_w_b_ladder=args.retry_water_cap_w_b_ladder,
                    retry_water_up_w_b_ladder=args.retry_water_up_w_b_ladder,
                    retry_water_min_w_b=args.retry_water_min_w_b,
                    reaction_model_id=args.reaction_model_id,
                    reaction_model_signature=args.reaction_model_signature,
                    reaction_model_config=args.reaction_model_config,
                    disable_plots=args.no_plots,
                    fail_fast=args.fail_fast,
                )
            )
            return 0
        if args.command in {"run-openai-task-query", "run-openai-task-query-mock"}:
            print(
                run_user_request_with_openai(
                    user_request=_request_text_from_args(args),
                    out=args.out,
                    db=args.db,
                    dat_lst=getattr(args, "dat_lst", None),
                    config=args.llm_config,
                    model=args.model,
                    max_repairs=args.max_repairs,
                    model_table=args.model_table,
                    model_bundle=args.model_bundle,
                    global_db=args.global_db,
                    global_n_candidates=args.global_n_candidates,
                    strict_materials=args.strict_materials if args.strict_materials else None,
                    model_registry=args.model_registry,
                    material_systems_config=args.material_systems_config,
                    route_target_policy=args.route_target_policy,
                    use_mock=args.command == "run-openai-task-query-mock",
                    skip_validation=args.skip_validation,
                    validation_top_k=args.validation_top_k,
                    output_selection=args.selection,
                    run_mode=args.run_mode,
                    normalize=not args.no_normalize,
                    allow_non_100=args.no_normalize,
                    temperature_celsius=args.temperature_celsius,
                    pressure=args.pressure,
                    force_rerun_xgems=args.force_rerun_xgems,
                    gems_class_path=args.gems_class_path,
                    xgems_input_mode=args.xgems_input_mode,
                    xgems_water_mode=args.xgems_water_mode,
                    xgems_water_factor=args.xgems_water_factor,
                    xgems_water_g=args.xgems_water_g,
                    xgems_water_w_b=args.xgems_water_w_b,
                    retry_water_on_failure=args.retry_water_on_failure,
                    retry_water_cap_w_b_ladder=args.retry_water_cap_w_b_ladder,
                    retry_water_up_w_b_ladder=args.retry_water_up_w_b_ladder,
                    retry_water_min_w_b=args.retry_water_min_w_b,
                    reaction_model_id=args.reaction_model_id,
                    reaction_model_signature=args.reaction_model_signature,
                    reaction_model_config=args.reaction_model_config,
                    disable_plots=args.no_plots,
                    fail_fast=args.fail_fast,
                )
            )
            return 0
        if args.command in {"run-inverse-forward-workflow", "run-inverse-forward-workflow-mock"}:
            print(
                run_inverse_forward_workflow(
                    inverse_request=_inverse_request_text_from_args(args),
                    out=args.out,
                    db=args.db,
                    dat_lst=getattr(args, "dat_lst", None),
                    global_db=args.global_db,
                    global_n_candidates=args.global_n_candidates,
                    strict_materials=args.strict_materials if args.strict_materials else None,
                    llm_config=args.llm_config,
                    model=args.model,
                    max_repairs=args.max_repairs,
                    model_registry=args.model_registry,
                    material_systems_config=args.material_systems_config,
                    route_target_policy=args.route_target_policy,
                    validation_top_k=args.validation_top_k,
                    output_selection=args.selection,
                    run_mode=args.run_mode,
                    normalize=not args.no_normalize,
                    allow_non_100=args.no_normalize,
                    temperature_celsius=args.temperature_celsius,
                    pressure=args.pressure,
                    force_rerun_xgems=args.force_rerun_xgems,
                    gems_class_path=args.gems_class_path,
                    xgems_input_mode=args.xgems_input_mode,
                    xgems_water_mode=args.xgems_water_mode,
                    xgems_water_factor=args.xgems_water_factor,
                    xgems_water_g=args.xgems_water_g,
                    xgems_water_w_b=args.xgems_water_w_b,
                    retry_water_on_failure=args.retry_water_on_failure,
                    retry_water_cap_w_b_ladder=args.retry_water_cap_w_b_ladder,
                    retry_water_up_w_b_ladder=args.retry_water_up_w_b_ladder,
                    retry_water_min_w_b=args.retry_water_min_w_b,
                    reaction_model_id=args.reaction_model_id,
                    reaction_model_signature=args.reaction_model_signature,
                    reaction_model_config=args.reaction_model_config,
                    disable_plots=args.no_plots,
                    fail_fast=args.fail_fast,
                    use_mock=args.command == "run-inverse-forward-workflow-mock",
                    candidate_rank=args.candidate_rank,
                    forward_age_values=args.forward_age_values,
                    forward_age_start=args.forward_age_start,
                    forward_age_stop=args.forward_age_stop,
                    forward_age_points=args.forward_age_points,
                    forward_age_spacing=args.forward_age_spacing,
                    forward_phases=args.forward_phase or None,
                    forward_phase_groups=args.forward_phase_group or None,
                    forward_scalars=args.forward_scalar or None,
                    forward_table_limit=args.forward_table_limit,
                    forward_top_phases=args.forward_top_phases,
                    forward_narrative_language=args.forward_narrative_language,
                )
            )
            return 0
        run_dir = run_forward_recipe(
            recipe_text=args.recipe,
            out=args.out,
            dat_lst=getattr(args, "dat_lst", None),
            use_mock=args.command == "forward-mock",
            run_mode=args.run_mode,
            normalize=not args.no_normalize,
            allow_non_100=args.no_normalize,
            temperature_celsius=args.temperature_celsius,
            command_name=args.command,
            gems_class_path=args.gems_class_path,
            xgems_input_mode=args.xgems_input_mode,
            xgems_water_mode=args.xgems_water_mode,
            xgems_water_factor=args.xgems_water_factor,
            xgems_water_g=args.xgems_water_g,
            xgems_water_w_b=args.xgems_water_w_b,
            reaction_model_id=args.reaction_model_id,
            reaction_model_config=args.reaction_model_config,
        )
    except Exception as exc:
        print(f"inverse-gems error: {exc}", file=sys.stderr)
        return 1
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

