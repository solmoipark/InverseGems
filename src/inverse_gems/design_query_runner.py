from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .candidate_search import run_surrogate_candidate_search
from .candidate_review import write_candidate_review
from .candidate_selection import select_candidates
from .candidate_validation import validate_candidates
from .design_query import compile_design_query, resolve_reaction_model_request
from .inverse_design_flow import write_inverse_design_flow_summary
from .utils import load_yaml, write_json


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


def _validation_top_k(compiled_selection: Path, explicit_top_k: int | None) -> int:
    if explicit_top_k is not None:
        return int(explicit_top_k)
    selection = load_yaml(compiled_selection)
    return int(selection.get("top_k") or 10)


def run_design_query(
    *,
    query: str | Path,
    out: str | Path,
    db: str | Path | None = None,
    dat_lst: str | Path | None = None,
    model_table: str | Path | None = None,
    model_bundle: str | Path | None = None,
    model_registry: str | Path | None = None,
    skip_validation: bool = False,
    use_mock_validation: bool = False,
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
    reaction_model_signature: str | None = None,
    reaction_model_config: str | Path | None = None,
    materials_config: str | Path | None = None,
    target_availability_policy: str = "warn",
    fail_fast: bool = False,
) -> Path:
    """Run a compiled design-query pipeline end-to-end.

    The runner keeps each stage in its own subdirectory:
    compiled_query, surrogate_search, validation, and selection.
    """
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    compiled_dir = out_dir / "compiled_query"
    search_dir = out_dir / "surrogate_search"
    validation_dir = out_dir / "validation"
    selection_dir = out_dir / "selection"

    reaction_request = resolve_reaction_model_request(
        load_yaml(query),
        reaction_model_id=reaction_model_id,
        reaction_model_signature=reaction_model_signature,
        reaction_model_config=reaction_model_config,
    )
    expected_reaction_model_id = reaction_request.get("id")
    expected_reaction_model_signature = reaction_request.get("signature")
    validation_reaction_model_id = expected_reaction_model_id or reaction_model_id
    validation_reaction_model_config = reaction_request.get("config") or reaction_model_config

    compiled = compile_design_query(
        query=query,
        out=compiled_dir,
        model_table=model_table,
        model_bundle=model_bundle,
        model_registry=model_registry,
        reaction_model_id=expected_reaction_model_id,
        reaction_model_signature=expected_reaction_model_signature,
        reaction_model_config=validation_reaction_model_config,
        target_availability_policy=target_availability_policy,
    )
    search_query = compiled / "surrogate_candidate_search.yaml"
    selection_query = compiled / "candidate_selection.yaml"

    search = run_surrogate_candidate_search(
        query=search_query,
        out=search_dir,
        reaction_model_id=expected_reaction_model_id,
        reaction_model_signature=expected_reaction_model_signature,
    )
    final_csv: str | None = None
    final_json: str | None = None
    stage = "surrogate_search"

    if skip_validation:
        final_csv = _copy_if_exists(search / "candidates.csv", out_dir / "final_candidates.csv")
        final_json = _copy_if_exists(search / "candidates.json", out_dir / "final_candidates.json")
    else:
        if db is None:
            raise ValueError("--db is required unless --skip-validation is used.")
        top_k = _validation_top_k(selection_query, validation_top_k)
        validation = validate_candidates(
            candidates=search / "candidates.csv",
            out=validation_dir,
            db=db,
            dat_lst=dat_lst,
            top_k=top_k,
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
            reaction_model_id=validation_reaction_model_id,
            reaction_model_config=validation_reaction_model_config,
            materials_config=materials_config,
            fail_fast=fail_fast,
        )
        selection = select_candidates(
            validation=validation / "validation_comparison.csv",
            config=selection_query,
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
            title=f"Candidate Review ({stage})",
        )

    summary = {
        "query": str(query),
        "out": str(out_dir),
        "skip_validation": skip_validation,
        "use_mock_validation": use_mock_validation,
        "final_stage": stage,
        "final_csv": final_csv,
        "final_json": final_json,
        "candidate_review": candidate_review,
        "paths": {
            "compiled_query": str(compiled_dir),
            "surrogate_search": str(search_dir),
            "validation": None if skip_validation else str(validation_dir),
            "selection": None if skip_validation else str(selection_dir),
        },
        "compiled_manifest": _read_json_if_exists(compiled_dir / "design_query_manifest.json"),
        "candidate_search_summary": _read_json_if_exists(search_dir / "candidate_search_summary.json"),
        "validation_summary": {} if skip_validation else _read_json_if_exists(validation_dir / "validation_summary.json"),
        "selection_summary": {} if skip_validation else _read_json_if_exists(selection_dir / "selection_summary.json"),
    }
    summary["inverse_design_flow_summary"] = write_inverse_design_flow_summary(run_summary=summary, out=out_dir)
    write_json(out_dir / "design_query_run_summary.json", summary)
    return out_dir
