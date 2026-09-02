from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .forward_query import run_forward_query
from .openai_task_router import parse_user_request_with_openai, run_user_request_with_openai
from .task_query import run_task_query
from .task_query_preview import run_confirmed_task_query
from .utils import write_json


RequestStatus = Literal["complete", "failed"]


@dataclass
class RequestResult:
    status: RequestStatus
    task_type: str | None
    run_dir: Path
    routed_out: Path | None = None
    answer_text: str | None = None
    files: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    missing_outputs: dict[str, list[str]] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "task_type": self.task_type,
            "run_dir": str(self.run_dir),
            "routed_out": str(self.routed_out) if self.routed_out else None,
            "answer_text": self.answer_text,
            "files": self.files,
            "warnings": self.warnings,
            "missing_outputs": self.missing_outputs,
            "summary": self.summary,
            "error": self.error,
        }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_text(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _string_path(value: Any) -> Path | None:
    if not value:
        return None
    return Path(str(value))


def _risk_counts(preview: dict[str, Any]) -> dict[str, int]:
    counts = {"error": 0, "warning": 0, "info": 0}
    for item in preview.get("risks") or []:
        level = str(item.get("level") or "info")
        counts[level] = counts.get(level, 0) + 1
    return counts


def _selected_model_from_preview(preview: dict[str, Any]) -> dict[str, Any] | None:
    inverse = preview.get("inverse_design") or {}
    routing = inverse.get("model_routing") or {}
    selected = routing.get("selected")
    return selected if isinstance(selected, dict) else None


def _preview_result_from_dir(*, run_dir: Path, preview_dir: Path) -> RequestResult:
    preview = _load_json(preview_dir / "parsed_query_preview.json")
    report = _load_json(preview_dir / "llm_task_router_report.json")
    selected_model = _selected_model_from_preview(preview)
    files: dict[str, str] = {"run_dir": str(run_dir), "preview_dir": str(preview_dir)}
    for name in [
        "task_query.yaml",
        "parsed_query_preview.json",
        "parsed_query_preview.md",
        "llm_task_router_report.json",
        "llm_raw_output_attempt_0.txt",
        "rendered_task_router_prompt.md",
        "user_request.txt",
    ]:
        path = preview_dir / name
        if path.exists():
            files[name.replace(".", "_")] = str(path)
    warnings = [
        f"{item.get('code')}: {item.get('message')}"
        for item in preview.get("risks") or []
        if str(item.get("level") or "") in {"warning", "error"}
    ]
    return RequestResult(
        status="complete" if preview.get("valid") else "failed",
        task_type=str(preview.get("task_type")) if preview.get("task_type") else None,
        run_dir=run_dir,
        routed_out=preview_dir,
        answer_text=_read_text(preview_dir / "parsed_query_preview.md"),
        files=files,
        warnings=warnings,
        summary={
            "mode": "parse_preview",
            "preview_status": preview.get("status"),
            "risk_counts": _risk_counts(preview),
            "selected_model": selected_model,
            "openai_model": report.get("model"),
            "parse_source": report.get("parse_source"),
            "usage": (report.get("attempts") or [{}])[-1].get("usage") if report.get("attempts") else None,
        },
    )


def _forward_result_from_dir(
    *,
    run_dir: Path,
    forward_dir: Path,
    task_type: str,
    summary: dict[str, Any],
) -> RequestResult:
    forward_summary = _load_json(forward_dir / "forward_query_summary.json")
    response_summary = _load_json(forward_dir / "response_summary.json")
    answer_json = _load_json(forward_dir / "answer.json")
    narrative_json = _load_json(forward_dir / "narrative_answer.json")

    files: dict[str, str] = {
        "run_dir": str(run_dir),
        "forward_dir": str(forward_dir),
    }
    for name in [
        "time_series.csv",
        "forward_query_summary.json",
        "response_summary.json",
        "response_summary.csv",
        "answer.json",
        "answer.md",
        "narrative_answer.json",
        "narrative_answer.md",
    ]:
        path = forward_dir / name
        if path.exists():
            files[name.replace(".", "_")] = str(path)

    answer_text = _read_text(forward_dir / "narrative_answer.md") or _read_text(forward_dir / "answer.md")
    failed_count = int(forward_summary.get("failed_count") or response_summary.get("failed_count") or 0)
    completed_count = int(forward_summary.get("completed_count") or response_summary.get("completed_count") or 0)
    row_count = int(forward_summary.get("age_count") or response_summary.get("row_count") or completed_count)
    warnings = [str(item) for item in forward_summary.get("warnings") or []]
    missing = answer_json.get("missing") or response_summary.get("missing") or {"phases": [], "scalars": []}

    return RequestResult(
        status="failed" if failed_count and completed_count == 0 else "complete",
        task_type=task_type,
        run_dir=run_dir,
        routed_out=forward_dir,
        answer_text=answer_text,
        files=files,
        warnings=warnings,
        missing_outputs={
            "phases": [str(item) for item in missing.get("phases") or []],
            "scalars": [str(item) for item in missing.get("scalars") or []],
        },
        summary={
            **summary,
            "row_count": row_count,
            "completed_count": completed_count,
            "failed_count": failed_count,
            "response_summary": forward_summary.get("response_summary"),
            "narrative": {
                "mode": narrative_json.get("mode"),
                "language": narrative_json.get("language"),
            }
            if narrative_json
            else None,
        },
    )


def _generic_result_from_task_dir(*, run_dir: Path, task_dir: Path) -> RequestResult:
    task_summary = _load_json(task_dir / "task_query_summary.json")
    task_type = task_summary.get("task_type")
    routed_out = _string_path(task_summary.get("routed_out"))
    if task_type in {"forward_calculation", "forward_time_series"} and routed_out:
        return _forward_result_from_dir(
            run_dir=run_dir,
            forward_dir=routed_out,
            task_type=str(task_type),
            summary={"task_query_summary": task_summary},
        )

    files: dict[str, str] = {"run_dir": str(run_dir), "task_dir": str(task_dir)}
    if routed_out:
        files["routed_out"] = str(routed_out)
    for name in ["task_query_summary.json", "task_query_manifest.json"]:
        path = task_dir / name
        if path.exists():
            files[path.stem] = str(path)
    design_summary = {}
    if routed_out and routed_out.exists():
        design_summary = _load_json(routed_out / "design_query_run_summary.json")
        for name in [
            "final_candidates.csv",
            "final_candidates.json",
            "design_query_run_summary.json",
            "candidate_review.csv",
            "candidate_review.json",
            "candidate_review.md",
            "candidate_review_summary.json",
            "selected_candidates.csv",
            "selected_candidates.json",
        ]:
            path = routed_out / name
            if path.exists():
                files[name.replace(".", "_")] = str(path)
                if name == "final_candidates.csv":
                    files.setdefault("final_candidates", str(path))
    return RequestResult(
        status="complete",
        task_type=str(task_type) if task_type else None,
        run_dir=run_dir,
        routed_out=routed_out,
        files=files,
        summary={
            "task_query_summary": task_summary,
            "design_query_run_summary": design_summary or None,
        },
    )


def _confirmed_result_from_dir(*, run_dir: Path, task_dir: Path) -> RequestResult:
    result = _generic_result_from_task_dir(run_dir=run_dir, task_dir=task_dir)
    manifest = _load_json(task_dir / "confirmed_run_manifest.json")
    preview = _load_json(task_dir / "confirmed_preview" / "parsed_query_preview.json")
    selected_model = _selected_model_from_preview(preview)
    result.files["confirmed_run_manifest"] = str(task_dir / "confirmed_run_manifest.json")
    for name in ["task_query.yaml", "parsed_query_preview.json", "parsed_query_preview.md"]:
        path = task_dir / "confirmed_preview" / name
        if path.exists():
            result.files[f"confirmed_{name.replace('.', '_')}"] = str(path)
    result.summary["confirmed_run_manifest"] = manifest
    result.summary["confirmed_preview"] = {
        "status": preview.get("status"),
        "risk_counts": _risk_counts(preview),
        "selected_model": selected_model,
    }
    return result


def _write_result(run_dir: Path, result: RequestResult) -> RequestResult:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "request_result.json", result.to_dict())
    return result


def run_forward_request(
    *,
    forward_query: str | Path,
    out: str | Path,
    db: str | Path,
    dat_lst: str | Path | None = None,
    use_mock: bool = False,
    run_mode: str = "reacted_only",
    normalize: bool = True,
    allow_non_100: bool = False,
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
    retry_water_policy: str = "ladder",
    retry_water_max_retries: int = 6,
    max_xgems_calls: int | None = None,
    reaction_model_id: str | None = None,
    reaction_model_config: str | Path | None = None,
    materials_config: str | Path | None = None,
    disable_plots: bool = False,
    fail_fast: bool = False,
) -> RequestResult:
    run_dir = Path(out)
    forward_dir = run_dir / "forward"
    run_forward_query(
        query=forward_query,
        out=forward_dir,
        db=db,
        dat_lst=dat_lst,
        use_mock=use_mock,
        run_mode=run_mode,
        normalize=normalize,
        allow_non_100=allow_non_100,
        pressure=pressure,
        force_rerun_xgems=force_rerun_xgems,
        gems_class_path=gems_class_path or "xgems:ChemicalEngineDicts",
        xgems_input_mode=xgems_input_mode,
        xgems_water_mode=xgems_water_mode,
        xgems_water_factor=xgems_water_factor,
        xgems_water_g=xgems_water_g,
        xgems_water_w_b=xgems_water_w_b,
        retry_water_on_failure=retry_water_on_failure,
        retry_water_cap_w_b_ladder=retry_water_cap_w_b_ladder,
        retry_water_up_w_b_ladder=retry_water_up_w_b_ladder,
        retry_water_min_w_b=retry_water_min_w_b,
        retry_water_policy=retry_water_policy,
        retry_water_max_retries=retry_water_max_retries,
        max_xgems_calls=max_xgems_calls,
        reaction_model_id=reaction_model_id,
        reaction_model_config=reaction_model_config,
        materials_config=materials_config,
        disable_plots=disable_plots,
        fail_fast=fail_fast,
    )
    result = _forward_result_from_dir(
        run_dir=run_dir,
        forward_dir=forward_dir,
        task_type="forward_calculation_or_time_series",
        summary={"forward_query": str(forward_query)},
    )
    return _write_result(run_dir, result)


def parse_request_preview(
    *,
    request: str,
    out: str | Path,
    llm_config: str | Path | None = None,
    model: str | None = None,
    max_repairs: int | None = None,
    client: Any | None = None,
    model_registry: str | Path | None = None,
    material_systems_config: str | Path | None = None,
    route_target_policy: str = "recommended",
    reaction_model_id: str | None = None,
    reaction_model_signature: str | None = None,
    reaction_model_config: str | Path | None = None,
) -> RequestResult:
    run_dir = Path(out)
    preview_dir = run_dir / "openai_parse"
    parse_user_request_with_openai(
        user_request=request,
        out=preview_dir,
        config=llm_config,
        model=model,
        max_repairs=max_repairs,
        model_registry=model_registry,
        material_systems_config=material_systems_config,
        route_target_policy=route_target_policy,
        reaction_model_id=reaction_model_id,
        reaction_model_signature=reaction_model_signature,
        reaction_model_config=reaction_model_config,
        client=client,
    )
    return _write_result(run_dir, _preview_result_from_dir(run_dir=run_dir, preview_dir=preview_dir))


def run_confirmed_request(
    *,
    confirmed_preview: str | Path,
    out: str | Path,
    db: str | Path,
    confirm_preview: bool = False,
    allow_preview_errors: bool = False,
    fail_on_preview_warnings: bool = False,
    dat_lst: str | Path | None = None,
    model_table: str | Path | None = None,
    model_bundle: str | Path | None = None,
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
    materials_config: str | Path | None = None,
    disable_plots: bool = False,
    fail_fast: bool = False,
) -> RequestResult:
    run_dir = Path(out)
    run_confirmed_task_query(
        preview_dir=confirmed_preview,
        out=run_dir,
        db=db,
        confirmed=confirm_preview,
        allow_preview_errors=allow_preview_errors,
        fail_on_preview_warnings=fail_on_preview_warnings,
        dat_lst=dat_lst,
        model_table=model_table,
        model_bundle=model_bundle,
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
        materials_config=materials_config,
        disable_plots=disable_plots,
        fail_fast=fail_fast,
    )
    return _write_result(run_dir, _confirmed_result_from_dir(run_dir=run_dir, task_dir=run_dir))


def run_request(
    *,
    out: str | Path,
    db: str | Path,
    request: str | None = None,
    task_query: str | Path | None = None,
    forward_query: str | Path | None = None,
    confirmed_preview: str | Path | None = None,
    confirm_preview: bool = False,
    allow_preview_errors: bool = False,
    fail_on_preview_warnings: bool = False,
    dat_lst: str | Path | None = None,
    use_mock: bool = False,
    use_openai: bool = False,
    llm_config: str | Path | None = None,
    model: str | None = None,
    max_repairs: int | None = None,
    client: Any | None = None,
    model_table: str | Path | None = None,
    model_bundle: str | Path | None = None,
    model_registry: str | Path | None = None,
    material_systems_config: str | Path | None = None,
    route_target_policy: str = "recommended",
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
    materials_config: str | Path | None = None,
    disable_plots: bool = False,
    fail_fast: bool = False,
) -> RequestResult:
    run_dir = Path(out)
    if sum(value is not None for value in [task_query, forward_query, request, confirmed_preview]) != 1:
        raise ValueError("Provide exactly one of request, task_query, forward_query, or confirmed_preview.")
    if request is not None and not use_openai:
        raise ValueError("Free-text request execution requires use_openai=True. Use task_query or forward_query for deterministic execution.")
    if confirmed_preview is not None:
        return run_confirmed_request(
            confirmed_preview=confirmed_preview,
            out=run_dir,
            db=db,
            confirm_preview=confirm_preview,
            allow_preview_errors=allow_preview_errors,
            fail_on_preview_warnings=fail_on_preview_warnings,
            dat_lst=dat_lst,
            model_table=model_table,
            model_bundle=model_bundle,
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
            materials_config=materials_config,
            disable_plots=disable_plots,
            fail_fast=fail_fast,
        )
    if forward_query is not None:
        return run_forward_request(
            forward_query=forward_query,
            out=run_dir,
            db=db,
            dat_lst=dat_lst,
            use_mock=use_mock,
            run_mode=run_mode,
            normalize=normalize,
            allow_non_100=allow_non_100,
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
            disable_plots=disable_plots,
            fail_fast=fail_fast,
        )
    if task_query is not None:
        task_dir = run_dir / "task_run"
        run_task_query(
            query=task_query,
            out=task_dir,
            db=db,
            dat_lst=dat_lst,
            model_table=model_table,
            model_bundle=model_bundle,
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
            materials_config=materials_config,
            disable_plots=disable_plots,
            fail_fast=fail_fast,
        )
        return _write_result(run_dir, _generic_result_from_task_dir(run_dir=run_dir, task_dir=task_dir))

    assert request is not None
    run_user_request_with_openai(
        user_request=request,
        out=run_dir,
        db=db,
        dat_lst=dat_lst,
        config=llm_config,
        model=model,
        max_repairs=max_repairs,
        client=client,
        model_table=model_table,
        model_bundle=model_bundle,
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
        materials_config=materials_config,
        disable_plots=disable_plots,
        fail_fast=fail_fast,
    )
    task_dir = run_dir / "task_run"
    result = _generic_result_from_task_dir(run_dir=run_dir, task_dir=task_dir)
    result.summary["openai_task_run_summary"] = _load_json(run_dir / "openai_task_run_summary.json")
    return _write_result(run_dir, result)
