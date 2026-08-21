from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .llm_task_router import (
    build_repair_prompt,
    render_task_router_prompt,
    validate_llm_task_query_output,
)
from .task_query import run_task_query
from .task_query_preview import preview_task_query_file
from .utils import config_path, load_project_env, load_yaml, write_json


def load_openai_task_router_config(path: str | Path | None = None) -> dict[str, Any]:
    load_project_env()
    config = load_yaml(path or config_path("openai_task_router.yaml"))
    return {
        "model": os.environ.get("INVERSE_GEMS_OPENAI_MODEL") or config.get("model", "gpt-4.1-mini"),
        "temperature": float(config.get("temperature", 0.0)),
        "max_output_tokens": int(config.get("max_output_tokens", 4096)),
        "max_repairs": int(config.get("max_repairs", 1)),
        "prompt": config.get("prompt") or config_path("llm_task_router.prompt.md"),
        "schema": config.get("schema") or config_path("task_query.schema.json"),
        "examples": config.get("examples") or config_path("task_query.examples.yaml"),
    }


def _make_openai_client() -> Any:
    load_project_env()
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError(
            "The openai Python package is required for this command. "
            "Install inverse-gems with the llm extra or install openai in this environment."
        ) from exc
    return OpenAI()


def _extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)
    if isinstance(response, dict):
        output_text = response.get("output_text")
        if output_text:
            return str(output_text)
        chunks: list[str] = []
        for item in response.get("output", []) or []:
            for content in item.get("content", []) or []:
                text = content.get("text") if isinstance(content, dict) else None
                if text:
                    chunks.append(str(text))
        if chunks:
            return "\n".join(chunks)
    output = getattr(response, "output", None)
    chunks = []
    for item in output or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(str(text))
    if chunks:
        return "\n".join(chunks)
    return str(response)


def _response_usage(response: Any) -> Any:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json")
    return usage


def _call_responses_api(
    *,
    client: Any,
    model: str,
    system_prompt: str,
    user_content: str,
    temperature: float,
    max_output_tokens: int,
) -> tuple[str, Any]:
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    return _extract_response_text(response), _response_usage(response)


def _write_task_query_yaml(path: Path, task_query: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(task_query, handle, sort_keys=False)


def parse_user_request_with_openai(
    *,
    user_request: str,
    out: str | Path,
    config: str | Path | None = None,
    model: str | None = None,
    max_repairs: int | None = None,
    model_registry: str | Path | None = None,
    material_systems_config: str | Path | None = None,
    route_target_policy: str = "recommended",
    reaction_model_id: str | None = None,
    reaction_model_signature: str | None = None,
    reaction_model_config: str | Path | None = None,
    client: Any | None = None,
) -> Path:
    router_config = load_openai_task_router_config(config)
    model_name = model or str(router_config["model"])
    repairs_allowed = int(router_config["max_repairs"] if max_repairs is None else max_repairs)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = render_task_router_prompt(
        prompt=router_config["prompt"],
        schema=router_config["schema"],
        examples=router_config["examples"],
    )
    (out_dir / "user_request.txt").write_text(user_request, encoding="utf-8")
    (out_dir / "rendered_task_router_prompt.md").write_text(system_prompt, encoding="utf-8")

    openai_client = client or _make_openai_client()
    attempts: list[dict[str, Any]] = []
    user_content = user_request
    final_report: dict[str, Any] | None = None
    for attempt_index in range(repairs_allowed + 1):
        raw_text, usage = _call_responses_api(
            client=openai_client,
            model=model_name,
            system_prompt=system_prompt,
            user_content=user_content,
            temperature=float(router_config["temperature"]),
            max_output_tokens=int(router_config["max_output_tokens"]),
        )
        raw_path = out_dir / f"llm_raw_output_attempt_{attempt_index}.txt"
        raw_path.write_text(raw_text, encoding="utf-8")
        validation = validate_llm_task_query_output(
            raw_text,
            original_user_request=user_request,
            base_prompt=system_prompt,
        )
        attempts.append(
            {
                "attempt_index": attempt_index,
                "valid": bool(validation.get("valid")),
                "raw_output": str(raw_path),
                "error": validation.get("error"),
                "usage": usage,
            }
        )
        if validation.get("valid"):
            final_report = validation
            break
        if attempt_index >= repairs_allowed:
            final_report = validation
            break
        repair_prompt = build_repair_prompt(
            original_user_request=user_request,
            previous_output=raw_text,
            error_message=str(validation.get("error")),
            base_prompt=system_prompt,
        )
        repair_path = out_dir / f"llm_repair_prompt_attempt_{attempt_index + 1}.md"
        repair_path.write_text(repair_prompt, encoding="utf-8")
        user_content = repair_prompt

    assert final_report is not None
    final_report["attempts"] = attempts
    final_report["model"] = model_name
    final_report["out"] = str(out_dir)
    write_json(out_dir / "llm_task_router_report.json", final_report)
    if final_report.get("valid"):
        _write_task_query_yaml(out_dir / "task_query.yaml", final_report["task_query"])
        preview_task_query_file(
            query=out_dir / "task_query.yaml",
            out=out_dir,
            model_registry=model_registry,
            material_systems_config=material_systems_config,
            route_target_policy=route_target_policy,
            reaction_model_id=reaction_model_id,
            reaction_model_signature=reaction_model_signature,
            reaction_model_config=reaction_model_config,
        )
    else:
        raise RuntimeError(
            "OpenAI task-router output did not validate. "
            f"See {out_dir / 'llm_task_router_report.json'} for the repair prompt and errors."
        )
    return out_dir


def run_user_request_with_openai(
    *,
    user_request: str,
    out: str | Path,
    db: str | Path,
    dat_lst: str | Path | None = None,
    config: str | Path | None = None,
    model: str | None = None,
    max_repairs: int | None = None,
    client: Any | None = None,
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
    out_dir = Path(out)
    parse_dir = out_dir / "openai_parse"
    parse_user_request_with_openai(
        user_request=user_request,
        out=parse_dir,
        config=config,
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
    run_dir = run_task_query(
        query=parse_dir / "task_query.yaml",
        out=out_dir / "task_run",
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
        out_dir / "openai_task_run_summary.json",
        {
            "user_request": user_request,
            "parse_dir": str(parse_dir),
            "task_query": str(parse_dir / "task_query.yaml"),
            "preview_json": str(parse_dir / "parsed_query_preview.json"),
            "preview_markdown": str(parse_dir / "parsed_query_preview.md"),
            "task_run": str(run_dir),
            "use_mock": use_mock,
            "global_db": str(global_db) if global_db is not None else None,
            "skip_validation": skip_validation,
            "material_systems_config": str(material_systems_config) if material_systems_config else None,
            "route_target_policy": route_target_policy,
        },
    )
    return out_dir
