from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from .design_query import DesignQuerySpec
from .design_query_runner import run_design_query
from .forward_query import ForwardQuerySpec, run_forward_query
from .model_router import route_design_query
from .utils import load_yaml, write_json


TaskType = Literal["forward_calculation", "forward_time_series", "inverse_design"]


class TaskQuerySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    user_request: str | None = None
    task_type: TaskType
    forward_query: ForwardQuerySpec | None = None
    design_query: DesignQuerySpec | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> "TaskQuerySpec":
        if self.task_type in {"forward_calculation", "forward_time_series"}:
            if self.forward_query is None:
                raise ValueError("forward_query is required for forward task_type.")
            if self.design_query is not None:
                raise ValueError("design_query must not be set for forward task_type.")
        if self.task_type == "inverse_design":
            if self.design_query is None:
                raise ValueError("design_query is required for inverse_design task_type.")
            if self.forward_query is not None:
                raise ValueError("forward_query must not be set for inverse_design task_type.")
        return self


def validate_task_query_data(data: dict[str, Any]) -> dict[str, Any]:
    return TaskQuerySpec.model_validate(data).model_dump(mode="json", exclude_none=True)


def validate_task_query_file(query: str | Path) -> dict[str, Any]:
    return validate_task_query_data(load_yaml(query))


def task_query_json_schema() -> dict[str, Any]:
    return TaskQuerySpec.model_json_schema()


def save_task_query_schema(out: str | Path) -> Path:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, task_query_json_schema())
    return out_path


def _write_nested_query(out_dir: Path, name: str, payload: dict[str, Any]) -> Path:
    path = out_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _needs_model_routing(design_query: dict[str, Any]) -> bool:
    if design_query.get("model_table") and design_query.get("model_bundle"):
        return False
    if design_query.get("model_id"):
        return False
    design_space = dict(design_query.get("design_space") or {})
    material_system = design_query.get("material_system") or design_space.get("material_systems")
    systems = [str(value).strip().lower() for value in _as_list(material_system) if str(value).strip()]
    if not systems:
        return True
    if len(systems) != 1:
        return True
    if systems[0] in {"auto", "any", "all"}:
        return True
    if design_query.get("age_days") is None and design_space.get("age_days") is None:
        return True
    return False


def run_task_query(
    *,
    query: str | Path,
    out: str | Path,
    db: str | Path,
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
    materials_config: str | Path | None = None,
    disable_plots: bool = False,
    fail_fast: bool = False,
) -> Path:
    query_path = Path(query)
    raw = load_yaml(query_path)
    spec = TaskQuerySpec.model_validate(raw)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(query_path, out_dir / "task_query_used.yaml")

    routed_path: Path
    routed_out: Path
    if spec.task_type in {"forward_calculation", "forward_time_series"}:
        assert spec.forward_query is not None
        routed_path = _write_nested_query(
            out_dir / "routed_query",
            "forward_query.yaml",
            spec.forward_query.model_dump(mode="json", exclude_none=True),
        )
        routed_out = out_dir / "forward"
        result_dir = run_forward_query(
            query=routed_path,
            out=routed_out,
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
            reaction_model_id=reaction_model_id,
            reaction_model_config=reaction_model_config,
            materials_config=materials_config,
            disable_plots=disable_plots,
            fail_fast=fail_fast,
        )
        forward_summary = _load_json(result_dir / "forward_query_summary.json")
    else:
        assert spec.design_query is not None
        design_payload = spec.design_query.model_dump(mode="json", exclude_none=True)
        model_route_report: dict[str, Any] | None = None
        if _needs_model_routing(design_payload):
            model_route_report = route_design_query(
                design_payload,
                model_registry=model_registry,
                material_systems_config=material_systems_config,
                reaction_model_id=reaction_model_id,
                reaction_model_signature=reaction_model_signature,
                reaction_model_config=reaction_model_config,
                target_policy=route_target_policy,  # type: ignore[arg-type]
            )
            design_payload = model_route_report["routed_query"]
        routed_path = _write_nested_query(
            out_dir / "routed_query",
            "design_query.yaml",
            design_payload,
        )
        if model_route_report is not None:
            write_json(out_dir / "routed_query" / "model_route_report.json", model_route_report)
        routed_out = out_dir / "design"
        if global_db is not None:
            from .global_chemistry_db import run_global_design_query

            result_dir = run_global_design_query(
                db=global_db,
                query=routed_path,
                out=routed_out,
                n_candidates=global_n_candidates,
                material_systems_config=material_systems_config,
                dat_lst=dat_lst,
                skip_validation=skip_validation,
                use_mock_validation=use_mock,
                validation_top_k=validation_top_k,
                output_selection=output_selection,
                run_mode=run_mode,
                normalize=normalize,
                allow_non_100=allow_non_100,
                temperature_celsius=temperature_celsius if temperature_celsius is not None else 20.0,
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
                strict_materials=strict_materials,
                fail_fast=fail_fast,
            )
        else:
            result_dir = run_design_query(
                query=routed_path,
                out=routed_out,
                db=db,
                dat_lst=dat_lst,
                model_table=model_table,
                model_bundle=model_bundle,
                model_registry=model_registry,
                skip_validation=skip_validation,
                use_mock_validation=use_mock,
                validation_top_k=validation_top_k,
                output_selection=output_selection,
                run_mode=run_mode,
                normalize=normalize,
                allow_non_100=allow_non_100,
                temperature_celsius=temperature_celsius if temperature_celsius is not None else 20.0,
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
                fail_fast=fail_fast,
            )
        forward_summary = {}

    write_json(
        out_dir / "task_query_summary.json",
        {
            "query": str(query_path),
            "out": str(out_dir),
            "task_type": spec.task_type,
            "name": spec.name,
            "user_request": spec.user_request,
            "routed_query": str(routed_path),
            "routed_out": str(result_dir),
            "use_mock": use_mock,
            "global_db": str(global_db) if global_db is not None else None,
            "skip_validation": skip_validation if spec.task_type == "inverse_design" else None,
            "response_summary": forward_summary.get("response_summary") if forward_summary else None,
            "model_route_report": None if spec.task_type != "inverse_design" else str(out_dir / "routed_query" / "model_route_report.json")
            if (out_dir / "routed_query" / "model_route_report.json").exists()
            else None,
        },
    )
    write_json(out_dir / "task_query_manifest.json", spec.model_dump(mode="json", exclude_none=True))
    return out_dir
