from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .cached_forward import run_forward_cached
from .database import InverseGemsDatabase, read_name_value_csv
from .env_check import check_environment
from .forward_query import run_forward_query
from .utils import timestamp_compact, write_json


DEFAULT_SINGLE_CASES: list[dict[str, Any]] = [
    {
        "case_id": "single_opc_28",
        "case_type": "single_recipe",
        "recipe": "OPC 100, w/b 0.4, age 28",
        "material_system": "OPC",
    },
    {
        "case_id": "single_opc_fly_ash_28",
        "case_type": "single_recipe",
        "recipe": "OPC 30, fly ash 70, w/b 0.4, age 28",
        "material_system": "OPC_fly_ash",
    },
    {
        "case_id": "single_opc_slag_1",
        "case_type": "single_recipe",
        "recipe": "OPC 50, slag 50, w/b 0.45, age 1",
        "material_system": "OPC_slag",
    },
    {
        "case_id": "single_opc_slag_28",
        "case_type": "single_recipe",
        "recipe": "OPC 50, slag 50, w/b 0.45, age 28",
        "material_system": "OPC_slag",
    },
    {
        "case_id": "single_opc_slag_limestone_gypsum_28",
        "case_type": "single_recipe",
        "recipe": "OPC 50, slag 40, limestone 8, gypsum 2, w/b 0.45, age 28",
        "material_system": "OPC_slag_limestone",
    },
    {
        "case_id": "single_opc_fly_ash_limestone_gypsum_28",
        "case_type": "single_recipe",
        "recipe": "OPC 55, fly ash 35, limestone 8, gypsum 2, w/b 0.45, age 28",
        "material_system": "OPC_fly_ash_limestone",
    },
    {
        "case_id": "single_lc3_like_28",
        "case_type": "single_recipe",
        "recipe": "OPC 50, metakaolin 30, limestone 15, gypsum 5, w/b 0.45, age 28",
        "material_system": "LC3_like",
    },
    {
        "case_id": "single_opc_silica_fume_28",
        "case_type": "single_recipe",
        "recipe": "OPC 90, silica fume 10, w/b 0.4, age 28",
        "material_system": "OPC_silica_fume",
    },
]


DEFAULT_FORWARD_QUERY_CASES: list[dict[str, Any]] = [
    {
        "case_id": "forward_query_single_age_opc_slag_28",
        "case_type": "forward_query",
        "query": {
            "name": "acceptance_single_age_opc_slag_28",
            "task": "forward_calculation",
            "material_system": "OPC_slag",
            "recipe": {"binders": {"OPC": 60, "slag": 40}, "w_b": 0.45},
            "age_grid": {"values": [28.0]},
            "temperature_celsius": 20.0,
            "outputs": {
                "phase_masses": "all",
                "phase_volumes": "all",
                "phase_volumes_reconstructed": "all",
                "aqueous_species": "all",
                "scalars": "all",
            },
            "plots": [],
            "response_summary": {
                "phases": ["CNASH", "Portlandite"],
                "scalars": ["pH", "porosity"],
                "top_phases": 8,
                "table_limit": 10,
                "narrative_enabled": True,
                "narrative_language": "ko",
            },
        },
    },
    {
        "case_id": "forward_query_time_series_opc_slag_fly_ash",
        "case_type": "forward_query",
        "query": {
            "name": "acceptance_time_series_opc_slag_fly_ash",
            "task": "forward_time_series",
            "material_system": "OPC_slag_fly_ash",
            "recipe": {"binders": {"OPC": 40, "slag": 30, "fly_ash": 30}, "w_b": 0.4},
            "age_grid": {"values": [0.1, 1.0, 28.0, 90.0]},
            "temperature_celsius": 20.0,
            "outputs": {
                "phase_masses": "all",
                "phase_volumes": "all",
                "phase_volumes_reconstructed": "all",
                "aqueous_species": "all",
                "scalars": ["pH", "system_volume", "system_mass"],
            },
            "plots": [
                {
                    "kind": "phase_volumes",
                    "names": "all_nonzero",
                    "top_n": 8,
                    "filename": "phase_volumes_vs_age.png",
                    "x_scale": "log",
                    "y_scale": "linear",
                }
            ],
            "response_summary": {
                "phases": ["CNASH", "Portlandite"],
                "scalars": ["pH", "porosity"],
                "top_phases": 8,
                "table_limit": 10,
                "narrative_enabled": True,
                "narrative_language": "ko",
            },
        },
    },
]


def default_acceptance_cases() -> list[dict[str, Any]]:
    return [*DEFAULT_SINGLE_CASES, *DEFAULT_FORWARD_QUERY_CASES]


def _json_cell(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _top_values(values: dict[str, Any], top_n: int) -> dict[str, float]:
    parsed: list[tuple[str, float]] = []
    for name, value in values.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if abs(number) > 0.0:
            parsed.append((str(name), number))
    parsed.sort(key=lambda item: abs(item[1]), reverse=True)
    return {name: value for name, value in parsed[:top_n]}


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": f"{type(exc).__name__}: {exc}"}


def _status_is_passed(status: Any) -> bool:
    text = str(status).lower()
    return bool(text) and "fail" not in text and "error" not in text and "bad" not in text


def _raw_from_cached_result(db: InverseGemsDatabase, result: dict[str, Any]) -> dict[str, Any]:
    chemistry_row = db.get_chemistry_run(str(result.get("chem_hash"))) or {}
    recipe_row = db.get_recipe_run(str(result.get("recipe_id"))) or {}
    raw_dir = Path(str(chemistry_row.get("xgems_run_dir") or result.get("chemistry_dir") or ""))
    phase_masses = read_name_value_csv(raw_dir / "xgems_phase_amounts_raw.csv")
    phase_volumes = read_name_value_csv(raw_dir / "xgems_phase_volumes_raw.csv")
    phase_volumes_reconstructed = read_name_value_csv(raw_dir / "xgems_phase_volumes_reconstructed.csv")
    aqueous_species = read_name_value_csv(raw_dir / "xgems_aqueous_species_raw.csv")
    scalars = _load_json_if_exists(raw_dir / "xgems_scalars_raw.json")
    return {
        "chemistry_row": chemistry_row,
        "recipe_row": recipe_row,
        "raw_dir": str(raw_dir) if str(raw_dir) else "",
        "phase_masses": phase_masses,
        "phase_volumes": phase_volumes,
        "phase_volumes_reconstructed": phase_volumes_reconstructed,
        "aqueous_species": aqueous_species,
        "scalars": scalars,
    }


def _summary_row_from_cached(
    *,
    case: dict[str, Any],
    result: dict[str, Any],
    raw: dict[str, Any],
    top_n: int,
) -> dict[str, Any]:
    recipe_row = raw.get("recipe_row") or {}
    scalars = raw.get("scalars") or {}
    status = "passed" if result.get("chemistry_status") == "complete" and _status_is_passed(result.get("solver_status")) else "failed"
    return {
        "case_id": case["case_id"],
        "case_type": case["case_type"],
        "status": status,
        "chemistry_status": result.get("chemistry_status"),
        "solver_status": result.get("solver_status"),
        "recipe": case.get("recipe"),
        "reported_age_days": recipe_row.get("age_days"),
        "age_count": 1,
        "completed_count": 1 if status == "passed" else 0,
        "failed_count": 0 if status == "passed" else 1,
        "recipe_id": result.get("recipe_id"),
        "chem_hash": result.get("chem_hash"),
        "porosity": result.get("porosity"),
        "pH": scalars.get("pH", scalars.get("ph")),
        "xgems_water_g": result.get("xgems_water_g"),
        "xgems_w_b": result.get("xgems_w_b"),
        "xgems_water_mode": result.get("xgems_water_mode"),
        "solver_rescued": result.get("solver_rescued"),
        "xgems_retry_count": result.get("xgems_retry_count"),
        "primary_chem_hash": result.get("primary_chem_hash"),
        "raw_dir": raw.get("raw_dir"),
        "output_dir": result.get("recipe_dir"),
        "phase_masses_nonzero_count": len(_top_values(raw.get("phase_masses") or {}, 100000)),
        "phase_volumes_nonzero_count": len(_top_values(raw.get("phase_volumes") or {}, 100000)),
        "phase_masses_top_json": _json_cell(_top_values(raw.get("phase_masses") or {}, top_n)),
        "phase_volumes_top_json": _json_cell(_top_values(raw.get("phase_volumes") or {}, top_n)),
        "phase_volumes_reconstructed_top_json": _json_cell(
            _top_values(raw.get("phase_volumes_reconstructed") or {}, top_n)
        ),
        "error_message": "",
    }


def _prefixed_values(row: pd.Series, prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    prefix_text = f"{prefix}__"
    for column, value in row.items():
        if not str(column).startswith(prefix_text):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        name = str(column)[len(prefix_text) :]
        out[name] = number
    return out


def _summary_row_from_forward_query(
    *,
    case: dict[str, Any],
    out_dir: Path,
    top_n: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = _load_json_if_exists(out_dir / "forward_query_summary.json")
    frame = pd.read_csv(out_dir / "time_series.csv") if (out_dir / "time_series.csv").exists() else pd.DataFrame()
    final_row: pd.Series | None = None
    if not frame.empty:
        complete = frame[frame["chemistry_status"].astype(str) == "complete"] if "chemistry_status" in frame else frame
        final_row = complete.sort_values("age_days").iloc[-1] if not complete.empty and "age_days" in complete else frame.iloc[-1]
    completed = int(summary.get("completed_count") or 0)
    failed = int(summary.get("failed_count") or 0)
    age_count = int(summary.get("age_count") or len(frame))
    status = "passed" if age_count > 0 and completed == age_count and failed == 0 else "failed"
    phase_masses = _prefixed_values(final_row, "phase_mass") if final_row is not None else {}
    phase_volumes = _prefixed_values(final_row, "phase_volume") if final_row is not None else {}
    phase_volumes_reconstructed = (
        _prefixed_values(final_row, "phase_volume_reconstructed") if final_row is not None else {}
    )
    scalars = _prefixed_values(final_row, "scalar") if final_row is not None else {}
    detail = {
        "summary": summary,
        "final_completed_row": final_row.to_dict() if final_row is not None else {},
        "phase_masses": phase_masses,
        "phase_volumes": phase_volumes,
        "phase_volumes_reconstructed": phase_volumes_reconstructed,
        "scalars": scalars,
    }
    row = {
        "case_id": case["case_id"],
        "case_type": case["case_type"],
        "status": status,
        "chemistry_status": "complete" if status == "passed" else "failed",
        "solver_status": final_row.get("solver_status") if final_row is not None else "",
        "recipe": _json_cell((case.get("query") or {}).get("recipe", {})),
        "reported_age_days": final_row.get("age_days") if final_row is not None else None,
        "age_count": age_count,
        "completed_count": completed,
        "failed_count": failed,
        "recipe_id": final_row.get("recipe_id") if final_row is not None else "",
        "chem_hash": final_row.get("chem_hash") if final_row is not None else "",
        "porosity": final_row.get("porosity") if final_row is not None else None,
        "pH": scalars.get("pH", scalars.get("ph")),
        "xgems_water_g": final_row.get("xgems_water_g") if final_row is not None else None,
        "xgems_w_b": final_row.get("xgems_w_b") if final_row is not None else None,
        "xgems_water_mode": final_row.get("xgems_water_mode") if final_row is not None else "",
        "solver_rescued": final_row.get("solver_rescued") if final_row is not None else None,
        "xgems_retry_count": final_row.get("xgems_retry_count") if final_row is not None else None,
        "primary_chem_hash": "",
        "raw_dir": final_row.get("xgems_run_dir") if final_row is not None else "",
        "output_dir": str(out_dir),
        "phase_masses_nonzero_count": len(_top_values(phase_masses, 100000)),
        "phase_volumes_nonzero_count": len(_top_values(phase_volumes, 100000)),
        "phase_masses_top_json": _json_cell(_top_values(phase_masses, top_n)),
        "phase_volumes_top_json": _json_cell(_top_values(phase_volumes, top_n)),
        "phase_volumes_reconstructed_top_json": _json_cell(_top_values(phase_volumes_reconstructed, top_n)),
        "error_message": "; ".join(summary.get("warnings") or []),
    }
    return row, detail


def _write_markdown_report(path: Path, payload: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    summary = payload["summary"]
    lines = [
        "# inverse_gems Acceptance Report",
        "",
        f"- Status: `{summary['overall_status']}`",
        f"- Passed cases: {summary['passed_count']} / {summary['case_count']}",
        f"- Mode: `{summary['mode']}`",
        f"- dat.lst: `{summary.get('dat_lst')}`",
        "",
        "| case_id | type | status | chemistry_status | solver_status | age_count | completed | failed | pH | porosity |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("case_id", "")),
                    str(row.get("case_type", "")),
                    str(row.get("status", "")),
                    str(row.get("chemistry_status", "")),
                    str(row.get("solver_status", "")).replace("|", "/"),
                    str(row.get("age_count", "")),
                    str(row.get("completed_count", "")),
                    str(row.get("failed_count", "")),
                    "" if row.get("pH") is None else f"{float(row.get('pH')):.6g}",
                    "" if row.get("porosity") is None else f"{float(row.get('porosity')):.6g}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Raw phase names are not cleaned, aliased, merged, or interpreted. "
            "`phase_masses_top_json` and `phase_volumes_top_json` in `acceptance_report.csv` preserve names exactly.",
            "",
            "Files:",
            f"- JSON: `{path.with_suffix('.json')}`",
            f"- CSV: `{path.with_suffix('.csv')}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_query_yaml(path: Path, query: dict[str, Any], disable_plots: bool) -> Path:
    query_data = dict(query)
    if disable_plots:
        query_data["plots"] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(query_data, sort_keys=False), encoding="utf-8")
    return path


def _selected_cases(case_ids: list[str] | None) -> list[dict[str, Any]]:
    cases = default_acceptance_cases()
    if not case_ids:
        return cases
    requested = set(case_ids)
    selected = [case for case in cases if str(case["case_id"]) in requested]
    missing = sorted(requested - {str(case["case_id"]) for case in selected})
    if missing:
        raise ValueError(f"Unknown acceptance case id(s): {missing}")
    return selected


def run_acceptance_suite(
    *,
    out: str | Path,
    dat_lst: str | Path | None = None,
    use_mock: bool = False,
    db: str | Path | None = None,
    case_ids: list[str] | None = None,
    run_mode: str = "reacted_only",
    normalize: bool = True,
    allow_non_100: bool = False,
    temperature_celsius: float = 20.0,
    pressure: float | None = None,
    force_rerun_xgems: bool = False,
    gems_class_path: str = "xgems:ChemicalEngineDicts",
    xgems_input_mode: str = "formula",
    xgems_water_mode: str = "initial",
    xgems_water_factor: float = 1.0,
    xgems_water_g: float | None = None,
    xgems_water_w_b: float | None = None,
    retry_water_on_failure: bool = True,
    retry_water_cap_w_b_ladder: list[float] | tuple[float, ...] | str = (0.45, 0.40, 0.35, 0.30),
    retry_water_up_w_b_ladder: list[float] | tuple[float, ...] | str = (0.30, 0.35, 0.40, 0.45),
    retry_water_min_w_b: float = 0.30,
    disable_plots: bool = True,
    top_n_phases: int = 8,
    fail_fast: bool = False,
) -> dict[str, Any]:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(db) if db is not None else out_dir / "db"
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    cases = _selected_cases(case_ids)
    write_json(out_dir / "acceptance_cases.json", cases)

    env_report = check_environment(
        dat_lst=dat_lst,
        gems_class_path=gems_class_path,
        xgems_input_mode=xgems_input_mode,
        require_xgems=not use_mock,
        instantiate_runner=not use_mock,
        out=out_dir / "environment_report.json",
    )

    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    if not use_mock and not env_report.get("ok"):
        rows.append(
            {
                "case_id": "environment_preflight",
                "case_type": "environment",
                "status": "failed",
                "chemistry_status": "not_run",
                "solver_status": "not_run",
                "recipe": "",
                "reported_age_days": None,
                "age_count": 0,
                "completed_count": 0,
                "failed_count": 1,
                "recipe_id": "",
                "chem_hash": "",
                "porosity": None,
                "pH": None,
                "xgems_water_g": None,
                "xgems_w_b": None,
                "xgems_water_mode": "",
                "solver_rescued": None,
                "xgems_retry_count": None,
                "primary_chem_hash": "",
                "raw_dir": "",
                "output_dir": str(out_dir),
                "phase_masses_nonzero_count": 0,
                "phase_volumes_nonzero_count": 0,
                "phase_masses_top_json": "{}",
                "phase_volumes_top_json": "{}",
                "phase_volumes_reconstructed_top_json": "{}",
                "error_message": "Environment preflight failed. See environment_report.json.",
            }
        )
    else:
        database = InverseGemsDatabase(db_path)
        for case in cases:
            case_dir = cases_dir / str(case["case_id"])
            case_dir.mkdir(parents=True, exist_ok=True)
            try:
                if case["case_type"] == "single_recipe":
                    result = run_forward_cached(
                        recipe_text=str(case["recipe"]),
                        db=db_path,
                        dat_lst=dat_lst,
                        use_mock=use_mock,
                        force_rerun_xgems=force_rerun_xgems,
                        run_mode=run_mode,
                        normalize=normalize,
                        allow_non_100=allow_non_100,
                        temperature_celsius=temperature_celsius,
                        pressure=pressure,
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
                        recipe_metadata={
                            "template_name": "acceptance",
                            "material_system": case.get("material_system", ""),
                            "acceptance_case_id": case["case_id"],
                        },
                    )
                    raw = _raw_from_cached_result(database, result)
                    row = _summary_row_from_cached(case=case, result=result, raw=raw, top_n=top_n_phases)
                    detail = {"case": case, "result": result, **raw}
                elif case["case_type"] == "forward_query":
                    query_path = _write_query_yaml(case_dir / "forward_query.yaml", dict(case["query"]), disable_plots)
                    query_out = case_dir / "run"
                    run_forward_query(
                        query=query_path,
                        out=query_out,
                        db=db_path,
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
                        disable_plots=disable_plots,
                        fail_fast=fail_fast,
                    )
                    row, detail = _summary_row_from_forward_query(case=case, out_dir=query_out, top_n=top_n_phases)
                    detail["case"] = case
                else:
                    raise ValueError(f"Unsupported acceptance case type: {case['case_type']}")
            except Exception as exc:
                row = {
                    "case_id": case.get("case_id", ""),
                    "case_type": case.get("case_type", ""),
                    "status": "failed",
                    "chemistry_status": "failed",
                    "solver_status": "",
                    "recipe": case.get("recipe", _json_cell((case.get("query") or {}).get("recipe", {}))),
                    "reported_age_days": None,
                    "age_count": 1,
                    "completed_count": 0,
                    "failed_count": 1,
                    "recipe_id": "",
                    "chem_hash": "",
                    "porosity": None,
                    "pH": None,
                    "xgems_water_g": None,
                    "xgems_w_b": None,
                    "xgems_water_mode": "",
                    "solver_rescued": None,
                    "xgems_retry_count": None,
                    "primary_chem_hash": "",
                    "raw_dir": "",
                    "output_dir": str(case_dir),
                    "phase_masses_nonzero_count": 0,
                    "phase_volumes_nonzero_count": 0,
                    "phase_masses_top_json": "{}",
                    "phase_volumes_top_json": "{}",
                    "phase_volumes_reconstructed_top_json": "{}",
                    "error_message": f"{type(exc).__name__}: {exc}",
                }
                detail = {"case": case, "error": row["error_message"]}
                if fail_fast:
                    rows.append(row)
                    details.append(detail)
                    break
            rows.append(row)
            details.append(detail)
            write_json(case_dir / "case_detail.json", detail)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.to_csv(out_dir / "acceptance_report.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "acceptance_report.csv", index=False)
    passed_count = sum(1 for row in rows if row.get("status") == "passed")
    failed_count = sum(1 for row in rows if row.get("status") != "passed")
    summary = {
        "run_id": f"acceptance_{timestamp_compact()}",
        "mode": "mock" if use_mock else "real",
        "overall_status": "passed" if rows and failed_count == 0 else "failed",
        "ok": bool(rows and failed_count == 0),
        "case_count": len(rows),
        "passed_count": passed_count,
        "failed_count": failed_count,
        "out": str(out_dir),
        "db": str(db_path),
        "dat_lst": str(dat_lst) if dat_lst else None,
        "gems_class_path": gems_class_path,
        "xgems_input_mode": xgems_input_mode,
        "retry_water_on_failure": retry_water_on_failure,
        "disable_plots": disable_plots,
    }
    payload = {
        "summary": summary,
        "environment": env_report,
        "rows": rows,
        "details": details,
    }
    write_json(out_dir / "acceptance_report.json", payload)
    _write_markdown_report(out_dir / "acceptance_report.md", payload, rows)
    return payload
