from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .forward_query import run_forward_query
from .openai_task_router import run_user_request_with_openai
from .uncertainty import (
    as_float,
    flag_counts,
    is_missing,
    pH_water_reliable as is_pH_water_reliable,
    uncertainty_flags,
    xgems_water_delta,
    xgems_water_matches_recipe,
)
from .utils import write_json


BINDER_COLUMNS = [
    "OPC",
    "slag",
    "fly_ash",
    "metakaolin",
    "silica_fume",
    "limestone",
    "gypsum",
]

DEFAULT_FORWARD_PHASES = [
    "CNASH",
    "Portlandite",
    "C4AH19",
    "C3(AF)S0.84H",
    "straetlingite",
    "OH-hydrotalcite",
    "OH_SO4_AFm",
    "SO4_OH_AFm",
    "ettringite",
]

DEFAULT_FORWARD_PHASE_GROUPS = [
    "C-A-S-H",
    "ettringite",
    "monosulfate",
    "hemicarbonate",
    "monocarbonate",
    "siliceous hydrogarnet",
    "straetlingite",
    "aluminosilicate gel",
    "Calcite",
    "Portlandite",
]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _value_or_none(value: Any) -> Any:
    return None if is_missing(value) else value


def _flags_text(row: dict[str, Any] | pd.Series) -> str:
    raw = row.get("uncertainty_flags")
    if isinstance(raw, str) and raw.strip().lower() not in {"", "nan", "none"}:
        return raw
    return ";".join(uncertainty_flags(row))


def _pH_flag(row: dict[str, Any] | pd.Series) -> str:
    reliable = is_pH_water_reliable(row)
    if reliable is True:
        return "pH_water_initial"
    if reliable is False:
        if _as_bool(row.get("solver_rescued")) or xgems_water_matches_recipe(row) is False:
            return "pH_uncertain_water_adjusted"
        return "pH_uncertain"
    return "pH_water_unknown"


def _load_top_candidate(path: Path, *, rank: int = 1) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Candidate review file not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Candidate review file is empty: {path}")
    if "review_rank" in frame.columns:
        ranked = frame.sort_values("review_rank")
        matches = ranked[ranked["review_rank"].astype(int) == int(rank)]
        row = matches.iloc[0] if not matches.empty else ranked.iloc[0]
    else:
        row = frame.iloc[max(0, int(rank) - 1)]
    return row.to_dict()


def _candidate_binders(candidate: dict[str, Any]) -> dict[str, float]:
    binders: dict[str, float] = {}
    for key in BINDER_COLUMNS:
        value = candidate.get(key)
        if value is None or pd.isna(value):
            continue
        amount = float(value)
        if abs(amount) > 1.0e-12:
            binders[key] = amount
    if not binders:
        raise ValueError("Top candidate does not contain any binder mass columns.")
    return binders


def _age_grid(
    *,
    candidate: dict[str, Any],
    forward_age_values: list[float] | None,
    forward_age_start: float | None,
    forward_age_stop: float | None,
    forward_age_points: int,
    forward_age_spacing: str,
) -> dict[str, Any]:
    if forward_age_values:
        return {"values": [float(value) for value in forward_age_values]}
    if forward_age_start is not None or forward_age_stop is not None:
        if forward_age_start is None or forward_age_stop is None:
            raise ValueError("Both forward_age_start and forward_age_stop are required for an age range.")
        return {
            "start": float(forward_age_start),
            "stop": float(forward_age_stop),
            "points": int(forward_age_points),
            "spacing": forward_age_spacing,
        }
    age = candidate.get("age_days")
    if age is None or pd.isna(age):
        raise ValueError("Top candidate has no age_days value; provide forward_age_values.")
    return {"values": [float(age)]}


def _write_forward_query(
    *,
    path: Path,
    candidate: dict[str, Any],
    age_grid: dict[str, Any],
    phases: list[str],
    phase_groups: list[str],
    scalars: list[str],
    table_limit: int,
    top_phases: int,
    narrative_language: str,
) -> Path:
    recipe: dict[str, Any] = {"binders": _candidate_binders(candidate)}
    if candidate.get("w_b") is not None and not pd.isna(candidate.get("w_b")):
        recipe["w_b"] = float(candidate["w_b"])
    elif candidate.get("water_g") is not None and not pd.isna(candidate.get("water_g")):
        recipe["water_g"] = float(candidate["water_g"])
    else:
        raise ValueError("Top candidate has neither w_b nor water_g.")

    task = "forward_time_series"
    if list(age_grid) == ["values"] and len(age_grid["values"]) == 1:
        task = "forward_calculation"
    payload = {
        "name": "inverse_top_candidate_forward",
        "task": task,
        "material_system": candidate.get("material_system"),
        "recipe": recipe,
        "age_grid": age_grid,
        "temperature_celsius": 20.0,
        "outputs": {
            "phase_masses": "all",
            "phase_volumes": "all",
            "phase_volumes_reconstructed": "all",
            "aqueous_species": "all",
            "scalars": list(dict.fromkeys(scalars)),
        },
        "plots": [],
        "response_summary": {
            "enabled": True,
            "phases": list(dict.fromkeys(phases)),
            "phase_groups": list(dict.fromkeys(phase_groups)),
            "scalars": list(dict.fromkeys(scalars)),
            "top_phases": int(top_phases),
            "table_limit": int(table_limit),
            "narrative_enabled": True,
            "narrative_language": narrative_language,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return path


def _read_unreacted(db: Path, recipe_id: Any) -> dict[str, float]:
    if recipe_id is None or pd.isna(recipe_id):
        return {}
    path = db / "recipe_runs" / str(recipe_id) / "unreacted_masses.json"
    data = _load_json(path)
    return {str(key): float(value) for key, value in data.items()}


def build_selected_forward_timeseries(
    *,
    time_series_path: str | Path,
    db: str | Path,
    out_csv: str | Path,
    phases: list[str] | None = None,
) -> Path:
    phases = phases or DEFAULT_FORWARD_PHASES
    frame = pd.read_csv(time_series_path)
    rows: list[dict[str, Any]] = []
    db_path = Path(db)
    for _, row in frame.iterrows():
        unreacted = _read_unreacted(db_path, row.get("recipe_id"))
        out: dict[str, Any] = {
            "age_days": float(row.get("age_days")),
            "chemistry_status": row.get("chemistry_status"),
            "solver_status": row.get("solver_status"),
            "solver_rescued": _as_bool(row.get("solver_rescued")),
            "pH_flag": _pH_flag(row),
            "pH_water_reliable": is_pH_water_reliable(row),
            "pH_unreliable_reason": _value_or_none(row.get("pH_unreliable_reason")),
            "xgems_water_delta_g": xgems_water_delta(row),
            "xgems_water_matches_recipe": xgems_water_matches_recipe(row),
            "uncertainty_flags": _flags_text(row),
            "preflight_dir": _value_or_none(row.get("preflight_dir")),
            "w_b_recipe": float(row.get("w_b")) if pd.notna(row.get("w_b")) else None,
            "water_g_recipe": float(row.get("water_g")) if pd.notna(row.get("water_g")) else None,
            "xgems_w_b": float(row.get("xgems_w_b")) if pd.notna(row.get("xgems_w_b")) else None,
            "xgems_water_g": float(row.get("xgems_water_g")) if pd.notna(row.get("xgems_water_g")) else None,
            "unreacted_OPC_g": float(unreacted.get("OPC", 0.0)),
            "unreacted_slag_g": float(unreacted.get("slag", 0.0)),
            "unreacted_fly_ash_g": float(unreacted.get("fly_ash", 0.0)),
            "unreacted_metakaolin_g": float(unreacted.get("metakaolin", 0.0)),
            "unreacted_silica_fume_g": float(unreacted.get("silica_fume", 0.0)),
            "pH": float(row.get("scalar__pH")) if pd.notna(row.get("scalar__pH")) else None,
            "porosity": float(row.get("porosity")) if pd.notna(row.get("porosity")) else None,
        }
        for phase in phases:
            mass_col = f"phase_mass__{phase}"
            volume_col = f"phase_volume__{phase}"
            recon_col = f"phase_volume_reconstructed__{phase}"
            out[f"{phase}_mass_g"] = float(row[mass_col]) * 1000.0 if mass_col in frame.columns and pd.notna(row.get(mass_col)) else 0.0
            if recon_col in frame.columns and pd.notna(row.get(recon_col)):
                out[f"{phase}_volume_cm3_reconstructed"] = float(row[recon_col]) * 1_000_000.0
            elif volume_col in frame.columns and pd.notna(row.get(volume_col)):
                out[f"{phase}_volume_cm3_reconstructed"] = float(row[volume_col]) * 1_000_000.0
            else:
                out[f"{phase}_volume_cm3_reconstructed"] = 0.0
        rows.append(out)
    selected = pd.DataFrame(rows)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_path, index=False)
    return out_path


def _write_markdown_summary(
    *,
    path: Path,
    workflow_summary: dict[str, Any],
    selected_csv: Path,
    sample_rows: int = 8,
) -> None:
    selected = pd.read_csv(selected_csv)
    if len(selected) > sample_rows:
        indices = sorted(set(round(i) for i in pd.Series(range(sample_rows)).mul((len(selected) - 1) / max(sample_rows - 1, 1))))
        display = selected.iloc[indices]
    else:
        display = selected
    preferred = [
        "age_days",
        "unreacted_OPC_g",
        "unreacted_metakaolin_g",
        "unreacted_fly_ash_g",
        "CNASH_mass_g",
        "Portlandite_mass_g",
        "C4AH19_mass_g",
        "straetlingite_mass_g",
        "pH",
        "pH_water_reliable",
        "xgems_water_delta_g",
        "porosity",
        "pH_flag",
        "uncertainty_flags",
    ]
    cols = [col for col in preferred if col in display.columns]
    table = display[cols].copy()
    for col in ["uncertainty_flags", "pH_unreliable_reason", "preflight_dir"]:
        if col in table.columns:
            table[col] = table[col].map(lambda value: "" if is_missing(value) else str(value))
    for col in table.select_dtypes(include="number").columns:
        table[col] = table[col].map(lambda value: f"{value:.5g}" if col in {"age_days", "pH", "porosity"} else f"{value:.4f}")
    with path.open("w", encoding="utf-8") as handle:
        top = workflow_summary["top_candidate"]
        handle.write("# Inverse-to-forward workflow summary\n\n")
        handle.write("## Top inverse candidate\n\n")
        for key, value in top.get("binders", {}).items():
            handle.write(f"- {key}: {float(value):.6f} g / 100 g binder\n")
        handle.write(f"- w/b: {top.get('w_b')}\n")
        handle.write(f"- inverse age days: {top.get('age_days')}\n")
        handle.write(f"- validation status: {top.get('validation_status')}\n")
        handle.write(f"- solver status: {top.get('solver_status')}\n")
        handle.write("\n## Forward check\n\n")
        forward = workflow_summary["forward"]
        handle.write(f"- completed ages: {forward.get('completed_count')} / {forward.get('age_count')}\n")
        handle.write(f"- rescued ages: {forward.get('rescued_ages_days')}\n")
        handle.write(f"- pH water-unreliable ages: `{forward.get('pH_water_unreliable_count')}`\n")
        handle.write(f"- uncertainty flag counts: `{forward.get('uncertainty_flag_counts')}`\n")
        handle.write(f"- preflight reports available: `{forward.get('preflight_available_count')}`\n")
        handle.write(f"- selected time series: `{selected_csv.name}`\n\n")
        handle.write("| " + " | ".join(cols) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for _, row in table.iterrows():
            handle.write("| " + " | ".join(str(row[col]) for col in cols) + " |\n")


def run_inverse_forward_workflow(
    *,
    inverse_request: str,
    out: str | Path,
    db: str | Path,
    dat_lst: str | Path | None = None,
    global_db: str | Path | None = None,
    global_n_candidates: int = 500,
    strict_materials: bool | None = None,
    llm_config: str | Path | None = None,
    model: str | None = None,
    max_repairs: int | None = None,
    client: Any | None = None,
    model_registry: str | Path | None = None,
    material_systems_config: str | Path | None = None,
    route_target_policy: str = "recommended",
    validation_top_k: int | None = 5,
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
    use_mock: bool = False,
    candidate_rank: int = 1,
    forward_age_values: list[float] | None = None,
    forward_age_start: float | None = None,
    forward_age_stop: float | None = None,
    forward_age_points: int = 24,
    forward_age_spacing: str = "log",
    forward_phases: list[str] | None = None,
    forward_phase_groups: list[str] | None = None,
    forward_scalars: list[str] | None = None,
    forward_table_limit: int = 20,
    forward_top_phases: int = 8,
    forward_narrative_language: str = "ko",
) -> Path:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    inverse_dir = out_dir / "01_inverse"
    forward_dir = out_dir / "02_forward_top_candidate"
    summary_dir = out_dir / "workflow_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    chemistry_db = Path(global_db) if global_db is not None else Path(db)
    run_user_request_with_openai(
        user_request=inverse_request,
        out=inverse_dir,
        db=db,
        dat_lst=dat_lst,
        config=llm_config,
        model=model,
        max_repairs=max_repairs,
        client=client,
        global_db=global_db,
        global_n_candidates=global_n_candidates,
        strict_materials=strict_materials,
        model_registry=model_registry,
        material_systems_config=material_systems_config,
        route_target_policy=route_target_policy,
        use_mock=use_mock,
        skip_validation=False,
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

    candidate_review = inverse_dir / "task_run" / "design" / "candidate_review.csv"
    candidate = _load_top_candidate(candidate_review, rank=candidate_rank)
    phases = forward_phases or DEFAULT_FORWARD_PHASES
    phase_groups = forward_phase_groups or DEFAULT_FORWARD_PHASE_GROUPS
    scalars = forward_scalars or ["pH", "porosity"]
    ages = _age_grid(
        candidate=candidate,
        forward_age_values=forward_age_values,
        forward_age_start=forward_age_start,
        forward_age_stop=forward_age_stop,
        forward_age_points=forward_age_points,
        forward_age_spacing=forward_age_spacing,
    )
    forward_query = _write_forward_query(
        path=out_dir / "forward_top_candidate_query.yaml",
        candidate=candidate,
        age_grid=ages,
        phases=phases,
        phase_groups=phase_groups,
        scalars=scalars,
        table_limit=forward_table_limit,
        top_phases=forward_top_phases,
        narrative_language=forward_narrative_language,
    )
    run_forward_query(
        query=forward_query,
        out=forward_dir,
        db=chemistry_db,
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
        disable_plots=disable_plots,
        fail_fast=fail_fast,
    )
    selected_csv = build_selected_forward_timeseries(
        time_series_path=forward_dir / "time_series.csv",
        db=chemistry_db,
        out_csv=summary_dir / "forward_top_candidate_selected_timeseries.csv",
        phases=phases,
    )
    forward_summary = _load_json(forward_dir / "forward_query_summary.json")
    selected = pd.read_csv(selected_csv)
    rescued_ages = [float(value) for value in selected.loc[selected["solver_rescued"], "age_days"].tolist()]
    selected_records = selected.to_dict(orient="records")
    pH_reliable_values = selected.get("pH_water_reliable", pd.Series(dtype=object)).map(
        lambda value: str(value).strip().lower()
    )
    pH_known = pH_reliable_values.isin({"true", "1", "1.0", "yes", "false", "0", "0.0", "no"})
    pH_unreliable = pH_reliable_values.isin({"false", "0", "0.0", "no"})
    preflight_available_count = (
        int(selected["preflight_dir"].map(lambda value: not is_missing(value)).sum())
        if "preflight_dir" in selected.columns
        else 0
    )
    workflow_summary = {
        "inverse_request": inverse_request,
        "top_candidate": {
            "rank": int(candidate_rank),
            "binders": _candidate_binders(candidate),
            "w_b": float(candidate["w_b"]) if candidate.get("w_b") is not None and not pd.isna(candidate.get("w_b")) else None,
            "water_g": float(candidate["water_g"]) if candidate.get("water_g") is not None and not pd.isna(candidate.get("water_g")) else None,
            "age_days": float(candidate["age_days"]) if candidate.get("age_days") is not None and not pd.isna(candidate.get("age_days")) else None,
            "validation_status": candidate.get("validation_status"),
            "solver_status": candidate.get("solver_status"),
            "solver_rescued": _as_bool(candidate.get("solver_rescued")),
            "xgems_retry_count": int(candidate.get("xgems_retry_count") or 0),
            "uncertainty_flags": candidate.get("uncertainty_flags"),
            "pH_water_reliable": _value_or_none(candidate.get("pH_water_reliable")),
            "xgems_water_delta_g": as_float(candidate.get("xgems_water_delta_g")),
            "preflight_dir": _value_or_none(candidate.get("preflight_dir")),
            "validated": {key.removeprefix("validated__"): float(value) for key, value in candidate.items() if str(key).startswith("validated__") and value is not None and not pd.isna(value)},
        },
        "forward": {
            "query": str(forward_query),
            "age_grid": ages,
            "age_count": int(forward_summary.get("age_count") or len(selected)),
            "completed_count": int(forward_summary.get("completed_count") or (selected["chemistry_status"] == "complete").sum()),
            "failed_count": int(forward_summary.get("failed_count") or 0),
            "rescued_count": int(selected["solver_rescued"].sum()) if "solver_rescued" in selected.columns else 0,
            "rescued_ages_days": rescued_ages,
            "pH_water_known_count": int(pH_known.sum()),
            "pH_water_unreliable_count": int((pH_known & pH_unreliable).sum()),
            "uncertainty_flag_counts": flag_counts(selected_records),
            "preflight_available_count": preflight_available_count,
            "selected_timeseries": str(selected_csv),
        },
        "paths": {
            "inverse_dir": str(inverse_dir),
            "inverse_candidate_review": str(candidate_review),
            "forward_dir": str(forward_dir),
            "forward_time_series": str(forward_dir / "time_series.csv"),
            "selected_timeseries": str(selected_csv),
        },
    }
    write_json(summary_dir / "workflow_summary.json", workflow_summary)
    _write_markdown_summary(
        path=summary_dir / "workflow_summary.md",
        workflow_summary=workflow_summary,
        selected_csv=selected_csv,
    )
    return out_dir
