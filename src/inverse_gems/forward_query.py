from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .cached_forward import run_forward_cached
from .call_budget import XGEMSCallBudget, XGEMSCallBudgetExceeded
from .database import InverseGemsDatabase, read_name_value_csv
from .forward_answer import write_forward_answer
from .forward_narrative import write_forward_narrative
from .forward_query_diagnostics import write_forward_query_diagnostics
from .forward_result_summary import summarize_forward_result
from .forward_single_result import write_forward_single_result
from .materials import BINDER_COMPONENTS, canonicalize_material_name, load_materials
from .uncertainty import pH_water_reliable, uncertainty_flags, xgems_water_delta, xgems_water_matches_recipe
from .utils import load_yaml, short_hash, timestamp_compact, write_json


class AgeGridSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    values: list[float] | None = None
    start: float | None = None
    stop: float | None = None
    points: int | None = Field(default=None, ge=1)
    spacing: Literal["linear", "log"] = "linear"

    @model_validator(mode="after")
    def validate_grid(self) -> "AgeGridSpec":
        if self.values is not None:
            if not self.values:
                raise ValueError("age_grid.values cannot be empty.")
            if any(float(value) <= 0.0 for value in self.values):
                raise ValueError("All age values must be positive.")
            return self
        if self.start is None or self.stop is None or self.points is None:
            raise ValueError("age_grid must define either values or start/stop/points.")
        if self.start <= 0.0 or self.stop <= 0.0:
            raise ValueError("age_grid start and stop must be positive.")
        if self.stop < self.start:
            raise ValueError("age_grid stop cannot be smaller than start.")
        return self


class ForwardRecipeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    binders: dict[str, float] | None = None
    OPC: float | None = None
    slag: float | None = None
    fly_ash: float | None = None
    metakaolin: float | None = None
    silica_fume: float | None = None
    limestone: float | None = None
    gypsum: float | None = None
    w_b: float | None = None
    water_g: float | None = None

    @model_validator(mode="after")
    def validate_recipe(self) -> "ForwardRecipeSpec":
        binders = self.binder_masses()
        if not binders:
            raise ValueError("recipe must include at least one binder component.")
        if any(value < 0.0 for value in binders.values()):
            raise ValueError("binder masses cannot be negative.")
        total = sum(binders.values())
        if total <= 0.0:
            raise ValueError("total binder mass must be positive.")
        if self.w_b is None and self.water_g is None:
            raise ValueError("recipe must include either w_b or water_g.")
        if self.w_b is not None and self.water_g is not None:
            raise ValueError("recipe must include only one of w_b or water_g.")
        if self.w_b is not None and self.w_b <= 0.0:
            raise ValueError("w_b must be positive.")
        if self.water_g is not None and self.water_g <= 0.0:
            raise ValueError("water_g must be positive.")
        return self

    def binder_masses(self) -> dict[str, float]:
        materials = load_materials()
        out: dict[str, float] = {}
        for name, value in (self.binders or {}).items():
            canonical = canonicalize_material_name(name, materials)
            if canonical not in BINDER_COMPONENTS:
                raise ValueError(f"{canonical} is not a supported binder component.")
            out[canonical] = out.get(canonical, 0.0) + float(value)
        for name in sorted(BINDER_COMPONENTS):
            value = getattr(self, name, None)
            if value is not None:
                out[name] = out.get(name, 0.0) + float(value)
        return {name: value for name, value in out.items() if abs(value) > 0.0}


class ForwardOutputsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase_masses: str | list[str] = "all"
    phase_volumes: str | list[str] = "all"
    phase_volumes_reconstructed: str | list[str] = "all"
    aqueous_species: str | list[str] = "all"
    scalars: str | list[str] = "all"


class ForwardPlotSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["phase_volumes", "phase_volumes_reconstructed", "phase_masses", "aqueous_species", "scalars"] = (
        "phase_volumes"
    )
    names: str | list[str] = "all_nonzero"
    top_n: int | None = Field(default=12, ge=1)
    filename: str | None = None
    x_scale: Literal["linear", "log"] = "log"
    y_scale: Literal["linear", "log"] = "linear"
    title: str | None = None


class ForwardResponseSummarySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    phases: list[str] = Field(default_factory=list)
    phase_groups: list[str] = Field(default_factory=list)
    scalars: list[str] = Field(default_factory=list)
    top_phases: int = Field(default=8, ge=0)
    table_limit: int = Field(default=20, ge=1)
    narrative_enabled: bool = True
    narrative_language: Literal["ko", "en"] = "ko"


class ForwardQuerySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    task: Literal["forward_calculation", "forward_time_series"] = "forward_time_series"
    recipe: ForwardRecipeSpec
    age_grid: AgeGridSpec
    material_system: str | None = None
    temperature_celsius: float = 20.0
    outputs: ForwardOutputsSpec = Field(default_factory=ForwardOutputsSpec)
    plots: list[ForwardPlotSpec] = Field(default_factory=lambda: [ForwardPlotSpec()])
    response_summary: ForwardResponseSummarySpec = Field(default_factory=ForwardResponseSummarySpec)


def expand_age_grid(spec: AgeGridSpec | dict[str, Any]) -> list[float]:
    spec = spec if isinstance(spec, AgeGridSpec) else AgeGridSpec.model_validate(spec)
    if spec.values is not None:
        return [float(value) for value in spec.values]
    assert spec.start is not None and spec.stop is not None and spec.points is not None
    if spec.points == 1:
        return [float(spec.start)]
    if spec.spacing == "log":
        return [float(value) for value in np.geomspace(float(spec.start), float(spec.stop), int(spec.points))]
    return [float(value) for value in np.linspace(float(spec.start), float(spec.stop), int(spec.points))]


def _recipe_text_for_age(recipe: ForwardRecipeSpec, age_days: float) -> str:
    label_map = {
        "OPC": "OPC",
        "slag": "slag",
        "fly_ash": "fly ash",
        "metakaolin": "metakaolin",
        "silica_fume": "silica fume",
        "limestone": "limestone",
        "gypsum": "gypsum",
    }
    parts = [f"{label_map[name]} {mass:g}" for name, mass in sorted(recipe.binder_masses().items()) if mass != 0.0]
    if recipe.w_b is not None:
        parts.append(f"w/b {float(recipe.w_b):g}")
    else:
        parts.append(f"water {float(recipe.water_g):g}")
    parts.append(f"age {float(age_days):.12g}")
    return ", ".join(parts)


def validate_forward_query_data(data: dict[str, Any]) -> dict[str, Any]:
    return ForwardQuerySpec.model_validate(data).model_dump(mode="json", exclude_none=True)


def validate_forward_query_file(query: str | Path) -> dict[str, Any]:
    return validate_forward_query_data(load_yaml(query))


def forward_query_json_schema() -> dict[str, Any]:
    return ForwardQuerySpec.model_json_schema()


def save_forward_query_schema(out: str | Path) -> Path:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, forward_query_json_schema())
    return out_path


def _load_scalars(raw_dir: Path) -> dict[str, Any]:
    path = raw_dir / "xgems_scalars_raw.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _names_requested(spec: str | list[str], values: dict[str, Any]) -> list[str]:
    if isinstance(spec, list):
        return [str(name) for name in spec]
    if spec in {"all", "all_nonzero"}:
        names = sorted(str(name) for name in values)
        if spec == "all_nonzero":
            return [name for name in names if abs(float(values.get(name) or 0.0)) > 0.0]
        return names
    return [str(spec)]


def _add_prefixed_values(row: dict[str, Any], prefix: str, requested: str | list[str], values: dict[str, Any]) -> None:
    for name in _names_requested(requested, values):
        row[f"{prefix}__{name}"] = values.get(name, 0.0)


def _row_from_result(
    *,
    database: InverseGemsDatabase,
    result: dict[str, Any],
    query_run_id: str,
    row_index: int,
    recipe_text: str,
    outputs: ForwardOutputsSpec,
) -> dict[str, Any]:
    recipe_row = database.get_recipe_run(str(result["recipe_id"])) or {}
    chemistry_row = database.get_chemistry_run(str(result["chem_hash"])) or {}
    raw_dir = Path(str(chemistry_row.get("xgems_run_dir") or ""))
    recipe_data = json.loads(recipe_row.get("recipe_json") or "{}")
    binder = recipe_data.get("binder_masses_g") or {}
    row: dict[str, Any] = {
        "query_run_id": query_run_id,
        "row_index": row_index,
        "recipe_id": result.get("recipe_id"),
        "chem_hash": result.get("chem_hash"),
        "chemistry_status": result.get("chemistry_status"),
        "solver_status": result.get("solver_status"),
        "reused_cache": result.get("reused_cache"),
        "solver_rescued": result.get("solver_rescued"),
        "xgems_retry_count": result.get("xgems_retry_count"),
        "recipe_text": recipe_text,
        "age_days": recipe_row.get("age_days", recipe_data.get("age_days")),
        "water_g": recipe_row.get("water_g", recipe_data.get("water_g")),
        "w_b": recipe_row.get("w_b", recipe_data.get("w_b")),
        "xgems_water_g": recipe_row.get("xgems_water_g"),
        "xgems_w_b": recipe_row.get("xgems_w_b"),
        "xgems_water_mode": recipe_row.get("xgems_water_mode"),
        "preflight_dir": result.get("preflight_dir"),
        "initial_volume_cm3": recipe_row.get("initial_volume_cm3"),
        "final_solid_volume_cm3": recipe_row.get("final_solid_volume_cm3"),
        "porosity": recipe_row.get("porosity"),
        "xgems_run_dir": chemistry_row.get("xgems_run_dir"),
    }
    row["xgems_water_delta_g"] = xgems_water_delta(row)
    row["xgems_water_matches_recipe"] = xgems_water_matches_recipe(row)
    row["pH_water_reliable"] = pH_water_reliable(row)
    if row["pH_water_reliable"] is False:
        reasons: list[str] = []
        if row.get("solver_rescued"):
            reasons.append("solver_rescued")
        if row.get("xgems_water_matches_recipe") is False:
            reasons.append("xgems_water_changed")
        row["pH_unreliable_reason"] = ";".join(reasons) or "unknown"
    else:
        row["pH_unreliable_reason"] = ""
    row["uncertainty_flags"] = ";".join(uncertainty_flags(row))
    for component in sorted(BINDER_COMPONENTS):
        row[component] = float(binder.get(component, 0.0))
    phase_masses = read_name_value_csv(raw_dir / "xgems_phase_amounts_raw.csv")
    phase_volumes = read_name_value_csv(raw_dir / "xgems_phase_volumes_raw.csv")
    phase_volumes_reconstructed = read_name_value_csv(raw_dir / "xgems_phase_volumes_reconstructed.csv")
    aqueous_species = read_name_value_csv(raw_dir / "xgems_aqueous_species_raw.csv")
    scalars = _load_scalars(raw_dir)
    _add_prefixed_values(row, "phase_mass", outputs.phase_masses, phase_masses)
    _add_prefixed_values(row, "phase_volume", outputs.phase_volumes, phase_volumes)
    _add_prefixed_values(row, "phase_volume_reconstructed", outputs.phase_volumes_reconstructed, phase_volumes_reconstructed)
    _add_prefixed_values(row, "aqueous", outputs.aqueous_species, aqueous_species)
    for name in _names_requested(outputs.scalars, scalars):
        value = scalars.get(name, 0.0)
        if isinstance(value, (int, float, bool)) or value is None:
            row[f"scalar__{name}"] = value
    return row


def _columns_for_plot(frame: pd.DataFrame, plot: ForwardPlotSpec) -> list[str]:
    prefix = {
        "phase_volumes": "phase_volume__",
        "phase_volumes_reconstructed": "phase_volume_reconstructed__",
        "phase_masses": "phase_mass__",
        "aqueous_species": "aqueous__",
        "scalars": "scalar__",
    }[plot.kind]
    if isinstance(plot.names, list):
        candidates = [f"{prefix}{name}" for name in plot.names]
    elif plot.names in {"all", "all_nonzero"}:
        candidates = [column for column in frame.columns if column.startswith(prefix)]
    else:
        candidates = [f"{prefix}{plot.names}"]
    existing = [column for column in candidates if column in frame.columns]
    if plot.names == "all_nonzero":
        existing = [
            column
            for column in existing
            if pd.to_numeric(frame[column], errors="coerce").fillna(0.0).abs().max() > 0.0
        ]
    if plot.top_n and len(existing) > plot.top_n:
        existing = sorted(
            existing,
            key=lambda column: pd.to_numeric(frame[column], errors="coerce").fillna(0.0).abs().max(),
            reverse=True,
        )[: plot.top_n]
    return existing


def _write_plot(frame: pd.DataFrame, plot: ForwardPlotSpec, out_dir: Path, warnings: list[str]) -> str | None:
    columns = _columns_for_plot(frame, plot)
    if not columns:
        warnings.append(f"No columns matched plot request for kind={plot.kind}, names={plot.names}.")
        return None
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        warnings.append(f"matplotlib unavailable; skipped plot {plot.kind}: {type(exc).__name__}: {exc}")
        return None

    fig = None
    try:
        filename = plot.filename or f"{plot.kind}_vs_age.png"
        path = out_dir / filename
        x = pd.to_numeric(frame["age_days"], errors="coerce")
        fig, axis = plt.subplots(figsize=(9, 5.5))
        for column in columns:
            y = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
            axis.plot(x, y, marker="o", linewidth=1.4, label=column.split("__", 1)[1])
        axis.set_xlabel("age_days")
        axis.set_ylabel(plot.kind)
        axis.set_title(plot.title or f"{plot.kind} vs age")
        if plot.x_scale == "log" and (x > 0).all():
            axis.set_xscale("log")
        if plot.y_scale == "log":
            positive = frame[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
            if (positive > 0).any():
                axis.set_yscale("log")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        return str(path)
    except KeyboardInterrupt:
        raise
    except BaseException as exc:
        warnings.append(f"Skipped plot {plot.kind}: {type(exc).__name__}: {exc}")
        return None
    finally:
        if fig is not None:
            try:
                plt.close(fig)
            except Exception:
                pass


def run_forward_query(
    *,
    query: str | Path,
    out: str | Path,
    db: str | Path,
    dat_lst: str | Path | None = None,
    use_mock: bool = False,
    run_mode: str = "reacted_only",
    normalize: bool = True,
    allow_non_100: bool = False,
    pressure: float | None = None,
    force_rerun_xgems: bool = False,
    gems_class_path: str = "xgems:ChemicalEngineDicts",
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
    max_xgems_calls: int | None = None,
    reaction_model_id: str | None = None,
    reaction_model_config: str | Path | None = None,
    disable_plots: bool = False,
    fail_fast: bool = False,
) -> Path:
    query_path = Path(query)
    query_data = load_yaml(query_path)
    spec = ForwardQuerySpec.model_validate(query_data)
    if not use_mock and dat_lst is None:
        raise ValueError("--dat-lst is required for real forward-query runs.")

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    query_run_id = f"forward_query_{timestamp_compact()}_{short_hash(query_data, 8)}_{uuid.uuid4().hex[:6]}"
    shutil.copy2(query_path, out_dir / "forward_query_used.yaml")
    database = InverseGemsDatabase(db)
    ages = expand_age_grid(spec.age_grid)
    call_budget = XGEMSCallBudget(max_xgems_calls) if max_xgems_calls else None
    rows: list[dict[str, Any]] = []
    run_results: list[dict[str, Any]] = []
    warnings: list[str] = []
    for index, age_days in enumerate(ages, 1):
        recipe_text = _recipe_text_for_age(spec.recipe, age_days)
        recipe_id = f"{query_run_id}_age_{index:04d}"
        try:
            result = run_forward_cached(
                recipe_text=recipe_text,
                db=db,
                dat_lst=dat_lst,
                use_mock=use_mock,
                force_rerun_xgems=force_rerun_xgems,
                run_mode=run_mode,
                normalize=normalize,
                allow_non_100=allow_non_100,
                temperature_celsius=spec.temperature_celsius,
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
                retry_water_policy=retry_water_policy,
                retry_water_max_retries=retry_water_max_retries,
                xgems_call_budget=call_budget,
                reaction_model_id=reaction_model_id,
                reaction_model_config=reaction_model_config,
                recipe_id=recipe_id,
                recipe_metadata={
                    "forward_query_run_id": query_run_id,
                    "material_system": spec.material_system or "",
                    "template_name": spec.name or "forward_query",
                },
            )
            run_results.append(result)
            rows.append(
                _row_from_result(
                    database=database,
                    result=result,
                    query_run_id=query_run_id,
                    row_index=index,
                    recipe_text=recipe_text,
                    outputs=spec.outputs,
                )
            )
        except XGEMSCallBudgetExceeded as exc:
            skipped_ages = [f"{value:g}" for value in ages[index - 1 :]]
            message = (
                f"xGEMS call budget exhausted at age_days={age_days:g}; "
                f"skipping remaining age(s) {skipped_ages}. {exc}"
            )
            warnings.append(message)
            for skip_offset, skip_age in enumerate(ages[index - 1 :]):
                rows.append(
                    {
                        "query_run_id": query_run_id,
                        "row_index": index + skip_offset,
                        "recipe_id": f"{query_run_id}_age_{index + skip_offset:04d}",
                        "chemistry_status": "skipped_budget",
                        "error_message": message,
                        "recipe_text": _recipe_text_for_age(spec.recipe, skip_age),
                        "age_days": skip_age,
                        "preflight_dir": "",
                    }
                )
            break
        except Exception as exc:
            message = f"age_days={age_days:g} failed: {type(exc).__name__}: {exc}"
            warnings.append(message)
            run_results.append(
                {
                    "recipe_id": recipe_id,
                    "age_days": age_days,
                    "chemistry_status": "failed",
                    "error_message": message,
                }
            )
            rows.append(
                {
                    "query_run_id": query_run_id,
                    "row_index": index,
                    "recipe_id": recipe_id,
                    "chemistry_status": "failed",
                    "error_message": message,
                    "recipe_text": recipe_text,
                    "age_days": age_days,
                }
            )
            if fail_fast:
                raise
    frame = pd.DataFrame(rows).sort_values("age_days")
    time_series_path = out_dir / "time_series.csv"
    frame.to_csv(time_series_path, index=False)
    diagnostics = write_forward_query_diagnostics(frame, out_dir)
    single_result = write_forward_single_result(frame, out_dir)
    response_summary: dict[str, Any] | None = None
    if spec.response_summary.enabled:
        response_dir = summarize_forward_result(
            run=out_dir,
            phases=spec.response_summary.phases,
            phase_groups=spec.response_summary.phase_groups,
            scalars=spec.response_summary.scalars,
            top_phases=spec.response_summary.top_phases,
            table_limit=spec.response_summary.table_limit,
        )
        response_summary = {
            "enabled": True,
            "out": str(response_dir),
            "json": str(response_dir / "response_summary.json"),
            "markdown": str(response_dir / "response_summary.md"),
            "csv": str(response_dir / "response_summary.csv"),
        }
        answer_dir = write_forward_answer(
            summary=response_dir / "response_summary.json",
            table_limit=min(spec.response_summary.table_limit, 10),
        )
        response_summary["answer"] = {
            "json": str(answer_dir / "answer.json"),
            "markdown": str(answer_dir / "answer.md"),
            "out": str(answer_dir),
        }
        if spec.response_summary.narrative_enabled:
            narrative_dir = write_forward_narrative(
                answer=answer_dir / "answer.json",
                language=spec.response_summary.narrative_language,
                use_openai=False,
            )
            response_summary["narrative"] = {
                "json": str(narrative_dir / "narrative_answer.json"),
                "markdown": str(narrative_dir / "narrative_answer.md"),
                "out": str(narrative_dir),
                "mode": "deterministic",
            }
    plot_files = []
    if disable_plots:
        warnings.append("Forward-query plots disabled by --no-plots.")
    else:
        for plot in spec.plots:
            path = _write_plot(frame, plot, out_dir, warnings)
            if path:
                plot_files.append(path)
    summary = {
        "query": str(query_path),
        "out": str(out_dir),
        "db": str(db),
        "dat_lst": str(dat_lst) if dat_lst else None,
        "use_mock": use_mock,
        "query_run_id": query_run_id,
        "age_count": len(ages),
        "completed_count": sum(1 for result in run_results if result.get("chemistry_status") == "complete"),
        "failed_count": sum(1 for result in run_results if result.get("chemistry_status") == "failed"),
        "time_series": str(time_series_path),
        "diagnostics": diagnostics,
        "single_result": single_result,
        "response_summary": response_summary or {"enabled": False},
        "plot_files": plot_files,
        "run_results": run_results,
        "warnings": warnings,
    }
    write_json(out_dir / "forward_query_summary.json", summary)
    write_json(out_dir / "forward_query_manifest.json", spec.model_dump(mode="json", exclude_none=True))
    with (out_dir / "forward_query_used.normalized.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(spec.model_dump(mode="json", exclude_none=True), handle, sort_keys=False)
    return out_dir
