from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .materials import BINDER_COMPONENTS
from .utils import to_jsonable, write_json


PREFIXES = {
    "phase_masses": "phase_mass__",
    "phase_volumes": "phase_volume__",
    "phase_volumes_reconstructed": "phase_volume_reconstructed__",
    "aqueous_species": "aqueous__",
    "raw_scalars": "scalar__",
}

METADATA_COLUMNS = [
    "query_run_id",
    "row_index",
    "recipe_id",
    "chem_hash",
    "chemistry_status",
    "solver_status",
    "reused_cache",
    "solver_rescued",
    "xgems_retry_count",
    "recipe_text",
    "age_days",
    "water_g",
    "w_b",
    "xgems_water_g",
    "xgems_w_b",
    "xgems_water_mode",
    "xgems_water_delta_g",
    "xgems_water_matches_recipe",
    "pH_water_reliable",
    "pH_unreliable_reason",
    "uncertainty_flags",
    "preflight_dir",
    "initial_volume_cm3",
    "final_solid_volume_cm3",
    "porosity",
    "xgems_run_dir",
    "error_message",
]


def _clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    return to_jsonable(value)


def _strip_prefixed(row: dict[str, Any], prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for column, value in row.items():
        if column.startswith(prefix):
            out[column[len(prefix) :]] = _clean_value(value)
    return dict(sorted(out.items()))


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {column: _clean_value(row[column]) for column in METADATA_COLUMNS if column in row}


def _binder_masses(row: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for component in sorted(BINDER_COMPONENTS):
        value = row.get(component)
        if value is not None and not pd.isna(value):
            out[component] = float(value)
    return out


def _write_name_value_csv(path: Path, values: dict[str, Any]) -> str:
    pd.DataFrame([{"name": name, "value": value} for name, value in values.items()], columns=["name", "value"]).to_csv(
        path, index=False
    )
    return str(path)


def _top_items(values: dict[str, Any], limit: int = 12) -> list[tuple[str, float]]:
    numeric: list[tuple[str, float]] = []
    for name, value in values.items():
        try:
            numeric.append((name, float(value or 0.0)))
        except Exception:
            continue
    return sorted(numeric, key=lambda item: abs(item[1]), reverse=True)[:limit]


def _markdown_table(items: list[tuple[str, float]]) -> str:
    if not items:
        return "_None._"
    lines = ["| raw_name | value |", "| --- | ---: |"]
    for name, value in items:
        lines.append(f"| {name} | {value:.12g} |")
    return "\n".join(lines)


def _write_summary_md(out_dir: Path, payload: dict[str, Any]) -> str:
    meta = payload["metadata"]
    lines = [
        "# Forward Calculation Summary",
        "",
        f"- age_days: {meta.get('age_days')}",
        f"- chemistry_status: {meta.get('chemistry_status')}",
        f"- solver_status: {meta.get('solver_status')}",
        f"- recipe_id: `{meta.get('recipe_id')}`",
        f"- chem_hash: `{meta.get('chem_hash')}`",
        f"- porosity: {meta.get('porosity')}",
        "",
        "Raw phase names are preserved exactly as returned in `time_series.csv`; no aliases or aggregation are applied.",
        "",
        "## Binder Masses",
        "",
        _markdown_table([(name, value) for name, value in payload["binder_masses_g"].items() if value != 0.0]),
        "",
        "## Largest Raw Phase Masses",
        "",
        _markdown_table(_top_items(payload["phase_masses"])),
        "",
        "## Largest Raw Phase Volumes",
        "",
        _markdown_table(_top_items(payload["phase_volumes"])),
        "",
        "## Raw Scalars",
        "",
    ]
    scalars = payload["raw_scalars"]
    if scalars:
        lines.extend(f"- {name}: {value}" for name, value in scalars.items())
    else:
        lines.append("_None._")
    path = out_dir / "calculation_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def write_forward_single_result(frame: pd.DataFrame, out_dir: str | Path) -> dict[str, Any] | None:
    if len(frame) != 1:
        return None
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    row = frame.iloc[0].to_dict()
    payload = {
        "metadata": _metadata(row),
        "binder_masses_g": _binder_masses(row),
        "phase_masses": _strip_prefixed(row, PREFIXES["phase_masses"]),
        "phase_volumes": _strip_prefixed(row, PREFIXES["phase_volumes"]),
        "phase_volumes_reconstructed": _strip_prefixed(row, PREFIXES["phase_volumes_reconstructed"]),
        "aqueous_species": _strip_prefixed(row, PREFIXES["aqueous_species"]),
        "raw_scalars": _strip_prefixed(row, PREFIXES["raw_scalars"]),
    }
    payload["calculation_scalars"] = {
        key: _clean_value(row[key])
        for key in ["porosity", "initial_volume_cm3", "final_solid_volume_cm3", "water_g", "w_b", "xgems_water_g", "xgems_w_b"]
        if key in row
    }
    files = {
        "single_result_json": str(out_path / "single_result.json"),
        "single_result_csv": str(out_path / "single_result.csv"),
        "raw_phase_masses": _write_name_value_csv(out_path / "raw_phase_masses.csv", payload["phase_masses"]),
        "raw_phase_volumes": _write_name_value_csv(out_path / "raw_phase_volumes.csv", payload["phase_volumes"]),
        "raw_phase_volumes_reconstructed": _write_name_value_csv(
            out_path / "raw_phase_volumes_reconstructed.csv", payload["phase_volumes_reconstructed"]
        ),
        "raw_aqueous_species": _write_name_value_csv(out_path / "raw_aqueous_species.csv", payload["aqueous_species"]),
        "raw_scalars": str(out_path / "raw_scalars.json"),
    }
    frame.to_csv(out_path / "single_result.csv", index=False)
    write_json(out_path / "single_result.json", payload)
    write_json(out_path / "raw_scalars.json", payload["raw_scalars"])
    files["calculation_summary"] = _write_summary_md(out_path, payload)
    return {
        "files": files,
        "age_days": payload["metadata"].get("age_days"),
        "chemistry_status": payload["metadata"].get("chemistry_status"),
        "solver_status": payload["metadata"].get("solver_status"),
        "phase_mass_count": len(payload["phase_masses"]),
        "phase_volume_count": len(payload["phase_volumes"]),
        "scalar_count": len(payload["raw_scalars"]),
    }
