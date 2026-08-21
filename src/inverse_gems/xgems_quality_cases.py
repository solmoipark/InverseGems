from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from .global_chemistry_db import load_global_manifest
from .utils import write_json


TRUE_VALUES = {"true", "1", "1.0", "yes", "y"}
COMPLETE_STATUS_VALUES = {"complete", "success", "ok", "cached"}
COMPONENTS = ["OPC", "slag", "fly_ash", "metakaolin", "silica_fume", "limestone", "gypsum"]
AGE_BANDS: tuple[tuple[str, float], ...] = (
    ("<=0.25d", 0.25),
    ("<=1d", 1.0),
    ("<=3d", 3.0),
    ("<=7d", 7.0),
    ("<=28d", 28.0),
    ("<=90d", 90.0),
    ("<=180d", 180.0),
    ("<=365d", 365.0),
)
W_B_BANDS: tuple[tuple[str, float], ...] = (
    ("<=0.35", 0.35),
    ("<=0.40", 0.40),
    ("<=0.45", 0.45),
    ("<=0.50", 0.50),
    ("<=0.55", 0.55),
)


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_csv(path)
    return pd.read_csv(path)


def _column(frame: pd.DataFrame, *names: str) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def _value_column(frame: pd.DataFrame, name: str) -> str | None:
    return _column(frame, name, f"meta__{name}", f"x__{name}")


def _series(frame: pd.DataFrame, *names: str, default: Any = None) -> pd.Series:
    column = _column(frame, *names)
    if column is None:
        return pd.Series([default] * len(frame), index=frame.index)
    return frame[column]


def _numeric_series(frame: pd.DataFrame, *names: str) -> pd.Series:
    return pd.to_numeric(_series(frame, *names, default=float("nan")), errors="coerce")


def _bool_series(frame: pd.DataFrame, *names: str, default: bool = False) -> pd.Series:
    return _series(frame, *names, default=default).fillna(default).astype(str).str.strip().str.lower().isin(TRUE_VALUES)


def _status_complete(frame: pd.DataFrame) -> pd.Series:
    status = _series(frame, "meta__chemistry_status", "chemistry_status", default="")
    return status.fillna("").astype(str).str.strip().str.lower().isin(COMPLETE_STATUS_VALUES)


def _age_band(age: Any) -> str:
    try:
        value = float(age)
    except (TypeError, ValueError):
        return "missing"
    if not math.isfinite(value):
        return "missing"
    for label, upper in AGE_BANDS:
        if value <= upper:
            return label
    return ">365d"


def _w_b_band(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "missing"
    if not math.isfinite(number):
        return "missing"
    for label, upper in W_B_BANDS:
        if number <= upper:
            return label
    return ">0.55"


def _source_model_table(*, db: str | Path | None, model_table: str | Path | None) -> tuple[Path, dict[str, Any]]:
    if model_table is not None:
        return Path(model_table), {}
    if db is None:
        raise ValueError("Provide either --db or --model-table.")
    manifest = load_global_manifest(db)
    value = (manifest.get("paths") or {}).get("model_table")
    if not value:
        raise ValueError("Global DB manifest does not include paths.model_table.")
    return Path(value), manifest


def _annotate(frame: pd.DataFrame, *, water_tolerance_g: float) -> pd.DataFrame:
    out = frame.copy()
    out["analysis__age_days"] = _numeric_series(out, "meta__age_days", "x__age_days", "age_days")
    out["analysis__age_band"] = out["analysis__age_days"].map(_age_band)
    out["analysis__w_b"] = _numeric_series(out, "meta__w_b", "x__w_b", "w_b")
    out["analysis__w_b_band"] = out["analysis__w_b"].map(_w_b_band)
    out["analysis__solver_rescued"] = _bool_series(out, "meta__solver_rescued", "solver_rescued")
    out["analysis__retry_count"] = _numeric_series(out, "meta__xgems_retry_count", "xgems_retry_count").fillna(0.0)
    water = _numeric_series(out, "meta__water_g", "water_g", "x__water_g")
    xgems_water = _numeric_series(out, "meta__xgems_water_g", "xgems_water_g", "x__xgems_water_g")
    out["analysis__water_delta_g"] = xgems_water - water
    out["analysis__abs_water_delta_g"] = out["analysis__water_delta_g"].abs()
    out["analysis__xgems_water_adjusted"] = out["analysis__abs_water_delta_g"] > float(water_tolerance_g)
    explicit_pH = _column(out, "pH_water_reliable", "meta__pH_water_reliable")
    if explicit_pH:
        out["analysis__pH_water_reliable"] = out[explicit_pH].fillna(True).astype(str).str.strip().str.lower().isin(TRUE_VALUES)
    else:
        out["analysis__pH_water_reliable"] = ~(out["analysis__solver_rescued"] | out["analysis__xgems_water_adjusted"].fillna(False))
    out["analysis__chemistry_complete"] = _status_complete(out)
    out["analysis__problem_case"] = (
        (~out["analysis__chemistry_complete"])
        | out["analysis__solver_rescued"]
        | out["analysis__xgems_water_adjusted"].fillna(False)
        | (~out["analysis__pH_water_reliable"])
    )
    return out


def _material_filter(frame: pd.DataFrame, material_system: str | None) -> pd.DataFrame:
    if material_system is None:
        return frame
    column = _column(frame, "meta__material_system", "material_system")
    if column is None:
        raise ValueError("Model table has no material system column.")
    return frame[frame[column].astype(str) == str(material_system)].copy()


def _preferred_case_columns(frame: pd.DataFrame) -> list[str]:
    names = [
        "meta__recipe_id",
        "meta__chem_hash",
        "meta__material_system",
        "analysis__age_days",
        "analysis__age_band",
        "analysis__w_b",
        "analysis__w_b_band",
        *[f"meta__{component}" for component in COMPONENTS],
        "meta__w_b",
        "meta__water_g",
        "meta__xgems_water_g",
        "meta__xgems_w_b",
        "analysis__water_delta_g",
        "analysis__abs_water_delta_g",
        "analysis__solver_rescued",
        "analysis__retry_count",
        "analysis__xgems_water_adjusted",
        "analysis__pH_water_reliable",
        "analysis__chemistry_complete",
        "analysis__problem_case",
        "meta__chemistry_status",
        "y__pH",
        "y__porosity",
        "y__amount_C_A_S_H",
        "y__amount_Portlandite",
        "y__amount_ettringite",
    ]
    available = [name for name in names if name in frame.columns]
    extras = [name for name in frame.columns if name.startswith("analysis__") and name not in available]
    return available + extras


def _summary_rows(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    if group_column not in frame.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for value, subset in frame.groupby(group_column, dropna=False):
        row_count = int(len(subset))
        problem_count = int(subset["analysis__problem_case"].sum()) if row_count else 0
        rows.append(
            {
                group_column: value,
                "row_count": row_count,
                "problem_count": problem_count,
                "problem_fraction": 0.0 if row_count == 0 else float(problem_count / row_count),
                "solver_rescued_count": int(subset["analysis__solver_rescued"].sum()),
                "xgems_water_adjusted_count": int(subset["analysis__xgems_water_adjusted"].fillna(False).sum()),
                "pH_water_unreliable_count": int((~subset["analysis__pH_water_reliable"]).sum()),
                "max_abs_water_delta_g": None
                if subset["analysis__abs_water_delta_g"].dropna().empty
                else float(subset["analysis__abs_water_delta_g"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["problem_fraction", "problem_count"], ascending=[False, False])


def _write_markdown(
    *,
    path: Path,
    summary: dict[str, Any],
    by_material: pd.DataFrame,
    by_age_band: pd.DataFrame,
    by_w_b_band: pd.DataFrame,
    top_cases: pd.DataFrame,
) -> None:
    lines = ["# xGEMS Quality Case Analysis", ""]
    lines.append(f"- model table: `{summary.get('model_table')}`")
    lines.append(f"- material system filter: `{summary.get('material_system')}`")
    lines.append(f"- source rows after filter: `{summary.get('source_row_count')}`")
    lines.append(f"- exported case rows: `{summary.get('case_row_count')}`")
    lines.append(f"- problem rows after filter: `{summary.get('problem_row_count')}`")
    lines.append(f"- water tolerance g: `{summary.get('water_tolerance_g')}`")
    if summary.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in summary["warnings"]:
            lines.append(f"- {warning}")
    for title, frame, columns in [
        (
            "By Material System",
            by_material,
            [
                "meta__material_system",
                "row_count",
                "problem_count",
                "problem_fraction",
                "solver_rescued_count",
                "xgems_water_adjusted_count",
                "pH_water_unreliable_count",
                "max_abs_water_delta_g",
            ],
        ),
        (
            "By Age Band",
            by_age_band,
            [
                "analysis__age_band",
                "row_count",
                "problem_count",
                "problem_fraction",
                "solver_rescued_count",
                "xgems_water_adjusted_count",
                "pH_water_unreliable_count",
                "max_abs_water_delta_g",
            ],
        ),
        (
            "By w/b Band",
            by_w_b_band,
            [
                "analysis__w_b_band",
                "row_count",
                "problem_count",
                "problem_fraction",
                "solver_rescued_count",
                "xgems_water_adjusted_count",
                "pH_water_unreliable_count",
                "max_abs_water_delta_g",
            ],
        ),
        (
            "Top Cases",
            top_cases,
            [
                "meta__recipe_id",
                "meta__material_system",
                "analysis__age_days",
                "analysis__w_b_band",
                "meta__OPC",
                "meta__slag",
                "meta__fly_ash",
                "meta__w_b",
                "meta__water_g",
                "meta__xgems_water_g",
                "analysis__water_delta_g",
                "analysis__retry_count",
                "y__pH",
                "y__porosity",
            ],
        ),
    ]:
        lines.extend(["", f"## {title}", ""])
        if frame.empty:
            lines.append("No data.")
            continue
        use_columns = [column for column in columns if column in frame.columns]
        lines.append("| " + " | ".join(use_columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(use_columns)) + " |")
        for row in frame[use_columns].to_dict(orient="records"):
            lines.append("| " + " | ".join(str(row.get(column, "")).replace("|", "\\|") for column in use_columns) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_xgems_quality_case_report(
    *,
    out: str | Path,
    db: str | Path | None = None,
    model_table: str | Path | None = None,
    material_system: str | None = None,
    only_problem_cases: bool = True,
    top_n: int = 50,
    water_tolerance_g: float = 1.0e-9,
) -> Path:
    model_table_path, manifest = _source_model_table(db=db, model_table=model_table)
    frame = _read_table(model_table_path)
    annotated = _annotate(frame, water_tolerance_g=water_tolerance_g)
    filtered = _material_filter(annotated, material_system)
    problems = filtered[filtered["analysis__problem_case"]].copy()
    cases = problems.copy() if only_problem_cases else filtered.copy()
    cases = cases.sort_values(
        ["analysis__problem_case", "analysis__abs_water_delta_g", "analysis__retry_count"],
        ascending=[False, False, False],
    )
    preferred = _preferred_case_columns(cases)
    cases_out = cases[preferred].copy() if preferred else cases.copy()
    top_cases = cases_out.head(int(top_n)).copy() if top_n and top_n > 0 else cases_out.copy()
    by_material = _summary_rows(filtered, _column(filtered, "meta__material_system", "material_system") or "meta__material_system")
    by_age_band = _summary_rows(filtered, "analysis__age_band")
    by_w_b_band = _summary_rows(filtered, "analysis__w_b_band")
    warnings: list[str] = []
    if filtered.empty:
        warnings.append("No rows matched the requested filter.")
    elif problems.empty:
        warnings.append("No problem cases matched the requested filter.")
    summary = {
        "db": None if db is None else str(db),
        "model_table": str(model_table_path),
        "material_system": material_system,
        "only_problem_cases": bool(only_problem_cases),
        "water_tolerance_g": float(water_tolerance_g),
        "source_row_count": int(len(filtered)),
        "problem_row_count": int(len(problems)),
        "case_row_count": int(len(cases_out)),
        "top_n": int(top_n),
        "manifest_db": manifest.get("db"),
        "warnings": warnings,
        "outputs": {
            "summary_json": str(Path(out) / "xgems_quality_case_summary.json"),
            "cases_csv": str(Path(out) / "xgems_quality_cases.csv"),
            "top_cases_csv": str(Path(out) / "xgems_quality_top_cases.csv"),
            "by_material_system_csv": str(Path(out) / "xgems_quality_by_material_system.csv"),
            "by_age_band_csv": str(Path(out) / "xgems_quality_by_age_band.csv"),
            "by_w_b_band_csv": str(Path(out) / "xgems_quality_by_w_b_band.csv"),
            "markdown": str(Path(out) / "xgems_quality_case_analysis.md"),
        },
    }
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_out.to_csv(out_dir / "xgems_quality_cases.csv", index=False)
    top_cases.to_csv(out_dir / "xgems_quality_top_cases.csv", index=False)
    by_material.to_csv(out_dir / "xgems_quality_by_material_system.csv", index=False)
    by_age_band.to_csv(out_dir / "xgems_quality_by_age_band.csv", index=False)
    by_w_b_band.to_csv(out_dir / "xgems_quality_by_w_b_band.csv", index=False)
    write_json(out_dir / "xgems_quality_case_summary.json", summary)
    _write_markdown(
        path=out_dir / "xgems_quality_case_analysis.md",
        summary=summary,
        by_material=by_material,
        by_age_band=by_age_band,
        by_w_b_band=by_w_b_band,
        top_cases=top_cases,
    )
    return out_dir
