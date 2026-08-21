from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import write_json


DEFAULT_FEATURE_COLUMNS = [
    "meta__age_days",
    "meta__OPC",
    "meta__slag",
    "meta__fly_ash",
    "meta__metakaolin",
    "meta__silica_fume",
    "meta__limestone",
    "meta__gypsum",
    "meta__w_b",
    "meta__xgems_water_g",
]


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_csv(path)
    return pd.read_csv(path, low_memory=False)


def _safe_target_name(value: str) -> str:
    text = str(value).strip()
    text = text.replace("-", "_")
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _norm(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def resolve_target_column(frame: pd.DataFrame, target: str, *, prefer: str = "amount") -> str:
    if target in frame.columns:
        return target
    safe = _safe_target_name(target)
    preferred = f"y__{prefer}_{safe}"
    if preferred in frame.columns:
        return preferred
    candidates = [f"y__amount_{safe}", f"y__volume_{safe}", f"y__{safe}"]
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    normalized = _norm(target)
    matches = [column for column in frame.columns if column.startswith("y__") and _norm(column).endswith(normalized)]
    if matches:
        amount_matches = [column for column in matches if column.startswith("y__amount_")]
        volume_matches = [column for column in matches if column.startswith("y__volume_")]
        if prefer == "amount" and amount_matches:
            return amount_matches[0]
        if prefer == "volume" and volume_matches:
            return volume_matches[0]
        return matches[0]
    raise ValueError(f"Could not resolve target '{target}'. Available targets include: {sorted([c for c in frame.columns if c.startswith('y__')])[:20]}")


def _quantile_rows(frame: pd.DataFrame, columns: list[str], *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            rows.append({"subset": label, "column": column, "count": 0})
            continue
        rows.append(
            {
                "subset": label,
                "column": column,
                "count": int(len(values)),
                "min": float(values.min()),
                "p05": float(values.quantile(0.05)),
                "p10": float(values.quantile(0.10)),
                "p25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "p75": float(values.quantile(0.75)),
                "p90": float(values.quantile(0.90)),
                "p95": float(values.quantile(0.95)),
                "max": float(values.max()),
            }
        )
    return rows


def _material_system_summary(frame: pd.DataFrame, values: pd.Series, *, threshold: float) -> pd.DataFrame:
    system_col = "meta__material_system" if "meta__material_system" in frame.columns else "material_system"
    if system_col not in frame.columns:
        return pd.DataFrame()
    work = frame.copy()
    work["_target_value"] = values
    work["_target_nonzero"] = values.abs() > threshold
    rows: list[dict[str, Any]] = []
    for system, subset in work.groupby(system_col, dropna=False):
        target_values = pd.to_numeric(subset["_target_value"], errors="coerce").fillna(0.0)
        nonzero = subset[subset["_target_nonzero"]]
        rows.append(
            {
                "material_system": "" if pd.isna(system) else str(system),
                "row_count": int(len(subset)),
                "nonzero_count": int(len(nonzero)),
                "nonzero_fraction": 0.0 if len(subset) == 0 else float(len(nonzero) / len(subset)),
                "target_max": float(target_values.max()) if len(target_values) else 0.0,
                "target_mean": float(target_values.mean()) if len(target_values) else 0.0,
                "target_nonzero_mean": float(nonzero["_target_value"].mean()) if len(nonzero) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["nonzero_count", "target_max"], ascending=[False, False])


def write_target_region_analysis(
    *,
    model_table: str | Path,
    target: str,
    out: str | Path,
    threshold: float = 1.0e-30,
    top_n: int = 50,
    feature_columns: list[str] | tuple[str, ...] | None = None,
    prefer: str = "amount",
) -> Path:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = _read_table(model_table)
    target_column = resolve_target_column(frame, target, prefer=prefer)
    values = pd.to_numeric(frame[target_column], errors="coerce").fillna(0.0)
    nonzero_mask = values.abs() > float(threshold)
    nonzero = frame[nonzero_mask].copy()
    zero = frame[~nonzero_mask].copy()
    nonzero["_target_value"] = values[nonzero_mask]
    zero["_target_value"] = values[~nonzero_mask]
    columns = [column for column in (feature_columns or DEFAULT_FEATURE_COLUMNS) if column in frame.columns]

    sort_columns = ["_target_value"]
    top_rows = nonzero.sort_values(sort_columns, ascending=False).head(int(top_n)).copy()
    material_summary = _material_system_summary(frame, values, threshold=float(threshold))
    quantiles = pd.DataFrame(
        _quantile_rows(nonzero, columns, label="nonzero")
        + _quantile_rows(zero, columns, label="zero")
        + _quantile_rows(frame, columns, label="all")
    )

    nonzero_path = out_dir / "target_region_nonzero_rows.csv"
    top_path = out_dir / "target_region_top_rows.csv"
    material_path = out_dir / "target_region_by_material_system.csv"
    quantile_path = out_dir / "target_region_quantiles.csv"
    nonzero.to_csv(nonzero_path, index=False)
    top_rows.to_csv(top_path, index=False)
    material_summary.to_csv(material_path, index=False)
    quantiles.to_csv(quantile_path, index=False)

    summary = {
        "model_table": str(model_table),
        "target_requested": str(target),
        "target_column": target_column,
        "threshold": float(threshold),
        "row_count": int(len(frame)),
        "nonzero_count": int(nonzero_mask.sum()),
        "zero_count": int((~nonzero_mask).sum()),
        "nonzero_fraction": 0.0 if len(frame) == 0 else float(nonzero_mask.mean()),
        "target_max": float(values.max()) if len(values) else 0.0,
        "target_mean": float(values.mean()) if len(values) else 0.0,
        "feature_columns": columns,
        "top_material_systems": material_summary.head(10).to_dict(orient="records"),
        "outputs": {
            "nonzero_rows": str(nonzero_path),
            "top_rows": str(top_path),
            "material_system_summary": str(material_path),
            "quantiles": str(quantile_path),
        },
    }
    write_json(out_dir / "target_region_summary.json", summary)

    lines = [
        f"# Target Region Analysis: `{target_column}`",
        "",
        f"- Rows: `{summary['row_count']}`",
        f"- Nonzero rows: `{summary['nonzero_count']}` ({summary['nonzero_fraction']:.4f})",
        f"- Target max: `{summary['target_max']:.6g}`",
        "",
        "## Top Material Systems",
        "",
    ]
    if material_summary.empty:
        lines.append("No material-system column was available.")
    else:
        lines.extend(["| material_system | rows | nonzero | fraction | max |", "| --- | ---: | ---: | ---: | ---: |"])
        for row in material_summary.head(10).to_dict(orient="records"):
            lines.append(
                f"| {row['material_system']} | {row['row_count']} | {row['nonzero_count']} | "
                f"{row['nonzero_fraction']:.4f} | {row['target_max']:.6g} |"
            )
    lines.extend(
        [
            "",
            "Files:",
            "- `target_region_nonzero_rows.csv`",
            "- `target_region_top_rows.csv`",
            "- `target_region_by_material_system.csv`",
            "- `target_region_quantiles.csv`",
        ]
    )
    (out_dir / "target_region_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_dir
