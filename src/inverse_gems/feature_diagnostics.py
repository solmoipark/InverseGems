from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import write_json


BINDER_COLUMNS = ["OPC", "slag", "fly_ash", "metakaolin", "silica_fume", "limestone", "gypsum"]
INPUT_COLUMNS = BINDER_COLUMNS + ["water_g", "w_b", "xgems_water_g", "xgems_w_b", "age_days", "temperature_celsius"]
EPS = 1.0e-30


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_csv(path)
    return pd.read_csv(path)


def _numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("float64", copy=False)


def _finite_values(series: pd.Series) -> pd.Series:
    numeric = _numeric_series(series)
    return numeric[np.isfinite(numeric)]


def _safe_corr(a: pd.Series, b: pd.Series) -> float | None:
    left = _numeric_series(a)
    right = _numeric_series(b)
    valid = left.notna() & right.notna() & np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return None
    left = left[valid]
    right = right[valid]
    if left.nunique(dropna=True) <= 1 or right.nunique(dropna=True) <= 1:
        return None
    value = left.corr(right)
    if value is None or math.isnan(float(value)):
        return None
    return float(value)


def _quantile(values: pd.Series, q: float) -> float | None:
    if values.empty:
        return None
    return float(values.quantile(q))


def _column_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in frame.columns:
        numeric = _numeric_series(frame[column])
        finite = numeric[np.isfinite(numeric)]
        rows.append(
            {
                "column": column,
                "is_numeric": bool(pd.api.types.is_numeric_dtype(frame[column]) or numeric.notna().any()),
                "missing_count": int(numeric.isna().sum()),
                "nonfinite_count": int(numeric.notna().sum() - np.isfinite(numeric.dropna()).sum()),
                "unique_count": int(frame[column].nunique(dropna=True)),
                "zero_count": int((finite.abs() <= EPS).sum()),
                "zero_fraction": None if finite.empty else float((finite.abs() <= EPS).mean()),
                "nonzero_count": int((finite.abs() > EPS).sum()),
                "negative_count": int((finite < 0.0).sum()),
                "min": None if finite.empty else float(finite.min()),
                "p01": _quantile(finite, 0.01),
                "p05": _quantile(finite, 0.05),
                "p25": _quantile(finite, 0.25),
                "median": _quantile(finite, 0.50),
                "mean": None if finite.empty else float(finite.mean()),
                "p75": _quantile(finite, 0.75),
                "p95": _quantile(finite, 0.95),
                "p99": _quantile(finite, 0.99),
                "max": None if finite.empty else float(finite.max()),
                "std": None if len(finite) <= 1 else float(finite.std()),
            }
        )
    return pd.DataFrame(rows)


def _prefixed(columns: list[str], prefix: str) -> list[str]:
    return [column for column in columns if column.startswith(prefix)]


def _phase_pair_summary(frame: pd.DataFrame, *, amount_prefix: str, volume_prefix: str) -> pd.DataFrame:
    columns = list(frame.columns)
    names = sorted(
        {column.removeprefix(amount_prefix) for column in _prefixed(columns, amount_prefix)}
        | {column.removeprefix(volume_prefix) for column in _prefixed(columns, volume_prefix)}
    )
    rows: list[dict[str, Any]] = []
    for name in names:
        amount_col = f"{amount_prefix}{name}"
        volume_col = f"{volume_prefix}{name}"
        amount = _numeric_series(frame[amount_col]) if amount_col in frame else pd.Series(0.0, index=frame.index)
        volume = _numeric_series(frame[volume_col]) if volume_col in frame else pd.Series(0.0, index=frame.index)
        amount_pos = amount.fillna(0.0).abs() > EPS
        volume_pos = volume.fillna(0.0).abs() > EPS
        density = amount[amount_pos & volume_pos] / volume[amount_pos & volume_pos]
        rows.append(
            {
                "name": name,
                "amount_column": amount_col if amount_col in frame else "",
                "volume_column": volume_col if volume_col in frame else "",
                "amount_nonzero_count": int(amount_pos.sum()),
                "volume_nonzero_count": int(volume_pos.sum()),
                "amount_nonzero_fraction": float(amount_pos.mean()),
                "volume_nonzero_fraction": float(volume_pos.mean()),
                "amount_positive_volume_zero_count": int((amount_pos & ~volume_pos).sum()),
                "volume_positive_amount_zero_count": int((volume_pos & ~amount_pos).sum()),
                "amount_sum": float(amount.fillna(0.0).sum()),
                "volume_sum": float(volume.fillna(0.0).sum()),
                "amount_max": float(amount.fillna(0.0).max()),
                "volume_max": float(volume.fillna(0.0).max()),
                "density_median_if_amount_volume_positive": None if density.empty else float(density.median()),
                "density_p05_if_amount_volume_positive": _quantile(density, 0.05),
                "density_p95_if_amount_volume_positive": _quantile(density, 0.95),
            }
        )
    return pd.DataFrame(rows)


def _correlation_with_targets(frame: pd.DataFrame, target_columns: list[str]) -> pd.DataFrame:
    fields = ["target", "column", "pearson", "abs_pearson"]
    numeric_columns = [
        column
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column]) or _numeric_series(frame[column]).notna().any()
    ]
    rows: list[dict[str, Any]] = []
    for target in target_columns:
        if target not in frame:
            continue
        for column in numeric_columns:
            if column == target:
                continue
            corr = _safe_corr(frame[column], frame[target])
            if corr is not None:
                rows.append({"target": target, "column": column, "pearson": corr, "abs_pearson": abs(corr)})
    if not rows:
        return pd.DataFrame(columns=fields)
    return pd.DataFrame(rows, columns=fields).sort_values(["target", "abs_pearson"], ascending=[True, False])


def _high_correlation_pairs(frame: pd.DataFrame, columns: list[str], threshold: float) -> pd.DataFrame:
    fields = ["column_a", "column_b", "pearson", "abs_pearson"]
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    nonconstant = [column for column in numeric.columns if numeric[column].nunique(dropna=True) > 1]
    if len(nonconstant) < 2:
        return pd.DataFrame(columns=fields)
    corr = numeric[nonconstant].corr()
    rows: list[dict[str, Any]] = []
    for i, left in enumerate(corr.columns):
        for right in corr.columns[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and abs(float(value)) >= threshold:
                rows.append({"column_a": left, "column_b": right, "pearson": float(value), "abs_pearson": abs(float(value))})
    if not rows:
        return pd.DataFrame(columns=fields)
    return pd.DataFrame(rows, columns=fields).sort_values("abs_pearson", ascending=False)


def _input_output_correlation(frame: pd.DataFrame, output_columns: list[str]) -> pd.DataFrame:
    fields = ["output", "input", "pearson", "abs_pearson"]
    input_columns = [column for column in INPUT_COLUMNS if column in frame.columns]
    rows: list[dict[str, Any]] = []
    for output in output_columns:
        for input_column in input_columns:
            corr = _safe_corr(frame[input_column], frame[output])
            if corr is not None:
                rows.append({"output": output, "input": input_column, "pearson": corr, "abs_pearson": abs(corr)})
    if not rows:
        return pd.DataFrame(columns=fields)
    return pd.DataFrame(rows, columns=fields).sort_values(["output", "abs_pearson"], ascending=[True, False])


def _write_plot_outputs(frame: pd.DataFrame, out_dir: Path, phase_group_summary: pd.DataFrame, warnings: list[str]) -> list[str]:
    plot_files: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        warnings.append(f"matplotlib unavailable; skipped plots: {type(exc).__name__}: {exc}")
        return plot_files

    def save_current(name: str) -> None:
        path = out_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        plot_files.append(name)

    if "porosity" in frame.columns:
        plt.figure(figsize=(8, 4.5))
        _finite_values(frame["porosity"]).hist(bins=60)
        plt.xlabel("porosity")
        plt.ylabel("count")
        plt.title("Porosity Distribution")
        save_current("porosity_hist.png")

    if "scalar__pH" in frame.columns:
        plt.figure(figsize=(8, 4.5))
        _finite_values(frame["scalar__pH"]).hist(bins=60)
        plt.xlabel("pH")
        plt.ylabel("count")
        plt.title("pH Distribution")
        save_current("pH_hist.png")

    binder_columns = [column for column in BINDER_COLUMNS if column in frame.columns]
    if binder_columns:
        plt.figure(figsize=(9, 4.8))
        frame[binder_columns].plot.box(ax=plt.gca(), rot=35)
        plt.ylabel("g per 100 g binder")
        plt.title("Binder Component Distribution")
        save_current("binder_boxplot.png")

    if not phase_group_summary.empty:
        subset = phase_group_summary.sort_values("amount_nonzero_fraction", ascending=True)
        plt.figure(figsize=(9, max(4.5, 0.35 * len(subset))))
        y = np.arange(len(subset))
        plt.barh(y - 0.17, subset["amount_nonzero_fraction"], height=0.34, label="amount")
        plt.barh(y + 0.17, subset["volume_nonzero_fraction"], height=0.34, label="volume")
        plt.yticks(y, subset["name"])
        plt.xlabel("nonzero fraction")
        plt.title("Phase Group Nonzero Fraction")
        plt.legend()
        save_current("phase_group_nonzero_fraction.png")

    corr_columns = [column for column in INPUT_COLUMNS if column in frame.columns]
    corr_columns += [
        column
        for column in frame.columns
        if column.startswith("phase_amount_group__") or column.startswith("phase_volume_group__") or column in {"porosity", "scalar__pH"}
    ]
    corr_columns = list(dict.fromkeys(corr_columns))
    if 2 <= len(corr_columns) <= 80:
        corr = frame[corr_columns].apply(pd.to_numeric, errors="coerce").corr()
        plt.figure(figsize=(max(8, 0.28 * len(corr_columns)), max(7, 0.28 * len(corr_columns))))
        image = plt.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")
        plt.colorbar(image, fraction=0.046, pad=0.04)
        plt.xticks(range(len(corr_columns)), corr_columns, rotation=90, fontsize=7)
        plt.yticks(range(len(corr_columns)), corr_columns, fontsize=7)
        plt.title("Input And Selected Output Correlation")
        save_current("input_output_correlation_heatmap.png")

    return plot_files


def run_feature_diagnostics(
    *,
    feature_table: str | Path,
    out: str | Path,
    correlation_threshold: float = 0.98,
    sparse_threshold: float = 0.01,
    plot: bool = True,
) -> Path:
    frame = _read_table(feature_table)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    columns = list(frame.columns)

    column_summary = _column_summary(frame)
    phase_summary = _phase_pair_summary(frame, amount_prefix="phase_amount__", volume_prefix="phase_volume__")
    phase_group_summary = _phase_pair_summary(
        frame,
        amount_prefix="phase_amount_group__",
        volume_prefix="phase_volume_group__",
    )

    target_columns = [column for column in ["porosity", "scalar__porosity", "scalar__pH"] if column in frame.columns]
    target_columns += [column for column in frame.columns if column.startswith("phase_amount_group__")]
    target_columns += [column for column in frame.columns if column.startswith("phase_volume_group__")]
    target_columns = list(dict.fromkeys(target_columns))

    corr_targets = _correlation_with_targets(frame, target_columns)
    input_output_corr = _input_output_correlation(frame, target_columns)
    model_candidate_columns = [
        column
        for column in frame.columns
        if column in INPUT_COLUMNS
        or column.startswith("alpha_")
        or column.endswith("_alpha_eff")
        or column.endswith("_D_eff")
        or column.startswith("phase_amount_group__")
        or column.startswith("phase_volume_group__")
        or column in {"porosity", "scalar__pH", "scalar__system_mass", "scalar__system_volume"}
    ]
    high_corr = _high_correlation_pairs(frame, model_candidate_columns, correlation_threshold)

    zero_or_sparse = column_summary[
        (column_summary["is_numeric"])
        & (column_summary["nonzero_count"] <= max(1, int(len(frame) * sparse_threshold)))
    ].copy()

    column_summary.to_csv(out_dir / "numeric_column_summary.csv", index=False)
    phase_summary.to_csv(out_dir / "phase_amount_volume_summary.csv", index=False)
    phase_group_summary.to_csv(out_dir / "phase_group_amount_volume_summary.csv", index=False)
    corr_targets.to_csv(out_dir / "correlation_with_targets.csv", index=False)
    input_output_corr.to_csv(out_dir / "input_output_correlations.csv", index=False)
    high_corr.to_csv(out_dir / "high_correlation_pairs.csv", index=False)
    zero_or_sparse.to_csv(out_dir / "zero_or_sparse_numeric_columns.csv", index=False)

    if "age_bin" in frame.columns:
        age_output_columns = [column for column in target_columns if column in frame.columns]
        frame.groupby("age_bin", dropna=False)[age_output_columns].mean(numeric_only=True).to_csv(
            out_dir / "age_bin_output_means.csv"
        )
    if "template_name" in frame.columns:
        template_output_columns = [column for column in target_columns if column in frame.columns]
        frame.groupby("template_name", dropna=False)[template_output_columns].mean(numeric_only=True).to_csv(
            out_dir / "template_output_means.csv"
        )

    plot_files = _write_plot_outputs(frame, out_dir, phase_group_summary, warnings) if plot else []

    mismatch_rows = phase_summary[
        (phase_summary["amount_positive_volume_zero_count"] > 0)
        | (phase_summary["volume_positive_amount_zero_count"] > 0)
    ]
    group_mismatch_rows = phase_group_summary[
        (phase_group_summary["amount_positive_volume_zero_count"] > 0)
        | (phase_group_summary["volume_positive_amount_zero_count"] > 0)
    ]
    summary = {
        "feature_table": str(feature_table),
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "chemistry_status_counts": frame["chemistry_status"].value_counts(dropna=False).to_dict()
        if "chemistry_status" in frame.columns
        else {},
        "duplicate_recipe_id_count": int(frame["recipe_id"].duplicated().sum()) if "recipe_id" in frame.columns else None,
        "selected_phase_count": int(len(phase_summary)),
        "selected_phase_group_count": int(len(phase_group_summary)),
        "selected_phase_amount_volume_mismatch_count": int(len(mismatch_rows)),
        "selected_phase_group_amount_volume_mismatch_count": int(len(group_mismatch_rows)),
        "all_zero_numeric_columns": column_summary[
            (column_summary["is_numeric"]) & (column_summary["nonzero_count"] == 0)
        ]["column"].tolist(),
        "zero_or_sparse_numeric_column_count": int(len(zero_or_sparse)),
        "high_correlation_pair_count": int(len(high_corr)),
        "porosity": _summary_for_column(frame, "porosity"),
        "pH": _summary_for_column(frame, "scalar__pH"),
        "plot_files": plot_files,
        "warnings": warnings,
    }
    write_json(out_dir / "feature_diagnostics_summary.json", summary)
    return out_dir


def _summary_for_column(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    if column not in frame.columns:
        return {}
    values = _finite_values(frame[column])
    if values.empty:
        return {"count": 0}
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "max": float(values.max()),
    }
