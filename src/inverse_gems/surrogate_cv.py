"""Repeated group-shuffle CV evaluation for surrogate targets.

Single-split R2 on sparse targets is dominated by which nonzero rows land in
the test split (the 8-cycle hemicarbonate campaign showed swings from -0.41
to +0.17 with nearly identical data). This module evaluates the same
estimator/config over N independent group-shuffle splits and reports
per-target mean/std/min/max, giving an honest uncertainty band for
improvement claims. Uses the exact split machinery of the training path
(group leakage control and sparse-target split support included).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from .surrogate import (
    _columns_from_schema_or_prefix,
    _load_schema,
    _make_estimator,
    _read_table,
    _rmse,
    _split_indices,
)
from .target_region_analysis import resolve_target_column
from .utils import config_path, load_yaml, write_json


def evaluate_surrogate_repeated_cv(
    *,
    model_table: str | Path,
    out: str | Path,
    config: str | Path | None = None,
    n_repeats: int = 5,
    targets: list[str] | tuple[str, ...] | None = None,
    base_random_state: int | None = None,
) -> Path:
    """Evaluate surrogate target metrics over repeated group-shuffle splits."""
    if int(n_repeats) < 2:
        raise ValueError("n_repeats must be at least 2 for a repeated-CV estimate.")
    config_data = load_yaml(Path(config) if config else config_path("surrogate_baseline.yaml"))
    frame = _read_table(model_table)
    schema = _load_schema(model_table)
    inputs = _columns_from_schema_or_prefix(schema, frame, "inputs", "x__")
    target_columns = _columns_from_schema_or_prefix(schema, frame, "targets", "y__")
    if not inputs or not target_columns:
        raise ValueError("Model table must contain input columns and target columns.")
    if targets:
        target_columns = [resolve_target_column(frame, str(name)) for name in targets]
    data = frame.dropna(subset=inputs + list(target_columns)).copy()

    split_cfg = dict(config_data.get("split") or {})
    base_state = int(
        base_random_state if base_random_state is not None else split_cfg.get("random_state", 42)
    )
    nonzero_threshold = float((config_data.get("evaluation") or {}).get("nonzero_threshold", 1.0e-12))

    per_repeat_rows: list[dict[str, Any]] = []
    for repeat in range(int(n_repeats)):
        repeat_config = copy.deepcopy(config_data)
        repeat_config.setdefault("split", {})
        repeat_config["split"]["random_state"] = base_state + 1000 * repeat
        train_idx, test_idx, split_report = _split_indices(
            data, repeat_config, target_columns=list(target_columns)
        )
        x_train = data.iloc[train_idx][inputs]
        x_test = data.iloc[test_idx][inputs]
        y_train = data.iloc[train_idx][list(target_columns)]
        y_test = data.iloc[test_idx][list(target_columns)]
        estimator = _make_estimator(repeat_config)
        estimator.fit(x_train, y_train)
        pred = estimator.predict(x_test)
        if pred.ndim == 1:
            pred = pred[:, None]
        for index, target in enumerate(target_columns):
            true = y_test[target].to_numpy()
            predicted = pred[:, index]
            row = {
                "repeat": repeat,
                "random_state": repeat_config["split"]["random_state"],
                "selected_random_state": split_report.get("selected_random_state"),
                "target": target,
                "test_rows": int(len(true)),
                "test_nonzero": int((np.abs(true) > nonzero_threshold).sum()),
                "r2": float(r2_score(true, predicted)) if len(true) > 1 else float("nan"),
                "rmse": _rmse(true, predicted),
            }
            per_repeat_rows.append(row)

    per_repeat = pd.DataFrame(per_repeat_rows)
    aggregated = (
        per_repeat.groupby("target")
        .agg(
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            r2_min=("r2", "min"),
            r2_max=("r2", "max"),
            rmse_mean=("rmse", "mean"),
            test_nonzero_mean=("test_nonzero", "mean"),
            test_nonzero_min=("test_nonzero", "min"),
            n_repeats=("repeat", "nunique"),
        )
        .reset_index()
        .sort_values("r2_mean", ascending=False)
    )

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_repeat.to_csv(out_dir / "cv_per_repeat_metrics.csv", index=False)
    aggregated.to_csv(out_dir / "cv_target_metrics.csv", index=False)
    summary = {
        "model_table": str(model_table),
        "n_repeats": int(n_repeats),
        "base_random_state": base_state,
        "rows": int(len(data)),
        "input_count": len(inputs),
        "target_count": len(target_columns),
        "targets": [
            {
                key: (value if isinstance(value, str) else (None if pd.isna(value) else float(value)))
                for key, value in record.items()
            }
            for record in aggregated.to_dict(orient="records")
        ],
    }
    write_json(out_dir / "cv_summary.json", summary)

    lines = [
        "# Repeated Group-Shuffle CV",
        "",
        f"- Model table: `{model_table}`  ·  rows: `{len(data)}`  ·  repeats: `{n_repeats}`",
        "",
        "| Target | R2 mean ± std | R2 min…max | Test nonzero (mean) |",
        "|---|---|---|---|",
    ]
    for record in aggregated.to_dict(orient="records"):
        lines.append(
            f"| {record['target']} | {record['r2_mean']:.3f} ± {record['r2_std']:.3f} "
            f"| {record['r2_min']:.3f}…{record['r2_max']:.3f} | {record['test_nonzero_mean']:.1f} |"
        )
    (out_dir / "cv_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_dir
