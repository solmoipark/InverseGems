from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.multioutput import MultiOutputRegressor

from .reaction_provenance import reaction_provenance_from_frame
from .utils import config_path, load_yaml, write_json


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_csv(path)
    return pd.read_csv(path)


def _load_schema(model_table: str | Path) -> dict[str, Any]:
    path = Path(model_table)
    schema_path = path.with_suffix(path.suffix + ".schema.json")
    if schema_path.exists():
        import json

        return json.loads(schema_path.read_text(encoding="utf-8"))
    return {}


def _columns_by_prefix(frame: pd.DataFrame, prefix: str) -> list[str]:
    return [column for column in frame.columns if column.startswith(prefix)]


def _columns_from_schema_or_prefix(schema: dict[str, Any], frame: pd.DataFrame, role: str, prefix: str) -> list[str]:
    roles = (schema.get("roles") or {}).get(role, [])
    columns = [item.get("column") for item in roles if item.get("column") in frame.columns]
    return [str(column) for column in columns] or _columns_by_prefix(frame, prefix)


def _base_groups(series: pd.Series, regex: str | None) -> np.ndarray:
    values = series.astype(str)
    if not regex:
        return values.to_numpy()
    pattern = re.compile(regex)
    return values.map(lambda value: (pattern.search(value).group(1) if pattern.search(value) else value)).to_numpy()


def _sparse_split_targets(
    frame: pd.DataFrame,
    target_columns: list[str],
    *,
    nonzero_threshold: float,
    sparse_fraction_threshold: float,
    include_targets: list[str] | None = None,
) -> list[dict[str, Any]]:
    include = {str(target) for target in include_targets or [] if str(target).strip()}
    rows: list[dict[str, Any]] = []
    for target in target_columns:
        if target not in frame.columns:
            continue
        values = pd.to_numeric(frame[target], errors="coerce").fillna(0.0).abs()
        nonzero_count = int((values > nonzero_threshold).sum())
        fraction = 0.0 if len(values) == 0 else float(nonzero_count / len(values))
        if include and target not in include:
            continue
        if include or (0 < fraction < sparse_fraction_threshold):
            rows.append(
                {
                    "target": target,
                    "full_nonzero_count": nonzero_count,
                    "full_nonzero_fraction": fraction,
                }
            )
    return rows


def _score_sparse_split(
    frame: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    sparse_targets: list[dict[str, Any]],
    *,
    nonzero_threshold: float,
    min_test_nonzero_count: int,
    desired_test_count: int,
) -> tuple[tuple[float, ...], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    satisfied = 0
    total_deficit = 0
    min_test_support = None
    capped_support = 0
    for item in sparse_targets:
        target = str(item["target"])
        train_values = pd.to_numeric(frame.iloc[train_idx][target], errors="coerce").fillna(0.0).abs()
        test_values = pd.to_numeric(frame.iloc[test_idx][target], errors="coerce").fillna(0.0).abs()
        train_count = int((train_values > nonzero_threshold).sum())
        test_count = int((test_values > nonzero_threshold).sum())
        target_satisfied = test_count >= min_test_nonzero_count
        deficit = max(0, min_test_nonzero_count - test_count)
        satisfied += int(target_satisfied)
        total_deficit += deficit
        min_test_support = test_count if min_test_support is None else min(min_test_support, test_count)
        capped_support += min(test_count, min_test_nonzero_count)
        rows.append(
            {
                "target": target,
                "full_nonzero_count": int(item["full_nonzero_count"]),
                "full_nonzero_fraction": float(item["full_nonzero_fraction"]),
                "train_nonzero_count": train_count,
                "test_nonzero_count": test_count,
                "test_nonzero_deficit": deficit,
                "satisfied": target_satisfied,
            }
        )
    score = (
        float(satisfied),
        float(-total_deficit),
        float(min_test_support or 0),
        float(capped_support),
        float(-abs(len(test_idx) - desired_test_count)),
    )
    return score, rows


def _split_indices(
    frame: pd.DataFrame,
    config: dict[str, Any],
    target_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    split = config.get("split", {}) or {}
    strategy = str(split.get("strategy", "group_shuffle"))
    test_size = float(split.get("test_size", 0.2))
    random_state = int(split.get("random_state", 42))
    indices = np.arange(len(frame))
    if strategy == "random":
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
        return np.asarray(train_idx), np.asarray(test_idx), {
            "strategy": strategy,
            "test_size": test_size,
            "random_state": random_state,
            "train_groups": None,
            "test_groups": None,
        }
    if strategy != "group_shuffle":
        raise ValueError("Surrogate split.strategy must be 'group_shuffle' or 'random'.")
    group_column = str(split.get("group_column", "meta__recipe_id"))
    if group_column not in frame.columns:
        raise ValueError(f"Group split requested missing group_column '{group_column}'.")
    groups = _base_groups(frame[group_column], split.get("group_regex"))
    sparse_cfg = split.get("sparse_target_support") or {}
    sparse_enabled = bool(sparse_cfg.get("enabled", False)) and bool(target_columns)
    evaluation = config.get("evaluation", {}) or {}
    sparse_targets: list[dict[str, Any]] = []
    selected_random_state = random_state
    sparse_report: dict[str, Any] = {
        "enabled": sparse_enabled,
        "target_count": 0,
        "targets": [],
    }
    if sparse_enabled:
        nonzero_threshold = float(sparse_cfg.get("nonzero_threshold", evaluation.get("nonzero_threshold", 1.0e-12)))
        sparse_fraction_threshold = float(
            sparse_cfg.get(
                "sparse_target_fraction_threshold",
                evaluation.get("sparse_target_fraction_threshold", 0.05),
            )
        )
        min_test_nonzero_count = int(
            sparse_cfg.get("min_test_nonzero_count", evaluation.get("min_test_nonzero_count", 10))
        )
        include_targets = sparse_cfg.get("targets")
        if isinstance(include_targets, str):
            include_targets = [item.strip() for item in include_targets.split(",") if item.strip()]
        sparse_targets = _sparse_split_targets(
            frame,
            list(target_columns or []),
            nonzero_threshold=nonzero_threshold,
            sparse_fraction_threshold=sparse_fraction_threshold,
            include_targets=include_targets,
        )
        sparse_report.update(
            {
                "nonzero_threshold": nonzero_threshold,
                "sparse_target_fraction_threshold": sparse_fraction_threshold,
                "min_test_nonzero_count": min_test_nonzero_count,
                "target_count": len(sparse_targets),
                "candidate_splits": int(sparse_cfg.get("candidate_splits", 100)),
            }
        )

    if sparse_enabled and sparse_targets:
        best: tuple[tuple[float, ...], np.ndarray, np.ndarray, int, list[dict[str, Any]]] | None = None
        candidate_splits = int(sparse_cfg.get("candidate_splits", 100))
        min_test_nonzero_count = int(
            sparse_cfg.get("min_test_nonzero_count", evaluation.get("min_test_nonzero_count", 10))
        )
        nonzero_threshold = float(sparse_cfg.get("nonzero_threshold", evaluation.get("nonzero_threshold", 1.0e-12)))
        desired_test_count = int(round(test_size * len(frame)))
        for offset in range(max(1, candidate_splits)):
            candidate_state = random_state + offset
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=candidate_state)
            candidate_train, candidate_test = next(splitter.split(indices, groups=groups))
            score, target_rows = _score_sparse_split(
                frame,
                np.asarray(candidate_train),
                np.asarray(candidate_test),
                sparse_targets,
                nonzero_threshold=nonzero_threshold,
                min_test_nonzero_count=min_test_nonzero_count,
                desired_test_count=desired_test_count,
            )
            if best is None or score > best[0]:
                best = (score, np.asarray(candidate_train), np.asarray(candidate_test), candidate_state, target_rows)
        assert best is not None
        _, train_idx, test_idx, selected_random_state, target_rows = best
        sparse_report["selected_random_state"] = selected_random_state
        sparse_report["targets"] = target_rows
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(indices, groups=groups))
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    overlap = sorted(train_groups & test_groups)
    return np.asarray(train_idx), np.asarray(test_idx), {
        "strategy": strategy,
        "test_size": test_size,
        "random_state": random_state,
        "group_column": group_column,
        "group_regex": split.get("group_regex"),
        "selected_random_state": selected_random_state,
        "train_group_count": len(train_groups),
        "test_group_count": len(test_groups),
        "group_overlap_count": len(overlap),
        "group_overlap_preview": overlap[:10],
        "sparse_target_support": sparse_report,
    }


def _make_estimator(config: dict[str, Any]) -> Any:
    model = config.get("model", {}) or {}
    kind = str(model.get("kind", "ExtraTreesRegressor"))
    kwargs = {
        "n_estimators": int(model.get("n_estimators", 300)),
        "random_state": int(model.get("random_state", 42)),
        "n_jobs": int(model.get("n_jobs", -1)),
        "min_samples_leaf": int(model.get("min_samples_leaf", 1)),
    }
    if model.get("max_features") is not None:
        kwargs["max_features"] = model.get("max_features")
    if kind == "ExtraTreesRegressor":
        return ExtraTreesRegressor(**kwargs)
    if kind == "RandomForestRegressor":
        return RandomForestRegressor(**kwargs)
    if kind == "MultiOutputExtraTreesRegressor":
        return MultiOutputRegressor(ExtraTreesRegressor(**kwargs), n_jobs=int(model.get("multioutput_n_jobs", 1)))
    raise ValueError(f"Unsupported surrogate model kind '{kind}'.")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def _classification_metrics(nonzero_true: np.ndarray, nonzero_pred: np.ndarray) -> dict[str, Any]:
    tp = int((nonzero_true & nonzero_pred).sum())
    tn = int((~nonzero_true & ~nonzero_pred).sum())
    fp = int((~nonzero_true & nonzero_pred).sum())
    fn = int((nonzero_true & ~nonzero_pred).sum())
    precision = None if (tp + fp) == 0 else tp / (tp + fp)
    recall = None if (tp + fn) == 0 else tp / (tp + fn)
    f1 = None if precision is None or recall is None or (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return {
        "nonzero_true_count": int(nonzero_true.sum()),
        "nonzero_pred_count": int(nonzero_pred.sum()),
        "zero_true_count": int((~nonzero_true).sum()),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "precision_nonzero": precision,
        "recall_nonzero": recall,
        "f1_nonzero": f1,
    }


def _target_metrics(
    *,
    y_full: pd.DataFrame,
    y_train: pd.DataFrame,
    y_true: pd.DataFrame,
    y_pred: np.ndarray,
    targets: list[str],
    nonzero_threshold: float,
    sparse_target_fraction_threshold: float,
    min_test_nonzero_count: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, target in enumerate(targets):
        full = y_full[target].to_numpy(dtype=float)
        train = y_train[target].to_numpy(dtype=float)
        true = y_true[target].to_numpy(dtype=float)
        pred = y_pred[:, index]
        baseline = np.full_like(true, float(np.mean(true)))
        full_nonzero = np.abs(full) > nonzero_threshold
        train_nonzero = np.abs(train) > nonzero_threshold
        test_nonzero_true = np.abs(true) > nonzero_threshold
        full_nonzero_count = int(full_nonzero.sum())
        train_nonzero_count = int(train_nonzero.sum())
        test_nonzero_count = int(test_nonzero_true.sum())
        full_nonzero_fraction = 0.0 if len(full) == 0 else float(full_nonzero_count / len(full))
        train_nonzero_fraction = 0.0 if len(train) == 0 else float(train_nonzero_count / len(train))
        test_nonzero_fraction = 0.0 if len(true) == 0 else float(test_nonzero_count / len(true))
        test_nonzero_coverage_fraction = (
            None if full_nonzero_count == 0 else float(test_nonzero_count / full_nonzero_count)
        )
        sparse_target = 0 < full_nonzero_fraction < sparse_target_fraction_threshold
        test_nonzero_missing = full_nonzero_count > 0 and test_nonzero_count == 0
        test_nonzero_too_low = full_nonzero_count > 0 and test_nonzero_count < min_test_nonzero_count
        warnings: list[str] = []
        if full_nonzero_count == 0:
            reliability = "all_zero_full_table"
            warnings.append("target is all zero in the full model table")
        elif test_nonzero_missing:
            reliability = "unstable_no_test_nonzero"
            warnings.append("test split contains no nonzero examples")
        elif test_nonzero_too_low:
            reliability = "unstable_low_test_nonzero"
            warnings.append(
                f"test split contains {test_nonzero_count} nonzero examples, below minimum {min_test_nonzero_count}"
            )
        elif sparse_target:
            reliability = "sparse_full_distribution"
            warnings.append(
                f"full-table nonzero fraction {full_nonzero_fraction:.4g} is below sparse threshold "
                f"{sparse_target_fraction_threshold:.4g}"
            )
        else:
            reliability = "ok"
        row = {
            "target": target,
            "n_total": int(len(full)),
            "n_train": int(len(train)),
            "n_test": int(len(true)),
            "r2": float(r2_score(true, pred)) if len(true) > 1 else None,
            "mae": float(mean_absolute_error(true, pred)),
            "rmse": _rmse(true, pred),
            "baseline_mean_mae": float(mean_absolute_error(true, baseline)),
            "baseline_mean_rmse": _rmse(true, baseline),
            "true_min": float(np.min(true)),
            "true_mean": float(np.mean(true)),
            "true_median": float(np.median(true)),
            "true_max": float(np.max(true)),
            "pred_min": float(np.min(pred)),
            "pred_mean": float(np.mean(pred)),
            "pred_median": float(np.median(pred)),
            "pred_max": float(np.max(pred)),
            "full_nonzero_count": full_nonzero_count,
            "full_nonzero_fraction": full_nonzero_fraction,
            "train_nonzero_count": train_nonzero_count,
            "train_nonzero_fraction": train_nonzero_fraction,
            "test_nonzero_fraction": test_nonzero_fraction,
            "test_nonzero_coverage_fraction": test_nonzero_coverage_fraction,
            "sparse_target": sparse_target,
            "test_nonzero_missing": test_nonzero_missing,
            "test_nonzero_too_low": test_nonzero_too_low,
            "evaluation_warning": bool(warnings),
            "evaluation_reliability": reliability,
            "diagnostic_warning": "; ".join(warnings),
        }
        row.update(_classification_metrics(test_nonzero_true, np.abs(pred) > nonzero_threshold))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("r2", ascending=False, na_position="last")


def _feature_importance(estimator: Any, inputs: list[str], targets: list[str]) -> pd.DataFrame:
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame(columns=["target", "input", "importance"])
    arr = np.asarray(importances)
    if arr.ndim == 1:
        rows = [{"target": "__multioutput_average__", "input": input_col, "importance": float(value)} for input_col, value in zip(inputs, arr)]
    else:
        rows = []
        for target, target_importance in zip(targets, arr):
            for input_col, value in zip(inputs, target_importance):
                rows.append({"target": target, "input": input_col, "importance": float(value)})
    return pd.DataFrame(rows).sort_values(["target", "importance"], ascending=[True, False])


def _permutation_importance_frame(
    *,
    estimator: Any,
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
    inputs: list[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    perm = ((config.get("evaluation") or {}).get("permutation_importance") or {})
    if not perm.get("enabled", True):
        return pd.DataFrame(columns=["input", "importance_mean", "importance_std"])
    max_rows = perm.get("max_test_rows")
    if max_rows is not None and len(x_test) > int(max_rows):
        sample = x_test.sample(n=int(max_rows), random_state=int(perm.get("random_state", 42))).index
        x_eval = x_test.loc[sample]
        y_eval = y_test.loc[sample]
    else:
        x_eval = x_test
        y_eval = y_test
    result = permutation_importance(
        estimator,
        x_eval,
        y_eval,
        n_repeats=int(perm.get("n_repeats", 5)),
        random_state=int(perm.get("random_state", 42)),
        n_jobs=int(perm.get("n_jobs", 1)),
        scoring="r2",
    )
    return pd.DataFrame(
        {
            "input": inputs,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)


def _write_plots(out_dir: Path, metrics: pd.DataFrame, predictions: pd.DataFrame, targets: list[str], warnings: list[str]) -> list[str]:
    files: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        warnings.append(f"matplotlib unavailable; skipped surrogate plots: {type(exc).__name__}: {exc}")
        return files

    plt.figure(figsize=(8, 4.8))
    ordered = metrics.sort_values("r2", ascending=True)
    plt.barh(ordered["target"], ordered["r2"])
    plt.axvline(0.0, color="black", linewidth=0.8)
    plt.xlabel("test R2")
    plt.title("Baseline Surrogate Target R2")
    plt.tight_layout()
    path = out_dir / "target_r2.png"
    plt.savefig(path, dpi=160)
    plt.close()
    files.append(path.name)

    key_targets = targets[: min(6, len(targets))]
    fig, axes = plt.subplots(len(key_targets), 1, figsize=(7, max(3, 2.5 * len(key_targets))))
    if len(key_targets) == 1:
        axes = [axes]
    for axis, target in zip(axes, key_targets):
        axis.scatter(predictions[f"true__{target}"], predictions[f"pred__{target}"], s=5, alpha=0.35)
        vals = pd.concat([predictions[f"true__{target}"], predictions[f"pred__{target}"]])
        low = float(vals.min())
        high = float(vals.max())
        axis.plot([low, high], [low, high], color="black", linewidth=0.8)
        axis.set_title(target)
        axis.set_xlabel("true")
        axis.set_ylabel("pred")
    plt.tight_layout()
    path = out_dir / "predicted_vs_true.png"
    plt.savefig(path, dpi=160)
    plt.close()
    files.append(path.name)
    return files


def train_baseline_surrogate(
    *,
    model_table: str | Path,
    config: str | Path | None = None,
    out: str | Path,
    save_model: bool = True,
) -> Path:
    config_path_value = Path(config) if config else config_path("surrogate_baseline.yaml")
    config_data = load_yaml(config_path_value)
    frame = _read_table(model_table)
    schema = _load_schema(model_table)
    inputs = _columns_from_schema_or_prefix(schema, frame, "inputs", "x__")
    targets = _columns_from_schema_or_prefix(schema, frame, "targets", "y__")
    if not inputs or not targets:
        raise ValueError("Model table must contain input columns and target columns.")
    data = frame.dropna(subset=inputs + targets).copy()
    if len(data) != len(frame):
        dropped = len(frame) - len(data)
    else:
        dropped = 0
    train_idx, test_idx, split_report = _split_indices(data, config_data, target_columns=targets)
    x_train = data.iloc[train_idx][inputs]
    x_test = data.iloc[test_idx][inputs]
    y_train = data.iloc[train_idx][targets]
    y_test = data.iloc[test_idx][targets]

    estimator = _make_estimator(config_data)
    estimator.fit(x_train, y_train)
    pred = estimator.predict(x_test)
    if pred.ndim == 1:
        pred = pred[:, None]

    evaluation = config_data.get("evaluation", {}) or {}
    nonzero_threshold = float(evaluation.get("nonzero_threshold", 1.0e-12))
    sparse_target_fraction_threshold = float(evaluation.get("sparse_target_fraction_threshold", 0.05))
    min_test_nonzero_count = int(evaluation.get("min_test_nonzero_count", 10))
    metrics = _target_metrics(
        y_full=data[targets],
        y_train=y_train,
        y_true=y_test,
        y_pred=pred,
        targets=targets,
        nonzero_threshold=nonzero_threshold,
        sparse_target_fraction_threshold=sparse_target_fraction_threshold,
        min_test_nonzero_count=min_test_nonzero_count,
    )
    feature_importance = _feature_importance(estimator, inputs, targets)
    permutation = _permutation_importance_frame(estimator=estimator, x_test=x_test, y_test=y_test, inputs=inputs, config=config_data)

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions = data.iloc[test_idx][[column for column in data.columns if column.startswith("meta__")]].copy()
    for index, target in enumerate(targets):
        predictions[f"true__{target}"] = y_test[target].to_numpy()
        predictions[f"pred__{target}"] = pred[:, index]
        predictions[f"residual__{target}"] = y_test[target].to_numpy() - pred[:, index]

    metrics.to_csv(out_dir / "target_metrics.csv", index=False)
    feature_importance.to_csv(out_dir / "feature_importance.csv", index=False)
    permutation.to_csv(out_dir / "permutation_importance.csv", index=False)
    if evaluation.get("save_predictions", True):
        predictions.to_csv(out_dir / "test_predictions.csv", index=False)

    warnings: list[str] = []
    reaction_provenance = reaction_provenance_from_frame(data, prefix="meta__")
    if reaction_provenance["reaction_model_signature_count"] == 0:
        warnings.append("Model table has no reaction_model_signature metadata; surrogate compatibility checks will be limited.")
    if reaction_provenance["reaction_model_signature_count"] > 1:
        warnings.append(
            "Model table contains multiple reaction_model_signatures; prefer training one surrogate per reaction parameter set."
        )
    plots_enabled = bool((evaluation.get("plots") or {}).get("enabled", True))
    plot_files = _write_plots(out_dir, metrics, predictions, targets, warnings) if plots_enabled else []
    if not plots_enabled:
        warnings.append("Surrogate plots disabled by evaluation.plots.enabled=false.")
    model_path = out_dir / "model.joblib"
    if save_model:
        joblib.dump(
            {
                "estimator": estimator,
                "inputs": inputs,
                "targets": targets,
                "config": config_data,
                "schema": schema,
                "reaction_provenance": reaction_provenance,
            },
            model_path,
        )

    summary = {
        "model_table": str(model_table),
        "config": str(config_path_value),
        "row_count_input": int(len(frame)),
        "row_count_after_dropna": int(len(data)),
        "rows_dropped_missing": int(dropped),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "input_columns": inputs,
        "target_columns": targets,
        "reaction_provenance": reaction_provenance,
        "split": split_report,
        "best_targets_by_r2": metrics.sort_values("r2", ascending=False).head(5).to_dict(orient="records"),
        "worst_targets_by_r2": metrics.sort_values("r2", ascending=True).head(5).to_dict(orient="records"),
        "model_path": str(model_path) if save_model else None,
        "plot_files": plot_files,
        "warnings": warnings,
    }
    write_json(out_dir / "surrogate_summary.json", summary)
    write_json(out_dir / "surrogate_model_manifest.json", summary)
    write_json(out_dir / "config_used.json", config_data)
    return out_dir
