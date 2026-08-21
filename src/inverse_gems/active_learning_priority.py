from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .utils import write_json


DEFAULT_RELIABILITY_WEIGHTS = {
    "unstable_no_test_nonzero": 120.0,
    "unstable_low_test_nonzero": 100.0,
    "sparse_full_distribution": 75.0,
    "all_zero_full_table": 25.0,
    "ok": 0.0,
    "": 0.0,
}


def _read_diagnostics(path: str | Path) -> pd.DataFrame:
    diagnostics_path = Path(path)
    if diagnostics_path.is_dir():
        diagnostics_path = diagnostics_path / "model_registry_diagnostics.csv"
    if not diagnostics_path.exists():
        raise ValueError(f"Diagnostics file does not exist: {diagnostics_path}")
    return pd.read_csv(diagnostics_path)


def _target_kind_from_column(column: Any) -> str:
    text = str(column or "")
    if text.startswith("y__amount_"):
        return "amount"
    if text.startswith("y__volume_"):
        return "volume"
    return "scalar"


def _allowed_by_kind(column: Any, kinds: set[str]) -> bool:
    if not kinds:
        return True
    expanded = set(kinds)
    if "phase" in expanded:
        expanded.update({"amount", "volume"})
    return _target_kind_from_column(column) in expanded


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(result):
        return default
    return result


def _as_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "1.0", "yes", "y"}


def _action_for_reliability(reliability: str, status: str) -> str:
    reliability = str(reliability or "")
    if reliability == "unstable_no_test_nonzero":
        return "add samples that are likely to make this phase nonzero; current test split has no support"
    if reliability == "unstable_low_test_nonzero":
        return "add validation samples inside the known nonzero region before trusting R2"
    if reliability == "sparse_full_distribution":
        return "expand targeted sampling around existing nonzero chemistry regions"
    if reliability == "all_zero_full_table":
        return "explore new material/age regions; the current surrogate has no nonzero signal"
    if str(status) == "usable_with_caution":
        return "add diverse samples to improve surrogate fit for this target"
    return "monitor"


def rank_active_learning_targets(
    diagnostics: str | Path,
    *,
    statuses: list[str] | tuple[str, ...] | None = None,
    target_kinds: list[str] | tuple[str, ...] | None = None,
    exclude_targets: list[str] | tuple[str, ...] | None = None,
    min_r2: float = 0.70,
    max_targets: int | None = None,
    reliability_weights: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rank diagnostic targets that deserve additional xGEMS/chemistry acquisition."""
    frame = _read_diagnostics(diagnostics)
    if frame.empty:
        return pd.DataFrame(), {"input_rows": 0, "ranked_rows": 0, "warnings": ["diagnostics table is empty"]}
    if "target_column" not in frame.columns:
        raise ValueError("Diagnostics table must contain a target_column column.")

    work = frame.copy()
    requested_statuses = tuple(statuses or ("usable_with_caution", "not_recommended"))
    wanted_statuses = {str(status) for status in requested_statuses}
    if "status" in work.columns:
        work = work[work["status"].astype(str).isin(wanted_statuses)]

    kinds = {str(kind).strip().lower() for kind in (target_kinds or ("amount",)) if str(kind).strip()}
    if kinds:
        work = work[work["target_column"].apply(lambda value: _allowed_by_kind(value, kinds))]

    excluded = {str(item).strip().lower() for item in (exclude_targets or []) if str(item).strip()}
    if excluded:
        labels = work.get("target_label", pd.Series("", index=work.index)).fillna("").astype(str).str.lower()
        columns = work["target_column"].fillna("").astype(str).str.lower()
        work = work[~labels.isin(excluded) & ~columns.isin(excluded)]

    if work.empty:
        return pd.DataFrame(), {
            "input_rows": int(len(frame)),
            "ranked_rows": 0,
            "statuses": list(requested_statuses),
            "target_kinds": sorted(kinds),
            "warnings": ["no diagnostics rows matched active-learning filters"],
        }

    rows: list[dict[str, Any]] = []
    weights = {**DEFAULT_RELIABILITY_WEIGHTS, **(reliability_weights or {})}
    for _, row in work.iterrows():
        target = str(row.get("target_column", ""))
        reliability = str(row.get("evaluation_reliability", "") or "")
        status = str(row.get("status", "") or "")
        r2 = _as_float(row.get("r2"), default=0.0)
        full_nonzero_count = _as_int(row.get("full_nonzero_count"))
        test_nonzero_count = _as_int(row.get("test_nonzero_count"))
        if test_nonzero_count is None:
            test_nonzero_count = _as_int(row.get("nonzero_true_count"))
        full_nonzero_fraction = _as_float(row.get("full_nonzero_fraction", row.get("nonzero_fraction")), default=0.0)
        r2_gap = max(0.0, float(min_r2) - r2)
        scarcity = max(0.0, 1.0 - full_nonzero_fraction) * 10.0 if full_nonzero_count else 0.0
        nonzero_support_penalty = 0.0 if (test_nonzero_count or 0) >= 10 else (10 - float(test_nonzero_count or 0))
        score = float(weights.get(reliability, 0.0)) + r2_gap * 25.0 + scarcity + nonzero_support_penalty
        if _as_bool(row.get("pH_water_adjusted")):
            score -= 50.0
        rows.append(
            {
                "target_column": target,
                "target_label": row.get("target_label", target),
                "target_kind": _target_kind_from_column(target),
                "status": status,
                "evaluation_reliability": reliability,
                "active_learning_priority_score": score,
                "r2": None if pd.isna(row.get("r2")) else row.get("r2"),
                "full_nonzero_count": full_nonzero_count,
                "full_nonzero_fraction": full_nonzero_fraction,
                "test_nonzero_count": test_nonzero_count,
                "reasons": row.get("reasons", ""),
                "recommended_action": _action_for_reliability(reliability, status),
            }
        )

    ranked = pd.DataFrame(rows)
    ranked = ranked.sort_values(
        ["active_learning_priority_score", "r2", "target_column"],
        ascending=[False, True, True],
        na_position="last",
    )
    ranked = ranked.drop_duplicates(subset=["target_column"], keep="first").reset_index(drop=True)
    if max_targets is not None:
        ranked = ranked.head(int(max_targets)).copy()
    summary = {
        "input_rows": int(len(frame)),
        "filtered_rows": int(len(work)),
        "ranked_rows": int(len(ranked)),
        "statuses": list(requested_statuses),
        "target_kinds": sorted(kinds),
        "max_targets": max_targets,
        "min_r2": float(min_r2),
        "top_targets": ranked.head(10).to_dict(orient="records"),
        "warnings": [],
    }
    return ranked, summary


def write_active_learning_target_priorities(
    *,
    diagnostics: str | Path,
    out: str | Path,
    statuses: list[str] | tuple[str, ...] | None = None,
    target_kinds: list[str] | tuple[str, ...] | None = None,
    exclude_targets: list[str] | tuple[str, ...] | None = None,
    min_r2: float = 0.70,
    max_targets: int | None = None,
) -> Path:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ranked, summary = rank_active_learning_targets(
        diagnostics,
        statuses=statuses,
        target_kinds=target_kinds,
        exclude_targets=exclude_targets,
        min_r2=min_r2,
        max_targets=max_targets,
    )
    ranked.to_csv(out_dir / "active_learning_target_priorities.csv", index=False)
    write_json(out_dir / "active_learning_target_priorities_summary.json", summary)
    lines = ["# Active Learning Target Priorities", ""]
    lines.append(f"- diagnostics: `{diagnostics}`")
    lines.append(f"- ranked targets: `{summary['ranked_rows']}`")
    lines.append("")
    if ranked.empty:
        lines.append("No targets matched the requested filters.")
    else:
        columns = [
            "target_column",
            "active_learning_priority_score",
            "evaluation_reliability",
            "r2",
            "full_nonzero_count",
            "test_nonzero_count",
            "recommended_action",
        ]
        use_columns = [column for column in columns if column in ranked.columns]
        lines.append("| " + " | ".join(use_columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(use_columns)) + " |")
        for row in ranked[use_columns].to_dict(orient="records"):
            lines.append("| " + " | ".join(str(row.get(column, "")).replace("|", "\\|") for column in use_columns) + " |")
    (out_dir / "active_learning_target_priorities.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return out_dir
