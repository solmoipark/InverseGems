from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .candidate_validation import candidate_recipe_text
from .utils import write_json


BINDER_COLUMNS: tuple[tuple[str, str], ...] = (
    ("x__OPC", "OPC"),
    ("x__slag", "slag"),
    ("x__fly_ash", "fly_ash"),
    ("x__metakaolin", "metakaolin"),
    ("x__silica_fume", "silica_fume"),
    ("x__limestone", "limestone"),
    ("x__gypsum", "gypsum"),
)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _as_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_bool(value: Any) -> bool | None:
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "1.0", "yes", "y"}:
        return True
    if text in {"false", "0", "0.0", "no", "n"}:
        return False
    return None


def _value(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and not _is_missing(row.get(name)):
            return row.get(name)
    return None


def _solver_status_is_ok(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text or text in {"none", "nan"}:
        return True
    if any(token in text for token in ["fail", "error", "bad"]):
        return False
    return any(token in text for token in ["success", "ok", "converged"])


def _safe_target_label(stem: str) -> str:
    label = str(stem)
    if label.startswith("y__"):
        label = label.removeprefix("y__")
    label = label.replace("amount_", "amount_").replace("volume_", "volume_")
    label = re.sub(r"[^0-9A-Za-z_]+", "_", label).strip("_")
    return re.sub(r"_+", "_", label) or "target"


def _target_stems(columns: list[str]) -> list[str]:
    stems: list[str] = []
    prefixes = [
        "pred__",
        "validated__",
        "source_true__",
        "true__",
        "abs_diff_validated_minus_pred__",
        "diff_validated_minus_pred__",
    ]
    for column in columns:
        for prefix in prefixes:
            if column.startswith(prefix):
                stem = column.removeprefix(prefix)
                if stem.startswith("y__") and stem not in stems:
                    stems.append(stem)
    return stems


def _review_rank(row: Mapping[str, Any], index: int) -> int:
    value = _as_float(_value(row, "selection_rank", "rank", "candidate_rank"))
    return index + 1 if value is None else int(value)


def _recipe_text(row: Mapping[str, Any]) -> str | None:
    existing = _value(row, "recipe_text")
    if existing is not None:
        return str(existing)
    try:
        return candidate_recipe_text(row)
    except Exception:
        return None


def _humanize_component_name(value: str) -> str:
    label = str(value).strip("_")
    replacements = {
        "C_A_S_H": "C-A-S-H",
        "pH": "pH",
        "w_b": "w/b",
    }
    return replacements.get(label, label.replace("_", " "))


def _ranking_component_label(column: str) -> str | None:
    parts = column.split("__")
    if column.startswith(("rank_component__", "selection_rank_component__")) and len(parts) >= 4:
        direction = parts[2]
        name = "__".join(parts[3:])
        return f"{direction} {_humanize_component_name(name)}"
    if column.startswith(("score_component__", "selection_score_component__")) and len(parts) >= 3:
        direction = parts[1]
        name = "__".join(parts[2:])
        return f"{direction} {_humanize_component_name(name)}"
    return None


def _ranking_reason(row: Mapping[str, Any]) -> str:
    rank_components = [
        (column, _ranking_component_label(column))
        for column in row
        if str(column).startswith(("rank_component__", "selection_rank_component__"))
    ]
    rank_labels = [label for _, label in sorted(rank_components) if label]
    if rank_labels:
        return "ordered preferences: " + "; then ".join(rank_labels)

    score_components = [
        (column, _ranking_component_label(column))
        for column in row
        if str(column).startswith(("score_component__", "selection_score_component__"))
    ]
    score_labels = [label for _, label in sorted(score_components) if label]
    if score_labels:
        return "weighted score: " + ", ".join(score_labels)

    source_rank = _value(row, "rank", "candidate_rank")
    if source_rank is not None:
        return f"carried from source rank {source_rank}"
    return "ranking metadata unavailable"


def _validation_note(row: Mapping[str, Any], *, stage: str, flags: list[str]) -> str:
    if stage == "surrogate_search" or "surrogate_only" in flags:
        return "surrogate-screened only; not xGEMS/GEMS validated"
    status = str(_value(row, "chemistry_status", "meta__chemistry_status") or "").strip()
    notes: list[str] = []
    if status:
        notes.append(f"xGEMS/GEMS status: {status}")
    if "solver_rescued" in flags:
        notes.append("solver used water rescue; pH is conditional on adjusted water")
    if "pH_water_uncertain" in flags:
        reason = _value(row, "pH_unreliable_reason")
        if reason:
            notes.append(f"pH uncertain: {reason}")
        else:
            notes.append("pH uncertain")
    if "out_of_domain" in flags:
        distance = _value(row, "nearest_scaled_distance", "meta__nearest_scaled_distance")
        if distance is not None:
            notes.append(f"outside reference domain; nearest scaled distance {distance}")
        else:
            notes.append("outside reference domain")
    return "; ".join(notes) if notes else "validated by xGEMS/GEMS"


def _uncertainty_flags(row: Mapping[str, Any], *, stage: str, target_stems: list[str]) -> list[str]:
    flags: list[str] = []
    has_validated = any(f"validated__{stem}" in row and not _is_missing(row.get(f"validated__{stem}")) for stem in target_stems)
    chemistry_status = str(_value(row, "chemistry_status", "meta__chemistry_status") or "").lower()
    if stage == "surrogate_search" or not has_validated:
        flags.append("surrogate_only")
    elif chemistry_status in {"complete", "success", "ok"}:
        flags.append("validated")
    if chemistry_status and chemistry_status not in {"complete", "success", "ok"}:
        flags.append(f"chemistry_status:{chemistry_status}")
    solver_status = str(_value(row, "solver_status") or "").lower()
    if solver_status and not _solver_status_is_ok(solver_status):
        flags.append(f"solver_status:{solver_status}")
    solver_rescued = _as_bool(_value(row, "solver_rescued", "meta__solver_rescued"))
    if solver_rescued:
        flags.append("solver_rescued")
    if _as_bool(_value(row, "pH_water_reliable")) is False:
        flags.append("pH_water_uncertain")
    if _value(row, "pH_unreliable_reason") is not None:
        flags.append("pH_water_uncertain")
    if _as_bool(_value(row, "xgems_water_matches_recipe")) is False:
        flags.append("xgems_water_adjusted")
    if _as_bool(_value(row, "out_of_domain", "meta__out_of_domain")):
        flags.append("out_of_domain")
    if row.get("error") and not _is_missing(row.get("error")):
        flags.append("has_error")
    return list(dict.fromkeys(flags))


def _candidate_row(row: Mapping[str, Any], *, index: int, stage: str, target_stems: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "review_rank": _review_rank(row, index),
        "source_rank": _value(row, "rank", "candidate_rank"),
        "selection_rank": _value(row, "selection_rank"),
        "score": _value(row, "selection_score", "score"),
        "source_recipe_id": _value(row, "source_recipe_id", "meta__recipe_id", "recipe_id"),
        "validation_recipe_id": _value(row, "validation_recipe_id"),
        "material_system": _value(row, "material_system", "meta__material_system"),
        "stage": stage,
        "validation_status": _value(row, "chemistry_status", "meta__chemistry_status") if stage != "surrogate_search" else "surrogate_only",
        "solver_status": _value(row, "solver_status"),
        "solver_rescued": _value(row, "solver_rescued", "meta__solver_rescued"),
        "xgems_retry_count": _value(row, "xgems_retry_count", "meta__xgems_retry_count"),
        "preflight_dir": _value(row, "preflight_dir", "meta__preflight_dir"),
        "recipe_text": _recipe_text(row),
        "age_days": _value(row, "age_days", "x__age_days", "meta__age_days"),
        "w_b": _value(row, "w_b", "x__w_b", "meta__w_b"),
        "water_g": _value(row, "water_g", "x__water_g", "meta__water_g"),
        "source_chem_hash": _value(row, "source_chem_hash", "meta__chem_hash", "chem_hash"),
        "validation_chem_hash": _value(row, "validation_chem_hash"),
        "reaction_model_id": _value(row, "meta__reaction_model_id"),
        "reaction_model_signature": _value(row, "meta__reaction_model_signature"),
        "out_of_domain": _value(row, "out_of_domain", "meta__out_of_domain"),
        "nearest_scaled_distance": _value(row, "nearest_scaled_distance", "meta__nearest_scaled_distance"),
        "outside_input_range_count": _value(row, "outside_input_range_count", "meta__outside_input_range_count"),
    }
    for column, label in BINDER_COLUMNS:
        out[label] = _value(row, label, column, "meta__" + column.removeprefix("x__"))
        if out[label] is None:
            out[label] = 0.0
    flags = _uncertainty_flags(row, stage=stage, target_stems=target_stems)
    out["uncertainty_flags"] = ";".join(flags)
    out["ranking_reason"] = _ranking_reason(row)
    out["validation_note"] = _validation_note(row, stage=stage, flags=flags)

    for stem in target_stems:
        label = _safe_target_label(stem)
        pred = _value(row, f"pred__{stem}")
        validated = _value(row, f"validated__{stem}")
        source_true = _value(row, f"source_true__{stem}", f"true__{stem}")
        diff = _value(row, f"diff_validated_minus_pred__{stem}")
        abs_diff = _value(row, f"abs_diff_validated_minus_pred__{stem}")
        if abs_diff is None:
            pred_float = _as_float(pred)
            validated_float = _as_float(validated)
            if pred_float is not None and validated_float is not None:
                abs_diff = abs(validated_float - pred_float)
                diff = validated_float - pred_float
        out[f"predicted__{label}"] = pred
        out[f"validated__{label}"] = validated
        out[f"source_true__{label}"] = source_true
        out[f"delta_validated_minus_predicted__{label}"] = diff
        out[f"abs_delta_validated_minus_predicted__{label}"] = abs_diff
    return out


def _ordered_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "review_rank",
        "source_rank",
        "selection_rank",
        "score",
        "source_recipe_id",
        "validation_recipe_id",
        "material_system",
        "stage",
        "validation_status",
        "solver_status",
        "solver_rescued",
        "xgems_retry_count",
        "preflight_dir",
        "uncertainty_flags",
        "ranking_reason",
        "validation_note",
        "recipe_text",
        "OPC",
        "slag",
        "fly_ash",
        "metakaolin",
        "silica_fume",
        "limestone",
        "gypsum",
        "w_b",
        "water_g",
        "age_days",
        "source_chem_hash",
        "validation_chem_hash",
        "reaction_model_id",
        "reaction_model_signature",
        "out_of_domain",
        "nearest_scaled_distance",
        "outside_input_range_count",
    ]
    target_columns = [
        column
        for column in frame.columns
        if column.startswith(
            (
                "predicted__",
                "validated__",
                "source_true__",
                "delta_validated_minus_predicted__",
                "abs_delta_validated_minus_predicted__",
            )
        )
    ]
    return [column for column in preferred if column in frame.columns] + target_columns


def _format_markdown_value(value: Any) -> str:
    if _is_missing(value):
        return ""
    number = _as_float(value)
    if number is not None:
        return f"{number:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _write_markdown(frame: pd.DataFrame, path: Path, *, title: str, limit: int) -> None:
    columns = [
        "review_rank",
        "source_recipe_id",
        "material_system",
        "stage",
        "validation_status",
        "uncertainty_flags",
        "preflight_dir",
        "ranking_reason",
        "validation_note",
        "OPC",
        "slag",
        "fly_ash",
        "metakaolin",
        "silica_fume",
        "limestone",
        "gypsum",
        "w_b",
        "age_days",
        "recipe_text",
    ]
    target_columns = [
        column
        for column in frame.columns
        if column.startswith(("predicted__", "validated__", "abs_delta_validated_minus_predicted__"))
    ]
    columns = [column for column in columns if column in frame.columns] + target_columns
    columns = list(dict.fromkeys(columns))
    display = frame[columns].head(limit).copy() if columns else pd.DataFrame()
    lines = [f"# {title}", ""]
    if display.empty:
        lines.append("No candidates available.")
    else:
        lines.append(f"Showing {len(display)} of {len(frame)} candidate row(s).")
        lines.append("")
        lines.append("| " + " | ".join(display.columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(display.columns)) + " |")
        for row in display.to_dict(orient="records"):
            lines.append("| " + " | ".join(_format_markdown_value(row.get(column)) for column in display.columns) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_candidate_review(
    *,
    candidates: str | Path,
    out: str | Path,
    stage: str,
    title: str = "Candidate Review",
    table_limit: int = 20,
) -> dict[str, Any]:
    candidates_path = Path(candidates)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(candidates_path)
    target_stems = _target_stems(list(frame.columns))
    rows = [
        _candidate_row(row, index=index, stage=stage, target_stems=target_stems)
        for index, row in enumerate(frame.to_dict(orient="records"))
    ]
    review = pd.DataFrame(rows)
    if not review.empty:
        review = review[_ordered_columns(review)]
    csv_path = out_dir / "candidate_review.csv"
    json_path = out_dir / "candidate_review.json"
    md_path = out_dir / "candidate_review.md"
    review.to_csv(csv_path, index=False)
    json_path.write_text(review.to_json(orient="records", indent=2), encoding="utf-8")
    _write_markdown(review, md_path, title=title, limit=table_limit)
    flag_counts: dict[str, int] = {}
    if "uncertainty_flags" in review.columns:
        for value in review["uncertainty_flags"].fillna("").astype(str):
            for flag in [item for item in value.split(";") if item]:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1
    summary = {
        "source_candidates": str(candidates_path),
        "out": str(out_dir),
        "stage": stage,
        "row_count": int(len(review)),
        "target_stems": target_stems,
        "uncertainty_flag_counts": flag_counts,
        "outputs": {
            "csv": str(csv_path),
            "json": str(json_path),
            "markdown": str(md_path),
        },
    }
    write_json(out_dir / "candidate_review_summary.json", summary)
    return summary
