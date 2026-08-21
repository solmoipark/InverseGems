from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import write_json


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _status_from_summary(summary: dict[str, Any]) -> str:
    return "complete" if summary else "missing"


def _pH_policy_notes(selection_summary: dict[str, Any]) -> list[str]:
    policy = selection_summary.get("pH_uncertainty_policy") or {}
    if not policy or policy.get("mode") in {None, "ignore"}:
        return []
    notes = [
        f"pH uncertainty policy: {policy.get('mode')}",
        f"pH unreliable rows: {policy.get('unreliable_rows', 0)}",
    ]
    if _as_int(policy.get("rejected_rows")):
        notes.append(f"pH unreliable rows rejected: {policy.get('rejected_rows')}")
    if _as_int(policy.get("penalized_rows")):
        notes.append(f"pH unreliable rows penalized: {policy.get('penalized_rows')}")
    return notes


def _stage(
    name: str,
    status: str,
    *,
    artifact: Any = None,
    rows_in: Any = None,
    rows_out: Any = None,
    runs: Any = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "artifact": artifact,
        "rows_in": _as_int(rows_in),
        "rows_out": _as_int(rows_out),
        "runs": _as_int(runs),
        "notes": notes or [],
    }


def build_inverse_design_flow_summary(run_summary: dict[str, Any]) -> dict[str, Any]:
    paths = run_summary.get("paths") or {}
    search_summary = run_summary.get("candidate_search_summary") or {}
    validation_summary = run_summary.get("validation_summary") or {}
    selection_summary = run_summary.get("selection_summary") or {}
    domain_summary = run_summary.get("domain_summary") or {}
    candidate_review = run_summary.get("candidate_review") or {}
    skip_validation = bool(run_summary.get("skip_validation"))

    stages: list[dict[str, Any]] = [
        _stage("compile_query", "complete", artifact=paths.get("compiled_query")),
    ]
    if paths.get("candidate_recipes"):
        stages.append(
            _stage(
                "candidate_generation",
                "complete",
                artifact=paths.get("candidate_recipes"),
                rows_out=run_summary.get("n_candidates"),
            )
        )
    if paths.get("chemistry_candidate_table"):
        stages.append(
            _stage(
                "reactive_chemistry_table",
                "complete",
                artifact=paths.get("chemistry_candidate_table"),
                rows_out=run_summary.get("n_candidates"),
            )
        )
    stages.append(
        _stage(
            "surrogate_screening",
            _status_from_summary(search_summary),
            artifact=paths.get("surrogate_search"),
            rows_in=search_summary.get("input_rows"),
            rows_out=search_summary.get("top_k_written"),
            notes=[
                f"rows after filters: {search_summary.get('candidate_rows_after_filters')}"
                if search_summary.get("candidate_rows_after_filters") is not None
                else ""
            ],
        )
    )
    if paths.get("domain_report"):
        stages.append(
            _stage(
                "domain_check",
                _status_from_summary(domain_summary),
                artifact=paths.get("domain_report"),
                rows_out=domain_summary.get("row_count"),
                notes=[
                    f"out of domain: {domain_summary.get('out_of_domain_count')}"
                    if domain_summary.get("out_of_domain_count") is not None
                    else ""
                ],
            )
        )

    if skip_validation:
        stages.append(_stage("xgems_validation", "skipped", artifact=paths.get("validation")))
        stages.append(_stage("final_selection", "skipped", artifact=paths.get("selection")))
    else:
        stages.append(
            _stage(
                "xgems_validation",
                _status_from_summary(validation_summary),
                artifact=paths.get("validation"),
                rows_in=validation_summary.get("candidate_rows_used"),
                runs=validation_summary.get("validation_runs"),
                notes=[f"status counts: {validation_summary.get('validation_status_counts') or {}}"],
            )
        )
        stages.append(
            _stage(
                "final_selection",
                _status_from_summary(selection_summary),
                artifact=paths.get("selection"),
                rows_in=selection_summary.get("input_rows"),
                rows_out=selection_summary.get("top_k_written"),
                notes=[
                    f"rows after constraints: {selection_summary.get('candidate_rows_after_constraints')}"
                    if selection_summary.get("candidate_rows_after_constraints") is not None
                    else ""
                ]
                + _pH_policy_notes(selection_summary),
            )
        )
    stages.append(
        _stage(
            "candidate_review",
            _status_from_summary(candidate_review),
            artifact=(candidate_review.get("outputs") or {}).get("csv"),
            rows_out=candidate_review.get("row_count"),
        )
    )

    notes: list[str] = []
    if skip_validation:
        notes.append("Final candidates are surrogate-screened only; run xGEMS/GEMS validation before treating them as confirmed.")
    pH_summary = validation_summary.get("pH_reliability") or {}
    if _as_int(pH_summary.get("pH_water_unreliable_count")):
        notes.append("Validated pH includes water-adjusted or rescued rows and should be treated as conditional.")
    validation_counts = validation_summary.get("validation_status_counts") or {}
    failed_count = sum(_as_int(value) or 0 for key, value in validation_counts.items() if str(key).lower() not in {"complete", "success", "ok"})
    if failed_count:
        notes.append(f"{failed_count} validation run(s) did not complete successfully.")
    if _as_int(domain_summary.get("out_of_domain_count")):
        notes.append(f"{domain_summary.get('out_of_domain_count')} candidate(s) are outside the reference chemistry domain.")
    pH_policy = selection_summary.get("pH_uncertainty_policy") or {}
    if pH_policy.get("mode") == "exclude" and _as_int(pH_policy.get("rejected_rows")):
        notes.append(
            f"{pH_policy.get('rejected_rows')} candidate(s) with pH-water-unreliable validation were excluded during final selection."
        )
    elif pH_policy.get("mode") == "penalize" and _as_int(pH_policy.get("penalized_rows")):
        notes.append(
            f"{pH_policy.get('penalized_rows')} candidate(s) with pH-water-unreliable validation were penalized during final ranking."
        )

    return {
        "final_stage": run_summary.get("final_stage"),
        "skip_validation": skip_validation,
        "final_csv": run_summary.get("final_csv"),
        "final_json": run_summary.get("final_json"),
        "stages": [
            {**stage, "notes": [note for note in stage["notes"] if note]}
            for stage in stages
        ],
        "interpretation_notes": notes,
    }


def _format(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = ["# Inverse Design Flow Summary", ""]
    lines.append(f"- final stage: `{summary.get('final_stage')}`")
    lines.append(f"- validation skipped: `{summary.get('skip_validation')}`")
    lines.append(f"- final csv: `{summary.get('final_csv')}`")
    lines.append("")
    lines.append("| Stage | Status | Rows in | Rows out | Runs | Artifact | Notes |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for stage in summary.get("stages") or []:
        lines.append(
            "| "
            + " | ".join(
                [
                    _format(stage.get("name")),
                    _format(stage.get("status")),
                    _format(stage.get("rows_in")),
                    _format(stage.get("rows_out")),
                    _format(stage.get("runs")),
                    _format(stage.get("artifact")),
                    _format("; ".join(stage.get("notes") or [])),
                ]
            )
            + " |"
        )
    notes = summary.get("interpretation_notes") or []
    if notes:
        lines.extend(["", "## Interpretation Notes", ""])
        for note in notes:
            lines.append(f"- {note}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_inverse_design_flow_summary(*, run_summary: dict[str, Any], out: str | Path) -> dict[str, Any]:
    out_dir = Path(out)
    flow = build_inverse_design_flow_summary(run_summary)
    json_path = out_dir / "inverse_design_flow_summary.json"
    md_path = out_dir / "inverse_design_flow_summary.md"
    write_json(json_path, flow)
    _write_markdown(flow, md_path)
    return {**flow, "outputs": {"json": str(json_path), "markdown": str(md_path)}}
