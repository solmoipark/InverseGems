import json

import pandas as pd

from inverse_gems.acceptance import default_acceptance_cases, run_acceptance_suite
from inverse_gems.cli import main


def test_default_acceptance_cases_include_single_and_forward_query():
    cases = default_acceptance_cases()
    case_types = {case["case_type"] for case in cases}
    assert "single_recipe" in case_types
    assert "forward_query" in case_types
    assert "single_opc_28" in {case["case_id"] for case in cases}


def test_acceptance_mock_writes_reports_and_preserves_phase_names(tmp_path):
    report = run_acceptance_suite(
        out=tmp_path / "acceptance",
        use_mock=True,
        case_ids=["single_opc_28", "forward_query_single_age_opc_slag_28"],
    )

    out = tmp_path / "acceptance"
    frame = pd.read_csv(out / "acceptance_report.csv")
    payload = json.loads((out / "acceptance_report.json").read_text(encoding="utf-8"))

    assert report["summary"]["ok"] is True
    assert payload["summary"]["passed_count"] == 2
    assert len(frame) == 2
    assert set(frame["status"]) == {"passed"}
    assert "Mock C-S-H raw phase" in frame.iloc[0]["phase_masses_top_json"]
    assert (out / "acceptance_report.md").exists()
    assert (out / "environment_report.json").exists()
    assert (out / "cases" / "single_opc_28" / "case_detail.json").exists()


def test_acceptance_mock_cli_returns_zero(tmp_path):
    code = main(
        [
            "acceptance-mock",
            "--out",
            str(tmp_path / "acceptance_cli"),
            "--case",
            "single_opc_28",
        ]
    )

    assert code == 0
    assert (tmp_path / "acceptance_cli" / "acceptance_report.csv").exists()


def test_acceptance_rejects_unknown_case(tmp_path):
    try:
        run_acceptance_suite(out=tmp_path / "acceptance", use_mock=True, case_ids=["missing_case"])
    except ValueError as exc:
        assert "Unknown acceptance case" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown acceptance case.")
