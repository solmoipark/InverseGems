import json

from inverse_gems.forward_answer import build_forward_answer, write_forward_answer


def _response_summary_payload():
    return {
        "mode": "time_series",
        "row_count": 2,
        "completed_count": 2,
        "failed_count": 0,
        "requested": {"phases": ["CNASH"], "phase_groups": ["C-A-S-H"], "scalars": ["pH", "porosity"]},
        "missing": {"phases": [], "phase_groups": [], "scalars": []},
        "selected": {
            "rows": [
                {
                    "age_days": 1.0,
                    "chemistry_status": "complete",
                    "solver_status": "ok",
                    "phases": {"CNASH": {"mass": 0.01, "volume": 1.0e-6}},
                    "phase_groups": {"C-A-S-H": {"mass": 0.01, "volume": 1.0e-6}},
                    "scalars": {"pH": 12.6, "porosity": 0.42},
                },
                {
                    "age_days": 28.0,
                    "chemistry_status": "complete",
                    "solver_status": "ok",
                    "phases": {"CNASH": {"mass": 0.05, "volume": 5.0e-6}},
                    "phase_groups": {"C-A-S-H": {"mass": 0.05, "volume": 5.0e-6}},
                    "scalars": {"pH": 12.3, "porosity": 0.36},
                },
            ]
        },
        "failed_ages": [],
        "uncertainty": {
            "flag_counts": {"pH_water_uncertain": 1},
            "preflight_available_count": 2,
            "rows": [
                {
                    "age_days": 1.0,
                    "uncertainty_flags": "",
                    "pH_water_reliable": True,
                    "xgems_water_delta_g": 0.0,
                    "preflight_dir": "preflight/a",
                },
                {
                    "age_days": 28.0,
                    "uncertainty_flags": "pH_water_uncertain",
                    "pH_water_reliable": False,
                    "xgems_water_delta_g": -5.0,
                    "preflight_dir": "preflight/b",
                },
            ],
        },
        "files": {"response_summary_csv": "response_summary.csv"},
    }


def test_build_forward_answer_creates_preview_and_numeric_highlights():
    answer = build_forward_answer(_response_summary_payload(), summary_path="response_summary.json")

    assert answer["headline"] == "Forward time_series completed successfully for 2 row(s)."
    assert answer["rows_preview"][0]["CNASH.mass"] == 0.01
    assert answer["rows_preview"][0]["phase_group.C-A-S-H.mass"] == 0.01
    assert answer["rows_preview"][0]["scalar.pH"] == 12.6
    names = {item["output"] for item in answer["series_highlights"]}
    assert "CNASH.mass" in names
    assert "phase_group.C-A-S-H.mass" in names
    assert "scalar.porosity" in names
    assert answer["uncertainty"]["flag_counts"]["pH_water_uncertain"] == 1
    assert answer["raw_phase_policy"].startswith("Raw xGEMS/GEMS names")


def test_write_forward_answer_writes_json_and_markdown(tmp_path):
    summary = tmp_path / "response_summary.json"
    summary.write_text(json.dumps(_response_summary_payload()), encoding="utf-8")

    out = write_forward_answer(summary=summary, out=tmp_path / "answer", table_limit=1)

    payload = json.loads((out / "answer.json").read_text(encoding="utf-8"))
    markdown = (out / "answer.md").read_text(encoding="utf-8")
    assert payload["rows_preview"][0]["CNASH.volume"] == 1.0e-6
    assert len(payload["rows_preview"]) == 1
    assert "CNASH.mass" in markdown
    assert "Selected phase groups" in markdown
    assert "Uncertainty And Preflight" in markdown
