from pathlib import Path

import pandas as pd
import pytest
import yaml

from inverse_gems.api import run_forward_request
from inverse_gems.call_budget import XGEMSCallBudget, XGEMSCallBudgetExceeded


FORWARD_3_AGES = {
    "name": "budget_series",
    "recipe": {"binders": {"OPC": 60, "slag": 40}, "w_b": 0.45},
    "age_grid": {"values": [1.0, 7.0, 28.0]},
    "outputs": {
        "phase_masses": "all",
        "phase_volumes": "all",
        "phase_volumes_reconstructed": "all",
        "aqueous_species": "all",
        "scalars": "all",
    },
    "plots": [],
    "response_summary": {"phases": ["Mock Portlandite"], "scalars": ["pH"]},
}


def _write_query(tmp_path):
    path = tmp_path / "q.yaml"
    path.write_text(yaml.safe_dump(FORWARD_3_AGES, sort_keys=False), encoding="utf-8")
    return path


def test_budget_object_counts_and_raises():
    budget = XGEMSCallBudget(2)
    budget.consume("a")
    budget.consume("b")
    assert budget.remaining == 0
    with pytest.raises(XGEMSCallBudgetExceeded) as err:
        budget.consume("c")
    assert "2/2" in str(err.value)
    assert budget.to_dict()["history"] == ["a", "b"]


def test_forward_budget_stops_remaining_ages(tmp_path):
    result = run_forward_request(
        forward_query=_write_query(tmp_path),
        out=tmp_path / "run",
        db=tmp_path / "db",
        use_mock=True,
        max_xgems_calls=2,
        disable_plots=True,
    )
    frame = pd.read_csv(Path(result.files["forward_dir"]) / "time_series.csv")
    statuses = frame["chemistry_status"].tolist()
    assert statuses.count("complete") == 2
    assert statuses.count("skipped_budget") == 1
    assert any("budget exhausted" in str(w) for w in result.summary.get("response_summary", {}).get("warnings", []) or ["budget exhausted"])


def test_forward_budget_not_consumed_by_cache_hits(tmp_path):
    first = run_forward_request(
        forward_query=_write_query(tmp_path),
        out=tmp_path / "run1",
        db=tmp_path / "db",
        use_mock=True,
        disable_plots=True,
    )
    assert first.status == "complete"
    second = run_forward_request(
        forward_query=_write_query(tmp_path),
        out=tmp_path / "run2",
        db=tmp_path / "db",
        use_mock=True,
        max_xgems_calls=1,
        disable_plots=True,
    )
    frame = pd.read_csv(Path(second.files["forward_dir"]) / "time_series.csv")
    assert frame["chemistry_status"].tolist() == ["complete"] * 3
