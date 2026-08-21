import json

import pandas as pd

from inverse_gems.feature_diagnostics import run_feature_diagnostics


def test_feature_diagnostics_writes_core_reports(tmp_path):
    feature_table = tmp_path / "features.csv"
    pd.DataFrame(
        {
            "recipe_id": ["a", "b", "c", "d"],
            "chemistry_status": ["complete", "complete", "failed", "complete"],
            "OPC": [100.0, 80.0, 60.0, 40.0],
            "slag": [0.0, 20.0, 40.0, 60.0],
            "w_b": [0.4, 0.4, 0.5, 0.6],
            "age_days": [1.0, 7.0, 28.0, 90.0],
            "porosity": [0.5, 0.45, 0.4, 0.35],
            "scalar__pH": [12.5, 12.4, 12.3, 12.2],
            "phase_amount__Calcite": [0.0, 1.0, 2.0, 3.0],
            "phase_volume__Calcite": [0.0, 0.5, 0.0, 1.5],
            "phase_amount_group__Calcite": [0.0, 1.0, 2.0, 3.0],
            "phase_volume_group__Calcite": [0.0, 0.5, 0.0, 1.5],
        }
    ).to_csv(feature_table, index=False)

    out = run_feature_diagnostics(feature_table=feature_table, out=tmp_path / "diagnostics", plot=False)

    assert (out / "feature_diagnostics_summary.json").exists()
    assert (out / "phase_amount_volume_summary.csv").exists()
    assert (out / "correlation_with_targets.csv").exists()

    summary = json.loads((out / "feature_diagnostics_summary.json").read_text(encoding="utf-8"))
    assert summary["row_count"] == 4
    assert summary["selected_phase_amount_volume_mismatch_count"] == 1

    phase = pd.read_csv(out / "phase_amount_volume_summary.csv")
    calcite = phase[phase["name"] == "Calcite"].iloc[0]
    assert int(calcite["amount_positive_volume_zero_count"]) == 1
