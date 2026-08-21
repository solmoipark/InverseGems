import json

import pandas as pd

from inverse_gems.target_region_analysis import resolve_target_column, write_target_region_analysis


def test_target_region_analysis_resolves_group_name_and_writes_reports(tmp_path):
    model_table = tmp_path / "model.csv"
    pd.DataFrame(
        {
            "meta__recipe_id": ["a", "b", "c"],
            "meta__material_system": ["OPC_slag", "OPC_slag", "OPC_fly_ash"],
            "meta__age_days": [7.0, 28.0, 90.0],
            "meta__OPC": [60.0, 55.0, 70.0],
            "meta__slag": [35.0, 40.0, 0.0],
            "meta__fly_ash": [0.0, 0.0, 25.0],
            "meta__limestone": [1.0, 0.5, 2.0],
            "meta__gypsum": [4.0, 2.0, 1.0],
            "meta__w_b": [0.4, 0.45, 0.5],
            "y__amount_hemicarbonate": [0.0, 0.002, 0.003],
        }
    ).to_csv(model_table, index=False)

    frame = pd.read_csv(model_table)
    assert resolve_target_column(frame, "hemicarbonate") == "y__amount_hemicarbonate"

    out = write_target_region_analysis(model_table=model_table, target="hemicarbonate", out=tmp_path / "analysis")

    summary = json.loads((out / "target_region_summary.json").read_text(encoding="utf-8"))
    nonzero = pd.read_csv(out / "target_region_nonzero_rows.csv")
    by_system = pd.read_csv(out / "target_region_by_material_system.csv")
    assert summary["target_column"] == "y__amount_hemicarbonate"
    assert summary["nonzero_count"] == 2
    assert nonzero["meta__recipe_id"].tolist() == ["b", "c"]
    assert by_system.set_index("material_system").loc["OPC_fly_ash", "nonzero_count"] == 1
    assert (out / "target_region_analysis.md").exists()
