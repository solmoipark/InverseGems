import json

import pandas as pd

from inverse_gems.active_learning_priority import write_active_learning_target_priorities


def test_active_learning_priority_ranks_sparse_evaluation_targets(tmp_path):
    diagnostics = tmp_path / "diagnostics.csv"
    pd.DataFrame(
        [
            {
                "target_column": "y__amount_hemicarbonate",
                "target_label": "hemicarbonate",
                "status": "usable_with_caution",
                "evaluation_reliability": "sparse_full_distribution",
                "r2": -0.05,
                "full_nonzero_count": 65,
                "full_nonzero_fraction": 0.018,
                "test_nonzero_count": 16,
                "reasons": "sparse",
            },
            {
                "target_column": "y__amount_aluminosilicate_gel",
                "target_label": "aluminosilicate gel",
                "status": "usable_with_caution",
                "evaluation_reliability": "unstable_low_test_nonzero",
                "r2": -1.3,
                "full_nonzero_count": 64,
                "full_nonzero_fraction": 0.018,
                "test_nonzero_count": 3,
                "reasons": "low test support",
            },
            {
                "target_column": "y__pH",
                "target_label": "pH",
                "status": "usable_with_caution",
                "evaluation_reliability": "ok",
                "r2": 0.85,
                "pH_water_adjusted": True,
                "full_nonzero_count": 100,
                "full_nonzero_fraction": 1.0,
                "test_nonzero_count": 20,
            },
        ]
    ).to_csv(diagnostics, index=False)

    out = write_active_learning_target_priorities(
        diagnostics=diagnostics,
        out=tmp_path / "priority",
        target_kinds=["amount"],
        max_targets=2,
    )

    ranked = pd.read_csv(out / "active_learning_target_priorities.csv")
    summary = json.loads((out / "active_learning_target_priorities_summary.json").read_text(encoding="utf-8"))
    assert ranked["target_column"].tolist() == ["y__amount_aluminosilicate_gel", "y__amount_hemicarbonate"]
    assert ranked.iloc[0]["active_learning_priority_score"] > ranked.iloc[1]["active_learning_priority_score"]
    assert summary["ranked_rows"] == 2
    assert (out / "active_learning_target_priorities.md").exists()
