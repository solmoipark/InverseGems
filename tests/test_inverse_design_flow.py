from inverse_gems.inverse_design_flow import build_inverse_design_flow_summary


def test_inverse_design_flow_mentions_ph_uncertainty_policy():
    summary = build_inverse_design_flow_summary(
        {
            "final_stage": "final_selection",
            "skip_validation": False,
            "paths": {
                "compiled_query": "compiled_query",
                "surrogate_search": "surrogate_search/candidates.csv",
                "validation": "validation/validated_candidates.csv",
                "selection": "selection/final_selected_candidates.csv",
            },
            "candidate_search_summary": {
                "input_rows": 20,
                "candidate_rows_after_filters": 10,
                "top_k_written": 5,
            },
            "validation_summary": {
                "candidate_rows_used": 5,
                "validation_runs": 5,
                "validation_status_counts": {"complete": 5},
            },
            "selection_summary": {
                "input_rows": 5,
                "candidate_rows_after_constraints": 3,
                "top_k_written": 2,
                "pH_uncertainty_policy": {
                    "mode": "exclude",
                    "unreliable_rows": 2,
                    "rejected_rows": 1,
                    "penalized_rows": 0,
                },
            },
        }
    )

    stages = {stage["name"]: stage for stage in summary["stages"]}
    final_notes = stages["final_selection"]["notes"]
    assert "pH uncertainty policy: exclude" in final_notes
    assert "pH unreliable rows rejected: 1" in final_notes
    assert any("pH-water-unreliable validation were excluded" in note for note in summary["interpretation_notes"])
