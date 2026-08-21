import json

import pandas as pd

from inverse_gems.candidate_review import write_candidate_review


def test_write_candidate_review_for_surrogate_candidates(tmp_path):
    candidates = tmp_path / "candidates.csv"
    pd.DataFrame(
        {
            "rank": [1],
            "score": [0.0],
            "meta__recipe_id": ["r1"],
            "meta__material_system": ["OPC_slag"],
            "meta__chem_hash": ["chem1"],
            "x__OPC": [60.0],
            "x__slag": [40.0],
            "x__w_b": [0.4],
            "x__age_days": [28.0],
            "pred__y__porosity": [0.38],
            "true__y__porosity": [0.39],
            "pred__y__amount_C_A_S_H": [0.045],
            "rank_component__01__maximize__C_A_S_H": [-0.045],
        }
    ).to_csv(candidates, index=False)

    summary = write_candidate_review(candidates=candidates, out=tmp_path / "review", stage="surrogate_search")

    review = pd.read_csv(tmp_path / "review" / "candidate_review.csv")
    assert review.iloc[0]["recipe_text"] == "OPC 60, slag 40, w/b 0.4, age 28"
    assert review.iloc[0]["fly_ash"] == 0.0
    assert review.iloc[0]["validation_status"] == "surrogate_only"
    assert review.iloc[0]["uncertainty_flags"] == "surrogate_only"
    assert review.iloc[0]["ranking_reason"] == "ordered preferences: maximize C-A-S-H"
    assert "not xGEMS/GEMS validated" in review.iloc[0]["validation_note"]
    assert review.iloc[0]["predicted__porosity"] == 0.38
    assert "predicted__amount_C_A_S_H" in review.columns
    assert summary["uncertainty_flag_counts"]["surrogate_only"] == 1
    assert (tmp_path / "review" / "candidate_review.md").exists()


def test_write_candidate_review_for_validated_candidates_flags_uncertainty(tmp_path):
    candidates = tmp_path / "selected.csv"
    pd.DataFrame(
        {
            "selection_rank": [1],
            "candidate_rank": [2],
            "source_recipe_id": ["source_a"],
            "validation_recipe_id": ["valid_a"],
            "material_system": ["OPC_fly_ash"],
            "chemistry_status": ["complete"],
            "solver_status": ["success"],
            "solver_rescued": [True],
            "out_of_domain": [True],
            "nearest_scaled_distance": [0.42],
            "pH_water_reliable": [False],
            "pH_unreliable_reason": ["solver_rescued"],
            "x__OPC": [35.0],
            "x__fly_ash": [65.0],
            "x__w_b": [0.45],
            "x__age_days": [28.0],
            "pred__y__pH": [12.8],
            "validated__y__pH": [12.7],
            "abs_diff_validated_minus_pred__y__pH": [0.1],
            "selection_rank_component__01__minimize__pH": [12.7],
        }
    ).to_csv(candidates, index=False)

    write_candidate_review(candidates=candidates, out=tmp_path / "review", stage="selection")

    review = pd.read_csv(tmp_path / "review" / "candidate_review.csv")
    flags = review.iloc[0]["uncertainty_flags"]
    assert "validated" in flags
    assert "solver_rescued" in flags
    assert "pH_water_uncertain" in flags
    assert "out_of_domain" in flags
    assert review.iloc[0]["validation_status"] == "complete"
    assert review.iloc[0]["ranking_reason"] == "ordered preferences: minimize pH"
    assert "pH uncertain" in review.iloc[0]["validation_note"]
    assert review.iloc[0]["out_of_domain"] is True or str(review.iloc[0]["out_of_domain"]).lower() == "true"
    assert review.iloc[0]["validated__pH"] == 12.7
    payload = json.loads((tmp_path / "review" / "candidate_review_summary.json").read_text(encoding="utf-8"))
    assert payload["uncertainty_flag_counts"]["solver_rescued"] == 1
