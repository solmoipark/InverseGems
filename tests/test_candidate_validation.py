import json

import pandas as pd

from inverse_gems.candidate_validation import candidate_recipe_text, validate_candidates
from inverse_gems.candidate_validation import _target_sources_from_candidates


def test_candidate_recipe_text_rebuilds_binder_recipe():
    row = {
        "x__OPC": 30.0,
        "x__slag": 0.0,
        "x__fly_ash": 70.0,
        "x__metakaolin": 0.0,
        "x__silica_fume": 0.0,
        "x__limestone": 0.0,
        "x__gypsum": 0.0,
        "x__w_b": 0.4,
        "x__age_days": 28.0,
    }

    assert candidate_recipe_text(row) == "OPC 30, fly ash 70, w/b 0.4, age 28"


def test_candidate_recipe_text_falls_back_to_meta_columns_for_chemistry_candidates():
    row = {
        "x__chem_oxide_equiv_mol_CaO": 1.2,
        "x__chem_oxide_equiv_mol_SiO2": 0.6,
        "x__xgems_water_g": 40.0,
        "meta__OPC": 60.0,
        "meta__slag": 40.0,
        "meta__fly_ash": 0.0,
        "meta__metakaolin": 0.0,
        "meta__silica_fume": 0.0,
        "meta__limestone": 0.0,
        "meta__gypsum": 0.0,
        "meta__w_b": 0.4,
        "meta__water_g": 40.0,
        "meta__age_days": 28.0,
    }

    assert candidate_recipe_text(row) == "OPC 60, slag 40, w/b 0.4, age 28"


def test_validate_candidates_mock_supports_chemistry_feature_candidates(tmp_path):
    candidates = tmp_path / "candidates.csv"
    pd.DataFrame(
        {
            "rank": [1],
            "score": [0.1],
            "meta__recipe_id": ["source_chem"],
            "meta__chem_hash": ["chem_hash_a"],
            "meta__OPC": [60.0],
            "meta__slag": [40.0],
            "meta__fly_ash": [0.0],
            "meta__metakaolin": [0.0],
            "meta__silica_fume": [0.0],
            "meta__limestone": [0.0],
            "meta__gypsum": [0.0],
            "meta__w_b": [0.4],
            "meta__water_g": [40.0],
            "meta__age_days": [28.0],
            "x__chem_oxide_equiv_mol_CaO": [1.2],
            "x__chem_oxide_equiv_mol_SiO2": [0.6],
            "x__xgems_water_g": [40.0],
            "pred__y__porosity": [0.5],
            "true__y__porosity": [0.51],
        }
    ).to_csv(candidates, index=False)

    out = validate_candidates(
        candidates=candidates,
        out=tmp_path / "validation",
        db=tmp_path / "validation_db",
        top_k=1,
        use_mock=True,
    )

    runs = pd.read_csv(out / "validation_runs.csv")
    comparison = pd.read_csv(out / "validation_comparison.csv")
    assert len(runs) == 1
    assert runs.iloc[0]["recipe_text"] == "OPC 60, slag 40, w/b 0.4, age 28"
    assert runs.iloc[0]["error"] is None or pd.isna(runs.iloc[0]["error"])
    assert len(comparison) == 1
    assert comparison.iloc[0]["chemistry_status"] == "complete"


def test_validate_candidates_mock_writes_comparison_outputs(tmp_path):
    candidates = tmp_path / "candidates.csv"
    pd.DataFrame(
        {
            "rank": [1, 2],
            "score": [0.1, 0.2],
            "meta__recipe_id": ["source_a", "source_b"],
            "meta__chem_hash": ["chem_a", "chem_b"],
            "x__OPC": [30.0, 60.0],
            "x__slag": [0.0, 10.0],
            "x__fly_ash": [70.0, 20.0],
            "x__metakaolin": [0.0, 0.0],
            "x__silica_fume": [0.0, 0.0],
            "x__limestone": [0.0, 5.0],
            "x__gypsum": [0.0, 5.0],
            "x__w_b": [0.4, 0.35],
            "x__age_days": [28.0, 90.0],
            "out_of_domain": [True, False],
            "nearest_scaled_distance": [0.31, 0.02],
            "outside_input_range_count": [1, 0],
            "pred__y__porosity": [0.5, 0.4],
            "true__y__porosity": [0.51, 0.39],
            "pred__y__pH": [12.5, 12.6],
            "true__y__pH": [12.55, 12.65],
        }
    ).to_csv(candidates, index=False)

    out = validate_candidates(
        candidates=candidates,
        out=tmp_path / "validation",
        db=tmp_path / "validation_db",
        top_k=1,
        use_mock=True,
    )

    expected = {
        "query_candidates_used.csv",
        "validation_runs.csv",
        "validated_feature_table.csv",
        "validated_feature_table_all.csv",
        "validation_comparison.csv",
        "validation_summary.json",
    }
    assert expected.issubset({path.name for path in out.iterdir()})

    runs = pd.read_csv(out / "validation_runs.csv")
    comparison = pd.read_csv(out / "validation_comparison.csv")
    assert len(runs) == 1
    assert len(comparison) == 1
    assert comparison.iloc[0]["source_recipe_id"] == "source_a"
    assert comparison.iloc[0]["recipe_text"] == "OPC 30, fly ash 70, w/b 0.4, age 28"
    assert bool(comparison.iloc[0]["out_of_domain"])
    assert comparison.iloc[0]["nearest_scaled_distance"] == 0.31
    assert "validated__y__porosity" in comparison.columns
    assert "diff_validated_minus_pred__y__porosity" in comparison.columns

    summary = json.loads((out / "validation_summary.json").read_text(encoding="utf-8"))
    assert summary["candidate_rows_used"] == 1
    assert summary["validation_status_counts"]["complete"] == 1
    assert summary["pH_reliability"]["pH_water_unreliable_count"] == 0


def test_target_sources_from_candidates_uses_selection_phase_groups():
    frame = pd.DataFrame(
        {
            "pred__y__amount_straetlingite": [0.01],
            "pred__y__amount_aluminosilicate_gel": [0.0],
            "pred__y__amount_monosulfate": [0.02],
        }
    )
    selection = {
        "phase_groups": {
            "amounts": {
                "straetlingite": {"include": ["straetlingite"]},
                "aluminosilicate gel": {"include": ["ZeoliteP", "Chabazite", "Silica-amorph"]},
                "monosulfate": {"include": ["OH_SO4_AFm", "SO4_OH_AFm", "C4AH19"]},
            }
        }
    }

    sources = _target_sources_from_candidates(frame, selection)

    assert sources["y__amount_straetlingite"] == "phase_amount_group__straetlingite"
    assert sources["y__amount_aluminosilicate_gel"] == "phase_amount_group__aluminosilicate gel"
    assert sources["y__amount_monosulfate"] == "phase_amount_group__monosulfate"
