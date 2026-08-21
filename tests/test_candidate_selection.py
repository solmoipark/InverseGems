import json

import pandas as pd
import yaml

from inverse_gems.candidate_selection import select_candidates


def test_select_candidates_filters_scores_and_writes_outputs(tmp_path):
    validation = tmp_path / "validation_comparison.csv"
    pd.DataFrame(
        {
            "candidate_rank": [1, 2, 3],
            "source_recipe_id": ["a", "b", "c"],
            "validation_recipe_id": ["va", "vb", "vc"],
            "chemistry_status": ["complete", "complete", "failed"],
            "recipe_text": ["OPC 30, fly ash 70, w/b 0.4, age 28", "OPC 50, slag 50, w/b 0.35, age 90", "OPC 80, w/b 0.4, age 28"],
            "validated__y__porosity": [0.30, 0.24, 0.20],
            "validated__y__amount_C_A_S_H": [0.050, 0.060, 0.070],
            "validated__y__amount_ettringite": [0.010, 0.020, 0.030],
            "validated__y__amount_Portlandite": [0.0, 0.0, 0.0],
            "pred__y__porosity": [0.31, 0.25, 0.21],
            "abs_diff_validated_minus_pred__y__porosity": [0.01, 0.01, 0.01],
        }
    ).to_csv(validation, index=False)
    feature_table = tmp_path / "validated_feature_table.csv"
    pd.DataFrame(
        {
            "recipe_id": ["va", "vb", "vc"],
            "OPC": [30.0, 50.0, 80.0],
            "slag": [0.0, 50.0, 0.0],
            "fly_ash": [70.0, 0.0, 0.0],
            "w_b": [0.40, 0.35, 0.40],
            "age_days": [28.0, 90.0, 28.0],
            "xgems_run_dir": ["raw_a", "raw_b", "raw_c"],
            "xgems_water_delta_g": [0.0, 2.0, 0.0],
            "pH_water_reliable": [True, False, True],
        }
    ).to_csv(feature_table, index=False)
    config = tmp_path / "selection.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "top_k": 5,
                "constraints": {
                    "chemistry_status": "complete",
                    "inputs": {"OPC": {"max": 60}},
                    "validated_targets": {"porosity": {"max": 0.31}, "C-A-S-H": {"min": 0.055}},
                    "inputs": {"age_days": {"equals": 90.0}},
                },
                "objectives": {
                    "minimize": {"validated_targets": {"porosity": 1.0}},
                    "maximize": {"validated_targets": {"C-A-S-H": 1.0}},
                },
            }
        ),
        encoding="utf-8",
    )

    out = select_candidates(validation=validation, feature_table=feature_table, config=config, out=tmp_path / "selected")

    selected = pd.read_csv(out / "selected_candidates.csv")
    assert len(selected) == 1
    assert selected.iloc[0]["source_recipe_id"] == "b"
    assert selected.iloc[0]["OPC"] == 50.0
    assert "pH_water_reliable" in selected.columns
    assert str(selected.iloc[0]["pH_water_reliable"]).lower() in {"false", "0"}
    assert "selection_score" in selected.columns
    assert (out / "selected_candidates.md").exists()
    assert (out / "selected_candidates.json").exists()

    summary = json.loads((out / "selection_summary.json").read_text(encoding="utf-8"))
    assert summary["input_rows"] == 3
    assert summary["candidate_rows_after_constraints"] == 1
    assert summary["top_k_written"] == 1


def test_select_candidates_lexicographic_preferences_use_input_order(tmp_path):
    validation = tmp_path / "validation_comparison.csv"
    pd.DataFrame(
        {
            "candidate_rank": [1, 2, 3],
            "source_recipe_id": ["high_opc", "low_opc", "lower_csh"],
            "validation_recipe_id": ["va", "vb", "vc"],
            "chemistry_status": ["complete", "complete", "complete"],
            "recipe_text": [
                "OPC 70, slag 30, w/b 0.4, age 28",
                "OPC 40, slag 60, w/b 0.4, age 28",
                "OPC 20, slag 80, w/b 0.4, age 28",
            ],
            "validated__y__amount_C_A_S_H": [0.0602, 0.0601, 0.050],
            "validated__y__porosity": [0.30, 0.35, 0.20],
        }
    ).to_csv(validation, index=False)
    feature_table = tmp_path / "validated_feature_table.csv"
    pd.DataFrame(
        {
            "recipe_id": ["va", "vb", "vc"],
            "OPC": [70.0, 40.0, 20.0],
            "slag": [30.0, 60.0, 80.0],
            "w_b": [0.40, 0.40, 0.40],
            "age_days": [28.0, 28.0, 28.0],
        }
    ).to_csv(feature_table, index=False)
    config = tmp_path / "selection.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "top_k": 3,
                "constraints": {"chemistry_status": "complete"},
                "ranking": {"mode": "lexicographic"},
                "preferences": [
                    {"target": "C-A-S-H", "direction": "maximize", "tolerance": 0.001},
                    {"input": "OPC", "direction": "minimize"},
                    {"target": "porosity", "direction": "minimize"},
                ],
            }
        ),
        encoding="utf-8",
    )

    out = select_candidates(validation=validation, feature_table=feature_table, config=config, out=tmp_path / "selected")

    selected = pd.read_csv(out / "selected_candidates.csv")
    assert selected["source_recipe_id"].tolist() == ["low_opc", "high_opc", "lower_csh"]
    assert "selection_rank_component__01__maximize__C_A_S_H" in selected.columns
    assert "selection_rank_component__02__minimize__OPC" in selected.columns
    summary = json.loads((out / "selection_summary.json").read_text(encoding="utf-8"))
    assert summary["resolved_objectives"][0]["mode"] == "lexicographic"


def test_select_candidates_treats_absent_zero_fixed_input_as_satisfied(tmp_path):
    validation = tmp_path / "validation_comparison.csv"
    pd.DataFrame(
        {
            "candidate_rank": [1, 2],
            "source_recipe_id": ["a", "b"],
            "validation_recipe_id": ["va", "vb"],
            "chemistry_status": ["complete", "complete"],
            "validated__y__porosity": [0.30, 0.35],
        }
    ).to_csv(validation, index=False)
    feature_table = tmp_path / "validated_feature_table.csv"
    pd.DataFrame(
        {
            "recipe_id": ["va", "vb"],
            "OPC": [45.0, 70.0],
            "slag": [55.0, 30.0],
            "w_b": [0.40, 0.40],
            "age_days": [28.0, 28.0],
        }
    ).to_csv(feature_table, index=False)
    config = tmp_path / "selection.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "top_k": 5,
                "constraints": {
                    "chemistry_status": "complete",
                    "inputs": {"fly_ash": {"max": 0.0}, "OPC": {"max": 60.0}},
                },
            }
        ),
        encoding="utf-8",
    )

    out = select_candidates(validation=validation, feature_table=feature_table, config=config, out=tmp_path / "selected")

    selected = pd.read_csv(out / "selected_candidates.csv")
    assert selected["source_recipe_id"].tolist() == ["a"]
    summary = json.loads((out / "selection_summary.json").read_text(encoding="utf-8"))
    absent = [item for item in summary["resolved_constraints"] if item["name"] == "fly_ash"][0]
    assert absent["column"] == "<absent input assumed zero>"
    assert absent["rejected"] == 0


def test_select_candidates_preserves_dynamic_validated_targets(tmp_path):
    validation = tmp_path / "validation_comparison.csv"
    pd.DataFrame(
        {
            "candidate_rank": [1, 2],
            "source_recipe_id": ["a", "b"],
            "validation_recipe_id": ["va", "vb"],
            "chemistry_status": ["complete", "complete"],
            "recipe_text": [
                "OPC 75, metakaolin 23, gypsum 2, w/b 0.45, age 28",
                "OPC 78, metakaolin 20, gypsum 2, w/b 0.45, age 28",
            ],
            "validated__y__amount_straetlingite": [0.025, 0.018],
            "pred__y__amount_straetlingite": [0.024, 0.019],
            "diff_validated_minus_pred__y__amount_straetlingite": [0.001, -0.001],
            "abs_diff_validated_minus_pred__y__amount_straetlingite": [0.001, 0.001],
        }
    ).to_csv(validation, index=False)
    feature_table = tmp_path / "validated_feature_table.csv"
    pd.DataFrame(
        {
            "recipe_id": ["va", "vb"],
            "OPC": [75.0, 78.0],
            "metakaolin": [23.0, 20.0],
            "gypsum": [2.0, 2.0],
            "w_b": [0.45, 0.45],
            "age_days": [28.0, 28.0],
        }
    ).to_csv(feature_table, index=False)
    config = tmp_path / "selection.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "top_k": 2,
                "constraints": {
                    "chemistry_status": "complete",
                    "validated_targets": {"straetlingite": {"min": 0.02}},
                    "prediction_errors": {"straetlingite": {"max_abs": 0.002}},
                },
                "ranking": {"mode": "lexicographic"},
                "preferences": [{"target": "straetlingite", "direction": "maximize"}],
            }
        ),
        encoding="utf-8",
    )

    out = select_candidates(validation=validation, feature_table=feature_table, config=config, out=tmp_path / "selected")

    selected = pd.read_csv(out / "selected_candidates.csv")
    assert selected["source_recipe_id"].tolist() == ["a"]
    assert "validated__y__amount_straetlingite" in selected.columns
    assert "pred__y__amount_straetlingite" in selected.columns
    assert "diff_validated_minus_pred__y__amount_straetlingite" in selected.columns


def test_select_candidates_auto_excludes_ph_unreliable_when_ph_is_constraint(tmp_path):
    validation = tmp_path / "validation_comparison.csv"
    pd.DataFrame(
        {
            "candidate_rank": [1, 2],
            "source_recipe_id": ["unreliable_high_ph", "reliable_lower_ph"],
            "validation_recipe_id": ["va", "vb"],
            "chemistry_status": ["complete", "complete"],
            "validated__y__pH": [13.2, 12.4],
        }
    ).to_csv(validation, index=False)
    feature_table = tmp_path / "validated_feature_table.csv"
    pd.DataFrame(
        {
            "recipe_id": ["va", "vb"],
            "OPC": [40.0, 45.0],
            "slag": [30.0, 30.0],
            "fly_ash": [30.0, 25.0],
            "w_b": [0.54, 0.42],
            "age_days": [28.0, 28.0],
            "pH_water_reliable": [False, True],
        }
    ).to_csv(feature_table, index=False)
    config = tmp_path / "selection.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "top_k": 2,
                "constraints": {
                    "chemistry_status": "complete",
                    "validated_targets": {"pH": {"min": 12.0}},
                },
                "ranking": {"mode": "lexicographic"},
                "preferences": [{"target": "pH", "direction": "maximize"}],
            }
        ),
        encoding="utf-8",
    )

    out = select_candidates(validation=validation, feature_table=feature_table, config=config, out=tmp_path / "selected")

    selected = pd.read_csv(out / "selected_candidates.csv")
    assert selected["source_recipe_id"].tolist() == ["reliable_lower_ph"]
    summary = json.loads((out / "selection_summary.json").read_text(encoding="utf-8"))
    policy = summary["pH_uncertainty_policy"]
    assert policy["mode"] == "exclude"
    assert policy["rejected_rows"] == 1
    assert policy["pH_used_as_constraint"] is True


def test_select_candidates_auto_penalizes_ph_unreliable_when_ph_is_ranking_target(tmp_path):
    validation = tmp_path / "validation_comparison.csv"
    pd.DataFrame(
        {
            "candidate_rank": [1, 2],
            "source_recipe_id": ["unreliable_high_ph", "reliable_lower_ph"],
            "validation_recipe_id": ["va", "vb"],
            "chemistry_status": ["complete", "complete"],
            "validated__y__pH": [13.2, 12.4],
            "validated__y__porosity": [0.42, 0.44],
        }
    ).to_csv(validation, index=False)
    feature_table = tmp_path / "validated_feature_table.csv"
    pd.DataFrame(
        {
            "recipe_id": ["va", "vb"],
            "OPC": [40.0, 45.0],
            "slag": [30.0, 30.0],
            "fly_ash": [30.0, 25.0],
            "w_b": [0.54, 0.42],
            "age_days": [28.0, 28.0],
            "pH_water_reliable": [False, True],
        }
    ).to_csv(feature_table, index=False)
    config = tmp_path / "selection.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "top_k": 2,
                "constraints": {"chemistry_status": "complete"},
                "ranking": {"mode": "lexicographic"},
                "preferences": [{"target": "pH", "direction": "maximize"}],
            }
        ),
        encoding="utf-8",
    )

    out = select_candidates(validation=validation, feature_table=feature_table, config=config, out=tmp_path / "selected")

    selected = pd.read_csv(out / "selected_candidates.csv")
    assert selected["source_recipe_id"].tolist() == ["reliable_lower_ph", "unreliable_high_ph"]
    assert "selection_rank_component__00__pH_water_reliability" in selected.columns
    assert selected.loc[selected["source_recipe_id"] == "unreliable_high_ph", "selection_pH_uncertainty_penalty"].iloc[0] > 0
    summary = json.loads((out / "selection_summary.json").read_text(encoding="utf-8"))
    policy = summary["pH_uncertainty_policy"]
    assert policy["mode"] == "penalize"
    assert policy["penalized_rows"] == 1
    assert policy["pH_used_in_ranking"] is True
