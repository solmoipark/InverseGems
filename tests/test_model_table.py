import json

import pandas as pd
import yaml

from inverse_gems.model_table import build_model_table


def test_build_model_table_filters_and_writes_schema(tmp_path):
    feature_table = tmp_path / "features.csv"
    pd.DataFrame(
        {
            "recipe_id": ["a", "b", "c"],
            "chem_hash": ["ha", "hb", "hc"],
            "template_name": ["t1", "t1", "t2"],
            "chemistry_status": ["complete", "failed", "complete"],
            "OPC": [80.0, 90.0, 70.0],
            "slag": [20.0, 0.0, 30.0],
            "w_b": [0.4, 0.5, 0.6],
            "age_days": [1.0, 7.0, 100.0],
            "porosity": [0.45, 0.50, 0.35],
            "scalar__pH": [12.4, 12.5, 12.2],
            "pH_water_reliable": [True, True, False],
            "xgems_water_delta_g": [0.0, 0.0, 5.0],
            "phase_amount_group__C-A-S-H": [0.1, 0.2, 0.3],
            "phase_amount_group__monosulfate": [0.0, 0.0, 0.01],
        }
    ).to_csv(feature_table, index=False)
    config = tmp_path / "model_dataset.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "filters": {"chemistry_status": "complete", "drop_missing_inputs": True, "drop_missing_targets": True},
                "metadata": {"include": ["recipe_id", "chem_hash", "template_name", "material_system", "chemistry_status"]},
                "inputs": {
                    "include": ["OPC", "slag", "w_b", "age_days"],
                    "derived": {"log_age_days": {"source": "age_days", "transform": "log10"}},
                },
                "targets": {
                    "scalars": {"porosity": "porosity", "pH": "scalar__pH"},
                    "phase_amount_groups": {"include": ["C-A-S-H", "monosulfate"]},
                },
                "output": {
                    "metadata_prefix": "meta__",
                    "input_prefix": "x__",
                    "target_prefix": "y__",
                    "sanitize_column_names": True,
                },
            }
        ),
        encoding="utf-8",
    )

    out = build_model_table(feature_table=feature_table, config=config, out=tmp_path / "model.csv")
    model = pd.read_csv(out)

    assert len(model) == 2
    assert "meta__recipe_id" in model.columns
    assert "meta__material_system" in model.columns
    assert model["meta__material_system"].fillna("").tolist() == ["", ""]
    assert "x__log_age_days" in model.columns
    assert "y__amount_C_A_S_H" in model.columns
    assert "y__amount_monosulfate" in model.columns
    assert model["meta__recipe_id"].tolist() == ["a", "c"]
    assert model["x__log_age_days"].tolist() == [0.0, 2.0]

    schema = json.loads(out.with_suffix(out.suffix + ".schema.json").read_text(encoding="utf-8"))
    assert schema["filters"]["input_rows"] == 3
    assert schema["filters"]["rows_after_status_filter"] == 2
    assert schema["row_count"] == 2
    assert set(item["column"] for item in schema["roles"]["targets"]) == {
        "y__porosity",
        "y__pH",
        "y__amount_C_A_S_H",
        "y__amount_monosulfate",
    }
    assert schema["target_summary"]["y__pH"]["pH_water_unreliable_count"] == 1
    assert schema["target_summary"]["y__pH"]["pH_water_unreliable_fraction"] == 0.5
    assert schema["target_summary"]["y__pH"]["pH_xgems_water_changed_count"] == 1


def test_build_model_table_applies_material_system_and_age_view_filters(tmp_path):
    feature_table = tmp_path / "features.csv"
    pd.DataFrame(
        {
            "recipe_id": ["slag28", "slag90", "lc3_28", "slag_failed"],
            "chem_hash": ["h1", "h2", "h3", "h4"],
            "template_name": ["OPC_slag", "OPC_slag", "LC3_like", "OPC_slag"],
            "material_system": ["OPC_slag", "OPC_slag", "LC3_like", "OPC_slag"],
            "chemistry_status": ["complete", "complete", "complete", "failed"],
            "OPC": [50.0, 50.0, 55.0, 50.0],
            "slag": [45.0, 45.0, 0.0, 45.0],
            "gypsum": [5.0, 5.0, 5.0, 5.0],
            "w_b": [0.4, 0.4, 0.4, 0.4],
            "age_days": [28.0, 90.0, 28.0, 28.0],
            "porosity": [0.30, 0.28, 0.32, 0.31],
            "scalar__pH": [13.0, 13.1, 12.8, 13.0],
            "phase_amount_group__C-A-S-H": [0.04, 0.05, 0.06, 0.04],
        }
    ).to_csv(feature_table, index=False)
    config = tmp_path / "model_dataset.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "filters": {
                    "chemistry_status": "complete",
                    "columns": {
                        "material_system": {"include": ["OPC_slag"]},
                        "age_days": {"equals": 28.0},
                    },
                    "drop_missing_inputs": True,
                    "drop_missing_targets": True,
                },
                "metadata": {"include": ["recipe_id", "material_system", "chemistry_status"]},
                "inputs": {"include": ["OPC", "slag", "gypsum", "w_b", "age_days"]},
                "targets": {
                    "scalars": {"porosity": "porosity", "pH": "scalar__pH"},
                    "phase_amount_groups": {"include": ["C-A-S-H"]},
                },
                "output": {
                    "metadata_prefix": "meta__",
                    "input_prefix": "x__",
                    "target_prefix": "y__",
                    "sanitize_column_names": True,
                },
            }
        ),
        encoding="utf-8",
    )

    out = build_model_table(feature_table=feature_table, config=config, out=tmp_path / "model.csv")
    model = pd.read_csv(out)

    assert model["meta__recipe_id"].tolist() == ["slag28"]
    assert model["meta__material_system"].tolist() == ["OPC_slag"]
    assert model["x__age_days"].tolist() == [28.0]

    schema = json.loads(out.with_suffix(out.suffix + ".schema.json").read_text(encoding="utf-8"))
    assert schema["filters"]["rows_after_status_filter"] == 3
    assert schema["filters"]["rows_after_view_filters"] == 1
    assert schema["filters"]["rejected_by_view_filters"]["material_system"] == 1
    assert schema["filters"]["rejected_by_view_filters"]["age_days"] == 1


def test_build_model_table_filters_and_preserves_reaction_provenance(tmp_path):
    feature_table = tmp_path / "features.csv"
    pd.DataFrame(
        {
            "recipe_id": ["a", "b"],
            "chem_hash": ["ha", "hb"],
            "prepared_id": ["pa", "pb"],
            "reaction_model_id": ["params_a", "params_b"],
            "reaction_model_signature": ["sig_a", "sig_b"],
            "chemistry_status": ["complete", "complete"],
            "OPC": [60.0, 60.0],
            "fly_ash": [40.0, 40.0],
            "w_b": [0.4, 0.4],
            "age_days": [28.0, 28.0],
            "porosity": [0.45, 0.35],
        }
    ).to_csv(feature_table, index=False)
    config = tmp_path / "model_dataset.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "filters": {"chemistry_status": "complete"},
                "metadata": {"include": ["recipe_id", "chem_hash"]},
                "inputs": {"include": ["OPC", "fly_ash", "w_b", "age_days"]},
                "targets": {"scalars": {"porosity": "porosity"}},
            }
        ),
        encoding="utf-8",
    )

    out = build_model_table(
        feature_table=feature_table,
        config=config,
        out=tmp_path / "model.csv",
        reaction_model_signature="sig_a",
    )
    model = pd.read_csv(out)
    schema = json.loads(out.with_suffix(out.suffix + ".schema.json").read_text(encoding="utf-8"))

    assert model["meta__recipe_id"].tolist() == ["a"]
    assert model["meta__prepared_id"].tolist() == ["pa"]
    assert model["meta__reaction_model_id"].tolist() == ["params_a"]
    assert model["meta__reaction_model_signature"].tolist() == ["sig_a"]
    assert schema["filters"]["rows_after_reaction_model_filter"] == 1
    assert schema["reaction_provenance"]["reaction_model_signatures"] == ["sig_a"]
