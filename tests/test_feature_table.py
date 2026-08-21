import pandas as pd
import yaml

from inverse_gems.cached_forward import run_forward_cached
from inverse_gems.feature_table import build_feature_table


def test_feature_table_one_row_per_recipe_and_missing_phase_zero(tmp_path):
    db_dir = tmp_path / "db"
    first = run_forward_cached(recipe_text="OPC 100, w/b 0.4, age 28", db=db_dir, use_mock=True)
    second = run_forward_cached(
        recipe_text="OPC 100, w/b 0.4, age 28",
        db=db_dir,
        use_mock=True,
        recipe_metadata={"target_profile": "hemi_test"},
    )
    selection = tmp_path / "selection.yaml"
    selection.write_text(
        yaml.safe_dump(
            {
                "phase_amounts": {"include": ["Mock C-S-H raw phase", "Mock Portlandite", "Definitely Missing Phase"]},
                "phase_volumes": {"include": ["Mock C-S-H raw phase", "Mock Portlandite"]},
                "aqueous_species": {"include": []},
                "phase_groups": {
                    "amounts": {
                        "mock hydrate sum": {
                            "include": ["Mock C-S-H raw phase", "Mock Portlandite"],
                        }
                    },
                    "volumes": {
                        "mock hydrate volume": {
                            "include": ["Mock C-S-H raw phase", "Mock Portlandite"],
                        }
                    },
                },
                "scalars": {"include": ["pH", "porosity"]},
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "features.csv"
    build_feature_table(db=db_dir, selection=selection, out=out)
    frame = pd.read_csv(out)
    assert len(frame) == 2
    assert set(frame["chem_hash"]) == {first["chem_hash"], second["chem_hash"]}
    assert frame["chemistry_status"].eq("complete").all()
    assert frame["xgems_run_dir"].notna().all()
    assert "prepared_id" in frame.columns
    assert "reaction_model_id" in frame.columns
    assert "reaction_model_signature" in frame.columns
    assert frame["prepared_id"].notna().all()
    assert frame["reaction_model_signature"].notna().all()
    assert "solver_rescued" in frame.columns
    assert "target_profile" in frame.columns
    assert frame.loc[frame["recipe_id"] == second["recipe_id"], "target_profile"].iloc[0] == "hemi_test"
    assert "xgems_retry_count" in frame.columns
    assert "pH_water_reliable" in frame.columns
    assert "xgems_water_delta_g" in frame.columns
    assert frame["pH_water_reliable"].astype(str).str.lower().eq("true").all()
    assert frame["xgems_water_delta_g"].fillna(0.0).abs().le(1.0e-9).all()
    assert "phase_amount__Definitely Missing Phase" in frame.columns
    assert frame["phase_amount__Definitely Missing Phase"].eq(0.0).all()
    amount_sum = frame["phase_amount__Mock C-S-H raw phase"] + frame["phase_amount__Mock Portlandite"]
    volume_sum = frame["phase_volume__Mock C-S-H raw phase"] + frame["phase_volume__Mock Portlandite"]
    assert frame["phase_amount_group__mock hydrate sum"].equals(amount_sum)
    assert frame["phase_volume_group__mock hydrate volume"].equals(volume_sum)
    assert out.with_suffix(out.suffix + ".warnings.json").exists()


def test_feature_table_can_filter_recipe_prefix_material_system_and_age(tmp_path):
    db_dir = tmp_path / "db"
    run_forward_cached(
        recipe_text="OPC 50, slag 45, gypsum 5, w/b 0.4, age 28",
        db=db_dir,
        use_mock=True,
        recipe_id="OPC_slag_age28_000001_age_28",
        recipe_metadata={"material_system": "OPC_slag", "template_name": "OPC_slag"},
    )
    run_forward_cached(
        recipe_text="OPC 50, slag 45, gypsum 5, w/b 0.4, age 90",
        db=db_dir,
        use_mock=True,
        recipe_id="OPC_slag_age90_000001_age_90",
        recipe_metadata={"material_system": "OPC_slag", "template_name": "OPC_slag"},
    )
    run_forward_cached(
        recipe_text="OPC 50, metakaolin 25, limestone 20, gypsum 5, w/b 0.4, age 28",
        db=db_dir,
        use_mock=True,
        recipe_id="LC3_age28_000001_age_28",
        recipe_metadata={"material_system": "LC3_like", "template_name": "LC3_like"},
    )

    selection = tmp_path / "selection.yaml"
    selection.write_text(
        yaml.safe_dump(
            {
                "phase_amounts": {"include": ["Mock C-S-H raw phase"]},
                "phase_volumes": {"include": ["Mock C-S-H raw phase"]},
                "aqueous_species": {"include": []},
                "scalars": {"include": ["pH", "porosity"]},
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "features.csv"
    build_feature_table(
        db=db_dir,
        selection=selection,
        out=out,
        recipe_id_prefix="OPC_slag_age28_",
        material_system="OPC_slag",
        age_days=28.0,
    )

    frame = pd.read_csv(out)
    assert frame["recipe_id"].tolist() == ["OPC_slag_age28_000001_age_28"]
    assert frame["material_system"].tolist() == ["OPC_slag"]
    assert frame["age_days"].tolist() == [28.0]
