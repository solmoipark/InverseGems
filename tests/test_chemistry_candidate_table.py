import pandas as pd

from inverse_gems.chemistry_candidate_table import build_chemistry_candidate_table


def test_build_chemistry_candidate_table_projects_age_dependent_reactive_chemistry(tmp_path):
    recipes = tmp_path / "recipes.csv"
    pd.DataFrame(
        [
            {
                "recipe_id": "osf_28",
                "material_system": "OPC_slag_fly_ash",
                "target_profile": "hemi_test",
                "OPC": 30.0,
                "slag": 40.0,
                "fly_ash": 30.0,
                "metakaolin": 0.0,
                "silica_fume": 0.0,
                "limestone": 0.0,
                "gypsum": 0.0,
                "w_b": 0.4,
                "water_mode": "wb_total",
                "age_days": 28.0,
                "temperature_celsius": 20.0,
            },
            {
                "recipe_id": "osf_56",
                "material_system": "OPC_slag_fly_ash",
                "target_profile": "hemi_test",
                "OPC": 30.0,
                "slag": 40.0,
                "fly_ash": 30.0,
                "metakaolin": 0.0,
                "silica_fume": 0.0,
                "limestone": 0.0,
                "gypsum": 0.0,
                "w_b": 0.4,
                "water_mode": "wb_total",
                "age_days": 56.0,
                "temperature_celsius": 20.0,
            },
        ]
    ).to_csv(recipes, index=False)

    out = build_chemistry_candidate_table(recipes_csv=recipes, out=tmp_path / "chem_candidates.csv")
    table = pd.read_csv(out)

    assert len(table) == 2
    assert table["meta__recipe_id"].tolist() == ["osf_28", "osf_56"]
    assert table["meta__target_profile"].tolist() == ["hemi_test", "hemi_test"]
    assert table["x__OPC"].tolist() == [30.0, 30.0]
    assert table["x__slag"].tolist() == [40.0, 40.0]
    assert table["x__fly_ash"].tolist() == [30.0, 30.0]
    assert table["x__age_days"].tolist() == [28.0, 56.0]
    assert table["x__chem_oxide_equiv_mol_CaO"].iloc[1] > table["x__chem_oxide_equiv_mol_CaO"].iloc[0]
    assert table["x__chem_oxide_equiv_mol_SiO2"].iloc[1] > table["x__chem_oxide_equiv_mol_SiO2"].iloc[0]
    assert table["meta__chem_hash"].nunique() == 2
    assert table["meta__reaction_model_signature"].nunique() == 1
    assert "x__temperature_celsius" in table.columns
    assert "x__xgems_water_g" in table.columns


def test_build_chemistry_candidate_table_records_reaction_parameter_change(tmp_path):
    recipes = tmp_path / "recipes.csv"
    pd.DataFrame(
        [
            {
                "recipe_id": "osf",
                "OPC": 30.0,
                "slag": 40.0,
                "fly_ash": 30.0,
                "w_b": 0.4,
                "age_days": 56.0,
            }
        ]
    ).to_csv(recipes, index=False)
    custom_params = tmp_path / "reaction_params.yaml"
    custom_params.write_text(
        """
id: faster_scm_for_test
scm_reaction:
  fly_ash:
    C: 10.0
  slag:
    C: 8.0
""",
        encoding="utf-8",
    )

    default_out = build_chemistry_candidate_table(recipes_csv=recipes, out=tmp_path / "default.csv")
    custom_out = build_chemistry_candidate_table(
        recipes_csv=recipes,
        out=tmp_path / "custom.csv",
        reaction_model_config=custom_params,
    )
    default = pd.read_csv(default_out)
    custom = pd.read_csv(custom_out)

    assert default.loc[0, "meta__reaction_model_signature"] != custom.loc[0, "meta__reaction_model_signature"]
    assert default.loc[0, "meta__chem_hash"] != custom.loc[0, "meta__chem_hash"]
    assert custom.loc[0, "x__chem_oxide_equiv_mol_SiO2"] > default.loc[0, "x__chem_oxide_equiv_mol_SiO2"]
