import yaml

from inverse_gems.cached_forward import run_forward_cached
from inverse_gems.materials import load_materials
from inverse_gems.reaction_parameters import load_reaction_parameters
from inverse_gems.recipe import parse_recipe
from inverse_gems.xgems_input_builder import build_xgems_input


def test_reaction_parameter_config_overrides_scm_alpha_and_availability(tmp_path):
    config = tmp_path / "reaction_parameters.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "id": "low_fly_ash_no_cap",
                "scm_reaction": {"fly_ash": {"D": 0.05}},
                "availability_modifier": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    recipe = parse_recipe("OPC 60, fly ash 40, w/b 0.4, age 28", materials=load_materials())
    default_input = build_xgems_input(recipe)
    custom_parameters = load_reaction_parameters(config)
    custom_input = build_xgems_input(recipe, reaction_parameters=custom_parameters)

    assert custom_parameters.id == "low_fly_ash_no_cap"
    assert custom_input.reaction_degrees["reaction_parameter_set"]["id"] == "low_fly_ash_no_cap"
    assert custom_input.reaction_degrees["availability_modifier"]["applied"] is False
    assert custom_input.reaction_degrees["scm"]["fly_ash"] < default_input.reaction_degrees["scm"]["fly_ash"]


def test_reaction_parameter_config_changes_prepared_and_chem_hash(tmp_path):
    config = tmp_path / "reaction_parameters.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "id": "low_fly_ash_no_cap",
                "scm_reaction": {"fly_ash": {"D": 0.05}},
                "availability_modifier": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    kwargs = {
        "recipe_text": "OPC 60, fly ash 40, w/b 0.4, age 28",
        "db": tmp_path / "db",
        "use_mock": True,
    }
    default = run_forward_cached(**kwargs)
    custom = run_forward_cached(**kwargs, reaction_model_config=config)

    assert default["chem_hash"] != custom["chem_hash"]
    assert default["prepared_id"] != custom["prepared_id"]
    assert default["reaction_model_signature"] != custom["reaction_model_signature"]
