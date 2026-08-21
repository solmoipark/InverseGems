from inverse_gems.materials import load_materials
from inverse_gems.porosity import compute_initial_volume_cm3, compute_porosity
from inverse_gems.recipe import parse_recipe


def test_initial_volume_from_masses_and_densities():
    recipe = parse_recipe("OPC 100, w/b 0.4, age 28")
    initial = compute_initial_volume_cm3(recipe)
    assert abs(initial["initial_volume_cm3"] - (40.0 + 100.0 / 3.15)) < 1.0e-9


def test_porosity_includes_unreacted_binders_in_reacted_only_mode():
    recipe = parse_recipe("OPC 30, fly ash 70, w/b 0.4, age 28")
    result = compute_porosity(
        recipe,
        xgems_phase_volumes={"raw phase": 10.0},
        unreacted_masses_g={"OPC": 10.0, "fly_ash": 40.0},
    )
    materials = load_materials()
    expected_unreacted = 10.0 / materials["OPC"].density_g_cm3 + 40.0 / materials["fly_ash"].density_g_cm3
    assert abs(sum(result["unreacted_binder_volumes_cm3"].values()) - expected_unreacted) < 1.0e-9


def test_porosity_warns_instead_of_clipping_by_default():
    recipe = parse_recipe("OPC 100, w/b 0.4, age 28")
    result = compute_porosity(recipe, xgems_phase_volumes={"huge solid": 1000.0})
    assert result["porosity_best_effort"] < 0
    assert result["warnings"]


def test_porosity_prefers_reconstructed_phase_volumes_when_available():
    recipe = parse_recipe("OPC 100, w/b 0.4, age 28")
    result = compute_porosity(
        recipe,
        xgems_phase_volumes={"Calcite": 0.0, "aq_gen": 100.0},
        xgems_phase_volumes_reconstructed={"Calcite": 1.0e-5, "aq_gen": 2.0e-5},
        config={
            "xgems_phase_volume_unit": "m3",
            "prefer_reconstructed_phase_volumes": True,
            "include_unreacted_binders": False,
            "non_solid_phase_name_patterns": ["aq"],
        },
    )

    assert abs(result["included_phase_volumes_cm3"]["Calcite"] - 10.0) < 1.0e-12
    assert result["included_phase_volume_sources"]["Calcite"] == "reconstructed_from_phase_species"
    assert "aq_gen" in result["excluded_non_solid_phase_volumes_raw"]
