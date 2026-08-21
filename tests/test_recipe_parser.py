import pytest

from inverse_gems.recipe import parse_recipe


def test_parses_opc_fly_ash_wb_age_example():
    recipe = parse_recipe("OPC 30, fly ash 70, w/b 0.4, age 28")
    assert recipe.binder_masses_g["OPC"] == 30
    assert recipe.binder_masses_g["fly_ash"] == 70
    assert recipe.water_g == 40
    assert recipe.age_days == 28


def test_normalizes_aliases():
    recipe = parse_recipe("PC 30, FA 70, w/b 0.4, age 28")
    assert "OPC" in recipe.binder_masses_g
    assert "fly_ash" in recipe.binder_masses_g


def test_validates_binder_sum_when_not_normalizing():
    with pytest.raises(ValueError):
        parse_recipe("OPC 30, fly ash 60, w/b 0.4, age 28", normalize=False)


def test_direct_water_mode():
    recipe = parse_recipe("OPC 50, slag 30, limestone 15, gypsum 5, water 40, age 90")
    assert recipe.water_mode == "direct_h2o"
    assert recipe.water_g == 40
