from inverse_gems.cached_forward import run_forward_cached
from inverse_gems.chemistry_vector import chemistry_vector_from_amounts_kg, chemistry_vector_from_source_ledger, vectors_close
from inverse_gems.materials import load_materials
from inverse_gems.recipe import parse_recipe
from inverse_gems.source_ledger import build_source_ledger
from inverse_gems.xgems_input_builder import build_xgems_input


def test_xgems_water_fraction_changes_species_water_not_pk_water():
    recipe = parse_recipe("OPC 100, w/b 0.4, age 28")
    xgems_input = build_xgems_input(recipe, xgems_water_mode="fraction_of_initial", xgems_water_factor=0.75)

    assert xgems_input.species_amounts_kg["H2O@"] == 0.03
    assert xgems_input.equilibrium_water_g == 30.0
    assert xgems_input.reaction_degrees["water_to_binder_for_pk"] == 0.4
    assert xgems_input.reaction_degrees["xgems_water"]["initial_water_g"] == 40.0
    assert xgems_input.reaction_degrees["xgems_water"]["equilibrium_water_g"] == 30.0


def test_source_ledger_uses_xgems_equilibrium_water():
    materials = load_materials()
    recipe = parse_recipe("OPC 30, fly ash 70, w/b 0.4, age 28", materials=materials)
    xgems_input = build_xgems_input(
        recipe,
        materials=materials,
        xgems_water_mode="fraction_of_initial",
        xgems_water_factor=0.75,
    )
    rows = build_source_ledger(recipe_id="r1", chem_hash="h1", recipe=recipe, xgems_input=xgems_input, materials=materials)

    water_rows = [row for row in rows if row["source_material"] == "water"]
    assert water_rows
    assert {row["source_mass_g_initial"] for row in water_rows} == {30.0}
    from_ledger = chemistry_vector_from_source_ledger(rows)
    from_amounts = chemistry_vector_from_amounts_kg(xgems_input.species_amounts_kg)
    assert vectors_close(from_ledger.element_mol, from_amounts.element_mol, tolerance=1.0e-10)


def test_cached_hash_changes_with_xgems_water_policy(tmp_path):
    base = run_forward_cached(recipe_text="OPC 100, w/b 0.4, age 28", db=tmp_path / "db", use_mock=True)
    reduced = run_forward_cached(
        recipe_text="OPC 100, w/b 0.4, age 28",
        db=tmp_path / "db",
        use_mock=True,
        xgems_water_mode="fraction_of_initial",
        xgems_water_factor=0.75,
    )

    assert base["chem_hash"] != reduced["chem_hash"]
    assert reduced["xgems_water_g"] == 30.0
    assert reduced["xgems_w_b"] == 0.3
