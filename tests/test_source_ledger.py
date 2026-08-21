from inverse_gems.chemistry_vector import chemistry_vector_from_amounts_kg, chemistry_vector_from_source_ledger, vectors_close
from inverse_gems.materials import load_materials
from inverse_gems.recipe import parse_recipe
from inverse_gems.source_ledger import build_source_ledger
from inverse_gems.xgems_input_builder import build_xgems_input


def test_source_ledger_sums_to_canonical_vector():
    materials = load_materials()
    recipe = parse_recipe("OPC 30, fly ash 70, w/b 0.4, age 28", materials=materials)
    xgems_input = build_xgems_input(recipe, materials=materials)
    rows = build_source_ledger(recipe_id="r1", chem_hash="h1", recipe=recipe, xgems_input=xgems_input, materials=materials)
    from_ledger = chemistry_vector_from_source_ledger(rows)
    from_amounts = chemistry_vector_from_amounts_kg(xgems_input.species_amounts_kg)
    assert vectors_close(from_ledger.element_mol, from_amounts.element_mol, tolerance=1.0e-10)
