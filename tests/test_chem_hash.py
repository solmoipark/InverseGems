from inverse_gems.chem_hash import compute_chem_hash
from inverse_gems.chemistry_vector import chemistry_vector_from_component_moles


def test_hash_equivalence_for_equivalent_chemistry():
    a = chemistry_vector_from_component_moles({"C3S": 1.0})
    b = chemistry_vector_from_component_moles({"CaO": 3.0, "SiO2": 1.0})
    ha = compute_chem_hash(a, water_mol=10.0, temperature_celsius=20.0, dat_lst_hash="db", species_map_hash="map")
    hb = compute_chem_hash(b, water_mol=10.0, temperature_celsius=20.0, dat_lst_hash="db", species_map_hash="map")
    assert ha.chem_hash == hb.chem_hash


def test_hash_changes_for_water_temperature_and_database():
    vector = chemistry_vector_from_component_moles({"C3S": 1.0})
    base = compute_chem_hash(vector, water_mol=10.0, temperature_celsius=20.0, dat_lst_hash="db", species_map_hash="map")
    assert base.chem_hash != compute_chem_hash(
        vector, water_mol=11.0, temperature_celsius=20.0, dat_lst_hash="db", species_map_hash="map"
    ).chem_hash
    assert base.chem_hash != compute_chem_hash(
        vector, water_mol=10.0, temperature_celsius=30.0, dat_lst_hash="db", species_map_hash="map"
    ).chem_hash
    assert base.chem_hash != compute_chem_hash(
        vector, water_mol=10.0, temperature_celsius=20.0, dat_lst_hash="other", species_map_hash="map"
    ).chem_hash
