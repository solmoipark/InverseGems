from inverse_gems.chemistry_vector import chemistry_vector_from_component_moles, vectors_close


def test_c3s_equals_three_cao_plus_sio2():
    c3s = chemistry_vector_from_component_moles({"C3S": 1.0})
    oxides = chemistry_vector_from_component_moles({"CaO": 3.0, "SiO2": 1.0})
    assert vectors_close(c3s.element_mol, oxides.element_mol)
    assert vectors_close(c3s.oxide_equivalent_mol, oxides.oxide_equivalent_mol)
