from inverse_gems.age_grids import age_metadata, get_age_values


def test_early_dense_v1_age_grid_metadata():
    ages = get_age_values("early_dense_v1")
    assert len(ages) == 28
    assert 0.0 not in ages
    assert any(age < 1 for age in ages)
    assert {1.0, 1.125, 1.25, 1.5}.issubset(set(ages))
    assert any(age != int(age) for age in ages)
    meta = age_metadata(0.0417)
    assert meta.age_hours == 0.0417 * 24
    assert meta.age_minutes == 0.0417 * 24 * 60
    assert meta.age_label.endswith("h")
    assert meta.age_bin == "early"
