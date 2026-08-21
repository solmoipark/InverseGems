from inverse_gems.sampling import expand_age_rows


def test_expand_ages_direct_list_preserves_float_ages():
    rows = expand_age_rows(
        [{"recipe_id": "r1", "OPC": 100, "w_b": 0.4, "water_g": 40, "water_mode": "wb_total"}],
        ages="0.0035,1,1.125",
    )
    assert [row["age_days"] for row in rows] == [0.0035, 1.0, 1.125]
    assert rows[0]["age_bin"] == "ultra_early"
    assert rows[2]["recipe_id"] == "r1_age_1p125"
