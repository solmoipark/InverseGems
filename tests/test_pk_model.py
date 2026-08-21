from inverse_gems.pk_model import parrot_killoh


def test_pk_model_accepts_single_age_and_bounds_values():
    result = parrot_killoh(28, wc=0.4)
    assert set(result) == {"C3S", "C2S", "C3A", "C4AF"}
    assert all(0.0 <= value <= 1.0 for value in result.values())


def test_pk_model_accepts_multiple_ages_and_bounds_values():
    result = parrot_killoh([1, 7, 28], wc=0.4)
    assert set(result) == {"C3S", "C2S", "C3A", "C4AF"}
    for values in result.values():
        assert len(values) == 3
        assert all(0.0 <= value <= 1.0 for value in values)
