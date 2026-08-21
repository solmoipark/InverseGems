from inverse_gems.scm_reaction import SCMLogisticParameters, load_scm_parameters, scm_alpha


def test_scm_alpha_increases_with_age_for_defaults():
    params = load_scm_parameters()["fly_ash"]
    values = scm_alpha([7, 28, 90, 365], params)
    assert values == sorted(values)


def test_scm_alpha_clips_to_unit_interval():
    params = SCMLogisticParameters(A=-1.0, B=1.0, C=1.0, D=2.0, G=1.0)
    values = scm_alpha([0, 1, 100], params)
    assert all(0.0 <= value <= 1.0 for value in values)


def test_scm_alpha_reasonable_approximate_values():
    params = load_scm_parameters()["slag"]
    values = scm_alpha([7, 28, 90, 365], params)
    assert 0.10 < values[0] < 0.30
    assert 0.30 < values[1] < 0.40
    assert 0.40 < values[2] < 0.50
    assert 0.48 < values[3] < 0.55
