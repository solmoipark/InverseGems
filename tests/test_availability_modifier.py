from inverse_gems.availability_modifier import C3S_C2SAvailabilityModifier
from inverse_gems.bogue import bogue_phases
from inverse_gems.materials import load_materials
from inverse_gems.scm_reaction import load_scm_parameters


def _opc_phases():
    opc = load_materials()["OPC"]
    return bogue_phases(opc.oxide_mass_percent).phase_mass_percent


def test_lower_opc_content_reduces_d_eff():
    modifier = C3S_C2SAvailabilityModifier()
    params = load_scm_parameters()
    phases = _opc_phases()
    low = modifier.modify_parameters({"OPC": 20, "fly_ash": 80}, phases, params)
    high = modifier.modify_parameters({"OPC": 60, "fly_ash": 40}, phases, params)
    assert low.parameters["fly_ash"].D < high.parameters["fly_ash"].D


def test_higher_opc_increases_or_preserves_d_eff():
    modifier = C3S_C2SAvailabilityModifier()
    params = load_scm_parameters()
    phases = _opc_phases()
    medium = modifier.modify_parameters({"OPC": 50, "slag": 50}, phases, params)
    high = modifier.modify_parameters({"OPC": 80, "slag": 20}, phases, params)
    assert high.parameters["slag"].D >= medium.parameters["slag"].D


def test_slag_is_less_sensitive_than_silica_fume():
    modifier = C3S_C2SAvailabilityModifier()
    params = load_scm_parameters()
    phases = _opc_phases()
    slag = modifier.modify_parameters({"OPC": 30, "slag": 70}, phases, params)
    sf = modifier.modify_parameters({"OPC": 30, "silica_fume": 70}, phases, params)
    slag_ratio = slag.parameters["slag"].D / params["slag"].D
    sf_ratio = sf.parameters["silica_fume"].D / params["silica_fume"].D
    assert slag_ratio > sf_ratio


def test_no_scm_returns_no_modification():
    modifier = C3S_C2SAvailabilityModifier()
    params = load_scm_parameters()
    result = modifier.modify_parameters({"OPC": 100}, _opc_phases(), params)
    assert result.metadata["applied"] is False
    assert result.parameters["slag"].D == params["slag"].D


def test_opc_zero_gives_d_eff_near_zero_for_pozzolanic_scms():
    modifier = C3S_C2SAvailabilityModifier()
    params = load_scm_parameters()
    result = modifier.modify_parameters({"OPC": 0, "silica_fume": 100}, _opc_phases(), params)
    assert result.parameters["silica_fume"].D == 0.0
