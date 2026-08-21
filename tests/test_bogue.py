from inverse_gems.bogue import calculate_bogue
from inverse_gems.materials import load_materials


def test_bogue_returns_expected_keys():
    opc = load_materials()["OPC"]
    result = calculate_bogue(opc.oxide_mass_percent)
    assert set(result) == {"C3S", "C2S", "C3A", "C4AF"}
    assert all(value >= 0 for value in result.values())
