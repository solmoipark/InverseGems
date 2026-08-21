from inverse_gems.phase_volume_reconstruction import reconstruct_phase_volumes
from inverse_gems.xgems_runner import capture_gems_object_state


class FakeGemsObject:
    phase_names = ["Calcite", "CNASH"]
    phase_masses = {"Calcite": 0.100087, "CNASH": 0.2}
    phases_masses = phase_masses
    phase_amounts = {"Calcite": 1.0, "CNASH": 2.0}
    phase_volumes = {"Calcite": 0.0, "CNASH": 4.0e-5}
    phase_molar_volume = {"Calcite": 0.0, "CNASH": 2.0e-5}
    species_molar_volumes = {"Cal": 3.6934e-5, "CNASH_a": 1.0e-5, "CNASH_b": 3.0e-5}
    species_molar_mass = {"Cal": 0.100087, "CNASH_a": 0.05, "CNASH_b": 0.15}
    species_in_phase = {"Calcite": ["Cal"], "CNASH": ["CNASH_a", "CNASH_b"]}
    system_volume = 7.6934e-5
    system_mass = 0.300087
    pH = 12.5

    def phase_species_moles(self):
        return {
            "Calcite": {"Cal": 1.0},
            "CNASH": {"CNASH_a": 1.0, "CNASH_b": 1.0},
        }


def test_reconstruct_phase_volumes_from_phase_species_moles():
    raw = {
        "phase_masses": {"Calcite": 0.100087},
        "phase_volumes": {"Calcite": 0.0},
        "phase_amounts": {"Calcite": 1.0},
        "phase_molar_volumes": {"Calcite": 0.0},
        "phase_species_moles": {"Calcite": {"Cal": 1.0}},
        "species_molar_volumes": {"Cal": 3.6934e-5},
        "species_molar_masses": {"Cal": 0.100087},
    }

    derived = reconstruct_phase_volumes(raw)

    assert derived["phase_volumes_reconstructed"]["Calcite"] == 3.6934e-5
    report = derived["phase_volume_reconstruction_report"][0]
    assert report["phase"] == "Calcite"
    assert report["phase_mass_matches_species_mass"] is True
    assert report["raw_volume_zero_but_reconstructed_nonzero"] is True
    assert abs(report["density_from_mass_reconstructed_volume_g_cm3"] - 2.71) < 0.01


def test_capture_gems_object_state_includes_reconstructed_volumes():
    raw = capture_gems_object_state(FakeGemsObject())

    assert raw["phase_volumes_reconstructed"]["Calcite"] == 3.6934e-5
    assert raw["phase_volumes_reconstructed"]["CNASH"] == 4.0e-5
    summary = raw["phase_volume_reconstruction_summary"]
    assert summary["raw_volume_zero_but_reconstructed_nonzero_count"] == 1
    assert summary["mass_mismatch_count"] == 0
