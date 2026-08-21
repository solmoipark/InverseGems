import json

import pandas as pd

from inverse_gems.xgems_output_capture import EXPECTED_OUTPUT_FILES, save_run_outputs


def test_output_capture_creates_all_expected_files_and_preserves_phase_names(tmp_path):
    run_dir = save_run_outputs(
        out_base=tmp_path,
        manifest={"command": "test"},
        user_request="OPC 100, w/b 0.4, age 28",
        recipe={"binder_masses_g": {"OPC": 100}},
        materials_used={},
        reaction_degrees={},
        xgems_species_amounts={"C3S": 0.01},
        raw_state={
            "phase_masses": {"raw phase/name 1": 1.2},
            "phase_volumes": {"raw phase/name 1": 0.5},
            "aqueous_species": {"Ca+2": 1e-6},
            "scalars": {"pH": 12.5},
            "attribute_report": {"missing_requested_attributes": []},
        },
        porosity={"porosity_best_effort": 0.3},
        warnings=[],
    )
    for filename in EXPECTED_OUTPUT_FILES:
        assert (run_dir / filename).exists()
    phase_csv = pd.read_csv(run_dir / "xgems_phase_amounts_raw.csv")
    assert phase_csv.loc[0, "name"] == "raw phase/name 1"
    with (run_dir / "manifest.json").open(encoding="utf-8") as handle:
        assert json.load(handle)["command"] == "test"
