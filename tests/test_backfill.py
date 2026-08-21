import csv
from pathlib import Path

from inverse_gems.backfill import RECONSTRUCTION_FILES, backfill_reconstructed_volumes_and_porosity
from inverse_gems.cached_forward import run_forward_cached
from inverse_gems.database import InverseGemsDatabase, read_name_value_csv


class ReconstructingRunner:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.species_amounts = {}

    def add_species_amounts(self, species_amounts, units="kg"):
        self.species_amounts = dict(species_amounts)

    def equilibrate(self):
        return "OK after GEM calculation with LPP AIA"

    def capture_raw_state(self):
        return {
            "phase_masses": {"Calcite": 10.0},
            "phase_volumes": {"Calcite": 0.0},
            "phase_amounts": {"Calcite": 1.0},
            "phase_molar_volumes": {"Calcite": 0.0},
            "phase_species_moles": {"Calcite": {"Cal": 1.0}},
            "species_molar_volumes": {"Cal": 5.0},
            "species_molar_masses": {"Cal": 10.0},
            "aqueous_species": {},
            "scalars": {},
            "attribute_report": {"attribute_names": [], "missing_requested_attributes": []},
        }


def test_backfill_reconstructed_volumes_updates_recipe_porosity(tmp_path):
    db_dir = tmp_path / "db"
    result = run_forward_cached(recipe_text="OPC 100, w/b 0.4, age 28", db=db_dir, use_mock=True)
    db = InverseGemsDatabase(db_dir)
    chem = db.get_chemistry_run(result["chem_hash"])
    raw_dir = Path(chem["xgems_run_dir"])
    chem_dir = db.chemistry_runs_dir / result["chem_hash"]
    for folder in (raw_dir, chem_dir):
        for name in RECONSTRUCTION_FILES:
            path = folder / name
            if path.exists():
                path.unlink()

    recipe_before = db.get_recipe_run(result["recipe_id"])
    dat_lst = tmp_path / "dummy.dat.lst"
    dat_lst.write_text("dummy\n", encoding="utf-8")

    summary = backfill_reconstructed_volumes_and_porosity(
        db=db_dir,
        dat_lst=dat_lst,
        limit=1,
        xgems_phase_volume_unit="cm3",
        runner_factory=lambda **kwargs: ReconstructingRunner(**kwargs),
    )

    assert summary["chemistry_backfill"]["complete"] == 1
    assert summary["porosity_recompute"]["updated"] == 1
    assert (raw_dir / "xgems_phase_volumes_reconstructed.csv").exists()
    assert read_name_value_csv(raw_dir / "xgems_phase_volumes_reconstructed.csv")["Calcite"] == 5.0

    recipe_after = db.get_recipe_run(result["recipe_id"])
    assert recipe_after["porosity"] != recipe_before["porosity"]
    assert recipe_after["final_solid_volume_cm3"] > recipe_before["final_solid_volume_cm3"]

    with (db_dir / "backfill_reconstructed_volumes_status.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[-1]["status"] == "complete"
