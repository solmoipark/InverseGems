from pathlib import Path

from inverse_gems.cached_forward import run_forward_cached
from inverse_gems.database import InverseGemsDatabase
from inverse_gems.porosity import compute_porosity
from inverse_gems.recipe import parse_recipe
from inverse_gems.xgems_runner import MockXGEMSRunner


class WaterSensitiveMockRunner(MockXGEMSRunner):
    def equilibrate(self):
        super().equilibrate()
        water_g = self.species_amounts.get("H2O@", self.species_amounts.get("H2O", 0.0))
        if water_g > 40.0:
            self.solver_status = "Failure (no result) in GEM calculation with LPP AIA"
        return self.solver_status


class LowWaterSensitiveMockRunner(MockXGEMSRunner):
    def equilibrate(self):
        super().equilibrate()
        water_g = self.species_amounts.get("H2O@", self.species_amounts.get("H2O", 0.0))
        if water_g < 30.0:
            self.solver_status = "Failure (no result) in GEM calculation with LPP AIA"
        return self.solver_status


def test_cached_forward_only_calls_mock_runner_once_for_same_chemistry(tmp_path):
    calls = {"count": 0}

    def factory(**kwargs):
        calls["count"] += 1
        return MockXGEMSRunner(**kwargs)

    kwargs = {
        "recipe_text": "OPC 100, w/b 0.4, age 28",
        "db": tmp_path / "db",
        "use_mock": True,
        "runner_factory": factory,
    }
    first = run_forward_cached(**kwargs)
    second = run_forward_cached(**kwargs)
    assert first["chem_hash"] == second["chem_hash"]
    assert second["reused_cache"] is True
    assert calls["count"] == 1


def test_cached_forward_writes_xgems_input_preflight(tmp_path):
    result = run_forward_cached(
        recipe_text="OPC 30, fly ash 70, w/b 0.4, age 28",
        db=tmp_path / "db",
        use_mock=True,
    )

    preflight_dir = result["preflight_dir"]
    assert preflight_dir
    assert Path(preflight_dir).exists()
    preflight_path = Path(preflight_dir) / "xgems_input_preflight.json"
    assert preflight_path.exists()
    assert '"ok": true' in preflight_path.read_text(encoding="utf-8")


def test_same_xgems_solids_can_give_different_porosity_from_source_identity():
    recipe_opc = parse_recipe("OPC 100, w/b 0.4, age 28")
    recipe_fa = parse_recipe("fly ash 100, w/b 0.4, age 28")
    opc = compute_porosity(recipe_opc, xgems_phase_volumes={"solid": 10.0}, unreacted_masses_g={"OPC": 50.0})
    fa = compute_porosity(recipe_fa, xgems_phase_volumes={"solid": 10.0}, unreacted_masses_g={"fly_ash": 50.0})
    assert opc["porosity_best_effort"] != fa["porosity_best_effort"]


def test_cached_forward_adaptive_water_retry_records_rescue(tmp_path):
    def factory(**kwargs):
        return WaterSensitiveMockRunner(**kwargs)

    result = run_forward_cached(
        recipe_text="OPC 100, w/b 0.6, age 28",
        db=tmp_path / "db",
        use_mock=True,
        runner_factory=factory,
        retry_water_on_failure=True,
        retry_water_cap_w_b_ladder=(0.45, 0.40, 0.35),
    )

    assert result["chemistry_status"] == "complete"
    assert result["solver_rescued"] is True
    assert result["xgems_w_b"] == 0.4
    assert result["xgems_retry_count"] == 2
    assert result["primary_chem_hash"] != result["chem_hash"]

    db = InverseGemsDatabase(tmp_path / "db")
    recipe_row = db.get_recipe_run(result["recipe_id"])
    assert recipe_row["solver_rescued"] == 1
    assert recipe_row["xgems_retry_count"] == 2
    statuses = []
    with db.connect() as conn:
        for row in conn.execute("SELECT status FROM chemistry_runs ORDER BY created_at, chem_hash").fetchall():
            statuses.append(row["status"])
    assert statuses.count("failed") == 2
    assert statuses.count("complete") == 1


def test_cached_forward_adaptive_water_retry_can_raise_low_xgems_water(tmp_path):
    def factory(**kwargs):
        return LowWaterSensitiveMockRunner(**kwargs)

    result = run_forward_cached(
        recipe_text="OPC 100, w/b 0.26, age 28",
        db=tmp_path / "db",
        use_mock=True,
        runner_factory=factory,
        retry_water_on_failure=True,
    )

    assert result["chemistry_status"] == "complete"
    assert result["solver_rescued"] is True
    assert result["xgems_w_b"] == 0.3
    assert result["xgems_retry_count"] == 1
    assert [record["label"] for record in result["retry_history"] if record["status"] == "complete"] == [
        "retry_fixed_w_b_0.3"
    ]
