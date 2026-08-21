import pytest

from inverse_gems.cached_forward import run_forward_cached
from inverse_gems.solver_recovery import (
    DiagnosisWaterPolicy,
    LadderWaterPolicy,
    classify_solver_failure,
    resolve_water_recovery_policy,
)
from inverse_gems.xgems_runner import MockXGEMSRunner

FAIL = "Failure (no result) in GEM calculation with LPP AIA"
OK = "OK after GEM calculation with LPP AIA"


def _record(label, status, solver_status, w_b):
    return {
        "label": label,
        "status": status,
        "solver_status": solver_status,
        "xgems_w_b": w_b,
        "xgems_water_g": None if w_b is None else 100.0 * w_b,
    }


def test_classify_solver_failure_categories():
    assert classify_solver_failure(_record("primary", "failed", FAIL, 0.5))["category"] == "no_result"
    assert classify_solver_failure(_record("primary", "failed", "GEM not converged", 0.5))["category"] == "nonconvergence"
    assert classify_solver_failure(_record("primary", "failed", "weird", 0.5))["category"] == "failed_status"


def test_ladder_policy_reproduces_fixed_sequence():
    specs = [{"label": f"retry_cap_w_b_{v:g}", "mode": "cap_w_b", "water_w_b": v} for v in (0.45, 0.40)]
    policy = LadderWaterPolicy(specs)
    history = [_record("primary", "failed", FAIL, 0.6)]
    first = policy.next_attempt(history)
    assert first["label"] == "retry_cap_w_b_0.45"
    history.append(_record(first["label"], "failed", FAIL, 0.45))
    second = policy.next_attempt(history)
    assert second["label"] == "retry_cap_w_b_0.4"
    history.append(_record(second["label"], "failed", FAIL, 0.40))
    assert policy.next_attempt(history) is None


def test_diagnosis_policy_shrinks_then_bisects_downward():
    policy = DiagnosisWaterPolicy(min_w_b=0.30)
    history = [_record("primary", "failed", FAIL, 0.6)]
    first = policy.next_attempt(history)
    assert first["mode"] == "cap_w_b"
    assert first["water_w_b"] == pytest.approx(0.51)
    assert first["diagnosis"]["category"] == "no_result"
    assert first["diagnosis"]["recovery_action"] == "reduce_water"

    history.append(_record(first["label"], "failed", FAIL, 0.51))
    second = policy.next_attempt(history)
    assert second["water_w_b"] == pytest.approx(0.405)

    history.append(_record(second["label"], "failed", FAIL, 0.405))
    third = policy.next_attempt(history)
    assert third["water_w_b"] == pytest.approx(0.3525)


def test_diagnosis_policy_switches_upward_after_downward_exhaustion():
    policy = DiagnosisWaterPolicy(min_w_b=0.30, max_w_b=0.80, max_down_attempts=2)
    history = [_record("primary", "failed", FAIL, 0.6)]
    for _ in range(2):
        spec = policy.next_attempt(history)
        assert spec["mode"] == "cap_w_b"
        history.append(_record(spec["label"], "failed", FAIL, spec["water_w_b"]))
    up = policy.next_attempt(history)
    assert up["mode"] == "fixed_w_b"
    assert up["water_w_b"] == pytest.approx(0.65)
    assert up["diagnosis"]["recovery_action"] == "raise_water"


def test_diagnosis_policy_raises_water_when_primary_below_floor():
    policy = DiagnosisWaterPolicy(min_w_b=0.30)
    history = [_record("primary", "failed", FAIL, 0.26)]
    spec = policy.next_attempt(history)
    assert spec["mode"] == "fixed_w_b"
    assert spec["water_w_b"] == pytest.approx(0.30)


def test_diagnosis_policy_respects_max_retries_and_max_w_b():
    policy = DiagnosisWaterPolicy(min_w_b=0.30, max_w_b=0.35, max_retries=10)
    history = [_record("primary", "failed", FAIL, 0.26)]
    spec = policy.next_attempt(history)
    history.append(_record(spec["label"], "failed", FAIL, spec["water_w_b"]))
    second = policy.next_attempt(history)
    assert second["water_w_b"] == pytest.approx(0.35)
    history.append(_record(second["label"], "failed", FAIL, second["water_w_b"]))
    assert policy.next_attempt(history) is None  # above max_w_b

    capped = DiagnosisWaterPolicy(max_retries=1)
    hist = [_record("primary", "failed", FAIL, 0.6)]
    spec = capped.next_attempt(hist)
    hist.append(_record(spec["label"], "failed", FAIL, spec["water_w_b"]))
    assert capped.next_attempt(hist) is None


def test_resolve_policy_names():
    assert isinstance(resolve_water_recovery_policy("ladder", ladder_specs=[], min_w_b=0.3, max_retries=6), LadderWaterPolicy)
    assert isinstance(resolve_water_recovery_policy("diagnosis", ladder_specs=[], min_w_b=0.3, max_retries=6), DiagnosisWaterPolicy)
    with pytest.raises(ValueError):
        resolve_water_recovery_policy("nope", ladder_specs=[], min_w_b=0.3, max_retries=6)


class WaterSensitiveMockRunner(MockXGEMSRunner):
    """Fails whenever xGEMS water exceeds 40 g (per 100 g binder)."""

    def equilibrate(self):
        super().equilibrate()
        water_g = self.species_amounts.get("H2O@", self.species_amounts.get("H2O", 0.0))
        if water_g > 40.0:
            self.solver_status = FAIL
        return self.solver_status


class LowWaterSensitiveMockRunner(MockXGEMSRunner):
    """Fails whenever xGEMS water is below 30 g (per 100 g binder)."""

    def equilibrate(self):
        super().equilibrate()
        water_g = self.species_amounts.get("H2O@", self.species_amounts.get("H2O", 0.0))
        if water_g < 30.0:
            self.solver_status = FAIL
        return self.solver_status


def test_cached_forward_diagnosis_policy_rescues_high_water(tmp_path):
    result = run_forward_cached(
        recipe_text="OPC 100, w/b 0.6, age 28",
        db=tmp_path / "db",
        use_mock=True,
        runner_factory=lambda **kwargs: WaterSensitiveMockRunner(**kwargs),
        retry_water_on_failure=True,
        retry_water_policy="diagnosis",
    )
    assert result["chemistry_status"] == "complete"
    assert result["solver_rescued"] is True
    # 0.6 -> 0.51 (fail) -> 0.405 (fail) -> 0.3525 (ok)
    assert result["xgems_w_b"] == pytest.approx(0.3525)
    assert result["xgems_retry_count"] == 3
    diagnosed = [r for r in result["retry_history"] if r.get("diagnosis")]
    assert diagnosed
    assert diagnosed[0]["diagnosis"]["category"] == "no_result"
    assert diagnosed[0]["diagnosis"]["recovery_action"] == "reduce_water"


def test_cached_forward_diagnosis_policy_rescues_low_water(tmp_path):
    result = run_forward_cached(
        recipe_text="OPC 100, w/b 0.26, age 28",
        db=tmp_path / "db",
        use_mock=True,
        runner_factory=lambda **kwargs: LowWaterSensitiveMockRunner(**kwargs),
        retry_water_on_failure=True,
        retry_water_policy="diagnosis",
    )
    assert result["chemistry_status"] == "complete"
    assert result["solver_rescued"] is True
    assert result["xgems_w_b"] == pytest.approx(0.30)
    assert result["retry_history"][-1]["diagnosis"]["recovery_action"] == "raise_water"
