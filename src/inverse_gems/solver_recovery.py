"""Scenario (b) of the agent roadmap: diagnosis-driven xGEMS solver recovery.

The cached forward runner retries failed equilibrium calculations by varying
the xGEMS water condition. Historically the retry schedule was a fixed
ladder; this module turns it into a swappable policy:

- :class:`LadderWaterPolicy` reproduces the fixed ladder exactly (default).
- :class:`DiagnosisWaterPolicy` inspects the failure history (solver status,
  water condition of each attempt) and proposes the next attempt adaptively:
  shrink-then-bisect the water downward, or raise it when the primary water
  was already at/below the floor. Each proposal carries the failure
  diagnosis, so run artifacts record *why* a retry was chosen.

Policies are deterministic and unit-testable; an agent host selects a policy
but never invents water values.
"""

from __future__ import annotations

from typing import Any, Protocol

_SKIP_STATUSES = {"skipped_duplicate"}


def classify_solver_failure(record: dict[str, Any]) -> dict[str, Any]:
    """Classify one failed attempt record from its solver status text."""
    status = str(record.get("solver_status") or "")
    lowered = status.lower()
    if "no result" in lowered or "failure" in lowered:
        category = "no_result"
    elif "not converged" in lowered or "diverg" in lowered or "max iter" in lowered:
        category = "nonconvergence"
    elif str(record.get("status") or "") == "failed":
        category = "failed_status"
    else:
        category = "unknown"
    return {
        "category": category,
        "solver_status": status,
        "attempt_label": record.get("label"),
        "xgems_w_b": record.get("xgems_w_b"),
        "xgems_water_g": record.get("xgems_water_g"),
    }


class WaterRecoveryPolicy(Protocol):
    def next_attempt(self, attempt_history: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Return the next attempt spec, or None to stop retrying."""


class LadderWaterPolicy:
    """Fixed retry schedule; reproduces the legacy ladder behavior exactly."""

    def __init__(self, specs: list[dict[str, Any]]):
        self._specs = list(specs)

    def next_attempt(self, attempt_history: list[dict[str, Any]]) -> dict[str, Any] | None:
        proposals_made = max(0, len(attempt_history) - 1)
        if proposals_made >= len(self._specs):
            return None
        return dict(self._specs[proposals_made])


class DiagnosisWaterPolicy:
    """Adaptive water recovery driven by the failure history.

    Downward phase (suspected excess water): first cap at ``shrink`` times the
    primary equilibrium w/b, then bisect toward ``min_w_b``. When the primary
    water already sits at or below the floor, the downward phase is skipped.
    Upward phase (suspected water starvation): raise the water in fixed steps
    from max(primary, floor) up to ``max_w_b``.
    """

    def __init__(
        self,
        *,
        min_w_b: float = 0.30,
        max_w_b: float = 0.80,
        max_retries: int = 6,
        max_down_attempts: int = 3,
        shrink: float = 0.85,
        up_step: float = 0.05,
        resolution: float = 0.01,
    ) -> None:
        self.min_w_b = float(min_w_b)
        self.max_w_b = float(max_w_b)
        self.max_retries = int(max_retries)
        self.max_down_attempts = int(max_down_attempts)
        self.shrink = float(shrink)
        self.up_step = float(up_step)
        self.resolution = float(resolution)

    def next_attempt(self, attempt_history: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not attempt_history:
            return None
        proposals_made = len(attempt_history) - 1
        if proposals_made >= self.max_retries:
            return None

        primary = attempt_history[0]
        executed = [r for r in attempt_history if str(r.get("status")) not in _SKIP_STATUSES]
        last_failed = executed[-1] if executed else primary
        diagnosis = classify_solver_failure(last_failed)

        primary_w_b = _to_float(primary.get("xgems_w_b"))
        down_records = [r for r in attempt_history if str(r.get("label", "")).startswith("retry_cap_w_b_")]
        up_records = [r for r in attempt_history if str(r.get("label", "")).startswith("retry_fixed_w_b_")]

        down_exhausted = (
            primary_w_b is None
            or primary_w_b <= self.min_w_b + self.resolution
            or len(down_records) >= self.max_down_attempts
        )
        if not down_exhausted:
            tried = [w for w in (_to_float(r.get("xgems_w_b")) for r in down_records) if w is not None]
            lowest = min(tried) if tried else primary_w_b
            if not down_records:
                candidate = max(self.min_w_b, primary_w_b * self.shrink)
            else:
                candidate = max(self.min_w_b, (self.min_w_b + lowest) / 2.0)
            if lowest - candidate >= self.resolution:
                return self._spec("cap_w_b", "retry_cap_w_b", candidate, diagnosis, "reduce_water")
            down_exhausted = True

        if down_exhausted:
            base = max(self.min_w_b, primary_w_b if primary_w_b is not None else self.min_w_b)
            tried_up = [w for w in (_to_float(r.get("xgems_w_b")) for r in up_records) if w is not None]
            if not tried_up:
                candidate = base if primary_w_b is not None and primary_w_b < self.min_w_b else base + self.up_step
                candidate = max(self.min_w_b, candidate)
            else:
                candidate = max(tried_up) + self.up_step
            if candidate > self.max_w_b + 1.0e-9:
                return None
            return self._spec("fixed_w_b", "retry_fixed_w_b", candidate, diagnosis, "raise_water")
        return None

    @staticmethod
    def _spec(
        mode: str, prefix: str, w_b: float, diagnosis: dict[str, Any], action: str
    ) -> dict[str, Any]:
        w_b = round(float(w_b), 4)
        return {
            "label": f"{prefix}_{w_b:g}",
            "mode": mode,
            "factor": 1.0,
            "water_g": None,
            "water_w_b": w_b,
            "diagnosis": {**diagnosis, "recovery_action": action},
        }


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_water_recovery_policy(
    policy: str | WaterRecoveryPolicy,
    *,
    ladder_specs: list[dict[str, Any]],
    min_w_b: float,
    max_retries: int,
) -> WaterRecoveryPolicy:
    """Map a policy name (or pass through a policy object) to an instance."""
    if not isinstance(policy, str):
        return policy
    if policy == "ladder":
        return LadderWaterPolicy(ladder_specs)
    if policy == "diagnosis":
        return DiagnosisWaterPolicy(min_w_b=min_w_b, max_retries=max_retries)
    raise ValueError(f"Unknown water recovery policy {policy!r}; expected 'ladder' or 'diagnosis'.")
