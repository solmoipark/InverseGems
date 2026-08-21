"""Per-request xGEMS call budget (roadmap Phase 2 guardrail).

An agent host attaches a budget to a request; every non-cached solver
invocation consumes one unit. When the budget is exhausted the run stops
with an explicit, machine-readable reason instead of silently burning
compute. Cache hits are free by design - reuse is the point of the
chemistry database.
"""

from __future__ import annotations

from typing import Any


class XGEMSCallBudgetExceeded(RuntimeError):
    def __init__(self, budget: "XGEMSCallBudget", label: str) -> None:
        self.budget = budget
        self.label = label
        super().__init__(
            f"xGEMS call budget exhausted: {budget.spent}/{budget.max_calls} calls used; "
            f"refused call {label!r}. Raise max_xgems_calls or narrow the request."
        )


class XGEMSCallBudget:
    def __init__(self, max_calls: int) -> None:
        if int(max_calls) < 1:
            raise ValueError("max_calls must be a positive integer.")
        self.max_calls = int(max_calls)
        self.spent = 0
        self.history: list[str] = []

    def consume(self, label: str) -> None:
        """Consume one call unit, raising when the budget is exhausted."""
        if self.spent >= self.max_calls:
            raise XGEMSCallBudgetExceeded(self, label)
        self.spent += 1
        self.history.append(str(label))

    @property
    def remaining(self) -> int:
        return max(0, self.max_calls - self.spent)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_calls": self.max_calls,
            "spent": self.spent,
            "remaining": self.remaining,
            "history": list(self.history),
        }
