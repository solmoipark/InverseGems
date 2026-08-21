from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any


PHASE_KEYS = ("C3S", "C2S", "C3A", "C4AF")


@dataclass
class BogueResult:
    phase_mass_percent: dict[str, float]
    raw_bogue: dict[str, float]
    warnings: list[str]
    normalized: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase_mass_percent": dict(self.phase_mass_percent),
            "raw_bogue": dict(self.raw_bogue),
            "warnings": list(self.warnings),
            "normalized": self.normalized,
        }


def bogue_phases(
    oxide_mass_percent: dict[str, float],
    *,
    clip_negative: bool = True,
    normalize_if_unrealistic: bool = False,
    unrealistic_upper_sum: float = 105.0,
) -> BogueResult:
    CaO = float(oxide_mass_percent.get("CaO", 0.0))
    SiO2 = float(oxide_mass_percent.get("SiO2", 0.0))
    Al2O3 = float(oxide_mass_percent.get("Al2O3", 0.0))
    Fe2O3 = float(oxide_mass_percent.get("Fe2O3", 0.0))
    SO3 = float(oxide_mass_percent.get("SO3", 0.0))

    raw = {
        "C3S": 4.07 * CaO - 7.6 * SiO2 - 6.72 * Al2O3 - 1.43 * Fe2O3 - 2.85 * SO3,
        "C2S": 0.0,
        "C3A": 2.65 * Al2O3 - 1.69 * Fe2O3,
        "C4AF": 3.04 * Fe2O3,
    }
    raw["C2S"] = 2.87 * SiO2 - 0.75 * raw["C3S"]

    messages: list[str] = []
    phases = dict(raw)
    if clip_negative:
        for key, value in list(phases.items()):
            if value < 0:
                message = f"Bogue phase {key} was negative ({value:g} mass-%); clipped to zero."
                warnings.warn(message, RuntimeWarning, stacklevel=2)
                messages.append(message)
                phases[key] = 0.0

    normalized = False
    phase_sum = sum(phases.values())
    if normalize_if_unrealistic and phase_sum > unrealistic_upper_sum and phase_sum > 0:
        scale = 100.0 / phase_sum
        phases = {key: value * scale for key, value in phases.items()}
        normalized = True
        messages.append(f"Bogue clinker phase sum {phase_sum:g} mass-% normalized to 100 mass-%.")

    return BogueResult(
        phase_mass_percent={key: float(phases.get(key, 0.0)) for key in PHASE_KEYS},
        raw_bogue={key: float(raw.get(key, 0.0)) for key in PHASE_KEYS},
        warnings=messages,
        normalized=normalized,
    )


def calculate_bogue(oxide_mass_percent: dict[str, float] | None = None, **oxides: float) -> dict[str, float]:
    data = dict(oxide_mass_percent or {})
    data.update({key: value for key, value in oxides.items() if value is not None})
    return bogue_phases(data).phase_mass_percent


def bouges_composition(CaO: float, SiO2: float, Al2O3: float, Fe2O3: float, SO3: float) -> dict[str, float]:
    return calculate_bogue({"CaO": CaO, "SiO2": SiO2, "Al2O3": Al2O3, "Fe2O3": Fe2O3, "SO3": SO3})
