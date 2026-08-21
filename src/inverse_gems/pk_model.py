from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.integrate import solve_ivp


PHASES = ("C3S", "C2S", "C3A", "C4AF")

K1 = {"C3S": 1.5, "C2S": 0.5, "C3A": 1.0, "C4AF": 0.37}
N1 = {"C3S": 0.7, "C2S": 1.0, "C3A": 0.85, "C4AF": 0.7}
K2 = {"C3S": 0.05, "C2S": 0.02, "C3A": 0.04, "C4AF": 0.015}
K3 = {"C3S": 1.1, "C2S": 0.7, "C3A": 1.0, "C4AF": 0.4}
N3 = {"C3S": 3.3, "C2S": 5.0, "C3A": 3.2, "C4AF": 3.7}
H = {"C3S": 1.8, "C2S": 1.35, "C3A": 1.6, "C4AF": 1.45}
Ea = {"C3S": 41570.0, "C2S": 20785.0, "C3A": 54040.0, "C4AF": 34087.0}
T0 = 20.0
ref_fineness = 385.0


@dataclass(frozen=True)
class PKParameters:
    K1: dict[str, float]
    N1: dict[str, float]
    K2: dict[str, float]
    K3: dict[str, float]
    N3: dict[str, float]
    H: dict[str, float]
    Ea: dict[str, float]
    T0: float
    ref_fineness: float

    def to_dict(self) -> dict[str, object]:
        return {
            "K1": dict(self.K1),
            "N1": dict(self.N1),
            "K2": dict(self.K2),
            "K3": dict(self.K3),
            "N3": dict(self.N3),
            "H": dict(self.H),
            "Ea": dict(self.Ea),
            "T0": self.T0,
            "ref_fineness": self.ref_fineness,
        }


def default_pk_parameters() -> PKParameters:
    return PKParameters(
        K1={phase: float(K1[phase]) for phase in PHASES},
        N1={phase: float(N1[phase]) for phase in PHASES},
        K2={phase: float(K2[phase]) for phase in PHASES},
        K3={phase: float(K3[phase]) for phase in PHASES},
        N3={phase: float(N3[phase]) for phase in PHASES},
        H={phase: float(H[phase]) for phase in PHASES},
        Ea={phase: float(Ea[phase]) for phase in PHASES},
        T0=float(T0),
        ref_fineness=float(ref_fineness),
    )


def _phase_mapping(name: str, values: dict[str, object], base: dict[str, float]) -> dict[str, float]:
    merged = dict(base)
    for phase, value in values.items():
        if phase not in PHASES:
            raise ValueError(f"Unknown PK phase '{phase}' in {name}; expected one of {', '.join(PHASES)}.")
        merged[phase] = float(value)
    return merged


def pk_parameters_from_dict(data: PKParameters | dict[str, object] | None = None) -> PKParameters:
    if data is None:
        return default_pk_parameters()
    if isinstance(data, PKParameters):
        return data
    base = default_pk_parameters()
    base_dict = base.to_dict()
    return PKParameters(
        K1=_phase_mapping("K1", dict(data.get("K1") or {}), base.K1),
        N1=_phase_mapping("N1", dict(data.get("N1") or {}), base.N1),
        K2=_phase_mapping("K2", dict(data.get("K2") or {}), base.K2),
        K3=_phase_mapping("K3", dict(data.get("K3") or {}), base.K3),
        N3=_phase_mapping("N3", dict(data.get("N3") or {}), base.N3),
        H=_phase_mapping("H", dict(data.get("H") or {}), base.H),
        Ea=_phase_mapping("Ea", dict(data.get("Ea") or {}), base.Ea),
        T0=float(data.get("T0", base_dict["T0"])),
        ref_fineness=float(data.get("ref_fineness", base_dict["ref_fineness"])),
    )


def nuclandgrowth(alpha_t: float, k1: float, n1: float, fineness: float, reference_fineness: float) -> float:
    alpha = min(max(float(alpha_t), 1.0e-15), 0.999999999)
    return (k1 / n1) * (1.0 - alpha) * (-np.log(1.0 - alpha)) ** (1.0 - n1) * (fineness / reference_fineness)


def diffusion(alpha_t: float, k2: float) -> float:
    alpha = min(max(float(alpha_t), 1.0e-15), 0.999999999)
    return k2 * (1.0 - alpha) ** (2.0 / 3.0) / alpha ** (1.0 / 3.0)


def hydrationshell(alpha_t: float, k3: float, n3: float) -> float:
    alpha = min(max(float(alpha_t), 0.0), 0.999999999)
    return k3 * (1.0 - alpha) ** n3


def f_wc_lothenbach(wc: float, h: float, alpha_t: float) -> float:
    alpha_cr = h * wc
    if alpha_t > alpha_cr:
        return (1.0 + 3.333 * (h * wc - alpha_t)) ** 4
    return 1.0


def f_rh(RH: float) -> float:
    if RH <= 0.55:
        return 0.0
    return ((RH - 0.55) / 0.45) ** 4


def f_temp(T_celsius: float, T0_celsius: float, ea: float) -> float:
    gas_constant = 8.314
    T_kelvin = T_celsius + 273.15
    T0_kelvin = T0_celsius + 273.15
    return float(np.exp((-ea / gas_constant) * ((1.0 / T_kelvin) - (1.0 / T0_kelvin))))


def overall_rate(
    _t: float,
    alpha_t: Iterable[float] | float,
    k1: float,
    n1: float,
    k2: float,
    k3: float,
    n3: float,
    h: float,
    wc: float,
    RH: float,
    fineness: float,
    reference_fineness: float,
    T_celsius: float,
    T0_celsius: float,
    ea: float,
) -> list[float]:
    alpha = float(np.asarray(alpha_t).reshape(-1)[0])
    alpha = min(max(alpha, 1.0e-15), 0.999999999)
    r1 = nuclandgrowth(alpha, k1, n1, fineness, reference_fineness)
    r2 = diffusion(alpha, k2)
    r3 = hydrationshell(alpha, k3, n3)
    rate = min(r1, r2, r3)
    rate *= f_wc_lothenbach(wc, h, alpha) * f_rh(RH) * f_temp(T_celsius, T0_celsius, ea)
    if not np.isfinite(rate):
        rate = 0.0
    return [max(float(rate), 0.0)]


def _prepare_ages(age_days: float | list[float] | tuple[float, ...] | np.ndarray) -> tuple[bool, np.ndarray, np.ndarray]:
    scalar = np.isscalar(age_days)
    ages = np.asarray([age_days] if scalar else age_days, dtype=float)
    if ages.ndim != 1 or len(ages) == 0:
        raise ValueError("age_days must be a positive scalar or one-dimensional list of ages.")
    if np.any(ages < 0):
        raise ValueError("age_days cannot contain negative values.")
    sorted_unique = np.unique(ages)
    return bool(scalar), ages, sorted_unique


def parrot_killoh(
    age_days: float | list[float],
    wc: float,
    RH: float = 1.0,
    T_celsius: float = 20.0,
    fineness_m2_kg: float = ref_fineness,
    parameters: PKParameters | dict[str, object] | None = None,
) -> dict[str, float | list[float]]:
    params = pk_parameters_from_dict(parameters)
    scalar, requested_ages, solve_ages = _prepare_ages(age_days)
    max_age = float(np.max(solve_ages))
    out: dict[str, float | list[float]] = {}

    for phase in PHASES:
        if max_age == 0.0:
            solved_values = np.zeros_like(solve_ages, dtype=float)
        else:
            solution = solve_ivp(
                overall_rate,
                (0.0, max_age),
                [1.0e-15],
                t_eval=solve_ages,
                args=(
                    params.K1[phase],
                    params.N1[phase],
                    params.K2[phase],
                    params.K3[phase],
                    params.N3[phase],
                    params.H[phase],
                    wc,
                    RH,
                    fineness_m2_kg,
                    params.ref_fineness,
                    T_celsius,
                    params.T0,
                    params.Ea[phase],
                ),
                rtol=1.0e-6,
                atol=1.0e-10,
            )
            if not solution.success:
                raise RuntimeError(f"Parrot-Killoh integration failed for {phase}: {solution.message}")
            solved_values = np.clip(np.asarray(solution.y[0], dtype=float), 0.0, 1.0)

        age_to_value = {float(age): float(value) for age, value in zip(solve_ages, solved_values)}
        values = [age_to_value[float(age)] for age in requested_ages]
        out[phase] = values[0] if scalar else values
    return out
