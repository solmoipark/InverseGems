from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from .materials import SCM_NAMES
from .utils import config_path, load_yaml


# ---------------------------------------------------------------------------
# Kinetics model registry
#
# The SCM reaction equation is a swappable strategy. A kinetics model is a
# function alpha(t, params) -> array, registered under a name that config
# files reference via a per-material ``model:`` key. The default remains the
# five-parameter logistic; a new equation only needs a registered function
# and updated YAML - no caller changes, and reaction provenance (to_dict)
# captures the model name so signatures/hashes distinguish equations.
# ---------------------------------------------------------------------------

KineticsFn = Callable[[np.ndarray, Mapping[str, float]], np.ndarray]

DEFAULT_KINETICS_MODEL = "five_param_logistic"


@dataclass(frozen=True)
class SCMKineticsModel:
    name: str
    fn: KineticsFn
    required: tuple[str, ...]
    # Parameter acting as the long-term reaction asymptote; the C3S/C2S
    # availability modifier caps this value via with_D().
    asymptote_key: str


_KINETICS_REGISTRY: dict[str, SCMKineticsModel] = {}


def register_scm_kinetics(
    name: str,
    *,
    required: tuple[str, ...] | list[str],
    asymptote_key: str,
) -> Callable[[KineticsFn], KineticsFn]:
    def decorator(fn: KineticsFn) -> KineticsFn:
        _KINETICS_REGISTRY[name] = SCMKineticsModel(
            name=name, fn=fn, required=tuple(required), asymptote_key=asymptote_key
        )
        return fn

    return decorator


def get_scm_kinetics_model(name: str) -> SCMKineticsModel:
    if name not in _KINETICS_REGISTRY:
        raise KeyError(
            f"Unknown SCM kinetics model '{name}'. Registered models: {sorted(_KINETICS_REGISTRY)}"
        )
    return _KINETICS_REGISTRY[name]


def registered_scm_kinetics_models() -> list[str]:
    return sorted(_KINETICS_REGISTRY)


@register_scm_kinetics(DEFAULT_KINETICS_MODEL, required=("A", "B", "C", "D", "G"), asymptote_key="D")
def _five_param_logistic(t: np.ndarray, p: Mapping[str, float]) -> np.ndarray:
    if p["C"] <= 0:
        raise ValueError("SCM logistic parameter C must be positive.")
    return p["D"] + (p["A"] - p["D"]) / (1.0 + (t / p["C"]) ** p["B"]) ** p["G"]


# ---------------------------------------------------------------------------
# Parameter containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SCMLogisticParameters:
    """Parameters of the default five-parameter logistic model.

    Kept as a dedicated container for backward compatibility: its to_dict()
    stays flat (no ``model`` key), so existing reaction signatures and
    chem_hash payloads are unchanged for the default equation.
    """

    A: float
    B: float
    C: float
    D: float
    G: float

    model: str = field(default=DEFAULT_KINETICS_MODEL, init=False, repr=False, compare=False)

    def to_dict(self) -> dict[str, float]:
        return {"A": self.A, "B": self.B, "C": self.C, "D": self.D, "G": self.G}

    def param_values(self) -> dict[str, float]:
        return self.to_dict()

    def with_D(self, D: float) -> "SCMLogisticParameters":
        return replace(self, D=float(D))


@dataclass(frozen=True)
class SCMKineticsParameters:
    """Generic parameter container for any registered kinetics model."""

    model: str
    values: dict[str, float]

    def __post_init__(self) -> None:
        spec = get_scm_kinetics_model(self.model)
        missing = [key for key in spec.required if key not in self.values]
        if missing:
            raise ValueError(
                f"SCM kinetics model '{self.model}' is missing required parameter(s): {missing}"
            )

    def __getattr__(self, item: str) -> float:
        values = object.__getattribute__(self, "values")
        if item in values:
            return values[item]
        raise AttributeError(item)

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, **self.values}

    def param_values(self) -> dict[str, float]:
        return dict(self.values)

    def with_D(self, D: float) -> "SCMKineticsParameters":
        """Cap the model's asymptote parameter (availability-modifier hook)."""
        spec = get_scm_kinetics_model(self.model)
        values = dict(self.values)
        values[spec.asymptote_key] = float(D)
        return SCMKineticsParameters(model=self.model, values=values)

    @property
    def D(self) -> float:  # asymptote alias used by the availability modifier
        spec = get_scm_kinetics_model(self.model)
        return float(self.values[spec.asymptote_key])


SCMParameters = SCMLogisticParameters | SCMKineticsParameters


def scm_kinetics_from_mapping(values: Mapping[str, Any]) -> SCMParameters:
    """Build a parameter container from a config mapping.

    Flat ``{A..G}`` mappings map to the default logistic (legacy format);
    mappings with a ``model:`` key select a registered kinetics model.
    """
    data = dict(values)
    model = str(data.pop("model", DEFAULT_KINETICS_MODEL))
    numeric = {key: float(value) for key, value in data.items()}
    if model == DEFAULT_KINETICS_MODEL and set(numeric) == {"A", "B", "C", "D", "G"}:
        return SCMLogisticParameters(**numeric)
    return SCMKineticsParameters(model=model, values=numeric)


def load_scm_parameters(path: str | Path | None = None) -> dict[str, SCMParameters]:
    raw = load_yaml(path or config_path("scm_reaction.yaml"))
    return {name: scm_kinetics_from_mapping(values) for name, values in raw.items()}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def scm_alpha(
    age_days: float | list[float] | np.ndarray,
    params: SCMParameters | Mapping[str, float],
) -> float | list[float]:
    if isinstance(params, Mapping):
        params = scm_kinetics_from_mapping(params)

    spec = get_scm_kinetics_model(params.model)
    values = params.param_values()

    scalar = np.isscalar(age_days)
    t = np.asarray([age_days] if scalar else age_days, dtype=float)
    if np.any(t < 0):
        raise ValueError("SCM reaction age cannot be negative.")

    alpha = np.asarray(spec.fn(t, values), dtype=float)
    alpha = np.clip(alpha, 0.0, 1.0)
    return float(alpha[0]) if scalar else [float(v) for v in alpha]


def scm_reaction_degrees(
    age_days: float,
    scm_masses_g: dict[str, float],
    params: dict[str, SCMParameters] | None = None,
) -> dict[str, float]:
    params = params or load_scm_parameters()
    result: dict[str, float] = {}
    for name, mass in scm_masses_g.items():
        if name in SCM_NAMES and mass > 0:
            result[name] = float(scm_alpha(age_days, params[name]))
    return result


def parameters_to_dict(params: dict[str, SCMParameters]) -> dict[str, Any]:
    return {name: values.to_dict() for name, values in params.items()}
