from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import config_path, load_yaml


DEFAULT_AGE_PRESET = "early_dense_v1"


@dataclass(frozen=True)
class AgeMetadata:
    age_days: float
    age_hours: float
    age_minutes: float
    age_label: str
    age_bin: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "age_days": self.age_days,
            "age_hours": self.age_hours,
            "age_minutes": self.age_minutes,
            "age_label": self.age_label,
            "age_bin": self.age_bin,
        }


def load_age_grids(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(path or config_path("age_grids.yaml"))


def get_age_values(preset: str = DEFAULT_AGE_PRESET, path: str | Path | None = None) -> list[float]:
    grids = load_age_grids(path)
    if preset not in grids or not isinstance(grids[preset], dict):
        raise ValueError(f"Unknown age grid preset '{preset}'.")
    values = [float(v) for v in grids[preset].get("values", [])]
    policy = grids.get("zero_age_policy", {})
    if not policy.get("include_zero_by_default", False):
        values = [v for v in values if v != 0.0]
    return values


def parse_age_list(ages: str) -> list[float]:
    values = [float(part.strip()) for part in ages.split(",") if part.strip()]
    return values


def age_bin(age_days: float) -> str:
    age_days = float(age_days)
    if age_days < 0.0417:
        return "ultra_early"
    if age_days < 0.5:
        return "early"
    if age_days <= 2:
        return "around_1d"
    if age_days <= 7:
        return "early_late"
    if age_days <= 90:
        return "standard"
    return "long_term"


def age_label(age_days: float) -> str:
    minutes = float(age_days) * 24.0 * 60.0
    hours = float(age_days) * 24.0
    if minutes < 60:
        return f"{minutes:g} min"
    if hours < 48:
        return f"{hours:g} h"
    return f"{float(age_days):g} d"


def age_metadata(age_days: float) -> AgeMetadata:
    age_days = float(age_days)
    return AgeMetadata(
        age_days=age_days,
        age_hours=age_days * 24.0,
        age_minutes=age_days * 24.0 * 60.0,
        age_label=age_label(age_days),
        age_bin=age_bin(age_days),
    )


def add_age_metadata(row: dict[str, Any], age_days: float) -> dict[str, Any]:
    out = dict(row)
    out.update(age_metadata(age_days).to_dict())
    return out


def detect_age_preset(ages: list[float], path: str | Path | None = None, tolerance: float = 1.0e-9) -> str | None:
    actual = sorted(float(age) for age in ages)
    grids = load_age_grids(path)
    for name, grid in grids.items():
        if not isinstance(grid, dict) or "values" not in grid:
            continue
        expected = sorted(float(age) for age in grid["values"])
        if len(expected) == len(actual) and all(abs(a - b) <= tolerance for a, b in zip(actual, expected)):
            return name
    return None
