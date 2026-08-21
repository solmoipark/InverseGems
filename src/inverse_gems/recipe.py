from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .materials import BINDER_COMPONENTS, Material, canonicalize_material_name, load_materials


NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


@dataclass
class Recipe:
    raw_text: str
    binder_masses_g: dict[str, float]
    age_days: float
    water_g: float
    water_mode: str
    w_b: float | None = None
    basis_g: float = 100.0
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "binder_masses_g": dict(self.binder_masses_g),
            "age_days": self.age_days,
            "water_g": self.water_g,
            "water_mode": self.water_mode,
            "w_b": self.w_b,
            "basis_g": self.basis_g,
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


def _last_number(token: str) -> tuple[float, tuple[int, int]]:
    matches = list(NUMBER_RE.finditer(token))
    if not matches:
        raise ValueError(f"Could not find a numeric value in recipe token '{token}'.")
    match = matches[-1]
    return float(match.group()), match.span()


def _looks_like_wb(token: str) -> bool:
    compact = token.lower().replace(" ", "")
    return compact.startswith(("w/b", "wb", "w_b", "water/binder", "waterbinder"))


def parse_recipe(
    text: str,
    *,
    materials: dict[str, Material] | None = None,
    normalize: bool = True,
    allow_non_100: bool = False,
    sum_tolerance_g: float = 1.0e-6,
) -> Recipe:
    materials = materials or load_materials()
    binder_masses: dict[str, float] = {}
    age_days: float | None = None
    w_b: float | None = None
    direct_water_g: float | None = None
    warnings: list[str] = []

    for raw_token in text.split(","):
        token = raw_token.strip()
        if not token:
            continue
        lower = token.lower()
        value, span = _last_number(token)

        if _looks_like_wb(token):
            w_b = value
            continue
        if lower.startswith(("age", "day", "days", "t ")):
            age_days = value
            continue
        if lower.startswith(("water", "h2o")):
            direct_water_g = value
            continue

        name = token[: span[0]].strip(" :=")
        if not name:
            raise ValueError(f"Could not identify binder component in recipe token '{token}'.")
        canonical = canonicalize_material_name(name, materials)
        if canonical not in BINDER_COMPONENTS:
            raise ValueError(f"'{canonical}' is not a supported binder component.")
        binder_masses[canonical] = binder_masses.get(canonical, 0.0) + value

    if not binder_masses:
        raise ValueError("Recipe must include at least one binder component.")
    if age_days is None:
        raise ValueError("Recipe must include age, for example 'age 28'.")
    if age_days <= 0:
        raise ValueError("Recipe age must be positive.")
    if w_b is not None and direct_water_g is not None:
        raise ValueError("Provide either w/b or direct water mass, not both.")
    if w_b is None and direct_water_g is None:
        raise ValueError("Recipe must include either w/b or direct water mass.")

    binder_total = sum(binder_masses.values())
    if binder_total <= 0:
        raise ValueError("Total binder mass must be positive.")

    basis_g = binder_total
    if normalize:
        basis_g = 100.0
        if abs(binder_total - 100.0) > sum_tolerance_g:
            scale = 100.0 / binder_total
            binder_masses = {k: v * scale for k, v in binder_masses.items()}
            warnings.append(f"Binder masses normalized from {binder_total:g} g to 100 g total binder.")
    elif not allow_non_100 and abs(binder_total - 100.0) > sum_tolerance_g:
        raise ValueError(
            f"Binder components sum to {binder_total:g} g, not 100 g. "
            "Use allow_non_100=True for explicit-mass recipes."
        )

    if w_b is not None:
        water_g = basis_g * w_b
        water_mode = "wb_total"
    else:
        water_g = float(direct_water_g)
        water_mode = "direct_h2o"

    return Recipe(
        raw_text=text,
        binder_masses_g=binder_masses,
        age_days=float(age_days),
        water_g=float(water_g),
        water_mode=water_mode,
        w_b=w_b,
        basis_g=float(basis_g),
        warnings=warnings,
    )
