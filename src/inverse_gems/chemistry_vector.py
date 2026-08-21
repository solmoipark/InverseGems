from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .constants import (
    ELEMENT_BASIS,
    FORMULAS,
    MOLAR_MASS_G_MOL,
    OXIDE_EQUIVALENT_BASIS,
    OXIDE_EQUIVALENTS,
)


def canonical_component_name(name: str) -> str:
    aliases = {
        "water": "H2O",
        "H2O@": "H2O@",
        "gypsum": "gypsum",
        "CSH2": "gypsum",
        "CaSO4:2H2O": "gypsum",
        "CaSO4·2H2O": "gypsum",
        "CaCO3": "CaCO3",
        "limestone": "CaCO3",
        "Cal": "Cal",
    }
    return aliases.get(name, name)


@dataclass
class CanonicalChemistryVector:
    element_mol: dict[str, float]
    oxide_equivalent_mol: dict[str, float]
    component_mol: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_mol": {key: float(self.element_mol.get(key, 0.0)) for key in ELEMENT_BASIS},
            "oxide_equivalent_mol": {
                key: float(self.oxide_equivalent_mol.get(key, 0.0)) for key in OXIDE_EQUIVALENT_BASIS
            },
            "component_mol": dict(sorted((key, float(value)) for key, value in self.component_mol.items())),
        }


def empty_vector() -> CanonicalChemistryVector:
    return CanonicalChemistryVector(
        element_mol={key: 0.0 for key in ELEMENT_BASIS},
        oxide_equivalent_mol={key: 0.0 for key in OXIDE_EQUIVALENT_BASIS},
        component_mol={},
    )


def add_component_moles(vector: CanonicalChemistryVector, component: str, moles: float) -> None:
    component = canonical_component_name(component)
    if component not in FORMULAS:
        raise ValueError(f"No formula is defined for component '{component}'.")
    moles = float(moles)
    vector.component_mol[component] = vector.component_mol.get(component, 0.0) + moles
    for element, coefficient in FORMULAS[component].items():
        if element in ELEMENT_BASIS:
            vector.element_mol[element] = vector.element_mol.get(element, 0.0) + moles * float(coefficient)
    for oxide, coefficient in OXIDE_EQUIVALENTS.get(component, {}).items():
        if oxide in OXIDE_EQUIVALENT_BASIS:
            vector.oxide_equivalent_mol[oxide] = vector.oxide_equivalent_mol.get(oxide, 0.0) + moles * float(coefficient)


def add_component_mass_g(vector: CanonicalChemistryVector, component: str, mass_g: float) -> None:
    component = canonical_component_name(component)
    if component not in MOLAR_MASS_G_MOL:
        raise ValueError(f"No molar mass is defined for component '{component}'.")
    add_component_moles(vector, component, float(mass_g) / MOLAR_MASS_G_MOL[component])


def chemistry_vector_from_component_moles(component_mol: dict[str, float]) -> CanonicalChemistryVector:
    vector = empty_vector()
    for component, moles in component_mol.items():
        add_component_moles(vector, component, moles)
    return vector


def chemistry_vector_from_amounts_kg(amounts_kg: dict[str, float]) -> CanonicalChemistryVector:
    vector = empty_vector()
    for component, kg in amounts_kg.items():
        add_component_mass_g(vector, component, float(kg) * 1000.0)
    return vector


def chemistry_vector_from_source_ledger(rows: Iterable[dict[str, Any]]) -> CanonicalChemistryVector:
    vector = empty_vector()
    for row in rows:
        component = str(row.get("component", ""))
        component_mol = float(row.get("component_mol") or 0.0)
        if component in ELEMENT_BASIS:
            vector.element_mol[component] = vector.element_mol.get(component, 0.0) + component_mol
        oxide = row.get("oxide_equivalent")
        oxide_mol = float(row.get("oxide_equivalent_mol") or 0.0)
        if oxide in OXIDE_EQUIVALENT_BASIS:
            vector.oxide_equivalent_mol[str(oxide)] = vector.oxide_equivalent_mol.get(str(oxide), 0.0) + oxide_mol
    return vector


def vector_difference(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    keys = sorted(set(a) | set(b))
    return {key: float(a.get(key, 0.0)) - float(b.get(key, 0.0)) for key in keys}


def vectors_close(a: dict[str, float], b: dict[str, float], tolerance: float = 1.0e-12) -> bool:
    return all(abs(value) <= tolerance for value in vector_difference(a, b).values())
