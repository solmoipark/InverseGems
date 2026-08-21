from __future__ import annotations

ELEMENT_MOLAR_MASS_G_MOL: dict[str, float] = {
    "Ca": 40.078,
    "Si": 28.0855,
    "Al": 26.9815385,
    "Fe": 55.845,
    "Mg": 24.305,
    "S": 32.065,
    "Na": 22.98976928,
    "K": 39.0983,
    "C": 12.011,
    "H": 1.008,
    "O": 15.999,
}

ELEMENT_BASIS: tuple[str, ...] = ("Ca", "Si", "Al", "Fe", "Mg", "S", "Na", "K", "C", "H", "O")
OXIDE_EQUIVALENT_BASIS: tuple[str, ...] = (
    "CaO",
    "SiO2",
    "Al2O3",
    "Fe2O3",
    "MgO",
    "SO3",
    "Na2O",
    "K2O",
    "CO2",
    "H2O",
)

FORMULAS: dict[str, dict[str, float]] = {
    "CaO": {"Ca": 1, "O": 1},
    "SiO2": {"Si": 1, "O": 2},
    "Al2O3": {"Al": 2, "O": 3},
    "Fe2O3": {"Fe": 2, "O": 3},
    "MgO": {"Mg": 1, "O": 1},
    "SO3": {"S": 1, "O": 3},
    "Na2O": {"Na": 2, "O": 1},
    "K2O": {"K": 2, "O": 1},
    "CO2": {"C": 1, "O": 2},
    "H2O": {"H": 2, "O": 1},
    "H2O@": {"H": 2, "O": 1},
    "O2": {"O": 2},
    "C3S": {"Ca": 3, "Si": 1, "O": 5},
    "C2S": {"Ca": 2, "Si": 1, "O": 4},
    "C3A": {"Ca": 3, "Al": 2, "O": 6},
    "C4AF": {"Ca": 4, "Al": 2, "Fe": 2, "O": 10},
    "CaCO3": {"Ca": 1, "C": 1, "O": 3},
    "Cal": {"Ca": 1, "C": 1, "O": 3},
    "gypsum": {"Ca": 1, "S": 1, "O": 6, "H": 4},
    "Gp": {"Ca": 1, "S": 1, "O": 6, "H": 4},
    "CaSO4·2H2O": {"Ca": 1, "S": 1, "O": 6, "H": 4},
    "CaSO4*2H2O": {"Ca": 1, "S": 1, "O": 6, "H": 4},
}

OXIDE_EQUIVALENTS: dict[str, dict[str, float]] = {
    "CaO": {"CaO": 1},
    "SiO2": {"SiO2": 1},
    "Al2O3": {"Al2O3": 1},
    "Fe2O3": {"Fe2O3": 1},
    "MgO": {"MgO": 1},
    "SO3": {"SO3": 1},
    "Na2O": {"Na2O": 1},
    "K2O": {"K2O": 1},
    "CO2": {"CO2": 1},
    "H2O": {"H2O": 1},
    "H2O@": {"H2O": 1},
    "C3S": {"CaO": 3, "SiO2": 1},
    "C2S": {"CaO": 2, "SiO2": 1},
    "C3A": {"CaO": 3, "Al2O3": 1},
    "C4AF": {"CaO": 4, "Al2O3": 1, "Fe2O3": 1},
    "CaCO3": {"CaO": 1, "CO2": 1},
    "Cal": {"CaO": 1, "CO2": 1},
    "gypsum": {"CaO": 1, "SO3": 1, "H2O": 2},
    "Gp": {"CaO": 1, "SO3": 1, "H2O": 2},
    "CaSO4·2H2O": {"CaO": 1, "SO3": 1, "H2O": 2},
    "CaSO4*2H2O": {"CaO": 1, "SO3": 1, "H2O": 2},
}


def molar_mass_from_formula(formula: dict[str, float]) -> float:
    return sum(ELEMENT_MOLAR_MASS_G_MOL[element] * count for element, count in formula.items())


MOLAR_MASS_G_MOL: dict[str, float] = {
    name: molar_mass_from_formula(formula) for name, formula in FORMULAS.items()
}
