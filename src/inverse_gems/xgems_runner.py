from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from .phase_volume_reconstruction import add_phase_volume_reconstruction
from .utils import config_path, load_yaml, to_jsonable


class XGEMSRunnerError(RuntimeError):
    pass


def _simple_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, type(None)))


class XGEMSRunner:
    def __init__(
        self,
        dat_lst_path: str | Path,
        temperature_celsius: float = 20.0,
        *,
        gems_class_path: str = "run.GEMSCalc:GEMS",
        add_o2: bool = True,
        o2_amount_kg: float = 1.0e-9,
        clear_existing_composition: bool = True,
        input_mode: str = "species",
        formula_map: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.dat_lst_path = Path(dat_lst_path)
        if not self.dat_lst_path.exists():
            raise XGEMSRunnerError(f"dat.lst path does not exist: {self.dat_lst_path}")
        self.temperature_celsius = temperature_celsius
        self.gems_class_path = gems_class_path
        self.add_o2 = add_o2
        self.o2_amount_kg = o2_amount_kg
        self.clear_existing_composition = clear_existing_composition
        self.input_mode = input_mode
        self.formula_map = formula_map
        self._composition_cleared = False
        self.gems = self._create_gems_object()
        self._set_temperature()

    def _create_gems_object(self) -> Any:
        module_name, _, class_name = self.gems_class_path.partition(":")
        if not module_name or not class_name:
            raise XGEMSRunnerError("gems_class_path must look like 'run.GEMSCalc:GEMS'.")
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            return cls(str(self.dat_lst_path))
        except Exception as exc:
            raise XGEMSRunnerError(
                "Could not initialize xGEMS/GEMS. "
                f"Tried '{self.gems_class_path}' with dat.lst '{self.dat_lst_path}'. "
                "Install/configure the local xGEMS Python module or pass a compatible class path. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc

    def _set_temperature(self) -> None:
        kelvin = self.temperature_celsius + 273.15
        if hasattr(self.gems, "T"):
            setattr(self.gems, "T", kelvin)
        elif hasattr(self.gems, "set_temperature"):
            self.gems.set_temperature(kelvin)

    def add_species_amounts(self, species_amounts: dict[str, float], units: str = "kg") -> None:
        if self.input_mode == "formula":
            self.add_formula_amounts(species_amounts, units=units)
            return
        if self.input_mode != "species":
            raise XGEMSRunnerError("input_mode must be 'species' or 'formula'.")
        available_species = self._available_species_names()
        attempted_species = dict(species_amounts)
        if self.add_o2 and "O2" not in attempted_species:
            attempted_species["O2"] = self.o2_amount_kg
        if available_species is not None:
            missing = sorted(species for species in attempted_species if species not in available_species)
            if missing:
                preview = sorted(available_species)[:80]
                raise XGEMSRunnerError(
                    "The loaded xGEMS/GEMS database does not contain required species. "
                    f"Missing species: {missing}. "
                    f"Attempted species dictionary: {species_amounts}. "
                    f"First available species names: {preview}"
                )
        if self.clear_existing_composition and not self._composition_cleared and hasattr(self.gems, "clear"):
            self.gems.clear()
            self._composition_cleared = True
        try:
            if hasattr(self.gems, "add_multiple_species_amt"):
                self.gems.add_multiple_species_amt(species_amounts, units=units)
            elif hasattr(self.gems, "add_species_amt"):
                for species, amount in species_amounts.items():
                    self.gems.add_species_amt(species, amount, units=units)
            else:
                raise XGEMSRunnerError("GEMS object has no add_multiple_species_amt or add_species_amt method.")
            if self.add_o2 and "O2" not in species_amounts and hasattr(self.gems, "add_species_amt"):
                self.gems.add_species_amt("O2", self.o2_amount_kg, units=units)
        except Exception as exc:
            raise XGEMSRunnerError(
                "Failed to add species amounts to xGEMS/GEMS. "
                f"Attempted species dictionary: {species_amounts}"
            ) from exc

    def add_formula_amounts(self, formula_amounts: dict[str, float], units: str = "kg") -> None:
        if not hasattr(self.gems, "add_amt_from_formula"):
            raise XGEMSRunnerError("GEMS object does not expose add_amt_from_formula for formula input mode.")
        formula_map = self.formula_map or load_formula_map()
        attempted = dict(formula_amounts)
        if self.add_o2 and "O2" not in attempted:
            attempted["O2"] = self.o2_amount_kg
        missing = sorted(name for name in attempted if name not in formula_map)
        if missing:
            raise XGEMSRunnerError(
                "Formula input mode requires formulas for every attempted input name. "
                f"Missing formulas: {missing}. Attempted amount dictionary: {formula_amounts}"
            )
        if self.clear_existing_composition and not self._composition_cleared and hasattr(self.gems, "clear"):
            self.gems.clear()
            self._composition_cleared = True
        try:
            for name, amount in attempted.items():
                self.gems.add_amt_from_formula(formula_map[name], float(amount), units=units)
        except Exception as exc:
            raise XGEMSRunnerError(
                "Failed to add formula amounts to xGEMS/GEMS. "
                f"Attempted amount dictionary: {formula_amounts}"
            ) from exc

    def _available_species_names(self) -> set[str] | None:
        if not hasattr(self.gems, "species_names"):
            return None
        try:
            names = getattr(self.gems, "species_names")
            return {str(name) for name in names}
        except Exception:
            return None

    def apply_lower_bounds(self, lower_bounds: dict[str, float], units: str = "kg") -> None:
        if not lower_bounds:
            return
        if not hasattr(self.gems, "species_lower_bound") and not hasattr(self.gems, "set_species_lower_bound"):
            raise XGEMSRunnerError(
                "GEMS object does not expose species_lower_bound or set_species_lower_bound for lower_bound_legacy mode."
            )
        for species, amount in lower_bounds.items():
            if hasattr(self.gems, "species_lower_bound"):
                self.gems.species_lower_bound(species, amount, units=units)
            else:
                self.gems.set_species_lower_bound(species, amount, units=units)

    def equilibrate(self) -> Any:
        if not hasattr(self.gems, "equilibrate"):
            raise XGEMSRunnerError("GEMS object has no equilibrate method.")
        return self.gems.equilibrate()

    def capture_raw_state(self) -> dict[str, Any]:
        return capture_gems_object_state(self.gems)


def load_formula_map(path: str | Path | None = None) -> dict[str, dict[str, float]]:
    raw = load_yaml(path or config_path("formula_map.yaml"))
    return {name: {element: float(count) for element, count in formula.items()} for name, formula in raw.items()}


def capture_gems_object_state(obj: Any) -> dict[str, Any]:
    attribute_names = sorted(name for name in dir(obj) if not name.startswith("__"))
    missing: list[str] = []

    def get_attr(*names: str) -> Any:
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
        missing.extend(names)
        return None

    def get_attr_or_call(*names: str) -> Any:
        for name in names:
            if hasattr(obj, name):
                value = getattr(obj, name)
                if callable(value):
                    try:
                        return value()
                    except Exception:
                        missing.append(name)
                        continue
                return value
        missing.extend(names)
        return None

    raw = {
        "phase_names": get_attr("phase_names", "phases_names"),
        "phase_masses": get_attr("phase_masses", "phases_masses"),
        "phase_amounts": get_attr("phase_amounts", "phases_moles"),
        "phase_volumes": get_attr("phase_volumes", "phases_volumes"),
        "phase_molar_volumes": get_attr("phase_molar_volume", "phases_molar_volume"),
        "phase_species_moles": get_attr_or_call("phase_species_moles", "phase_species_amounts"),
        "system_volume": get_attr("system_volume", "volume"),
        "system_mass": get_attr("system_mass", "mass"),
        "pH": get_attr("pH", "ph"),
        "aqueous_species": get_attr("aqueous_species", "aq_species", "aqueous_species_amounts"),
        "species_amounts": get_attr("species_amounts", "amounts", "species_moles"),
        "species_in_phase": get_attr("species_in_phase"),
        "species_molar_volumes": get_attr("species_molar_volumes", "species_molar_volume"),
        "species_molar_masses": get_attr("species_molar_masses", "species_molar_mass"),
        "solver_status": get_attr("solver_status", "status"),
        "attribute_report": {
            "attribute_names": attribute_names,
            "missing_requested_attributes": sorted(set(missing)),
        },
    }

    scalars: dict[str, Any] = {}
    for name in attribute_names:
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if _simple_scalar(value):
            scalars[name] = value
    raw["scalars"] = scalars
    add_phase_volume_reconstruction(raw)
    return to_jsonable(raw)


class MockXGEMSRunner:
    def __init__(self, dat_lst_path: str | Path | None = None, temperature_celsius: float = 20.0, **_: Any) -> None:
        self.dat_lst_path = str(dat_lst_path) if dat_lst_path else None
        self.temperature_celsius = temperature_celsius
        self.species_amounts: dict[str, float] = {}
        self.lower_bounds: dict[str, float] = {}
        self.phase_masses: dict[str, float] = {}
        self.phase_volumes: dict[str, float] = {}
        self.aqueous_species: dict[str, float] = {}
        self.system_volume = 0.0
        self.system_mass = 0.0
        self.pH = 12.6
        self.solver_status = "not_run"

    def add_species_amounts(self, species_amounts: dict[str, float], units: str = "kg") -> None:
        factor = 1000.0 if units == "kg" else 1.0
        for species, amount in species_amounts.items():
            self.species_amounts[species] = self.species_amounts.get(species, 0.0) + float(amount) * factor

    def apply_lower_bounds(self, lower_bounds: dict[str, float], units: str = "kg") -> None:
        factor = 1000.0 if units == "kg" else 1.0
        self.lower_bounds = {species: float(amount) * factor for species, amount in lower_bounds.items()}

    def equilibrate(self) -> str:
        total_input_g = sum(self.species_amounts.values())
        water_g = self.species_amounts.get("H2O@", self.species_amounts.get("H2O", 0.0))
        reacted_solids_g = max(total_input_g - water_g, 0.0)
        self.phase_masses = {
            "Mock C-S-H raw phase": reacted_solids_g * 0.42,
            "Mock Portlandite": reacted_solids_g * 0.18,
            "Mock Ettringite/AFt": reacted_solids_g * 0.08,
        }
        self.phase_volumes = {
            "Mock C-S-H raw phase": self.phase_masses["Mock C-S-H raw phase"] / 2.15,
            "Mock Portlandite": self.phase_masses["Mock Portlandite"] / 2.24,
            "Mock Ettringite/AFt": self.phase_masses["Mock Ettringite/AFt"] / 1.78,
        }
        self.aqueous_species = {
            "Ca+2": max(1.0e-6, reacted_solids_g * 1.0e-5),
            "OH-": max(1.0e-6, water_g * 1.0e-5),
        }
        self.system_mass = total_input_g
        self.system_volume = sum(self.phase_volumes.values()) + water_g
        self.solver_status = "mock_success"
        return self.solver_status

    def capture_raw_state(self) -> dict[str, Any]:
        raw = capture_gems_object_state(self)
        raw["solver_status"] = self.solver_status
        return raw
