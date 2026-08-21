from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bogue import bogue_phases
from .materials import SCM_NAMES, Material, load_materials
from .scm_reaction import SCMLogisticParameters, load_scm_parameters, parameters_to_dict
from .utils import config_path, load_yaml


@dataclass
class AvailabilityModification:
    parameters: dict[str, SCMLogisticParameters]
    metadata: dict[str, Any]


class C3S_C2SAvailabilityModifier:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        config_file: str | Path | None = None,
        materials: dict[str, Material] | None = None,
    ) -> None:
        self.config = config or load_yaml(config_file or config_path("c3s_c2s_availability.yaml"))
        self.materials = materials or load_materials()

    @staticmethod
    def _fraction(value: float) -> float:
        value = float(value)
        return value / 100.0 if value > 1.0 else value

    def _reference_availability(self) -> tuple[float, dict[str, float], float, float]:
        reference = self.config.get("reference_system", {})
        opc_mass = float(reference.get("OPC", 60.0))
        scm_total = float(reference.get("SCM_total", 40.0))
        c2s_weight = float(self.config.get("c2s_weight", 0.30))
        opc = self.materials["OPC"]
        if opc.oxide_mass_percent is None:
            raise ValueError("Default OPC material must include oxide_mass_percent for Bogue reference.")
        ref_bogue = bogue_phases(opc.oxide_mass_percent).phase_mass_percent
        supply = opc_mass * (self._fraction(ref_bogue["C3S"]) + c2s_weight * self._fraction(ref_bogue["C2S"]))
        demand = scm_total
        availability = supply / demand if demand > 0 else 0.0
        return availability, ref_bogue, supply, demand

    def modify_parameters(
        self,
        binder_masses_g: dict[str, float],
        opc_phase_mass_percent: dict[str, float],
        scm_params: dict[str, SCMLogisticParameters] | None = None,
        *,
        use_time_dependent_supply: bool = False,
    ) -> AvailabilityModification:
        params = scm_params or load_scm_parameters()
        effective = dict(params)
        present_scms = {name: mass for name, mass in binder_masses_g.items() if name in SCM_NAMES and mass > 0}
        metadata: dict[str, Any] = {
            "applied": False,
            "future_hook_use_time_dependent_supply": bool(use_time_dependent_supply),
            "input_parameters": parameters_to_dict(params),
            "per_scm": {},
        }

        if not present_scms:
            metadata["reason"] = "no_scm"
            return AvailabilityModification(effective, metadata)

        demand_coefficients = self.config.get("demand_coefficients", {})
        eta = self.config.get("eta", {})
        absolute_max = self.config.get("absolute_max_D", {})
        c2s_weight = float(self.config.get("c2s_weight", 0.30))

        demand = sum(float(mass) * float(demand_coefficients.get(name, 1.0)) for name, mass in present_scms.items())
        if demand <= 0:
            metadata["reason"] = "zero_scm_demand"
            return AvailabilityModification(effective, metadata)

        opc_mass = float(binder_masses_g.get("OPC", 0.0))
        supply = opc_mass * (
            self._fraction(opc_phase_mass_percent.get("C3S", 0.0))
            + c2s_weight * self._fraction(opc_phase_mass_percent.get("C2S", 0.0))
        )
        availability_mix = supply / demand if demand > 0 else 0.0
        availability_ref, ref_bogue, ref_supply, ref_demand = self._reference_availability()
        R = availability_mix / availability_ref if availability_ref > 0 else 0.0
        R = max(R, 0.0)

        for name in present_scms:
            if name not in effective:
                continue
            d_ref = float(params[name].D)
            eta_i = float(eta.get(name, 1.0))
            d_absmax = float(absolute_max.get(name, 1.0))
            d_eff = min(d_absmax, d_ref * (R**eta_i))
            d_eff = max(0.0, min(1.0, d_eff))
            effective[name] = params[name].with_D(d_eff)
            metadata["per_scm"][name] = {
                "D_ref": d_ref,
                "D_eff": d_eff,
                "eta": eta_i,
                "D_absmax": d_absmax,
                "demand_coefficient": float(demand_coefficients.get(name, 1.0)),
            }

        metadata.update(
            {
                "applied": True,
                "C3S_C2S_supply_index": supply,
                "SCM_demand_index": demand,
                "availability_mix": availability_mix,
                "availability_reference": availability_ref,
                "R": R,
                "reference_bogue_phase_mass_percent": ref_bogue,
                "reference_supply_index": ref_supply,
                "reference_demand_index": ref_demand,
            }
        )
        return AvailabilityModification(effective, metadata)
