from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .chem_hash import file_sha256
from .pk_model import PKParameters, default_pk_parameters, pk_parameters_from_dict
from .scm_reaction import (
    SCMParameters,
    load_scm_parameters,
    parameters_to_dict,
    scm_kinetics_from_mapping,
)
from .utils import config_path, load_yaml, to_jsonable


DEFAULT_REACTION_PARAMETER_SET_ID = "local_default_parameters"

# OPC oxides that the Bogue conversion does not carry into the clinker phases. Before this
# policy existed they were silently dropped, so the xGEMS system had no sulfur (no AFt/AFm)
# and no alkalis (pH pinned at the portlandite buffer). Degrees: a number, or
# "clinker_mean" (mass-weighted mean of the Parrot-Killoh clinker degrees).
DEFAULT_OPC_MINOR_OXIDES: dict[str, Any] = {
    "enabled": True,
    "degrees": {"SO3": 1.0, "Na2O": 1.0, "K2O": 1.0, "MgO": "clinker_mean"},
    # SO3 enters as CaSO4: the CaO tied to the sulfate (Bogue subtracts 2.85·SO3 from C3S) is
    # added alongside the SO3 so calcium is conserved.
    "sulfate_as_calcium_sulfate": True,
}


@dataclass
class ReactionParameterSet:
    id: str
    config_path: str | None
    config_hash: str
    raw_config: dict[str, Any]
    pk_parameters: PKParameters
    relative_humidity: float
    fineness_m2_kg: float
    scm_parameters: dict[str, SCMParameters]
    availability_config: dict[str, Any]
    apply_availability_modifier: bool
    warnings: list[str]
    # OPC minor-oxide policy (SO3 as CaSO4, MgO, Na2O, K2O added to the xGEMS input);
    # see DEFAULT_OPC_MINOR_OXIDES. Part of the signature payload.
    opc_minor_oxides: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_OPC_MINOR_OXIDES))

    def to_dict(self) -> dict[str, Any]:
        return {
            "opc_minor_oxides": dict(self.opc_minor_oxides),
            "id": self.id,
            "config_path": self.config_path,
            "config_hash": self.config_hash,
            "raw_config": self.raw_config,
            "pk_model": {
                "relative_humidity": self.relative_humidity,
                "fineness_m2_kg": self.fineness_m2_kg,
                "parameters": self.pk_parameters.to_dict(),
            },
            "scm_reaction": parameters_to_dict(self.scm_parameters),
            "c3s_c2s_availability": self.availability_config,
            "availability_modifier": {"enabled": self.apply_availability_modifier},
            "warnings": list(self.warnings),
        }

    def signature_payload(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("warnings", None)
        return payload


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _load_raw_config(path: str | Path | None) -> tuple[Path | None, dict[str, Any], str]:
    if path is None:
        return None, {}, ""
    config = Path(path)
    raw = load_yaml(config)
    return config, raw, file_sha256(config, fallback=f"missing:{config}")


def _load_scm_parameters(raw: dict[str, Any]) -> dict[str, SCMParameters]:
    base = parameters_to_dict(load_scm_parameters())
    overrides = raw.get("scm_reaction") or raw.get("scm_parameters") or {}
    merged: dict[str, Any] = {name: dict(values) for name, values in base.items()}
    for name, override in overrides.items():
        if isinstance(override, dict) and "model" in override:
            # Switching the kinetics model replaces the entry wholesale;
            # merging would leak parameters from the previous equation.
            merged[name] = dict(override)
        elif isinstance(override, dict) and isinstance(merged.get(name), dict):
            merged[name] = _deep_merge(dict(merged[name]), override)
        else:
            merged[name] = override
    return {name: scm_kinetics_from_mapping(values) for name, values in merged.items()}


def _load_availability_config(raw: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    base = load_yaml(config_path("c3s_c2s_availability.yaml"))
    availability_section_raw = raw.get("availability_modifier")
    if isinstance(availability_section_raw, bool):
        availability_section: dict[str, Any] = {"enabled": availability_section_raw}
    else:
        availability_section = availability_section_raw or {}
    overrides = raw.get("c3s_c2s_availability") or availability_section.get("config") or {}
    enabled = bool(availability_section.get("enabled", raw.get("apply_availability_modifier", True)))
    return _deep_merge(base, overrides), enabled


def _load_pk_parameters(raw: dict[str, Any]) -> tuple[PKParameters, float, float]:
    pk_section = raw.get("pk_model") or raw.get("parrot_killoh") or {}
    constants = pk_section.get("constants") or {
        key: value
        for key, value in pk_section.items()
        if key in {"K1", "N1", "K2", "K3", "N3", "H", "Ea", "T0", "ref_fineness"}
    }
    pk_parameters = pk_parameters_from_dict(constants)
    relative_humidity = float(pk_section.get("relative_humidity", pk_section.get("RH", 1.0)))
    fineness = float(pk_section.get("fineness_m2_kg", pk_section.get("fineness", pk_parameters.ref_fineness)))
    return pk_parameters, relative_humidity, fineness


def load_reaction_parameters(
    path: str | Path | None = None,
    *,
    reaction_model_id: str | None = None,
) -> ReactionParameterSet:
    config, raw, config_hash = _load_raw_config(path)
    scm_parameters = _load_scm_parameters(raw)
    availability_config, apply_availability_modifier = _load_availability_config(raw)
    pk_parameters, relative_humidity, fineness = _load_pk_parameters(raw)
    set_id = str(reaction_model_id or raw.get("id") or DEFAULT_REACTION_PARAMETER_SET_ID)
    warnings: list[str] = []
    minor_raw = raw.get("opc_minor_oxides")
    if isinstance(minor_raw, bool):
        minor_raw = {"enabled": minor_raw}
    opc_minor = _deep_merge(dict(DEFAULT_OPC_MINOR_OXIDES), dict(minor_raw or {}))
    return ReactionParameterSet(
        opc_minor_oxides=opc_minor,
        id=set_id,
        config_path=str(config) if config else None,
        config_hash=config_hash,
        raw_config=to_jsonable(raw),
        pk_parameters=pk_parameters,
        relative_humidity=relative_humidity,
        fineness_m2_kg=fineness,
        scm_parameters=scm_parameters,
        availability_config=availability_config,
        apply_availability_modifier=apply_availability_modifier,
        warnings=warnings,
    )
