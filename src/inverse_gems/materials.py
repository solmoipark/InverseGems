from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import config_path, load_yaml


SCM_NAMES = {"slag", "fly_ash", "metakaolin", "silica_fume"}
BINDER_COMPONENTS = {"OPC", "slag", "fly_ash", "metakaolin", "silica_fume", "limestone", "gypsum"}


@dataclass(frozen=True)
class Material:
    name: str
    density_g_cm3: float
    aliases: tuple[str, ...] = ()
    oxide_mass_percent: dict[str, float] | None = None
    compound_mass_percent: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "density_g_cm3": self.density_g_cm3,
            "aliases": list(self.aliases),
        }
        if self.oxide_mass_percent is not None:
            out["oxide_mass_percent"] = dict(self.oxide_mass_percent)
        if self.compound_mass_percent is not None:
            out["compound_mass_percent"] = dict(self.compound_mass_percent)
        return out


def _norm_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def load_materials(path: str | Path | None = None) -> dict[str, Material]:
    raw = load_yaml(path or config_path("materials.yaml"))
    materials: dict[str, Material] = {}
    for name, data in raw.items():
        materials[name] = Material(
            name=name,
            density_g_cm3=float(data["density_g_cm3"]),
            aliases=tuple(data.get("aliases", []) or ()),
            oxide_mass_percent={k: float(v) for k, v in (data.get("oxide_mass_percent") or {}).items()} or None,
            compound_mass_percent={k: float(v) for k, v in (data.get("compound_mass_percent") or {}).items()} or None,
        )
    return materials


def build_alias_map(materials: dict[str, Material] | None = None) -> dict[str, str]:
    materials = materials or load_materials()
    aliases: dict[str, str] = {}
    for name, material in materials.items():
        aliases[_norm_name(name)] = name
        aliases[name.lower()] = name
        for alias in material.aliases:
            aliases[_norm_name(alias)] = name
            aliases[alias.lower()] = name
    aliases.update(
        {
            "pc": "OPC",
            "portland_cement": "OPC",
            "ordinary_portland_cement": "OPC",
            "flyash": "fly_ash",
            "fly_ash": "fly_ash",
            "fly ash": "fly_ash",
            "bfs": "slag",
            "blast_furnace_slag": "slag",
            "mk": "metakaolin",
            "sf": "silica_fume",
            "silica_fume": "silica_fume",
            "silica fume": "silica_fume",
            "ls": "limestone",
            "gp": "gypsum",
            "csh2": "gypsum",
        }
    )
    return aliases


def canonicalize_material_name(name: str, materials: dict[str, Material] | None = None) -> str:
    aliases = build_alias_map(materials)
    candidates = [name.strip(), _norm_name(name)]
    for candidate in candidates:
        if candidate in aliases:
            return aliases[candidate]
        lowered = candidate.lower()
        if lowered in aliases:
            return aliases[lowered]
    raise ValueError(f"Unknown binder component '{name}'. Edit configs/materials.yaml to add an alias.")


def material_subset(materials: dict[str, Material], masses_g: dict[str, float]) -> dict[str, Any]:
    return {name: materials[name].to_dict() for name, mass in masses_g.items() if mass > 0 and name in materials}
