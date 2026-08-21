from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .chemistry_vector import CanonicalChemistryVector
from .utils import config_path, to_jsonable

CHEM_HASH_VERSION = 1


@dataclass
class ChemHashResult:
    chem_hash: str
    chem_hash_version: int
    exact_vector: dict[str, Any]
    rounded_vector: dict[str, Any]
    hash_payload: dict[str, Any]
    dat_lst_hash: str
    species_map_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "chem_hash": self.chem_hash,
            "chem_hash_version": self.chem_hash_version,
            "exact_vector": self.exact_vector,
            "rounded_vector": self.rounded_vector,
            "hash_payload": self.hash_payload,
            "dat_lst_hash": self.dat_lst_hash,
            "species_map_hash": self.species_map_hash,
        }


def file_sha256(path: str | Path | None, *, fallback: str = "") -> str:
    if path is None:
        return fallback
    path = Path(path)
    if not path.exists():
        return fallback or f"missing:{path}"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def round_sig(value: float, digits: int = 12) -> float:
    value = float(value)
    if value == 0.0 or not math.isfinite(value):
        return value
    return round(value, digits - int(math.floor(math.log10(abs(value)))) - 1)


def round_mapping(mapping: dict[str, float], digits: int = 12) -> dict[str, float]:
    return {key: round_sig(float(mapping.get(key, 0.0)), digits) for key in sorted(mapping)}


def compute_chem_hash(
    canonical_vector: CanonicalChemistryVector,
    *,
    water_mol: float,
    temperature_celsius: float,
    pressure: float | None = None,
    dat_lst_path: str | Path | None = None,
    dat_lst_hash: str | None = None,
    species_map_path: str | Path | None = None,
    species_map_hash: str | None = None,
    xgems_run_mode: str = "reacted_only",
    thermodynamic_database_identifier: str | None = None,
    rounding_sig_digits: int = 12,
    chem_hash_version: int = CHEM_HASH_VERSION,
) -> ChemHashResult:
    exact = canonical_vector.to_dict()
    rounded_vector = {
        "element_mol": round_mapping(exact["element_mol"], rounding_sig_digits),
        "oxide_equivalent_mol": round_mapping(exact["oxide_equivalent_mol"], rounding_sig_digits),
    }
    dat_hash = dat_lst_hash or file_sha256(dat_lst_path, fallback="mock-dat")
    species_hash = species_map_hash or file_sha256(species_map_path or config_path("species_map.yaml"))
    payload = {
        "chem_hash_version": chem_hash_version,
        "canonical_component_vector": rounded_vector["element_mol"],
        "water_mol": round_sig(float(water_mol), rounding_sig_digits),
        "temperature_celsius": round_sig(float(temperature_celsius), rounding_sig_digits),
        "pressure": None if pressure is None else round_sig(float(pressure), rounding_sig_digits),
        "dat_lst_hash": dat_hash,
        "species_map_hash": species_hash,
        "xgems_run_mode": xgems_run_mode,
        "thermodynamic_database_identifier": thermodynamic_database_identifier,
    }
    encoded = json.dumps(to_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return ChemHashResult(
        chem_hash=hashlib.sha256(encoded).hexdigest(),
        chem_hash_version=chem_hash_version,
        exact_vector=exact,
        rounded_vector=rounded_vector,
        hash_payload=payload,
        dat_lst_hash=dat_hash,
        species_map_hash=species_hash,
    )
