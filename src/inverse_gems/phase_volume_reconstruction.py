from __future__ import annotations

from typing import Any

import pandas as pd

from .utils import as_float


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _nested_mapping(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, inner in value.items():
        if isinstance(inner, dict):
            out[str(key)] = inner
    return out


def _relative_error(a: float, b: float) -> float:
    denominator = max(abs(a), abs(b), 1.0e-30)
    return abs(a - b) / denominator


def reconstruct_phase_volumes(raw_state: dict[str, Any]) -> dict[str, Any]:
    """Derive phase volumes from phase-specific species moles and species molar volumes.

    xGEMS/GEMS may report zero phase volumes for some phases even when phase mass
    and species molar volumes are nonzero. This derived layer does not replace raw
    capture; it provides an auditable fallback for downstream calculations.
    """

    phase_masses = _mapping(raw_state.get("phase_masses"))
    raw_phase_volumes = _mapping(raw_state.get("phase_volumes"))
    phase_amounts = _mapping(raw_state.get("phase_amounts"))
    phase_molar_volumes = _mapping(raw_state.get("phase_molar_volumes"))
    phase_species_moles = _nested_mapping(raw_state.get("phase_species_moles"))
    species_molar_volumes = _mapping(raw_state.get("species_molar_volumes"))
    species_molar_masses = _mapping(raw_state.get("species_molar_masses"))

    phases = sorted(
        set(phase_masses)
        | set(raw_phase_volumes)
        | set(phase_amounts)
        | set(phase_molar_volumes)
        | set(phase_species_moles)
    )
    reconstructed: dict[str, float] = {}
    report: list[dict[str, Any]] = []

    for phase in phases:
        species_amounts = phase_species_moles.get(phase, {})
        reconstructed_mass = 0.0
        reconstructed_volume = 0.0
        nonzero_species: list[str] = []
        for species, amount in species_amounts.items():
            n_mol = as_float(amount, 0.0) or 0.0
            if n_mol:
                nonzero_species.append(str(species))
            reconstructed_mass += n_mol * (as_float(species_molar_masses.get(species), 0.0) or 0.0)
            reconstructed_volume += n_mol * (as_float(species_molar_volumes.get(species), 0.0) or 0.0)

        phase_mass = as_float(phase_masses.get(phase), 0.0) or 0.0
        raw_volume = as_float(raw_phase_volumes.get(phase), 0.0) or 0.0
        phase_moles = as_float(phase_amounts.get(phase), 0.0) or 0.0
        phase_molar_volume = as_float(phase_molar_volumes.get(phase), 0.0) or 0.0

        if abs(reconstructed_volume) > 0.0:
            reconstructed[phase] = reconstructed_volume

        mass_abs_error = phase_mass - reconstructed_mass
        volume_abs_error = raw_volume - reconstructed_volume
        density_kg_m3 = None
        if abs(reconstructed_volume) > 0.0:
            density_kg_m3 = phase_mass / reconstructed_volume

        report.append(
            {
                "phase": phase,
                "phase_mass_raw": phase_mass,
                "phase_moles_raw": phase_moles,
                "phase_molar_volume_raw": phase_molar_volume,
                "phase_volume_raw": raw_volume,
                "phase_mass_reconstructed_from_species": reconstructed_mass,
                "phase_volume_reconstructed_from_species": reconstructed_volume,
                "mass_abs_error": mass_abs_error,
                "mass_rel_error": _relative_error(phase_mass, reconstructed_mass),
                "volume_abs_error": volume_abs_error,
                "volume_rel_error": _relative_error(raw_volume, reconstructed_volume),
                "density_from_mass_reconstructed_volume_kg_m3": density_kg_m3,
                "density_from_mass_reconstructed_volume_g_cm3": None if density_kg_m3 is None else density_kg_m3 / 1000.0,
                "raw_volume_zero_but_reconstructed_nonzero": abs(raw_volume) <= 1.0e-18
                and abs(reconstructed_volume) > 1.0e-18
                and abs(phase_mass) > 1.0e-18,
                "raw_volume_positive_but_reconstruction_missing": abs(raw_volume) > 1.0e-18
                and abs(reconstructed_volume) <= 1.0e-18,
                "phase_mass_matches_species_mass": _relative_error(phase_mass, reconstructed_mass) < 1.0e-8
                or abs(mass_abs_error) < 1.0e-12,
                "phase_volume_matches_species_volume": _relative_error(raw_volume, reconstructed_volume) < 1.0e-6
                or abs(volume_abs_error) < 1.0e-12,
                "species_count": len(species_amounts),
                "nonzero_species": ";".join(nonzero_species),
            }
        )

    summary = {
        "phase_count": len(report),
        "reconstructed_nonzero_count": sum(1 for value in reconstructed.values() if abs(value) > 0.0),
        "raw_volume_zero_but_reconstructed_nonzero_count": sum(
            1 for row in report if row["raw_volume_zero_but_reconstructed_nonzero"]
        ),
        "raw_volume_positive_but_reconstruction_missing_count": sum(
            1 for row in report if row["raw_volume_positive_but_reconstruction_missing"]
        ),
        "mass_mismatch_count": sum(1 for row in report if not row["phase_mass_matches_species_mass"]),
        "volume_mismatch_count": sum(1 for row in report if not row["phase_volume_matches_species_volume"]),
        "max_mass_rel_error": max((float(row["mass_rel_error"]) for row in report), default=0.0),
        "max_volume_rel_error": max((float(row["volume_rel_error"]) for row in report), default=0.0),
    }
    return {
        "phase_volumes_reconstructed": reconstructed,
        "phase_volume_reconstruction_report": report,
        "phase_volume_reconstruction_summary": summary,
    }


def add_phase_volume_reconstruction(raw_state: dict[str, Any]) -> dict[str, Any]:
    derived = reconstruct_phase_volumes(raw_state)
    raw_state.update(derived)
    return raw_state


def write_phase_volume_reconstruction_files(raw_dir: Any, raw_state: dict[str, Any]) -> None:
    from pathlib import Path

    from .utils import write_json

    raw_dir = Path(raw_dir)
    if "phase_volumes_reconstructed" not in raw_state:
        add_phase_volume_reconstruction(raw_state)
    volumes = raw_state.get("phase_volumes_reconstructed") or {}
    report = raw_state.get("phase_volume_reconstruction_report") or []
    summary = raw_state.get("phase_volume_reconstruction_summary") or {}
    pd.DataFrame(
        [{"name": str(name), "value": value} for name, value in volumes.items()],
        columns=["name", "value"],
    ).to_csv(raw_dir / "xgems_phase_volumes_reconstructed.csv", index=False)
    pd.DataFrame(report).to_csv(raw_dir / "xgems_phase_volume_reconstruction_report.csv", index=False)
    write_json(raw_dir / "xgems_phase_volume_reconstruction_summary.json", summary)
