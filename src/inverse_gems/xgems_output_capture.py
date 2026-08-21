from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .phase_volume_reconstruction import write_phase_volume_reconstruction_files
from .utils import short_hash, timestamp_compact, to_jsonable, write_json


EXPECTED_OUTPUT_FILES = [
    "manifest.json",
    "input_user_request.txt",
    "input_recipe.json",
    "input_materials_used.json",
    "input_reaction_degrees.json",
    "input_xgems_species_amounts.json",
    "run_provenance.json",
    "xgems_phase_amounts_raw.csv",
    "xgems_phase_volumes_raw.csv",
    "xgems_phase_volumes_reconstructed.csv",
    "xgems_phase_volume_reconstruction_report.csv",
    "xgems_phase_volume_reconstruction_summary.json",
    "xgems_aqueous_species_raw.csv",
    "xgems_scalars_raw.json",
    "xgems_attribute_report.json",
    "xgems_stdout.txt",
    "xgems_stderr.txt",
    "porosity.json",
    "warnings.json",
]


def create_run_directory(out_base: str | Path, identity_payload: Any) -> Path:
    out_base = Path(out_base)
    out_base.mkdir(parents=True, exist_ok=True)
    run_dir = out_base / f"run_{timestamp_compact()}_{short_hash(identity_payload)}"
    suffix = 1
    while run_dir.exists():
        run_dir = out_base / f"{run_dir.name}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def _dict_to_csv(path: Path, data: Any) -> None:
    if isinstance(data, dict):
        rows = [{"name": str(name), "value": to_jsonable(value)} for name, value in data.items()]
    elif data is None:
        rows = []
    else:
        rows = [{"name": "raw", "value": to_jsonable(data)}]
    pd.DataFrame(rows, columns=["name", "value"]).to_csv(path, index=False)


def save_run_outputs(
    *,
    out_base: str | Path,
    manifest: dict[str, Any],
    user_request: str,
    recipe: dict[str, Any],
    materials_used: dict[str, Any],
    reaction_degrees: dict[str, Any],
    xgems_species_amounts: dict[str, float],
    raw_state: dict[str, Any],
    porosity: dict[str, Any],
    warnings: list[str] | dict[str, Any],
    stdout_text: str = "",
    stderr_text: str = "",
    run_provenance: dict[str, Any] | None = None,
) -> Path:
    identity = {"recipe": recipe, "species": xgems_species_amounts, "manifest": manifest}
    run_dir = create_run_directory(out_base, identity)
    manifest = dict(manifest)
    manifest["expected_files"] = EXPECTED_OUTPUT_FILES

    write_json(run_dir / "manifest.json", manifest)
    (run_dir / "input_user_request.txt").write_text(user_request, encoding="utf-8")
    write_json(run_dir / "input_recipe.json", recipe)
    write_json(run_dir / "input_materials_used.json", materials_used)
    write_json(run_dir / "input_reaction_degrees.json", reaction_degrees)
    write_json(run_dir / "input_xgems_species_amounts.json", xgems_species_amounts)
    write_json(run_dir / "run_provenance.json", run_provenance or {})

    _dict_to_csv(run_dir / "xgems_phase_amounts_raw.csv", raw_state.get("phase_masses"))
    _dict_to_csv(run_dir / "xgems_phase_volumes_raw.csv", raw_state.get("phase_volumes"))
    write_phase_volume_reconstruction_files(run_dir, raw_state)
    _dict_to_csv(run_dir / "xgems_aqueous_species_raw.csv", raw_state.get("aqueous_species"))
    write_json(run_dir / "xgems_scalars_raw.json", raw_state.get("scalars") or {})
    write_json(run_dir / "xgems_attribute_report.json", raw_state.get("attribute_report") or {})
    (run_dir / "xgems_stdout.txt").write_text(stdout_text, encoding="utf-8")
    (run_dir / "xgems_stderr.txt").write_text(stderr_text, encoding="utf-8")
    write_json(run_dir / "porosity.json", porosity)
    write_json(run_dir / "warnings.json", warnings if isinstance(warnings, dict) else {"warnings": warnings})
    return run_dir
