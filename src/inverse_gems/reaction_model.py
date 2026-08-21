from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .chem_hash import file_sha256
from .utils import config_path, load_yaml, project_root, to_jsonable


REACTION_MODEL_SIGNATURE_VERSION = 1
DEFAULT_REACTION_MODEL_ID = "local_reaction_model_v1"


DEFAULT_SIGNATURE_FILES = [
    "src/inverse_gems/pk_model.py",
    "src/inverse_gems/scm_reaction.py",
    "src/inverse_gems/availability_modifier.py",
    "src/inverse_gems/bogue.py",
    "src/inverse_gems/xgems_input_builder.py",
    "configs/materials.yaml",
    "configs/scm_reaction.yaml",
    "configs/c3s_c2s_availability.yaml",
    "configs/species_map.yaml",
]


def _resolve_project_file(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return project_root() / path


def _hash_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(to_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def current_reaction_model_metadata(
    *,
    reaction_model_id: str | None = None,
    reaction_model_config: str | Path | None = None,
    extra_signature_files: list[str | Path] | tuple[str | Path, ...] | None = None,
    reaction_parameters: Any | None = None,
) -> dict[str, Any]:
    """Return deterministic provenance for recipe-to-chemistry projection settings.

    The chemistry hash remains the thermodynamic cache key. This metadata tracks the
    upstream reaction model and configuration that produced a recipe's reacted
    chemistry vector, so changed PK/SCM/availability parameters do not look
    identical at the recipe layer.
    """

    config_path_value = Path(reaction_model_config) if reaction_model_config else config_path("reaction_model.yaml")
    config_data: dict[str, Any] = {}
    if config_path_value.exists():
        config_data = load_yaml(config_path_value)

    parameter_payload: dict[str, Any] | None = None
    if reaction_parameters is not None:
        parameter_payload = reaction_parameters.signature_payload()

    model_id = str(
        reaction_model_id
        or (getattr(reaction_parameters, "id", None) if reaction_parameters is not None else None)
        or config_data.get("id")
        or DEFAULT_REACTION_MODEL_ID
    )
    configured_files = config_data.get("signature_files") or DEFAULT_SIGNATURE_FILES
    signature_files = list(configured_files) + list(extra_signature_files or [])
    file_hashes = {
        str(path): file_sha256(_resolve_project_file(path), fallback=f"missing:{_resolve_project_file(path)}")
        for path in signature_files
    }
    payload = {
        "reaction_model_signature_version": REACTION_MODEL_SIGNATURE_VERSION,
        "reaction_model_id": model_id,
        "reaction_model_config_path": str(config_path_value) if config_path_value.exists() else None,
        "reaction_model_config_hash": file_sha256(config_path_value, fallback=""),
        "reaction_model_config": config_data,
        "resolved_reaction_parameters": parameter_payload,
        "file_hashes": file_hashes,
    }
    return {
        "reaction_model_id": model_id,
        "reaction_model_signature": _hash_json(payload),
        "reaction_model_signature_version": REACTION_MODEL_SIGNATURE_VERSION,
        "reaction_model_payload": payload,
    }
