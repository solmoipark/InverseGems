from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def config_path(filename: str) -> Path:
    candidates = [
        project_root() / "configs" / filename,
        Path.cwd() / "configs" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _parse_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_env_file(path: str | Path, *, override: bool = False) -> list[str]:
    """Load simple KEY=VALUE lines from an .env file and return loaded keys."""
    env_path = Path(path)
    if not env_path.exists():
        return []
    loaded: list[str] = []
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            continue
        if not override and key in os.environ:
            continue
        os.environ[key] = _parse_env_value(value)
        loaded.append(key)
    return loaded


def load_project_env(*, override: bool = False) -> list[str]:
    """Load .env from the current working directory or project root."""
    loaded: list[str] = []
    seen: set[Path] = set()
    for candidate in [Path.cwd() / ".env", project_root() / ".env"]:
        resolved = candidate.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        loaded.extend(load_env_file(candidate, override=override))
    return loaded


def to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return to_jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    return value


def write_json(path: str | Path, payload: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def short_hash(payload: Any, length: int = 8) -> str:
    encoded = json.dumps(to_jsonable(payload), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:length]


def timestamp_compact() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def as_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def add_amount(amounts: dict[str, float], species: str, value: float) -> None:
    if abs(value) <= 0.0:
        return
    amounts[species] = amounts.get(species, 0.0) + float(value)
