from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(math.isnan(float(value))) if not isinstance(value, str) else value.strip().lower() in {"", "nan", "none"}
    except (TypeError, ValueError):
        return False


def as_bool(value: Any) -> bool | None:
    if is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "1.0", "yes", "y"}:
        return True
    if text in {"false", "0", "0.0", "no", "n"}:
        return False
    return None


def as_float(value: Any) -> float | None:
    if is_missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def solver_status_is_ok(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text or text in {"none", "nan"}:
        return True
    if any(token in text for token in ["fail", "error", "bad"]):
        return False
    return any(token in text for token in ["success", "ok", "converged", "complete", "solved", "cached"])


def xgems_water_delta(row: Mapping[str, Any]) -> float | None:
    water = as_float(row.get("water_g"))
    xgems_water = as_float(row.get("xgems_water_g"))
    if water is None or xgems_water is None:
        return None
    return xgems_water - water


def xgems_water_matches_recipe(row: Mapping[str, Any], *, tolerance_g: float = 1.0e-9) -> bool | None:
    explicit = as_bool(row.get("xgems_water_matches_recipe"))
    if explicit is not None:
        return explicit
    delta = xgems_water_delta(row)
    if delta is None:
        return None
    return abs(delta) <= tolerance_g


def pH_water_reliable(row: Mapping[str, Any]) -> bool | None:
    explicit = as_bool(row.get("pH_water_reliable"))
    if explicit is not None:
        return explicit
    if as_bool(row.get("solver_rescued")) is True:
        return False
    matches = xgems_water_matches_recipe(row)
    if matches is False:
        return False
    return matches


def uncertainty_flags(row: Mapping[str, Any]) -> list[str]:
    flags: list[str] = []
    chemistry_status = str(row.get("chemistry_status") or "").strip().lower()
    if chemistry_status and chemistry_status not in {"complete", "success", "ok", "cached"}:
        flags.append(f"chemistry_status:{chemistry_status}")
    solver_status = row.get("solver_status")
    if solver_status and not solver_status_is_ok(solver_status):
        flags.append("solver_non_success")
    if as_bool(row.get("solver_rescued")) is True:
        flags.append("solver_rescued")
    if xgems_water_matches_recipe(row) is False:
        flags.append("xgems_water_adjusted")
    reliable = pH_water_reliable(row)
    if reliable is False:
        flags.append("pH_water_uncertain")
    reason = row.get("pH_unreliable_reason")
    if not is_missing(reason):
        flags.append("pH_water_uncertain")
    if as_bool(row.get("out_of_domain")) is True:
        flags.append("out_of_domain")
    if row.get("error_message") and not is_missing(row.get("error_message")):
        flags.append("has_error")
    if row.get("error") and not is_missing(row.get("error")):
        flags.append("has_error")
    return list(dict.fromkeys(flags))


def read_preflight_summary(preflight_dir: str | Path | None) -> dict[str, Any]:
    if not isinstance(preflight_dir, (str, Path)) or not preflight_dir:
        # rows without a preflight (failed/skipped ages) round-trip through
        # CSV as NaN floats
        return {}
    path = Path(preflight_dir) / "xgems_input_preflight.json"
    if not path.exists():
        return {"preflight_dir": str(preflight_dir), "exists": False}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"preflight_dir": str(preflight_dir), "exists": True, "error": f"{type(exc).__name__}: {exc}"}
    water = payload.get("water") or {}
    compatibility = payload.get("input_compatibility") or {}
    return {
        "preflight_dir": str(preflight_dir),
        "exists": True,
        "ok": payload.get("ok"),
        "flags": payload.get("flags") or [],
        "input_mode": payload.get("xgems_input_mode"),
        "attempted_input_count": len(
            compatibility.get("attempted_input_names") or compatibility.get("attempted_species_names") or []
        ),
        "used_elements": compatibility.get("used_elements") or [],
        "water_matches_recipe": water.get("matches_recipe_water"),
        "pH_interpretation": water.get("pH_interpretation"),
    }


def flag_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        raw = row.get("uncertainty_flags")
        if isinstance(raw, str) and raw.strip().lower() not in {"", "nan", "none"}:
            flags = [flag for flag in raw.split(";") if flag]
        else:
            flags = uncertainty_flags(row)
        for flag in flags:
            counts[flag] = counts.get(flag, 0) + 1
    return dict(sorted(counts.items()))
