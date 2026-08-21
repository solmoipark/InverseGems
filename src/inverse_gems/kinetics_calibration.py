"""User-data SCM kinetics calibration route.

A user supplies degree-of-reaction measurements (CSV with ``scm``, ``age_d``,
``dor`` columns); this module fits any registered kinetics model per SCM and
emits a ready-to-use reaction parameter config. The config carries full
calibration provenance (data hash, fit metrics, bounds) and plugs into every
existing entry point via ``reaction_model_config`` - forward runs, batches,
and DB growth then happen under a new ``reaction_model_signature``, so a
recalibrated parameter set coexists with prior results instead of replacing
them. Rebuilding coverage for the new set is the existing campaign/batch
machinery pointed at the new config; no separate rebuild pipeline is needed.

Validation gates: minimum sample count, fitted-curve monotonicity and [0, 1]
range on the observed age window, and per-SCM R2/RMSE reporting with
explicit extrapolation warnings beyond the data's age range.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .scm_reaction import (
    DEFAULT_KINETICS_MODEL,
    get_scm_kinetics_model,
    scm_alpha,
    scm_kinetics_from_mapping,
)
from .utils import write_json

# Sensible fitting defaults for the built-in five-parameter logistic.
_DEFAULT_INIT: dict[str, dict[str, float]] = {
    DEFAULT_KINETICS_MODEL: {"A": 0.0, "B": 0.8, "C": 20.0, "D": 0.6, "G": 1.0},
}
_DEFAULT_BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    DEFAULT_KINETICS_MODEL: {
        "A": (0.0, 0.2),
        "B": (0.05, 5.0),
        "C": (0.1, 500.0),
        "D": (0.05, 1.0),
        "G": (0.05, 5.0),
    },
}

_COLUMN_ALIASES = {
    "scm": ["scm", "scm_role", "material"],
    "age_d": ["age_d", "age_days", "age"],
    "dor": ["dor", "alpha", "reaction_degree", "degree_of_reaction"],
}


def _resolve_columns(frame: pd.DataFrame) -> dict[str, str]:
    resolved: dict[str, str] = {}
    lowered = {str(c).lower(): str(c) for c in frame.columns}
    for canonical, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lowered:
                resolved[canonical] = lowered[alias]
                break
        else:
            raise ValueError(
                f"Calibration data needs a {canonical!r} column (aliases: {aliases}). "
                f"Found columns: {list(frame.columns)}"
            )
    return resolved


def _normalize_dor(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.dropna().abs().max() > 1.5:  # percent scale
        numeric = numeric / 100.0
    return numeric


def calibrate_scm_kinetics(
    *,
    data_csv: str | Path,
    out: str | Path,
    model: str = DEFAULT_KINETICS_MODEL,
    config_id: str | None = None,
    scms: list[str] | tuple[str, ...] | None = None,
    fixed_params: dict[str, float] | None = None,
    param_init: dict[str, float] | None = None,
    param_bounds: dict[str, tuple[float, float]] | None = None,
    min_points: int = 8,
    make_plot: bool = True,
) -> dict[str, Any]:
    """Fit a registered kinetics model to user DoR data; write a reaction config."""
    from scipy.optimize import curve_fit

    spec = get_scm_kinetics_model(model)  # raises for unknown models
    data_path = Path(data_csv)
    frame = pd.read_csv(data_path)
    columns = _resolve_columns(frame)
    work = pd.DataFrame(
        {
            "scm": frame[columns["scm"]].astype(str).str.strip(),
            "age_d": pd.to_numeric(frame[columns["age_d"]], errors="coerce"),
            "dor": _normalize_dor(frame[columns["dor"]]),
        }
    ).dropna()
    work = work[(work["age_d"] > 0) & (work["dor"] >= 0) & (work["dor"] <= 1.0)]

    fixed = {str(k): float(v) for k, v in (fixed_params or {}).items()}
    free_names = [name for name in spec.required if name not in fixed]
    init_defaults = {**_DEFAULT_INIT.get(model, {}), **(param_init or {})}
    bounds_defaults = {**_DEFAULT_BOUNDS.get(model, {}), **(param_bounds or {})}
    missing_setup = [
        name for name in free_names if name not in init_defaults or name not in bounds_defaults
    ]
    if missing_setup:
        raise ValueError(
            f"Model {model!r} has no default fitting setup for parameter(s) {missing_setup}; "
            "provide param_init and param_bounds for them."
        )

    requested = [str(s) for s in scms] if scms else sorted(work["scm"].unique())
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    fits: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    scm_section: dict[str, dict[str, Any]] = {}
    for scm in requested:
        subset = work[work["scm"] == scm]
        if len(subset) < int(min_points):
            warnings.append(
                f"{scm}: only {len(subset)} usable point(s) (< {min_points}); skipped."
            )
            continue
        t = subset["age_d"].to_numpy()
        y = subset["dor"].to_numpy()

        def _predict(t_values: np.ndarray, *free_values: float) -> np.ndarray:
            params = dict(fixed)
            params.update(dict(zip(free_names, free_values)))
            return np.asarray(spec.fn(np.asarray(t_values, dtype=float), params), dtype=float)

        p0 = [float(init_defaults[name]) for name in free_names]
        lower = [float(bounds_defaults[name][0]) for name in free_names]
        upper = [float(bounds_defaults[name][1]) for name in free_names]
        try:
            popt, _ = curve_fit(_predict, t, y, p0=p0, bounds=(lower, upper), maxfev=20000)
        except Exception as exc:  # noqa: BLE001 - report per-SCM fit failures
            warnings.append(f"{scm}: fit failed: {type(exc).__name__}: {exc}")
            continue
        values = dict(fixed)
        values.update({name: float(v) for name, v in zip(free_names, popt)})

        pred = _predict(t, *popt)
        residual = y - pred
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = float(1 - np.sum(residual**2) / ss_tot) if ss_tot > 0 else float("nan")
        rmse = float(np.sqrt(np.mean(residual**2)))

        grid = np.logspace(np.log10(max(t.min(), 0.05)), np.log10(t.max()), 100)
        curve = np.clip(_predict(grid, *popt), None, None)
        gate_warnings = []
        if np.any(np.diff(curve) < -1.0e-9):
            gate_warnings.append("fitted curve is non-monotonic on the data age range")
        if curve.min() < -1.0e-6 or curve.max() > 1.0 + 1.0e-6:
            gate_warnings.append("fitted curve leaves [0, 1] on the data age range")
        at_bounds = [
            name
            for name, value in zip(free_names, popt)
            if abs(value - bounds_defaults[name][0]) < 1.0e-9
            or abs(value - bounds_defaults[name][1]) < 1.0e-9
        ]
        if at_bounds:
            gate_warnings.append(f"parameter(s) at fit bounds: {at_bounds}")
        for message in gate_warnings:
            warnings.append(f"{scm}: {message}")

        entry: dict[str, Any] = dict(values)
        if model != DEFAULT_KINETICS_MODEL:
            entry = {"model": model, **values}
        # round-trips through the kinetics registry (raises on inconsistency)
        scm_kinetics_from_mapping(entry)
        scm_section[scm] = entry
        fits[scm] = {
            "n_points": int(len(subset)),
            "age_min_d": float(t.min()),
            "age_max_d": float(t.max()),
            "parameters": values,
            "fixed_parameters": fixed,
            "r2": r2,
            "rmse": rmse,
            "gate_warnings": gate_warnings,
            "extrapolation_note": (
                f"Data covers {t.min():g}-{t.max():g} d; predictions outside this window "
                "are extrapolation."
            ),
        }

    if not scm_section:
        raise ValueError(
            "No SCM could be calibrated. " + ("; ".join(warnings) if warnings else "")
        )

    set_id = str(
        config_id
        or f"user_calibrated_{model}_{hashlib.sha256(data_path.read_bytes()).hexdigest()[:8]}"
    )
    config_payload = {
        "id": set_id,
        "scm_reaction": scm_section,
        "calibration": {
            "source_data": str(data_path),
            "source_data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "kinetics_model": model,
            "fitted_scms": sorted(scm_section),
            "fits": {name: {k: v for k, v in fit.items() if k != "parameters"} for name, fit in fits.items()},
        },
    }
    config_path_out = out_dir / f"reaction_parameters.{set_id}.yaml"
    config_path_out.write_text(
        yaml.safe_dump(config_payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    report = {
        "config_path": str(config_path_out),
        "id": set_id,
        "model": model,
        "fits": fits,
        "warnings": warnings,
        "usage": {
            "forward": f"inverse-gems run-forward-query --reaction-model-config {config_path_out} ...",
            "note": (
                "Any run started with this reaction_model_config gets its own "
                "reaction_model_signature; DB entries and surrogates for the new "
                "parameter set coexist with existing ones."
            ),
        },
    }
    write_json(out_dir / "calibration_report.json", report)

    if make_plot:
        report["plot"] = _write_plot(out_dir, work, fits, scm_section, model)
    return report


def _write_plot(
    out_dir: Path,
    work: pd.DataFrame,
    fits: dict[str, dict[str, Any]],
    scm_section: dict[str, dict[str, Any]],
    model: str,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted(fits)
    fig, axes = plt.subplots(1, max(1, len(names)), figsize=(5.2 * max(1, len(names)), 4.2), squeeze=False)
    for ax, name in zip(axes[0], names):
        subset = work[work["scm"] == name]
        ax.scatter(subset["age_d"], subset["dor"], s=18, alpha=0.6, color="#36454F", label="data")
        grid = np.logspace(
            np.log10(max(fits[name]["age_min_d"], 0.05)), np.log10(fits[name]["age_max_d"]), 120
        )
        curve = scm_alpha(list(grid), scm_kinetics_from_mapping(scm_section[name]))
        ax.plot(grid, curve, color="#B85042", lw=2.2, label=f"fit (R2={fits[name]['r2']:.2f})")
        ax.set_xscale("log")
        ax.set_xlabel("Age (days, log)")
        ax.set_ylabel("Degree of reaction (-)")
        ax.set_title(f"{name} - {model}")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.tight_layout()
    path = out_dir / "calibration_fits.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)
