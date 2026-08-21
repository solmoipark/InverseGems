"""One-at-a-time (OAT) sensitivity analysis for reaction-model parameters.

The manuscript's reaction parameters (Parrot-Killoh constants, SCM logistic
parameters, C3S/C2S availability settings) are provisional. This module
quantifies how much the model outputs move when each parameter is perturbed
by a relative delta (default +-20%): each perturbation becomes a reaction
parameter config override, runs through the normal forward pipeline (mock or
real xGEMS), and the tracked outputs (porosity, pH, selected phase volumes,
reaction degrees) are compared against the unperturbed baseline.

Reported per parameter/output: the perturbed values, relative changes, and a
normalized elasticity S = (y(+d) - y(-d)) / (2 d y0) - the paper's tornado
figure comes straight from the emitted CSV.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .cached_forward import run_forward_cached
from .database import read_name_value_csv
from .reaction_parameters import load_reaction_parameters
from .utils import write_json

DEFAULT_PARAMETERS: list[str] = [
    # SCM logistic kinetics
    "scm_reaction.slag.B",
    "scm_reaction.slag.C",
    "scm_reaction.slag.D",
    "scm_reaction.fly_ash.B",
    "scm_reaction.fly_ash.C",
    "scm_reaction.fly_ash.D",
    # Parrot-Killoh constants (per clinker phase)
    "pk_model.constants.K1.C3S",
    "pk_model.constants.N1.C3S",
    "pk_model.constants.K3.C3S",
    "pk_model.constants.K1.C2S",
    "pk_model.constants.K1.C3A",
    "pk_model.constants.ref_fineness",
    # C3S/C2S availability modifier
    "c3s_c2s_availability.eta.slag",
    "c3s_c2s_availability.eta.fly_ash",
    "c3s_c2s_availability.c2s_weight",
]

_AVAILABILITY_FALLBACKS = {"eta": 1.0, "demand_coefficients": 1.0}


def _base_structures() -> dict[str, Any]:
    defaults = load_reaction_parameters(None)
    return {
        "scm_reaction": {name: params.to_dict() for name, params in defaults.scm_parameters.items()},
        "pk_model": {"constants": defaults.pk_parameters.to_dict()},
        "c3s_c2s_availability": defaults.availability_config,
    }


def _lookup_base_value(base: dict[str, Any], path: str) -> float:
    parts = path.split(".")
    node: Any = base
    for index, part in enumerate(parts):
        if isinstance(node, dict) and part in node:
            node = node[part]
            continue
        # fallbacks for availability keys that default implicitly
        if parts[0] == "c3s_c2s_availability" and index >= 1:
            section = parts[1]
            if section in _AVAILABILITY_FALLBACKS:
                return float(_AVAILABILITY_FALLBACKS[section])
        raise KeyError(f"Cannot resolve base value for parameter path {path!r} (stuck at {part!r}).")
    return float(node)


def _set_path(payload: dict[str, Any], path: str, value: float) -> None:
    parts = path.split(".")
    node = payload
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = float(value)


def _run_outputs(
    *,
    recipe_text: str,
    db: Path,
    reaction_model_config: Path | None,
    use_mock: bool,
    dat_lst: str | Path | None,
    track_phases: list[str],
    retry_water_on_failure: bool,
) -> dict[str, Any]:
    result = run_forward_cached(
        recipe_text=recipe_text,
        db=db,
        dat_lst=dat_lst,
        use_mock=use_mock,
        reaction_model_config=reaction_model_config,
        retry_water_on_failure=retry_water_on_failure,
    )
    outputs: dict[str, Any] = {
        "chemistry_status": result["chemistry_status"],
        "porosity": result.get("porosity"),
    }
    chem_dir = Path(result["chemistry_dir"])
    scalars_path = chem_dir / "xgems_scalars_raw.json"
    if scalars_path.exists():
        scalars = json.loads(scalars_path.read_text(encoding="utf-8"))
        outputs["pH"] = scalars.get("pH")
    volumes: dict[str, float] = {}
    for name in ["xgems_phase_volumes_reconstructed.csv", "xgems_phase_volumes_raw.csv"]:
        path = chem_dir / name
        if path.exists():
            volumes = read_name_value_csv(path)
            break
    for phase in track_phases:
        outputs[f"phase_volume__{phase}"] = volumes.get(phase)

    from .database import InverseGemsDatabase

    row = InverseGemsDatabase(db).get_recipe_run(result["recipe_id"]) or {}
    degrees = json.loads(row.get("reaction_degrees_json") or "{}")
    for scm, alpha in (degrees.get("scm") or {}).items():
        outputs[f"alpha_scm__{scm}"] = float(alpha)
    opc = degrees.get("opc") or {}
    if opc:
        outputs["alpha_opc_mean"] = float(sum(map(float, opc.values())) / len(opc))
    return outputs


def run_reaction_parameter_sensitivity(
    *,
    out: str | Path,
    recipes: list[str] | tuple[str, ...],
    parameters: list[str] | tuple[str, ...] | None = None,
    rel_delta: float = 0.2,
    use_mock: bool = False,
    dat_lst: str | Path | None = None,
    track_phases: list[str] | tuple[str, ...] | None = None,
    retry_water_on_failure: bool = True,
) -> dict[str, Any]:
    """Run the OAT sensitivity study; returns the report (also written to disk)."""
    out_dir = Path(out)
    (out_dir / "configs").mkdir(parents=True, exist_ok=True)
    db = out_dir / "db"
    parameter_paths = list(parameters or DEFAULT_PARAMETERS)
    phases = list(track_phases or ["CNASH", "Portlandite", "ettringite"])
    base = _base_structures()

    rows: list[dict[str, Any]] = []
    baselines: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for recipe_text in recipes:
        baselines[recipe_text] = _run_outputs(
            recipe_text=recipe_text,
            db=db,
            reaction_model_config=None,
            use_mock=use_mock,
            dat_lst=dat_lst,
            track_phases=phases,
            retry_water_on_failure=retry_water_on_failure,
        )
        if baselines[recipe_text]["chemistry_status"] != "complete":
            warnings.append(f"baseline failed for recipe: {recipe_text}")

    for path in parameter_paths:
        try:
            base_value = _lookup_base_value(base, path)
        except KeyError as exc:
            warnings.append(str(exc))
            continue
        for direction, sign in [("minus", -1.0), ("plus", 1.0)]:
            perturbed_value = base_value * (1.0 + sign * float(rel_delta))
            payload: dict[str, Any] = {"id": f"sensitivity_{path.replace('.', '_')}_{direction}"}
            _set_path(payload, path, perturbed_value)
            config_path = out_dir / "configs" / f"{payload['id']}.yaml"
            config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
            for recipe_text in recipes:
                try:
                    outputs = _run_outputs(
                        recipe_text=recipe_text,
                        db=db,
                        reaction_model_config=config_path,
                        use_mock=use_mock,
                        dat_lst=dat_lst,
                        track_phases=phases,
                        retry_water_on_failure=retry_water_on_failure,
                    )
                except Exception as exc:  # noqa: BLE001 - record and continue the sweep
                    warnings.append(f"{path} {direction} failed for {recipe_text!r}: {exc}")
                    continue
                baseline = baselines[recipe_text]
                for output_name, value in outputs.items():
                    if output_name == "chemistry_status":
                        continue
                    base_output = baseline.get(output_name)
                    rel_change = None
                    if (
                        value is not None
                        and base_output not in (None, 0)
                        and pd.notna(value)
                        and pd.notna(base_output)
                    ):
                        rel_change = (float(value) - float(base_output)) / abs(float(base_output))
                    rows.append(
                        {
                            "parameter": path,
                            "base_value": base_value,
                            "direction": direction,
                            "perturbed_value": perturbed_value,
                            "rel_delta": rel_delta,
                            "recipe": recipe_text,
                            "output": output_name,
                            "baseline_output": base_output,
                            "perturbed_output": value,
                            "rel_change": rel_change,
                        }
                    )

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "sensitivity_runs.csv", index=False)

    elasticity_rows: list[dict[str, Any]] = []
    if not frame.empty:
        for (parameter, recipe, output), group in frame.groupby(["parameter", "recipe", "output"]):
            plus = group[group.direction == "plus"]
            minus = group[group.direction == "minus"]
            if plus.empty or minus.empty:
                continue
            y_plus, y_minus = plus.iloc[0]["perturbed_output"], minus.iloc[0]["perturbed_output"]
            y_base = plus.iloc[0]["baseline_output"]
            if any(v is None or pd.isna(v) for v in (y_plus, y_minus, y_base)) or float(y_base) == 0:
                continue
            elasticity = (float(y_plus) - float(y_minus)) / (2.0 * float(rel_delta) * abs(float(y_base)))
            max_abs_rel = max(
                abs(v) for v in [plus.iloc[0]["rel_change"], minus.iloc[0]["rel_change"]] if v is not None
            )
            elasticity_rows.append(
                {
                    "parameter": parameter,
                    "recipe": recipe,
                    "output": output,
                    "elasticity": elasticity,
                    "max_abs_rel_change": max_abs_rel,
                }
            )
    elasticity = pd.DataFrame(elasticity_rows)
    elasticity.to_csv(out_dir / "sensitivity_elasticity.csv", index=False)

    report = {
        "recipes": list(recipes),
        "parameters": parameter_paths,
        "rel_delta": rel_delta,
        "use_mock": use_mock,
        "run_count": int(len(frame)),
        "baselines": baselines,
        "warnings": warnings,
        "outputs": {
            "runs_csv": str(out_dir / "sensitivity_runs.csv"),
            "elasticity_csv": str(out_dir / "sensitivity_elasticity.csv"),
        },
    }
    write_json(out_dir / "sensitivity_summary.json", report)
    return report


def write_sensitivity_tornado(
    *,
    elasticity_csv: str | Path,
    out: str | Path,
    outputs: list[str] | tuple[str, ...] = ("porosity", "pH"),
    recipe: str | None = None,
    top_n: int = 12,
) -> Path:
    """Render tornado charts (one panel per output) from the elasticity CSV."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.read_csv(elasticity_csv)
    if recipe is not None:
        frame = frame[frame.recipe == recipe]
    outputs = [o for o in outputs if o in set(frame.output)]
    if not outputs:
        raise ValueError(f"No requested outputs present in {elasticity_csv}.")
    fig, axes = plt.subplots(1, len(outputs), figsize=(6.2 * len(outputs), 5.2), squeeze=False)
    for ax, output in zip(axes[0], outputs):
        subset = (
            frame[frame.output == output]
            .groupby("parameter", as_index=False)
            .agg(elasticity=("elasticity", "mean"))
            .assign(magnitude=lambda d: d.elasticity.abs())
            .sort_values("magnitude")
            .tail(top_n)
        )
        colors = ["#B85042" if v < 0 else "#36454F" for v in subset.elasticity]
        ax.barh(subset.parameter, subset.elasticity, color=colors)
        ax.axvline(0, color="#AEB6BD", lw=1)
        ax.set_title(f"Elasticity of {output}")
        ax.set_xlabel("d(output)/output per d(param)/param")
        ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
