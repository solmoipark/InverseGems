from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from .chem_hash import file_sha256
from .materials import load_materials
from .recipe import Recipe, parse_recipe
from .reaction_parameters import load_reaction_parameters
from .utils import write_json
from .xgems_input_builder import XGEMSInput, build_xgems_input
from .xgems_runner import XGEMSRunner, XGEMSRunnerError, load_formula_map


def _is_nonzero(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and abs(number) > 0.0


def _amount_rows(amounts_kg: dict[str, float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, amount_kg in sorted(amounts_kg.items()):
        rows.append(
            {
                "input_name": name,
                "amount_kg": float(amount_kg),
                "amount_g": float(amount_kg) * 1000.0,
            }
        )
    return rows


def _dat_lst_report(dat_lst: str | Path | None) -> dict[str, Any]:
    if dat_lst is None:
        return {"ok": False, "path": None, "error": "No dat.lst path provided."}
    path = Path(dat_lst)
    exists = path.exists()
    return {
        "ok": exists,
        "path": str(path),
        "size_bytes": path.stat().st_size if exists else None,
        "sha256": file_sha256(path, fallback=None) if exists else None,
        "error": None if exists else "dat.lst path does not exist.",
    }


def _formula_input_report(amounts_kg: dict[str, float]) -> dict[str, Any]:
    formula_map = load_formula_map()
    attempted = sorted(name for name, amount in amounts_kg.items() if _is_nonzero(amount))
    missing = sorted(name for name in attempted if name not in formula_map)
    elements: dict[str, float] = {}
    expanded: dict[str, dict[str, float]] = {}
    for name in attempted:
        formula = formula_map.get(name)
        if not formula:
            continue
        expanded[name] = dict(formula)
        amount_kg = float(amounts_kg[name])
        for element, coefficient in formula.items():
            elements[element] = elements.get(element, 0.0) + amount_kg * float(coefficient)
    return {
        "ok": not missing,
        "input_mode": "formula",
        "attempted_input_names": attempted,
        "missing_formula_names": missing,
        "used_elements": sorted(elements),
        "formula_map_entry_count": len(formula_map),
        "expanded_formulas": expanded,
    }


def _species_input_report(
    amounts_kg: dict[str, float],
    *,
    available_species: set[str] | None = None,
) -> dict[str, Any]:
    attempted = sorted(name for name, amount in amounts_kg.items() if _is_nonzero(amount))
    missing: list[str] = []
    preview: list[str] | None = None
    if available_species is not None:
        missing = sorted(name for name in attempted if name not in available_species)
        preview = sorted(available_species)[:80]
    return {
        "ok": not missing,
        "input_mode": "species",
        "attempted_species_names": attempted,
        "missing_species_names": missing,
        "available_species_count": None if available_species is None else len(available_species),
        "available_species_preview": preview,
        "available_species_checked": available_species is not None,
    }


def _runner_api_report(
    *,
    dat_lst: str | Path | None,
    temperature_celsius: float,
    gems_class_path: str,
    xgems_input_mode: str,
    instantiate_runner: bool,
) -> tuple[dict[str, Any], set[str] | None]:
    if not instantiate_runner:
        return {"ok": True, "skipped": True}, None
    if dat_lst is None or not Path(dat_lst).exists():
        return {"ok": False, "error": "Cannot instantiate runner without an existing dat.lst."}, None
    try:
        runner = XGEMSRunner(
            dat_lst,
            temperature_celsius=temperature_celsius,
            gems_class_path=gems_class_path,
            input_mode=xgems_input_mode,
        )
        available_species = runner._available_species_names()
        attribute_names = sorted(name for name in dir(runner.gems) if not name.startswith("__"))
        return (
            {
                "ok": True,
                "gems_object_type": type(runner.gems).__name__,
                "gems_class_path": gems_class_path,
                "attribute_count": len(attribute_names),
                "has_equilibrate": hasattr(runner.gems, "equilibrate"),
                "has_clear": hasattr(runner.gems, "clear"),
                "has_add_amt_from_formula": hasattr(runner.gems, "add_amt_from_formula"),
                "has_add_multiple_species_amt": hasattr(runner.gems, "add_multiple_species_amt"),
                "has_add_species_amt": hasattr(runner.gems, "add_species_amt"),
                "has_species_names": hasattr(runner.gems, "species_names"),
                "attribute_names_preview": attribute_names[:120],
            },
            available_species,
        )
    except (XGEMSRunnerError, Exception) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, None


def _water_report(xgems_input: XGEMSInput) -> dict[str, Any]:
    policy = dict(xgems_input.water_policy)
    initial = float(policy.get("initial_water_g") or 0.0)
    equilibrium = float(policy.get("equilibrium_water_g") or 0.0)
    delta = equilibrium - initial
    rel_delta = None if initial == 0 else delta / initial
    adjusted = abs(delta) > 1.0e-12
    flags = []
    if adjusted:
        flags.extend(["xgems_water_adjusted", "pH_uncertain_if_run"])
    if policy.get("mode") != "initial":
        flags.append("xgems_water_policy_not_initial")
    return {
        "initial_water_g": initial,
        "equilibrium_water_g": equilibrium,
        "initial_w_b": policy.get("initial_w_b"),
        "equilibrium_w_b": policy.get("equilibrium_w_b"),
        "delta_water_g": delta,
        "relative_delta": rel_delta,
        "matches_recipe_water": not adjusted,
        "pH_interpretation": "conditional_on_adjusted_xgems_water" if adjusted else "recipe_water",
        "flags": list(dict.fromkeys(flags)),
    }


def _overall_flags(report: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if not (report.get("dat_lst") or {}).get("ok"):
        flags.append("dat_lst_missing")
    input_report = report.get("input_compatibility") or {}
    if input_report.get("missing_formula_names"):
        flags.append("missing_formula_map_entries")
    if input_report.get("missing_species_names"):
        flags.append("missing_xgems_species")
    if not (report.get("runner_api") or {}).get("ok", True):
        flags.append("runner_api_unavailable")
    flags.extend((report.get("water") or {}).get("flags") or [])
    if report.get("xgems_input_warnings"):
        flags.append("xgems_input_warnings")
    return list(dict.fromkeys(flags))


def _write_markdown(report: dict[str, Any], path: Path, *, table_limit: int = 40) -> None:
    amounts = pd.DataFrame(report.get("xgems_species_amount_rows") or [])
    lines = [
        "# xGEMS Input Preflight",
        "",
        f"- ok: `{report.get('ok')}`",
        f"- recipe: `{(report.get('recipe') or {}).get('raw_text')}`",
        f"- dat.lst: `{(report.get('dat_lst') or {}).get('path')}`",
        f"- input mode: `{report.get('xgems_input_mode')}`",
        f"- run mode: `{report.get('run_mode')}`",
        f"- flags: `{', '.join(report.get('flags') or []) or 'none'}`",
        "",
        "## Water",
        "",
        f"- initial water g: `{(report.get('water') or {}).get('initial_water_g')}`",
        f"- xGEMS water g: `{(report.get('water') or {}).get('equilibrium_water_g')}`",
        f"- pH interpretation: `{(report.get('water') or {}).get('pH_interpretation')}`",
        "",
        "## Input Names",
        "",
    ]
    if amounts.empty:
        lines.append("No xGEMS input amounts were generated.")
    else:
        display = amounts.head(table_limit)
        lines.append("| input_name | amount_kg | amount_g |")
        lines.append("| --- | ---: | ---: |")
        for row in display.to_dict(orient="records"):
            lines.append(f"| {row['input_name']} | {row['amount_kg']:.8g} | {row['amount_g']:.8g} |")
        if len(amounts) > len(display):
            lines.append(f"\nShowing {len(display)} of {len(amounts)} input rows.")
    compatibility = report.get("input_compatibility") or {}
    if compatibility.get("used_elements"):
        lines.extend(["", "## Formula Elements", "", "`" + ", ".join(compatibility["used_elements"]) + "`"])
    if compatibility.get("missing_formula_names") or compatibility.get("missing_species_names"):
        lines.extend(
            [
                "",
                "## Missing Inputs",
                "",
                f"- missing formula names: `{compatibility.get('missing_formula_names') or []}`",
                f"- missing species names: `{compatibility.get('missing_species_names') or []}`",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_xgems_input_preflight(
    *,
    recipe_text: str,
    out: str | Path,
    dat_lst: str | Path | None = None,
    run_mode: str = "reacted_only",
    normalize: bool = True,
    allow_non_100: bool = False,
    temperature_celsius: float = 20.0,
    xgems_input_mode: str = "formula",
    gems_class_path: str = "xgems:ChemicalEngineDicts",
    xgems_water_mode: str = "initial",
    xgems_water_factor: float = 1.0,
    xgems_water_g: float | None = None,
    xgems_water_w_b: float | None = None,
    reaction_model_id: str | None = None,
    reaction_model_config: str | Path | None = None,
    instantiate_runner: bool = False,
    table_limit: int = 40,
) -> dict[str, Any]:
    """Build the exact xGEMS input for a recipe and diagnose it before equilibrium."""
    out_dir = Path(out)
    materials = load_materials()
    recipe = parse_recipe(recipe_text, materials=materials, normalize=normalize, allow_non_100=allow_non_100)
    reaction_parameters = load_reaction_parameters(reaction_model_config, reaction_model_id=reaction_model_id)
    xgems_input = build_xgems_input(
        recipe,
        materials=materials,
        run_mode=run_mode,
        temperature_celsius=temperature_celsius,
        reaction_parameters=reaction_parameters,
        xgems_water_mode=xgems_water_mode,
        xgems_water_factor=xgems_water_factor,
        xgems_water_g=xgems_water_g,
        xgems_water_w_b=xgems_water_w_b,
    )
    return write_xgems_input_preflight_artifacts(
        recipe=recipe,
        xgems_input=xgems_input,
        out=out_dir,
        dat_lst=dat_lst,
        run_mode=run_mode,
        temperature_celsius=temperature_celsius,
        xgems_input_mode=xgems_input_mode,
        gems_class_path=gems_class_path,
        reaction_parameter_set=reaction_parameters.to_dict(),
        instantiate_runner=instantiate_runner,
        table_limit=table_limit,
    )


def write_xgems_input_preflight_artifacts(
    *,
    recipe: Recipe | dict[str, Any],
    xgems_input: XGEMSInput,
    out: str | Path,
    dat_lst: str | Path | None = None,
    run_mode: str = "reacted_only",
    temperature_celsius: float = 20.0,
    xgems_input_mode: str = "formula",
    gems_class_path: str = "xgems:ChemicalEngineDicts",
    reaction_parameter_set: dict[str, Any] | None = None,
    instantiate_runner: bool = False,
    table_limit: int = 40,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write pre-equilibrium xGEMS input diagnostics from an already built input."""
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    recipe_payload = recipe.to_dict() if isinstance(recipe, Recipe) else dict(recipe)
    context_data = context or {}
    if context_data.get("use_mock") and dat_lst is None:
        dat_report = {
            "ok": True,
            "path": None,
            "skipped": True,
            "reason": "mock runner does not require dat.lst.",
        }
    else:
        dat_report = _dat_lst_report(dat_lst)
    runner_report, available_species = _runner_api_report(
        dat_lst=dat_lst,
        temperature_celsius=temperature_celsius,
        gems_class_path=gems_class_path,
        xgems_input_mode=xgems_input_mode,
        instantiate_runner=instantiate_runner,
    )
    if xgems_input_mode == "formula":
        input_report = _formula_input_report(xgems_input.species_amounts_kg)
    elif xgems_input_mode == "species":
        input_report = _species_input_report(xgems_input.species_amounts_kg, available_species=available_species)
    else:
        raise ValueError("xgems_input_mode must be 'species' or 'formula'.")

    amount_rows = _amount_rows(xgems_input.species_amounts_kg)
    report: dict[str, Any] = {
        "ok": bool(dat_report.get("ok")) and bool(input_report.get("ok")) and bool(runner_report.get("ok", True)),
        "recipe": recipe_payload,
        "dat_lst": dat_report,
        "run_mode": run_mode,
        "temperature_celsius": temperature_celsius,
        "xgems_input_mode": xgems_input_mode,
        "gems_class_path": gems_class_path,
        "reaction_parameter_set": reaction_parameter_set or {},
        "context": context_data,
        "water": _water_report(xgems_input),
        "input_compatibility": input_report,
        "runner_api": runner_report,
        "xgems_species_amounts_kg": xgems_input.species_amounts_kg,
        "xgems_species_amount_rows": amount_rows,
        "lower_bounds_kg": xgems_input.lower_bounds_kg,
        "unreacted_masses_g": xgems_input.unreacted_masses_g,
        "reaction_degrees": xgems_input.reaction_degrees,
        "xgems_input_warnings": xgems_input.warnings,
        "outputs": {
            "report_json": str(out_dir / "xgems_input_preflight.json"),
            "summary_markdown": str(out_dir / "xgems_input_preflight.md"),
            "amounts_csv": str(out_dir / "xgems_input_amounts.csv"),
            "amounts_json": str(out_dir / "xgems_input_amounts.json"),
        },
    }
    report["flags"] = _overall_flags(report)
    report["ok"] = report["ok"] and not any(
        flag in set(report["flags"])
        for flag in ["missing_formula_map_entries", "missing_xgems_species", "runner_api_unavailable"]
    )

    pd.DataFrame(amount_rows).to_csv(out_dir / "xgems_input_amounts.csv", index=False)
    write_json(out_dir / "xgems_input_amounts.json", amount_rows)
    write_json(out_dir / "xgems_input_preflight.json", report)
    _write_markdown(report, out_dir / "xgems_input_preflight.md", table_limit=table_limit)
    return report
