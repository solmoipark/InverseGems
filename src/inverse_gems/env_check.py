from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path
from typing import Any

from .utils import write_json
from .xgems_runner import XGEMSRunner, XGEMSRunnerError, load_formula_map


CORE_IMPORTS = {
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "pydantic": "pydantic",
    "pyyaml": "yaml",
    "scikit-learn": "sklearn",
    "matplotlib": "matplotlib",
    "typer": "typer",
}


def _check_import(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    version = getattr(module, "__version__", None)
    return {"ok": True, "version": str(version) if version is not None else None}


def _check_gems_class(gems_class_path: str) -> dict[str, Any]:
    module_name, _, class_name = gems_class_path.partition(":")
    if not module_name or not class_name:
        return {"ok": False, "error": "gems_class_path must look like 'module.path:ClassName'."}
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "ok": True,
        "module": module_name,
        "class": class_name,
        "repr": repr(cls),
    }


def check_environment(
    *,
    dat_lst: str | Path | None = None,
    gems_class_path: str = "xgems:ChemicalEngineDicts",
    xgems_input_mode: str = "formula",
    require_xgems: bool = False,
    instantiate_runner: bool = False,
    out: str | Path | None = None,
) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "imports": {name: _check_import(module) for name, module in CORE_IMPORTS.items()},
        "xgems": _check_import("xgems"),
        "dat_lst": None,
        "gems_class": _check_gems_class(gems_class_path),
        "xgems_input_mode": xgems_input_mode,
        "formula_map": None,
        "runner_instantiation": None,
        "recommendations": [],
    }

    dat_path = Path(dat_lst) if dat_lst is not None else None
    if dat_path is None:
        checks["dat_lst"] = {"ok": not instantiate_runner, "path": None, "error": "No dat.lst path provided."}
    else:
        checks["dat_lst"] = {
            "ok": dat_path.exists(),
            "path": str(dat_path),
            "size_bytes": dat_path.stat().st_size if dat_path.exists() else None,
            "error": None if dat_path.exists() else "dat.lst path does not exist.",
        }

    if xgems_input_mode == "formula":
        try:
            formula_map = load_formula_map()
            checks["formula_map"] = {"ok": True, "entry_count": len(formula_map)}
        except Exception as exc:
            checks["formula_map"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    else:
        checks["formula_map"] = {"ok": True, "skipped": True}

    if instantiate_runner:
        if dat_path is None or not dat_path.exists():
            checks["runner_instantiation"] = {"ok": False, "error": "Cannot instantiate runner without an existing dat.lst."}
        else:
            try:
                runner = XGEMSRunner(
                    dat_path,
                    gems_class_path=gems_class_path,
                    input_mode=xgems_input_mode,
                )
                attr_count = len(getattr(runner.gems, "__dir__", lambda: [])())
                checks["runner_instantiation"] = {
                    "ok": True,
                    "gems_object_type": type(runner.gems).__name__,
                    "attribute_count": attr_count,
                }
            except (XGEMSRunnerError, Exception) as exc:
                checks["runner_instantiation"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    else:
        checks["runner_instantiation"] = {"ok": True, "skipped": True}

    required_flags = [
        all(item.get("ok") for item in checks["imports"].values()),
        checks["dat_lst"]["ok"] if dat_lst is not None else True,
        checks["formula_map"]["ok"],
    ]
    if require_xgems or instantiate_runner:
        required_flags.append(checks["xgems"]["ok"])
        required_flags.append(checks["gems_class"]["ok"])
    if instantiate_runner:
        required_flags.append(checks["runner_instantiation"]["ok"])

    if not checks["xgems"]["ok"]:
        checks["recommendations"].append("Install xgems in the active environment for real xGEMS/GEMS runs.")
    missing_core = [name for name, item in checks["imports"].items() if not item.get("ok")]
    if missing_core:
        checks["recommendations"].append(
            "Install missing runtime packages: " + ", ".join(missing_core)
        )
    if dat_lst is not None and not checks["dat_lst"]["ok"]:
        checks["recommendations"].append("Pass an existing dat.lst path.")
    if require_xgems and not checks["xgems"]["ok"]:
        checks["recommendations"].append("On this Windows machine, prefer invoking the xGEMS Conda env python.exe directly.")

    checks["ok"] = all(required_flags)
    if out is not None:
        write_json(out, checks)
    return checks
