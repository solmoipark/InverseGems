import json

import joblib
import pandas as pd
import pytest
import yaml

from inverse_gems.model_router import route_design_query
from inverse_gems.task_query import run_task_query


class RouterToyEstimator:
    def predict(self, x):
        porosity = 0.20 + 0.002 * x["x__OPC"].to_numpy()
        return pd.DataFrame({"y__porosity": porosity}).to_numpy()


def _write_profiles(path):
    path.write_text(
        yaml.safe_dump(
            {
                "OPC_slag": {
                    "allowed": ["OPC", "slag", "gypsum"],
                    "bounds": {"OPC": [30, 85], "slag": [15, 70], "gypsum": [0, 8]},
                    "default_age_days": 28,
                    "default_w_b": [0.30, 0.55],
                },
                "OPC_slag_fly_ash": {
                    "allowed": ["OPC", "slag", "fly_ash", "gypsum"],
                    "bounds": {"OPC": [25, 70], "slag": [10, 55], "fly_ash": [10, 55], "gypsum": [0, 8]},
                    "default_age_days": 28,
                    "default_w_b": [0.30, 0.55],
                },
                "OPC_fly_ash": {
                    "allowed": ["OPC", "fly_ash", "gypsum"],
                    "bounds": {"OPC": [25, 85], "fly_ash": [15, 70], "gypsum": [0, 8]},
                    "default_age_days": 28,
                    "default_w_b": [0.30, 0.55],
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_model_assets(tmp_path, system, *, targets=None, ph_caution=False, estimator=False):
    targets = targets or {"y__porosity": {"name": "porosity", "values": [0.30, 0.35, 0.40]}}
    table = tmp_path / f"{system}.csv"
    data = {
        "meta__recipe_id": [f"{system}_a", f"{system}_b", f"{system}_c"],
        "meta__material_system": [system, system, system],
        "x__OPC": [70.0, 50.0, 30.0],
        "x__slag": [30.0, 50.0, 70.0],
        "x__fly_ash": [0.0, 0.0, 0.0],
        "x__gypsum": [0.0, 0.0, 0.0],
        "x__w_b": [0.4, 0.4, 0.4],
        "x__age_days": [28.0, 28.0, 28.0],
    }
    for column, spec in targets.items():
        data[column] = spec["values"]
    if system == "OPC_slag_fly_ash":
        data["x__fly_ash"] = [10.0, 20.0, 30.0]
    if system == "OPC_fly_ash":
        data["x__slag"] = [0.0, 0.0, 0.0]
        data["x__fly_ash"] = [30.0, 50.0, 70.0]
    pd.DataFrame(data).to_csv(table, index=False)

    target_summary = {}
    role_targets = []
    metric_rows = []
    for column, spec in targets.items():
        values = spec["values"]
        target_summary[column] = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "median": sorted(values)[len(values) // 2],
            "nonzero_count": sum(abs(value) > 1.0e-30 for value in values),
            "nonzero_fraction": 1.0,
        }
        if column == "y__pH" and ph_caution:
            target_summary[column]["pH_water_unreliable_count"] = 1
            target_summary[column]["pH_water_unreliable_fraction"] = 1.0 / len(values)
        kind = "scalar" if column in {"y__porosity", "y__pH"} else "phase_amount_group"
        source = "porosity" if column == "y__porosity" else "scalar__pH" if column == "y__pH" else f"phase_amount_group__{spec['name']}"
        role_targets.append({"column": column, "kind": kind, "name": spec["name"], "source": source})
        metric_rows.append({"target": column, "r2": 0.95, "mae": 0.001})
    (tmp_path / f"{system}.csv.schema.json").write_text(
        json.dumps({"filters": {"rows_after_status_filter": 3}, "roles": {"targets": role_targets}, "target_summary": target_summary}),
        encoding="utf-8",
    )

    bundle_dir = tmp_path / f"{system}_bundle"
    bundle_dir.mkdir()
    bundle = bundle_dir / "model.joblib"
    if estimator:
        joblib.dump(
            {
                "estimator": RouterToyEstimator(),
                "inputs": ["x__OPC", "x__slag", "x__fly_ash", "x__gypsum", "x__w_b", "x__age_days"],
                "targets": ["y__porosity"],
            },
            bundle,
        )
    else:
        bundle.write_text("placeholder", encoding="utf-8")
    pd.DataFrame(metric_rows).to_csv(bundle_dir / "target_metrics.csv", index=False)
    return table, bundle


def _write_registry(path, entries):
    path.write_text(yaml.safe_dump({"models": entries}), encoding="utf-8")
    return path


def test_route_design_query_selects_specific_allowed_material_system(tmp_path):
    profiles = _write_profiles(tmp_path / "profiles.yaml")
    slag_table, slag_bundle = _write_model_assets(
        tmp_path,
        "OPC_slag",
        targets={
            "y__porosity": {"name": "porosity", "values": [0.30, 0.35, 0.40]},
            "y__amount_C_A_S_H": {"name": "C-A-S-H", "values": [0.04, 0.05, 0.06]},
        },
    )
    ternary_table, ternary_bundle = _write_model_assets(
        tmp_path,
        "OPC_slag_fly_ash",
        targets={
            "y__porosity": {"name": "porosity", "values": [0.31, 0.36, 0.41]},
            "y__amount_C_A_S_H": {"name": "C-A-S-H", "values": [0.04, 0.05, 0.06]},
        },
    )
    registry = _write_registry(
        tmp_path / "registry.yaml",
        [
            {
                "id": "slag",
                "material_system": "OPC_slag",
                "age_days": 28.0,
                "model_table": str(slag_table),
                "model_bundle": str(slag_bundle),
                "reaction_model_signature": "current",
            },
            {
                "id": "slag_fly_ash",
                "material_system": "OPC_slag_fly_ash",
                "age_days": 28.0,
                "model_table": str(ternary_table),
                "model_bundle": str(ternary_bundle),
                "reaction_model_signature": "current",
            },
        ],
    )
    query = {
        "design_space": {"allowed_materials": ["OPC", "slag", "fly_ash"], "age_days": 28},
        "output_constraints": {"porosity": {"max": 0.4}},
        "preferences": [{"target": "C-A-S-H", "direction": "maximize"}],
    }

    result = route_design_query(query, model_registry=registry, material_systems_config=profiles)

    assert result["selected"]["material_system"] == "OPC_slag_fly_ash"
    assert result["routed_query"]["material_system"] == "OPC_slag_fly_ash"
    assert result["routed_query"]["model_id"] == "slag_fly_ash"
    assert result["routed_query"]["design_space"]["allowed_materials"] == ["OPC", "slag", "fly_ash", "gypsum"]
    explanation = result["selection_explanation"]
    assert explanation["selected"]["id"] == "slag_fly_ash"
    assert explanation["selected"]["material_score"] == result["selected"]["material_score"]
    assert "highest-ranked eligible model" in explanation["summary"]
    assert explanation["top_alternatives"][0]["id"] == "slag_fly_ash"


def test_route_design_query_blocks_caution_ph_unless_allowed(tmp_path):
    profiles = _write_profiles(tmp_path / "profiles.yaml")
    table, bundle = _write_model_assets(
        tmp_path,
        "OPC_fly_ash",
        targets={"y__pH": {"name": "pH", "values": [12.8, 13.0, 13.2]}},
        ph_caution=True,
    )
    registry = _write_registry(
        tmp_path / "registry.yaml",
        [
            {
                "id": "fly_ash",
                "material_system": "OPC_fly_ash",
                "age_days": 28.0,
                "model_table": str(table),
                "model_bundle": str(bundle),
            }
        ],
    )
    query = {
        "design_space": {"allowed_materials": ["OPC", "fly_ash"], "age_days": 28},
        "output_constraints": {"pH": {"min": 13.0}},
    }

    with pytest.raises(ValueError, match="No eligible"):
        route_design_query(query, model_registry=registry, material_systems_config=profiles)

    routed = route_design_query(query, model_registry=registry, material_systems_config=profiles, target_policy="allow_caution")
    assert routed["selected"]["material_system"] == "OPC_fly_ash"
    assert routed["selected"]["matched_targets"][0]["status"] == "usable_with_caution"


def test_route_design_query_accepts_continuous_age_registry_entry(tmp_path):
    profiles = _write_profiles(tmp_path / "profiles.yaml")
    table, bundle = _write_model_assets(tmp_path, "OPC_fly_ash")
    registry = _write_registry(
        tmp_path / "registry.yaml",
        [
            {
                "id": "global_fly_ash",
                "material_system": "OPC_fly_ash",
                "age_min_days": 0.1,
                "age_max_days": 365.0,
                "model_table": str(table),
                "model_bundle": str(bundle),
                "reaction_model_signature": "current",
            }
        ],
    )
    query = {
        "design_space": {"allowed_materials": ["OPC", "fly_ash"], "age_days": 100},
        "output_constraints": {"porosity": {"max": 0.5}},
    }

    routed = route_design_query(query, model_registry=registry, material_systems_config=profiles)

    assert routed["selected"]["id"] == "global_fly_ash"
    assert routed["selected"]["age_days"] == 100.0
    assert routed["selected"]["age_score"] == 0.0
    assert routed["routed_query"]["model_id"] == "global_fly_ash"


def test_run_task_query_auto_routes_inverse_design(tmp_path):
    profiles = _write_profiles(tmp_path / "profiles.yaml")
    table, bundle = _write_model_assets(tmp_path, "OPC_slag", estimator=True)
    registry = _write_registry(
        tmp_path / "registry.yaml",
        [
            {
                "id": "slag_current",
                "material_system": "OPC_slag",
                "age_days": 28.0,
                "model_table": str(table),
                "model_bundle": str(bundle),
                "reaction_model_signature": "current",
            }
        ],
    )
    task = tmp_path / "task.yaml"
    task.write_text(
        yaml.safe_dump(
            {
                "task_type": "inverse_design",
                "design_query": {
                    "design_space": {"allowed_materials": ["OPC", "slag"], "age_days": 28},
                    "output_constraints": {"porosity": {"max": 2.0}},
                    "preferences": [{"input": "OPC", "direction": "minimize"}],
                    "search_top_k": 1,
                    "selection_top_k": 1,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    out = run_task_query(
        query=task,
        out=tmp_path / "out",
        db=tmp_path / "db",
        model_registry=registry,
        material_systems_config=profiles,
        skip_validation=True,
    )

    routed = yaml.safe_load((out / "routed_query" / "design_query.yaml").read_text(encoding="utf-8"))
    summary = json.loads((out / "task_query_summary.json").read_text(encoding="utf-8"))
    candidates = pd.read_csv(out / "design" / "final_candidates.csv")
    assert routed["material_system"] == "OPC_slag"
    assert routed["model_id"] == "slag_current"
    assert summary["model_route_report"].endswith("model_route_report.json")
    assert candidates.iloc[0]["meta__material_system"] == "OPC_slag"
