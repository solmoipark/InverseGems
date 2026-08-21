import json

import pandas as pd
import pytest
import yaml

from inverse_gems.design_query import compile_design_query, design_query_json_schema, validate_design_query_data


def test_compile_design_query_writes_search_and_selection_configs(tmp_path):
    query = tmp_path / "design_query.yaml"
    query.write_text(
        yaml.safe_dump(
            {
                "name": "ordered_user_query",
                "material_system": "OPC_slag",
                "age_days": 28,
                "age_bin": "standard",
                "model_table": "data/model.csv",
                "model_bundle": "reports/model.joblib",
                "inputs": {
                    "OPC": {"min": 30, "max": 85},
                    "slag": {"min": 15, "max": 70},
                    "w_b": {"min": 0.30, "max": 0.55},
                },
                "targets": {
                    "porosity": {"max": 0.40},
                    "C-A-S-H": {"min": 0.038},
                    "ettringite": {"max": 0.012},
                },
                "prediction_errors": {"porosity": {"max_abs": 0.015}},
                "ranking": {"mode": "lexicographic"},
                "preferences": [
                    {"target": "C-A-S-H", "direction": "maximize", "tolerance": 0.001},
                    {"input": "OPC", "direction": "minimize"},
                    {"target": "porosity", "direction": "minimize"},
                ],
                "search_top_k": 50,
                "selection_top_k": 10,
            }
        ),
        encoding="utf-8",
    )

    out = compile_design_query(query=query, out=tmp_path / "compiled")

    search = yaml.safe_load((out / "surrogate_candidate_search.yaml").read_text(encoding="utf-8"))
    selection = yaml.safe_load((out / "candidate_selection.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((out / "design_query_manifest.json").read_text(encoding="utf-8"))

    assert search["constraints"]["metadata"]["material_system"]["include"] == ["OPC_slag"]
    assert search["constraints"]["inputs"]["age_days"]["equals"] == 28.0
    assert search["constraints"]["predicted_targets"]["C-A-S-H"]["min"] == 0.038
    assert search["preferences"][0] == {
        "section": "predicted_targets",
        "name": "C-A-S-H",
        "direction": "maximize",
        "tolerance": 0.001,
    }
    assert selection["constraints"]["validated_targets"]["ettringite"]["max"] == 0.012
    assert selection["constraints"]["prediction_errors"]["porosity"]["max_abs"] == 0.015
    assert selection["preferences"][1] == {"section": "inputs", "name": "OPC", "direction": "minimize"}
    assert manifest["ranking"]["mode"] == "lexicographic"


def test_compile_design_query_resolves_model_paths_from_registry(tmp_path):
    query = tmp_path / "design_query.yaml"
    query.write_text(
        yaml.safe_dump(
            {
                "name": "registry_user_query",
                "material_system": "OPC_slag",
                "age_days": 28,
                "targets": {"porosity": {"max": 0.40}},
                "preferences": [{"target": "porosity", "direction": "minimize"}],
                "search_top_k": 5,
                "selection_top_k": 2,
            }
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "id": "opc_slag_28",
                        "material_system": "OPC_slag",
                        "age_days": 28.0,
                        "model_table": "data/model.csv",
                        "model_bundle": "reports/model.joblib",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    out = compile_design_query(query=query, out=tmp_path / "compiled", model_registry=registry)

    search = yaml.safe_load((out / "surrogate_candidate_search.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((out / "design_query_manifest.json").read_text(encoding="utf-8"))
    assert search["model_table"] == "data/model.csv"
    assert search["model_bundle"] == "reports/model.joblib"
    assert manifest["model_registry_entry"]["id"] == "opc_slag_28"


def test_compile_design_query_resolves_registry_by_reaction_signature(tmp_path):
    query = tmp_path / "design_query.yaml"
    query.write_text(
        yaml.safe_dump(
            {
                "name": "registry_reaction_query",
                "material_system": "OPC_slag",
                "age_days": 28,
                "reaction_model": {"id": "params_b", "signature": "sig_b"},
                "targets": {"porosity": {"max": 0.40}},
            }
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "id": "opc_slag_28_a",
                        "material_system": "OPC_slag",
                        "age_days": 28.0,
                        "reaction_model_id": "params_a",
                        "reaction_model_signature": "sig_a",
                        "model_table": "data/model_a.csv",
                        "model_bundle": "reports/model_a.joblib",
                    },
                    {
                        "id": "opc_slag_28_b",
                        "material_system": "OPC_slag",
                        "age_days": 28.0,
                        "reaction_model_id": "params_b",
                        "reaction_model_signature": "sig_b",
                        "model_table": "data/model_b.csv",
                        "model_bundle": "reports/model_b.joblib",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    out = compile_design_query(query=query, out=tmp_path / "compiled", model_registry=registry)

    search = yaml.safe_load((out / "surrogate_candidate_search.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((out / "design_query_manifest.json").read_text(encoding="utf-8"))
    assert search["model_table"] == "data/model_b.csv"
    assert search["model_bundle"] == "reports/model_b.joblib"
    assert search["reaction_model"] == {"id": "params_b", "signature": "sig_b"}
    assert manifest["model_registry_entry"]["id"] == "opc_slag_28_b"
    assert manifest["requested_reaction_model"]["signature"] == "sig_b"


def test_compile_design_query_prefers_unversioned_registry_entry_without_reaction_request(tmp_path):
    query = tmp_path / "design_query.yaml"
    query.write_text(
        yaml.safe_dump(
            {
                "name": "registry_default_query",
                "material_system": "OPC_slag",
                "age_days": 28,
                "targets": {"porosity": {"max": 0.40}},
            }
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "id": "opc_slag_28_legacy",
                        "material_system": "OPC_slag",
                        "age_days": 28.0,
                        "model_table": "data/legacy.csv",
                        "model_bundle": "reports/legacy.joblib",
                    },
                    {
                        "id": "opc_slag_28_sig_a",
                        "material_system": "OPC_slag",
                        "age_days": 28.0,
                        "reaction_model_id": "params_a",
                        "reaction_model_signature": "sig_a",
                        "model_table": "data/model_a.csv",
                        "model_bundle": "reports/model_a.joblib",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    out = compile_design_query(query=query, out=tmp_path / "compiled", model_registry=registry)

    search = yaml.safe_load((out / "surrogate_candidate_search.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((out / "design_query_manifest.json").read_text(encoding="utf-8"))
    assert search["model_table"] == "data/legacy.csv"
    assert "reaction_model" not in search
    assert manifest["model_registry_entry"]["id"] == "opc_slag_28_legacy"


def test_compile_design_query_rejects_registry_reaction_mismatch(tmp_path):
    query = tmp_path / "design_query.yaml"
    query.write_text(
        yaml.safe_dump(
            {
                "name": "registry_reaction_query",
                "material_system": "OPC_slag",
                "age_days": 28,
                "reaction_model": {"signature": "sig_missing"},
                "targets": {"porosity": {"max": 0.40}},
            }
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "id": "opc_slag_28_a",
                        "material_system": "OPC_slag",
                        "age_days": 28.0,
                        "reaction_model_signature": "sig_a",
                        "model_table": "data/model_a.csv",
                        "model_bundle": "reports/model_a.joblib",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="No registry models match requested reaction model"):
        compile_design_query(query=query, out=tmp_path / "compiled", model_registry=registry)


def _write_availability_test_model(tmp_path):
    table = tmp_path / "model_table.csv"
    pd.DataFrame(
        {
            "x__OPC": [40.0, 60.0, 80.0],
            "y__pH": [12.650, 12.651, 12.652],
            "y__porosity": [0.30, 0.35, 0.42],
        }
    ).to_csv(table, index=False)
    (tmp_path / "model_table.csv.schema.json").write_text(
        json.dumps(
            {
                "filters": {"rows_after_status_filter": 3},
                "roles": {
                    "targets": [
                        {"column": "y__pH", "kind": "scalar", "name": "pH", "source": "scalar__pH"},
                        {"column": "y__porosity", "kind": "scalar", "name": "porosity", "source": "porosity"},
                    ]
                },
                "target_summary": {
                    "y__pH": {
                        "count": 3,
                        "min": 12.650,
                        "max": 12.652,
                        "mean": 12.651,
                        "median": 12.651,
                        "nonzero_count": 3,
                        "nonzero_fraction": 1.0,
                    },
                    "y__porosity": {
                        "count": 3,
                        "min": 0.30,
                        "max": 0.42,
                        "mean": 0.356,
                        "median": 0.35,
                        "nonzero_count": 3,
                        "nonzero_fraction": 1.0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.joblib").write_text("placeholder", encoding="utf-8")
    pd.DataFrame(
        {
            "target": ["y__pH", "y__porosity"],
            "r2": [0.99, 0.95],
            "mae": [0.0001, 0.01],
            "rmse": [0.0001, 0.01],
            "nonzero_true_count": [1, 1],
        }
    ).to_csv(bundle / "target_metrics.csv", index=False)
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "id": "opc_test_28",
                        "material_system": "OPC_test",
                        "age_days": 28.0,
                        "model_table": str(table),
                        "model_bundle": str(bundle / "model.joblib"),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return registry


def test_compile_design_query_writes_target_availability_warning_report(tmp_path):
    registry = _write_availability_test_model(tmp_path)
    query = tmp_path / "query.yaml"
    query.write_text(
        yaml.safe_dump(
            {
                "material_system": "OPC_test",
                "age_days": 28,
                "targets": {"pH": {"min": 12.65}, "porosity": {"max": 0.40}},
                "preferences": [{"target": "pH", "direction": "maximize"}],
            }
        ),
        encoding="utf-8",
    )

    out = compile_design_query(query=query, out=tmp_path / "compiled", model_registry=registry)

    report = json.loads((out / "target_availability_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "design_query_manifest.json").read_text(encoding="utf-8"))
    assert report["policy"] == "warn"
    assert any(issue["target"] == "pH" and issue["status"] == "not_recommended" for issue in report["issues"])
    assert manifest["target_availability"]["issues"] == report["issues"]


def test_compile_design_query_can_error_on_unavailable_target(tmp_path):
    registry = _write_availability_test_model(tmp_path)
    query = tmp_path / "query.yaml"
    query.write_text(
        yaml.safe_dump(
            {
                "material_system": "OPC_test",
                "age_days": 28,
                "targets": {"pH": {"min": 12.65}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Target availability check failed"):
        compile_design_query(
            query=query,
            out=tmp_path / "compiled",
            model_registry=registry,
            target_availability_policy="error",
        )


def test_compile_target_first_design_query_contract(tmp_path):
    query = tmp_path / "target_first_design_query.yaml"
    query.write_text(
        yaml.safe_dump(
            {
                "name": "target_first_query",
                "model_table": "data/model.csv",
                "model_bundle": "reports/model.joblib",
                "design_space": {
                    "material_systems": ["OPC_slag"],
                    "allowed_materials": ["OPC", "slag", "gypsum"],
                    "input_constraints": {
                        "OPC": {"max": 40},
                        "w_b": {"min": 0.30, "max": 0.50},
                    },
                    "age_days": 28,
                },
                "output_constraints": {
                    "porosity": {"max": 0.38},
                    "ettringite": {"max": 0.01},
                    "C-A-S-H": {"min": 0.04},
                },
                "objectives": [
                    {"target": "C-A-S-H", "direction": "maximize"},
                    {"input": "OPC", "direction": "minimize"},
                    {"target": "porosity", "direction": "minimize"},
                ],
                "validation": {
                    "top_k_xgems": 12,
                    "search_top_k": 60,
                    "use_thermo_cache": True,
                    "use_nearest_neighbors": True,
                    "prediction_errors": {"porosity": {"max_abs": 0.02}},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    out = compile_design_query(query=query, out=tmp_path / "compiled_target_first")

    search = yaml.safe_load((out / "surrogate_candidate_search.yaml").read_text(encoding="utf-8"))
    selection = yaml.safe_load((out / "candidate_selection.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((out / "design_query_manifest.json").read_text(encoding="utf-8"))

    assert search["top_k"] == 60
    assert selection["top_k"] == 12
    assert search["constraints"]["metadata"]["material_system"]["include"] == ["OPC_slag"]
    assert search["constraints"]["inputs"]["age_days"]["equals"] == 28.0
    assert search["constraints"]["inputs"]["OPC"]["max"] == 40
    assert search["constraints"]["inputs"]["fly_ash"]["max"] == 0.0
    assert search["constraints"]["inputs"]["limestone"]["max"] == 0.0
    assert "slag" not in search["constraints"]["inputs"]
    assert search["constraints"]["predicted_targets"]["porosity"]["max"] == 0.38
    assert selection["constraints"]["validated_targets"]["ettringite"]["max"] == 0.01
    assert selection["constraints"]["prediction_errors"]["porosity"]["max_abs"] == 0.02
    assert search["preferences"][0] == {
        "section": "predicted_targets",
        "name": "C-A-S-H",
        "direction": "maximize",
    }
    assert selection["preferences"][1] == {"section": "inputs", "name": "OPC", "direction": "minimize"}
    assert manifest["design_space"]["allowed_materials"] == ["OPC", "slag", "gypsum"]
    assert manifest["validation"]["top_k_xgems"] == 12


def test_target_first_design_query_resolves_registry_from_design_space(tmp_path):
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "id": "opc_slag_28",
                        "material_system": "OPC_slag",
                        "age_days": 28.0,
                        "model_table": "data/model.csv",
                        "model_bundle": "reports/model.joblib",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    query = {
        "design_space": {"material_systems": "OPC_slag", "age_days": 28},
        "output_constraints": {"porosity": {"max": 0.4}},
    }

    validated = validate_design_query_data(query, model_registry=registry, require_model_paths=True)

    assert validated["design_space"]["material_systems"] == "OPC_slag"


def test_design_query_resolves_continuous_age_registry_entry(tmp_path):
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "id": "global_opc_fly_ash",
                        "material_system": "OPC_fly_ash",
                        "age_min_days": 0.1,
                        "age_max_days": 365.0,
                        "model_table": "data/global.csv",
                        "model_bundle": "reports/global/model.joblib",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    query = {
        "design_space": {"material_systems": "OPC_fly_ash", "age_days": 100},
        "output_constraints": {"porosity": {"max": 0.5}},
    }

    validated = validate_design_query_data(query, model_registry=registry, require_model_paths=True)

    assert validated["design_space"]["material_systems"] == "OPC_fly_ash"


def test_design_query_schema_contains_llm_contract_fields():
    schema = design_query_json_schema()
    assert schema["title"] == "DesignQuerySpec"
    assert "properties" in schema
    assert "design_space" in schema["properties"]
    assert "output_constraints" in schema["properties"]
    assert "validation" in schema["properties"]
    assert "preferences" in schema["properties"]
    assert "ranking" in schema["properties"]
    assert "additionalProperties" in schema
    assert schema["additionalProperties"] is False


def test_design_query_validation_rejects_ambiguous_preference():
    with pytest.raises(ValueError, match="Preference cannot set both maximize and minimize"):
        validate_design_query_data(
            {
                "name": "bad_query",
                "preferences": [{"maximize": "C-A-S-H", "minimize": "porosity"}],
            }
        )


def test_design_query_validation_rejects_unknown_root_field():
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        validate_design_query_data({"name": "bad_query", "llm_free_text": "please optimize everything"})


def test_design_query_validation_require_model_paths_allows_registry(tmp_path):
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "id": "opc_slag_28",
                        "material_system": "OPC_slag",
                        "age_days": 28.0,
                        "model_table": "data/model.csv",
                        "model_bundle": "reports/model.joblib",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    validated = validate_design_query_data(
        {"material_system": "OPC_slag", "age_days": 28},
        model_registry=registry,
        require_model_paths=True,
    )

    assert validated["material_system"] == "OPC_slag"
