import json

import pandas as pd
import pytest
import yaml

from inverse_gems.task_query_preview import preview_task_query_file, run_confirmed_task_query


def _write_preview_router_assets(tmp_path):
    profiles = tmp_path / "profiles.yaml"
    profiles.write_text(
        yaml.safe_dump(
            {
                "OPC_slag": {
                    "allowed": ["OPC", "slag", "gypsum"],
                    "bounds": {"OPC": [30, 85], "slag": [15, 70], "gypsum": [0, 8]},
                    "default_age_days": 28,
                    "default_w_b": [0.30, 0.55],
                }
            }
        ),
        encoding="utf-8",
    )
    table = tmp_path / "OPC_slag.csv"
    pd.DataFrame(
        {
            "meta__recipe_id": ["a", "b", "c"],
            "meta__material_system": ["OPC_slag", "OPC_slag", "OPC_slag"],
            "x__OPC": [70.0, 50.0, 30.0],
            "x__slag": [30.0, 50.0, 70.0],
            "x__gypsum": [0.0, 0.0, 0.0],
            "x__w_b": [0.4, 0.4, 0.4],
            "x__age_days": [28.0, 28.0, 28.0],
            "y__porosity": [0.30, 0.35, 0.40],
        }
    ).to_csv(table, index=False)
    (tmp_path / "OPC_slag.csv.schema.json").write_text(
        json.dumps(
            {
                "filters": {"rows_after_status_filter": 3},
                "roles": {"targets": [{"column": "y__porosity", "kind": "scalar", "name": "porosity", "source": "porosity"}]},
                "target_summary": {
                    "y__porosity": {
                        "count": 3,
                        "min": 0.30,
                        "max": 0.40,
                        "mean": 0.35,
                        "median": 0.35,
                        "nonzero_count": 3,
                        "nonzero_fraction": 1.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    bundle = bundle_dir / "model.joblib"
    bundle.write_text("placeholder", encoding="utf-8")
    pd.DataFrame([{"target": "y__porosity", "r2": 0.95, "mae": 0.001}]).to_csv(bundle_dir / "target_metrics.csv", index=False)
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "id": "slag_current",
                        "material_system": "OPC_slag",
                        "age_days": 28.0,
                        "model_table": str(table),
                        "model_bundle": str(bundle),
                        "reaction_model_signature": "current",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return registry, profiles


def test_preview_task_query_forward_writes_json_and_markdown(tmp_path):
    query = tmp_path / "forward_task.yaml"
    query.write_text(
        yaml.safe_dump(
            {
                "name": "forward_preview",
                "task_type": "forward_time_series",
                "forward_query": {
                    "recipe": {"binders": {"OPC": 40, "slag": 30, "fly_ash": 30}, "w_b": 0.4},
                    "age_grid": {"values": [1, 28]},
                    "plots": [],
                    "response_summary": {"phases": ["CNASH"], "scalars": ["pH", "porosity"]},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    out = preview_task_query_file(query=query, out=tmp_path / "preview")

    preview = json.loads((out / "parsed_query_preview.json").read_text(encoding="utf-8"))
    markdown = (out / "parsed_query_preview.md").read_text(encoding="utf-8")
    assert preview["task_type"] == "forward_time_series"
    assert preview["forward_query"]["recipe"]["binder_total_g"] == 100.0
    assert preview["forward_query"]["age_grid"]["values"] == [1.0, 28.0]
    assert "Forward query" in markdown
    assert (out / "task_query.yaml").exists()


def test_preview_task_query_inverse_auto_routes_model(tmp_path):
    registry, profiles = _write_preview_router_assets(tmp_path)
    query = tmp_path / "inverse_task.yaml"
    query.write_text(
        yaml.safe_dump(
            {
                "task_type": "inverse_design",
                "design_query": {
                    "design_space": {"allowed_materials": ["OPC", "slag"], "age_days": 28},
                    "output_constraints": {"porosity": {"max": 0.42}},
                    "preferences": [{"input": "OPC", "direction": "minimize"}],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    out = preview_task_query_file(
        query=query,
        out=tmp_path / "preview_inverse",
        model_registry=registry,
        material_systems_config=profiles,
    )

    preview = json.loads((out / "parsed_query_preview.json").read_text(encoding="utf-8"))
    markdown = (out / "parsed_query_preview.md").read_text(encoding="utf-8")
    routing = preview["inverse_design"]["model_routing"]
    assert routing["status"] == "selected"
    assert routing["selected"]["id"] == "slag_current"
    assert routing["selected"]["material_system"] == "OPC_slag"
    assert routing["selection_explanation"]["selected"]["id"] == "slag_current"
    assert routing["selection_explanation"]["target_support"][0]["target"] == "porosity"
    assert "Top routing candidates" in markdown
    assert "Score breakdown" in markdown
    assert any(item["code"] == "material_system_auto" for item in preview["risks"])


def test_preview_task_query_inverse_describes_ph_uncertainty_policy(tmp_path):
    query = tmp_path / "inverse_ph_task.yaml"
    query.write_text(
        yaml.safe_dump(
            {
                "task_type": "inverse_design",
                "design_query": {
                    "name": "ph_sensitive",
                    "model_table": str(tmp_path / "model.csv"),
                    "model_bundle": str(tmp_path / "model.joblib"),
                    "design_space": {"allowed_materials": ["OPC", "slag"], "age_days": 28},
                    "output_constraints": {"pH": {"min": 12.5}},
                    "preferences": [{"target": "porosity", "direction": "minimize"}],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    out = preview_task_query_file(query=query, out=tmp_path / "preview_ph")

    preview = json.loads((out / "parsed_query_preview.json").read_text(encoding="utf-8"))
    markdown = (out / "parsed_query_preview.md").read_text(encoding="utf-8")
    policy = preview["inverse_design"]["pH_uncertainty_policy"]
    assert policy["resolved_mode"] == "exclude"
    assert policy["pH_used_as_constraint"] is True
    assert "pH uncertainty policy" in markdown
    assert any(item["code"] == "ph_uncertainty_policy" for item in preview["risks"])


def test_run_confirmed_task_query_requires_confirmation(tmp_path):
    query = tmp_path / "forward_task.yaml"
    query.write_text(
        yaml.safe_dump(
            {
                "task_type": "forward_calculation",
                "forward_query": {
                    "recipe": {"binders": {"OPC": 100}, "w_b": 0.4},
                    "age_grid": {"values": [28]},
                    "plots": [],
                },
            }
        ),
        encoding="utf-8",
    )
    preview_dir = preview_task_query_file(query=query, out=tmp_path / "preview")

    with pytest.raises(ValueError, match="confirm"):
        run_confirmed_task_query(preview_dir=preview_dir, out=tmp_path / "run", db=tmp_path / "db", use_mock=True)


def test_run_confirmed_task_query_executes_reviewed_forward_mock(tmp_path):
    query = tmp_path / "forward_task.yaml"
    query.write_text(
        yaml.safe_dump(
            {
                "task_type": "forward_calculation",
                "forward_query": {
                    "recipe": {"binders": {"OPC": 100}, "w_b": 0.4},
                    "age_grid": {"values": [28]},
                    "plots": [],
                    "response_summary": {"scalars": ["pH", "porosity"], "narrative_language": "en"},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    preview_dir = preview_task_query_file(query=query, out=tmp_path / "preview")

    out = run_confirmed_task_query(
        preview_dir=preview_dir,
        out=tmp_path / "confirmed_run",
        db=tmp_path / "db",
        confirmed=True,
        use_mock=True,
        disable_plots=True,
    )

    manifest = json.loads((out / "confirmed_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["confirmed"] is True
    assert manifest["preview_risk_counts"]["error"] == 0
    assert (out / "confirmed_preview" / "task_query.yaml").exists()
    assert (out / "confirmed_preview" / "parsed_query_preview.json").exists()
    assert (out / "forward" / "time_series.csv").exists()
