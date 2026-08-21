from types import SimpleNamespace

import json
import joblib
import pandas as pd
import yaml

from inverse_gems.openai_task_router import parse_user_request_with_openai, run_user_request_with_openai


class DesignToyEstimator:
    def predict(self, x):
        return pd.DataFrame({"y__porosity": 0.20 + 0.002 * x["x__OPC"].to_numpy()}).to_numpy()


def _valid_forward_task_query() -> str:
    return yaml.safe_dump(
        {
            "task_type": "forward_time_series",
            "forward_query": {
                "recipe": {"binders": {"OPC": 40, "slag": 30, "fly_ash": 30}, "w_b": 0.4},
                "age_grid": {"values": [0.1, 1.0, 28.0]},
                "plots": [{"kind": "phase_volumes", "names": "all_nonzero", "filename": "volumes.png"}],
                "response_summary": {
                    "phases": ["Mock C-S-H raw phase"],
                    "scalars": ["pH", "porosity"],
                },
            },
        },
        sort_keys=False,
    )


def _valid_inverse_auto_task_query() -> str:
    return yaml.safe_dump(
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
    )


def _write_design_router_assets(tmp_path):
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
    joblib.dump(
        {
            "estimator": DesignToyEstimator(),
            "inputs": ["x__OPC", "x__slag", "x__gypsum", "x__w_b", "x__age_days"],
            "targets": ["y__porosity"],
        },
        bundle,
    )
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


class FakeResponses:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.outputs:
            raise AssertionError("No fake response left.")
        return SimpleNamespace(output_text=self.outputs.pop(0), usage={"total_tokens": 10})


class FakeOpenAIClient:
    def __init__(self, outputs):
        self.responses = FakeResponses(outputs)


def test_parse_user_request_with_openai_writes_validated_task_query(tmp_path):
    client = FakeOpenAIClient([_valid_forward_task_query()])

    out = parse_user_request_with_openai(
        user_request="OPC 40 slag 30 fly ash 30 volume vs time",
        out=tmp_path / "parse",
        client=client,
    )

    assert (out / "task_query.yaml").exists()
    assert (out / "llm_task_router_report.json").exists()
    assert (out / "llm_raw_output_attempt_0.txt").exists()
    assert (out / "parsed_query_preview.json").exists()
    assert (out / "parsed_query_preview.md").exists()
    assert client.responses.calls[0]["model"] == "gpt-4.1-mini"


def test_parse_user_request_with_openai_repairs_once(tmp_path):
    client = FakeOpenAIClient(["task_type: forward_time_series\n", _valid_forward_task_query()])

    out = parse_user_request_with_openai(
        user_request="OPC 40 slag 30 fly ash 30 volume vs time",
        out=tmp_path / "parse",
        client=client,
        max_repairs=1,
    )

    assert (out / "task_query.yaml").exists()
    assert (out / "llm_repair_prompt_attempt_1.md").exists()
    assert len(client.responses.calls) == 2


def test_run_user_request_with_openai_mock_executes_forward_query(tmp_path):
    client = FakeOpenAIClient([_valid_forward_task_query()])

    out = run_user_request_with_openai(
        user_request="OPC 40 slag 30 fly ash 30 volume vs time",
        out=tmp_path / "run",
        db=tmp_path / "db",
        client=client,
        use_mock=True,
    )

    assert (out / "openai_parse" / "task_query.yaml").exists()
    assert (out / "task_run" / "forward" / "time_series.csv").exists()
    assert (out / "task_run" / "forward" / "volumes.png").exists()
    assert (out / "task_run" / "forward" / "response_summary.json").exists()
    assert (out / "task_run" / "forward" / "answer.md").exists()
    assert (out / "task_run" / "forward" / "narrative_answer.md").exists()
    assert (out / "openai_task_run_summary.json").exists()


def test_run_user_request_with_openai_auto_routes_inverse_design(tmp_path):
    registry, profiles = _write_design_router_assets(tmp_path)
    client = FakeOpenAIClient([_valid_inverse_auto_task_query()])

    out = run_user_request_with_openai(
        user_request="Use OPC and slag at 28 days; find low-porosity binders with low OPC.",
        out=tmp_path / "run_inverse",
        db=tmp_path / "db",
        client=client,
        model_registry=registry,
        material_systems_config=profiles,
        skip_validation=True,
        use_mock=True,
    )

    routed = yaml.safe_load((out / "task_run" / "routed_query" / "design_query.yaml").read_text(encoding="utf-8"))
    summary = json.loads((out / "openai_task_run_summary.json").read_text(encoding="utf-8"))
    assert routed["material_system"] == "OPC_slag"
    assert routed["model_id"] == "slag_current"
    assert routed["design_space"]["allowed_materials"] == ["OPC", "slag", "gypsum"]
    assert summary["material_systems_config"] == str(profiles)
    parse_preview = json.loads((out / "openai_parse" / "parsed_query_preview.json").read_text(encoding="utf-8"))
    assert parse_preview["inverse_design"]["model_routing"]["selected"]["id"] == "slag_current"
    assert (out / "task_run" / "routed_query" / "model_route_report.json").exists()
    assert (out / "task_run" / "design" / "final_candidates.csv").exists()
