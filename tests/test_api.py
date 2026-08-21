from types import SimpleNamespace

import json
import yaml

import inverse_gems.api as api_module
from inverse_gems.api import parse_request_preview, run_confirmed_request, run_forward_request, run_request
from inverse_gems.cli import main
from inverse_gems.task_query_preview import preview_task_query_file


def _forward_query_yaml(path):
    path.write_text(
        yaml.safe_dump(
            {
                "name": "api_forward",
                "recipe": {"binders": {"OPC": 40, "slag": 30, "fly_ash": 30}, "w_b": 0.4},
                "age_grid": {"values": [1.0, 28.0]},
                "plots": [],
                "response_summary": {
                    "phases": ["Mock C-S-H raw phase"],
                    "scalars": ["pH", "porosity"],
                    "narrative_language": "en",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _task_query_yaml(path):
    path.write_text(
        yaml.safe_dump(
            {
                "name": "api_task",
                "task_type": "forward_time_series",
                "forward_query": {
                    "recipe": {"binders": {"OPC": 40, "slag": 30, "fly_ash": 30}, "w_b": 0.4},
                    "age_grid": {"values": [1.0, 28.0]},
                    "plots": [],
                    "response_summary": {
                        "phases": ["Mock C-S-H raw phase"],
                        "scalars": ["pH", "porosity"],
                        "narrative_language": "en",
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


class FakeResponses:
    def __init__(self, output):
        self.output = output

    def create(self, **kwargs):
        return SimpleNamespace(output_text=self.output, usage={"total_tokens": 10})


class FakeOpenAIClient:
    def __init__(self, output):
        self.responses = FakeResponses(output)


def test_run_forward_request_mock_returns_standard_result(tmp_path):
    query = _forward_query_yaml(tmp_path / "forward.yaml")

    result = run_forward_request(forward_query=query, out=tmp_path / "request", db=tmp_path / "db", use_mock=True)

    assert result.status == "complete"
    assert result.task_type == "forward_calculation_or_time_series"
    assert "Forward time_series completed successfully" in result.answer_text
    assert result.missing_outputs == {"phases": [], "scalars": []}
    assert (tmp_path / "request" / "request_result.json").exists()
    assert (tmp_path / "request" / "forward" / "narrative_answer.md").exists()


def test_run_request_task_query_mock_returns_standard_result(tmp_path):
    query = _task_query_yaml(tmp_path / "task.yaml")

    result = run_request(task_query=query, out=tmp_path / "request", db=tmp_path / "db", use_mock=True)

    assert result.status == "complete"
    assert result.task_type == "forward_time_series"
    assert result.files["narrative_answer_md"].endswith("narrative_answer.md")
    assert "Mock C-S-H raw phase.mass" in result.answer_text


def test_run_request_openai_mock_uses_parser_then_local_execution(tmp_path):
    query_text = _task_query_yaml(tmp_path / "task.yaml").read_text(encoding="utf-8")
    client = FakeOpenAIClient(query_text)

    result = run_request(
        request="OPC 40 slag 30 fly ash 30, show pH and porosity",
        out=tmp_path / "request",
        db=tmp_path / "db",
        use_mock=True,
        use_openai=True,
        client=client,
    )

    assert result.status == "complete"
    assert result.task_type == "forward_time_series"
    assert (tmp_path / "request" / "openai_parse" / "task_query.yaml").exists()
    assert (tmp_path / "request" / "request_result.json").exists()


def test_parse_request_preview_openai_mock_returns_standard_result(tmp_path):
    query_text = _task_query_yaml(tmp_path / "task.yaml").read_text(encoding="utf-8")
    client = FakeOpenAIClient(query_text)

    result = parse_request_preview(
        request="OPC 40 slag 30 fly ash 30, show pH and porosity",
        out=tmp_path / "preview_request",
        client=client,
    )

    assert result.status == "complete"
    assert result.task_type == "forward_time_series"
    assert result.summary["mode"] == "parse_preview"
    assert result.summary["risk_counts"]["error"] == 0
    assert result.files["parsed_query_preview_json"].endswith("parsed_query_preview.json")
    assert (tmp_path / "preview_request" / "request_result.json").exists()


def test_run_request_requires_openai_for_free_text(tmp_path):
    try:
        run_request(request="OPC 100, age 28", out=tmp_path / "request", db=tmp_path / "db", use_mock=True)
    except ValueError as exc:
        assert "requires use_openai=True" in str(exc)
    else:
        raise AssertionError("Expected ValueError for free text without OpenAI.")


def test_run_request_forwards_auto_routing_options_to_openai_runner(tmp_path, monkeypatch):
    captured = {}

    def fake_run_user_request_with_openai(**kwargs):
        captured.update(kwargs)
        out = kwargs["out"]
        task_run = out / "task_run"
        task_run.mkdir(parents=True)
        (out / "openai_task_run_summary.json").write_text("{}", encoding="utf-8")
        (task_run / "task_query_summary.json").write_text(
            json.dumps({"task_type": "inverse_design", "routed_out": str(task_run / "design")}),
            encoding="utf-8",
        )
        (task_run / "task_query_manifest.json").write_text("{}", encoding="utf-8")
        (task_run / "design").mkdir()
        return out

    monkeypatch.setattr(api_module, "run_user_request_with_openai", fake_run_user_request_with_openai)
    profiles = tmp_path / "profiles.yaml"
    registry = tmp_path / "registry.yaml"

    result = run_request(
        request="Find OPC slag mixtures with low porosity.",
        out=tmp_path / "request",
        db=tmp_path / "db",
        use_openai=True,
        use_mock=True,
        model_registry=registry,
        material_systems_config=profiles,
        route_target_policy="allow_caution",
        reaction_model_signature="current",
        skip_validation=True,
    )

    assert result.task_type == "inverse_design"
    assert captured["model_registry"] == registry
    assert captured["material_systems_config"] == profiles
    assert captured["route_target_policy"] == "allow_caution"
    assert captured["reaction_model_signature"] == "current"


def test_run_confirmed_request_mock_returns_standard_result(tmp_path):
    task = _task_query_yaml(tmp_path / "task.yaml")
    preview_dir = preview_task_query_file(query=task, out=tmp_path / "preview")

    result = run_confirmed_request(
        confirmed_preview=preview_dir,
        out=tmp_path / "confirmed",
        db=tmp_path / "db",
        confirm_preview=True,
        use_mock=True,
        disable_plots=True,
    )

    assert result.status == "complete"
    assert result.task_type == "forward_time_series"
    assert result.summary["confirmed_run_manifest"]["confirmed"] is True
    assert result.files["confirmed_task_query_yaml"].endswith("task_query.yaml")
    assert (tmp_path / "confirmed" / "request_result.json").exists()


def test_run_request_mock_accepts_confirmed_preview(tmp_path):
    task = _task_query_yaml(tmp_path / "task.yaml")
    preview_dir = preview_task_query_file(query=task, out=tmp_path / "preview_request_facade")

    result = run_request(
        confirmed_preview=preview_dir,
        confirm_preview=True,
        out=tmp_path / "confirmed_request",
        db=tmp_path / "db",
        use_mock=True,
        disable_plots=True,
    )

    assert result.task_type == "forward_time_series"
    assert result.summary["confirmed_preview"]["risk_counts"]["error"] == 0


def test_run_request_mock_cli_with_forward_query(tmp_path):
    query = _forward_query_yaml(tmp_path / "forward.yaml")

    code = main(
        [
            "run-request-mock",
            "--forward-query",
            str(query),
            "--out",
            str(tmp_path / "cli_request"),
            "--db",
            str(tmp_path / "cli_db"),
            "--no-plots",
        ]
    )

    assert code == 0
    assert (tmp_path / "cli_request" / "request_result.json").exists()


def test_run_request_mock_cli_with_confirmed_preview(tmp_path):
    task = _task_query_yaml(tmp_path / "task.yaml")
    preview_dir = preview_task_query_file(query=task, out=tmp_path / "cli_preview")

    code = main(
        [
            "run-request-mock",
            "--confirmed-preview",
            str(preview_dir),
            "--confirm-preview",
            "--out",
            str(tmp_path / "cli_confirmed_request"),
            "--db",
            str(tmp_path / "cli_confirmed_db"),
            "--no-plots",
        ]
    )

    assert code == 0
    payload = json.loads((tmp_path / "cli_confirmed_request" / "request_result.json").read_text(encoding="utf-8"))
    assert payload["summary"]["confirmed_run_manifest"]["confirmed"] is True


def test_generic_result_collects_design_candidate_outputs(tmp_path):
    run_dir = tmp_path / "request"
    task_dir = run_dir / "task_run"
    design_dir = task_dir / "design"
    design_dir.mkdir(parents=True)
    (task_dir / "task_query_summary.json").write_text(
        json.dumps({"task_type": "inverse_design", "routed_out": str(design_dir)}),
        encoding="utf-8",
    )
    (task_dir / "task_query_manifest.json").write_text("{}", encoding="utf-8")
    (design_dir / "final_candidates.csv").write_text("candidate_id,score\nc1,0.9\n", encoding="utf-8")
    (design_dir / "final_candidates.json").write_text('[{"candidate_id": "c1", "score": 0.9}]', encoding="utf-8")
    (design_dir / "candidate_review.csv").write_text("review_rank,source_recipe_id\n1,c1\n", encoding="utf-8")
    (design_dir / "candidate_review.json").write_text('[{"review_rank": 1, "source_recipe_id": "c1"}]', encoding="utf-8")
    (design_dir / "candidate_review.md").write_text("# Candidate Review\n", encoding="utf-8")
    (design_dir / "candidate_review_summary.json").write_text("{}", encoding="utf-8")
    (design_dir / "design_query_run_summary.json").write_text(
        json.dumps({"stage": "surrogate_search"}),
        encoding="utf-8",
    )

    result = api_module._generic_result_from_task_dir(run_dir=run_dir, task_dir=task_dir)

    assert result.task_type == "inverse_design"
    assert result.files["final_candidates_csv"].endswith("final_candidates.csv")
    assert result.files["final_candidates_json"].endswith("final_candidates.json")
    assert result.files["candidate_review_csv"].endswith("candidate_review.csv")
    assert result.files["candidate_review_md"].endswith("candidate_review.md")
    assert result.files["candidate_review_summary_json"].endswith("candidate_review_summary.json")
    assert result.files["design_query_run_summary_json"].endswith("design_query_run_summary.json")
    assert result.files["final_candidates"] == result.files["final_candidates_csv"]
    assert result.summary["design_query_run_summary"]["stage"] == "surrogate_search"
