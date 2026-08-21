from pathlib import Path

import yaml

from inverse_gems.llm_task_router import (
    parse_llm_task_query_output,
    render_task_router_prompt,
    validate_llm_task_query_output,
    validate_with_repair_callback,
)
from inverse_gems.task_query import validate_task_query_data
from inverse_gems.utils import config_path, load_yaml


def _valid_forward_yaml() -> str:
    return yaml.safe_dump(
        {
            "task_type": "forward_time_series",
            "forward_query": {
                "recipe": {"binders": {"OPC": 40, "slag": 30, "fly_ash": 30}, "w_b": 0.4},
                "age_grid": {"values": [0.1, 1.0, 28.0]},
            },
        },
        sort_keys=False,
    )


def test_parse_llm_task_query_output_accepts_fenced_yaml():
    parsed, source = parse_llm_task_query_output(f"Here is the query:\n```yaml\n{_valid_forward_yaml()}```")
    assert source == "fenced_code_block"
    assert parsed["task_type"] == "forward_time_series"


def test_validate_llm_task_query_output_returns_normalized_query():
    result = validate_llm_task_query_output(_valid_forward_yaml())
    assert result["valid"] is True
    assert result["task_type"] == "forward_time_series"
    assert result["task_query"]["forward_query"]["recipe"]["binders"]["slag"] == 30.0


def test_invalid_llm_task_query_output_produces_repair_prompt():
    result = validate_llm_task_query_output(
        "task_type: forward_time_series\n",
        original_user_request="calculate OPC 100 at 28 days",
    )
    assert result["valid"] is False
    assert "forward_query is required" in result["error"]
    assert "Original user request" in result["repair_prompt"]
    assert "Return only corrected YAML or JSON" in result["repair_prompt"]


def test_validate_with_repair_callback_accepts_one_repair():
    def repair_callback(_prompt):
        return _valid_forward_yaml()

    result = validate_with_repair_callback(
        "task_type: forward_time_series\n",
        repair_callback=repair_callback,
        max_repairs=1,
    )
    assert result["valid"] is True
    assert len(result["attempts"]) == 2


def test_render_task_router_prompt_includes_schema_and_examples(tmp_path):
    prompt = tmp_path / "prompt.md"
    schema = tmp_path / "schema.json"
    examples = tmp_path / "examples.yaml"
    prompt.write_text("Route requests.", encoding="utf-8")
    schema.write_text('{"type":"object","properties":{"task_type":{"type":"string"}}}', encoding="utf-8")
    examples.write_text("examples:\n  - id: one\n    user_request: test\n", encoding="utf-8")

    rendered = render_task_router_prompt(prompt=prompt, schema=schema, examples=examples)

    assert "Route requests." in rendered
    assert "Task Query JSON Schema" in rendered
    assert "task_type" in rendered
    assert "Examples" in rendered


def test_task_query_examples_validate_against_schema():
    data = load_yaml(config_path("task_query.examples.yaml"))
    examples = data.get("examples") or []
    assert examples
    for example in examples:
        validate_task_query_data(example["task_query"])
