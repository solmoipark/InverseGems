from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from .task_query import validate_task_query_data
from .utils import config_path, load_yaml, write_json


FENCE_RE = re.compile(r"```(?:json|yaml|yml)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _json_object_candidate(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _candidate_texts(text: str) -> list[tuple[str, str]]:
    stripped = text.strip()
    candidates: list[tuple[str, str]] = []
    for match in FENCE_RE.finditer(stripped):
        candidates.append(("fenced_code_block", match.group(1).strip()))
    json_candidate = _json_object_candidate(stripped)
    if json_candidate:
        candidates.append(("json_object_substring", json_candidate))
    candidates.append(("full_text", stripped))
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for source, candidate in candidates:
        if candidate and candidate not in seen:
            unique.append((source, candidate))
            seen.add(candidate)
    return unique


def _load_mapping(candidate: str) -> dict[str, Any]:
    data = yaml.safe_load(candidate)
    if isinstance(data, dict) and set(data) == {"task_query"} and isinstance(data["task_query"], dict):
        data = data["task_query"]
    if not isinstance(data, dict):
        raise ValueError("LLM output did not parse to a mapping/object.")
    return data


def parse_llm_task_query_output(text: str) -> tuple[dict[str, Any], str]:
    errors: list[str] = []
    for source, candidate in _candidate_texts(text):
        try:
            return _load_mapping(candidate), source
        except Exception as exc:
            errors.append(f"{source}: {type(exc).__name__}: {exc}")
    raise ValueError("Could not parse LLM output as YAML/JSON task query. " + "; ".join(errors))


def build_repair_prompt(
    *,
    original_user_request: str | None,
    previous_output: str,
    error_message: str,
    base_prompt: str | None = None,
) -> str:
    prompt = base_prompt or (
        "You convert user requests into inverse_gems task-query YAML/JSON. "
        "Return only a corrected task_query object and no explanatory text."
    )
    parts = [
        prompt.strip(),
        "",
        "The previous output failed deterministic validation.",
    ]
    if original_user_request:
        parts.extend(["", "Original user request:", original_user_request.strip()])
    parts.extend(
        [
            "",
            "Validation error:",
            error_message.strip(),
            "",
            "Previous output:",
            "```",
            previous_output.strip(),
            "```",
            "",
            "Return only corrected YAML or JSON matching the task_query schema.",
        ]
    )
    return "\n".join(parts)


def validate_llm_task_query_output(
    text: str,
    *,
    original_user_request: str | None = None,
    base_prompt: str | None = None,
) -> dict[str, Any]:
    try:
        parsed, parse_source = parse_llm_task_query_output(text)
        normalized = validate_task_query_data(parsed)
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        return {
            "valid": False,
            "error": error_message,
            "repair_prompt": build_repair_prompt(
                original_user_request=original_user_request,
                previous_output=text,
                error_message=error_message,
                base_prompt=base_prompt,
            ),
        }
    return {
        "valid": True,
        "parse_source": parse_source,
        "task_type": normalized.get("task_type"),
        "task_query": normalized,
        "repair_prompt": None,
    }


def validate_llm_task_query_file(
    path: str | Path,
    *,
    original_user_request: str | None = None,
    base_prompt: str | None = None,
) -> dict[str, Any]:
    return validate_llm_task_query_output(
        _read_text(path),
        original_user_request=original_user_request,
        base_prompt=base_prompt,
    )


def validate_with_repair_callback(
    text: str,
    *,
    repair_callback: Any,
    original_user_request: str | None = None,
    base_prompt: str | None = None,
    max_repairs: int = 1,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    current = text
    for attempt_index in range(max_repairs + 1):
        result = validate_llm_task_query_output(
            current,
            original_user_request=original_user_request,
            base_prompt=base_prompt,
        )
        attempts.append({"attempt_index": attempt_index, "valid": result["valid"], "error": result.get("error")})
        if result["valid"] or attempt_index >= max_repairs:
            result["attempts"] = attempts
            return result
        current = str(repair_callback(result["repair_prompt"]))
    raise RuntimeError("unreachable")


def render_task_router_prompt(
    *,
    prompt: str | Path | None = None,
    schema: str | Path | None = None,
    examples: str | Path | None = None,
) -> str:
    prompt_text = _read_text(prompt or config_path("llm_task_router.prompt.md")).strip()
    schema_path = Path(schema or config_path("task_query.schema.json"))
    examples_path = Path(examples or config_path("task_query.examples.yaml"))
    schema_payload = json.loads(schema_path.read_text(encoding="utf-8")) if schema_path.exists() else {}
    examples_payload = load_yaml(examples_path) if examples_path.exists() else {}
    rendered = [
        prompt_text,
        "",
        "## Task Query JSON Schema",
        "",
        "```json",
        json.dumps(schema_payload, indent=2, sort_keys=True),
        "```",
    ]
    if examples_payload:
        rendered.extend(
            [
                "",
                "## Examples",
                "",
                "```yaml",
                yaml.safe_dump(examples_payload, sort_keys=False),
                "```",
            ]
        )
    return "\n".join(rendered)


def save_rendered_task_router_prompt(
    out: str | Path,
    *,
    prompt: str | Path | None = None,
    schema: str | Path | None = None,
    examples: str | Path | None = None,
) -> Path:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_task_router_prompt(prompt=prompt, schema=schema, examples=examples),
        encoding="utf-8",
    )
    return out_path


def write_llm_task_validation_report(report: dict[str, Any], out: str | Path) -> Path:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, report)
    return out_path
