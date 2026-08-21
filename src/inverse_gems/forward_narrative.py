from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from .utils import load_project_env, write_json


NarrativeLanguage = Literal["ko", "en"]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Answer JSON does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return "" if value is None else str(value)


def _join_names(names: list[str]) -> str:
    return ", ".join(names) if names else "none"


def _highlight_lines(answer: dict[str, Any], *, limit: int = 8, language: NarrativeLanguage = "ko") -> list[str]:
    lines: list[str] = []
    for item in (answer.get("series_highlights") or [])[:limit]:
        output = item.get("output")
        first = _format_value(item.get("first"))
        first_age = _format_value(item.get("first_age_days"))
        final = _format_value(item.get("final"))
        final_age = _format_value(item.get("final_age_days"))
        minimum = _format_value(item.get("min"))
        min_age = _format_value(item.get("min_age_days"))
        maximum = _format_value(item.get("max"))
        max_age = _format_value(item.get("max_age_days"))
        if language == "ko":
            lines.append(
                f"- `{output}`: 첫 값 {first} at {first_age} days, 마지막 값 {final} at {final_age} days, "
                f"최소 {minimum} at {min_age} days, 최대 {maximum} at {max_age} days."
            )
        else:
            lines.append(
                f"- `{output}`: first {first} at {first_age} days, final {final} at {final_age} days, "
                f"min {minimum} at {min_age} days, max {maximum} at {max_age} days."
            )
    return lines


def build_deterministic_narrative(
    answer: dict[str, Any],
    *,
    language: NarrativeLanguage = "ko",
    max_highlights: int = 8,
) -> str:
    requested = answer.get("requested") or {}
    missing = answer.get("missing") or {}
    phase_names = [str(name) for name in requested.get("phases") or []]
    phase_group_names = [str(name) for name in requested.get("phase_groups") or []]
    scalar_names = [str(name) for name in requested.get("scalars") or []]
    missing_phases = [str(name) for name in missing.get("phases") or []]
    missing_phase_groups = [str(name) for name in missing.get("phase_groups") or []]
    missing_scalars = [str(name) for name in missing.get("scalars") or []]
    highlights = _highlight_lines(answer, limit=max_highlights, language=language)

    if language == "ko":
        lines = [
            "# Forward Narrative Answer",
            "",
            str(answer.get("headline") or "Forward calculation finished."),
            "",
            f"요청 raw phase는 `{_join_names(phase_names)}`, selected phase group은 `{_join_names(phase_group_names)}`, 요청 scalar는 `{_join_names(scalar_names)}`입니다.",
            "아래 내용은 이미 계산된 `answer.json` 값만 사용해 작성되었습니다.",
            "",
            "## 주요 수치",
            "",
        ]
        lines.extend(highlights or ["- 요약할 numeric output이 없습니다."])
        if missing_phases or missing_phase_groups or missing_scalars:
            lines.extend(["", "## 누락된 요청", ""])
            if missing_phases:
                lines.append(f"- 누락 phase: {_join_names(missing_phases)}")
            if missing_phase_groups:
                lines.append(f"- 누락 phase group: {_join_names(missing_phase_groups)}")
            if missing_scalars:
                lines.append(f"- 누락 scalar: {_join_names(missing_scalars)}")
        lines.extend(
            [
                "",
                "## 주의",
                "",
                "- phase 이름은 xGEMS/GEMS raw 이름 그대로 보존했습니다.",
                "- selected phase group은 설정 파일에 명시된 raw phase 합으로 별도 표기하며, raw phase 출력 자체를 바꾸지 않습니다.",
                "- 이 답변은 설정 파일 밖의 phase alias 또는 과학적 재해석을 수행하지 않습니다.",
            ]
        )
    else:
        lines = [
            "# Forward Narrative Answer",
            "",
            str(answer.get("headline") or "Forward calculation finished."),
            "",
            f"Requested raw phases: `{_join_names(phase_names)}`. Requested selected phase groups: `{_join_names(phase_group_names)}`. Requested scalars: `{_join_names(scalar_names)}`.",
            "This narrative uses only the precomputed values in `answer.json`.",
            "",
            "## Key Numbers",
            "",
        ]
        lines.extend(highlights or ["- No numeric outputs were available to summarize."])
        if missing_phases or missing_phase_groups or missing_scalars:
            lines.extend(["", "## Missing Requests", ""])
            if missing_phases:
                lines.append(f"- Missing phases: {_join_names(missing_phases)}")
            if missing_phase_groups:
                lines.append(f"- Missing phase groups: {_join_names(missing_phase_groups)}")
            if missing_scalars:
                lines.append(f"- Missing scalars: {_join_names(missing_scalars)}")
        lines.extend(
            [
                "",
                "## Note",
                "",
                "- Phase names are preserved as raw xGEMS/GEMS names.",
                "- Selected phase groups are reported separately as configured sums of raw phases; raw phase outputs are not renamed.",
                "- This answer does not create phase aliases or scientific reinterpretations outside the configured grouping file.",
            ]
        )
    return "\n".join(lines) + "\n"


def build_narrative_prompt(answer: dict[str, Any], *, language: NarrativeLanguage = "ko") -> str:
    language_label = "Korean" if language == "ko" else "English"
    payload = json.dumps(answer, ensure_ascii=False, indent=2, sort_keys=True)
    return (
        "You write concise user-facing responses for inverse_gems forward thermodynamic calculations.\n"
        "Use only the provided answer JSON. Do not calculate new values. Do not infer missing values.\n"
        "Preserve raw xGEMS/GEMS phase names exactly. If selected phase groups are present, describe them as configured sums separate from raw phases.\n"
        "Do not create aliases, phase groups, or phase aggregations beyond what is already present in the answer JSON.\n"
        "Mention missing requested phases or scalars if the JSON reports them.\n"
        f"Write in {language_label}.\n\n"
        "ANSWER_JSON:\n"
        f"{payload}\n"
    )


def _make_openai_client() -> Any:
    load_project_env()
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError(
            "The openai Python package is required for --use-openai. "
            "Install inverse-gems with the llm extra or install openai in this environment."
        ) from exc
    return OpenAI()


def _extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)
    if isinstance(response, dict):
        output_text = response.get("output_text")
        if output_text:
            return str(output_text)
        chunks: list[str] = []
        for item in response.get("output", []) or []:
            for content in item.get("content", []) or []:
                text = content.get("text") if isinstance(content, dict) else None
                if text:
                    chunks.append(str(text))
        if chunks:
            return "\n".join(chunks)
    output = getattr(response, "output", None)
    chunks = []
    for item in output or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(str(text))
    return "\n".join(chunks) if chunks else str(response)


def _response_usage(response: Any) -> Any:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json")
    return usage


def _call_openai_narrative(
    *,
    answer: dict[str, Any],
    language: NarrativeLanguage,
    model: str,
    max_output_tokens: int,
    temperature: float,
    client: Any | None,
) -> tuple[str, Any, str]:
    prompt = build_narrative_prompt(answer, language=language)
    openai_client = client or _make_openai_client()
    response = openai_client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": "You convert precomputed inverse_gems result JSON into concise user-facing prose.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    return _extract_response_text(response), _response_usage(response), prompt


def write_forward_narrative(
    *,
    answer: str | Path,
    out: str | Path | None = None,
    language: NarrativeLanguage = "ko",
    use_openai: bool = False,
    model: str | None = None,
    max_output_tokens: int = 1200,
    temperature: float = 0.0,
    client: Any | None = None,
) -> Path:
    answer_path = Path(answer)
    out_dir = Path(out) if out is not None else answer_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    answer_payload = _load_json(answer_path)
    model_name = model or os.environ.get("INVERSE_GEMS_OPENAI_MODEL", "gpt-4.1-mini")
    if use_openai:
        text, usage, prompt = _call_openai_narrative(
            answer=answer_payload,
            language=language,
            model=model_name,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            client=client,
        )
        mode = "openai"
        (out_dir / "narrative_prompt.md").write_text(prompt, encoding="utf-8")
        (out_dir / "llm_narrative_raw.md").write_text(text, encoding="utf-8")
    else:
        text = build_deterministic_narrative(answer_payload, language=language)
        usage = None
        mode = "deterministic"
        (out_dir / "narrative_prompt.md").write_text(build_narrative_prompt(answer_payload, language=language), encoding="utf-8")

    md_path = out_dir / "narrative_answer.md"
    json_path = out_dir / "narrative_answer.json"
    md_path.write_text(text, encoding="utf-8")
    write_json(
        json_path,
        {
            "answer_type": "forward_narrative_answer",
            "mode": mode,
            "language": language,
            "model": model_name if use_openai else None,
            "usage": usage,
            "source_answer_json": str(answer_path),
            "narrative_markdown": str(md_path),
            "narrative_json": str(json_path),
            "raw_phase_policy": "Raw xGEMS/GEMS names are preserved exactly; configured selected phase groups are reported separately when present.",
            "text": text,
        },
    )
    return out_dir
