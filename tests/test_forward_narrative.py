import json
from types import SimpleNamespace

from inverse_gems.forward_narrative import (
    build_deterministic_narrative,
    build_narrative_prompt,
    write_forward_narrative,
)


def _answer_payload():
    return {
        "headline": "Forward time_series completed successfully for 2 row(s).",
        "requested": {"phases": ["CNASH"], "phase_groups": ["C-A-S-H"], "scalars": ["pH", "porosity"]},
        "missing": {"phases": [], "phase_groups": [], "scalars": []},
        "series_highlights": [
            {
                "output": "CNASH.mass",
                "first": 0.01,
                "first_age_days": 1.0,
                "final": 0.05,
                "final_age_days": 28.0,
                "min": 0.01,
                "min_age_days": 1.0,
                "max": 0.05,
                "max_age_days": 28.0,
            },
            {
                "output": "phase_group.C-A-S-H.mass",
                "first": 0.01,
                "first_age_days": 1.0,
                "final": 0.05,
                "final_age_days": 28.0,
                "min": 0.01,
                "min_age_days": 1.0,
                "max": 0.05,
                "max_age_days": 28.0,
            },
            {
                "output": "scalar.porosity",
                "first": 0.42,
                "first_age_days": 1.0,
                "final": 0.36,
                "final_age_days": 28.0,
                "min": 0.36,
                "min_age_days": 28.0,
                "max": 0.42,
                "max_age_days": 1.0,
            },
        ],
        "files": {"answer_json": "answer.json"},
    }


class FakeResponses:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.text, usage={"total_tokens": 25})


class FakeOpenAIClient:
    def __init__(self, text):
        self.responses = FakeResponses(text)


def test_build_deterministic_narrative_uses_answer_values_only():
    text = build_deterministic_narrative(_answer_payload(), language="ko")

    assert "Forward time_series completed successfully" in text
    assert "`CNASH.mass`" in text
    assert "selected phase group" in text
    assert "재해석" in text


def test_build_narrative_prompt_warns_against_aliasing():
    prompt = build_narrative_prompt(_answer_payload(), language="en")

    assert "Use only the provided answer JSON" in prompt
    assert "Preserve raw xGEMS/GEMS phase names exactly" in prompt
    assert "phase_group.C-A-S-H.mass" in prompt


def test_write_forward_narrative_deterministic(tmp_path):
    answer = tmp_path / "answer.json"
    answer.write_text(json.dumps(_answer_payload()), encoding="utf-8")

    out = write_forward_narrative(answer=answer, out=tmp_path / "narrative", language="en")

    payload = json.loads((out / "narrative_answer.json").read_text(encoding="utf-8"))
    markdown = (out / "narrative_answer.md").read_text(encoding="utf-8")
    assert payload["mode"] == "deterministic"
    assert payload["source_answer_json"] == str(answer)
    assert "CNASH.mass" in markdown
    assert (out / "narrative_prompt.md").exists()


def test_write_forward_narrative_openai_with_fake_client(tmp_path):
    answer = tmp_path / "answer.json"
    answer.write_text(json.dumps(_answer_payload()), encoding="utf-8")
    client = FakeOpenAIClient("LLM narrative based only on answer.json")

    out = write_forward_narrative(
        answer=answer,
        out=tmp_path / "narrative",
        language="en",
        use_openai=True,
        model="fake-model",
        client=client,
    )

    payload = json.loads((out / "narrative_answer.json").read_text(encoding="utf-8"))
    assert payload["mode"] == "openai"
    assert payload["model"] == "fake-model"
    assert payload["usage"] == {"total_tokens": 25}
    assert (out / "llm_narrative_raw.md").read_text(encoding="utf-8") == "LLM narrative based only on answer.json"
    assert "Use only the provided answer JSON" in client.responses.calls[0]["input"][1]["content"]
