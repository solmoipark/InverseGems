import os

from inverse_gems.utils import load_env_file


def test_load_env_file_sets_keys_without_overriding(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# local secrets",
                "OPENAI_API_KEY=from_file",
                "INVERSE_GEMS_OPENAI_MODEL='gpt-test'",
                "export INVERSE_GEMS_FLAG=enabled",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "from_environment")
    monkeypatch.delenv("INVERSE_GEMS_OPENAI_MODEL", raising=False)
    monkeypatch.delenv("INVERSE_GEMS_FLAG", raising=False)

    try:
        loaded = load_env_file(env_file)

        assert "OPENAI_API_KEY" not in loaded
        assert os.environ["OPENAI_API_KEY"] == "from_environment"
        assert os.environ["INVERSE_GEMS_OPENAI_MODEL"] == "gpt-test"
        assert os.environ["INVERSE_GEMS_FLAG"] == "enabled"
    finally:
        os.environ.pop("INVERSE_GEMS_OPENAI_MODEL", None)
        os.environ.pop("INVERSE_GEMS_FLAG", None)


def test_load_env_file_can_override(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=from_file\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "from_environment")

    loaded = load_env_file(env_file, override=True)

    assert loaded == ["OPENAI_API_KEY"]
    assert os.environ["OPENAI_API_KEY"] == "from_file"
