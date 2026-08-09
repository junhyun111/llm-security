import pytest

from llm_security.config import AppConfig


def test_models_and_key_are_loaded_from_env(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENROUTER_API_KEY=test-key\n"
        "OPENROUTER_EXPERT_MODEL=qwen/qwen3-coder\n"
        "OPENROUTER_VALIDATOR_MODEL=anthropic/claude-sonnet-4.5\n"
        "OPENROUTER_PATCH_MODEL=moonshotai/kimi-k2.7-code\n"
        "OPENROUTER_STRUCTURED_OUTPUT=false\n"
        "OPENROUTER_REASONING_ENABLED=false\n",
        encoding="utf-8",
    )

    config = AppConfig.from_env(env_file)

    assert config.model.api_key == "test-key"
    assert config.model.expert_model == "qwen/qwen3-coder"
    assert config.model.structured_output is False
    assert config.model.reasoning_enabled is False


def test_latest_alias_is_rejected(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENROUTER_EXPERT_MODEL=~google/gemini-flash-latest\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical model IDs"):
        AppConfig.from_env(env_file)
