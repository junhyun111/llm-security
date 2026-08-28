import pytest

from llm_security.config import AppConfig
from llm_security.models import ExpertFamily


MODEL_ENV = (
    "OPENROUTER_EXPERT_MODEL=model/expert\n"
    "OPENROUTER_VALIDATOR_MODEL=model/validator\n"
    "OPENROUTER_PATCH_MODEL=model/patch\n"
)


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


def test_model_roles_must_be_explicitly_configured(tmp_path, monkeypatch) -> None:
    for key in (
        "OPENROUTER_EXPERT_MODEL",
        "OPENROUTER_VALIDATOR_MODEL",
        "OPENROUTER_PATCH_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(ValueError, match="OPENROUTER_EXPERT_MODEL must be set"):
        AppConfig.from_env(tmp_path / "missing.env")


def test_optional_temperature_and_provider_routing_are_loaded_from_env(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        MODEL_ENV
        + "OPENROUTER_TEMPERATURE=\n"
        + "OPENROUTER_REQUIRE_PARAMETERS=false\n"
        + "OPENROUTER_ALLOW_FALLBACKS=true\n",
        encoding="utf-8",
    )

    config = AppConfig.from_env(env_file)

    assert config.model.temperature is None
    assert config.model.require_parameters is False
    assert config.model.allow_fallbacks is True


def test_blank_expert_override_inherits_the_required_expert_model(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        MODEL_ENV + "OPENROUTER_MEMORY_MODEL=\n",
        encoding="utf-8",
    )

    config = AppConfig.from_env(env_file)

    assert all(
        model == "model/expert"
        for model in config.model.expert_models.values()
    )


def test_adaptive_router_policy_is_loaded_from_env(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        MODEL_ENV
        + "ROUTER_HIGH_CONFIDENCE=0.80\n"
        "ROUTER_MIN_MARGIN=0.25\n"
        "ROUTER_MAX_ENTROPY=0.90\n"
        "ROUTER_MAX_EXPERTS=2\n"
        "ROUTER_TARGET_COVERAGE=0.96\n"
        "USE_RULE_FALLBACK=false\n",
        encoding="utf-8",
    )

    config = AppConfig.from_env(env_file)

    assert config.router.high_confidence == 0.80
    assert config.router.min_margin == 0.25
    assert config.router.max_entropy == 0.90
    assert config.router.target_coverage == 0.96
    assert config.router.use_rule_fallback is False


def test_latest_alias_is_rejected(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENROUTER_EXPERT_MODEL=~google/gemini-flash-latest\n"
        "OPENROUTER_VALIDATOR_MODEL=model/validator\n"
        "OPENROUTER_PATCH_MODEL=model/patch\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical model IDs"):
        AppConfig.from_env(env_file)


def test_analysis_backend_and_gate_are_loaded_from_env(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        MODEL_ENV
        + "ANALYSIS_BACKEND=semantic\n"
        "CANDIDATE_GATE_ENABLED=false\n"
        "CANDIDATE_GATE_THRESHOLD=0.25\n",
        encoding="utf-8",
    )

    config = AppConfig.from_env(env_file)

    assert config.analysis.backend == "semantic"
    assert config.candidate_gate.enabled is False
    assert config.candidate_gate.threshold == 0.25


def test_candidate_gate_is_off_by_default(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("CANDIDATE_GATE_ENABLED", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(MODEL_ENV, encoding="utf-8")
    config = AppConfig.from_env(env_file)
    assert config.candidate_gate.enabled is False


def test_candidate_ranker_path_is_resolved_relative_to_env_file(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        MODEL_ENV
        + "CANDIDATE_RANKER_PATH=artifacts/ranker.pkl\n"
        + "CANDIDATE_RANKER_REQUIRED=true\n",
        encoding="utf-8",
    )

    config = AppConfig.from_env(env_file)

    assert config.analysis.max_candidates_per_project == 4
    assert config.analysis.candidate_ranker_path == str(
        (tmp_path / "artifacts/ranker.pkl").resolve()
    )
    assert config.analysis.candidate_ranker_required is True


def test_required_candidate_ranker_needs_a_path(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        MODEL_ENV + "CANDIDATE_RANKER_REQUIRED=true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="CANDIDATE_RANKER_PATH is required"):
        AppConfig.from_env(env_file)


def test_per_expert_validator_thresholds_are_loaded(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        MODEL_ENV
        + "MINIMUM_CONFIDENCE=0.55\n"
        + "MINIMUM_CONFIDENCE_INTEGER_SIZE_TYPE=0.73\n",
        encoding="utf-8",
    )

    config = AppConfig.from_env(env_file)

    assert config.validation.minimum_confidence == 0.55
    assert config.validation.minimum_confidence_by_expert[
        ExpertFamily.INTEGER_SIZE_TYPE
    ] == 0.73
