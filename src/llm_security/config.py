from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


DEFAULT_EXPERT_MODEL = "openai/gpt-5.4-mini"
DEFAULT_VALIDATOR_MODEL = "anthropic/claude-sonnet-4.5"
DEFAULT_PATCH_MODEL = "moonshotai/kimi-k2.7-code"
DEFAULT_STRONG_MODEL = "openai/gpt-5.4"


@dataclass(slots=True)
class ModelConfig:
    api_key: str | None = None
    expert_model: str = DEFAULT_EXPERT_MODEL
    validator_model: str = DEFAULT_VALIDATOR_MODEL
    patch_model: str = DEFAULT_PATCH_MODEL
    strong_model: str | None = DEFAULT_STRONG_MODEL
    sweep_models: tuple[str, ...] = ()
    temperature: float = 0.0
    max_output_tokens: int = 2500
    reasoning_enabled: bool | None = None
    reasoning_effort: str | None = "medium"
    provider: str | None = None
    structured_output: bool = True


@dataclass(slots=True)
class RouterConfig:
    kind: Literal["rule", "learned"] = "learned"
    threshold: float = 0.25
    max_experts: int = 3


@dataclass(slots=True)
class AnalysisConfig:
    max_candidates_per_project: int = 50
    context_lines: int = 25
    max_context_characters: int = 30_000


@dataclass(slots=True)
class ValidationConfig:
    minimum_confidence: float = 0.60
    use_llm_for_uncertain: bool = True


@dataclass(slots=True)
class RuntimeConfig:
    seed: int = 2026
    request_timeout_seconds: float = 120.0
    max_retries: int = 2
    allow_paid_experiments: bool = False
    run_model_sweep: bool = False


@dataclass(slots=True)
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_env(cls, path: str | Path = ".env") -> "AppConfig":
        values = _read_env_file(path)
        values.update(os.environ)
        sweep_models = tuple(
            item.strip()
            for item in values.get("OPENROUTER_SWEEP_MODELS", "").split(",")
            if item.strip()
        )
        config = cls(
            model=ModelConfig(
                api_key=_optional(values.get("OPENROUTER_API_KEY")),
                expert_model=values.get("OPENROUTER_EXPERT_MODEL", DEFAULT_EXPERT_MODEL),
                validator_model=values.get(
                    "OPENROUTER_VALIDATOR_MODEL", DEFAULT_VALIDATOR_MODEL
                ),
                patch_model=values.get("OPENROUTER_PATCH_MODEL", DEFAULT_PATCH_MODEL),
                strong_model=_optional(
                    values.get("OPENROUTER_STRONG_MODEL", DEFAULT_STRONG_MODEL)
                ),
                sweep_models=sweep_models,
                temperature=float(values.get("OPENROUTER_TEMPERATURE", "0")),
                max_output_tokens=int(values.get("OPENROUTER_MAX_OUTPUT_TOKENS", "2500")),
                reasoning_enabled=_as_optional_bool(
                    values.get("OPENROUTER_REASONING_ENABLED")
                ),
                reasoning_effort=_optional(
                    values.get("OPENROUTER_REASONING_EFFORT", "medium")
                ),
                provider=_optional(values.get("OPENROUTER_PROVIDER")),
                structured_output=_as_bool(
                    values.get("OPENROUTER_STRUCTURED_OUTPUT", "true")
                ),
            ),
            router=RouterConfig(
                kind=values.get("ROUTER_KIND", "learned"),
                threshold=float(values.get("ROUTER_THRESHOLD", "0.25")),
                max_experts=int(values.get("ROUTER_MAX_EXPERTS", "3")),
            ),
            analysis=AnalysisConfig(
                max_candidates_per_project=int(values.get("MAX_CANDIDATES", "50")),
                context_lines=int(values.get("CONTEXT_LINES", "25")),
                max_context_characters=int(values.get("MAX_CONTEXT_CHARACTERS", "30000")),
            ),
            validation=ValidationConfig(
                minimum_confidence=float(values.get("MINIMUM_CONFIDENCE", "0.60")),
                use_llm_for_uncertain=_as_bool(
                    values.get("USE_LLM_FOR_UNCERTAIN", "true")
                ),
            ),
            runtime=RuntimeConfig(
                seed=int(values.get("EXPERIMENT_SEED", "2026")),
                request_timeout_seconds=float(values.get("REQUEST_TIMEOUT_SECONDS", "120")),
                max_retries=int(values.get("MAX_RETRIES", "2")),
                allow_paid_experiments=_as_bool(
                    values.get("RUN_PAID_EXPERIMENTS", "false")
                ),
                run_model_sweep=_as_bool(values.get("RUN_MODEL_SWEEP", "false")),
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.router.kind not in {"rule", "learned"}:
            raise ValueError("ROUTER_KIND must be rule or learned")
        if not 0.0 <= self.router.threshold <= 1.0:
            raise ValueError("ROUTER_THRESHOLD must be between 0 and 1")
        if not 1 <= self.router.max_experts <= 6:
            raise ValueError("ROUTER_MAX_EXPERTS must be between 1 and 6")
        if self.analysis.max_candidates_per_project < 1:
            raise ValueError("MAX_CANDIDATES must be positive")
        for model_id in (
            self.model.expert_model,
            self.model.validator_model,
            self.model.patch_model,
        ):
            if not model_id:
                raise ValueError("OpenRouter role model names cannot be empty")
            if model_id.startswith("~"):
                raise ValueError("Experiments require canonical model IDs, not latest aliases")


def _read_env_file(path: str | Path) -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        return {}
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        env_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid .env entry at line {line_number}")
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _optional(value: str | None) -> str | None:
    return value if value and value.strip() else None


def _as_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _as_optional_bool(value: str | None) -> bool | None:
    if value is None or not value.strip():
        return None
    return _as_bool(value)
