from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .models import ExpertFamily


@dataclass(slots=True)
class ModelConfig:
    api_key: str | None = None
    expert_model: str = ""
    expert_models: dict[ExpertFamily, str] = field(default_factory=dict)
    validator_model: str = ""
    patch_model: str = ""
    strong_model: str | None = None
    sweep_models: tuple[str, ...] = ()
    temperature: float | None = None
    max_output_tokens: int = 2500
    reasoning_enabled: bool | None = None
    reasoning_effort: str | None = "medium"
    provider: str | None = None
    require_parameters: bool = True
    allow_fallbacks: bool = False
    structured_output: bool = True


@dataclass(slots=True)
class RouterConfig:
    high_confidence: float = 0.72
    min_margin: float = 0.18
    max_entropy: float = 1.0
    max_experts: int = 2
    target_coverage: float = 0.95
    use_rule_fallback: bool = True


@dataclass(slots=True)
class CandidateGateConfig:
    enabled: bool = False
    threshold: float = 0.40


@dataclass(slots=True)
class AnalysisConfig:
    backend: str = "semantic"
    max_candidates_per_project: int = 50
    context_lines: int = 25
    max_context_characters: int = 30_000
    security_knowledge_path: str | None = None


@dataclass(slots=True)
class ValidationConfig:
    minimum_confidence: float = 0.60
    use_llm_for_uncertain: bool = True
    falsify_all_supported: bool = True


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
    candidate_gate: CandidateGateConfig = field(default_factory=CandidateGateConfig)
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
        default_expert_model = _required(values, "OPENROUTER_EXPERT_MODEL")
        expert_models = {
            family: _optional(values.get(env_name)) or default_expert_model
            for family, env_name in _EXPERT_MODEL_ENV.items()
        }
        config = cls(
            model=ModelConfig(
                api_key=_optional(values.get("OPENROUTER_API_KEY")),
                expert_model=default_expert_model,
                expert_models=expert_models,
                validator_model=_required(values, "OPENROUTER_VALIDATOR_MODEL"),
                patch_model=_required(values, "OPENROUTER_PATCH_MODEL"),
                strong_model=_optional(values.get("OPENROUTER_STRONG_MODEL")),
                sweep_models=sweep_models,
                temperature=_as_optional_float(
                    values.get("OPENROUTER_TEMPERATURE")
                ),
                max_output_tokens=int(values.get("OPENROUTER_MAX_OUTPUT_TOKENS", "2500")),
                reasoning_enabled=_as_optional_bool(
                    values.get("OPENROUTER_REASONING_ENABLED")
                ),
                reasoning_effort=_optional(
                    values.get("OPENROUTER_REASONING_EFFORT", "medium")
                ),
                provider=_optional(values.get("OPENROUTER_PROVIDER")),
                require_parameters=_as_bool(
                    values.get("OPENROUTER_REQUIRE_PARAMETERS", "true")
                ),
                allow_fallbacks=_as_bool(
                    values.get("OPENROUTER_ALLOW_FALLBACKS", "false")
                ),
                structured_output=_as_bool(
                    values.get("OPENROUTER_STRUCTURED_OUTPUT", "true")
                ),
            ),
            router=RouterConfig(
                high_confidence=float(
                    values.get("ROUTER_HIGH_CONFIDENCE", "0.72")
                ),
                min_margin=float(values.get("ROUTER_MIN_MARGIN", "0.18")),
                max_entropy=float(values.get("ROUTER_MAX_ENTROPY", "1.0")),
                max_experts=int(values.get("ROUTER_MAX_EXPERTS", "2")),
                target_coverage=float(
                    values.get("ROUTER_TARGET_COVERAGE", "0.95")
                ),
                use_rule_fallback=_as_bool(
                    values.get("USE_RULE_FALLBACK", "true")
                ),
            ),
            candidate_gate=CandidateGateConfig(
                enabled=_as_bool(values.get("CANDIDATE_GATE_ENABLED", "false")),
                threshold=float(values.get("CANDIDATE_GATE_THRESHOLD", "0.40"))
            ),
            analysis=AnalysisConfig(
                backend=values.get("ANALYSIS_BACKEND", "semantic").strip().lower(),
                max_candidates_per_project=int(values.get("MAX_CANDIDATES", "50")),
                context_lines=int(values.get("CONTEXT_LINES", "25")),
                max_context_characters=int(values.get("MAX_CONTEXT_CHARACTERS", "30000")),
                security_knowledge_path=_optional(
                    values.get("SECURITY_KNOWLEDGE_PATH")
                ),
            ),
            validation=ValidationConfig(
                minimum_confidence=float(values.get("MINIMUM_CONFIDENCE", "0.60")),
                use_llm_for_uncertain=_as_bool(
                    values.get("USE_LLM_FOR_UNCERTAIN", "true")
                ),
                falsify_all_supported=_as_bool(
                    values.get("FALSIFY_ALL_SUPPORTED", "true")
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
        if not 0.0 <= self.candidate_gate.threshold <= 1.0:
            raise ValueError("CANDIDATE_GATE_THRESHOLD must be between 0 and 1")
        if not 0.0 <= self.router.high_confidence <= 1.0:
            raise ValueError("ROUTER_HIGH_CONFIDENCE must be between 0 and 1")
        if not 0.0 <= self.router.min_margin <= 1.0:
            raise ValueError("ROUTER_MIN_MARGIN must be between 0 and 1")
        if self.router.max_entropy < 0.0:
            raise ValueError("ROUTER_MAX_ENTROPY cannot be negative")
        if self.router.max_experts not in {1, 2}:
            raise ValueError("ROUTER_MAX_EXPERTS must be 1 or 2")
        if not 0.0 <= self.router.target_coverage <= 1.0:
            raise ValueError("ROUTER_TARGET_COVERAGE must be between 0 and 1")
        if self.analysis.max_candidates_per_project < 1:
            raise ValueError("MAX_CANDIDATES must be positive")
        if self.analysis.backend not in {"legacy", "semantic"}:
            raise ValueError("ANALYSIS_BACKEND must be legacy or semantic")
        for model_id in (
            self.model.expert_model,
            self.model.validator_model,
            self.model.patch_model,
            *self.model.expert_models.values(),
        ):
            if not model_id:
                raise ValueError("OpenRouter role model names cannot be empty")
            if model_id.startswith("~"):
                raise ValueError("Experiments require canonical model IDs, not latest aliases")


_EXPERT_MODEL_ENV = {
    ExpertFamily.MEMORY_BOUNDS: "OPENROUTER_MEMORY_MODEL",
    ExpertFamily.LIFETIME_RESOURCE: "OPENROUTER_LIFETIME_MODEL",
    ExpertFamily.INTEGER_SIZE_TYPE: "OPENROUTER_INTEGER_MODEL",
    ExpertFamily.TAINT_API_CONTRACT: "OPENROUTER_TAINT_MODEL",
    ExpertFamily.CONTROL_STATE_ERROR: "OPENROUTER_CONTROL_MODEL",
    ExpertFamily.CONCURRENCY_TOCTOU: "OPENROUTER_CONCURRENCY_MODEL",
}


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


def _required(values: dict[str, str], key: str) -> str:
    value = _optional(values.get(key))
    if value is None:
        raise ValueError(f"{key} must be set in .env or the process environment")
    return value


def _as_optional_float(value: str | None) -> float | None:
    normalized = _optional(value)
    return float(normalized) if normalized is not None else None


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
