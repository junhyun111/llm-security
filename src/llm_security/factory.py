from __future__ import annotations

from .aggregation import FindingAggregator
from .config import AppConfig
from .evidence import ContextBuilder
from .experts import ExpertRunner
from .llm import OpenRouterClient
from .pipeline import VulnerabilityPipeline
from .routing import Router
from .static_analysis import LightweightStaticAnalyzer
from .validation import EvidenceValidator


def build_pipeline(config: AppConfig, router: Router) -> VulnerabilityPipeline:
    config.validate()
    client = OpenRouterClient(
        api_key=config.model.api_key,
        timeout_seconds=config.runtime.request_timeout_seconds,
        max_retries=config.runtime.max_retries,
        temperature=config.model.temperature,
        max_output_tokens=config.model.max_output_tokens,
        reasoning_enabled=config.model.reasoning_enabled,
        reasoning_effort=config.model.reasoning_effort,
        provider=config.model.provider,
        structured_output=config.model.structured_output,
    )
    analyzer = LightweightStaticAnalyzer(
        max_candidates=config.analysis.max_candidates_per_project,
        context_lines=config.analysis.context_lines,
    )
    return VulnerabilityPipeline(
        analyzer=analyzer,
        router=router,
        expert_runner=ExpertRunner(
            client=client,
            model=config.model.expert_model,
            context_builder=ContextBuilder(config.analysis.max_context_characters),
        ),
        aggregator=FindingAggregator(),
        validator=EvidenceValidator(
            minimum_confidence=config.validation.minimum_confidence,
            client=client,
            model=config.model.validator_model,
            use_llm_for_uncertain=config.validation.use_llm_for_uncertain,
        ),
    )
