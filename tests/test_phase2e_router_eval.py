from __future__ import annotations

from llm_security.datasets import RouterSample
from llm_security.experiments.router_eval import (
    evaluate_full_hybrid_router,
    evaluate_supported_router,
)
from llm_security.models import Candidate, ExpertFamily
from llm_security.routing import AdaptiveExpertRouter, RoutingPolicyConfig


def _candidate(candidate_id: str, features: dict[str, float]) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        project_id="p",
        file="a.c",
        function="f",
        line_start=1,
        line_end=2,
        code="x",
        evidence=[],
        features=features,
        suspicion_score=0.8,
        feature_schema_version="semantic-v1",
    )


def _trained_router() -> AdaptiveExpertRouter:
    samples = [
        RouterSample(_candidate("m1", {"memory_copy_without_guard_count": 2.0}), [ExpertFamily.MEMORY_BOUNDS]),
        RouterSample(_candidate("m2", {"memory_copy_without_guard_count": 1.0}), [ExpertFamily.MEMORY_BOUNDS]),
        RouterSample(_candidate("l1", {"use_after_release_count": 2.0}), [ExpertFamily.LIFETIME_RESOURCE]),
        RouterSample(_candidate("l2", {"use_after_release_count": 1.0}), [ExpertFamily.LIFETIME_RESOURCE]),
    ]
    return AdaptiveExpertRouter.fit(
        samples,
        policy_config=RoutingPolicyConfig(max_experts=2),
        seed=2026,
    )


def test_supported_metrics_ignore_unseen_family_without_crashing():
    router = _trained_router()
    samples = [
        RouterSample(_candidate("m", {"memory_copy_without_guard_count": 2.0}), [ExpertFamily.MEMORY_BOUNDS]),
        RouterSample(_candidate("l", {"use_after_release_count": 2.0}), [ExpertFamily.LIFETIME_RESOURCE]),
        RouterSample(_candidate("i", {"arithmetic_to_memory_sink_count": 2.0}), [ExpertFamily.INTEGER_SIZE_TYPE]),
    ]
    metrics = evaluate_supported_router(router, samples)
    assert metrics.supported_sample_count == 2
    assert set(metrics.supported_families) == {
        ExpertFamily.MEMORY_BOUNDS.value,
        ExpertFamily.LIFETIME_RESOURCE.value,
    }


def test_forced_fallback_selects_unseen_trigger_and_limits_experts():
    router = _trained_router()
    unseen = RouterSample(
        _candidate("integer", {"arithmetic_to_memory_sink_count": 1.0}),
        [ExpertFamily.INTEGER_SIZE_TYPE],
    )
    metrics = evaluate_full_hybrid_router(
        router,
        [unseen],
        fallback_enabled=True,
        fallback_mode="forced_trigger",
    )
    assert metrics.unsupported_family_coverage == 1.0
    assert metrics.adaptive_coverage == 1.0
    assert metrics.fallback_activation_count == 1
    assert metrics.fallback_correct_activation_count == 1
    assert metrics.average_experts_per_candidate <= 2.0
