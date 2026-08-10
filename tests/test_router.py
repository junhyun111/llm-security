from llm_security.models import ExpertFamily
from llm_security.routing import (
    AdaptiveExpertRouter,
    AdaptiveTopKPolicy,
    RoutingPolicyConfig,
    RuleTriggerFallback,
)
from tests.helpers import router_samples


def test_adaptive_policy_selects_top1_or_top2() -> None:
    policy = AdaptiveTopKPolicy(
        RoutingPolicyConfig(
            high_confidence=0.70,
            min_margin=0.20,
            max_entropy=10.0,
            max_experts=2,
        )
    )

    confident = policy.decide(
        {
            ExpertFamily.MEMORY_BOUNDS: 0.82,
            ExpertFamily.INTEGER_SIZE_TYPE: 0.10,
            ExpertFamily.LIFETIME_RESOURCE: 0.08,
        }
    )
    uncertain = policy.decide(
        {
            ExpertFamily.MEMORY_BOUNDS: 0.46,
            ExpertFamily.INTEGER_SIZE_TYPE: 0.42,
            ExpertFamily.LIFETIME_RESOURCE: 0.12,
        }
    )

    assert confident.selected == [ExpertFamily.MEMORY_BOUNDS]
    assert uncertain.selected == [
        ExpertFamily.MEMORY_BOUNDS,
        ExpertFamily.INTEGER_SIZE_TYPE,
    ]
    assert "routing uncertainty" in uncertain.reasons[0]


def test_adaptive_router_training_metrics_and_serialization(tmp_path) -> None:
    train, test = router_samples()
    router = AdaptiveExpertRouter.fit(
        train,
        policy_config=RoutingPolicyConfig(max_entropy=10.0),
        use_rule_fallback=False,
    )
    metrics = router.evaluate(test)

    assert metrics.routing_accuracy >= 0.9
    assert metrics.coverage_at_2 == 1.0
    assert set(router.available_families) == set(ExpertFamily)

    artifact = tmp_path / "router.pkl"
    router.save(artifact)
    restored = AdaptiveExpertRouter.load(artifact)

    original = router.route(test[0].candidate)
    loaded = restored.route(test[0].candidate)
    assert loaded.selected == original.selected
    assert loaded.top1_confidence == original.top1_confidence


def test_rule_fallback_is_separate_from_learned_probabilities() -> None:
    _, samples = router_samples()
    integer_candidate = next(
        sample.candidate
        for sample in samples
        if sample.labels == [ExpertFamily.INTEGER_SIZE_TYPE]
    )

    class MemoryOnlyModel:
        available_families = (ExpertFamily.MEMORY_BOUNDS,)

        def predict_proba(self, _features):
            return {ExpertFamily.MEMORY_BOUNDS: 1.0}

    router = AdaptiveExpertRouter(
        model=MemoryOnlyModel(),
        policy=AdaptiveTopKPolicy(RoutingPolicyConfig(max_entropy=10.0)),
        triggers=RuleTriggerFallback(enabled=True),
    )
    decision = router.route(integer_candidate)

    assert decision.learned_scores == {ExpertFamily.MEMORY_BOUNDS: 1.0}
    assert ExpertFamily.INTEGER_SIZE_TYPE in decision.trigger_scores
    assert any("rule fallback" in reason for reason in decision.reasons)
