import pytest

from llm_security.models import ExpertFamily
from llm_security.routing import (
    AdaptiveExpertRouter,
    AdaptiveTopKPolicy,
    RoutingPolicyConfig,
    RuleTriggerFallback,
)
from tests.helpers import router_samples
from llm_security.datasets import RouterSample
from llm_security.models import Candidate


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


def _semantic_candidate(candidate_id: str, feature: str) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        project_id="project",
        file="x.c",
        function="f",
        line_start=1,
        line_end=1,
        code="",
        evidence=[],
        features={feature: 1.0},
        feature_schema_version="semantic-v1",
    )


def test_router_accepts_matching_semantic_schema_and_rejects_legacy_schema() -> None:
    samples = [
        RouterSample(_semantic_candidate("m1", "memory_copy_count"), [ExpertFamily.MEMORY_BOUNDS]),
        RouterSample(_semantic_candidate("m2", "memory_copy_count"), [ExpertFamily.MEMORY_BOUNDS]),
        RouterSample(_semantic_candidate("l1", "use_after_release_count"), [ExpertFamily.LIFETIME_RESOURCE]),
        RouterSample(_semantic_candidate("l2", "use_after_release_count"), [ExpertFamily.LIFETIME_RESOURCE]),
    ]
    router = AdaptiveExpertRouter.fit(samples, use_rule_fallback=False)

    assert router.feature_schema_version == "semantic-v1"
    assert router.route(samples[0].candidate).selected

    legacy = Candidate(
        "legacy", "project", "x.c", "f", 1, 1, "", [], {"memory_api_count": 1.0}
    )
    with pytest.raises(ValueError, match="expects semantic-v1.*legacy-v1"):
        router.route(legacy)


def test_router_training_rejects_mixed_feature_schemas() -> None:
    semantic = RouterSample(
        _semantic_candidate("semantic", "memory_copy_count"),
        [ExpertFamily.MEMORY_BOUNDS],
    )
    legacy_train, _ = router_samples()

    with pytest.raises(ValueError, match="exactly one feature schema"):
        AdaptiveExpertRouter.fit([semantic, legacy_train[0]])


def test_semantic_rule_fallback_triggers_integer_taint_and_toctou() -> None:
    class MemoryOnlyModel:
        available_families = (ExpertFamily.MEMORY_BOUNDS,)

        def predict_proba(self, _features):
            return {ExpertFamily.MEMORY_BOUNDS: 1.0}

    router = AdaptiveExpertRouter(
        model=MemoryOnlyModel(),
        policy=AdaptiveTopKPolicy(RoutingPolicyConfig(max_entropy=10.0)),
        triggers=RuleTriggerFallback(enabled=True),
        feature_schema_version="semantic-v1",
    )
    cases = {
        ExpertFamily.INTEGER_SIZE_TYPE: "arithmetic_to_memory_sink_count",
        ExpertFamily.TAINT_API_CONTRACT: "source_to_sink_count",
        ExpertFamily.CONCURRENCY_TOCTOU: "toctou_check_use_count",
    }

    for expected, feature in cases.items():
        decision = router.route(_semantic_candidate(expected.value, feature))
        assert expected in decision.trigger_scores


def test_old_router_artifact_version_is_rejected(tmp_path) -> None:
    train, _ = router_samples()
    router = AdaptiveExpertRouter.fit(train, use_rule_fallback=False)
    router._artifact_version = 3
    artifact = tmp_path / "old-router.pkl"
    router.save(artifact)

    with pytest.raises(TypeError, match="Incompatible Router artifact"):
        AdaptiveExpertRouter.load(artifact)
