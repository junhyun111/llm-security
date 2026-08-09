from llm_security.router import LearnedRouter, train_and_evaluate_router
from tests.helpers import router_samples


def test_learned_router_training_and_serialization(tmp_path) -> None:
    train, test = router_samples()
    router, metrics = train_and_evaluate_router(train, test, threshold=0.25)

    assert metrics.micro_f1 >= 0.9
    assert metrics.coverage_at_2 == 1.0

    artifact = tmp_path / "router.pkl"
    router.save(artifact)
    restored = LearnedRouter.load(artifact)

    assert restored.route(test[0].candidate).selected == router.route(test[0].candidate).selected
