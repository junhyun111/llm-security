from llm_security.datasets import (
    load_router_samples_jsonl,
    write_router_samples_jsonl,
)
from tests.helpers import router_samples
from llm_security.datasets import RouterSample
from llm_security.models import Candidate, ExpertFamily


def test_router_sample_jsonl_round_trip(tmp_path) -> None:
    train, _ = router_samples()
    destination = tmp_path / "router.jsonl"
    write_router_samples_jsonl(train[:2], destination)

    restored = load_router_samples_jsonl(destination)

    assert [item.labels for item in restored] == [item.labels for item in train[:2]]
    assert restored[0].candidate.features == train[0].candidate.features
    assert restored[0].candidate.feature_schema_version == "legacy-v1"


def test_semantic_feature_schema_round_trip(tmp_path) -> None:
    candidate = Candidate(
        "semantic",
        "project",
        "x.c",
        "f",
        1,
        1,
        "",
        [],
        {"use_after_release_count": 1.0},
        feature_schema_version="semantic-v1",
    )
    destination = tmp_path / "semantic-router.jsonl"
    write_router_samples_jsonl(
        [RouterSample(candidate, [ExpertFamily.LIFETIME_RESOURCE])], destination
    )

    restored = load_router_samples_jsonl(destination)

    assert restored[0].candidate.feature_schema_version == "semantic-v1"
