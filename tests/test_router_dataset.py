from llm_security.datasets import (
    load_router_samples_jsonl,
    write_router_samples_jsonl,
)
from tests.helpers import router_samples


def test_router_sample_jsonl_round_trip(tmp_path) -> None:
    train, _ = router_samples()
    destination = tmp_path / "router.jsonl"
    write_router_samples_jsonl(train[:2], destination)

    restored = load_router_samples_jsonl(destination)

    assert [item.labels for item in restored] == [item.labels for item in train[:2]]
    assert restored[0].candidate.features == train[0].candidate.features
