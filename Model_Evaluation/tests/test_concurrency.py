from __future__ import annotations

import threading
import time

from model_evaluation.concurrency import run_completion_pool


def test_completion_pool_reuses_each_finished_slot() -> None:
    lock = threading.Lock()
    active = 0
    peak = 0
    completion_order: list[int] = []

    def work(item: int) -> int:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.04 if item == 0 else 0.005)
        with lock:
            active -= 1
        return item * 2

    results = run_completion_pool(
        range(12),
        work,
        item_key=str,
        max_concurrency=4,
        max_attempts=1,
        retry_base_seconds=0.0,
        retry_max_seconds=0.0,
        retryable=lambda error: False,
        on_completed=lambda result: completion_order.append(result.item),
    )

    assert peak == 4
    assert sorted(result.value for result in results) == list(range(0, 24, 2))
    assert completion_order[0] != 0


def test_completion_pool_retries_transient_failure() -> None:
    attempts = 0

    def work(item: str) -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("OpenRouter HTTP 429")
        return item

    result = run_completion_pool(
        ["case"],
        work,
        item_key=str,
        max_concurrency=1,
        max_attempts=3,
        retry_base_seconds=0.0,
        retry_max_seconds=0.0,
        retryable=lambda error: "429" in str(error),
    )[0]

    assert result.succeeded
    assert result.attempts == 3
    assert result.value == "case"
