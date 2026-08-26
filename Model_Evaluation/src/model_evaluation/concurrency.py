from __future__ import annotations

import asyncio
import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Generic, Iterable, TypeVar


T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True, slots=True)
class CompletionResult(Generic[T, R]):
    item: T
    value: R | None
    attempts: int
    error: Exception | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


def run_completion_pool(
    items: Iterable[T],
    work: Callable[[T], R],
    *,
    item_key: Callable[[T], str],
    max_concurrency: int,
    max_attempts: int,
    retry_base_seconds: float,
    retry_max_seconds: float,
    retryable: Callable[[Exception], bool],
    on_completed: Callable[[CompletionResult[T, R]], None] | None = None,
) -> list[CompletionResult[T, R]]:
    """Run synchronous network work through a semaphore-controlled async pool.

    Results are consumed with ``asyncio.as_completed``. Consequently, one slow
    request never holds a batch barrier: as soon as any request releases the
    semaphore, the next waiting item can start.
    """
    pending = list(items)
    if not pending:
        return []
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if retry_base_seconds < 0 or retry_max_seconds < retry_base_seconds:
        raise ValueError("Invalid retry delay configuration")

    def launch() -> list[CompletionResult[T, R]]:
        return asyncio.run(
            _run_completion_pool(
                pending,
                work,
                item_key=item_key,
                max_concurrency=max_concurrency,
                max_attempts=max_attempts,
                retry_base_seconds=retry_base_seconds,
                retry_max_seconds=retry_max_seconds,
                retryable=retryable,
                on_completed=on_completed,
            )
        )

    # Jupyter already owns an event loop in the notebook thread. Keep the
    # public collector synchronous by hosting our event loop in one bridge
    # thread when necessary.
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return launch()
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="eval-async-bridge") as bridge:
        return bridge.submit(launch).result()


async def _run_completion_pool(
    items: list[T],
    work: Callable[[T], R],
    *,
    item_key: Callable[[T], str],
    max_concurrency: int,
    max_attempts: int,
    retry_base_seconds: float,
    retry_max_seconds: float,
    retryable: Callable[[Exception], bool],
    on_completed: Callable[[CompletionResult[T, R]], None] | None,
) -> list[CompletionResult[T, R]]:
    limit = min(max_concurrency, len(items))
    semaphore = asyncio.Semaphore(limit)
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(
        max_workers=limit,
        thread_name_prefix="eval-api",
    )

    async def run_one(item: T) -> CompletionResult[T, R]:
        key = item_key(item)
        for attempt in range(1, max_attempts + 1):
            try:
                async with semaphore:
                    value = await loop.run_in_executor(executor, work, item)
                return CompletionResult(item=item, value=value, attempts=attempt)
            except Exception as error:  # surfaced in the structured result below
                if attempt >= max_attempts or not retryable(error):
                    return CompletionResult(
                        item=item,
                        value=None,
                        attempts=attempt,
                        error=error,
                    )
                delay = min(
                    retry_max_seconds,
                    retry_base_seconds * (2 ** (attempt - 1)),
                )
                # Deterministic jitter prevents a large 429 retry wave from
                # synchronizing on the same second while remaining reproducible.
                digest = hashlib.sha256(f"{key}:{attempt}".encode("utf-8")).digest()
                jitter = (int.from_bytes(digest[:2], "big") / 65535.0) * delay * 0.25
                await asyncio.sleep(delay + jitter)
        raise AssertionError("unreachable")

    tasks = [asyncio.create_task(run_one(item)) for item in items]
    results: list[CompletionResult[T, R]] = []
    try:
        for future in asyncio.as_completed(tasks):
            result = await future
            results.append(result)
            if on_completed is not None:
                on_completed(result)
    finally:
        executor.shutdown(wait=True, cancel_futures=False)
    return results
