from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .jsonl import append_jsonl


class BudgetExceeded(Exception):
    """Raised before a request that would violate the configured hard guard."""


@dataclass(slots=True)
class ApiBudget:
    max_requests: int
    max_usd: float
    reserve_usd_per_request: float = 0.10
    requests: int = 0
    actual_usd: float = 0.0

    def validate(self) -> None:
        if self.max_requests < 1:
            raise ValueError("max_requests must be positive for paid execution")
        if self.max_usd <= 0.0:
            raise ValueError("max_usd must be positive for paid execution")
        if self.reserve_usd_per_request < 0.0:
            raise ValueError("reserve_usd_per_request cannot be negative")

    def check(self) -> None:
        if self.requests >= self.max_requests:
            raise BudgetExceeded(f"request limit reached ({self.max_requests})")
        if self.actual_usd + self.reserve_usd_per_request > self.max_usd:
            raise BudgetExceeded(
                f"USD guard reached (${self.actual_usd:.6f} spent, "
                f"${self.reserve_usd_per_request:.6f} reserved, cap=${self.max_usd:.6f})"
            )


class BudgetedLLMClient:
    def __init__(self, client, budget: ApiBudget, ledger_path: str | Path) -> None:
        budget.validate()
        self.client = client
        self.budget = budget
        self.ledger_path = Path(ledger_path)

    def complete(self, *, model, messages, response_schema, metadata=None):
        self.budget.check()
        request_number = self.budget.requests + 1
        task = str((metadata or {}).get("task", "unknown"))
        try:
            response = self.client.complete(
                model=model,
                messages=messages,
                response_schema=response_schema,
                metadata=metadata,
            )
        except Exception as error:
            self.budget.requests += 1
            append_jsonl(
                self.ledger_path,
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "request_number": request_number,
                    "task": task,
                    "model": model,
                    "status": "error",
                    "error_type": type(error).__name__,
                },
            )
            raise
        self.budget.requests += 1
        self.budget.actual_usd += float(response.usage.cost)
        append_jsonl(
            self.ledger_path,
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "request_number": request_number,
                "task": task,
                "model": response.usage.model,
                "provider": response.usage.provider,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "reasoning_tokens": response.usage.reasoning_tokens,
                "cost_usd": response.usage.cost,
                "latency_seconds": response.usage.latency_seconds,
                "status": "ok",
            },
        )
        return response
