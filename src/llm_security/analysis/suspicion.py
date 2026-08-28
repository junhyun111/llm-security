from __future__ import annotations

import math


RISK_WEIGHTS = {
    "use_after_release_count": 1.0,
    "double_release_count": 1.0,
    "uninitialized_use_count": 0.9,
    "unchecked_nullable_dereference_count": 0.9,
    "unsanitized_source_to_sink_count": 0.85,
    "memory_copy_without_guard_count": 0.75,
    "unchecked_index_count": 0.70,
    "toctou_check_use_count": 0.70,
    "cwe_252_score": 0.65,
    "unchecked_call_result_count": 0.70,
    "integer_arithmetic_count": 0.30,
    "state_transition_count": 0.20,
    "numeric_conversion_count": 0.35,
    "cwe_integer_score": 0.25,
    "arithmetic_to_memory_sink_count": 0.55,
    "arithmetic_to_allocation_count": 0.45,
    "source_to_sink_count": 0.40,
    "memory_copy_count": 0.15,
    "array_access_count": 0.10,
    "dereference_count": 0.10,
}


class SuspicionScorer:
    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = dict(weights or RISK_WEIGHTS)

    def score(self, features: dict[str, float]) -> float:
        raw_score = sum(
            weight * min(max(float(features.get(name, 0.0)), 0.0), 3.0)
            for name, weight in self.weights.items()
        )
        return 1.0 - math.exp(-raw_score)

    def reasons(self, features: dict[str, float]) -> list[str]:
        contributions = [
            (self.weights[name] * min(max(float(features.get(name, 0.0)), 0.0), 3.0), name, float(features.get(name, 0.0)))
            for name in self.weights
            if features.get(name, 0.0) > 0.0
        ]
        return [
            f"{name}={value:g}"
            for _, name, value in sorted(contributions, key=lambda item: (-item[0], item[1]))
        ]
