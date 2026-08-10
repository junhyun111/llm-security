from __future__ import annotations

import math
from dataclasses import dataclass

from ..models import ExpertFamily


@dataclass(slots=True, frozen=True)
class RoutingPolicyConfig:
    high_confidence: float = 0.72
    min_margin: float = 0.18
    max_entropy: float = 1.0
    max_experts: int = 2


@dataclass(slots=True)
class PolicySelection:
    selected: list[ExpertFamily]
    top1_confidence: float
    top1_top2_margin: float
    entropy: float
    reasons: list[str]


class AdaptiveTopKPolicy:
    name = "adaptive_top_k"

    def __init__(self, config: RoutingPolicyConfig | None = None) -> None:
        self.config = config or RoutingPolicyConfig()
        self._validate()

    def select(self, scores: dict[ExpertFamily, float]) -> list[ExpertFamily]:
        return self.decide(scores).selected

    def decide(self, scores: dict[ExpertFamily, float]) -> PolicySelection:
        if not scores:
            raise ValueError("AdaptiveTopKPolicy requires at least one Expert score")
        ranked = sorted(scores, key=lambda family: (-scores[family], family.value))
        first = ranked[0]
        p1 = float(scores[first])
        p2 = float(scores[ranked[1]]) if len(ranked) > 1 else 0.0
        margin = p1 - p2
        entropy = -sum(
            probability * math.log(probability)
            for probability in scores.values()
            if probability > 0.0
        )
        confident = (
            p1 >= self.config.high_confidence
            and margin >= self.config.min_margin
            and entropy <= self.config.max_entropy
        )
        if confident or self.config.max_experts == 1 or len(ranked) == 1:
            selected = [first]
            reasons = [
                "top-1 selected: confidence, margin, and entropy satisfy policy"
            ]
        else:
            selected = ranked[:2]
            reasons = ["top-2 selected: routing uncertainty"]
            if p1 < self.config.high_confidence:
                reasons.append("top-1 confidence below threshold")
            if margin < self.config.min_margin:
                reasons.append("top-1/top-2 margin below threshold")
            if entropy > self.config.max_entropy:
                reasons.append("routing entropy above threshold")
        return PolicySelection(selected, p1, margin, entropy, reasons)

    def _validate(self) -> None:
        if not 0.0 <= self.config.high_confidence <= 1.0:
            raise ValueError("high_confidence must be between 0 and 1")
        if not 0.0 <= self.config.min_margin <= 1.0:
            raise ValueError("min_margin must be between 0 and 1")
        if self.config.max_entropy < 0.0:
            raise ValueError("max_entropy cannot be negative")
        if self.config.max_experts not in {1, 2}:
            raise ValueError("Adaptive routing supports at most two Experts")
