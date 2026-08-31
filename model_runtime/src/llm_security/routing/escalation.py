from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from ..models import ACTIVE_UTILITY_EXPERTS, Candidate, ExpertFamily
from .model import BinaryRoutingModel


@dataclass(slots=True)
class EscalationTrainingRow:
    candidate: Candidate
    family_probabilities: dict[ExpertFamily, float]
    ranked_experts: list[ExpertFamily]
    top2_sufficient: bool


class EscalationGate:
    """Estimate whether the Utility-ranked Top-2 covers all candidate truths."""

    def __init__(self, *, seed: int = 2026) -> None:
        self.seed = seed
        self.model = BinaryRoutingModel(seed=seed)
        self._fitted = False

    def fit(self, rows: Iterable[EscalationTrainingRow]) -> "EscalationGate":
        materialized = list(rows)
        if not materialized:
            raise ValueError("Escalation Gate training requires outcome rows")
        self.model.fit(
            [
                escalation_features(
                    row.candidate,
                    row.family_probabilities,
                    row.ranked_experts,
                )
                for row in materialized
            ],
            [row.top2_sufficient for row in materialized],
        )
        self._fitted = True
        return self

    def predict_proba(
        self,
        candidate: Candidate,
        family_probabilities: dict[ExpertFamily, float],
        ranked_experts: list[ExpertFamily],
    ) -> float:
        if not self._fitted:
            raise RuntimeError("EscalationGate.fit must be called before inference")
        return self.model.predict_proba(
            escalation_features(candidate, family_probabilities, ranked_experts)
        )


def escalation_features(
    candidate: Candidate,
    family_probabilities: dict[ExpertFamily, float],
    ranked_experts: list[ExpertFamily],
) -> dict[str, float]:
    probabilities = [
        max(0.0, min(1.0, family_probabilities.get(family, 0.0)))
        for family in ACTIVE_UTILITY_EXPERTS
    ]
    ranked_probabilities = sorted(probabilities, reverse=True)
    p1 = ranked_probabilities[0] if ranked_probabilities else 0.0
    p2 = ranked_probabilities[1] if len(ranked_probabilities) > 1 else 0.0
    features = {
        f"candidate::{name}": float(value)
        for name, value in candidate.features.items()
    }
    features.update(
        {
            "candidate::suspicion_score": candidate.suspicion_score,
            "gate::p1": p1,
            "gate::p2": p2,
            "gate::margin": p1 - p2,
            "gate::independence_confidence": 1.0 - (1.0 - p1) * (1.0 - p2),
            "gate::mean_binary_entropy": sum(_binary_entropy(p) for p in probabilities)
            / max(1, len(probabilities)),
        }
    )
    for family, probability in zip(ACTIVE_UTILITY_EXPERTS, probabilities, strict=True):
        features[f"probability::{family.value}"] = probability
    for rank, family in enumerate(ranked_experts[:2], start=1):
        features[f"top{rank}::{family.value}"] = 1.0
    return features


def independence_top2_confidence(probabilities: list[float]) -> float:
    if not probabilities:
        return 0.0
    miss_probability = 1.0
    for probability in probabilities[:2]:
        miss_probability *= 1.0 - max(0.0, min(1.0, probability))
    return 1.0 - miss_probability


def _binary_entropy(probability: float) -> float:
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return -(
        probability * math.log2(probability)
        + (1.0 - probability) * math.log2(1.0 - probability)
    )
