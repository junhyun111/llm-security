from __future__ import annotations

from dataclasses import dataclass

from ..analysis.suspicion import SuspicionScorer
from ..models import Candidate


@dataclass(slots=True, frozen=True)
class GateDecision:
    candidate_id: str
    score: float
    threshold: float
    accepted: bool
    reasons: list[str]


@dataclass(slots=True, frozen=True)
class GateMetrics:
    candidate_count: int
    accepted_count: int
    rejected_count: int
    reduction_rate: float


@dataclass(slots=True, frozen=True)
class GateCalibration:
    threshold: float
    target_recall: float
    achieved_recall: float
    accepted_count: int
    target_met: bool


class CandidateGate:
    def __init__(
        self,
        *,
        threshold: float = 0.40,
        enabled: bool = True,
        scorer: SuspicionScorer | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Candidate gate threshold must be between 0 and 1")
        self.threshold = threshold
        self.enabled = enabled
        self.scorer = scorer or SuspicionScorer()

    def decide(self, candidate: Candidate) -> GateDecision:
        accepted = not self.enabled or candidate.suspicion_score >= self.threshold
        reasons = self.scorer.reasons(candidate.features)
        if not reasons:
            reasons = [f"suspicion_score={candidate.suspicion_score:.4f}"]
        reasons.append(
            "candidate gate disabled"
            if not self.enabled
            else (
                f"accepted: score >= {self.threshold:.4f}"
                if accepted
                else f"rejected: score < {self.threshold:.4f}"
            )
        )
        return GateDecision(
            candidate_id=candidate.candidate_id,
            score=candidate.suspicion_score,
            threshold=self.threshold,
            accepted=accepted,
            reasons=reasons,
        )

    def filter(
        self, candidates: list[Candidate]
    ) -> tuple[list[Candidate], list[GateDecision]]:
        decisions = [self.decide(candidate) for candidate in candidates]
        accepted = [
            candidate
            for candidate, decision in zip(candidates, decisions, strict=True)
            if decision.accepted
        ]
        return accepted, decisions

    def metrics(self, decisions: list[GateDecision]) -> GateMetrics:
        accepted = sum(decision.accepted for decision in decisions)
        count = len(decisions)
        return GateMetrics(
            candidate_count=count,
            accepted_count=accepted,
            rejected_count=count - accepted,
            reduction_rate=(count - accepted) / count if count else 0.0,
        )
