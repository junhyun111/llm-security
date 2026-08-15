from __future__ import annotations

import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..datasets import UtilitySample
from ..models import Candidate, ExpertAssignment, ExpertFamily, RouteDecision
from .anchor import DEFAULT_ANCHORS
from .model import UtilityRoutingModel


@dataclass(slots=True)
class UtilityPolicyConfig:
    max_assignments: int = 3
    max_expected_cost: float = 0.0
    minimum_success_probability: float = 0.20
    cost_weight: float = 1.0
    false_positive_weight: float = 0.50
    unsupported_weight: float = 0.25
    anchor_families: tuple[ExpertFamily, ...] = DEFAULT_ANCHORS

    def validate(self) -> None:
        if self.max_assignments < 1:
            raise ValueError("max_assignments must be positive")
        if self.max_expected_cost < 0.0:
            raise ValueError("max_expected_cost cannot be negative")
        if not 0.0 <= self.minimum_success_probability <= 1.0:
            raise ValueError("minimum_success_probability must be between 0 and 1")


@dataclass(slots=True)
class AssignmentStatistics:
    samples: int
    success_rate: float
    false_positive_rate: float
    unsupported_claim_rate: float
    average_cost: float


@dataclass(slots=True)
class UtilityRouterMetrics:
    candidate_count: int
    success_coverage: float
    oracle_coverage: float
    average_assignments: float
    average_expected_cost: float
    average_realized_reward: float
    average_oracle_reward: float
    average_regret: float


class BudgetedUtilityRouter:
    """Select Expert x model jobs by predicted success and measured penalties."""

    artifact_version = 1

    def __init__(
        self,
        model: UtilityRoutingModel,
        assignments: dict[str, ExpertAssignment],
        statistics: dict[str, AssignmentStatistics],
        *,
        policy: UtilityPolicyConfig | None = None,
        feature_schema_version: str = "semantic-v1",
    ) -> None:
        self.model = model
        self.assignments = dict(assignments)
        self.statistics = dict(statistics)
        self.policy = policy or UtilityPolicyConfig()
        self.policy.validate()
        self.feature_schema_version = feature_schema_version
        self._artifact_version = self.artifact_version

    @classmethod
    def fit(
        cls,
        samples: Iterable[UtilitySample],
        *,
        policy: UtilityPolicyConfig | None = None,
        seed: int = 2026,
    ) -> "BudgetedUtilityRouter":
        materialized = list(samples)
        if not materialized:
            raise ValueError("Utility Router training requires outcome samples")
        schemas = {sample.candidate.feature_schema_version for sample in materialized}
        if len(schemas) != 1:
            raise ValueError(
                "Utility samples must use exactly one feature schema; got: "
                + ", ".join(sorted(schemas))
            )
        assignments = {
            sample.assignment.assignment_id: sample.assignment for sample in materialized
        }
        model = UtilityRoutingModel(seed=seed).fit(
            [sample.candidate.features for sample in materialized],
            [sample.assignment.assignment_id for sample in materialized],
            [sample.success for sample in materialized],
        )
        grouped: dict[str, list[UtilitySample]] = defaultdict(list)
        for sample in materialized:
            grouped[sample.assignment.assignment_id].append(sample)
        statistics = {}
        for assignment_id, rows in grouped.items():
            count = len(rows)
            statistics[assignment_id] = AssignmentStatistics(
                samples=count,
                success_rate=sum(item.success for item in rows) / count,
                false_positive_rate=sum(item.false_positive for item in rows) / count,
                unsupported_claim_rate=sum(item.unsupported_claims for item in rows) / count,
                average_cost=sum(item.cost for item in rows) / count,
            )
        return cls(
            model,
            assignments,
            statistics,
            policy=policy,
            feature_schema_version=next(iter(schemas)),
        )

    def route(self, candidate: Candidate) -> RouteDecision:
        if candidate.feature_schema_version != self.feature_schema_version:
            raise ValueError(
                f"Router expects {self.feature_schema_version} but candidate uses "
                f"{candidate.feature_schema_version}. Train or load a matching Router."
            )
        probabilities = self.model.predict_proba(candidate.features)
        utilities = {
            assignment_id: self._utility(assignment_id, probability)
            for assignment_id, probability in probabilities.items()
        }
        ranked = sorted(
            probabilities,
            key=lambda item: (-utilities[item], -probabilities[item], item),
        )
        selected_ids: list[str] = []
        for family in self.policy.anchor_families:
            choices = [
                item for item in ranked if self.assignments[item].expert == family
            ]
            if choices:
                self._try_add(choices[0], selected_ids)
        for assignment_id in ranked:
            if len(selected_ids) >= self.policy.max_assignments:
                break
            if assignment_id in selected_ids:
                continue
            if probabilities[assignment_id] < self.policy.minimum_success_probability:
                continue
            self._try_add(assignment_id, selected_ids)
        selected_assignments = [self.assignments[item] for item in selected_ids]
        selected_families = list(
            dict.fromkeys(assignment.expert for assignment in selected_assignments)
        )
        family_scores: dict[ExpertFamily, float] = {}
        for assignment_id, probability in probabilities.items():
            family = self.assignments[assignment_id].expert
            family_scores[family] = max(family_scores.get(family, 0.0), probability)
        ranked_family_scores = sorted(family_scores.values(), reverse=True)
        top = ranked_family_scores[0] if ranked_family_scores else 0.0
        second = ranked_family_scores[1] if len(ranked_family_scores) > 1 else 0.0
        expected_cost = sum(self.statistics[item].average_cost for item in selected_ids)
        reasons = [
            (
                f"selected {item}: p_success={probabilities[item]:.3f}, "
                f"utility={utilities[item]:.3f}, "
                f"expected_cost={self.statistics[item].average_cost:.6f}"
            )
            for item in selected_ids
        ]
        return RouteDecision(
            candidate_id=candidate.candidate_id,
            scores=family_scores,
            selected=selected_families,
            top1_confidence=top,
            top1_top2_margin=top - second,
            policy="budgeted_utility",
            reasons=reasons,
            available_families=sorted(family_scores, key=lambda family: family.value),
            learned_scores=family_scores,
            assignments=selected_assignments,
            expected_cost=expected_cost,
        )

    def evaluate(self, samples: Iterable[UtilitySample]) -> UtilityRouterMetrics:
        grouped: dict[str, list[UtilitySample]] = defaultdict(list)
        for sample in samples:
            grouped[sample.candidate.candidate_id].append(sample)
        if not grouped:
            raise ValueError("Utility Router evaluation requires outcome samples")
        selected_successes = oracle_successes = assignments = 0
        expected_cost = realized_reward = oracle_reward = 0.0
        for rows in grouped.values():
            decision = self.route(rows[0].candidate)
            selected_ids = {item.assignment_id for item in decision.assignments}
            selected_rows = [
                item for item in rows if item.assignment.assignment_id in selected_ids
            ]
            selected_successes += int(any(item.success for item in selected_rows))
            oracle_successes += int(any(item.success for item in rows))
            assignments += len(decision.assignments)
            expected_cost += decision.expected_cost
            selected_reward = (
                float(any(item.success for item in selected_rows))
                - self.policy.false_positive_weight
                * sum(item.false_positive for item in selected_rows)
                - self.policy.unsupported_weight
                * sum(item.unsupported_claims for item in selected_rows)
                - self.policy.cost_weight * sum(item.cost for item in selected_rows)
            )
            best_reward = max(
                (
                    item.reward(
                        false_positive_weight=self.policy.false_positive_weight,
                        unsupported_weight=self.policy.unsupported_weight,
                        cost_weight=self.policy.cost_weight,
                    )
                    for item in rows
                ),
                default=0.0,
            )
            realized_reward += selected_reward
            oracle_reward += best_reward
        count = len(grouped)
        return UtilityRouterMetrics(
            candidate_count=count,
            success_coverage=selected_successes / count,
            oracle_coverage=oracle_successes / count,
            average_assignments=assignments / count,
            average_expected_cost=expected_cost / count,
            average_realized_reward=realized_reward / count,
            average_oracle_reward=oracle_reward / count,
            average_regret=max(0.0, oracle_reward - realized_reward) / count,
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "BudgetedUtilityRouter":
        with Path(path).open("rb") as handle:
            router = pickle.load(handle)  # noqa: S301 - trusted local artifact
        if not isinstance(router, cls) or getattr(router, "_artifact_version", None) != cls.artifact_version:
            raise TypeError("Incompatible Utility Router artifact; retrain it")
        return router

    def _utility(self, assignment_id: str, probability: float) -> float:
        stats = self.statistics[assignment_id]
        return (
            probability
            - self.policy.false_positive_weight * stats.false_positive_rate
            - self.policy.unsupported_weight * stats.unsupported_claim_rate
            - self.policy.cost_weight * stats.average_cost
        )

    def _try_add(self, assignment_id: str, selected_ids: list[str]) -> bool:
        if len(selected_ids) >= self.policy.max_assignments:
            return False
        cost = self.statistics[assignment_id].average_cost
        current = sum(self.statistics[item].average_cost for item in selected_ids)
        if self.policy.max_expected_cost and current + cost > self.policy.max_expected_cost:
            return False
        selected_ids.append(assignment_id)
        return True
