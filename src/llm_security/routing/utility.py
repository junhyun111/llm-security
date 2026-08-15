from __future__ import annotations

import hashlib
import math
import pickle
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable

from ..datasets import UTILITY_OUTCOME_LABEL_VERSION, UtilitySample
from ..models import (
    ACTIVE_UTILITY_EXPERTS,
    Candidate,
    ExpertAssignment,
    ExpertFamily,
    RouteDecision,
)
from .escalation import (
    EscalationGate,
    EscalationTrainingRow,
    independence_top2_confidence,
)
from .model import UtilityRoutingModel


@dataclass(slots=True)
class UtilityPolicyConfig:
    """Adaptive Top-2 / Full-5 allocation policy."""

    normal_top_k: int = 2
    full_expert_count: int = 5
    escalation_threshold: float = 0.85
    cost_weight: float = 1.0
    false_positive_weight: float = 0.50
    unsupported_weight: float = 0.25

    def validate(self) -> None:
        if self.normal_top_k != 2:
            raise ValueError("normal_top_k must be 2 for the Top-2 policy")
        if self.full_expert_count != len(ACTIVE_UTILITY_EXPERTS):
            raise ValueError(
                f"full_expert_count must be {len(ACTIVE_UTILITY_EXPERTS)}"
            )
        if not 0.0 <= self.escalation_threshold <= 1.0:
            raise ValueError("escalation_threshold must be between 0 and 1")
        for name, value in (
            ("cost_weight", self.cost_weight),
            ("false_positive_weight", self.false_positive_weight),
            ("unsupported_weight", self.unsupported_weight),
        ):
            if value < 0.0:
                raise ValueError(f"{name} cannot be negative")


@dataclass(slots=True)
class AssignmentStatistics:
    samples: int
    success_rate: float
    false_positive_rate: float
    unsupported_claim_rate: float
    average_cost: float
    average_prompt_tokens: float = 0.0
    average_completion_tokens: float = 0.0
    average_latency_seconds: float = 0.0


@dataclass(slots=True)
class EscalationCalibration:
    threshold: float
    target_truth_recall: float
    achieved_truth_recall: float
    exact_coverage: float
    average_assignments: float
    average_cost: float
    full5_rate: float
    candidate_count: int
    truth_count: int
    feasible: bool


@dataclass(slots=True)
class BaselineCalibration:
    """Dev-only choices used for honest Best-Single and Fixed-2 baselines."""

    best_single_assignment_id: str
    best_fixed_pair_assignment_ids: tuple[str, str]
    best_single_metrics: "UtilityRouterMetrics"
    best_fixed2_metrics: "UtilityRouterMetrics"


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
    truth_recall: float = 0.0
    exact_coverage: float = 0.0
    outcome_precision: float = 0.0
    outcome_f1: float = 0.0
    full5_rate: float = 0.0
    average_realized_cost: float = 0.0
    average_prompt_tokens: float = 0.0
    average_completion_tokens: float = 0.0
    average_latency_seconds: float = 0.0
    logical_expert_tasks: int = 0
    research_physical_requests: int = 0
    web_batched_requests: int = 0
    estimated_api_requests: int = 0
    brier_score: float = 0.0
    expected_calibration_error: float = 0.0
    escalation_recall: float = 0.0
    missed_escalation_rate: float = 0.0
    unnecessary_escalation_rate: float = 0.0
    gate_brier_score: float = 0.0
    gate_expected_calibration_error: float = 0.0


@dataclass(slots=True)
class _RankedCandidate:
    assignment_probabilities: dict[str, float]
    assignment_utilities: dict[str, float]
    ranked_assignment_ids: list[str]
    family_probabilities: dict[ExpertFamily, float]


class BudgetedUtilityRouter:
    """Rank five Experts by Utility, then choose Top-2 or escalate to Full-5.

    Independent models estimate assignment success. Utility is used only for
    ranking; escalation confidence comes from success probabilities or a
    learned Top2Sufficient gate.
    """

    artifact_version = 3
    expert_taxonomy_version = "five-expert-memory-safety-v1"

    def __init__(
        self,
        model: UtilityRoutingModel,
        assignments: dict[str, ExpertAssignment],
        statistics: dict[str, AssignmentStatistics],
        *,
        policy: UtilityPolicyConfig | None = None,
        feature_schema_version: str = "semantic-v1",
        escalation_gate: EscalationGate | None = None,
    ) -> None:
        self.model = model
        self.assignments = dict(assignments)
        self.statistics = dict(statistics)
        self.policy = policy or UtilityPolicyConfig()
        self.policy.validate()
        self.feature_schema_version = feature_schema_version
        self.escalation_gate = escalation_gate
        self.execution_model_id: str | None = None
        self.best_single_assignment_id: str | None = None
        self.best_fixed_pair_assignment_ids: tuple[str, str] | None = None
        self.training_project_ids: set[str] = set()
        self.development_project_ids: set[str] = set()
        self._artifact_version = self.artifact_version
        self._expert_taxonomy_version = self.expert_taxonomy_version
        self._validate_assignment_pool()

    @property
    def available_families(self) -> tuple[ExpertFamily, ...]:
        return ACTIVE_UTILITY_EXPERTS

    def restrict_to_model(self, model_id: str) -> "BudgetedUtilityRouter":
        """Bind routing to the physical model used by a one-request web run."""
        available = {
            assignment.expert
            for assignment in self.assignments.values()
            if assignment.model_id == model_id
            and assignment.expert in ACTIVE_UTILITY_EXPERTS
        }
        missing = set(ACTIVE_UTILITY_EXPERTS) - available
        if missing:
            raise ValueError(
                f"Utility artifact has no {model_id!r} outcome model for: "
                + ", ".join(sorted(family.value for family in missing))
                + ". Recollect/train with the same OPENROUTER_EXPERT_MODEL used by web."
            )
        self.execution_model_id = model_id
        return self

    @classmethod
    def fit(
        cls,
        samples: Iterable[UtilitySample],
        *,
        policy: UtilityPolicyConfig | None = None,
        seed: int = 2026,
    ) -> "BudgetedUtilityRouter":
        materialized = [
            sample
            for sample in samples
            if sample.assignment.expert in ACTIVE_UTILITY_EXPERTS
        ]
        if not materialized:
            raise ValueError("Utility Router training requires five-Expert outcome samples")
        invalid_labels = [
            sample
            for sample in materialized
            if not sample.truth_labels_available
            or sample.label_version != UTILITY_OUTCOME_LABEL_VERSION
            or not sample.case_id
        ]
        if invalid_labels:
            raise ValueError(
                "Utility Router requires semantic outcome labels with case IDs. "
                f"Expected label_version={UTILITY_OUTCOME_LABEL_VERSION}; "
                "recollect legacy outcomes before training."
            )
        schemas = {sample.candidate.feature_schema_version for sample in materialized}
        if len(schemas) != 1:
            raise ValueError(
                "Utility samples must use exactly one feature schema; got: "
                + ", ".join(sorted(schemas))
            )
        assignments = {
            sample.assignment.assignment_id: sample.assignment for sample in materialized
        }
        present = {assignment.expert for assignment in assignments.values()}
        missing = set(ACTIVE_UTILITY_EXPERTS) - present
        if missing:
            raise ValueError(
                "Utility outcomes are missing active Experts: "
                + ", ".join(sorted(family.value for family in missing))
            )
        _validate_complete_matrix(materialized, set(assignments))
        model = UtilityRoutingModel(seed=seed).fit(
            [sample.candidate.features for sample in materialized],
            [sample.assignment.assignment_id for sample in materialized],
            [sample.success for sample in materialized],
        )
        grouped: dict[str, list[UtilitySample]] = defaultdict(list)
        for sample in materialized:
            grouped[sample.assignment.assignment_id].append(sample)
        statistics: dict[str, AssignmentStatistics] = {}
        for assignment_id, rows in grouped.items():
            count = len(rows)
            statistics[assignment_id] = AssignmentStatistics(
                samples=count,
                success_rate=sum(item.success for item in rows) / count,
                false_positive_rate=sum(item.false_positive for item in rows) / count,
                unsupported_claim_rate=sum(item.unsupported_claims > 0 for item in rows)
                / count,
                average_cost=sum(item.cost for item in rows) / count,
                average_prompt_tokens=sum(item.prompt_tokens for item in rows) / count,
                average_completion_tokens=sum(item.completion_tokens for item in rows)
                / count,
                average_latency_seconds=sum(item.latency_seconds for item in rows) / count,
            )
        router = cls(
            model,
            assignments,
            statistics,
            policy=policy,
            feature_schema_version=next(iter(schemas)),
        )
        router.training_project_ids = {
            sample.candidate.project_id for sample in materialized
        }
        return router

    def fit_escalation_gate(
        self,
        samples: Iterable[UtilitySample],
        *,
        seed: int = 2026,
    ) -> int:
        groups = _group_samples(samples)
        self._require_truth_labels(groups)
        self.development_project_ids.update(
            row.candidate.project_id for rows in groups.values() for row in rows
        )
        rows: list[EscalationTrainingRow] = []
        for outcome_rows in groups.values():
            candidate = outcome_rows[0].candidate
            ranked = self._rank(candidate)
            top2_ids = ranked.ranked_assignment_ids[: self.policy.normal_top_k]
            rows.append(
                EscalationTrainingRow(
                    candidate=candidate,
                    family_probabilities=ranked.family_probabilities,
                    ranked_experts=[
                        self.assignments[item].expert
                        for item in ranked.ranked_assignment_ids
                    ],
                    top2_sufficient=_top2_sufficient(outcome_rows, top2_ids),
                )
            )
        self.escalation_gate = EscalationGate(seed=seed).fit(rows)
        return len(rows)

    def calibrate_threshold(
        self,
        samples: Iterable[UtilitySample],
        *,
        target_truth_recall: float = 0.95,
    ) -> EscalationCalibration:
        if not 0.0 <= target_truth_recall <= 1.0:
            raise ValueError("target_truth_recall must be between 0 and 1")
        groups = _group_samples(samples)
        if not groups:
            raise ValueError("Threshold calibration requires outcome samples")
        self._require_truth_labels(groups)
        self.development_project_ids.update(
            row.candidate.project_id for rows in groups.values() for row in rows
        )
        confidences = [
            self._escalation_confidence(rows[0].candidate, self._rank(rows[0].candidate))
            for rows in groups.values()
        ]
        thresholds = {0.0, 1.0, self.policy.escalation_threshold}
        thresholds.update(
            min(1.0, math.nextafter(confidence, math.inf))
            for confidence in confidences
        )
        candidates = [
            self._calibration_at(groups, threshold, target_truth_recall)
            for threshold in sorted(thresholds)
        ]
        feasible = [item for item in candidates if item.feasible]
        if feasible:
            selected = min(
                feasible,
                key=lambda item: (
                    item.average_cost,
                    item.average_assignments,
                    -item.achieved_truth_recall,
                    item.threshold,
                ),
            )
        else:
            selected = min(
                candidates,
                key=lambda item: (
                    -item.achieved_truth_recall,
                    item.average_cost,
                    item.average_assignments,
                ),
            )
        self.policy.escalation_threshold = selected.threshold
        return selected

    def calibrate_baselines(
        self, samples: Iterable[UtilitySample]
    ) -> BaselineCalibration:
        """Choose comparison baselines on dev data, never on the test split."""
        groups = _group_samples(samples)
        if not groups:
            raise ValueError("Baseline calibration requires outcome samples")
        self._require_truth_labels(groups)
        self.development_project_ids.update(
            row.candidate.project_id for rows in groups.values() for row in rows
        )

        assignment_ids = sorted(self.assignments)
        single_results = [
            (
                (assignment_id,),
                self._evaluate_groups(
                    groups, lambda rows, item=assignment_id: [item]
                ),
            )
            for assignment_id in assignment_ids
        ]
        pair_results = [
            (
                pair,
                self._evaluate_groups(
                    groups, lambda rows, items=pair: list(items)
                ),
            )
            for pair in combinations(assignment_ids, 2)
            if self.assignments[pair[0]].expert != self.assignments[pair[1]].expert
        ]
        if not pair_results:
            raise ValueError("At least two distinct Expert assignments are required")

        def selection_key(
            item: tuple[tuple[str, ...], UtilityRouterMetrics]
        ) -> tuple[float, float, float, tuple[str, ...]]:
            ids, metrics = item
            return (
                -metrics.truth_recall,
                -metrics.outcome_f1,
                metrics.average_realized_cost,
                ids,
            )

        single_ids, single_metrics = min(single_results, key=selection_key)
        pair_ids, pair_metrics = min(pair_results, key=selection_key)
        self.best_single_assignment_id = single_ids[0]
        self.best_fixed_pair_assignment_ids = (pair_ids[0], pair_ids[1])
        return BaselineCalibration(
            best_single_assignment_id=single_ids[0],
            best_fixed_pair_assignment_ids=(pair_ids[0], pair_ids[1]),
            best_single_metrics=single_metrics,
            best_fixed2_metrics=pair_metrics,
        )

    def assert_test_projects_unseen(
        self, samples: Iterable[UtilitySample]
    ) -> None:
        test_projects = {sample.candidate.project_id for sample in samples}
        overlap = test_projects & (
            self.training_project_ids | self.development_project_ids
        )
        if overlap:
            raise ValueError(
                "Test outcome projects overlap Router train/dev projects: "
                + ", ".join(sorted(overlap)[:5])
            )

    def route(self, candidate: Candidate) -> RouteDecision:
        ranked = self._rank(candidate)
        top2_ids = ranked.ranked_assignment_ids[: self.policy.normal_top_k]
        confidence = self._escalation_confidence(candidate, ranked)
        escalated = confidence < self.policy.escalation_threshold
        selected_ids = ranked.ranked_assignment_ids if escalated else top2_ids
        selected_assignments = [self.assignments[item] for item in selected_ids]
        ranked_experts = [
            self.assignments[item].expert for item in ranked.ranked_assignment_ids
        ]
        selected_experts = [assignment.expert for assignment in selected_assignments]
        top_probabilities = sorted(ranked.family_probabilities.values(), reverse=True)
        top = top_probabilities[0] if top_probabilities else 0.0
        second = top_probabilities[1] if len(top_probabilities) > 1 else 0.0
        method = "learned_gate" if self.escalation_gate is not None else "independence"
        reasons = [
            (
                f"rank {index}: {self.assignments[item].expert.value} via "
                f"{self.assignments[item].model_id}; "
                f"p_success={ranked.assignment_probabilities[item]:.3f}, "
                f"utility={ranked.assignment_utilities[item]:.3f}"
            )
            for index, item in enumerate(ranked.ranked_assignment_ids, start=1)
        ]
        reasons.append(
            f"{method} Top-2 sufficiency={confidence:.3f}; threshold="
            f"{self.policy.escalation_threshold:.3f}; "
            + ("Full-5 escalation" if escalated else "Top-2 path")
        )
        return RouteDecision(
            candidate_id=candidate.candidate_id,
            scores=ranked.family_probabilities,
            selected=selected_experts,
            top1_confidence=top,
            top1_top2_margin=top - second,
            policy="utility_full5_escalation" if escalated else "utility_top2",
            reasons=reasons,
            available_families=list(ACTIVE_UTILITY_EXPERTS),
            learned_scores=ranked.family_probabilities,
            assignments=selected_assignments,
            expected_cost=sum(
                self.statistics[item].average_cost for item in selected_ids
            ),
            ranked_experts=ranked_experts,
            top2_experts=ranked_experts[: self.policy.normal_top_k],
            escalation_confidence=confidence,
            escalated=escalated,
            escalation_method=method,
        )

    def evaluate(self, samples: Iterable[UtilitySample]) -> UtilityRouterMetrics:
        groups = _group_samples(samples)
        if not groups:
            raise ValueError("Utility Router evaluation requires outcome samples")
        self._require_truth_labels(groups)
        return self._evaluate_groups(
            groups,
            lambda rows: [
                assignment.assignment_id
                for assignment in self.route(rows[0].candidate).assignments
            ],
        )

    def evaluate_baselines(
        self,
        samples: Iterable[UtilitySample],
        *,
        anchor_router=None,
    ) -> dict[str, UtilityRouterMetrics]:
        groups = _group_samples(samples)
        if not groups:
            raise ValueError("Utility Router evaluation requires outcome samples")
        self._require_truth_labels(groups)

        def ranked_ids(rows: list[UtilitySample]) -> list[str]:
            return self._rank(rows[0].candidate).ranked_assignment_ids

        def fixed_top2(rows: list[UtilitySample]) -> list[str]:
            ranked = self._rank(rows[0].candidate)
            by_family = {
                self.assignments[item].expert: item
                for item in ranked.ranked_assignment_ids
            }
            return [
                by_family[family]
                for family in ACTIVE_UTILITY_EXPERTS[:2]
                if family in by_family
            ]

        def formula(rows: list[UtilitySample]) -> list[str]:
            ranked = self._rank(rows[0].candidate)
            top2 = ranked.ranked_assignment_ids[:2]
            confidence = independence_top2_confidence(
                [ranked.assignment_probabilities[item] for item in top2]
            )
            return (
                top2
                if confidence >= self.policy.escalation_threshold
                else ranked.ranked_assignment_ids
            )

        report = {
            "full5": self._evaluate_groups(groups, ranked_ids),
            "fixed_top2_e1_e3": self._evaluate_groups(groups, fixed_top2),
            "utility_top2": self._evaluate_groups(
                groups, lambda rows: ranked_ids(rows)[:2]
            ),
            "formula_escalation": self._evaluate_groups(groups, formula),
            "adaptive_gate": self._evaluate_groups(
                groups,
                lambda rows: [
                    item.assignment_id
                    for item in self.route(rows[0].candidate).assignments
                ],
            ),
        }
        if self.best_single_assignment_id is not None:
            assignment_id = self.best_single_assignment_id
            report["best_single"] = self._evaluate_groups(
                groups, lambda rows: [assignment_id]
            )
        if self.best_fixed_pair_assignment_ids is not None:
            pair = self.best_fixed_pair_assignment_ids
            report["best_fixed2"] = self._evaluate_groups(
                groups, lambda rows: list(pair)
            )
        if anchor_router is not None:
            def anchor_selection(rows: list[UtilitySample]) -> list[str]:
                ranked = self._rank(rows[0].candidate)
                by_family = {
                    self.assignments[item].expert: item
                    for item in ranked.ranked_assignment_ids
                }
                legacy = anchor_router.route(rows[0].candidate).selected
                mapped = [
                    ExpertFamily.MEMORY_SAFETY
                    if family == ExpertFamily.LIFETIME_RESOURCE
                    else family
                    for family in legacy
                ]
                return [
                    by_family[family]
                    for family in dict.fromkeys(mapped)
                    if family in by_family
                ]

            report["anchor_rare"] = self._evaluate_groups(
                groups, anchor_selection
            )
        return report

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "BudgetedUtilityRouter":
        with Path(path).open("rb") as handle:
            router = pickle.load(handle)  # noqa: S301 - trusted local artifact
        if (
            not isinstance(router, cls)
            or getattr(router, "_artifact_version", None) != cls.artifact_version
            or getattr(router, "_expert_taxonomy_version", None)
            != cls.expert_taxonomy_version
        ):
            raise TypeError("Incompatible Utility Router artifact; retrain it")
        return router

    def _validate_assignment_pool(self) -> None:
        available = {
            assignment.expert
            for assignment in self.assignments.values()
            if assignment.expert in ACTIVE_UTILITY_EXPERTS
        }
        missing = set(ACTIVE_UTILITY_EXPERTS) - available
        if missing:
            raise ValueError(
                "Utility Router assignment pool is missing: "
                + ", ".join(sorted(family.value for family in missing))
            )

    def _rank(self, candidate: Candidate) -> _RankedCandidate:
        if candidate.feature_schema_version != self.feature_schema_version:
            raise ValueError(
                f"Router expects {self.feature_schema_version} but candidate uses "
                f"{candidate.feature_schema_version}. Train or load a matching Router."
            )
        probabilities = self.model.predict_proba(candidate.features)
        utilities = {
            assignment_id: self._utility(assignment_id, probability)
            for assignment_id, probability in probabilities.items()
            if self.assignments[assignment_id].expert in ACTIVE_UTILITY_EXPERTS
            and (
                self.execution_model_id is None
                or self.assignments[assignment_id].model_id
                == self.execution_model_id
            )
        }
        best_by_family: dict[ExpertFamily, str] = {}
        for assignment_id in sorted(
            utilities,
            key=lambda item: (-utilities[item], -probabilities[item], item),
        ):
            family = self.assignments[assignment_id].expert
            best_by_family.setdefault(family, assignment_id)
        missing = set(ACTIVE_UTILITY_EXPERTS) - set(best_by_family)
        if missing:
            raise RuntimeError(
                "No routable assignment for: "
                + ", ".join(sorted(family.value for family in missing))
            )
        ranked_ids = sorted(
            best_by_family.values(),
            key=lambda item: (-utilities[item], -probabilities[item], item),
        )
        family_probabilities = {
            family: probabilities[assignment_id]
            for family, assignment_id in best_by_family.items()
        }
        return _RankedCandidate(
            assignment_probabilities=probabilities,
            assignment_utilities=utilities,
            ranked_assignment_ids=ranked_ids,
            family_probabilities=family_probabilities,
        )

    def _utility(self, assignment_id: str, probability: float) -> float:
        stats = self.statistics[assignment_id]
        return (
            probability
            - self.policy.false_positive_weight * stats.false_positive_rate
            - self.policy.unsupported_weight * stats.unsupported_claim_rate
            - self.policy.cost_weight * stats.average_cost
        )

    def _escalation_confidence(
        self, candidate: Candidate, ranked: _RankedCandidate
    ) -> float:
        if self.escalation_gate is not None:
            return self.escalation_gate.predict_proba(
                candidate,
                ranked.family_probabilities,
                [self.assignments[item].expert for item in ranked.ranked_assignment_ids],
            )
        return independence_top2_confidence(
            [
                ranked.assignment_probabilities[item]
                for item in ranked.ranked_assignment_ids[: self.policy.normal_top_k]
            ]
        )

    def _calibration_at(
        self,
        groups: dict[tuple[str, str], list[UtilitySample]],
        threshold: float,
        target: float,
    ) -> EscalationCalibration:
        total_truths = matched_truths = assignments = full5 = 0
        exact = 0
        cost = 0.0
        for rows in groups.values():
            ranked = self._rank(rows[0].candidate)
            confidence = self._escalation_confidence(rows[0].candidate, ranked)
            escalated = confidence < threshold
            selected_ids = (
                ranked.ranked_assignment_ids
                if escalated
                else ranked.ranked_assignment_ids[: self.policy.normal_top_k]
            )
            selected_rows = _selected_rows(rows, selected_ids)
            truths = _truth_ids(rows)
            matched = _matched_truth_ids(selected_rows) & truths
            total_truths += len(truths)
            matched_truths += len(matched)
            exact += int(truths <= matched)
            assignments += len(selected_ids)
            full5 += int(escalated)
            cost += sum(item.cost for item in selected_rows)
        count = len(groups)
        recall = matched_truths / total_truths if total_truths else 1.0
        return EscalationCalibration(
            threshold=threshold,
            target_truth_recall=target,
            achieved_truth_recall=recall,
            exact_coverage=exact / count,
            average_assignments=assignments / count,
            average_cost=cost / count,
            full5_rate=full5 / count,
            candidate_count=count,
            truth_count=total_truths,
            feasible=recall >= target,
        )

    def _evaluate_groups(
        self,
        groups: dict[tuple[str, str], list[UtilitySample]],
        selector: Callable[[list[UtilitySample]], list[str]],
    ) -> UtilityRouterMetrics:
        selected_successes = oracle_successes = assignments = full5 = exact = 0
        selected_true_outcomes = selected_false_outcomes = 0
        total_truths = matched_truths = 0
        expected_cost = realized_cost = realized_reward = oracle_reward = 0.0
        prompt_tokens = completion_tokens = latency = 0.0
        calibration_pairs: list[tuple[float, bool]] = []
        gate_calibration_pairs: list[tuple[float, bool]] = []
        web_cases: set[str] = set()
        escalation_tp = escalation_fn = escalation_fp = escalation_tn = 0
        for rows in groups.values():
            candidate = rows[0].candidate
            selected_ids = selector(rows)
            selected_rows = _selected_rows(rows, selected_ids)
            if selected_ids:
                web_cases.add(
                    rows[0].case_id
                    or f"{candidate.project_id}:{candidate.candidate_id}"
                )
            selected_successes += int(any(item.success for item in selected_rows))
            oracle_successes += int(any(item.success for item in rows))
            assignments += len(selected_ids)
            full5 += int(len(selected_ids) == len(ACTIVE_UTILITY_EXPERTS))
            ranked = self._rank(candidate)
            top2_ids = ranked.ranked_assignment_ids[: self.policy.normal_top_k]
            top2_sufficient = _top2_sufficient(rows, top2_ids)
            predicted_escalation = len(selected_ids) == len(ACTIVE_UTILITY_EXPERTS)
            gate_confidence = self._escalation_confidence(candidate, ranked)
            gate_calibration_pairs.append((gate_confidence, top2_sufficient))
            if not top2_sufficient and predicted_escalation:
                escalation_tp += 1
            elif not top2_sufficient:
                escalation_fn += 1
            elif predicted_escalation:
                escalation_fp += 1
            else:
                escalation_tn += 1
            truths = _truth_ids(rows)
            matched = _matched_truth_ids(selected_rows) & truths
            total_truths += len(truths)
            matched_truths += len(matched)
            exact += int(truths <= matched)
            selected_true_outcomes += sum(
                _validated_true_count(item) for item in selected_rows
            )
            selected_false_outcomes += sum(
                _validated_false_count(item) for item in selected_rows
            )
            expected_cost += sum(
                self.statistics[item].average_cost for item in selected_ids
            )
            realized_cost += sum(item.cost for item in selected_rows)
            prompt_tokens += sum(item.prompt_tokens for item in selected_rows)
            completion_tokens += sum(item.completion_tokens for item in selected_rows)
            latency += sum(item.latency_seconds for item in selected_rows)
            selected_reward = (
                float(any(item.success for item in selected_rows))
                - self.policy.false_positive_weight
                * sum(item.false_positive for item in selected_rows)
                - self.policy.unsupported_weight
                * sum(item.unsupported_claims for item in selected_rows)
                - self.policy.cost_weight * sum(item.cost for item in selected_rows)
            )
            realized_reward += selected_reward
            oracle_reward += max(
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
            predicted = self.model.predict_proba(candidate.features)
            calibration_pairs.extend(
                (predicted[item.assignment.assignment_id], item.success)
                for item in rows
                if item.assignment.assignment_id in predicted
            )
        count = len(groups)
        precision_denominator = selected_true_outcomes + selected_false_outcomes
        precision = (
            selected_true_outcomes / precision_denominator
            if precision_denominator
            else 1.0
        )
        recall = matched_truths / total_truths if total_truths else 1.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        brier = (
            sum((probability - float(label)) ** 2 for probability, label in calibration_pairs)
            / len(calibration_pairs)
            if calibration_pairs
            else 0.0
        )
        gate_brier = (
            sum(
                (probability - float(label)) ** 2
                for probability, label in gate_calibration_pairs
            )
            / len(gate_calibration_pairs)
            if gate_calibration_pairs
            else 0.0
        )
        required_escalations = escalation_tp + escalation_fn
        sufficient_top2 = escalation_fp + escalation_tn
        escalation_recall = (
            escalation_tp / required_escalations if required_escalations else 1.0
        )
        missed_escalation_rate = (
            escalation_fn / required_escalations if required_escalations else 0.0
        )
        unnecessary_escalation_rate = (
            escalation_fp / sufficient_top2 if sufficient_top2 else 0.0
        )
        return UtilityRouterMetrics(
            candidate_count=count,
            success_coverage=selected_successes / count,
            oracle_coverage=oracle_successes / count,
            average_assignments=assignments / count,
            average_expected_cost=expected_cost / count,
            average_realized_reward=realized_reward / count,
            average_oracle_reward=oracle_reward / count,
            average_regret=max(0.0, oracle_reward - realized_reward) / count,
            truth_recall=recall,
            exact_coverage=exact / count,
            outcome_precision=precision,
            outcome_f1=f1,
            full5_rate=full5 / count,
            average_realized_cost=realized_cost / count,
            average_prompt_tokens=prompt_tokens / count,
            average_completion_tokens=completion_tokens / count,
            average_latency_seconds=latency / count,
            logical_expert_tasks=assignments,
            research_physical_requests=assignments,
            web_batched_requests=len(web_cases),
            estimated_api_requests=len(web_cases),
            brier_score=brier,
            expected_calibration_error=_expected_calibration_error(calibration_pairs),
            escalation_recall=escalation_recall,
            missed_escalation_rate=missed_escalation_rate,
            unnecessary_escalation_rate=unnecessary_escalation_rate,
            gate_brier_score=gate_brier,
            gate_expected_calibration_error=_expected_calibration_error(
                gate_calibration_pairs
            ),
        )

    @staticmethod
    def _require_truth_labels(
        groups: dict[tuple[str, str], list[UtilitySample]],
    ) -> None:
        missing = [
            f"{key[0]}/{key[1]}"
            for key, rows in groups.items()
            if not all(
                item.truth_labels_available
                and item.case_id
                and item.label_version == UTILITY_OUTCOME_LABEL_VERSION
                for item in rows
            )
        ]
        if missing:
            preview = ", ".join(missing[:3])
            raise ValueError(
                "Utility evaluation requires semantic outcome rows with case IDs; "
                f"expected label_version={UTILITY_OUTCOME_LABEL_VERSION}. "
                f"Recollect legacy rows. Invalid labels for {preview}"
            )


def split_gate_calibration_samples(
    samples: Iterable[UtilitySample],
    *,
    seed: int = 2026,
    gate_fraction: float = 0.5,
) -> tuple[list[UtilitySample], list[UtilitySample]]:
    """Split a dev outcome matrix by project for gate fitting and calibration."""
    if not 0.0 < gate_fraction < 1.0:
        raise ValueError("gate_fraction must be between 0 and 1")
    rows = list(samples)
    projects = sorted({item.candidate.project_id for item in rows})
    if len(projects) < 2:
        raise ValueError("At least two dev projects are required to split gate/calibration")
    ordered = sorted(
        projects,
        key=lambda project: hashlib.sha256(
            f"{seed}:{project}".encode("utf-8")
        ).hexdigest(),
    )
    cut = min(len(ordered) - 1, max(1, round(len(ordered) * gate_fraction)))
    gate_projects = set(ordered[:cut])
    return (
        [item for item in rows if item.candidate.project_id in gate_projects],
        [item for item in rows if item.candidate.project_id not in gate_projects],
    )


def assert_project_disjoint(
    first: Iterable[UtilitySample],
    second: Iterable[UtilitySample],
    *,
    first_name: str,
    second_name: str,
) -> None:
    first_projects = {item.candidate.project_id for item in first}
    second_projects = {item.candidate.project_id for item in second}
    overlap = first_projects & second_projects
    if overlap:
        raise ValueError(
            f"{first_name}/{second_name} outcome projects overlap: "
            + ", ".join(sorted(overlap)[:5])
        )


def _group_samples(
    samples: Iterable[UtilitySample],
) -> dict[tuple[str, str], list[UtilitySample]]:
    grouped: dict[tuple[str, str], list[UtilitySample]] = defaultdict(list)
    for sample in samples:
        if sample.assignment.expert in ACTIVE_UTILITY_EXPERTS:
            grouped[(sample.case_id, sample.candidate.candidate_id)].append(
                sample
            )
    return dict(grouped)


def _validate_complete_matrix(
    samples: list[UtilitySample], expected_assignment_ids: set[str]
) -> None:
    groups = _group_samples(samples)
    invalid: list[str] = []
    for (case_id, candidate_id), rows in groups.items():
        ids = [item.assignment.assignment_id for item in rows]
        if set(ids) != expected_assignment_ids or len(ids) != len(set(ids)):
            invalid.append(f"{case_id}/{candidate_id}")
    if invalid:
        raise ValueError(
            "Utility training requires a complete, duplicate-free Expert x model "
            "matrix for every candidate. Invalid rows: "
            + ", ".join(invalid[:5])
        )


def _selected_rows(
    rows: list[UtilitySample], selected_ids: list[str]
) -> list[UtilitySample]:
    by_assignment = {item.assignment.assignment_id: item for item in rows}
    missing = [item for item in selected_ids if item not in by_assignment]
    if missing:
        raise ValueError(
            "Evaluation outcome matrix is missing selected assignments: "
            + ", ".join(missing)
        )
    return [by_assignment[item] for item in selected_ids]


def _truth_ids(rows: list[UtilitySample]) -> set[str]:
    return {truth_id for item in rows for truth_id in item.ground_truth_ids}


def _matched_truth_ids(rows: list[UtilitySample]) -> set[str]:
    return {truth_id for item in rows for truth_id in item.matched_truth_ids}


def _top2_sufficient(rows: list[UtilitySample], top2_ids: list[str]) -> bool:
    truths = _truth_ids(rows)
    matched = _matched_truth_ids(_selected_rows(rows, top2_ids))
    return truths <= matched


def _validated_true_count(sample: UtilitySample) -> int:
    if sample.validated_true_findings or sample.validated_false_findings:
        return sample.validated_true_findings
    return int(sample.success)


def _validated_false_count(sample: UtilitySample) -> int:
    if sample.validated_true_findings or sample.validated_false_findings:
        return sample.validated_false_findings
    return int(sample.false_positive)


def _expected_calibration_error(
    pairs: list[tuple[float, bool]], *, bins: int = 10
) -> float:
    if not pairs:
        return 0.0
    total = len(pairs)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        bucket = [
            (probability, label)
            for probability, label in pairs
            if lower <= probability < upper
            or (index == bins - 1 and probability == 1.0)
        ]
        if not bucket:
            continue
        confidence = sum(item[0] for item in bucket) / len(bucket)
        accuracy = sum(item[1] for item in bucket) / len(bucket)
        error += len(bucket) / total * abs(confidence - accuracy)
    return error
