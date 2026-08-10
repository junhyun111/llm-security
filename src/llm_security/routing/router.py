from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable, Protocol

from ..datasets import RouterSample
from ..models import Candidate, ExpertFamily, RouteDecision
from .model import ExpertRoutingModel, SoftmaxRoutingModel
from .policy import AdaptiveTopKPolicy, RoutingPolicyConfig


class Router(Protocol):
    def route(self, candidate: Candidate) -> RouteDecision: ...


@dataclass(slots=True)
class RouterMetrics:
    routing_accuracy: float
    coverage_at_1: float
    coverage_at_2: float
    adaptive_coverage: float
    average_experts_per_candidate: float
    average_top1_confidence: float
    average_top1_top2_margin: float
    expected_calibration_error: float
    routing_latency_ms: float
    llm_calls_saved: int
    sample_count: int


@dataclass(slots=True)
class PolicyCalibration:
    high_confidence: float
    min_margin: float
    target_coverage: float
    achieved_coverage: float
    average_experts_per_candidate: float
    target_met: bool


class RuleTriggerFallback:
    """Rule assistance for Expert families absent from Router training data."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled

    def score(
        self,
        candidate: Candidate,
        learned_families: Iterable[ExpertFamily],
    ) -> tuple[dict[ExpertFamily, float], list[str]]:
        if not self.enabled:
            return {}, []
        learned = set(learned_families)
        features = candidate.features
        integer_trigger = (
            features.get("integer_size_arithmetic_count", 0.0) > 0.0
            and (
                features.get("cast_count", 0.0) > 0.0
                or features.get("arithmetic_to_memory_sink", 0.0) > 0.0
                or features.get("allocation_count", 0.0) > 0.0
            )
        )
        taint_trigger = (
            features.get("source_sink_path", 0.0) > 0.0
            or features.get("external_input_to_sink", 0.0) > 0.0
        )
        concurrency_trigger = (
            features.get("toctou_pair", 0.0) > 0.0
            or (
                features.get("thread_signal_count", 0.0) > 0.0
                and features.get("lock_signal_count", 0.0) > 0.0
            )
        )
        raw = {
            ExpertFamily.INTEGER_SIZE_TYPE: 0.30 if integer_trigger else 0.0,
            ExpertFamily.TAINT_API_CONTRACT: 0.30 if taint_trigger else 0.0,
            ExpertFamily.CONCURRENCY_TOCTOU: 0.30 if concurrency_trigger else 0.0,
        }
        scores = {
            family: value
            for family, value in raw.items()
            if family not in learned and value > 0.0
        }
        reasons = [
            f"rule fallback activated for unlearned family: {family.value}"
            for family in sorted(scores, key=lambda item: item.value)
        ]
        return scores, reasons


class AdaptiveExpertRouter:
    artifact_version = 2

    def __init__(
        self,
        model: ExpertRoutingModel,
        policy: AdaptiveTopKPolicy,
        triggers: RuleTriggerFallback | None = None,
    ) -> None:
        self.model = model
        self.policy = policy
        self.triggers = triggers or RuleTriggerFallback(enabled=False)

    @property
    def available_families(self) -> tuple[ExpertFamily, ...]:
        return self.model.available_families

    @classmethod
    def fit(
        cls,
        samples: Iterable[RouterSample],
        *,
        policy_config: RoutingPolicyConfig | None = None,
        seed: int = 2026,
        use_rule_fallback: bool = True,
    ) -> "AdaptiveExpertRouter":
        materialized = list(samples)
        labels = [_single_label(sample) for sample in materialized]
        model = SoftmaxRoutingModel(seed=seed).fit(
            [sample.candidate.features for sample in materialized], labels
        )
        return cls(
            model=model,
            policy=AdaptiveTopKPolicy(policy_config),
            triggers=RuleTriggerFallback(enabled=use_rule_fallback),
        )

    def route(self, candidate: Candidate) -> RouteDecision:
        learned_scores = self.model.predict_proba(candidate.features)
        trigger_scores, trigger_reasons = self.triggers.score(
            candidate, learned_scores
        )
        combined = dict(learned_scores)
        for family, score in trigger_scores.items():
            combined[family] = max(combined.get(family, 0.0), score)
        total = sum(combined.values())
        scores = (
            {family: score / total for family, score in combined.items()}
            if total > 0.0
            else combined
        )
        selection = self.policy.decide(scores)
        return RouteDecision(
            candidate_id=candidate.candidate_id,
            scores=scores,
            selected=selection.selected,
            top1_confidence=selection.top1_confidence,
            top1_top2_margin=selection.top1_top2_margin,
            policy=self.policy.name,
            reasons=selection.reasons + trigger_reasons,
            available_families=sorted(scores, key=lambda family: family.value),
            learned_scores=learned_scores,
            trigger_scores=trigger_scores,
        )

    def evaluate(self, samples: Iterable[RouterSample]) -> RouterMetrics:
        materialized = list(samples)
        if not materialized:
            raise ValueError("At least one routing evaluation sample is required")
        labels = [_single_label(sample) for sample in materialized]
        unseen = set(labels) - set(self.model.available_families)
        if unseen:
            raise ValueError(
                "Evaluation contains Expert families absent from training: "
                + ", ".join(sorted(family.value for family in unseen))
            )
        started = perf_counter()
        decisions = [self.route(sample.candidate) for sample in materialized]
        elapsed = perf_counter() - started
        correct_top1 = 0
        covered_top2 = 0
        adaptive_covered = 0
        confidences: list[float] = []
        margins: list[float] = []
        correctness: list[int] = []
        expert_calls = 0
        for label, decision in zip(labels, decisions, strict=True):
            ranked = sorted(
                decision.scores,
                key=lambda family: (-decision.scores[family], family.value),
            )
            correct = int(label == ranked[0])
            correct_top1 += correct
            covered_top2 += int(label in ranked[:2])
            adaptive_covered += int(label in decision.selected)
            expert_calls += len(decision.selected)
            confidences.append(decision.top1_confidence)
            margins.append(decision.top1_top2_margin)
            correctness.append(correct)
        count = len(materialized)
        average_experts = expert_calls / count
        return RouterMetrics(
            routing_accuracy=correct_top1 / count,
            coverage_at_1=correct_top1 / count,
            coverage_at_2=covered_top2 / count,
            adaptive_coverage=adaptive_covered / count,
            average_experts_per_candidate=average_experts,
            average_top1_confidence=sum(confidences) / count,
            average_top1_top2_margin=sum(margins) / count,
            expected_calibration_error=_expected_calibration_error(
                confidences, correctness
            ),
            routing_latency_ms=elapsed * 1_000.0 / count,
            llm_calls_saved=len(ExpertFamily) * count - expert_calls,
            sample_count=count,
        )

    def calibrate_policy(
        self,
        samples: Iterable[RouterSample],
        *,
        target_coverage: float = 0.95,
    ) -> PolicyCalibration:
        materialized = list(samples)
        if not materialized:
            raise ValueError("Policy calibration requires dev samples")
        labels = [_single_label(sample) for sample in materialized]
        score_rows = [self.route(sample.candidate).scores for sample in materialized]
        candidates: list[tuple[float, float, float, float]] = []
        for confidence_step in range(50, 96, 5):
            for margin_step in range(5, 41, 5):
                policy = AdaptiveTopKPolicy(
                    RoutingPolicyConfig(
                        high_confidence=confidence_step / 100.0,
                        min_margin=margin_step / 100.0,
                        max_entropy=self.policy.config.max_entropy,
                        max_experts=self.policy.config.max_experts,
                    )
                )
                selected = [policy.select(scores) for scores in score_rows]
                coverage = sum(
                    label in families
                    for label, families in zip(labels, selected, strict=True)
                ) / len(labels)
                average = sum(map(len, selected)) / len(selected)
                candidates.append((coverage, average, confidence_step / 100.0, margin_step / 100.0))
        feasible = [item for item in candidates if item[0] >= target_coverage]
        if feasible:
            chosen = min(feasible, key=lambda item: (item[1], -item[0], item[2], item[3]))
        else:
            chosen = max(candidates, key=lambda item: (item[0], -item[1], -item[2], -item[3]))
        coverage, average, confidence, margin = chosen
        self.policy = AdaptiveTopKPolicy(
            RoutingPolicyConfig(
                high_confidence=confidence,
                min_margin=margin,
                max_entropy=self.policy.config.max_entropy,
                max_experts=self.policy.config.max_experts,
            )
        )
        return PolicyCalibration(
            high_confidence=confidence,
            min_margin=margin,
            target_coverage=target_coverage,
            achieved_coverage=coverage,
            average_experts_per_candidate=average,
            target_met=coverage >= target_coverage,
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "AdaptiveExpertRouter":
        with Path(path).open("rb") as handle:
            router = pickle.load(handle)  # noqa: S301 - trusted local artifact
        if not isinstance(router, cls) or router.artifact_version != cls.artifact_version:
            raise TypeError(
                "Incompatible Router artifact. Run 01_train_router.ipynb again."
            )
        return router


def _single_label(sample: RouterSample) -> ExpertFamily:
    if len(sample.labels) != 1:
        raise ValueError(
            f"Multiclass routing requires exactly one label for {sample.candidate.candidate_id}"
        )
    return sample.labels[0]


def _expected_calibration_error(
    confidences: list[float], correctness: list[int], *, bins: int = 10
) -> float:
    total = len(confidences)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            item
            for item, confidence in enumerate(confidences)
            if lower <= confidence < upper or (index == bins - 1 and confidence == 1.0)
        ]
        if not members:
            continue
        accuracy = sum(correctness[item] for item in members) / len(members)
        confidence = sum(confidences[item] for item in members) / len(members)
        error += len(members) / total * abs(accuracy - confidence)
    return error
