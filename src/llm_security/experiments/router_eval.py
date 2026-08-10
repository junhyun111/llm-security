from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import numpy as np

from ..datasets import RouterSample
from ..models import ExpertFamily
from ..routing import AdaptiveExpertRouter


@dataclass(slots=True)
class SupportedRouterMetrics:
    supported_sample_count: int
    supported_families: list[str]
    learned_top1_accuracy: float
    learned_coverage_at_2: float
    learned_ece: float
    routing_latency_ms: float


@dataclass(slots=True)
class FullHybridRouterMetrics:
    sample_count: int
    coverage_at_1: float
    coverage_at_2: float
    adaptive_coverage: float
    average_experts_per_candidate: float
    supported_family_coverage: float
    unsupported_family_coverage: float
    fallback_activation_count: int
    fallback_correct_activation_count: int
    routing_latency_ms: float
    llm_calls_saved_vs_all_six: int


def evaluate_supported_router(
    router: AdaptiveExpertRouter, samples: Iterable[RouterSample]
) -> SupportedRouterMetrics:
    supported = [
        sample
        for sample in samples
        if len(sample.labels) == 1 and sample.labels[0] in router.available_families
    ]
    mismatched = [
        sample.candidate.candidate_id
        for sample in supported
        if sample.candidate.feature_schema_version != router.feature_schema_version
    ]
    if mismatched:
        raise ValueError(
            f"Router expects {router.feature_schema_version}; schema mismatch for "
            + ", ".join(mismatched[:5])
        )
    labels = [sample.labels[0] for sample in supported]
    started = perf_counter()
    score_rows = [router.model.predict_proba(sample.candidate.features) for sample in supported]
    elapsed = perf_counter() - started
    top1_hits = 0
    top2_hits = 0
    confidences: list[float] = []
    correctness: list[int] = []
    for label, scores in zip(labels, score_rows, strict=True):
        ranked = _rank(scores)
        correct = int(bool(ranked) and label == ranked[0])
        top1_hits += correct
        top2_hits += int(label in ranked[:2])
        confidences.append(float(scores[ranked[0]]) if ranked else 0.0)
        correctness.append(correct)
    count = len(supported)
    return SupportedRouterMetrics(
        supported_sample_count=count,
        supported_families=[family.value for family in router.available_families],
        learned_top1_accuracy=_ratio(top1_hits, count),
        learned_coverage_at_2=_ratio(top2_hits, count),
        learned_ece=_expected_calibration_error(confidences, correctness),
        routing_latency_ms=(elapsed * 1_000.0 / count) if count else 0.0,
    )


def evaluate_full_hybrid_router(
    router: AdaptiveExpertRouter,
    samples: Iterable[RouterSample],
    *,
    selection_mode: str = "adaptive",
    fallback_enabled: bool = True,
    fallback_mode: str = "score_fusion",
) -> FullHybridRouterMetrics:
    if selection_mode not in {"top1", "top2", "adaptive"}:
        raise ValueError("selection_mode must be top1, top2, or adaptive")
    materialized = [sample for sample in samples if len(sample.labels) == 1]
    previous_enabled = router.triggers.enabled
    previous_mode = router.triggers.mode
    router.triggers.enabled = fallback_enabled
    router.triggers.mode = fallback_mode
    try:
        started = perf_counter()
        decisions = [router.route(sample.candidate) for sample in materialized]
        elapsed = perf_counter() - started
    finally:
        router.triggers.enabled = previous_enabled
        router.triggers.mode = previous_mode

    top1_hits = top2_hits = adaptive_hits = expert_calls = 0
    supported_hits = unsupported_hits = 0
    supported_count = unsupported_count = 0
    fallback_count = fallback_correct = 0
    supported_set = set(router.available_families)
    for sample, decision in zip(materialized, decisions, strict=True):
        label = sample.labels[0]
        ranked = _rank(decision.scores)
        top1_hits += int(label in ranked[:1])
        top2_hits += int(label in ranked[:2])
        selected = _selected_for_mode(decision.selected, ranked, selection_mode)
        adaptive_hits += int(label in selected)
        expert_calls += len(selected)
        if label in supported_set:
            supported_count += 1
            supported_hits += int(label in selected)
        else:
            unsupported_count += 1
            unsupported_hits += int(label in selected)
        activated = bool(decision.trigger_scores)
        fallback_count += int(activated)
        fallback_correct += int(activated and label in decision.trigger_scores)
    count = len(materialized)
    return FullHybridRouterMetrics(
        sample_count=count,
        coverage_at_1=_ratio(top1_hits, count),
        coverage_at_2=_ratio(top2_hits, count),
        adaptive_coverage=_ratio(adaptive_hits, count),
        average_experts_per_candidate=_ratio(expert_calls, count),
        supported_family_coverage=_ratio(supported_hits, supported_count),
        unsupported_family_coverage=_ratio(unsupported_hits, unsupported_count),
        fallback_activation_count=fallback_count,
        fallback_correct_activation_count=fallback_correct,
        routing_latency_ms=(elapsed * 1_000.0 / count) if count else 0.0,
        llm_calls_saved_vs_all_six=len(ExpertFamily) * count - expert_calls,
    )


def all_six_baseline(sample_count: int) -> dict[str, int | float]:
    return {
        "sample_count": sample_count,
        "coverage": 1.0 if sample_count else 0.0,
        "average_experts_per_candidate": float(len(ExpertFamily)),
        "expert_calls": len(ExpertFamily) * sample_count,
        "llm_calls_saved_vs_all_six": 0,
    }


def confusion_matrix(
    router: AdaptiveExpertRouter, samples: Iterable[RouterSample]
) -> dict[str, object]:
    labels = list(router.available_families)
    positions = {family: index for index, family in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    sample_count = 0
    for sample in samples:
        if len(sample.labels) != 1 or sample.labels[0] not in positions:
            continue
        scores = router.model.predict_proba(sample.candidate.features)
        predicted = _rank(scores)[0]
        matrix[positions[sample.labels[0]]][positions[predicted]] += 1
        sample_count += 1
    return {
        "labels": [family.value for family in labels],
        "matrix": matrix,
        "sample_count": sample_count,
    }


def feature_importance(
    router: AdaptiveExpertRouter, *, top_n: int = 10
) -> dict[str, dict[str, list[dict[str, float | str]]]]:
    model = router.model
    if not hasattr(model, "vectorizer") or not hasattr(model, "classifier"):
        raise TypeError("Feature importance requires SoftmaxRoutingModel")
    names = list(model.vectorizer.get_feature_names_out())
    coefficients = np.asarray(model.classifier.coef_)
    classes = [ExpertFamily(value) for value in model.classifier.classes_]
    if len(classes) == 2 and coefficients.shape[0] == 1:
        coefficients = np.vstack([-coefficients[0], coefficients[0]])
    output: dict[str, dict[str, list[dict[str, float | str]]]] = {}
    for family, row in zip(classes, coefficients, strict=True):
        positive_indices = sorted(
            range(len(names)), key=lambda index: (-row[index], names[index])
        )[:top_n]
        negative_indices = sorted(
            range(len(names)), key=lambda index: (row[index], names[index])
        )[:top_n]
        output[family.value] = {
            "positive": [
                {"feature": names[index], "coefficient": float(row[index])}
                for index in positive_indices
            ],
            "negative": [
                {"feature": names[index], "coefficient": float(row[index])}
                for index in negative_indices
            ],
        }
    return output


def _selected_for_mode(selected, ranked, mode: str):
    if mode == "top1":
        return ranked[:1]
    if mode == "top2":
        return ranked[:2]
    return selected


def _rank(scores):
    return sorted(scores, key=lambda family: (-scores[family], family.value))


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _expected_calibration_error(
    confidences: list[float], correctness: list[int], *, bins: int = 10
) -> float:
    if not confidences:
        return 0.0
    error = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [
            item
            for item, confidence in enumerate(confidences)
            if lower <= confidence < upper
            or (index == bins - 1 and confidence == 1.0)
        ]
        if members:
            accuracy = sum(correctness[item] for item in members) / len(members)
            confidence = sum(confidences[item] for item in members) / len(members)
            error += len(members) / len(confidences) * abs(accuracy - confidence)
    return error
