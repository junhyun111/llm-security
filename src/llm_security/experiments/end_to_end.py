from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Mapping

from ..models import Candidate, ExpertFamily, ProjectCase, RouteDecision
from ..routing import AdaptiveExpertRouter
from .analyzer_eval import candidate_matches_truth


@dataclass(slots=True)
class EndToEndMetrics:
    ground_truth_count: int
    pre_gate_candidate_hit_count: int
    post_gate_candidate_hit_count: int
    successful_ground_truth_count: int
    pre_gate_candidate_recall: float
    gate_retention: float
    post_gate_candidate_recall: float
    conditional_router_coverage: float
    end_to_end_routing_recall: float
    candidate_count_before: int
    candidate_count_after: int
    average_expert_calls_per_case: float
    llm_calls_saved_vs_all_six: int
    routing_latency_ms_per_case: float


@dataclass(slots=True)
class PerFamilyMetrics:
    learned_supported: bool
    ground_truth_count: int
    candidate_recall: float
    gate_retention: float
    top1_accuracy: float
    adaptive_coverage: float
    fallback_activation: float
    end_to_end_routing_recall: float


@dataclass(slots=True)
class EndToEndEvaluation:
    metrics: EndToEndMetrics
    per_family: dict[str, PerFamilyMetrics]


def evaluate_end_to_end(
    cases: list[ProjectCase],
    candidates_by_case: Mapping[str, list[Candidate]],
    router: AdaptiveExpertRouter,
    *,
    gate_threshold: float | None = None,
    selection_mode: str = "adaptive",
    fallback_enabled: bool = False,
    fallback_mode: str = "score_fusion",
) -> EndToEndEvaluation:
    if selection_mode not in {"top1", "top2", "adaptive"}:
        raise ValueError("selection_mode must be top1, top2, or adaptive")
    previous_enabled = router.triggers.enabled
    previous_mode = router.triggers.mode
    router.triggers.enabled = fallback_enabled
    router.triggers.mode = fallback_mode

    family_counts = {
        family: {
            "truth": 0,
            "pre": 0,
            "post": 0,
            "top1": 0,
            "adaptive": 0,
            "fallback": 0,
            "success": 0,
        }
        for family in ExpertFamily
    }
    truth_count = pre_hits = post_hits = successes = 0
    candidate_before = candidate_after = expert_calls = 0
    routing_elapsed = 0.0
    try:
        for case in cases:
            before = candidates_by_case.get(case.case_id, [])
            after = (
                before
                if gate_threshold is None
                else [
                    candidate
                    for candidate in before
                    if candidate.suspicion_score >= gate_threshold
                ]
            )
            candidate_before += len(before)
            candidate_after += len(after)
            started = perf_counter()
            decisions = [router.route(candidate) for candidate in after]
            routing_elapsed += perf_counter() - started
            expert_calls += sum(
                len(_selected(decision, selection_mode)) for decision in decisions
            )

            for truth in case.ground_truth:
                truth_count += 1
                pre_matches = [
                    candidate
                    for candidate in before
                    if candidate_matches_truth(candidate, truth)
                ]
                post_pairs = [
                    (candidate, decision)
                    for candidate, decision in zip(after, decisions, strict=True)
                    if candidate_matches_truth(candidate, truth)
                ]
                pre_hit = bool(pre_matches)
                post_hit = bool(post_pairs)
                top1_hit = any(
                    bool(_rank(decision)) and _rank(decision)[0] in truth.experts
                    for _, decision in post_pairs
                )
                adaptive_hit = any(
                    any(
                        family in _selected(decision, selection_mode)
                        for family in truth.experts
                    )
                    for _, decision in post_pairs
                )
                fallback_hit = any(
                    any(family in decision.trigger_scores for family in truth.experts)
                    for _, decision in post_pairs
                )
                pre_hits += int(pre_hit)
                post_hits += int(post_hit)
                successes += int(adaptive_hit)

                for family in truth.experts:
                    counts = family_counts[family]
                    counts["truth"] += 1
                    counts["pre"] += int(pre_hit)
                    counts["post"] += int(post_hit)
                    counts["top1"] += int(
                        any(
                            bool(_rank(decision)) and _rank(decision)[0] == family
                            for _, decision in post_pairs
                        )
                    )
                    counts["adaptive"] += int(
                        any(
                            family in _selected(decision, selection_mode)
                            for _, decision in post_pairs
                        )
                    )
                    counts["fallback"] += int(
                        any(family in decision.trigger_scores for _, decision in post_pairs)
                    )
                    counts["success"] += int(
                        any(
                            family in _selected(decision, selection_mode)
                            for _, decision in post_pairs
                        )
                    )
    finally:
        router.triggers.enabled = previous_enabled
        router.triggers.mode = previous_mode

    case_count = len(cases)
    all_six_calls = len(ExpertFamily) * candidate_after
    metrics = EndToEndMetrics(
        ground_truth_count=truth_count,
        pre_gate_candidate_hit_count=pre_hits,
        post_gate_candidate_hit_count=post_hits,
        successful_ground_truth_count=successes,
        pre_gate_candidate_recall=_ratio(pre_hits, truth_count),
        gate_retention=_ratio(post_hits, pre_hits) if pre_hits else 1.0,
        post_gate_candidate_recall=_ratio(post_hits, truth_count),
        conditional_router_coverage=_ratio(successes, post_hits),
        end_to_end_routing_recall=_ratio(successes, truth_count),
        candidate_count_before=candidate_before,
        candidate_count_after=candidate_after,
        average_expert_calls_per_case=_ratio(expert_calls, case_count),
        llm_calls_saved_vs_all_six=all_six_calls - expert_calls,
        routing_latency_ms_per_case=(routing_elapsed * 1_000.0 / case_count)
        if case_count
        else 0.0,
    )
    supported = set(router.available_families)
    per_family = {
        family.value: PerFamilyMetrics(
            learned_supported=family in supported,
            ground_truth_count=counts["truth"],
            candidate_recall=_ratio(counts["pre"], counts["truth"]),
            gate_retention=_ratio(counts["post"], counts["pre"])
            if counts["pre"]
            else 1.0,
            top1_accuracy=_ratio(counts["top1"], counts["post"]),
            adaptive_coverage=_ratio(counts["adaptive"], counts["post"]),
            fallback_activation=_ratio(counts["fallback"], counts["post"]),
            end_to_end_routing_recall=_ratio(counts["success"], counts["truth"]),
        )
        for family, counts in family_counts.items()
    }
    return EndToEndEvaluation(metrics=metrics, per_family=per_family)


def _selected(decision: RouteDecision, mode: str) -> list[ExpertFamily]:
    ranked = _rank(decision)
    if mode == "top1":
        return ranked[:1]
    if mode == "top2":
        return ranked[:2]
    return decision.selected


def _rank(decision: RouteDecision) -> list[ExpertFamily]:
    return sorted(
        decision.scores,
        key=lambda family: (-decision.scores[family], family.value),
    )


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0
