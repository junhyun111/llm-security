from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

from ..datasets import UtilitySample
from ..models import Candidate, GroundTruth, ProjectCase
from ..routing import BudgetedUtilityRouter, CandidateGate
from .analyzer_eval import candidate_matches_truth


class CandidateAnalyzer(Protocol):
    def analyze(self, case: ProjectCase) -> list[Candidate]: ...


@dataclass(slots=True)
class UtilityEndToEndMetrics:
    """Stage-wise recall and request accounting for the measured-outcome pipeline."""

    case_count: int
    analysis_error_count: int
    ground_truth_count: int
    analyzer_candidate_count: int
    gated_candidate_count: int
    outcome_matrix_candidate_count: int
    analyzer_truth_hits: int
    gated_truth_hits: int
    outcome_matrix_truth_hits: int
    full5_oracle_truth_hits: int
    routed_truth_hits: int
    analyzer_candidate_recall: float
    candidate_gate_gt_retention: float
    post_gate_candidate_recall: float
    outcome_matrix_gt_coverage: float
    full5_oracle_detection_recall: float
    routed_detection_recall: float
    routing_retention_vs_full5: float
    validated_outcome_precision: float
    end_to_end_f1: float
    exact_case_coverage: float
    average_assignments_per_routed_candidate: float
    logical_expert_tasks: int
    research_physical_requests: int
    web_batched_requests: int
    prompt_tokens: int
    completion_tokens: int
    realized_cost: float


def evaluate_utility_end_to_end(
    cases: Iterable[ProjectCase],
    outcomes: Iterable[UtilitySample],
    *,
    analyzer: CandidateAnalyzer,
    candidate_gate: CandidateGate,
    router: BudgetedUtilityRouter,
    max_candidates_per_case: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> UtilityEndToEndMetrics:
    """Replay a frozen outcome matrix through every stage without LLM calls.

    The matrix contains post-validator outcomes collected earlier. Missing matrix
    candidates are reported as coverage loss instead of being silently treated as
    negative Expert results.
    """
    rows_by_candidate: dict[tuple[str, str], list[UtilitySample]] = defaultdict(list)
    for sample in outcomes:
        if not sample.case_id:
            raise ValueError("End-to-end evaluation requires outcome rows with case_id")
        rows_by_candidate[(sample.case_id, sample.candidate.candidate_id)].append(sample)

    case_count = analysis_errors = 0
    analyzer_candidates = gated_candidates = matrix_candidates = 0
    total_truths = analyzer_hits = gated_hits = matrix_hits = 0
    oracle_hits = routed_hits = exact_cases = 0
    logical_tasks = prompt_tokens = completion_tokens = 0
    true_outcomes = false_outcomes = 0
    realized_cost = 0.0
    web_cases: set[str] = set()

    for case in cases:
        case_count += 1
        truth_keys = {(case.case_id, truth.truth_id) for truth in case.ground_truth}
        total_truths += len(truth_keys)
        try:
            before = analyzer.analyze(case)
        except (RuntimeError, ValueError, TimeoutError) as error:
            analysis_errors += 1
            if progress:
                progress(f"skip {case.case_id}: analyzer failed: {error}")
            continue
        analyzer_candidates += len(before)
        after, _ = candidate_gate.filter(before)
        after.sort(
            key=lambda candidate: (
                -candidate.suspicion_score,
                candidate.file,
                candidate.line_start,
                candidate.candidate_id,
            )
        )
        if max_candidates_per_case is not None:
            after = after[:max_candidates_per_case]
        gated_candidates += len(after)

        analyzer_truths = _truths_hit(case, before)
        gated_truths = _truths_hit(case, after)
        analyzer_hits += len(analyzer_truths)
        gated_hits += len(gated_truths)

        case_matrix_truths: set[tuple[str, str]] = set()
        case_oracle_truths: set[tuple[str, str]] = set()
        case_routed_truths: set[tuple[str, str]] = set()
        for candidate in after:
            rows = rows_by_candidate.get((case.case_id, candidate.candidate_id), [])
            if not rows:
                continue
            matrix_candidates += 1
            case_matrix_truths.update(
                (case.case_id, truth_id)
                for row in rows
                for truth_id in row.ground_truth_ids
            )
            case_oracle_truths.update(
                (case.case_id, truth_id)
                for row in rows
                for truth_id in row.matched_truth_ids
            )
            decision = router.route(candidate)
            selected_ids = [item.assignment_id for item in decision.assignments]
            by_assignment = {
                row.assignment.assignment_id: row for row in rows
            }
            missing = [item for item in selected_ids if item not in by_assignment]
            if missing:
                raise ValueError(
                    f"Outcome matrix for {case.case_id}/{candidate.candidate_id} "
                    "is missing routed assignments: " + ", ".join(missing)
                )
            selected = [by_assignment[item] for item in selected_ids]
            if selected:
                web_cases.add(case.case_id)
            logical_tasks += len(selected)
            prompt_tokens += sum(item.prompt_tokens for item in selected)
            completion_tokens += sum(item.completion_tokens for item in selected)
            realized_cost += sum(item.cost for item in selected)
            true_outcomes += sum(_validated_true_count(item) for item in selected)
            false_outcomes += sum(_validated_false_count(item) for item in selected)
            case_routed_truths.update(
                (case.case_id, truth_id)
                for row in selected
                for truth_id in row.matched_truth_ids
            )

        matrix_truths = case_matrix_truths & truth_keys
        oracle_truths = case_oracle_truths & truth_keys
        selected_truths = case_routed_truths & truth_keys
        matrix_hits += len(matrix_truths)
        oracle_hits += len(oracle_truths)
        routed_hits += len(selected_truths)
        exact_cases += int(truth_keys <= selected_truths)
        if progress:
            progress(f"end-to-end: {case_count} cases")

    precision_denominator = true_outcomes + false_outcomes
    precision = _ratio(true_outcomes, precision_denominator, empty=1.0)
    recall = _ratio(routed_hits, total_truths, empty=1.0)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return UtilityEndToEndMetrics(
        case_count=case_count,
        analysis_error_count=analysis_errors,
        ground_truth_count=total_truths,
        analyzer_candidate_count=analyzer_candidates,
        gated_candidate_count=gated_candidates,
        outcome_matrix_candidate_count=matrix_candidates,
        analyzer_truth_hits=analyzer_hits,
        gated_truth_hits=gated_hits,
        outcome_matrix_truth_hits=matrix_hits,
        full5_oracle_truth_hits=oracle_hits,
        routed_truth_hits=routed_hits,
        analyzer_candidate_recall=_ratio(analyzer_hits, total_truths, empty=1.0),
        candidate_gate_gt_retention=_ratio(gated_hits, analyzer_hits, empty=1.0),
        post_gate_candidate_recall=_ratio(gated_hits, total_truths, empty=1.0),
        outcome_matrix_gt_coverage=_ratio(matrix_hits, total_truths, empty=1.0),
        full5_oracle_detection_recall=_ratio(oracle_hits, total_truths, empty=1.0),
        routed_detection_recall=recall,
        routing_retention_vs_full5=_ratio(routed_hits, oracle_hits, empty=1.0),
        validated_outcome_precision=precision,
        end_to_end_f1=f1,
        exact_case_coverage=_ratio(exact_cases, case_count, empty=1.0),
        average_assignments_per_routed_candidate=_ratio(
            logical_tasks, matrix_candidates
        ),
        logical_expert_tasks=logical_tasks,
        # Outcome research currently sends one request per Expert assignment.
        research_physical_requests=logical_tasks,
        # The web runner batches all logical Expert tasks for one upload/case.
        web_batched_requests=len(web_cases),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        realized_cost=realized_cost,
    )


def _truths_hit(
    case: ProjectCase, candidates: list[Candidate]
) -> set[tuple[str, str]]:
    return {
        (case.case_id, truth.truth_id)
        for truth in case.ground_truth
        if any(candidate_matches_truth(candidate, truth) for candidate in candidates)
    }


def _validated_true_count(sample: UtilitySample) -> int:
    if sample.validated_true_findings or sample.validated_false_findings:
        return sample.validated_true_findings
    return int(sample.success)


def _validated_false_count(sample: UtilitySample) -> int:
    if sample.validated_true_findings or sample.validated_false_findings:
        return sample.validated_false_findings
    return int(sample.false_positive)


def _ratio(
    numerator: int | float, denominator: int | float, *, empty: float = 0.0
) -> float:
    return float(numerator / denominator) if denominator else empty
