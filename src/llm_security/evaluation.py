from __future__ import annotations

from dataclasses import dataclass

from .models import Finding, GroundTruth, PipelineResult, ProjectCase


@dataclass(slots=True)
class CaseMetrics:
    case_id: str
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    candidate_recall: float
    route_coverage: float
    cost: float


@dataclass(slots=True)
class AggregateMetrics:
    case_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    candidate_recall: float
    route_coverage: float
    total_cost: float
    total_requests: int


def evaluate_case(case: ProjectCase, result: PipelineResult) -> CaseMetrics:
    findings = result.validated_findings
    matched_truth: set[str] = set()
    matched_findings: set[str] = set()
    for finding in findings:
        for truth in case.ground_truth:
            if truth.truth_id in matched_truth:
                continue
            if _finding_matches_truth(finding, truth):
                matched_truth.add(truth.truth_id)
                matched_findings.add(finding.finding_id)
                break
    true_positives = len(matched_truth)
    false_positives = len(findings) - len(matched_findings)
    false_negatives = len(case.ground_truth) - true_positives
    precision = _safe_ratio(true_positives, true_positives + false_positives, empty=1.0)
    recall = _safe_ratio(true_positives, true_positives + false_negatives, empty=1.0)
    f1 = _f1(precision, recall)

    candidate_hits = sum(
        int(any(_candidate_matches_truth(candidate, truth) for candidate in result.candidates))
        for truth in case.ground_truth
    )
    candidate_recall = _safe_ratio(candidate_hits, len(case.ground_truth), empty=1.0)
    route_hits = 0
    for truth in case.ground_truth:
        for candidate, route in zip(result.candidates, result.routes, strict=True):
            if _candidate_matches_truth(candidate, truth) and set(route.selected).intersection(
                truth.experts
            ):
                route_hits += 1
                break
    route_coverage = _safe_ratio(route_hits, len(case.ground_truth), empty=1.0)
    return CaseMetrics(
        case_id=case.case_id,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        candidate_recall=candidate_recall,
        route_coverage=route_coverage,
        cost=sum(item.cost for item in result.usage),
    )


def aggregate_metrics(metrics: list[CaseMetrics], results: list[PipelineResult]) -> AggregateMetrics:
    tp = sum(item.true_positives for item in metrics)
    fp = sum(item.false_positives for item in metrics)
    fn = sum(item.false_negatives for item in metrics)
    precision = _safe_ratio(tp, tp + fp, empty=1.0)
    recall = _safe_ratio(tp, tp + fn, empty=1.0)
    return AggregateMetrics(
        case_count=len(metrics),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=_f1(precision, recall),
        candidate_recall=(
            sum(item.candidate_recall for item in metrics) / len(metrics) if metrics else 0.0
        ),
        route_coverage=(
            sum(item.route_coverage for item in metrics) / len(metrics) if metrics else 0.0
        ),
        total_cost=sum(item.cost for item in metrics),
        total_requests=sum(len(result.usage) for result in results),
    )


def _finding_matches_truth(finding: Finding, truth: GroundTruth) -> bool:
    return (
        finding.file == truth.file
        and finding.line_start <= truth.line_end
        and finding.line_end >= truth.line_start
        and finding.expert in truth.experts
    )


def _candidate_matches_truth(candidate, truth: GroundTruth) -> bool:
    return (
        candidate.file == truth.file
        and candidate.line_start <= truth.line_end
        and candidate.line_end >= truth.line_start
    )


def _safe_ratio(numerator: int, denominator: int, *, empty: float) -> float:
    return numerator / denominator if denominator else empty


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
