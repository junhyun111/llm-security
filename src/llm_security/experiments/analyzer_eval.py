from __future__ import annotations

import statistics
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Mapping

from ..analysis.protocols import CandidateAnalyzer
from ..models import Candidate, GroundTruth, ProjectCase


@dataclass(slots=True)
class AnalyzerMetrics:
    ground_truth_count: int
    candidate_hit_count: int
    candidate_recall: float
    case_count: int
    candidate_count: int
    average_candidates_per_case: float
    average_candidates_per_ground_truth: float
    mean_candidate_loc: float
    median_candidate_loc: float
    analysis_latency_ms_per_case: float
    analysis_error_count: int = 0


@dataclass(slots=True)
class AnalyzerEvaluation:
    metrics: AnalyzerMetrics
    candidates_by_case: dict[str, list[Candidate]]


def candidate_matches_truth(candidate: Candidate, truth: GroundTruth) -> bool:
    return (
        candidate.file == truth.file
        and candidate.line_start <= truth.line_end
        and candidate.line_end >= truth.line_start
    )


def evaluate_analyzer(
    analyzer: CandidateAnalyzer,
    cases: list[ProjectCase],
    *,
    progress: Callable[[int, int], None] | None = None,
    progress_every: int = 50,
) -> AnalyzerEvaluation:
    candidates_by_case: dict[str, list[Candidate]] = {}
    elapsed = 0.0
    total = len(cases)
    for index, case in enumerate(cases, start=1):
        started = perf_counter()
        candidates = analyzer.analyze(case)
        elapsed += perf_counter() - started
        candidates_by_case[case.case_id] = sorted(
            candidates,
            key=lambda item: (-item.suspicion_score, item.file, item.line_start, item.candidate_id),
        )
        if progress is not None and (
            index == total or (progress_every > 0 and index % progress_every == 0)
        ):
            progress(index, total)
    metrics = analyzer_metrics(cases, candidates_by_case, elapsed_seconds=elapsed)
    return AnalyzerEvaluation(metrics=metrics, candidates_by_case=candidates_by_case)


def analyzer_metrics(
    cases: list[ProjectCase],
    candidates_by_case: Mapping[str, list[Candidate]],
    *,
    elapsed_seconds: float = 0.0,
) -> AnalyzerMetrics:
    truths = [truth for case in cases for truth in case.ground_truth]
    hits = sum(
        any(
            candidate_matches_truth(candidate, truth)
            for candidate in candidates_by_case.get(case.case_id, [])
        )
        for case in cases
        for truth in case.ground_truth
    )
    candidates = [
        candidate
        for case in cases
        for candidate in candidates_by_case.get(case.case_id, [])
    ]
    loc = [max(1, item.line_end - item.line_start + 1) for item in candidates]
    truth_count = len(truths)
    case_count = len(cases)
    return AnalyzerMetrics(
        ground_truth_count=truth_count,
        candidate_hit_count=hits,
        candidate_recall=_ratio(hits, truth_count),
        case_count=case_count,
        candidate_count=len(candidates),
        average_candidates_per_case=_ratio(len(candidates), case_count),
        average_candidates_per_ground_truth=_ratio(len(candidates), truth_count),
        mean_candidate_loc=float(statistics.fmean(loc)) if loc else 0.0,
        median_candidate_loc=float(statistics.median(loc)) if loc else 0.0,
        analysis_latency_ms_per_case=(elapsed_seconds * 1_000.0 / case_count)
        if case_count
        else 0.0,
    )


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0
