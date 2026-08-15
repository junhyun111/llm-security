from __future__ import annotations

import gc
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

from ..analysis.protocols import CandidateAnalyzer
from ..datasets import RouterSample, iter_cases_jsonl
from ..models import Candidate, ProjectCase
from .analyzer_eval import AnalyzerMetrics, candidate_matches_truth
from .dataset import router_samples_from_cached_candidates, single_label_samples


@dataclass(slots=True)
class StreamingAnalysisOutput:
    metrics: AnalyzerMetrics
    router_samples: list[RouterSample]
    retained_cases: list[ProjectCase]
    retained_candidates_by_case: dict[str, list[Candidate]]


def analyze_split_jsonl(
    analyzer: CandidateAnalyzer,
    cases_path: str | Path,
    *,
    retain_compact_analysis: bool,
    expected_case_count: int,
    progress: Callable[[int, int], None] | None = None,
    case_start_progress: Callable[[int, int, ProjectCase], None] | None = None,
    progress_every: int = 50,
    failure_log_path: str | Path | None = None,
) -> StreamingAnalysisOutput:
    """Analyze one case at a time and retain no source or candidate code."""
    case_count = 0
    truth_count = 0
    hit_count = 0
    candidate_count = 0
    elapsed_seconds = 0.0
    analysis_error_count = 0
    candidate_locs: list[int] = []
    samples: list[RouterSample] = []
    retained_cases: list[ProjectCase] = []
    retained_candidates: dict[str, list[Candidate]] = {}

    failure_handle = None
    if failure_log_path is not None:
        failure_path = Path(failure_log_path)
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_handle = failure_path.open("w", encoding="utf-8", newline="\n")
    try:
        for case_count, case in enumerate(iter_cases_jsonl(cases_path), start=1):
            if case_start_progress is not None:
                case_start_progress(case_count, expected_case_count, case)
            started = perf_counter()
            try:
                candidates = analyzer.analyze(case)
            except Exception as error:  # keep benchmark misses instead of aborting a long run
                candidates = []
                analysis_error_count += 1
                if failure_handle is not None:
                    failure_handle.write(
                        json.dumps(
                            {
                                "case_id": case.case_id,
                                "project_id": case.project_id,
                                "error_type": type(error).__name__,
                                "message": str(error),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    failure_handle.flush()
            elapsed_seconds += perf_counter() - started
            truth_count += len(case.ground_truth)
            hit_count += sum(
                any(candidate_matches_truth(candidate, truth) for candidate in candidates)
                for truth in case.ground_truth
            )
            candidate_count += len(candidates)
            candidate_locs.extend(
                max(1, candidate.line_end - candidate.line_start + 1)
                for candidate in candidates
            )

            current_samples = single_label_samples(
                router_samples_from_cached_candidates(
                    [case], {case.case_id: candidates}
                )
            )
            samples.extend(
                RouterSample(
                    candidate=_compact_candidate(sample.candidate),
                    labels=list(sample.labels),
                )
                for sample in current_samples
            )
            if retain_compact_analysis:
                retained_cases.append(_compact_case(case))
                retained_candidates[case.case_id] = [
                    _compact_candidate(candidate) for candidate in candidates
                ]

            if progress is not None and (
                case_count == expected_case_count
                or (progress_every > 0 and case_count % progress_every == 0)
            ):
                progress(case_count, expected_case_count)
            del candidates, current_samples, case
            if case_count % 100 == 0:
                gc.collect()
    finally:
        if failure_handle is not None:
            failure_handle.close()

    if case_count != expected_case_count:
        raise ValueError(
            f"Expected {expected_case_count} cases in {cases_path}, found {case_count}"
        )
    return StreamingAnalysisOutput(
        metrics=AnalyzerMetrics(
            ground_truth_count=truth_count,
            candidate_hit_count=hit_count,
            candidate_recall=_ratio(hit_count, truth_count),
            case_count=case_count,
            candidate_count=candidate_count,
            average_candidates_per_case=_ratio(candidate_count, case_count),
            average_candidates_per_ground_truth=_ratio(candidate_count, truth_count),
            mean_candidate_loc=(
                float(statistics.fmean(candidate_locs)) if candidate_locs else 0.0
            ),
            median_candidate_loc=(
                float(statistics.median(candidate_locs)) if candidate_locs else 0.0
            ),
            analysis_latency_ms_per_case=(
                elapsed_seconds * 1_000.0 / case_count if case_count else 0.0
            ),
            analysis_error_count=analysis_error_count,
        ),
        router_samples=samples,
        retained_cases=retained_cases,
        retained_candidates_by_case=retained_candidates,
    )


def _compact_candidate(candidate: Candidate) -> Candidate:
    return Candidate(
        candidate_id=candidate.candidate_id,
        project_id=candidate.project_id,
        file=candidate.file,
        function=candidate.function,
        line_start=candidate.line_start,
        line_end=candidate.line_end,
        code="",
        evidence=[],
        features=dict(candidate.features),
        suspicion_score=candidate.suspicion_score,
        callers=list(candidate.callers),
        callees=list(candidate.callees),
        feature_schema_version=candidate.feature_schema_version,
    )


def _compact_case(case: ProjectCase) -> ProjectCase:
    return ProjectCase(
        case_id=case.case_id,
        project_id=case.project_id,
        source_files={},
        split=case.split,
        vulnerable_revision=case.vulnerable_revision,
        fixed_revision=case.fixed_revision,
        ground_truth=list(case.ground_truth),
        metadata={},
    )


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0
