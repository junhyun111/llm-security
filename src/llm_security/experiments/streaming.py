from __future__ import annotations

import gc
import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

from ..analysis.protocols import CandidateAnalyzer
from ..datasets import (
    RouterSample,
    candidate_from_dict,
    case_from_dict,
    iter_cases_jsonl,
    router_sample_from_dict,
    router_sample_to_dict,
)
from ..models import Candidate, ProjectCase, to_dict
from .analyzer_eval import AnalyzerMetrics, candidate_matches_truth
from .dataset import router_samples_from_cached_candidates, single_label_samples


@dataclass(slots=True)
class StreamingAnalysisOutput:
    metrics: AnalyzerMetrics
    router_samples: list[RouterSample]
    retained_cases: list[ProjectCase]
    retained_candidates_by_case: dict[str, list[Candidate]]
    resumed_case_count: int = 0


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
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 100,
    resume: bool = True,
    resume_progress: Callable[[int, int], None] | None = None,
    checkpoint_progress: Callable[[int, int], None] | None = None,
) -> StreamingAnalysisOutput:
    """Analyze one split with atomic, resumable checkpoints.

    A checkpoint contains only compact candidates, labels, and aggregate metrics;
    it never stores original source code. Interruptions resume from the most
    recent complete case checkpoint, normally at most ``checkpoint_every - 1``
    cases behind.
    """
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    restored = _restore_checkpoint(
        checkpoint,
        cases_path=cases_path,
        expected_case_count=expected_case_count,
        retain_compact_analysis=retain_compact_analysis,
        resume=resume,
    )
    case_count = restored["completed_cases"]
    truth_count = 0
    hit_count = 0
    candidate_count = 0
    elapsed_seconds = 0.0
    analysis_error_count = 0
    candidate_locs: list[int] = []
    samples: list[RouterSample] = []
    retained_cases: list[ProjectCase] = []
    retained_candidates: dict[str, list[Candidate]] = {}
    records: list[dict] = restored["records"]
    last_case_id = restored["last_case_id"]
    for record in records:
        truth_count += int(record["ground_truth_count"])
        hit_count += int(record["candidate_hit_count"])
        candidate_count += int(record["candidate_count"])
        elapsed_seconds += float(record["elapsed_seconds"])
        analysis_error_count += int(bool(record["analysis_error"]))
        candidate_locs.extend(int(value) for value in record["candidate_locs"])
        samples.extend(
            router_sample_from_dict(item) for item in record["router_samples"]
        )
        if retain_compact_analysis:
            retained_case = record.get("retained_case")
            if retained_case is not None:
                retained_cases.append(case_from_dict(retained_case))
            retained_candidates[str(record["case_id"])] = [
                candidate_from_dict(item)
                for item in record.get("retained_candidates", [])
            ]

    if case_count and resume_progress is not None:
        resume_progress(case_count, expected_case_count)
    if case_count == expected_case_count:
        return _streaming_output(
            truth_count=truth_count,
            hit_count=hit_count,
            case_count=case_count,
            candidate_count=candidate_count,
            candidate_locs=candidate_locs,
            elapsed_seconds=elapsed_seconds,
            analysis_error_count=analysis_error_count,
            samples=samples,
            retained_cases=retained_cases,
            retained_candidates=retained_candidates,
            resumed_case_count=case_count,
        )

    failure_handle = None
    if failure_log_path is not None:
        failure_path = Path(failure_log_path)
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_handle = failure_path.open(
            "a" if case_count and resume else "w",
            encoding="utf-8",
            newline="\n",
        )
    try:
        for case_index, case in enumerate(iter_cases_jsonl(cases_path), start=1):
            if case_index <= case_count:
                if case_index == case_count and last_case_id != case.case_id:
                    raise ValueError(
                        "Checkpoint does not match the frozen case split; "
                        "rerun with --no-resume."
                    )
                continue
            if case_start_progress is not None:
                case_start_progress(case_index, expected_case_count, case)
            started = perf_counter()
            analysis_error = False
            try:
                candidates = analyzer.analyze(case)
            except Exception as error:  # keep benchmark misses instead of aborting a long run
                candidates = []
                analysis_error = True
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
            analysis_elapsed = perf_counter() - started
            elapsed_seconds += analysis_elapsed
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
            compact_samples = [
                RouterSample(
                    candidate=_compact_candidate(sample.candidate),
                    labels=list(sample.labels),
                )
                for sample in current_samples
            ]
            samples.extend(compact_samples)
            if retain_compact_analysis:
                compact_case = _compact_case(case)
                compact_candidates = [
                    _compact_candidate(candidate) for candidate in candidates
                ]
                retained_cases.append(compact_case)
                retained_candidates[case.case_id] = compact_candidates
            else:
                compact_case = None
                compact_candidates = []

            record = {
                "case_id": case.case_id,
                "ground_truth_count": len(case.ground_truth),
                "candidate_hit_count": sum(
                    any(candidate_matches_truth(candidate, truth) for candidate in candidates)
                    for truth in case.ground_truth
                ),
                "candidate_count": len(candidates),
                "candidate_locs": [
                    max(1, candidate.line_end - candidate.line_start + 1)
                    for candidate in candidates
                ],
                "elapsed_seconds": analysis_elapsed,
                "analysis_error": analysis_error,
                "router_samples": [
                    router_sample_to_dict(sample) for sample in compact_samples
                ],
                "retained_case": (
                    to_dict(compact_case) if compact_case is not None else None
                ),
                "retained_candidates": [
                    to_dict(candidate) for candidate in compact_candidates
                ],
            }
            records.append(record)
            case_count = case_index
            last_case_id = case.case_id

            if progress is not None and (
                case_count == expected_case_count
                or (progress_every > 0 and case_count % progress_every == 0)
            ):
                progress(case_count, expected_case_count)
            if checkpoint is not None and (
                case_count % checkpoint_every == 0
                or case_count == expected_case_count
            ):
                _write_checkpoint(
                    checkpoint,
                    records=records,
                    completed_cases=case_count,
                    expected_case_count=expected_case_count,
                    cases_path=cases_path,
                    last_case_id=last_case_id,
                    retain_compact_analysis=retain_compact_analysis,
                )
                if checkpoint_progress is not None:
                    checkpoint_progress(case_count, expected_case_count)
            del candidates, current_samples, compact_samples, case
            if case_count % 100 == 0:
                gc.collect()
    except BaseException:
        if checkpoint is not None and case_count:
            _write_checkpoint(
                checkpoint,
                records=records,
                completed_cases=case_count,
                expected_case_count=expected_case_count,
                cases_path=cases_path,
                last_case_id=last_case_id,
                retain_compact_analysis=retain_compact_analysis,
            )
        raise
    finally:
        if failure_handle is not None:
            failure_handle.close()

    if case_count != expected_case_count:
        raise ValueError(
            f"Expected {expected_case_count} cases in {cases_path}, found {case_count}"
        )
    return _streaming_output(
        truth_count=truth_count,
        hit_count=hit_count,
        case_count=case_count,
        candidate_count=candidate_count,
        candidate_locs=candidate_locs,
        elapsed_seconds=elapsed_seconds,
        analysis_error_count=analysis_error_count,
        samples=samples,
        retained_cases=retained_cases,
        retained_candidates=retained_candidates,
        resumed_case_count=restored["completed_cases"],
    )


def _streaming_output(
    *,
    truth_count: int,
    hit_count: int,
    case_count: int,
    candidate_count: int,
    candidate_locs: list[int],
    elapsed_seconds: float,
    analysis_error_count: int,
    samples: list[RouterSample],
    retained_cases: list[ProjectCase],
    retained_candidates: dict[str, list[Candidate]],
    resumed_case_count: int,
) -> StreamingAnalysisOutput:
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
        resumed_case_count=resumed_case_count,
    )


def _restore_checkpoint(
    checkpoint: Path | None,
    *,
    cases_path: str | Path,
    expected_case_count: int,
    retain_compact_analysis: bool,
    resume: bool,
) -> dict:
    empty = {"records": [], "completed_cases": 0, "last_case_id": ""}
    if checkpoint is None:
        return empty
    if not resume:
        checkpoint.unlink(missing_ok=True)
        return empty
    if not checkpoint.exists():
        return empty
    try:
        raw = json.loads(checkpoint.read_text(encoding="utf-8"))
        completed = int(raw["completed_cases"])
        records = list(raw["records"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Invalid analysis checkpoint {checkpoint}; rerun with --no-resume."
        ) from error
    expected_path = str(Path(cases_path).resolve())
    valid = (
        raw.get("checkpoint_version") == 1
        and raw.get("cases_path") == expected_path
        and raw.get("expected_case_count") == expected_case_count
        and raw.get("retain_compact_analysis") is retain_compact_analysis
        and 0 <= completed <= expected_case_count
        and len(records) == completed
    )
    if not valid:
        raise ValueError(
            f"Checkpoint {checkpoint} is incompatible with this run; rerun with --no-resume."
        )
    return {
        "records": records,
        "completed_cases": completed,
        "last_case_id": str(raw.get("last_case_id", "")),
    }


def _write_checkpoint(
    checkpoint: Path,
    *,
    records: list[dict],
    completed_cases: int,
    expected_case_count: int,
    cases_path: str | Path,
    last_case_id: str,
    retain_compact_analysis: bool,
) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": 1,
        "cases_path": str(Path(cases_path).resolve()),
        "expected_case_count": expected_case_count,
        "completed_cases": completed_cases,
        "last_case_id": last_case_id,
        "retain_compact_analysis": retain_compact_analysis,
        "records": records,
    }
    temporary = checkpoint.with_name(f".{checkpoint.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, checkpoint)


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
        cwe_hypotheses=list(candidate.cwe_hypotheses),
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
