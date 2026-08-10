from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..models import Candidate, ProjectCase
from .analyzer_eval import analyzer_metrics


@dataclass(slots=True)
class GateCalibrationResult:
    threshold: float
    pre_gate_candidate_recall: float
    gate_retention: float
    post_gate_candidate_recall: float
    candidate_count_before: int
    candidate_count_after: int
    reduction_rate: float
    target_retention: float
    target_met: bool


def apply_gate(
    candidates_by_case: Mapping[str, list[Candidate]], threshold: float
) -> dict[str, list[Candidate]]:
    return {
        case_id: [
            candidate
            for candidate in candidates
            if candidate.suspicion_score >= threshold
        ]
        for case_id, candidates in candidates_by_case.items()
    }


def calibrate_gate(
    dev_cases: list[ProjectCase],
    dev_candidates_by_case: Mapping[str, list[Candidate]],
    *,
    target_retention: float = 0.98,
) -> GateCalibrationResult:
    """Calibrate only from explicitly supplied development data."""
    if not 0.0 <= target_retention <= 1.0:
        raise ValueError("target_retention must be between 0 and 1")
    scores = sorted(
        {
            float(candidate.suspicion_score)
            for candidates in dev_candidates_by_case.values()
            for candidate in candidates
        }
    )
    if any(score < 0.0 or score > 1.0 for score in scores):
        raise ValueError(
            "Gate calibration requires normalized suspicion scores between 0 and 1"
        )
    # A value below the minimum is the exact Gate-OFF behavior.  Unique observed
    # scores are sufficient because acceptance changes only at those points.
    thresholds = [0.0, *scores, 1.0]
    results = [
        evaluate_gate(
            dev_cases,
            dev_candidates_by_case,
            threshold=threshold,
            target_retention=target_retention,
        )
        for threshold in sorted(set(thresholds))
    ]
    feasible = [result for result in results if result.target_met]
    if feasible:
        return min(
            feasible,
            key=lambda result: (
                result.candidate_count_after,
                -result.threshold,
            ),
        )
    return max(
        results,
        key=lambda result: (
            result.gate_retention,
            -result.candidate_count_after,
            result.threshold,
        ),
    )


def evaluate_gate(
    cases: list[ProjectCase],
    candidates_by_case: Mapping[str, list[Candidate]],
    *,
    threshold: float,
    target_retention: float = 0.98,
) -> GateCalibrationResult:
    before = analyzer_metrics(cases, candidates_by_case)
    accepted = apply_gate(candidates_by_case, threshold)
    after = analyzer_metrics(cases, accepted)
    retention = (
        after.candidate_hit_count / before.candidate_hit_count
        if before.candidate_hit_count
        else 1.0
    )
    reduction = (
        1.0 - after.candidate_count / before.candidate_count
        if before.candidate_count
        else 0.0
    )
    return GateCalibrationResult(
        threshold=float(threshold),
        pre_gate_candidate_recall=before.candidate_recall,
        gate_retention=retention,
        post_gate_candidate_recall=after.candidate_recall,
        candidate_count_before=before.candidate_count,
        candidate_count_after=after.candidate_count,
        reduction_rate=reduction,
        target_retention=target_retention,
        target_met=retention >= target_retention,
    )
