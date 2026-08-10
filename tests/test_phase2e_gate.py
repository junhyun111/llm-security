from __future__ import annotations

from llm_security.experiments.gate_eval import calibrate_gate, evaluate_gate
from llm_security.models import Candidate, ExpertFamily, GroundTruth, ProjectCase


def _candidate(candidate_id: str, line: int, score: float) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        project_id="p",
        file="a.c",
        function="f",
        line_start=line,
        line_end=line,
        code="x",
        evidence=[],
        features={"x": score},
        suspicion_score=score,
    )


def _case(case_id: str, line: int) -> ProjectCase:
    return ProjectCase(
        case_id=case_id,
        project_id=case_id,
        source_files={"a.c": "x\n" * 20},
        ground_truth=[
            GroundTruth(
                truth_id=f"{case_id}-truth",
                file="a.c",
                function="f",
                line_start=line,
                line_end=line,
                experts=[ExpertFamily.MEMORY_BOUNDS],
            )
        ],
    )


def test_gate_calibration_maximizes_reduction_subject_to_retention():
    cases = [_case("a", 5), _case("b", 10)]
    candidates = {
        "a": [_candidate("a-hit", 5, 0.8), _candidate("a-noise", 2, 0.1)],
        "b": [_candidate("b-hit", 10, 0.4), _candidate("b-noise", 3, 0.2)],
    }
    result = calibrate_gate(cases, candidates, target_retention=0.98)
    assert result.threshold == 0.4
    assert result.target_met
    assert result.gate_retention == 1.0
    assert result.candidate_count_after == 2
    assert result.reduction_rate == 0.5


def test_test_evaluation_uses_frozen_dev_threshold():
    dev_cases = [_case("dev", 5)]
    dev_candidates = {
        "dev": [_candidate("hit", 5, 0.6), _candidate("noise", 1, 0.2)]
    }
    calibrated = calibrate_gate(dev_cases, dev_candidates, target_retention=1.0)
    test_cases = [_case("test", 5)]
    test_candidates = {"test": [_candidate("test-hit", 5, 0.1)]}
    evaluated = evaluate_gate(
        test_cases,
        test_candidates,
        threshold=calibrated.threshold,
        target_retention=1.0,
    )
    assert calibrated.threshold == 0.6
    assert evaluated.threshold == calibrated.threshold
    assert evaluated.post_gate_candidate_recall == 0.0
