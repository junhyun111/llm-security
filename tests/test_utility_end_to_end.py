from __future__ import annotations

from llm_security.datasets import UTILITY_OUTCOME_LABEL_VERSION, UtilitySample
from llm_security.experiments import evaluate_utility_end_to_end
from llm_security.models import (
    ACTIVE_UTILITY_EXPERTS,
    Candidate,
    Evidence,
    ExpertAssignment,
    ExpertFamily,
    GroundTruth,
    ProjectCase,
)
from llm_security.routing import BudgetedUtilityRouter, CandidateGate, UtilityPolicyConfig


def _candidate(candidate_id: str, line: int, score: float) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        project_id="project",
        file="a.c",
        function="f",
        line_start=line,
        line_end=line,
        code="void f(void) {}",
        evidence=[Evidence("EV1", "memory_sink", "a.c", line, "p[i]", "f")],
        features={"memory_sink_count": 1.0},
        suspicion_score=score,
        feature_schema_version="semantic-v1",
    )


class _FixedAnalyzer:
    def __init__(self, candidates: list[Candidate]) -> None:
        self.candidates = candidates

    def analyze(self, case: ProjectCase) -> list[Candidate]:
        return list(self.candidates)


def _matrix(candidate: Candidate) -> list[UtilitySample]:
    rows = []
    for family in ACTIVE_UTILITY_EXPERTS:
        success = family == ExpertFamily.MEMORY_SAFETY
        rows.append(
            UtilitySample(
                candidate=candidate,
                assignment=ExpertAssignment(family, f"model/{family.value}"),
                success=success,
                false_positive=False,
                cost=0.01,
                matched_truth_ids=["truth-1"] if success else [],
                ground_truth_ids=["truth-1"],
                truth_labels_available=True,
                case_id="case-1",
                label_version=UTILITY_OUTCOME_LABEL_VERSION,
            )
        )
    return rows


def test_utility_end_to_end_replays_all_stages_without_llm_calls() -> None:
    kept = _candidate("kept", 4, 0.9)
    gated = _candidate("gated", 20, 0.1)
    rows = _matrix(kept)
    router = BudgetedUtilityRouter.fit(
        rows,
        policy=UtilityPolicyConfig(escalation_threshold=0.0),
    )
    case = ProjectCase(
        case_id="case-1",
        project_id="project",
        source_files={"a.c": "\n" * 30},
        split="test",
        ground_truth=[
            GroundTruth(
                "truth-1",
                "a.c",
                "f",
                4,
                4,
                [ExpertFamily.MEMORY_SAFETY],
                ["CWE-787"],
            ),
            GroundTruth(
                "truth-2",
                "a.c",
                "f",
                20,
                20,
                [ExpertFamily.MEMORY_SAFETY],
                ["CWE-787"],
            ),
        ],
    )

    metrics = evaluate_utility_end_to_end(
        [case],
        rows,
        analyzer=_FixedAnalyzer([kept, gated]),
        candidate_gate=CandidateGate(enabled=True, threshold=0.5),
        router=router,
    )

    assert metrics.analyzer_candidate_recall == 1.0
    assert metrics.candidate_gate_gt_retention == 0.5
    assert metrics.outcome_matrix_gt_coverage == 0.5
    assert metrics.full5_oracle_detection_recall == 0.5
    assert metrics.routed_detection_recall == 0.5
    assert metrics.logical_expert_tasks == 2
    assert metrics.research_physical_requests == 2
    assert metrics.web_batched_requests == 1


def test_best_single_and_fixed2_are_frozen_on_calibration_rows() -> None:
    candidate = _candidate("candidate", 4, 0.9)
    rows = _matrix(candidate)
    router = BudgetedUtilityRouter.fit(rows)

    selected = router.calibrate_baselines(rows)
    report = router.evaluate_baselines(rows)

    assert router.assignments[selected.best_single_assignment_id].expert == (
        ExpertFamily.MEMORY_SAFETY
    )
    assert "best_single" in report
    assert "best_fixed2" in report
    assert report["best_single"].truth_recall == 1.0
