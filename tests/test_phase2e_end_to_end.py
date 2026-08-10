from __future__ import annotations

from types import SimpleNamespace

from llm_security.experiments.end_to_end import evaluate_end_to_end
from llm_security.models import (
    Candidate,
    ExpertFamily,
    GroundTruth,
    ProjectCase,
    RouteDecision,
)


class FixedRouter:
    available_families = (ExpertFamily.MEMORY_BOUNDS,)

    def __init__(self) -> None:
        self.triggers = SimpleNamespace(enabled=False, mode="score_fusion")

    def route(self, candidate: Candidate) -> RouteDecision:
        return RouteDecision(
            candidate_id=candidate.candidate_id,
            scores={ExpertFamily.MEMORY_BOUNDS: 1.0},
            selected=[ExpertFamily.MEMORY_BOUNDS],
            top1_confidence=1.0,
            top1_top2_margin=1.0,
            policy="fixed",
            reasons=[],
        )


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
        features={"x": 1.0},
        suspicion_score=score,
    )


def test_end_to_end_metrics_decompose_analyzer_gate_and_router_losses():
    case = ProjectCase(
        case_id="case",
        project_id="p",
        source_files={"a.c": "x\n" * 30},
        ground_truth=[
            GroundTruth("t1", "a.c", "f", 1, 1, [ExpertFamily.MEMORY_BOUNDS]),
            GroundTruth("t2", "a.c", "f", 10, 10, [ExpertFamily.MEMORY_BOUNDS]),
            GroundTruth("t3", "a.c", "f", 20, 20, [ExpertFamily.MEMORY_BOUNDS]),
        ],
    )
    evaluation = evaluate_end_to_end(
        [case],
        {"case": [_candidate("kept", 1, 0.9), _candidate("gated", 10, 0.1)]},
        FixedRouter(),  # type: ignore[arg-type]
        gate_threshold=0.5,
    )
    metrics = evaluation.metrics
    assert metrics.pre_gate_candidate_recall == 2 / 3
    assert metrics.gate_retention == 0.5
    assert metrics.post_gate_candidate_recall == 1 / 3
    assert metrics.conditional_router_coverage == 1.0
    assert metrics.end_to_end_routing_recall == 1 / 3
