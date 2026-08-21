from __future__ import annotations

import json

import pytest

from model_evaluation.api_budget import ApiBudget, BudgetExceeded
from model_evaluation.candidates import StreamingCachedCandidateAnalyzer
from model_evaluation.paths import EVALUATION_ROOT
from model_evaluation.workflow import audit_outcome_matrix


class _Case:
    def __init__(self, case_id: str) -> None:
        self.case_id = case_id


def test_streaming_candidate_cache_handles_zero_candidate_case(tmp_path) -> None:
    # Candidate fields are intentionally minimal but valid for parent restoration.
    row = {
        "case_id": "case-b",
        "candidate": {
            "candidate_id": "candidate-b",
            "project_id": "project-b",
            "file": "unit.c",
            "function": "f",
            "line_start": 1,
            "line_end": 2,
            "code": "int f(void) { return 0; }",
            "evidence": [],
            "features": {"bias": 1.0},
            "suspicion_score": 0.1,
            "feature_schema_version": "semantic-cwe-v2",
            "cwe_hypotheses": [],
        },
    }
    path = tmp_path / "candidates.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    analyzer = StreamingCachedCandidateAnalyzer(path)
    assert analyzer.analyze(_Case("case-a")) == []
    assert [item.candidate_id for item in analyzer.analyze(_Case("case-b"))] == [
        "candidate-b"
    ]


def test_budget_stops_before_request_or_reserved_overspend() -> None:
    budget = ApiBudget(max_requests=1, max_usd=0.05, reserve_usd_per_request=0.10)
    budget.validate()
    with pytest.raises(BudgetExceeded):
        budget.check()
    budget = ApiBudget(max_requests=1, max_usd=1.0, reserve_usd_per_request=0.10)
    budget.validate()
    budget.requests = 1
    with pytest.raises(BudgetExceeded):
        budget.check()


def test_matrix_audit_detects_wholly_missing_candidate_group(tmp_path) -> None:
    # Output paths are required to stay in Model_Evaluation, so use its cache area.
    work = EVALUATION_ROOT / "cache" / "workflow-test"
    work.mkdir(parents=True, exist_ok=True)
    selection = work / "selection.jsonl"
    selection.write_text(
        json.dumps({"case_id": "case-a", "candidate_id": "candidate-a"})
        + "\n"
        + json.dumps({"case_id": "case-b", "candidate_id": "candidate-b"})
        + "\n",
        encoding="utf-8",
    )
    outcome = work / "outcome.jsonl"
    candidate = {
        "candidate_id": "candidate-a",
        "project_id": "project-a",
        "file": "unit.c",
        "function": "f",
        "line_start": 1,
        "line_end": 2,
        "features": {"bias": 1.0},
        "suspicion_score": 0.1,
        "feature_schema_version": "semantic-cwe-v2",
        "cwe_hypotheses": [],
    }
    assignment = {
        "expert": "memory_bounds",
        "model_id": "example/model",
        "prompt_version": "expert-v4-cwe-hypothesis",
        "expected_cost": 0.0,
    }
    assignment_id = "memory_bounds::example/model::expert-v4-cwe-hypothesis"
    outcome.write_text(
        json.dumps(
            {
                "candidate": candidate,
                "assignment": assignment,
                "success": False,
                "truth_labels_available": True,
                "case_id": "case-a",
                "label_version": "semantic-causal-v1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    audit = audit_outcome_matrix(
        outcome,
        expected_assignment_ids=[assignment_id],
        selection_manifest=selection,
    )
    assert not audit["complete"]
    assert audit["missing_candidate_group_count"] == 1
