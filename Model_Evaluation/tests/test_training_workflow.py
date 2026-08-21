from __future__ import annotations

import json
import uuid

import pytest

from model_evaluation.api_budget import ApiBudget, BudgetExceeded
from model_evaluation.candidates import StreamingCachedCandidateAnalyzer
from model_evaluation.paths import EVALUATION_ROOT
from model_evaluation.workflow import audit_outcome_matrix
from model_evaluation.workflow import collect_outcome_matrix
from model_evaluation.workflow import resolve_models
from model_evaluation.adapters.llm_security import activate_parent_package


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


def test_default_evaluation_model_uses_declared_sweep_model(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENROUTER_API_KEY=test-only\n"
        "OPENROUTER_SWEEP_MODELS=deepseek/deepseek-v4-flash-0731\n",
        encoding="utf-8",
    )
    assert resolve_models(env_file) == ["deepseek/deepseek-v4-flash-0731"]
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


def test_batched_collection_uses_one_request_per_case_and_resumes(
    tmp_path, monkeypatch
) -> None:
    activate_parent_package()
    import llm_security.factory as factory
    from llm_security.llm import LLMResponse
    from llm_security.models import ACTIVE_UTILITY_EXPERTS, UsageRecord

    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, *, model, messages, response_schema, metadata=None):
            self.calls += 1
            results = [
                {
                    "task_id": f"T{index:05d}",
                    "candidate_id": "candidate-one",
                    "expert": expert.value,
                    "findings": [],
                }
                for index, expert in enumerate(ACTIVE_UTILITY_EXPERTS, start=1)
            ]
            return LLMResponse(
                data={
                    "reviewed_task_ids": [item["task_id"] for item in results],
                    "expert_results": results,
                },
                usage=UsageRecord(
                    model=model,
                    prompt_tokens=100,
                    completion_tokens=50,
                    cost=0.01,
                    latency_seconds=1.0,
                ),
                raw={},
            )

    fake = FakeClient()
    monkeypatch.setattr(factory, "build_openrouter_client", lambda config: fake)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENROUTER_API_KEY=test-only\n"
        "OPENROUTER_EXPERT_MODEL=example/model\n",
        encoding="utf-8",
    )
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        json.dumps(
            {
                "case_id": "case-one",
                "project_id": "project-one",
                "source_files": {"unit.c": "int f(void) { return 0; }\n"},
                "split": "train",
                "ground_truth": [],
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    candidate_cache = tmp_path / "candidates.jsonl"
    candidate_cache.write_text(
        json.dumps(
            {
                "case_id": "case-one",
                "candidate": {
                    "candidate_id": "candidate-one",
                    "project_id": "project-one",
                    "file": "unit.c",
                    "function": "f",
                    "line_start": 1,
                    "line_end": 1,
                    "code": "int f(void) { return 0; }",
                    "evidence": [],
                    "features": {"bias": 1.0},
                    "suspicion_score": 0.1,
                    "feature_schema_version": "semantic-cwe-v2",
                    "cwe_hypotheses": [],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    run_dir = EVALUATION_ROOT / "cache" / f"batched-collector-test-{uuid.uuid4().hex}"
    outcome = run_dir / "outcomes.jsonl"
    result = collect_outcome_matrix(
        env_file=env_file,
        cases_path=cases,
        candidate_cache=candidate_cache,
        outcome_path=outcome,
        ledger_path=run_dir / "ledger.jsonl",
        model_ids=["example/model"],
        max_candidates_per_case=1,
        hard_negatives_per_case=1,
    )
    assert result["status"] == "complete"
    assert result["physical_requests_this_run"] == 1
    assert fake.calls == 1
    assert len(outcome.read_text(encoding="utf-8").splitlines()) == 5

    resumed = collect_outcome_matrix(
        env_file=env_file,
        cases_path=cases,
        candidate_cache=candidate_cache,
        outcome_path=outcome,
        ledger_path=run_dir / "ledger.jsonl",
        model_ids=["example/model"],
        max_candidates_per_case=1,
        hard_negatives_per_case=1,
    )
    assert resumed["physical_requests_this_run"] == 0
    assert fake.calls == 1
    assert len(outcome.read_text(encoding="utf-8").splitlines()) == 5
