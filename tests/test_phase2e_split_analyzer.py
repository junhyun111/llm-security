from __future__ import annotations

import json

from llm_security.experiments.analyzer_eval import evaluate_analyzer
from llm_security.datasets import write_cases_jsonl
from llm_security.experiments.split import (
    freeze_project_split,
    freeze_project_split_jsonl,
)
from llm_security.experiments.streaming import analyze_split_jsonl
from llm_security.models import Candidate, ExpertFamily, GroundTruth, ProjectCase


def _case(index: int) -> ProjectCase:
    return ProjectCase(
        case_id=f"case-{index}",
        project_id=f"project-{index}",
        source_files={"a.c": "void f(void) {}\n"},
        ground_truth=[
            GroundTruth(
                truth_id=f"truth-{index}",
                file="a.c",
                function="f",
                line_start=1,
                line_end=1,
                experts=[ExpertFamily.MEMORY_BOUNDS],
            )
        ],
    )


class FirstCaseOnlyAnalyzer:
    def analyze(self, case: ProjectCase) -> list[Candidate]:
        if case.case_id != "case-0":
            return []
        return [
            Candidate(
                candidate_id="candidate-0",
                project_id=case.project_id,
                file="a.c",
                function="f",
                line_start=1,
                line_end=1,
                code=case.source_files["a.c"],
                evidence=[],
                features={"x": 1.0},
                suspicion_score=0.9,
            )
        ]


class FailingAnalyzer:
    def analyze(self, case: ProjectCase) -> list[Candidate]:
        raise RecursionError(f"too deep: {case.case_id}")


def test_frozen_split_is_project_disjoint_and_persisted(tmp_path):
    frozen = freeze_project_split([_case(i) for i in range(8)], tmp_path, seed=2026)
    projects = {
        name: {case.project_id for case in cases}
        for name, cases in frozen.cases.items()
    }
    assert not projects["train"] & projects["dev"]
    assert not projects["train"] & projects["test"]
    assert not projects["dev"] & projects["test"]
    manifest = json.loads((tmp_path / "split_manifest.json").read_text("utf-8"))
    assert manifest["seed"] == 2026
    assert sum(item["case_count"] for item in manifest["splits"].values()) == 8


def test_analyzer_misses_remain_in_benchmark_denominator():
    cases = [_case(0), _case(1), _case(2)]
    evaluation = evaluate_analyzer(FirstCaseOnlyAnalyzer(), cases)
    assert evaluation.metrics.ground_truth_count == 3
    assert evaluation.metrics.candidate_hit_count == 1
    assert evaluation.metrics.candidate_recall == 1 / 3
    assert set(evaluation.candidates_by_case) == {case.case_id for case in cases}


def test_streaming_split_matches_in_memory_project_assignment(tmp_path):
    cases = [_case(i) for i in range(8)]
    input_path = tmp_path / "cases.jsonl"
    write_cases_jsonl(cases, input_path)
    in_memory = freeze_project_split(cases, tmp_path / "memory", seed=2026)
    streaming = freeze_project_split_jsonl(
        input_path, tmp_path / "streaming", seed=2026
    )
    for split in ("train", "dev", "test"):
        assert (
            in_memory.manifest["splits"][split]["projects"]
            == streaming.manifest["splits"][split]["projects"]
        )


def test_streaming_analysis_records_case_error_and_continues(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    failure_path = tmp_path / "failures.jsonl"
    write_cases_jsonl([_case(i) for i in range(3)], cases_path)
    output = analyze_split_jsonl(
        FailingAnalyzer(),
        cases_path,
        retain_compact_analysis=False,
        expected_case_count=3,
        failure_log_path=failure_path,
    )

    assert output.metrics.case_count == 3
    assert output.metrics.ground_truth_count == 3
    assert output.metrics.analysis_error_count == 3
    assert output.metrics.candidate_recall == 0.0
    assert len(failure_path.read_text(encoding="utf-8").splitlines()) == 3
