from __future__ import annotations

import json

import pytest

from llm_security.datasets import load_router_samples_jsonl, write_cases_jsonl
from llm_security.experiments import (
    Phase2EConfig,
    prepare_phase2e_jsonl,
    run_phase2e,
    run_phase2e_jsonl,
)
from llm_security.models import ExpertFamily, GroundTruth, ProjectCase
from llm_security.experiments.streaming import analyze_split_jsonl


SOURCE = """\
#include <stdlib.h>
#include <string.h>
void copy_bad(char *dst, const char *src, int len) {
    memcpy(dst, src, len);
}
void lifetime_bad(char *ptr) {
    free(ptr);
    ptr[0] = 1;
}
"""


def _workflow_case(index: int) -> ProjectCase:
    return ProjectCase(
        case_id=f"workflow-{index}",
        project_id=f"workflow-project-{index}",
        source_files={"sample.c": SOURCE},
        ground_truth=[
            GroundTruth(
                f"memory-{index}",
                "sample.c",
                "copy_bad",
                3,
                5,
                [ExpertFamily.MEMORY_BOUNDS],
            ),
            GroundTruth(
                f"lifetime-{index}",
                "sample.c",
                "lifetime_bad",
                6,
                9,
                [ExpertFamily.LIFETIME_RESOURCE],
            ),
        ],
    )


def test_phase2e_workflow_writes_all_offline_artifacts(tmp_path):
    result = run_phase2e(
        [_workflow_case(index) for index in range(3)],
        config=Phase2EConfig(
            data_directory=tmp_path / "data",
            artifact_directory=tmp_path / "artifacts",
            output_directory=tmp_path / "results",
        ),
    )
    expected = {
        "experiment_manifest.json",
        "analyzer_comparison.json",
        "gate_calibration.json",
        "router_legacy.json",
        "router_semantic.json",
        "routing_ablation.json",
        "per_family.json",
        "confusion_matrix.json",
        "confusion_matrix.csv",
        "feature_importance.json",
    }
    assert expected <= {path.name for path in (tmp_path / "results").iterdir()}
    manifest = json.loads(
        (tmp_path / "results" / "experiment_manifest.json").read_text("utf-8")
    )
    assert manifest["offline_only"] is True
    assert manifest["llm_api_calls"] == 0
    assert result["manifest"]["gate_calibration_split"] == "dev"


def test_phase2e_streaming_workflow_keeps_source_out_of_router_data(tmp_path):
    cases_path = tmp_path / "cases_all.jsonl"
    write_cases_jsonl([_workflow_case(index) for index in range(3)], cases_path)
    result = run_phase2e_jsonl(
        cases_path,
        config=Phase2EConfig(
            data_directory=tmp_path / "data",
            artifact_directory=tmp_path / "artifacts",
            output_directory=tmp_path / "results",
        ),
    )
    samples = load_router_samples_jsonl(
        tmp_path / "data" / "semantic" / "router_train.jsonl"
    )
    assert samples
    assert all(sample.candidate.code == "" for sample in samples)
    assert result["manifest"]["case_counts"] == {"train": 1, "dev": 1, "test": 1}


def test_phase2e_prepare_stops_before_training(tmp_path):
    cases_path = tmp_path / "cases_all.jsonl"
    write_cases_jsonl([_workflow_case(index) for index in range(3)], cases_path)
    summary = prepare_phase2e_jsonl(
        cases_path,
        config=Phase2EConfig(data_directory=tmp_path / "data"),
    )
    assert summary["training_performed"] is False
    assert (tmp_path / "data" / "legacy" / "router_train.jsonl").exists()
    assert (tmp_path / "data" / "semantic" / "router_dev.jsonl").exists()
    assert not (tmp_path / "artifacts").exists()


class _InterruptingAnalyzer:
    def __init__(self, *, interrupt_on_call: int | None = None) -> None:
        self.interrupt_on_call = interrupt_on_call
        self.calls = 0

    def analyze(self, case: ProjectCase):
        self.calls += 1
        if self.calls == self.interrupt_on_call:
            raise KeyboardInterrupt()
        return []


def test_streaming_analysis_resumes_from_atomic_100_case_checkpoints(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    cases = [
        ProjectCase(
            case_id=f"checkpoint-{index}",
            project_id=f"checkpoint-project-{index}",
            source_files={"sample.c": "void f(void) {}"},
        )
        for index in range(205)
    ]
    write_cases_jsonl(cases, cases_path)
    checkpoint = tmp_path / "analysis_checkpoints" / "semantic_train_prepare.json"

    interrupted = _InterruptingAnalyzer(interrupt_on_call=151)
    with pytest.raises(KeyboardInterrupt):
        analyze_split_jsonl(
            interrupted,
            cases_path,
            retain_compact_analysis=False,
            expected_case_count=205,
            checkpoint_path=checkpoint,
            checkpoint_every=100,
        )

    raw_checkpoint = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert raw_checkpoint["completed_cases"] == 150
    assert "void f(void)" not in checkpoint.read_text(encoding="utf-8")

    resumed = _InterruptingAnalyzer()
    output = analyze_split_jsonl(
        resumed,
        cases_path,
        retain_compact_analysis=False,
        expected_case_count=205,
        checkpoint_path=checkpoint,
        checkpoint_every=100,
    )

    assert output.resumed_case_count == 150
    assert output.metrics.case_count == 205
    assert resumed.calls == 55
    assert json.loads(checkpoint.read_text(encoding="utf-8"))["completed_cases"] == 205

    restarted = _InterruptingAnalyzer()
    analyze_split_jsonl(
        restarted,
        cases_path,
        retain_compact_analysis=False,
        expected_case_count=205,
        checkpoint_path=checkpoint,
        checkpoint_every=100,
        resume=False,
    )
    assert restarted.calls == 205
