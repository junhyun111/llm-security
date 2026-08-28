from __future__ import annotations

import json

import pytest

from model_evaluation.candidates import (
    CandidateSelectionManifest,
    build_candidate_selection_manifest,
)
from model_evaluation.paths import EVALUATION_ROOT


def _case(*, truth_line: int | None) -> dict:
    ground_truth = []
    if truth_line is not None:
        ground_truth.append(
            {
                "truth_id": "truth-1",
                "file": "unit.c",
                "function": "f",
                "line_start": truth_line,
                "line_end": truth_line,
                "experts": ["integer_size_type"],
                "cwes": ["CWE-190"],
            }
        )
    return {
        "case_id": "case-1",
        "project_id": "project-1",
        "source_files": {"unit.c": "int f(void) { return 0; }"},
        "split": "test",
        "ground_truth": ground_truth,
        "metadata": {},
    }


def _candidate(candidate_id: str, *, line: int, score: float) -> dict:
    return {
        "case_id": "case-1",
        "candidate": {
            "candidate_id": candidate_id,
            "project_id": "project-1",
            "file": "unit.c",
            "function": "f",
            "line_start": line,
            "line_end": line,
            "code": "int f(void) { return 0; }",
            "evidence": [],
            "features": {"bias": 1.0},
            "suspicion_score": score,
            "feature_schema_version": "test-v1",
            "cwe_hypotheses": [],
        },
    }


def _write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_deployment_selection_is_invariant_to_ground_truth() -> None:
    work = EVALUATION_ROOT / "cache" / "candidate-selection-policy-test"
    cache = work / "candidates.jsonl"
    cases_with_truth = work / "cases_with_truth.jsonl"
    cases_without_truth = work / "cases_without_truth.jsonl"
    manifest_with_truth = work / "deployment_with_truth.jsonl"
    manifest_without_truth = work / "deployment_without_truth.jsonl"
    _write_jsonl(
        cache,
        [
            _candidate("high-score-negative", line=50, score=0.9),
            _candidate("low-score-positive", line=10, score=0.1),
        ],
    )
    _write_jsonl(cases_with_truth, [_case(truth_line=10)])
    _write_jsonl(cases_without_truth, [_case(truth_line=None)])

    first = build_candidate_selection_manifest(
        cases_with_truth,
        cache,
        manifest_with_truth,
        selection_policy="deployment_top_k",
        max_candidates_per_case=1,
        hard_negatives_per_case=0,
    )
    second = build_candidate_selection_manifest(
        cases_without_truth,
        cache,
        manifest_without_truth,
        selection_policy="deployment_top_k",
        max_candidates_per_case=1,
        hard_negatives_per_case=0,
    )

    first_row = json.loads(manifest_with_truth.read_text(encoding="utf-8"))
    second_row = json.loads(manifest_without_truth.read_text(encoding="utf-8"))
    assert first_row["candidate_id"] == "high-score-negative"
    assert first_row["candidate_id"] == second_row["candidate_id"]
    assert "positive_candidate" not in first_row
    assert first["ground_truth_access"] == "none-during-selection"
    assert second["ground_truth_access"] == "none-during-selection"


def test_training_selection_remains_explicitly_ground_truth_aware() -> None:
    work = EVALUATION_ROOT / "cache" / "candidate-selection-policy-test"
    cache = work / "candidates.jsonl"
    cases = work / "cases_with_truth.jsonl"
    manifest = work / "training.jsonl"
    _write_jsonl(
        cache,
        [
            _candidate("high-score-negative", line=50, score=0.9),
            _candidate("low-score-positive", line=10, score=0.1),
        ],
    )
    _write_jsonl(cases, [_case(truth_line=10)])

    summary = build_candidate_selection_manifest(
        cases,
        cache,
        manifest,
        selection_policy="training_matrix",
        max_candidates_per_case=1,
        hard_negatives_per_case=0,
    )

    row = json.loads(manifest.read_text(encoding="utf-8"))
    assert row["candidate_id"] == "low-score-positive"
    assert row["positive_candidate"] is True
    assert summary["ground_truth_access"] == "training-selection"


def test_deployment_selection_rejects_gt_aware_options_and_wrong_policy() -> None:
    work = EVALUATION_ROOT / "cache" / "candidate-selection-policy-test"
    cache = work / "candidates.jsonl"
    cases = work / "cases_with_truth.jsonl"
    manifest = work / "training-for-policy-check.jsonl"
    _write_jsonl(cache, [_candidate("candidate", line=10, score=0.9)])
    _write_jsonl(cases, [_case(truth_line=10)])

    with pytest.raises(ValueError, match="does not accept hard negatives"):
        build_candidate_selection_manifest(
            cases,
            cache,
            work / "invalid.jsonl",
            selection_policy="deployment_top_k",
            hard_negatives_per_case=1,
        )

    build_candidate_selection_manifest(
        cases,
        cache,
        manifest,
        selection_policy="training_matrix",
        max_candidates_per_case=1,
        hard_negatives_per_case=0,
    )
    with pytest.raises(ValueError, match="policy mismatch"):
        CandidateSelectionManifest(
            manifest,
            expected_policy="deployment_top_k",
        )
