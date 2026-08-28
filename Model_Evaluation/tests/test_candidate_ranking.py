from __future__ import annotations

import json

from model_evaluation.candidate_ranking import (
    evaluate_candidate_ranker,
    rank_candidate_cache,
    train_candidate_ranker_suite,
)
from model_evaluation.adapters.llm_security import activate_parent_package
from model_evaluation.paths import EVALUATION_ROOT


activate_parent_package()
from llm_security.analysis.features import FEATURE_SCHEMA_SEMANTIC_CWE_V3


def _features(value: float) -> dict[str, float]:
    return {name: value for name in FEATURE_SCHEMA_SEMANTIC_CWE_V3}


def _case(case_id: str, truth_line: int) -> dict:
    return {
        "case_id": case_id,
        "project_id": case_id,
        "source_files": {"unit.c": "int f(void) { return 0; }"},
        "split": "train",
        "ground_truth": [{
            "truth_id": f"truth-{case_id}", "file": "unit.c", "function": "f",
            "line_start": truth_line, "line_end": truth_line,
            "experts": ["integer_size_type"], "cwes": ["CWE-190"],
        }],
        "metadata": {},
    }


def _candidate(case_id: str, suffix: str, line: int, value: float) -> dict:
    return {
        "case_id": case_id,
        "candidate": {
            "candidate_id": f"candidate-{case_id}-{suffix}", "project_id": case_id,
            "file": "unit.c", "function": "f", "line_start": line, "line_end": line,
            "code": "int f(void) { return 0; }", "evidence": [],
            "features": _features(value), "suspicion_score": 1.0 - value,
            "feature_schema_version": "semantic-cwe-v3", "cwe_hypotheses": [],
        },
    }


def test_candidate_ranker_suite_selects_and_rewrites_cache() -> None:
    work = EVALUATION_ROOT / "cache" / "candidate-ranker-test"
    work.mkdir(parents=True, exist_ok=True)
    train_cases = work / "train.jsonl"
    dev_cases = work / "dev.jsonl"
    train_cache = work / "train_candidates.jsonl"
    dev_cache = work / "dev_candidates.jsonl"
    cases = [_case("case-a", 10), _case("case-b", 20)]
    candidates = [
        _candidate("case-a", "positive", 10, 1.0),
        _candidate("case-a", "negative", 50, 0.0),
        _candidate("case-b", "positive", 20, 0.9),
        _candidate("case-b", "negative", 60, 0.1),
    ]
    text = "".join(json.dumps(row) + "\n" for row in cases)
    cache_text = "".join(json.dumps(row) + "\n" for row in candidates)
    train_cases.write_text(text, encoding="utf-8")
    dev_cases.write_text(text, encoding="utf-8")
    train_cache.write_text(cache_text, encoding="utf-8")
    dev_cache.write_text(cache_text, encoding="utf-8")

    report = train_candidate_ranker_suite(
        train_cases=train_cases,
        train_candidate_cache=train_cache,
        dev_cases=dev_cases,
        dev_candidate_cache=dev_cache,
        artifact_dir=work / "artifacts",
        report_path=work / "report.json",
        backends=("logistic_regression", "gradient_boosting"),
        seed=7,
    )
    assert report["selected_backend"] in {"logistic_regression", "gradient_boosting"}
    assert len(report["selected_artifact_sha256"]) == 64
    assert report["variants"][report["selected_backend"]]["recall_at_k"]["1"] == 1.0

    ranked_path = work / "dev_candidates_ranked.jsonl"
    summary = rank_candidate_cache(
        cases_path=dev_cases,
        input_cache=dev_cache,
        output_path=ranked_path,
        artifact_path=report["selected_artifact"],
    )
    assert summary["candidate_count"] == 4
    metrics = evaluate_candidate_ranker(
        cases_path=dev_cases,
        candidate_cache=ranked_path,
        ranker=__import__("llm_security.analysis", fromlist=["LearnedCandidateRanker"])
        .LearnedCandidateRanker.load(report["selected_artifact"]),
        recall_ks=(1, 2),
    )
    assert metrics["recall_at_k"]["1"] == 1.0
