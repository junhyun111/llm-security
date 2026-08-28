from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .adapters.llm_security import (
    activate_parent_package,
    candidate_from_dict,
    candidate_to_dict,
    case_from_dict,
)
from .jsonl import iter_jsonl, write_jsonl
from .paths import EVALUATION_ROOT, require_within, write_json


DEFAULT_RECALL_KS = (1, 2, 4, 8)


def train_candidate_ranker_suite(
    *,
    train_cases: str | Path,
    train_candidate_cache: str | Path,
    dev_cases: str | Path,
    dev_candidate_cache: str | Path,
    artifact_dir: str | Path,
    report_path: str | Path,
    backends: Iterable[str] = (
        "logistic_regression",
        "gradient_boosting",
        "small_mlp",
    ),
    recall_ks: Iterable[int] = DEFAULT_RECALL_KS,
    selection_k: int = 4,
    seed: int = 2026,
) -> dict[str, object]:
    """Fit candidate rankers on train only and select one by dev Recall@K.

    Labels are derived solely from candidate/ground-truth line overlap. No Expert
    output and no API call is used in this stage.
    """
    activate_parent_package()
    from llm_security.analysis import LearnedCandidateRanker

    backend_names = tuple(dict.fromkeys(str(item) for item in backends))
    ks = _normalize_ks(recall_ks)
    if selection_k not in ks:
        ks = tuple(sorted((*ks, selection_k)))
    destination = require_within(artifact_dir, EVALUATION_ROOT)
    destination.mkdir(parents=True, exist_ok=True)

    train_rows, train_labels, train_summary = _training_rows(
        train_cases, train_candidate_cache
    )
    if not train_rows:
        raise ValueError("Candidate Ranker training set contains no candidates")
    if len(set(train_labels)) != 2:
        raise ValueError(
            "Candidate Ranker training requires both positive and negative candidates"
        )

    variants: dict[str, dict[str, object]] = {}
    rankers = {}
    for backend in backend_names:
        ranker = LearnedCandidateRanker.fit(
            train_rows,
            train_labels,
            backend=backend,
            seed=seed,
        )
        artifact = ranker.save(destination / f"candidate_ranker_{backend}.pkl")
        metrics = evaluate_candidate_ranker(
            cases_path=dev_cases,
            candidate_cache=dev_candidate_cache,
            ranker=ranker,
            recall_ks=ks,
        )
        variants[backend] = {
            "artifact": str(artifact),
            "artifact_sha256": _sha256(artifact),
            **metrics,
        }
        rankers[backend] = ranker

    selected_backend = max(
        backend_names,
        key=lambda backend: (
            float(variants[backend]["recall_at_k"][str(selection_k)]),
            *(float(variants[backend]["recall_at_k"][str(k)]) for k in ks),
            backend,
        ),
    )
    selected_artifact = rankers[selected_backend].save(
        destination / "candidate_ranker.pkl"
    )
    report = {
        "selection_metric": f"dev_recall_at_{selection_k}",
        "selected_backend": selected_backend,
        "selected_artifact": str(selected_artifact),
        "selected_artifact_sha256": _sha256(selected_artifact),
        "feature_schema": rankers[selected_backend].feature_schema_version,
        "seed": seed,
        "train": train_summary,
        "dev": {
            "cases": str(Path(dev_cases).resolve()),
            "candidate_cache": str(Path(dev_candidate_cache).resolve()),
        },
        "variants": variants,
    }
    write_json(report_path, report)
    return report


def evaluate_candidate_ranker(
    *,
    cases_path: str | Path,
    candidate_cache: str | Path,
    ranker,
    recall_ks: Iterable[int] = DEFAULT_RECALL_KS,
) -> dict[str, object]:
    ks = _normalize_ks(recall_ks)
    candidates_by_case = _load_candidate_cache(candidate_cache)
    hits = {k: 0 for k in ks}
    truth_count = candidate_count = case_count = 0
    positive_candidate_count = 0
    for raw_case in iter_jsonl(cases_path):
        case_count += 1
        case = case_from_dict(raw_case)
        candidates = candidates_by_case.get(case.case_id, [])
        candidate_count += len(candidates)
        positive_candidate_count += sum(
            any(_matches(candidate, truth) for truth in case.ground_truth)
            for candidate in candidates
        )
        ranked = ranker.rank(candidates)
        for truth in case.ground_truth:
            truth_count += 1
            for k in ks:
                hits[k] += int(any(_matches(item, truth) for item in ranked[:k]))
    return {
        "case_count": case_count,
        "candidate_count": candidate_count,
        "positive_candidate_count": positive_candidate_count,
        "ground_truth_count": truth_count,
        "recall_at_k": {
            str(k): (hits[k] / truth_count if truth_count else 1.0) for k in ks
        },
        "ground_truth_hits_at_k": {str(k): hits[k] for k in ks},
    }


def rank_candidate_cache(
    *,
    cases_path: str | Path,
    input_cache: str | Path,
    output_path: str | Path,
    artifact_path: str | Path,
) -> dict[str, object]:
    """Write a score-updated cache ordered by the selected learned ranker."""
    activate_parent_package()
    from llm_security.analysis import LearnedCandidateRanker

    destination = require_within(output_path, EVALUATION_ROOT)
    ranker = LearnedCandidateRanker.load(artifact_path)
    candidates_by_case = _load_candidate_cache(input_cache)
    rows: list[dict] = []
    case_count = candidate_count = 0
    for raw_case in iter_jsonl(cases_path):
        case_count += 1
        case = case_from_dict(raw_case)
        ranked = ranker.rank(candidates_by_case.get(case.case_id, []))
        candidate_count += len(ranked)
        rows.extend(
            {"case_id": case.case_id, "candidate": candidate_to_dict(candidate)}
            for candidate in ranked
        )
    write_jsonl(destination, rows)
    summary = {
        "cases": str(Path(cases_path).resolve()),
        "input_candidate_cache": str(Path(input_cache).resolve()),
        "candidate_cache": str(destination),
        "candidate_ranker_artifact": str(Path(artifact_path).resolve()),
        "candidate_ranker_artifact_sha256": _sha256(artifact_path),
        "candidate_ranker_backend": ranker.backend,
        "feature_schema": ranker.feature_schema_version,
        "case_count": case_count,
        "candidate_count": candidate_count,
    }
    write_json(destination.with_suffix(".summary.json"), summary)
    return summary


def _training_rows(cases_path, candidate_cache):
    candidates_by_case = _load_candidate_cache(candidate_cache)
    rows = []
    labels: list[bool] = []
    case_count = truth_count = 0
    for raw_case in iter_jsonl(cases_path):
        case_count += 1
        case = case_from_dict(raw_case)
        truth_count += len(case.ground_truth)
        for candidate in candidates_by_case.get(case.case_id, []):
            rows.append(candidate)
            labels.append(
                any(_matches(candidate, truth) for truth in case.ground_truth)
            )
    return rows, labels, {
        "cases": str(Path(cases_path).resolve()),
        "candidate_cache": str(Path(candidate_cache).resolve()),
        "case_count": case_count,
        "candidate_count": len(rows),
        "positive_candidate_count": sum(labels),
        "negative_candidate_count": len(labels) - sum(labels),
        "ground_truth_count": truth_count,
    }


def _load_candidate_cache(path: str | Path) -> dict[str, list]:
    grouped: dict[str, list] = defaultdict(list)
    for row in iter_jsonl(path):
        grouped[str(row["case_id"])].append(candidate_from_dict(row["candidate"]))
    return dict(grouped)


def _normalize_ks(values: Iterable[int]) -> tuple[int, ...]:
    ks = tuple(sorted(set(int(value) for value in values)))
    if not ks or any(value < 1 for value in ks):
        raise ValueError("Recall K values must be positive integers")
    return ks


def _matches(candidate, truth) -> bool:
    return (
        candidate.file == truth.file
        and candidate.line_start <= truth.line_end
        and candidate.line_end >= truth.line_start
    )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
