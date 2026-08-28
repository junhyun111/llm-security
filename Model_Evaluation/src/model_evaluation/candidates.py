from __future__ import annotations

import json
import hashlib
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable

from .adapters.llm_security import (
    candidate_from_dict,
    candidate_to_dict,
    case_from_dict,
    semantic_analyzer,
    semantic_analyzer_version,
    semantic_feature_schema,
)
from .jsonl import iter_jsonl, write_jsonl
from .paths import EVALUATION_ROOT, require_within, write_json


TRAINING_MATRIX_SELECTION = "training_matrix"
DEPLOYMENT_TOP_K_SELECTION = "deployment_top_k"
CANDIDATE_SELECTION_POLICIES = {
    TRAINING_MATRIX_SELECTION,
    DEPLOYMENT_TOP_K_SELECTION,
}


def cache_candidates(
    cases_path: str | Path,
    output_path: str | Path,
    *,
    max_source_bytes: int | None = None,
    parse_timeout_ms: int = 30_000,
    reuse_existing: bool = True,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Analyze cases once and freeze candidates used by every later stage."""
    destination = require_within(output_path, EVALUATION_ROOT)
    cases_source = Path(cases_path).resolve()
    source_stat = cases_source.stat()
    feature_schema = semantic_feature_schema()
    analyzer_version = semantic_analyzer_version()
    summary_path = destination.with_suffix(".summary.json")
    if reuse_existing and destination.is_file() and summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            existing.get("cases") == str(cases_source)
            and int(existing.get("cases_size", -1)) == source_stat.st_size
            and int(existing.get("cases_mtime_ns", -1)) == source_stat.st_mtime_ns
            and existing.get("max_source_bytes") == max_source_bytes
            and int(existing.get("parse_timeout_ms", -1)) == parse_timeout_ms
            and existing.get("feature_schema") == feature_schema
            and existing.get("analyzer_version") == analyzer_version
        ):
            return existing
    analyzer = semantic_analyzer(
        max_source_bytes=max_source_bytes,
        parse_timeout_ms=parse_timeout_ms,
    )
    failures: list[dict] = []
    case_count = candidate_count = 0
    cases_with_candidates = ground_truth_count = ground_truth_hits = 0
    started = perf_counter()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for raw_case in iter_jsonl(cases_path):
            case_count += 1
            case = case_from_dict(raw_case)
            try:
                candidates = sorted(
                    analyzer.analyze(case),
                    key=lambda item: (
                        -item.suspicion_score,
                        item.file,
                        item.line_start,
                        item.candidate_id,
                    ),
                )
            except (RuntimeError, TimeoutError, ValueError) as error:
                failures.append({"case_id": case.case_id, "error": str(error)})
                candidates = []
            for candidate in candidates:
                handle.write(json.dumps(
                    {"case_id": case.case_id, "candidate": candidate_to_dict(candidate)},
                    ensure_ascii=False,
                    sort_keys=True,
                ) + "\n")
            candidate_count += len(candidates)
            cases_with_candidates += int(bool(candidates))
            ground_truth_count += len(case.ground_truth)
            ground_truth_hits += sum(
                any(_matches(candidate, truth) for candidate in candidates)
                for truth in case.ground_truth
            )
            if progress and (case_count == 1 or case_count % 100 == 0):
                progress(
                    f"candidate cache: {case_count} cases / "
                    f"{candidate_count} candidates"
                )
    temporary.replace(destination)
    failure_path = destination.with_name(destination.stem + "_failures.jsonl")
    write_jsonl(failure_path, failures)
    summary = {
        "cases": str(cases_source),
        "cases_size": source_stat.st_size,
        "cases_mtime_ns": source_stat.st_mtime_ns,
        "max_source_bytes": max_source_bytes,
        "parse_timeout_ms": parse_timeout_ms,
        "candidate_cache": str(destination),
        "case_count": case_count,
        "candidate_count": candidate_count,
        "cases_with_candidates": cases_with_candidates,
        "zero_candidate_case_count": case_count - cases_with_candidates,
        "ground_truth_count": ground_truth_count,
        "ground_truth_candidate_hits": ground_truth_hits,
        "candidate_recall": (
            ground_truth_hits / ground_truth_count if ground_truth_count else 1.0
        ),
        "analysis_failure_count": len(failures),
        "elapsed_seconds": perf_counter() - started,
        "feature_schema": feature_schema,
        "analyzer_version": analyzer_version,
    }
    write_json(summary_path, summary)
    return summary


class CachedCandidateAnalyzer:
    """CandidateAnalyzer implementation backed by a frozen JSONL cache."""

    def __init__(self, path: str | Path) -> None:
        grouped: dict[str, list] = defaultdict(list)
        for row in iter_jsonl(path):
            grouped[str(row["case_id"])].append(candidate_from_dict(row["candidate"]))
        self._by_case = dict(grouped)

    def analyze(self, case) -> list:
        return list(self._by_case.get(case.case_id, ()))


class StreamingCachedCandidateAnalyzer:
    """Constant-memory cache reader for cases consumed in source JSONL order."""

    def __init__(self, path: str | Path) -> None:
        self._rows = iter(iter_jsonl(path))
        self._pending: dict | None = next(self._rows, None)

    def analyze(self, case) -> list:
        return self.analyze_case_id(case.case_id)

    def analyze_case_id(self, case_id: str) -> list:
        if self._pending is None or str(self._pending["case_id"]) != case_id:
            return []
        candidates = []
        while self._pending is not None and str(self._pending["case_id"]) == case_id:
            candidates.append(candidate_from_dict(self._pending["candidate"]))
            self._pending = next(self._rows, None)
        return candidates


def build_candidate_selection_manifest(
    cases_path: str | Path,
    candidate_cache: str | Path,
    output_path: str | Path,
    *,
    selection_policy: str,
    max_candidates_per_case: int = 4,
    hard_negatives_per_case: int = 1,
) -> dict[str, object]:
    if selection_policy not in CANDIDATE_SELECTION_POLICIES:
        raise ValueError(
            "selection_policy must be training_matrix or deployment_top_k"
        )
    if selection_policy == DEPLOYMENT_TOP_K_SELECTION and hard_negatives_per_case:
        raise ValueError(
            "deployment_top_k does not accept hard negatives or ground-truth selection"
        )
    analyzer = StreamingCachedCandidateAnalyzer(candidate_cache)
    rows: list[dict] = []
    positive_count = negative_count = case_count = 0
    for raw in iter_jsonl(cases_path):
        case_count += 1
        case_id = str(raw["case_id"])
        candidates = analyzer.analyze_case_id(case_id)
        if selection_policy == TRAINING_MATRIX_SELECTION:
            case = case_from_dict(raw)
            selected = select_training_matrix_candidates(
                candidates,
                case.ground_truth,
                max_candidates=max_candidates_per_case,
                hard_negatives=hard_negatives_per_case,
            )
        else:
            selected = select_deployment_candidates(
                candidates,
                max_candidates=max_candidates_per_case,
            )
        for rank, candidate in enumerate(selected, start=1):
            row = {
                "case_id": case_id,
                "candidate_id": candidate.candidate_id,
                "rank": rank,
                "ranker_score": candidate.suspicion_score,
                "selection_policy": selection_policy,
            }
            if selection_policy == TRAINING_MATRIX_SELECTION:
                positive = any(
                    _matches(candidate, truth) for truth in case.ground_truth
                )
                positive_count += int(positive)
                negative_count += int(not positive)
                row["positive_candidate"] = positive
            rows.append(row)
    destination = write_jsonl(output_path, rows)
    summary = {
        "selection_contract_version": "candidate-selection-v2",
        "selection_policy": selection_policy,
        "selection_manifest": str(destination),
        "selection_manifest_sha256": _sha256(destination),
        "cases": str(Path(cases_path).resolve()),
        "cases_sha256": _sha256(cases_path),
        "candidate_cache": str(Path(candidate_cache).resolve()),
        "candidate_cache_sha256": _sha256(candidate_cache),
        "max_candidates_per_case": max_candidates_per_case,
        "case_count": case_count,
        "selected_candidate_count": len(rows),
    }
    if selection_policy == TRAINING_MATRIX_SELECTION:
        summary.update(
            {
                "hard_negatives_per_case": hard_negatives_per_case,
                "positive_candidate_count": positive_count,
                "hard_negative_candidate_count": negative_count,
                "ground_truth_access": "training-selection",
            }
        )
    else:
        summary["ground_truth_access"] = "none-during-selection"
    write_json(destination.with_suffix(".summary.json"), summary)
    return summary


def select_training_matrix_candidates(
    candidates, truths, *, max_candidates: int, hard_negatives: int
):
    if max_candidates < 1 or hard_negatives < 0:
        raise ValueError("Candidate limits are invalid")
    positives = [item for item in candidates if any(_matches(item, truth) for truth in truths)]
    negatives = [item for item in candidates if item not in positives]
    positives.sort(key=lambda item: (-item.suspicion_score, item.candidate_id))
    negatives.sort(key=lambda item: (-item.suspicion_score, item.candidate_id))
    slots = min(hard_negatives, len(negatives), max_candidates)
    selected = positives[: max_candidates - slots] + negatives[:slots]
    return list({item.candidate_id: item for item in selected}.values())


def select_deployment_candidates(candidates, *, max_candidates: int):
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    return sorted(
        candidates,
        key=lambda item: (
            -item.suspicion_score,
            item.file,
            item.line_start,
            item.candidate_id,
        ),
    )[:max_candidates]


class CandidateSelectionManifest:
    """Resolve the exact preplanned candidate IDs without consulting labels."""

    def __init__(
        self,
        path: str | Path,
        *,
        expected_policy: str,
        cases_path: str | Path | None = None,
        candidate_cache: str | Path | None = None,
        max_candidates_per_case: int | None = None,
    ) -> None:
        if expected_policy not in CANDIDATE_SELECTION_POLICIES:
            raise ValueError(f"Unknown candidate selection policy: {expected_policy}")
        self.path = Path(path).resolve()
        self.sha256 = _sha256(self.path)
        self.expected_policy = expected_policy
        summary_path = self.path.with_suffix(".summary.json")
        if not summary_path.is_file():
            raise ValueError(
                f"Candidate selection summary is missing: {summary_path}"
            )
        self.summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_policy = str(self.summary.get("selection_policy", ""))
        if summary_policy != expected_policy:
            raise ValueError(
                f"Selection manifest policy mismatch: {summary_policy!r} != "
                f"{expected_policy!r}"
            )
        expected_summary = {
            "selection_manifest_sha256": self.sha256,
        }
        if cases_path is not None:
            expected_summary["cases_sha256"] = _sha256(cases_path)
        if candidate_cache is not None:
            expected_summary["candidate_cache_sha256"] = _sha256(candidate_cache)
        if max_candidates_per_case is not None:
            expected_summary["max_candidates_per_case"] = max_candidates_per_case
        mismatches = {
            key: (self.summary.get(key), value)
            for key, value in expected_summary.items()
            if self.summary.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "Candidate selection summary contract mismatch: " + str(mismatches)
            )
        self._by_case: dict[str, list[str]] = defaultdict(list)
        row_count = 0
        for row in iter_jsonl(self.path):
            row_count += 1
            policy = str(row.get("selection_policy", ""))
            if policy != expected_policy:
                raise ValueError(
                    f"Selection manifest policy mismatch: {policy!r} != "
                    f"{expected_policy!r}"
                )
            self._by_case[str(row["case_id"])].append(str(row["candidate_id"]))
        if row_count != int(self.summary.get("selected_candidate_count", -1)):
            raise ValueError("Candidate selection row count does not match its summary")

    def select(self, case_id: str, candidates) -> list:
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        expected_ids = self._by_case.get(case_id, [])
        missing = [candidate_id for candidate_id in expected_ids if candidate_id not in by_id]
        if missing:
            raise ValueError(
                f"Selection manifest references missing candidates for {case_id}: {missing}"
            )
        return [by_id[candidate_id] for candidate_id in expected_ids]


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matches(candidate, truth) -> bool:
    return (
        candidate.file == truth.file
        and candidate.line_start <= truth.line_end
        and candidate.line_end >= truth.line_start
    )
