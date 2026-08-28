from __future__ import annotations

import json
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
        if self._pending is None or str(self._pending["case_id"]) != case.case_id:
            return []
        candidates = []
        while self._pending is not None and str(self._pending["case_id"]) == case.case_id:
            candidates.append(candidate_from_dict(self._pending["candidate"]))
            self._pending = next(self._rows, None)
        return candidates


def selected_candidate_manifest(
    cases_path: str | Path,
    candidate_cache: str | Path,
    output_path: str | Path,
    *,
    max_candidates_per_case: int = 4,
    hard_negatives_per_case: int = 1,
) -> dict[str, object]:
    analyzer = StreamingCachedCandidateAnalyzer(candidate_cache)
    rows: list[dict] = []
    positive_count = negative_count = 0
    for raw in iter_jsonl(cases_path):
        case = case_from_dict(raw)
        selected = select_matrix_candidates(
            analyzer.analyze(case),
            case.ground_truth,
            max_candidates=max_candidates_per_case,
            hard_negatives=hard_negatives_per_case,
        )
        for candidate in selected:
            positive = any(_matches(candidate, truth) for truth in case.ground_truth)
            positive_count += int(positive)
            negative_count += int(not positive)
            rows.append(
                {
                    "case_id": case.case_id,
                    "candidate_id": candidate.candidate_id,
                    "positive_candidate": positive,
                }
            )
    destination = write_jsonl(output_path, rows)
    return {
        "selection_manifest": str(destination),
        "selected_candidate_count": len(rows),
        "positive_candidate_count": positive_count,
        "hard_negative_candidate_count": negative_count,
    }


def select_matrix_candidates(
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


def _matches(candidate, truth) -> bool:
    return (
        candidate.file == truth.file
        and candidate.line_start <= truth.line_end
        and candidate.line_end >= truth.line_start
    )
