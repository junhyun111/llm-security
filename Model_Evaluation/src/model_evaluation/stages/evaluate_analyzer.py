from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable

from ..adapters.llm_security import (
    candidate_to_dict,
    case_from_dict,
    semantic_analyzer,
)
from ..config import EvaluationConfig, ExpertMapping
from ..paths import EVALUATION_ROOT, require_within, write_json
from ..juliet.splitter import SPLITS


@dataclass(slots=True)
class _MetricAccumulator:
    case_count: int = 0
    analysis_error_count: int = 0
    truth_count: int = 0
    truth_hits: int = 0
    negative_region_count: int = 0
    negative_region_hits: int = 0
    candidate_count: int = 0
    positive_candidate_count: int = 0
    negative_candidate_count: int = 0
    unscored_candidate_count: int = 0
    elapsed_seconds: float = 0.0
    candidate_loc: list[int] = field(default_factory=list)

    def report(self) -> dict[str, object]:
        precision_denominator = self.candidate_count
        return {
            "case_count": self.case_count,
            "analysis_error_count": self.analysis_error_count,
            "ground_truth_count": self.truth_count,
            "candidate_hit_count": self.truth_hits,
            "candidate_recall": _ratio(self.truth_hits, self.truth_count),
            "target_safe_region_count": self.negative_region_count,
            "target_safe_region_candidate_hits": self.negative_region_hits,
            "target_safe_region_candidate_rate": _ratio(
                self.negative_region_hits, self.negative_region_count
            ),
            "candidate_count": self.candidate_count,
            "positive_matching_candidate_count": self.positive_candidate_count,
            "negative_matching_candidate_count": self.negative_candidate_count,
            "unscored_candidate_count": self.unscored_candidate_count,
            "gt_localization_precision_proxy": _ratio(
                self.positive_candidate_count, precision_denominator
            ),
            "average_candidates_per_case": _ratio(
                self.candidate_count, self.case_count
            ),
            "average_candidates_per_ground_truth": _ratio(
                self.candidate_count, self.truth_count
            ),
            "mean_candidate_loc": (
                float(statistics.fmean(self.candidate_loc))
                if self.candidate_loc
                else 0.0
            ),
            "median_candidate_loc": (
                float(statistics.median(self.candidate_loc))
                if self.candidate_loc
                else 0.0
            ),
            "analysis_latency_ms_per_case": (
                self.elapsed_seconds * 1_000.0 / self.case_count
                if self.case_count
                else 0.0
            ),
        }


def evaluate_semantic_analyzer(
    config: EvaluationConfig,
    mapping: ExpertMapping,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    analyzer = semantic_analyzer(
        max_source_bytes=config.max_source_bytes,
        parse_timeout_ms=config.parse_timeout_ms,
    )
    candidate_directory = require_within(config.paths.candidate_directory, EVALUATION_ROOT)
    candidate_directory.mkdir(parents=True, exist_ok=True)
    split_reports: dict[str, dict[str, object]] = {}
    overall = _MetricAccumulator()
    per_expert: dict[str, _MetricAccumulator] = defaultdict(_MetricAccumulator)
    per_cwe: dict[str, _MetricAccumulator] = defaultdict(_MetricAccumulator)
    failure_rows: list[dict[str, str]] = []

    for split in SPLITS:
        case_path = config.paths.pilot_directory / f"cases_{split}.jsonl"
        raw_cases = _load_jsonl(case_path)
        split_metrics = _MetricAccumulator()
        candidate_rows: list[dict] = []
        for index, raw_case in enumerate(raw_cases, start=1):
            case = case_from_dict(raw_case)
            expert = str(raw_case["metadata"]["expected_expert"])
            cwe = str(raw_case["metadata"]["target_cwe"])
            selected_accumulators = (
                split_metrics,
                overall,
                per_expert[expert],
                per_cwe[cwe],
            )
            for accumulator in selected_accumulators:
                accumulator.case_count += 1
            started = perf_counter()
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
            except (RuntimeError, ValueError, TimeoutError) as error:
                elapsed = perf_counter() - started
                for accumulator in selected_accumulators:
                    accumulator.analysis_error_count += 1
                    accumulator.elapsed_seconds += elapsed
                failure_rows.append(
                    {
                        "case_id": case.case_id,
                        "split": split,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            elapsed = perf_counter() - started
            truths = list(case.ground_truth)
            negative_regions = list(raw_case["metadata"].get("negative_regions", []))
            truth_hits = sum(
                any(_candidate_matches(candidate, truth) for candidate in candidates)
                for truth in truths
            )
            negative_hits = sum(
                any(_candidate_matches(candidate, region) for candidate in candidates)
                for region in negative_regions
            )
            positive_candidates = sum(
                any(_candidate_matches(candidate, truth) for truth in truths)
                for candidate in candidates
            )
            negative_candidates = sum(
                not any(_candidate_matches(candidate, truth) for truth in truths)
                and any(
                    _candidate_matches(candidate, region) for region in negative_regions
                )
                for candidate in candidates
            )
            unscored_candidates = len(candidates) - positive_candidates - negative_candidates
            loc = [max(1, item.line_end - item.line_start + 1) for item in candidates]
            for accumulator in selected_accumulators:
                accumulator.elapsed_seconds += elapsed
                accumulator.truth_count += len(truths)
                accumulator.truth_hits += truth_hits
                accumulator.negative_region_count += len(negative_regions)
                accumulator.negative_region_hits += negative_hits
                accumulator.candidate_count += len(candidates)
                accumulator.positive_candidate_count += positive_candidates
                accumulator.negative_candidate_count += negative_candidates
                accumulator.unscored_candidate_count += unscored_candidates
                accumulator.candidate_loc.extend(loc)
            candidate_rows.extend(
                {
                    "case_id": case.case_id,
                    "split": split,
                    "target_cwe": cwe,
                    "expected_expert": expert,
                    "candidate": candidate_to_dict(candidate),
                    "matches_ground_truth": any(
                        _candidate_matches(candidate, truth) for truth in truths
                    ),
                    "matches_target_safe_region": any(
                        _candidate_matches(candidate, region)
                        for region in negative_regions
                    ),
                }
                for candidate in candidates
            )
            if progress:
                progress(f"analyzer: {split} {index}/{len(raw_cases)} cases")
        _write_jsonl(candidate_directory / f"candidates_{split}.jsonl", candidate_rows)
        split_reports[split] = split_metrics.report()

    _write_jsonl(candidate_directory / "analysis_failures.jsonl", failure_rows)
    report = {
        "schema_version": config.schema_version,
        "mapping_version": mapping.schema_version,
        "mapping_hash": mapping.mapping_hash,
        "analyzer": "SemanticStaticAnalyzer",
        "feature_schema": analyzer.candidate_builder.feature_extractor.schema_version,
        "splits": split_reports,
        "overall": overall.report(),
        "per_expert": {
            key: value.report() for key, value in sorted(per_expert.items())
        },
        "per_cwe": {
            key: value.report() for key, value in sorted(per_cwe.items())
        },
        "failure_count": len(failure_rows),
        "candidate_files": {
            split: str(candidate_directory / f"candidates_{split}.jsonl")
            for split in SPLITS
        },
    }
    write_json(config.paths.analyzer_report, report)
    return report


def _candidate_matches(candidate, region) -> bool:
    file = region.file if hasattr(region, "file") else region["file"]
    line_start = (
        region.line_start if hasattr(region, "line_start") else int(region["line_start"])
    )
    line_end = region.line_end if hasattr(region, "line_end") else int(region["line_end"])
    return (
        candidate.file == file
        and candidate.line_start <= line_end
        and candidate.line_end >= line_start
    )


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    destination = require_within(path, EVALUATION_ROOT)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(destination)


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0

