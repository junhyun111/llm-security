from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

from ..config import ACTIVE_EXPERTS, EvaluationConfig, ExpertMapping
from ..paths import EVALUATION_ROOT, require_within, write_json
from ..schemas import IndexedScenario
from ..juliet.audit import audit_pilot
from ..juliet.function_regions import materialize_scenario
from ..juliet.indexer import iter_scenarios
from ..juliet.splitter import SPLITS


def build_pilot(
    config: EvaluationConfig,
    mapping: ExpertMapping,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    destination = require_within(config.paths.pilot_directory, EVALUATION_ROOT)
    destination.mkdir(parents=True, exist_ok=True)
    scenarios = list(iter_scenarios(config.paths.index, include_duplicates=False))
    if any(not item.split or not item.leakage_group for item in scenarios):
        raise ValueError("Frozen split assignments are missing; run the index stage first")

    split_quota = _allocate_split_quota(
        config.pilot_per_expert,
        train=config.train_fraction,
        dev=config.dev_fraction,
        test=config.test_fraction,
    )
    selected_cases: list[dict] = []
    review_rows: list[dict[str, object]] = []
    warning_rows: list[dict[str, object]] = []
    for expert in ACTIVE_EXPERTS:
        for split in SPLITS:
            required = split_quota[split]
            candidates = [
                item
                for item in scenarios
                if item.expert == expert and item.split == split
            ]
            chosen = 0
            for scenario in _diverse_order(candidates, config.seed, expert, split):
                try:
                    materialized = materialize_scenario(scenario, config, mapping)
                except (OSError, ValueError, RuntimeError) as error:
                    warning_rows.append(
                        {
                            "case_id": scenario.case_id,
                            "expert": expert,
                            "split": split,
                            "status": "materialization_failed",
                            "warnings": [str(error)],
                        }
                    )
                    continue
                if materialized.oracle_leaks or not materialized.case["metadata"][
                    "negative_regions"
                ]:
                    warning_rows.append(
                        {
                            "case_id": scenario.case_id,
                            "expert": expert,
                            "split": split,
                            "status": "rejected_by_pilot_audit",
                            "warnings": materialized.warnings + materialized.oracle_leaks,
                        }
                    )
                    continue
                selected_cases.append(materialized.case)
                review_rows.extend(materialized.review_rows)
                if materialized.warnings:
                    warning_rows.append(
                        {
                            "case_id": scenario.case_id,
                            "expert": expert,
                            "split": split,
                            "status": "selected_with_warnings",
                            "warnings": materialized.warnings,
                        }
                    )
                chosen += 1
                if progress:
                    progress(
                        f"pilot: {expert}/{split} {chosen}/{required}; total={len(selected_cases)}"
                    )
                if chosen >= required:
                    break
            if chosen < required:
                raise ValueError(
                    f"Pilot needs {required} usable {expert}/{split} cases, found {chosen}"
                )

    selected_cases.sort(key=lambda item: (item["split"], item["case_id"]))
    for split in SPLITS:
        _write_jsonl(
            destination / f"cases_{split}.jsonl",
            [item for item in selected_cases if item["split"] == split],
        )
    _write_jsonl(destination / "cases_all.jsonl", selected_cases)
    _write_review_csv(destination / "pilot_review.csv", review_rows)
    _write_jsonl(destination / "label_warnings.jsonl", warning_rows)
    audit = audit_pilot(selected_cases, config, mapping)
    write_json(destination / "pilot_audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError(
            "Pilot audit failed: " + "; ".join(str(item) for item in audit["problems"])
        )
    summary = {
        "schema_version": config.schema_version,
        "seed": config.seed,
        "mapping_version": mapping.schema_version,
        "mapping_hash": mapping.mapping_hash,
        "case_count": len(selected_cases),
        "split_quota_per_expert": split_quota,
        "family_distribution": dict(
            sorted(
                Counter(
                    str(item["metadata"]["expected_expert"])
                    for item in selected_cases
                ).items()
            )
        ),
        "split_distribution": dict(
            sorted(Counter(str(item["split"]) for item in selected_cases).items())
        ),
        "warning_record_count": len(warning_rows),
        "audit": audit,
    }
    write_json(destination / "pilot_summary.json", summary)
    return summary


def _allocate_split_quota(
    total: int, *, train: float, dev: float, test: float
) -> dict[str, int]:
    fractions = {"train": train, "dev": dev, "test": test}
    raw = {name: total * fraction for name, fraction in fractions.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remainder = total - sum(counts.values())
    for name in sorted(
        fractions,
        key=lambda item: (-(raw[item] - counts[item]), -fractions[item], item),
    )[:remainder]:
        counts[name] += 1
    return counts


def _diverse_order(
    scenarios: list[IndexedScenario], seed: int, expert: str, split: str
) -> list[IndexedScenario]:
    buckets: dict[str, list[IndexedScenario]] = defaultdict(list)
    for item in scenarios:
        buckets[item.cwe].append(item)
    for cwe, rows in buckets.items():
        rows.sort(
            key=lambda item: (
                hashlib.sha256(
                    f"{seed}:{expert}:{split}:{cwe}:{item.case_id}".encode("utf-8")
                ).hexdigest(),
                item.case_id,
            )
        )
    output: list[IndexedScenario] = []
    seen_groups: set[str] = set()
    deferred: list[IndexedScenario] = []
    keys = sorted(buckets)
    while keys:
        remaining: list[str] = []
        for key in keys:
            if buckets[key]:
                item = buckets[key].pop(0)
                if item.leakage_group in seen_groups:
                    deferred.append(item)
                else:
                    output.append(item)
                    seen_groups.add(item.leakage_group)
            if buckets[key]:
                remaining.append(key)
        keys = remaining
    return output + deferred


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    destination = require_within(path, EVALUATION_ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(destination)


def _write_review_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "case_id",
        "package_id",
        "package_path",
        "split",
        "cwe",
        "expert",
        "template_group",
        "leakage_group",
        "raw_file",
        "raw_function",
        "raw_line_start",
        "raw_line_end",
        "virtual_file",
        "virtual_function",
        "label",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
