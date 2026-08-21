from __future__ import annotations

from collections import Counter, defaultdict

from ..config import ACTIVE_EXPERTS, EvaluationConfig, ExpertMapping
from .sanitizer import find_oracle_leaks


def audit_pilot(
    cases: list[dict], config: EvaluationConfig, mapping: ExpertMapping
) -> dict[str, object]:
    problems: list[str] = []
    family_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    positives = negatives = 0
    projects_by_split: dict[str, set[str]] = defaultdict(set)
    hashes: dict[str, dict[str, set[str]]] = {
        "exact_hash": defaultdict(set),
        "canonical_hash": defaultdict(set),
    }
    oracle_leaks: list[str] = []
    for case in cases:
        split = str(case["split"])
        metadata = case["metadata"]
        expert = str(metadata["expected_expert"])
        family_counts[expert] += 1
        split_counts[split] += 1
        projects_by_split[split].add(str(case["project_id"]))
        positives += len(case["ground_truth"])
        negatives += len(metadata.get("negative_regions", []))
        oracle_leaks.extend(
            f"{case['case_id']}:{item}" for item in find_oracle_leaks(case["source_files"])
        )
        for attribute in hashes:
            hashes[attribute][str(metadata[attribute])].add(split)
        for truth in case["ground_truth"]:
            expected = mapping.expert_for(truth["cwes"][0])
            if expected != truth["experts"][0]:
                problems.append(
                    f"{case['case_id']}: mapping mismatch {truth['cwes'][0]}"
                )
            if not _valid_region(case, truth):
                problems.append(f"{case['case_id']}: invalid positive region")
        for region in metadata.get("negative_regions", []):
            if not _valid_region(case, region):
                problems.append(f"{case['case_id']}: invalid negative region")

    for first_index, first in enumerate(("train", "dev", "test")):
        for second in ("train", "dev", "test")[first_index + 1 :]:
            overlap = projects_by_split[first] & projects_by_split[second]
            if overlap:
                problems.append(f"project overlap {first}/{second}: {sorted(overlap)[:3]}")
    for attribute, values in hashes.items():
        overlaps = [value for value, splits in values.items() if len(splits) > 1]
        if overlaps:
            problems.append(f"{attribute} crosses splits: {overlaps[:3]}")
    expected_per_family = config.pilot_per_expert
    for expert in ACTIVE_EXPERTS:
        if family_counts[expert] != expected_per_family:
            problems.append(
                f"{expert}: expected {expected_per_family}, got {family_counts[expert]}"
            )
    if oracle_leaks:
        problems.append(f"oracle leakage remains: {oracle_leaks[:5]}")
    if negatives < len(cases):
        problems.append(
            f"paired safe regions are incomplete: {negatives} regions for {len(cases)} cases"
        )
    return {
        "passed": not problems,
        "problems": problems,
        "case_count": len(cases),
        "positive_region_count": positives,
        "negative_region_count": negatives,
        "family_distribution": dict(sorted(family_counts.items())),
        "split_distribution": dict(sorted(split_counts.items())),
        "oracle_leak_count": len(oracle_leaks),
        "oracle_leak_preview": oracle_leaks[:20],
    }


def _valid_region(case: dict, region: dict) -> bool:
    source = case["source_files"].get(region["file"])
    if source is None:
        return False
    line_count = max(1, len(source.splitlines()))
    return 1 <= int(region["line_start"]) <= int(region["line_end"]) <= line_count

