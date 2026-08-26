from __future__ import annotations

import hashlib
import json
import tomllib
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..config import ACTIVE_EXPERTS, EvaluationConfig
from ..jsonl import iter_jsonl, write_jsonl
from ..juliet.indexer import build_index, iter_scenarios
from ..juliet.splitter import SPLITS, assign_frozen_splits
from ..paths import EVALUATION_ROOT, require_within, write_json
from ..schemas import IndexedScenario


@dataclass(frozen=True, slots=True)
class SplitSamplingPolicy:
    mode: str
    quotas: dict[str, int]


@dataclass(frozen=True, slots=True)
class CohortSamplingConfig:
    schema_version: str
    seed: int
    splits: dict[str, SplitSamplingPolicy]
    config_hash: str


def ensure_frozen_index(
    config: EvaluationConfig,
    mapping,
    *,
    progress=None,
) -> dict[str, object]:
    """Create the deterministic index/split only when the frozen files are absent."""
    if config.paths.index.is_file() and config.paths.split_manifest.is_file():
        return {
            "status": "reused",
            "index": str(config.paths.index),
            "split_manifest": str(config.paths.split_manifest),
        }
    index = build_index(config, mapping, rebuild=False, progress=progress)
    split = assign_frozen_splits(config, mapping)
    return {"status": "created", "index": index, "split": split}


def load_cohort_config(path: str | Path) -> CohortSamplingConfig:
    source = Path(path).resolve()
    if not source.is_file():
        raise ValueError(f"Cohort config does not exist: {source}")
    raw_bytes = source.read_bytes()
    raw = tomllib.loads(raw_bytes.decode("utf-8"))
    policies: dict[str, SplitSamplingPolicy] = {}
    for split in SPLITS:
        block = raw.get(split)
        if not isinstance(block, dict):
            raise ValueError(f"Cohort config is missing [{split}]")
        mode = str(block.get("mode", "quota")).strip().lower()
        if mode not in {"quota", "all"}:
            raise ValueError(f"Unsupported cohort mode for {split}: {mode}")
        quotas = {
            expert: int(block.get(expert, 0))
            for expert in ACTIVE_EXPERTS
        }
        if any(value < 0 for value in quotas.values()):
            raise ValueError(f"Negative Expert quota in [{split}]")
        if mode == "quota" and any(value == 0 for value in quotas.values()):
            raise ValueError(
                f"Quota mode must explicitly preserve every Expert in [{split}]"
            )
        policies[split] = SplitSamplingPolicy(mode=mode, quotas=quotas)
    return CohortSamplingConfig(
        schema_version=str(raw.get("schema_version", "juliet-cohort-v1")),
        seed=int(raw.get("seed", 2026)),
        splits=policies,
        config_hash=hashlib.sha256(raw_bytes).hexdigest(),
    )


def build_cohort_manifests(
    config: EvaluationConfig,
    cohort: CohortSamplingConfig,
    *,
    output_directory: str | Path,
) -> dict[str, object]:
    """Select deterministic leakage-safe cohorts without consulting candidates."""
    if config.seed != cohort.seed:
        raise ValueError(
            f"Evaluation/cohort seeds differ: {config.seed} != {cohort.seed}"
        )
    destination = require_within(output_directory, EVALUATION_ROOT)
    destination.mkdir(parents=True, exist_ok=True)
    scenarios_by_split: dict[str, list[IndexedScenario]] = defaultdict(list)
    for scenario in iter_scenarios(config.paths.index, include_duplicates=False):
        if not scenario.split or not scenario.leakage_group:
            raise ValueError("Frozen split assignments are missing; run index first")
        scenarios_by_split[scenario.split].append(scenario)

    selected_by_split: dict[str, list[tuple[IndexedScenario, str, int]]] = {}
    summary_splits: dict[str, object] = {}
    for split in SPLITS:
        available = scenarios_by_split[split]
        policy = cohort.splits[split]
        selected_by_expert: dict[str, list[tuple[IndexedScenario, str, int]]] = {}
        for expert in ACTIVE_EXPERTS:
            rows = [item for item in available if item.expert == expert]
            ranked = _diversity_rank(rows, cohort.seed, split, expert)
            quota = len(rows) if policy.mode == "all" else policy.quotas[expert]
            if quota > len(rows):
                raise ValueError(
                    f"{split}/{expert} requests {quota} cases but only {len(rows)} exist"
                )
            selected_by_expert[expert] = ranked[:quota]
        merged = _weighted_fair_merge(selected_by_expert)
        selected_by_split[split] = merged

        cwe_available = Counter((item.expert, item.cwe) for item in available)
        cwe_selected = Counter((item.expert, item.cwe) for item, _, _ in merged)
        manifest_rows = []
        for global_rank, (scenario, phase, expert_rank) in enumerate(merged, start=1):
            available_count = cwe_available[(scenario.expert, scenario.cwe)]
            selected_count = cwe_selected[(scenario.expert, scenario.cwe)]
            manifest_rows.append(
                {
                    "case_id": scenario.case_id,
                    "split": split,
                    "expert": scenario.expert,
                    "cwe": scenario.cwe,
                    "template_group": scenario.template_group,
                    "leakage_group": scenario.leakage_group,
                    "exact_hash": scenario.exact_hash,
                    "canonical_hash": scenario.canonical_hash,
                    "selection_phase": phase,
                    "expert_rank": expert_rank,
                    "selection_order": global_rank,
                    "stratum_available": available_count,
                    "stratum_selected": selected_count,
                    "selection_fraction": selected_count / available_count,
                    "sampling_weight": available_count / selected_count,
                }
            )
        manifest_path = write_jsonl(
            destination / f"cohort_{split}.jsonl", manifest_rows
        )
        expert_available = Counter(item.expert for item in available)
        expert_selected = Counter(item.expert for item, _, _ in merged)
        cwe_coverage = {
            expert: {
                "available": len({item.cwe for item in available if item.expert == expert}),
                "selected": len({item.cwe for item, _, _ in merged if item.expert == expert}),
            }
            for expert in ACTIVE_EXPERTS
        }
        group_coverage = {
            expert: {
                "available": len({
                    item.leakage_group for item in available if item.expert == expert
                }),
                "selected": len({
                    item.leakage_group for item, _, _ in merged
                    if item.expert == expert
                }),
            }
            for expert in ACTIVE_EXPERTS
        }
        summary_splits[split] = {
            "mode": policy.mode,
            "available_cases": len(available),
            "selected_cases": len(merged),
            "manifest": str(manifest_path),
            "manifest_sha256": _file_sha256(manifest_path),
            "expert_available": dict(sorted(expert_available.items())),
            "expert_selected": dict(sorted(expert_selected.items())),
            "cwe_coverage": cwe_coverage,
            "leakage_group_coverage": group_coverage,
        }

    audit = _audit_selection(scenarios_by_split, selected_by_split, cohort)
    if not audit["passed"]:
        raise RuntimeError("Cohort audit failed: " + "; ".join(audit["problems"]))
    report = {
        "schema_version": cohort.schema_version,
        "seed": cohort.seed,
        "cohort_config_hash": cohort.config_hash,
        "frozen_split_manifest": str(config.paths.split_manifest),
        "frozen_split_manifest_sha256": _file_sha256(config.paths.split_manifest),
        "selection_inputs": "Expert/CWE/leakage-group metadata only; no candidate data",
        "splits": summary_splits,
        "audit": audit,
    }
    write_json(destination / "cohort_summary.json", report)
    return report


def manifest_case_ids(path: str | Path) -> list[str]:
    return [str(row["case_id"]) for row in iter_jsonl(path)]


def _diversity_rank(
    scenarios: list[IndexedScenario], seed: int, split: str, expert: str
) -> list[tuple[IndexedScenario, str, int]]:
    by_cwe_group: dict[str, dict[str, list[IndexedScenario]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for item in scenarios:
        group = item.leakage_group or (
            f"{item.template_group}:{item.canonical_hash}:{item.exact_hash}"
        )
        by_cwe_group[item.cwe][group].append(item)
    for cwe, groups in by_cwe_group.items():
        for group, rows in groups.items():
            rows.sort(key=lambda item: (
                _stable_hash(seed, split, expert, cwe, group, item.case_id),
                item.case_id,
            ))

    cwes = sorted(by_cwe_group, key=_cwe_key)
    group_queues: dict[str, deque[str]] = {}
    for cwe in cwes:
        groups = sorted(
            by_cwe_group[cwe],
            key=lambda group: (_stable_hash(seed, split, expert, cwe, group), group),
        )
        group_queues[cwe] = deque(groups)

    ranked: list[tuple[IndexedScenario, str, int]] = []
    # Phase one and two: every CWE, then every unseen leakage family, before
    # admitting a second variation from any family.
    first_cwe_round = True
    while any(group_queues.values()):
        for cwe in cwes:
            if not group_queues[cwe]:
                continue
            group = group_queues[cwe].popleft()
            item = by_cwe_group[cwe][group].pop(0)
            phase = "new_cwe" if first_cwe_round else "new_leakage_group"
            ranked.append((item, phase, len(ranked) + 1))
        first_cwe_round = False

    # Phase three: round-robin across CWE and family for remaining variations.
    remaining_groups = {
        cwe: deque(sorted(
            (group for group, rows in groups.items() if rows),
            key=lambda group: (_stable_hash(seed, split, expert, cwe, group), group),
        ))
        for cwe, groups in by_cwe_group.items()
    }
    while any(remaining_groups.values()):
        added = False
        for cwe in cwes:
            queue = remaining_groups[cwe]
            if not queue:
                continue
            group = queue.popleft()
            rows = by_cwe_group[cwe][group]
            ranked.append((rows.pop(0), "family_variation", len(ranked) + 1))
            added = True
            if rows:
                queue.append(group)
        if not added:
            break
    if len(ranked) != len(scenarios):
        raise RuntimeError(
            f"Diversity ranking lost cases for {split}/{expert}: "
            f"{len(ranked)} != {len(scenarios)}"
        )
    return ranked


def _weighted_fair_merge(
    selected: dict[str, list[tuple[IndexedScenario, str, int]]]
) -> list[tuple[IndexedScenario, str, int]]:
    positions = {expert: 0 for expert in ACTIVE_EXPERTS}
    output: list[tuple[IndexedScenario, str, int]] = []
    while True:
        available = [
            expert
            for expert in ACTIVE_EXPERTS
            if positions[expert] < len(selected[expert])
        ]
        if not available:
            return output
        expert = min(
            available,
            key=lambda item: (
                positions[item] / max(1, len(selected[item])),
                ACTIVE_EXPERTS.index(item),
            ),
        )
        output.append(selected[expert][positions[expert]])
        positions[expert] += 1


def _audit_selection(
    available: dict[str, list[IndexedScenario]],
    selected: dict[str, list[tuple[IndexedScenario, str, int]]],
    cohort: CohortSamplingConfig,
) -> dict[str, object]:
    problems: list[str] = []
    all_case_ids: set[str] = set()
    groups_by_split: dict[str, set[str]] = {}
    for split in SPLITS:
        rows = [item for item, _, _ in selected[split]]
        case_ids = [item.case_id for item in rows]
        if len(case_ids) != len(set(case_ids)):
            problems.append(f"duplicate selected case in {split}")
        if any(item.split != split for item in rows):
            problems.append(f"foreign frozen split member selected in {split}")
        overlap = all_case_ids & set(case_ids)
        if overlap:
            problems.append(f"case overlap across splits: {sorted(overlap)[:3]}")
        all_case_ids.update(case_ids)
        groups_by_split[split] = {item.leakage_group for item in rows}
        policy = cohort.splits[split]
        for expert in ACTIVE_EXPERTS:
            chosen = sum(item.expert == expert for item in rows)
            available_count = sum(item.expert == expert for item in available[split])
            expected = available_count if policy.mode == "all" else policy.quotas[expert]
            if chosen != expected:
                problems.append(
                    f"{split}/{expert} selected {chosen}, expected {expected}"
                )
            if expert == "concurrency_toctou" and chosen != available_count:
                problems.append(f"E6 was subsampled in {split}")
    for index, first in enumerate(SPLITS):
        for second in SPLITS[index + 1 :]:
            overlap = groups_by_split[first] & groups_by_split[second]
            if overlap:
                problems.append(
                    f"leakage groups overlap for {first}/{second}: {sorted(overlap)[:3]}"
                )
    return {
        "passed": not problems,
        "problems": problems,
        "selected_case_count": len(all_case_ids),
        "candidate_dependent_selection": False,
    }


def _stable_hash(*values: object) -> str:
    return hashlib.sha256(":".join(map(str, values)).encode("utf-8")).hexdigest()


def _cwe_key(value: str) -> tuple[int, str]:
    digits = "".join(character for character in value if character.isdigit())
    return (int(digits) if digits else 10**9, value)


def _file_sha256(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        return ""
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
