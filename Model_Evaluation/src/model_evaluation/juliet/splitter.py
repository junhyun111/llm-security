from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..config import EvaluationConfig, ExpertMapping
from ..paths import write_json
from ..schemas import IndexedScenario
from .indexer import index_summary, iter_scenarios


SPLITS = ("train", "dev", "test")


@dataclass(slots=True)
class _Cluster:
    group_id: str
    expert: str
    case_ids: list[str]
    cwes: set[str]
    weight: int


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, first: str, second: str) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            smaller, larger = sorted((first_root, second_root))
            self.parent[larger] = smaller


def assign_frozen_splits(
    config: EvaluationConfig, mapping: ExpertMapping
) -> dict[str, object]:
    scenarios = list(iter_scenarios(config.paths.index, include_duplicates=False))
    if not scenarios:
        raise ValueError("Juliet index has no supported non-duplicate scenarios")
    by_id = {item.case_id: item for item in scenarios}
    union = _UnionFind(by_id)
    _union_shared(scenarios, union, key=lambda item: item.template_group)
    _union_shared(
        scenarios,
        union,
        key=lambda item: f"{item.expert}:{item.canonical_hash}",
    )

    members: dict[str, list[IndexedScenario]] = defaultdict(list)
    for scenario in scenarios:
        members[union.find(scenario.case_id)].append(scenario)
    clusters: list[_Cluster] = []
    for rows in members.values():
        experts = {item.expert for item in rows}
        if len(experts) != 1:
            raise ValueError(
                "Leakage cluster crosses Expert labels: " + ", ".join(sorted(experts))
            )
        case_ids = sorted(item.case_id for item in rows)
        seed = "\0".join(case_ids)
        clusters.append(
            _Cluster(
                group_id="group-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20],
                expert=next(iter(experts)),
                case_ids=case_ids,
                cwes={item.cwe for item in rows},
                weight=len(rows),
            )
        )

    assignment: dict[str, str] = {}
    fractions = {
        "train": config.train_fraction,
        "dev": config.dev_fraction,
        "test": config.test_fraction,
    }
    for expert in mapping.experts:
        family_clusters = [item for item in clusters if item.expert == expert]
        if len(family_clusters) < 3:
            raise ValueError(
                f"Expert {expert} has only {len(family_clusters)} leakage groups; "
                "at least three are required"
            )
        ordered = _interleave_cwes(family_clusters, config.seed, expert)
        targets = {
            split: sum(item.weight for item in family_clusters) * fraction
            for split, fraction in fractions.items()
        }
        observed = Counter()
        for index, cluster in enumerate(ordered):
            if index < len(SPLITS):
                split = SPLITS[index]
            else:
                split = max(
                    SPLITS,
                    key=lambda name: (
                        (targets[name] - observed[name]) / max(targets[name], 1.0),
                        fractions[name],
                        -SPLITS.index(name),
                    ),
                )
            assignment[cluster.group_id] = split
            observed[split] += cluster.weight

    case_to_group = {
        case_id: cluster.group_id
        for cluster in clusters
        for case_id in cluster.case_ids
    }
    connection = sqlite3.connect(config.paths.index)
    try:
        for case_id, group_id in case_to_group.items():
            connection.execute(
                "UPDATE scenarios SET leakage_group = ?, split = ? WHERE case_id = ?",
                (group_id, assignment[group_id], case_id),
            )
        duplicates = list(
            connection.execute(
                "SELECT case_id, duplicate_of FROM scenarios WHERE duplicate_of != ''"
            )
        )
        for case_id, duplicate_of in duplicates:
            original = connection.execute(
                "SELECT leakage_group, split FROM scenarios WHERE case_id = ?",
                (duplicate_of,),
            ).fetchone()
            if original:
                connection.execute(
                    "UPDATE scenarios SET leakage_group = ?, split = ? WHERE case_id = ?",
                    (original[0], original[1], case_id),
                )
        connection.commit()
    finally:
        connection.close()

    refreshed = list(iter_scenarios(config.paths.index, include_duplicates=False))
    split_counts = Counter(item.split for item in refreshed)
    family_distribution: dict[str, Counter[str]] = {
        split: Counter() for split in SPLITS
    }
    cwe_distribution: dict[str, Counter[str]] = {
        split: Counter() for split in SPLITS
    }
    projects: dict[str, set[str]] = {split: set() for split in SPLITS}
    for item in refreshed:
        family_distribution[item.split][item.expert] += 1
        cwe_distribution[item.split][item.cwe] += 1
        projects[item.split].add(item.leakage_group)
    leakage_audit = audit_index_splits(refreshed)
    metadata = index_summary(config.paths.index)
    manifest: dict[str, object] = {
        "schema_version": config.schema_version,
        "seed": config.seed,
        "policy": "Expert-balanced template-and-canonical-hash-disjoint 70/15/15",
        "mapping_version": mapping.schema_version,
        "mapping_hash": mapping.mapping_hash,
        "dataset_fingerprint": metadata.get("dataset_fingerprint", ""),
        "case_count": len(refreshed),
        "leakage_group_count": len(clusters),
        "split_assignment_hash": hashlib.sha256(
            json.dumps(assignment, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "splits": {
            split: {
                "case_count": split_counts[split],
                "project_count": len(projects[split]),
                "family_distribution": dict(sorted(family_distribution[split].items())),
                "cwe_distribution": dict(sorted(cwe_distribution[split].items())),
            }
            for split in SPLITS
        },
        "leakage_audit": leakage_audit,
    }
    write_json(config.paths.split_manifest, manifest)
    return manifest


def audit_index_splits(scenarios: list[IndexedScenario]) -> dict[str, object]:
    problems: list[str] = []
    for attribute in ("leakage_group", "template_group", "exact_hash", "canonical_hash"):
        values: dict[str, set[str]] = defaultdict(set)
        for item in scenarios:
            values[str(getattr(item, attribute))].add(item.split)
        overlaps = sorted(key for key, splits in values.items() if len(splits) > 1)
        if overlaps:
            problems.append(f"{attribute} crosses splits: {', '.join(overlaps[:5])}")
    return {"passed": not problems, "problems": problems}


def _union_shared(
    scenarios: list[IndexedScenario],
    union: _UnionFind,
    *,
    key,
) -> None:
    first_by_key: dict[str, str] = {}
    for scenario in scenarios:
        value = key(scenario)
        previous = first_by_key.setdefault(value, scenario.case_id)
        union.union(previous, scenario.case_id)


def _interleave_cwes(
    clusters: list[_Cluster], seed: int, expert: str
) -> list[_Cluster]:
    buckets: dict[str, list[_Cluster]] = defaultdict(list)
    for cluster in clusters:
        cwe_key = "+".join(sorted(cluster.cwes))
        buckets[cwe_key].append(cluster)
    for cwe, rows in buckets.items():
        rows.sort(
            key=lambda item: hashlib.sha256(
                f"{seed}:{expert}:{cwe}:{item.group_id}".encode("utf-8")
            ).hexdigest()
        )
    output: list[_Cluster] = []
    keys = sorted(buckets)
    while keys:
        remaining: list[str] = []
        for key in keys:
            if buckets[key]:
                output.append(buckets[key].pop(0))
            if buckets[key]:
                remaining.append(key)
        keys = remaining
    return output

