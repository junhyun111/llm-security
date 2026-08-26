from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable

from ..config import ACTIVE_EXPERTS, EvaluationConfig, ExpertMapping
from ..jsonl import write_jsonl
from ..juliet.function_regions import materialize_scenario
from ..juliet.indexer import iter_scenarios
from ..juliet.splitter import SPLITS
from ..paths import EVALUATION_ROOT, require_within, write_json
from ..schemas import IndexedScenario
from .select_cohort import manifest_case_ids


def materialize_dataset(
    config: EvaluationConfig,
    mapping: ExpertMapping,
    *,
    output_directory: str | Path,
    splits: Iterable[str] = SPLITS,
    limits: dict[str, int] | None = None,
    selection_manifests: dict[str, str | Path] | None = None,
    reuse_existing: bool = True,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Create leakage-safe case JSONL files from the frozen Juliet index.

    A split limit of 0 means every indexed scenario in that split. Limited
    cohorts are deterministically balanced over Expert and CWE and therefore
    remain reproducible.
    """
    destination = require_within(output_directory, EVALUATION_ROOT)
    destination.mkdir(parents=True, exist_ok=True)
    requested = tuple(dict.fromkeys(str(item) for item in splits))
    unknown = sorted(set(requested) - set(SPLITS))
    if unknown:
        raise ValueError("Unknown splits: " + ", ".join(unknown))
    caps = {split: max(0, int((limits or {}).get(split, 0))) for split in requested}
    manifests = {
        split: Path((selection_manifests or {}).get(split, "")).resolve()
        if (selection_manifests or {}).get(split)
        else None
        for split in requested
    }
    for split, manifest in manifests.items():
        if manifest is not None:
            if not manifest.is_file():
                raise ValueError(f"Selection manifest does not exist: {manifest}")
            caps[split] = len(manifest_case_ids(manifest))
    manifest_hashes = {
        split: hashlib.sha256(manifest.read_bytes()).hexdigest()
        if manifest is not None
        else None
        for split, manifest in manifests.items()
    }
    summary_path = destination / "materialization_summary.json"
    if reuse_existing and summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        existing_splits = existing.get("splits", {})
        if (
            existing.get("schema_version") == config.schema_version
            and existing.get("mapping_hash") == mapping.mapping_hash
            and int(existing.get("seed", -1)) == config.seed
            and all(
                split in existing_splits
                and int(existing_splits[split].get("requested_limit", -1)) == caps[split]
                and existing_splits[split].get("selection_manifest")
                == (str(manifests[split]) if manifests[split] else None)
                and existing_splits[split].get("selection_manifest_sha256")
                == manifest_hashes[split]
                and Path(existing_splits[split].get("cases", "")).is_file()
                for split in requested
            )
        ):
            return existing
    scenarios_by_split: dict[str, list[IndexedScenario]] = defaultdict(list)
    for scenario in iter_scenarios(config.paths.index, include_duplicates=False):
        if scenario.split in requested:
            scenarios_by_split[scenario.split].append(scenario)

    summary: dict[str, object] = {
        "schema_version": config.schema_version,
        "mapping_hash": mapping.mapping_hash,
        "seed": config.seed,
        "splits": {},
    }
    for split in requested:
        ordered = _balanced_order(scenarios_by_split[split], config.seed, split)
        cap = caps[split]
        manifest = manifests[split]
        if manifest is not None:
            by_id = {item.case_id: item for item in ordered}
            requested_ids = manifest_case_ids(manifest)
            missing = [case_id for case_id in requested_ids if case_id not in by_id]
            if missing:
                raise ValueError(
                    f"Selection manifest contains unknown {split} cases: {missing[:5]}"
                )
            selected = [by_id[case_id] for case_id in requested_ids]
            cap = len(selected)
        else:
            selected = ordered if cap == 0 else ordered[:cap]
        failures: list[dict] = []
        expert_distribution: Counter[str] = Counter()
        cwe_distribution: Counter[str] = Counter()
        materialized_count = 0
        case_path = require_within(destination / f"cases_{split}.jsonl", EVALUATION_ROOT)
        temporary = case_path.with_suffix(case_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for index, scenario in enumerate(selected, start=1):
                try:
                    materialized = materialize_scenario(scenario, config, mapping)
                    if materialized.oracle_leaks:
                        raise ValueError(
                            "oracle leakage: " + "; ".join(materialized.oracle_leaks)
                        )
                    handle.write(
                        json.dumps(
                            materialized.case,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    materialized_count += 1
                    expert_distribution[
                        str(materialized.case["metadata"]["expected_expert"])
                    ] += 1
                    cwe_distribution[
                        str(materialized.case["metadata"]["target_cwe"])
                    ] += 1
                except (OSError, RuntimeError, ValueError) as error:
                    failures.append({"case_id": scenario.case_id, "error": str(error)})
                if progress and (
                    index == 1 or index % 100 == 0 or index == len(selected)
                ):
                    progress(f"materialize {split}: {index}/{len(selected)}")
        temporary.replace(case_path)
        failure_path = write_jsonl(destination / f"failures_{split}.jsonl", failures)
        split_summary = {
            "available_indexed_cases": len(ordered),
            "requested_limit": cap,
            "selection_manifest": str(manifest) if manifest else None,
            "selection_manifest_sha256": manifest_hashes[split],
            "attempted_cases": len(selected),
            "materialized_cases": materialized_count,
            "failure_count": len(failures),
            "cases": str(case_path),
            "failures": str(failure_path),
            "expert_distribution": dict(sorted(expert_distribution.items())),
            "cwe_distribution": dict(sorted(cwe_distribution.items())),
        }
        summary["splits"][split] = split_summary
    write_json(summary_path, summary)
    return summary


def _balanced_order(
    scenarios: list[IndexedScenario], seed: int, split: str
) -> list[IndexedScenario]:
    buckets: dict[tuple[str, str], list[IndexedScenario]] = defaultdict(list)
    for scenario in scenarios:
        buckets[(scenario.expert, scenario.cwe)].append(scenario)
    for key, rows in buckets.items():
        rows.sort(key=lambda item: (
            hashlib.sha256(
                f"{seed}:{split}:{key[0]}:{key[1]}:{item.case_id}".encode("utf-8")
            ).hexdigest(),
            item.case_id,
        ))
    expert_keys = {
        expert: sorted(key for key in buckets if key[0] == expert)
        for expert in ACTIVE_EXPERTS
    }
    output: list[IndexedScenario] = []
    while True:
        added = False
        for expert in ACTIVE_EXPERTS:
            for key in expert_keys[expert]:
                if buckets[key]:
                    output.append(buckets[key].pop(0))
                    added = True
        if not added:
            return output
