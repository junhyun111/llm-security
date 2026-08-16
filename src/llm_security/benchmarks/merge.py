from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from ..datasets import (
    iter_cases_jsonl,
    load_router_samples_jsonl,
    router_sample_to_dict,
)
from ..models import to_dict


def merge_case_split_directories(
    input_directories: Iterable[str | Path],
    output_directory: str | Path,
) -> dict[str, object]:
    """Merge frozen splits while rejecting project and case leakage."""
    sources = [Path(item) for item in input_directories]
    if len(sources) < 2:
        raise ValueError("At least two split directories are required")
    destination = Path(output_directory)
    project_split: dict[str, str] = {}
    seen_cases: set[str] = set()
    destination.mkdir(parents=True, exist_ok=True)
    temporary_files = {
        split: destination / f".cases_{split}.jsonl.tmp"
        for split in ("train", "dev", "test")
    }
    all_temporary = destination / ".cases_all.jsonl.tmp"
    handles = {
        split: path.open("w", encoding="utf-8", newline="\n")
        for split, path in temporary_files.items()
    }
    all_handle = all_temporary.open("w", encoding="utf-8", newline="\n")
    case_counts: Counter[str] = Counter()
    family_counts = {split: Counter() for split in handles}
    split_projects = {split: set() for split in handles}
    try:
        for source in sources:
            for split in ("train", "dev", "test"):
                path = source / f"cases_{split}.jsonl"
                if not path.is_file():
                    raise ValueError(f"Missing frozen split: {path}")
                for case in iter_cases_jsonl(path):
                    previous = project_split.setdefault(case.project_id, split)
                    if previous != split:
                        raise ValueError(
                            f"Project leakage for {case.project_id}: {previous} and {split}"
                        )
                    if case.case_id in seen_cases:
                        raise ValueError(
                            f"Duplicate case_id while merging: {case.case_id}"
                        )
                    seen_cases.add(case.case_id)
                    case.split = split
                    line = json.dumps(
                        to_dict(case), ensure_ascii=False, sort_keys=True
                    ) + "\n"
                    handles[split].write(line)
                    all_handle.write(line)
                    case_counts[split] += 1
                    split_projects[split].add(case.project_id)
                    family_counts[split].update(
                        family.value
                        for truth in case.ground_truth
                        for family in truth.experts
                    )
    except BaseException:
        for handle in handles.values():
            handle.close()
        all_handle.close()
        for path in (*temporary_files.values(), all_temporary):
            path.unlink(missing_ok=True)
        raise
    else:
        for handle in handles.values():
            handle.close()
        all_handle.close()

    manifest: dict[str, object] = {
        "split_policy": "source-frozen, project-disjoint",
        "sources": [str(path) for path in sources],
        "case_count": len(seen_cases),
        "splits": {},
    }
    for split in ("train", "dev", "test"):
        temporary_files[split].replace(destination / f"cases_{split}.jsonl")
        manifest["splits"][split] = {  # type: ignore[index]
            "case_count": case_counts[split],
            "project_count": len(split_projects[split]),
            "projects": sorted(split_projects[split]),
            "family_distribution": dict(sorted(family_counts[split].items())),
        }
    all_temporary.replace(destination / "cases_all.jsonl")
    (destination / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def merge_router_split_directories(
    input_directories: Iterable[str | Path],
    output_directory: str | Path,
) -> dict[str, object]:
    """Merge compact Router JSONL while preserving every source split."""
    sources = [Path(path) for path in input_directories]
    if len(sources) < 2:
        raise ValueError("At least two Router split directories are required")
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    project_split: dict[str, str] = {}
    seen_candidates: set[str] = set()
    schemas: set[str] = set()
    split_projects = {split: set() for split in ("train", "dev", "test")}
    split_counts: Counter[str] = Counter()
    label_counts = {
        split: Counter() for split in ("train", "dev", "test")
    }
    temporary_files = {
        split: destination / f".router_{split}.jsonl.tmp"
        for split in ("train", "dev", "test")
    }
    handles = {
        split: path.open("w", encoding="utf-8", newline="\n")
        for split, path in temporary_files.items()
    }
    try:
        for source in sources:
            for split in ("train", "dev", "test"):
                path = source / f"router_{split}.jsonl"
                if not path.is_file():
                    raise ValueError(f"Missing Router split: {path}")
                for sample in load_router_samples_jsonl(path):
                    candidate = sample.candidate
                    previous = project_split.setdefault(candidate.project_id, split)
                    if previous != split:
                        raise ValueError(
                            f"Router project leakage for {candidate.project_id}: "
                            f"{previous} and {split}"
                        )
                    if candidate.candidate_id in seen_candidates:
                        raise ValueError(
                            "Duplicate candidate_id while merging Router data: "
                            + candidate.candidate_id
                        )
                    seen_candidates.add(candidate.candidate_id)
                    schemas.add(candidate.feature_schema_version)
                    handles[split].write(
                        json.dumps(
                            router_sample_to_dict(sample),
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    split_counts[split] += 1
                    split_projects[split].add(candidate.project_id)
                    label_counts[split].update(label.value for label in sample.labels)
    except BaseException:
        for handle in handles.values():
            handle.close()
        for path in temporary_files.values():
            path.unlink(missing_ok=True)
        raise
    else:
        for handle in handles.values():
            handle.close()

    if len(schemas) != 1:
        for path in temporary_files.values():
            path.unlink(missing_ok=True)
        raise ValueError(
            "Merged Router data must use one feature schema; got: "
            + ", ".join(sorted(schemas))
        )
    for split, temporary in temporary_files.items():
        temporary.replace(destination / f"router_{split}.jsonl")
    manifest: dict[str, object] = {
        "split_policy": "source-frozen, project-disjoint",
        "sources": [str(path) for path in sources],
        "feature_schema": next(iter(schemas)),
        "sample_count": len(seen_candidates),
        "splits": {
            split: {
                "sample_count": split_counts[split],
                "project_count": len(split_projects[split]),
                "projects": sorted(split_projects[split]),
                "label_distribution": dict(sorted(label_counts[split].items())),
            }
            for split in ("train", "dev", "test")
        },
    }
    (destination / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
