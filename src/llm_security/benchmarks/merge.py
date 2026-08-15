from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from ..datasets import load_cases_jsonl, write_cases_jsonl
from ..models import ProjectCase


def merge_case_split_directories(
    input_directories: Iterable[str | Path],
    output_directory: str | Path,
) -> dict[str, object]:
    """Merge frozen splits while rejecting project and case leakage."""
    sources = [Path(item) for item in input_directories]
    if len(sources) < 2:
        raise ValueError("At least two split directories are required")
    destination = Path(output_directory)
    merged: dict[str, list[ProjectCase]] = {name: [] for name in ("train", "dev", "test")}
    project_split: dict[str, str] = {}
    seen_cases: set[str] = set()

    for source in sources:
        for split in ("train", "dev", "test"):
            path = source / f"cases_{split}.jsonl"
            if not path.is_file():
                raise ValueError(f"Missing frozen split: {path}")
            for case in load_cases_jsonl(path):
                previous = project_split.setdefault(case.project_id, split)
                if previous != split:
                    raise ValueError(
                        f"Project leakage for {case.project_id}: {previous} and {split}"
                    )
                if case.case_id in seen_cases:
                    raise ValueError(f"Duplicate case_id while merging: {case.case_id}")
                seen_cases.add(case.case_id)
                case.split = split
                merged[split].append(case)

    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "split_policy": "source-frozen, project-disjoint",
        "sources": [str(path) for path in sources],
        "case_count": len(seen_cases),
        "splits": {},
    }
    all_cases: list[ProjectCase] = []
    for split in ("train", "dev", "test"):
        cases = sorted(merged[split], key=lambda case: case.case_id)
        all_cases.extend(cases)
        write_cases_jsonl(cases, destination / f"cases_{split}.jsonl")
        distribution = Counter(
            family.value
            for case in cases
            for truth in case.ground_truth
            for family in truth.experts
        )
        manifest["splits"][split] = {  # type: ignore[index]
            "case_count": len(cases),
            "project_count": len({case.project_id for case in cases}),
            "family_distribution": dict(sorted(distribution.items())),
        }
    write_cases_jsonl(sorted(all_cases, key=lambda case: case.case_id), destination / "cases_all.jsonl")
    (destination / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
