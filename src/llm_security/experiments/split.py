from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ..arvo import split_cases_by_project
from ..models import ProjectCase
from .io import sorted_jsonl, write_json


@dataclass(slots=True)
class FrozenProjectSplit:
    seed: int
    cases: dict[str, list[ProjectCase]]
    manifest: dict[str, object]


def freeze_project_split(
    cases: list[ProjectCase], output_directory: str | Path, *, seed: int = 2026
) -> FrozenProjectSplit:
    """Create the one project-disjoint split used by every Phase 2E backend."""
    splits = split_cases_by_project(cases, seed=seed)
    _assert_project_disjoint(splits)
    destination = Path(output_directory)
    manifest: dict[str, object] = {
        "seed": seed,
        "split_policy": "project-disjoint 70/15/15",
        "case_count": sum(len(items) for items in splits.values()),
        "splits": {},
    }
    for split_name in ("train", "dev", "test"):
        split_cases = sorted(splits[split_name], key=lambda case: case.case_id)
        splits[split_name] = split_cases
        projects = sorted({case.project_id for case in split_cases})
        distribution = Counter(
            family.value
            for case in split_cases
            for truth in case.ground_truth
            for family in truth.experts
        )
        manifest["splits"][split_name] = {  # type: ignore[index]
            "case_count": len(split_cases),
            "project_count": len(projects),
            "projects": projects,
            "family_distribution": dict(sorted(distribution.items())),
        }
        sorted_jsonl(
            split_cases,
            destination / f"cases_{split_name}.jsonl",
            key=lambda case: case.case_id,
        )
    write_json(manifest, destination / "split_manifest.json")
    return FrozenProjectSplit(seed=seed, cases=splits, manifest=manifest)


def _assert_project_disjoint(splits: dict[str, list[ProjectCase]]) -> None:
    project_sets = {
        name: {case.project_id for case in cases} for name, cases in splits.items()
    }
    names = ("train", "dev", "test")
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = project_sets[left] & project_sets[right]
            if overlap:
                raise ValueError(
                    f"Project leakage between {left} and {right}: {sorted(overlap)}"
                )
