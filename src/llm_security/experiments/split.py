from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..arvo import split_cases_by_project
from ..datasets import iter_cases_jsonl
from ..models import ProjectCase, to_dict
from .io import sorted_jsonl, write_json


@dataclass(slots=True)
class FrozenProjectSplit:
    seed: int
    cases: dict[str, list[ProjectCase]]
    manifest: dict[str, object]


@dataclass(slots=True)
class FrozenProjectFiles:
    seed: int
    files: dict[str, Path]
    manifest: dict[str, object]


def load_frozen_project_files(
    input_directory: str | Path,
    *,
    seed: int = 2026,
) -> FrozenProjectFiles:
    """Load already assigned train/dev/test files without reshuffling projects."""
    source = Path(input_directory)
    manifest_path = source / "split_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing frozen split manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split_items = manifest["splits"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid frozen split manifest: {manifest_path}") from error
    files = {
        split: source / f"cases_{split}.jsonl"
        for split in ("train", "dev", "test")
    }
    for split, path in files.items():
        if not path.is_file():
            raise ValueError(f"Missing frozen {split} split: {path}")
        try:
            item = split_items[split]
            count = int(item["case_count"])
            projects = {str(value) for value in item["projects"]}
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Frozen split manifest lacks {split} counts/projects"
            ) from error
        if count < 1 or not projects:
            raise ValueError(f"Frozen split {split} must contain cases and projects")
    project_sets = {
        split: {str(value) for value in split_items[split]["projects"]}
        for split in files
    }
    for index, left in enumerate(("train", "dev", "test")):
        for right in ("train", "dev", "test")[index + 1 :]:
            overlap = project_sets[left] & project_sets[right]
            if overlap:
                raise ValueError(
                    f"Project leakage between frozen {left}/{right}: "
                    + ", ".join(sorted(overlap)[:10])
                )
    return FrozenProjectFiles(
        seed=int(manifest.get("seed", seed)),
        files=files,
        manifest=manifest,
    )


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


def freeze_project_split_jsonl(
    cases_path: str | Path,
    output_directory: str | Path,
    *,
    seed: int = 2026,
    progress: Callable[[str], None] | None = None,
) -> FrozenProjectFiles:
    """Freeze a split with two streaming passes and no corpus-sized list."""
    source = Path(cases_path)
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    project_case_counts: Counter[str] = Counter()
    project_family_counts: dict[str, Counter[str]] = {}
    case_count = 0
    for case_count, case in enumerate(iter_cases_jsonl(source), start=1):
        project_case_counts[case.project_id] += 1
        distribution = project_family_counts.setdefault(case.project_id, Counter())
        distribution.update(
            family.value
            for truth in case.ground_truth
            for family in truth.experts
        )
        if progress is not None and case_count % 250 == 0:
            progress(f"      split scan: {case_count} cases")
    if case_count == 0:
        raise ValueError(f"No cases found in {source}")

    project_to_split = _project_assignment(list(project_case_counts), seed=seed)
    files = {
        split: destination / f"cases_{split}.jsonl"
        for split in ("train", "dev", "test")
    }
    split_counts = Counter()
    handles = {
        split: path.open("w", encoding="utf-8", newline="\n")
        for split, path in files.items()
    }
    try:
        for index, case in enumerate(iter_cases_jsonl(source), start=1):
            split = project_to_split[case.project_id]
            case.split = split
            handles[split].write(
                json.dumps(to_dict(case), ensure_ascii=False, sort_keys=True) + "\n"
            )
            split_counts[split] += 1
            if progress is not None and index % 250 == 0:
                progress(f"      split write: {index}/{case_count} cases")
    finally:
        for handle in handles.values():
            handle.close()

    manifest: dict[str, object] = {
        "seed": seed,
        "split_policy": "project-disjoint 70/15/15",
        "case_count": case_count,
        "streaming": True,
        "splits": {},
    }
    for split in ("train", "dev", "test"):
        projects = sorted(
            project for project, assigned in project_to_split.items() if assigned == split
        )
        family_distribution: Counter[str] = Counter()
        for project in projects:
            family_distribution.update(project_family_counts[project])
        manifest["splits"][split] = {  # type: ignore[index]
            "case_count": split_counts[split],
            "project_count": len(projects),
            "projects": projects,
            "family_distribution": dict(sorted(family_distribution.items())),
        }
    write_json(manifest, destination / "split_manifest.json")
    return FrozenProjectFiles(seed=seed, files=files, manifest=manifest)


def _project_assignment(projects: list[str], *, seed: int) -> dict[str, str]:
    ordered = sorted(projects)
    if len(ordered) < 3:
        raise ValueError("At least three projects are required for project-level splits")
    random.Random(seed).shuffle(ordered)
    test_count = max(1, round(len(ordered) * 0.15))
    dev_count = max(1, round(len(ordered) * 0.15))
    train_count = len(ordered) - dev_count - test_count
    if train_count < 1:
        raise ValueError("Project split leaves no training projects")
    return {
        **{project: "train" for project in ordered[:train_count]},
        **{
            project: "dev"
            for project in ordered[train_count : train_count + dev_count]
        },
        **{project: "test" for project in ordered[train_count + dev_count :]},
    }


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
