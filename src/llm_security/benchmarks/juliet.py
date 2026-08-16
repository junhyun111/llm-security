from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from ..analysis.frontend import AnalysisLimitError, TreeSitterFrontend
from ..cwe import expert_for_cwe
from ..datasets import iter_cases_jsonl
from ..models import ExpertFamily, GroundTruth, ProjectCase, to_dict


SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx"}
DEFAULT_FAMILIES = (
    ExpertFamily.INTEGER_SIZE_TYPE,
    ExpertFamily.TAINT_API_CONTRACT,
    ExpertFamily.CONCURRENCY_TOCTOU,
)
_CWE_RE = re.compile(r"CWE[_-]?(\d+)", re.I)
_FLOW_VARIANT_RE = re.compile(r"_\d{2}[a-z]?$", re.I)
_FLAW_RE = re.compile(r"\b(?:POTENTIAL\s+)?FLAW\b", re.I)


def prepare_juliet_dataset(
    source_directory: str | Path,
    output_directory: str | Path,
    *,
    families: Iterable[ExpertFamily] = DEFAULT_FAMILIES,
    seed: int = 2026,
    max_cases: int = 0,
) -> dict[str, object]:
    """Convert an unpacked Juliet C/C++ tree into project-disjoint JSONL.

    Juliet's numbered flow variants share a template group. The converter uses
    that group as ``project_id`` so near-clones cannot cross train/dev/test.
    Only functions explicitly named ``bad``/``badSink``/``badSource`` become
    vulnerable ground truth. When Juliet marks a FLAW line, that line is used
    instead of labeling the whole function.
    """
    source_root = Path(source_directory).resolve()
    destination = Path(output_directory)
    if not source_root.is_dir():
        raise ValueError(f"Juliet source directory does not exist: {source_root}")
    allowed = set(families)
    if not allowed:
        raise ValueError("At least one Juliet Expert family must be selected")

    frontend = TreeSitterFrontend(max_source_bytes=None, parse_timeout_ms=30_000)
    destination.mkdir(parents=True, exist_ok=True)
    unassigned_path = destination / ".juliet_unassigned.jsonl"
    case_count = 0
    projects: set[str] = set()
    projects_by_family: dict[ExpertFamily, set[str]] = defaultdict(set)
    seen_functions: set[str] = set()
    skipped_parse = duplicate_functions = unsupported_cwe = 0
    family_counts: Counter[str] = Counter()

    files = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES
    )
    with unassigned_path.open("w", encoding="utf-8", newline="\n") as output:
        for path in files:
            relative = path.relative_to(source_root).as_posix()
            cwe_match = _CWE_RE.search(relative)
            if not cwe_match:
                continue
            cwe = f"CWE-{int(cwe_match.group(1))}"
            family = expert_for_cwe(cwe)
            if family is None or family not in allowed:
                unsupported_cwe += 1
                continue
            source = path.read_text(encoding="utf-8", errors="replace")
            try:
                functions = frontend.parse_file(relative, source)
            except (AnalysisLimitError, ValueError):
                skipped_parse += 1
                continue
            flaw_lines = {
                index
                for index, line in enumerate(source.splitlines(), start=1)
                if _FLAW_RE.search(line)
            }
            truths: list[GroundTruth] = []
            for function in functions:
                if "bad" not in function.name.lower():
                    continue
                fingerprint = hashlib.sha256(
                    _normalize_code(function.code).encode("utf-8")
                ).hexdigest()
                if fingerprint in seen_functions:
                    duplicate_functions += 1
                    continue
                seen_functions.add(fingerprint)
                marked = sorted(
                    line
                    for line in flaw_lines
                    if function.line_start <= line <= function.line_end
                )
                spans = [(line, line) for line in marked] or [
                    (function.line_start, function.line_end)
                ]
                for index, (line_start, line_end) in enumerate(spans, start=1):
                    truth_seed = f"{relative}:{function.name}:{line_start}:{index}"
                    truths.append(
                        GroundTruth(
                            truth_id="juliet-truth-" + _short_hash(truth_seed),
                            file=relative,
                            function=function.name,
                            line_start=line_start,
                            line_end=line_end,
                            experts=[family],
                            cwes=[cwe],
                        )
                    )
            if not truths:
                continue
            template = _template_group(path.stem)
            project_id = f"juliet-{cwe.lower()}-{template.lower()}"
            case = ProjectCase(
                case_id="juliet-case-" + _short_hash(relative),
                project_id=project_id,
                source_files={relative: source},
                split="unassigned",
                ground_truth=truths,
                metadata={
                    "source": "NIST Juliet Test Suite for C/C++",
                    "benchmark_kind": "synthetic",
                    "cwe": cwe,
                    "template_group": project_id,
                    "relative_path": relative,
                },
            )
            output.write(json.dumps(to_dict(case), ensure_ascii=False) + "\n")
            case_count += 1
            projects.add(project_id)
            projects_by_family[family].add(project_id)
            family_counts[family.value] += len(truths)
            if max_cases and case_count >= max_cases:
                break

    if not case_count:
        raise ValueError(
            "No supported Juliet bad functions were found. Check the extracted "
            "source path and selected E3/E4/E6 families."
        )
    if len(projects) < 3:
        raise ValueError(
            "Juliet conversion needs at least three template groups for a "
            "project-disjoint train/dev/test split"
        )

    split_files, split_manifest = _write_stratified_splits(
        unassigned_path,
        destination,
        projects_by_family=projects_by_family,
        selected_families=allowed,
        seed=seed,
    )
    with (destination / "cases_all.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as output:
        for split in ("train", "dev", "test"):
            with split_files[split].open("r", encoding="utf-8") as source:
                shutil.copyfileobj(source, output)
    unassigned_path.unlink()
    manifest = dict(split_manifest)
    manifest["source"] = "NIST Juliet Test Suite for C/C++"
    manifest["selected_families"] = sorted(family.value for family in allowed)
    manifest["family_ground_truth_distribution"] = dict(sorted(family_counts.items()))
    manifest["duplicate_functions_removed"] = duplicate_functions
    manifest["files_skipped_on_parse"] = skipped_parse
    manifest["files_skipped_for_unsupported_cwe"] = unsupported_cwe
    (destination / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _write_stratified_splits(
    unassigned_path: Path,
    destination: Path,
    *,
    projects_by_family: dict[ExpertFamily, set[str]],
    selected_families: set[ExpertFamily],
    seed: int,
) -> tuple[dict[str, Path], dict[str, object]]:
    project_to_split: dict[str, str] = {}
    for family in sorted(selected_families, key=lambda item: item.value):
        projects = sorted(projects_by_family.get(family, set()))
        if len(projects) < 3:
            raise ValueError(
                f"Juliet family {family.value} needs at least three template "
                "projects so train/dev/test each contain positives"
            )
        family_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{family.value}".encode("utf-8")).digest()[:8],
            "big",
        )
        random.Random(family_seed).shuffle(projects)
        test_count = max(1, round(len(projects) * 0.15))
        dev_count = max(1, round(len(projects) * 0.15))
        train_count = len(projects) - dev_count - test_count
        if train_count < 1:
            raise ValueError(f"Juliet family {family.value} has no training project")
        for project in projects[:train_count]:
            project_to_split[project] = "train"
        for project in projects[train_count : train_count + dev_count]:
            project_to_split[project] = "dev"
        for project in projects[train_count + dev_count :]:
            project_to_split[project] = "test"

    files = {
        split: destination / f"cases_{split}.jsonl"
        for split in ("train", "dev", "test")
    }
    handles = {
        split: path.open("w", encoding="utf-8", newline="\n")
        for split, path in files.items()
    }
    case_counts: Counter[str] = Counter()
    family_counts: dict[str, Counter[str]] = {
        split: Counter() for split in files
    }
    split_projects: dict[str, set[str]] = {
        split: set() for split in files
    }
    try:
        for case in iter_cases_jsonl(unassigned_path):
            split = project_to_split[case.project_id]
            case.split = split
            handles[split].write(
                json.dumps(to_dict(case), ensure_ascii=False, sort_keys=True) + "\n"
            )
            case_counts[split] += 1
            split_projects[split].add(case.project_id)
            family_counts[split].update(
                family.value
                for truth in case.ground_truth
                for family in truth.experts
            )
    finally:
        for handle in handles.values():
            handle.close()

    manifest: dict[str, object] = {
        "seed": seed,
        "split_policy": "family-stratified template-project-disjoint 70/15/15",
        "case_count": sum(case_counts.values()),
        "streaming": True,
        "splits": {},
    }
    for split in ("train", "dev", "test"):
        missing = sorted(
            family.value
            for family in selected_families
            if family_counts[split][family.value] == 0
        )
        if missing:
            raise RuntimeError(
                f"Internal Juliet split error: {split} lacks " + ", ".join(missing)
            )
        manifest["splits"][split] = {  # type: ignore[index]
            "case_count": case_counts[split],
            "project_count": len(split_projects[split]),
            "projects": sorted(split_projects[split]),
            "family_distribution": dict(sorted(family_counts[split].items())),
        }
    return files, manifest


def _template_group(stem: str) -> str:
    normalized = _FLOW_VARIANT_RE.sub("", stem)
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", normalized).strip("-")


def _normalize_code(code: str) -> str:
    without_comments = re.sub(r"/\*.*?\*/|//[^\n]*", "", code, flags=re.S)
    return re.sub(r"\s+", " ", without_comments).strip()


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
