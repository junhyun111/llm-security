from __future__ import annotations

import hashlib
import json
import os
import posixpath
import random
import re
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .datasets import RouterSample, write_cases_jsonl, write_router_samples_jsonl
from .models import Candidate, ExpertFamily, GroundTruth, ProjectCase
from .static_analysis import LightweightStaticAnalyzer


SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp"}
_HUNK_RE = re.compile(
    r"^@@\s+-(?P<old_start>\d+)(?:,(?P<old_count>\d+))?\s+"
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))?\s+@@(?P<context>.*)$"
)
_FRAME_RE = re.compile(
    r"#\d+[^\r\n]*?\s(?:in\s+)?(?P<function>.*?)\s+"
    r"(?P<file>/src/[^:\r\n]+):(?P<line>\d+)(?::\d+)?"
)


@dataclass(slots=True)
class ArvoRecord:
    local_id: int
    project: str
    crash_type: str
    crash_output: str
    severity: str
    report: str
    fix_commit: str
    repo_addr: str
    patch_url: str
    sanitizer: str
    fuzz_target: str
    fuzz_engine: str
    language: str


@dataclass(slots=True)
class PatchHunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    context: str
    lines: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FilePatch:
    old_path: str
    new_path: str
    hunks: list[PatchHunk]


@dataclass(slots=True)
class StackFrame:
    function: str
    file: str
    line: int


class GitHubClient:
    def __init__(
        self,
        token: str | None = None,
        timeout_seconds: float = 60.0,
        cache_directory: str | Path | None = None,
    ) -> None:
        self.token = token
        self.timeout_seconds = timeout_seconds
        self.cache_directory = Path(cache_directory) if cache_directory else None
        self._memory_cache: dict[str, bytes] = {}

    def commit(self, owner: str, repository: str, revision: str) -> dict[str, Any]:
        url = f"https://api.github.com/repos/{owner}/{repository}/commits/{revision}"
        return json.loads(self._request(url).decode("utf-8"))

    def commit_patch(self, owner: str, repository: str, revision: str) -> str:
        url = f"https://github.com/{owner}/{repository}/commit/{revision}.patch"
        return self._request(url).decode("utf-8", errors="replace")

    def raw_file(
        self, owner: str, repository: str, revision: str, file_path: str
    ) -> str:
        quoted_path = "/".join(urllib.parse.quote(part) for part in file_path.split("/"))
        url = (
            f"https://raw.githubusercontent.com/{owner}/{repository}/"
            f"{revision}/{quoted_path}"
        )
        return self._request(url).decode("utf-8", errors="replace")

    def _request(self, url: str) -> bytes:
        if url in self._memory_cache:
            return self._memory_cache[url]
        cache_path = self._cache_path(url)
        if cache_path and cache_path.exists():
            payload = cache_path.read_bytes()
            self._memory_cache[url] = payload
            return payload
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "llm-security-arvo-preparer",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = response.read()
        self._memory_cache[url] = payload
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(payload)
        return payload

    def _cache_path(self, url: str) -> Path | None:
        if self.cache_directory is None:
            return None
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_directory / digest[:2] / digest


def prepare_arvo_cases(
    database: str | Path,
    output: str | Path,
    *,
    count: int = 30,
    case_ids: Iterable[int] = (),
    github_token: str | None = None,
    unique_projects: bool = False,
    balanced: bool = True,
    seed: int = 2026,
    cache_directory: str | Path = "data/arvo/cache/github",
    require_routable: bool = True,
) -> list[ProjectCase]:
    if count < 1:
        raise ValueError("count must be positive")
    records = load_arvo_records(database, case_ids=case_ids)
    if balanced and not tuple(case_ids):
        records = balanced_records(records, seed=seed)
    client = GitHubClient(
        token=github_token or os.getenv("GITHUB_TOKEN"),
        cache_directory=cache_directory,
    )
    analyzer = LightweightStaticAnalyzer(max_candidates=500)
    cases: list[ProjectCase] = []
    selected_projects: set[str] = set()
    failures: list[str] = []
    for record in records:
        if len(cases) >= count:
            break
        if unique_projects and record.project in selected_projects:
            continue
        try:
            case = build_case(record, client=client)
        except urllib.error.HTTPError as error:
            if error.code in {403, 429}:
                raise RuntimeError(
                    "GitHub API rate limit reached. Set GITHUB_TOKEN or retry after reset."
                ) from error
            failures.append(f"ARVO-{record.local_id}: HTTP {error.code}")
            continue
        except (ValueError, KeyError, urllib.error.URLError, TimeoutError) as error:
            failures.append(f"ARVO-{record.local_id}: {error}")
            continue
        if case is None:
            continue
        if require_routable and not truth_candidates(case, analyzer):
            failures.append(f"ARVO-{record.local_id}: patch location has no static candidate")
            continue
        cases.append(case)
        selected_projects.add(case.project_id)
    if len(cases) < count:
        detail = "; ".join(failures[:5])
        raise RuntimeError(
            f"Prepared only {len(cases)} of {count} requested ARVO cases. {detail}"
        )
    write_cases_jsonl(cases, output)
    if failures:
        print(f"Skipped {len(failures)} records before collecting {len(cases)} cases.")
    return cases


def prepare_arvo_training_dataset(
    cases: list[ProjectCase],
    output_directory: str | Path,
    *,
    seed: int = 2026,
) -> dict[str, Any]:
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    split_cases = split_cases_by_project(cases, seed=seed)
    analyzer = LightweightStaticAnalyzer(max_candidates=500)
    manifest: dict[str, Any] = {
        "source": "ARVO v3.0.0",
        "seed": seed,
        "split_policy": "project-disjoint 70/15/15",
        "splits": {},
    }
    for split in ("train", "dev", "test"):
        current_cases = split_cases[split]
        samples = router_samples_from_cases(current_cases, analyzer=analyzer)
        write_cases_jsonl(current_cases, destination / f"cases_{split}.jsonl")
        write_router_samples_jsonl(samples, destination / f"router_{split}.jsonl")
        families = Counter(
            family.value for sample in samples for family in sample.labels
        )
        manifest["splits"][split] = {
            "case_count": len(current_cases),
            "project_count": len({case.project_id for case in current_cases}),
            "router_sample_count": len(samples),
            "projects": sorted({case.project_id for case in current_cases}),
            "families": dict(sorted(families.items())),
        }
    _validate_project_disjointness(split_cases)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def load_arvo_records(
    database: str | Path, *, case_ids: Iterable[int] = ()
) -> list[ArvoRecord]:
    path = Path(database).resolve()
    if not path.exists():
        raise FileNotFoundError(f"ARVO database not found: {path}")
    selected_ids = tuple(int(value) for value in case_ids)
    uri = f"file:{path.as_posix()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        where = (
            "reproduced = 1 AND patch_located = 1 AND submodule_bug = 0 "
            "AND language IN ('c', 'c++') AND repo_addr LIKE 'https://github.com/%'"
        )
        parameters: tuple[Any, ...] = ()
        if selected_ids:
            placeholders = ",".join("?" for _ in selected_ids)
            where += f" AND localId IN ({placeholders})"
            parameters = selected_ids
        rows = connection.execute(
            f"SELECT * FROM arvo WHERE {where} ORDER BY localId", parameters
        ).fetchall()
    finally:
        connection.close()
    return [
        ArvoRecord(
            local_id=int(row["localId"]),
            project=str(row["project"]),
            crash_type=str(row["crash_type"] or ""),
            crash_output=str(row["crash_output"] or ""),
            severity=str(row["severity"] or ""),
            report=str(row["report"] or ""),
            fix_commit=str(row["fix_commit"] or ""),
            repo_addr=str(row["repo_addr"] or ""),
            patch_url=str(row["patch_url"] or ""),
            sanitizer=str(row["sanitizer"] or ""),
            fuzz_target=str(row["fuzz_target"] or ""),
            fuzz_engine=str(row["fuzz_engine"] or ""),
            language=str(row["language"] or ""),
        )
        for row in rows
    ]


def build_case(record: ArvoRecord, *, client: GitHubClient) -> ProjectCase | None:
    owner, repository = parse_github_repository(record.repo_addr)
    fix_commits = [item for item in record.fix_commit.split() if item]
    if not fix_commits:
        raise ValueError("ARVO record has no fix commit")
    primary_fix_commit = fix_commits[0]
    commit_patch = client.commit_patch(owner, repository, primary_fix_commit)
    file_patches = parse_git_patch(commit_patch)
    frames = parse_stack_frames(record.crash_output, record.project)
    family, cwes = classify_crash(record.crash_type)
    for file_patch in ordered_file_patches(file_patches, frames):
        file_path = file_patch.old_path
        try:
            fixed_source = client.raw_file(
                owner, repository, primary_fix_commit, file_patch.new_path
            )
        except urllib.error.HTTPError as error:
            if error.code == 404:
                continue
            raise
        vulnerable_source = reverse_apply_patch(fixed_source, file_patch)
        hunks = file_patch.hunks
        matching_frames = [frame for frame in frames if paths_match(file_path, frame.file)]
        if matching_frames:
            frame = matching_frames[0]
            hunk = min(hunks, key=lambda item: abs(item.old_start - frame.line))
            target_line = frame.line
            function = normalize_function_name(frame.function)
            location_source = "crash-stack+patch"
        else:
            hunk = hunks[0]
            target_line = max(1, hunk.old_start)
            function = function_from_hunk_context(hunk.context)
            location_source = "patch-hunk"
        line_count = max(1, vulnerable_source.count("\n") + 1)
        if target_line > line_count:
            target_line = min(max(1, hunk.old_start), line_count)
            location_source = "patch-hunk-clamped"
        return ProjectCase(
            case_id=f"arvo-{record.local_id}",
            project_id=record.project,
            source_files={file_path: vulnerable_source},
            split="unassigned",
            vulnerable_revision=f"{primary_fix_commit}^",
            fixed_revision=fix_commits[-1],
            ground_truth=[
                GroundTruth(
                    truth_id=f"arvo-{record.local_id}-truth-1",
                    file=file_path,
                    function=function or "<unknown>",
                    line_start=target_line,
                    line_end=target_line,
                    experts=[family],
                    cwes=cwes,
                )
            ],
            metadata={
                "arvo_local_id": record.local_id,
                "source": "ARVO v3.0.0",
                "repo_addr": record.repo_addr,
                "report": record.report,
                "patch_url": record.patch_url,
                "severity": record.severity,
                "sanitizer": record.sanitizer,
                "crash_type": record.crash_type,
                "fuzz_target": record.fuzz_target,
                "fuzz_engine": record.fuzz_engine,
                "language": record.language,
                "ground_truth_source": location_source,
                "patch_old_start": hunk.old_start,
                "patch_old_count": hunk.old_count,
                "fix_commits": fix_commits,
            },
        )
    return None


def split_cases_by_project(
    cases: list[ProjectCase], *, seed: int = 2026
) -> dict[str, list[ProjectCase]]:
    by_project: dict[str, list[ProjectCase]] = defaultdict(list)
    for case in cases:
        by_project[case.project_id].append(case)
    projects = sorted(by_project)
    if len(projects) < 3:
        raise ValueError("At least three projects are required for project-level splits")
    random.Random(seed).shuffle(projects)
    test_count = max(1, round(len(projects) * 0.15))
    dev_count = max(1, round(len(projects) * 0.15))
    train_count = len(projects) - dev_count - test_count
    if train_count < 1:
        raise ValueError("Project split leaves no training projects")
    project_split = {
        **{project: "train" for project in projects[:train_count]},
        **{
            project: "dev"
            for project in projects[train_count : train_count + dev_count]
        },
        **{project: "test" for project in projects[train_count + dev_count :]},
    }
    split_cases: dict[str, list[ProjectCase]] = {"train": [], "dev": [], "test": []}
    for project, project_cases in by_project.items():
        split = project_split[project]
        for case in project_cases:
            case.split = split
            split_cases[split].append(case)
    return split_cases


def router_samples_from_cases(
    cases: Iterable[ProjectCase], *, analyzer: LightweightStaticAnalyzer
) -> list[RouterSample]:
    samples: list[RouterSample] = []
    for case in cases:
        candidates = analyzer.analyze(case)
        by_candidate: dict[str, tuple[Candidate, set[ExpertFamily]]] = {}
        for truth in case.ground_truth:
            matching = [
                candidate
                for candidate in candidates
                if candidate.file == truth.file
                and candidate.line_start <= truth.line_end
                and candidate.line_end >= truth.line_start
            ]
            if not matching:
                continue
            candidate = min(
                matching,
                key=lambda item: (
                    item.line_end - item.line_start,
                    -item.static_score,
                    item.candidate_id,
                ),
            )
            stored_candidate, labels = by_candidate.setdefault(
                candidate.candidate_id, (candidate, set())
            )
            labels.update(truth.experts)
            by_candidate[candidate.candidate_id] = (stored_candidate, labels)
        samples.extend(
            RouterSample(candidate=candidate, labels=sorted(labels, key=lambda x: x.value))
            for candidate, labels in by_candidate.values()
        )
    return samples


def truth_candidates(
    case: ProjectCase, analyzer: LightweightStaticAnalyzer
) -> list[Candidate]:
    candidates = analyzer.analyze(case)
    return [
        candidate
        for candidate in candidates
        if any(
            candidate.file == truth.file
            and candidate.line_start <= truth.line_end
            and candidate.line_end >= truth.line_start
            for truth in case.ground_truth
        )
    ]


def balanced_records(records: list[ArvoRecord], *, seed: int) -> list[ArvoRecord]:
    grouped: dict[ExpertFamily, list[ArvoRecord]] = defaultdict(list)
    for record in records:
        family, _ = classify_crash(record.crash_type)
        grouped[family].append(record)
    rng = random.Random(seed)
    for family_records in grouped.values():
        rng.shuffle(family_records)
    ordered: list[ArvoRecord] = []
    active = [family for family in ExpertFamily if grouped.get(family)]
    while active:
        next_active: list[ExpertFamily] = []
        for family in active:
            if grouped[family]:
                ordered.append(grouped[family].pop())
            if grouped[family]:
                next_active.append(family)
        active = next_active
    return ordered


def parse_github_repository(repo_addr: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(repo_addr)
    if parsed.hostname not in {"github.com", "www.github.com"}:
        raise ValueError(f"not a GitHub repository: {repo_addr}")
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 2:
        raise ValueError(f"invalid GitHub repository URL: {repo_addr}")
    repository = parts[1][:-4] if parts[1].endswith(".git") else parts[1]
    return parts[0], repository


def parse_patch_hunks(patch: str) -> list[PatchHunk]:
    hunks: list[PatchHunk] = []
    current: PatchHunk | None = None
    for line in patch.splitlines(keepends=True):
        match = _HUNK_RE.match(line.rstrip("\r\n"))
        if match:
            current = PatchHunk(
                old_start=int(match.group("old_start")),
                old_count=int(match.group("old_count") or "1"),
                new_start=int(match.group("new_start")),
                new_count=int(match.group("new_count") or "1"),
                context=match.group("context").strip(),
            )
            hunks.append(current)
            continue
        if current is not None and line[:1] in {" ", "+", "-", "\\"}:
            current.lines.append(line)
    return hunks


def parse_git_patch(patch: str) -> list[FilePatch]:
    """Parse changed files and unified-diff hunks from a git-format patch."""
    file_patches: list[FilePatch] = []
    old_path = ""
    new_path = ""
    body: list[str] = []

    def finish_file() -> None:
        nonlocal old_path, new_path, body
        if old_path and new_path and old_path != "/dev/null" and new_path != "/dev/null":
            hunks = parse_patch_hunks("".join(body))
            if hunks:
                file_patches.append(FilePatch(old_path, new_path, hunks))
        old_path = ""
        new_path = ""
        body = []

    for line in patch.splitlines(keepends=True):
        if line.startswith("diff --git "):
            finish_file()
        elif line.startswith("--- "):
            old_path = _parse_patch_path(line[4:])
        elif line.startswith("+++ "):
            new_path = _parse_patch_path(line[4:])
        elif old_path or new_path:
            body.append(line)
    finish_file()
    return file_patches


def _parse_patch_path(value: str) -> str:
    path = value.rstrip("\r\n").split("\t", 1)[0]
    if path.startswith('"') and path.endswith('"'):
        path = bytes(path[1:-1], "utf-8").decode("unicode_escape")
    if path.startswith(("a/", "b/")):
        path = path[2:]
    return path


def ordered_file_patches(
    file_patches: list[FilePatch], frames: list[StackFrame]
) -> list[FilePatch]:
    eligible = [
        item
        for item in file_patches
        if Path(item.old_path).suffix.lower() in SOURCE_SUFFIXES and item.hunks
    ]

    def priority(item: FilePatch) -> tuple[int, str]:
        frame_match = any(paths_match(item.old_path, frame.file) for frame in frames)
        return (0 if frame_match else 1, item.old_path)

    return sorted(eligible, key=priority)


def reverse_apply_patch(fixed_source: str, file_patch: FilePatch) -> str:
    """Reconstruct a vulnerable file by reverse-applying its fix hunks."""
    fixed_lines = fixed_source.splitlines(keepends=True)
    output: list[str] = []
    cursor = 0
    for hunk in sorted(file_patch.hunks, key=lambda item: item.new_start):
        meaningful = [line for line in hunk.lines if not line.startswith("\\")]
        new_tokens = [line[1:] for line in meaningful if line[:1] in {" ", "+"}]
        expected = max(cursor, max(0, hunk.new_start - 1))
        position = _locate_hunk(fixed_lines, new_tokens, expected, cursor)
        if position is None:
            raise ValueError(
                f"cannot reverse-apply hunk at fixed line {hunk.new_start} "
                f"for {file_patch.new_path}"
            )
        output.extend(fixed_lines[cursor:position])
        consumed = position
        for line in meaningful:
            marker, content = line[:1], line[1:]
            if marker == " ":
                output.append(fixed_lines[consumed])
                consumed += 1
            elif marker == "+":
                consumed += 1
            elif marker == "-":
                output.append(content)
        cursor = consumed
    output.extend(fixed_lines[cursor:])
    return "".join(output)


def _locate_hunk(
    lines: list[str], tokens: list[str], expected: int, minimum: int
) -> int | None:
    if not tokens:
        return min(max(expected, minimum), len(lines))

    def matches(start: int) -> bool:
        end = start + len(tokens)
        if start < minimum or end > len(lines):
            return False
        return all(
            _line_without_ending(actual) == _line_without_ending(wanted)
            for actual, wanted in zip(lines[start:end], tokens)
        )

    if matches(expected):
        return expected
    last_start = len(lines) - len(tokens)
    candidates = sorted(
        range(minimum, max(minimum, last_start) + 1),
        key=lambda index: (abs(index - expected), index),
    )
    return next((index for index in candidates if matches(index)), None)


def _line_without_ending(value: str) -> str:
    return value.rstrip("\r\n")


def first_old_hunk_line(patch: str) -> int | None:
    hunks = parse_patch_hunks(patch)
    return hunks[0].old_start if hunks else None


def parse_stack_frames(crash_output: str, project: str) -> list[StackFrame]:
    return [
        StackFrame(
            function=match.group("function").strip(),
            file=normalize_frame_path(match.group("file"), project),
            line=int(match.group("line")),
        )
        for match in _FRAME_RE.finditer(crash_output)
    ]


def classify_crash(crash_type: str) -> tuple[ExpertFamily, list[str]]:
    normalized = crash_type.lower()
    if "double-free" in normalized:
        return ExpertFamily.LIFETIME_RESOURCE, ["CWE-415"]
    if "use-after" in normalized:
        return ExpertFamily.LIFETIME_RESOURCE, ["CWE-416"]
    if "memory-leak" in normalized or "resource-leak" in normalized:
        return ExpertFamily.LIFETIME_RESOURCE, ["CWE-401"]
    if "integer-overflow" in normalized or "integer overflow" in normalized:
        return ExpertFamily.INTEGER_SIZE_TYPE, ["CWE-190"]
    if "data-race" in normalized or "deadlock" in normalized:
        return ExpertFamily.CONCURRENCY_TOCTOU, ["CWE-362"]
    if any(
        marker in normalized
        for marker in ("buffer-overflow", "buffer-underflow", "out-of-bounds")
    ):
        cwe = "CWE-787" if "write" in normalized else "CWE-125"
        return ExpertFamily.MEMORY_BOUNDS, [cwe]
    if "null-dereference" in normalized:
        return ExpertFamily.CONTROL_STATE_ERROR, ["CWE-476"]
    if "uninitialized" in normalized:
        return ExpertFamily.CONTROL_STATE_ERROR, ["CWE-457"]
    return ExpertFamily.CONTROL_STATE_ERROR, []


def normalize_frame_path(frame_path: str, project: str) -> str:
    normalized = posixpath.normpath(frame_path.replace("\\", "/"))
    prefixes = (f"/src/{project}/", "/src/")
    for prefix in prefixes:
        if normalized.startswith(prefix):
            return normalized[len(prefix) :]
    return normalized.lstrip("/")


def paths_match(left: str, right: str) -> bool:
    normalized_left = posixpath.normpath(left.replace("\\", "/")).lstrip("/")
    normalized_right = posixpath.normpath(right.replace("\\", "/")).lstrip("/")
    return (
        normalized_left == normalized_right
        or normalized_left.endswith("/" + normalized_right)
        or normalized_right.endswith("/" + normalized_left)
    )


def normalize_function_name(function: str) -> str:
    prefix = function.split("(", 1)[0].strip()
    if "::" in prefix:
        prefix = prefix.rsplit("::", 1)[-1]
    return prefix.split()[-1] if prefix else ""


def function_from_hunk_context(context: str) -> str:
    if not context:
        return ""
    matches = list(re.finditer(r"([A-Za-z_~]\w*)\s*\(", context))
    return matches[-1].group(1) if matches else ""


def _validate_project_disjointness(
    split_cases: dict[str, list[ProjectCase]],
) -> None:
    project_sets = {
        split: {case.project_id for case in cases}
        for split, cases in split_cases.items()
    }
    if project_sets["train"] & project_sets["dev"]:
        raise RuntimeError("train/dev project leakage detected")
    if project_sets["train"] & project_sets["test"]:
        raise RuntimeError("train/test project leakage detected")
    if project_sets["dev"] & project_sets["test"]:
        raise RuntimeError("dev/test project leakage detected")
