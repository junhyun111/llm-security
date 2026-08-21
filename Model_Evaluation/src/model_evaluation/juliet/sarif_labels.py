from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ..config import ExpertMapping


SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx"}


@dataclass(frozen=True, slots=True)
class ManifestRegion:
    cwe: str
    expert: str
    uri: str
    line_start: int
    line_end: int


@dataclass(frozen=True, slots=True)
class ManifestScenario:
    package_id: str
    package_path: Path
    language: str
    state: str
    cwe: str
    expert: str
    source_uris: tuple[str, ...]
    positive_regions: tuple[ManifestRegion, ...]


def parse_sarif_package(
    package_directory: Path, mapping: ExpertMapping
) -> tuple[list[ManifestScenario], list[str]]:
    manifest = package_directory / "manifest.sarif"
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    scenarios: list[ManifestScenario] = []
    warnings: list[str] = []
    for run_index, run in enumerate(raw.get("runs", [])):
        properties = dict(run.get("properties", {}))
        artifacts = list(run.get("artifacts", []))
        artifact_uris = [_artifact_uri(item) for item in artifacts]
        source_uris = tuple(
            uri
            for uri in artifact_uris
            if uri and _is_testcase_source(uri)
        )
        if not source_uris:
            warnings.append(f"run {run_index}: no testcase C/C++ source artifacts")
            continue
        by_label: dict[tuple[str, str], list[ManifestRegion]] = defaultdict(list)
        for result in run.get("results", []):
            cwe = _result_cwe(result)
            expert = mapping.expert_for(cwe)
            if not cwe or expert is None:
                continue
            for location in result.get("locations", []):
                physical = location.get("physicalLocation", {})
                artifact = physical.get("artifactLocation", {})
                uri = str(artifact.get("uri", "")).replace("\\", "/")
                if not uri and "index" in artifact:
                    index = int(artifact["index"])
                    if 0 <= index < len(artifact_uris):
                        uri = artifact_uris[index]
                region = physical.get("region", {})
                start = int(region.get("startLine", 0) or 0)
                end = int(region.get("endLine", start) or start)
                if not uri or start < 1:
                    warnings.append(f"run {run_index}: {cwe} has incomplete location")
                    continue
                by_label[(cwe, expert)].append(
                    ManifestRegion(cwe, expert, uri, start, max(start, end))
                )
        for (cwe, expert), regions in sorted(by_label.items()):
            scenarios.append(
                ManifestScenario(
                    package_id=str(properties.get("id", package_directory.name)),
                    package_path=package_directory,
                    language=str(properties.get("language", "unknown")),
                    state=str(properties.get("state", "unknown")),
                    cwe=cwe,
                    expert=expert,
                    source_uris=source_uris,
                    positive_regions=tuple(regions),
                )
            )
    return scenarios, warnings


def resolve_artifact(package_directory: Path, uri: str) -> Path:
    relative = Path(*PurePosixPath(uri.replace("\\", "/")).parts)
    resolved = (package_directory / relative).resolve()
    try:
        resolved.relative_to(package_directory.resolve())
    except ValueError as error:
        raise ValueError(f"SARIF artifact escapes package directory: {uri}") from error
    if not resolved.is_file():
        raise ValueError(f"SARIF artifact does not exist: {resolved}")
    return resolved


def _artifact_uri(artifact: dict) -> str:
    return str(artifact.get("location", {}).get("uri", "")).replace("\\", "/")


def _is_testcase_source(uri: str) -> bool:
    pure = PurePosixPath(uri.lower())
    return pure.suffix in SOURCE_SUFFIXES and "testcases" in pure.parts


def _result_cwe(result: dict) -> str:
    candidates = [result.get("ruleId", "")]
    candidates.extend(item.get("id", "") for item in result.get("taxa", []))
    for candidate in candidates:
        digits = "".join(character for character in str(candidate) if character.isdigit())
        if digits:
            return f"CWE-{int(digits)}"
    return ""

