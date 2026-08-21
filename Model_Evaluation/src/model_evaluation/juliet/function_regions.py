from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from ..adapters.llm_security import frontend
from ..config import EvaluationConfig, ExpertMapping
from ..schemas import IndexedScenario, Region
from .sanitizer import SanitizedSources, find_oracle_leaks, sanitize_source_files


@dataclass(slots=True)
class MaterializedScenario:
    case: dict
    review_rows: list[dict[str, object]]
    warnings: list[str]
    oracle_leaks: list[str]


def materialize_scenario(
    scenario: IndexedScenario,
    config: EvaluationConfig,
    mapping: ExpertMapping,
) -> MaterializedScenario:
    raw_sources: dict[str, str] = {}
    warnings = list(scenario.warnings)
    for artifact in scenario.source_artifacts:
        raw_path = Path(artifact.raw_path).resolve()
        package = Path(scenario.package_path).resolve()
        try:
            raw_path.relative_to(package)
        except ValueError as error:
            raise ValueError(f"Indexed artifact escapes package: {raw_path}") from error
        source = raw_path.read_text(encoding="utf-8", errors="replace")
        observed_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if observed_hash != artifact.sha256:
            raise ValueError(f"Raw source changed after indexing: {raw_path}")
        raw_sources[artifact.raw_uri] = source

    parser = frontend(
        max_source_bytes=config.max_source_bytes,
        parse_timeout_ms=config.parse_timeout_ms,
    )
    functions_by_file = {
        file: parser.parse_file(file, source)
        for file, source in sorted(raw_sources.items())
    }
    sanitized = sanitize_source_files(raw_sources, case_id=scenario.case_id)
    positive_regions: list[Region] = []
    review_rows: list[dict[str, object]] = []
    for region in scenario.positive_regions:
        function = _function_at(functions_by_file.get(region.file, []), region.line_start)
        if function is None:
            warnings.append(
                f"SARIF location outside parsed function: {region.file}:{region.line_start}"
            )
            raw_function = ""
            virtual_function = ""
        else:
            raw_function = function.name
            virtual_function = sanitized.alias_text(function.name)
        virtual_file = sanitized.raw_to_virtual.get(region.file)
        if virtual_file is None:
            raise ValueError(f"SARIF result references unindexed source: {region.file}")
        positive = Region(
            file=virtual_file,
            function=virtual_function,
            line_start=region.line_start,
            line_end=region.line_end,
            label="vulnerable",
            source="sarif",
        )
        positive_regions.append(positive)
        review_rows.append(
            {
                "case_id": scenario.case_id,
                "package_id": scenario.package_id,
                "package_path": scenario.package_path,
                "split": scenario.split,
                "cwe": scenario.cwe,
                "expert": scenario.expert,
                "template_group": scenario.template_group,
                "leakage_group": scenario.leakage_group,
                "raw_file": region.file,
                "raw_function": raw_function,
                "raw_line_start": region.line_start,
                "raw_line_end": region.line_end,
                "virtual_file": virtual_file,
                "virtual_function": virtual_function,
                "label": "vulnerable",
            }
        )

    negative_regions: list[Region] = []
    for raw_file, functions in functions_by_file.items():
        for function in functions:
            if "good" not in function.name.lower() or _dispatcher_only(function):
                continue
            virtual_file = sanitized.raw_to_virtual[raw_file]
            negative = Region(
                file=virtual_file,
                function=sanitized.alias_text(function.name),
                line_start=function.line_start,
                line_end=function.line_end,
                label="target_cwe_safe",
                source="juliet_good_function",
            )
            if any(_overlaps(negative, positive) for positive in positive_regions):
                warnings.append(
                    f"ignored good-labelled function overlapping SARIF GT: {raw_file}:{function.name}"
                )
                continue
            negative_regions.append(negative)
            review_rows.append(
                {
                    "case_id": scenario.case_id,
                    "package_id": scenario.package_id,
                    "package_path": scenario.package_path,
                    "split": scenario.split,
                    "cwe": scenario.cwe,
                    "expert": scenario.expert,
                    "template_group": scenario.template_group,
                    "leakage_group": scenario.leakage_group,
                    "raw_file": raw_file,
                    "raw_function": function.name,
                    "raw_line_start": function.line_start,
                    "raw_line_end": function.line_end,
                    "virtual_file": virtual_file,
                    "virtual_function": negative.function,
                    "label": "target_cwe_safe",
                }
            )
    if not negative_regions:
        warnings.append("no non-wrapper Juliet good function was found")

    truths = [
        {
            "truth_id": "truth-"
            + hashlib.sha256(
                f"{scenario.case_id}:{item.file}:{item.line_start}:{index}".encode("utf-8")
            ).hexdigest()[:20],
            "file": item.file,
            "function": item.function,
            "line_start": item.line_start,
            "line_end": item.line_end,
            "experts": [scenario.expert],
            "cwes": [scenario.cwe],
        }
        for index, item in enumerate(positive_regions, start=1)
    ]
    case = {
        "case_id": scenario.case_id,
        "project_id": scenario.leakage_group,
        "source_files": sanitized.source_files,
        "split": scenario.split,
        "vulnerable_revision": None,
        "fixed_revision": None,
        "ground_truth": truths,
        "metadata": {
            "source": "NIST SARD Juliet Test Suite for C/C++",
            "benchmark_kind": "synthetic_mixed",
            "schema_version": config.schema_version,
            "mapping_version": mapping.schema_version,
            "mapping_hash": mapping.mapping_hash,
            "target_cwe": scenario.cwe,
            "expected_expert": scenario.expert,
            "sard_package_id": scenario.package_id,
            "raw_package_path": scenario.package_path,
            "template_group": scenario.template_group,
            "leakage_group": scenario.leakage_group,
            "exact_hash": scenario.exact_hash,
            "canonical_hash": scenario.canonical_hash,
            "negative_regions": [
                {
                    "file": item.file,
                    "function": item.function,
                    "line_start": item.line_start,
                    "line_end": item.line_end,
                    "label": item.label,
                    "source": item.source,
                }
                for item in negative_regions
            ],
            "raw_to_virtual": sanitized.raw_to_virtual,
            "warnings": warnings,
        },
    }
    return MaterializedScenario(
        case=case,
        review_rows=review_rows,
        warnings=warnings,
        oracle_leaks=find_oracle_leaks(sanitized.source_files),
    )


def _function_at(functions: list, line: int):
    containing = [
        function
        for function in functions
        if function.line_start <= line <= function.line_end
    ]
    return min(
        containing,
        key=lambda item: (item.line_end - item.line_start, item.line_start),
        default=None,
    )


def _dispatcher_only(function) -> bool:
    if not function.calls:
        return False
    return (
        not function.assignments
        and not function.memory_accesses
        and all("good" in call.callee.lower() for call in function.calls)
    )


def _overlaps(first: Region, second: Region) -> bool:
    return (
        first.file == second.file
        and first.line_start <= second.line_end
        and first.line_end >= second.line_start
    )

