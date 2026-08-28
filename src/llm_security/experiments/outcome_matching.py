from __future__ import annotations

import re
from dataclasses import dataclass

from ..cwe import cwe_categories, expert_for_cwe
from ..models import Candidate, ExpertFamily, Finding, GroundTruth
from ..datasets import UTILITY_OUTCOME_LABEL_VERSION


OUTCOME_LABEL_VERSION = UTILITY_OUTCOME_LABEL_VERSION


@dataclass(frozen=True, slots=True)
class OutcomeMatch:
    matched: bool
    location_match: bool
    evidence_valid: bool
    semantic_compatible: bool
    truth_categories: tuple[str, ...]
    finding_categories: tuple[str, ...]
    reason: str


class FindingTruthMatcher:
    """Deterministic GT matcher that avoids line-overlap-only success labels."""

    label_version = OUTCOME_LABEL_VERSION

    def evaluate(
        self,
        finding: Finding,
        truth: GroundTruth,
        candidate: Candidate,
    ) -> OutcomeMatch:
        location = _location_matches(finding, truth)
        evidence_by_id = {item.evidence_id: item for item in candidate.evidence}
        evidence_valid = bool(finding.evidence_ids) and set(
            finding.evidence_ids
        ).issubset(evidence_by_id)
        cited_kinds = {
            evidence_by_id[evidence_id].kind
            for evidence_id in finding.evidence_ids
            if evidence_id in evidence_by_id
        }
        truth_categories = _truth_categories(truth)
        finding_categories = _finding_categories(finding, cited_kinds)
        semantic = _semantically_compatible(
            truth_categories,
            finding_categories,
            finding=finding,
            cited_kinds=cited_kinds,
        )
        matched = location and evidence_valid and semantic
        failed = []
        if not location:
            failed.append("location")
        if not evidence_valid:
            failed.append("evidence")
        if not semantic:
            failed.append("semantics")
        return OutcomeMatch(
            matched=matched,
            location_match=location,
            evidence_valid=evidence_valid,
            semantic_compatible=semantic,
            truth_categories=tuple(sorted(truth_categories)),
            finding_categories=tuple(sorted(finding_categories)),
            reason="matched" if matched else "failed: " + ", ".join(failed),
        )

    def matches(
        self,
        finding: Finding,
        truth: GroundTruth,
        candidate: Candidate,
    ) -> bool:
        return self.evaluate(finding, truth, candidate).matched


_EXPERT_CATEGORIES: dict[ExpertFamily, set[str]] = {
    # E1 intentionally combines spatial and temporal memory safety.  The
    # MEMORY_SAFETY enum retains the historical ``memory_bounds`` wire value.
    ExpertFamily.MEMORY_SAFETY: {"memory_spatial", "memory_temporal"},
    ExpertFamily.LIFETIME_RESOURCE: {"memory_temporal"},
    ExpertFamily.INTEGER_SIZE_TYPE: {"integer"},
    ExpertFamily.TAINT_API_CONTRACT: {"taint_api"},
    ExpertFamily.CONTROL_STATE_ERROR: {"control_state"},
    ExpertFamily.CONCURRENCY_TOCTOU: {"concurrency"},
}


_EVIDENCE_CATEGORIES: dict[str, str] = {
    "memory_sink": "memory_spatial",
    "memory_access": "memory_spatial",
    "memory_copy": "memory_spatial",
    "memory_copy_without_guard": "memory_spatial",
    "unchecked_index": "memory_spatial",
    "release": "memory_temporal",
    "use_after_release": "memory_temporal",
    "double_release": "memory_temporal",
    "integer_arithmetic": "integer",
    "type_conversion": "integer",
    "numeric_conversion": "integer",
    "arithmetic_to_allocation": "integer",
    "arithmetic_to_memory_sink": "integer",
    "cast_to_size_sink": "integer",
    "taint_source": "taint_api",
    "taint_sink": "taint_api",
    "source_to_sink": "taint_api",
    "unsanitized_source_to_sink": "taint_api",
    "state": "control_state",
    "error_path": "control_state",
    "uninitialized_use": "control_state",
    "unchecked_nullable_dereference": "control_state",
    "unchecked_call_result": "control_state",
    "concurrency": "concurrency",
    "synchronization": "concurrency",
    "toctou": "concurrency",
    "thread_spawn": "concurrency",
    "toctou_check_use": "concurrency",
}


_ROOT_CAUSE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("memory_temporal", re.compile(r"use[- ]after[- ]free|double[- ]free|invalid[- ]free|dangling|released?\s+(?:memory|pointer)", re.I)),
    ("integer", re.compile(r"integer\s+(?:over|under)flow|wraparound|truncat|signedness|sign extension|size arithmetic", re.I)),
    ("memory_spatial", re.compile(r"out[- ]of[- ]bounds|buffer\s+(?:over|under)flow|unchecked\s+(?:index|length|copy)|bounds", re.I)),
    ("concurrency", re.compile(r"race condition|data race|toctou|time[- ]of[- ]check|deadlock|missing lock|atomicity", re.I)),
    ("taint_api", re.compile(r"unsanitized|attacker[- ]controlled|command injection|path traversal|format string|api (?:contract|precondition)|taint", re.I)),
    ("control_state", re.compile(r"unchecked return|error path|uninitialized|state machine|invalid state|null dereference|missing (?:error|state) check", re.I)),
)


_INTEGER_TO_MEMORY_EVIDENCE = {
    "arithmetic_to_allocation",
    "arithmetic_to_memory_sink",
    "cast_to_size_sink",
}
_TAINT_TO_SINK_EVIDENCE = {"source_to_sink", "unsanitized_source_to_sink"}


def _truth_categories(truth: GroundTruth) -> set[str]:
    categories = cwe_categories(truth.cwes)
    if categories:
        return categories
    return {
        category
        for expert in truth.experts
        for category in _EXPERT_CATEGORIES.get(expert, set())
    }


def _finding_categories(finding: Finding, cited_kinds: set[str]) -> set[str]:
    # An explicit CWE is the strongest semantic claim. Evidence must not rescue
    # a contradictory CWE merely because it occurs at the same location.
    categories = cwe_categories(finding.cwes)
    if categories:
        return categories
    text = " ".join((finding.title, finding.root_cause, finding.consequence))
    root_categories = {
        category for category, pattern in _ROOT_CAUSE_PATTERNS if pattern.search(text)
    }
    if root_categories:
        return root_categories
    return {
        _EVIDENCE_CATEGORIES[kind]
        for kind in cited_kinds
        if kind in _EVIDENCE_CATEGORIES
    }


def _semantically_compatible(
    truth_categories: set[str],
    finding_categories: set[str],
    *,
    finding: Finding,
    cited_kinds: set[str],
) -> bool:
    if not truth_categories or not finding_categories:
        return False
    if truth_categories & finding_categories:
        return True
    pair = (finding_categories, truth_categories)
    if (
        "integer" in pair[0]
        and "memory_spatial" in pair[1]
        and bool(cited_kinds & _INTEGER_TO_MEMORY_EVIDENCE)
    ):
        return True
    if (
        "taint_api" in pair[0]
        and bool(pair[1] & {"memory_spatial", "control_state"})
        and bool(cited_kinds & _TAINT_TO_SINK_EVIDENCE)
        and finding.source
        and finding.sink
    ):
        return True
    return False


def _location_matches(finding: Finding, truth: GroundTruth) -> bool:
    return (
        finding.file == truth.file
        and finding.line_start <= truth.line_end
        and finding.line_end >= truth.line_start
    )
