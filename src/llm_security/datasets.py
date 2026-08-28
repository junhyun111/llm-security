from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from .models import (
    Candidate,
    CweHypothesis,
    Evidence,
    ExpertAssignment,
    ExpertFamily,
    Finding,
    GroundTruth,
    ProjectCase,
    RelatedFunctionSummary,
    to_dict,
)


UTILITY_OUTCOME_LABEL_VERSION = "semantic-causal-v2"


@dataclass(slots=True)
class RouterSample:
    candidate: Candidate
    labels: list[ExpertFamily]


@dataclass(slots=True)
class UtilitySample:
    """Observed result of running one Expert x model assignment."""

    candidate: Candidate
    assignment: ExpertAssignment
    success: bool
    false_positive: bool = False
    unsupported_claims: int = 0
    cost: float = 0.0
    matched_truth_ids: list[str] = field(default_factory=list)
    ground_truth_ids: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    truth_labels_available: bool = False
    case_id: str = ""
    label_version: str = "legacy-line-v1"
    validated_true_findings: int = 0
    validated_false_findings: int = 0
    rejected_findings: int = 0
    uncertain_findings: int = 0

    def reward(
        self,
        *,
        false_positive_weight: float = 0.5,
        unsupported_weight: float = 0.25,
        cost_weight: float = 1.0,
    ) -> float:
        return (
            float(self.success)
            - false_positive_weight * float(self.false_positive)
            - unsupported_weight * float(self.unsupported_claims)
            - cost_weight * self.cost
        )


def write_cases_jsonl(cases: Iterable[ProjectCase], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(to_dict(case), ensure_ascii=False) + "\n")


def load_cases_jsonl(path: str | Path) -> list[ProjectCase]:
    return list(iter_cases_jsonl(path))


def iter_cases_jsonl(path: str | Path) -> Iterator[ProjectCase]:
    """Yield one case at a time so large source corpora stay off the heap."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                yield _case_from_raw(raw)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid case JSONL at line {line_number}: {error}") from error


def _case_from_raw(raw: dict) -> ProjectCase:
    truths = [
        GroundTruth(
            truth_id=item["truth_id"],
            file=item["file"],
            function=item["function"],
            line_start=int(item["line_start"]),
            line_end=int(item["line_end"]),
            experts=[ExpertFamily(value) for value in item.get("experts", [])],
            cwes=[str(value) for value in item.get("cwes", [])],
        )
        for item in raw.get("ground_truth", [])
    ]
    return ProjectCase(
        case_id=raw["case_id"],
        project_id=raw["project_id"],
        source_files={
            str(name): str(content) for name, content in raw["source_files"].items()
        },
        split=raw.get("split", "dev"),
        vulnerable_revision=raw.get("vulnerable_revision"),
        fixed_revision=raw.get("fixed_revision"),
        ground_truth=truths,
        metadata=dict(raw.get("metadata", {})),
    )


def case_from_dict(raw: dict) -> ProjectCase:
    """Restore a ProjectCase from the JSON-compatible project schema."""
    return _case_from_raw(raw)


def write_router_samples_jsonl(samples: Iterable[RouterSample], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(
                json.dumps(router_sample_to_dict(sample), ensure_ascii=False)
                + "\n"
            )


def load_router_samples_jsonl(path: str | Path) -> list[RouterSample]:
    samples: list[RouterSample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                samples.append(router_sample_from_dict(raw))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Invalid router sample JSONL at line {line_number}: {error}"
                ) from error
    return samples


def router_sample_to_dict(sample: RouterSample) -> dict:
    return {
        "candidate": to_dict(sample.candidate),
        "labels": [label.value for label in sample.labels],
    }


def router_sample_from_dict(raw: dict) -> RouterSample:
    return RouterSample(
        candidate=candidate_from_dict(raw["candidate"]),
        labels=[ExpertFamily(value) for value in raw["labels"]],
    )


def write_utility_samples_jsonl(
    samples: Iterable[UtilitySample], path: str | Path
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(utility_sample_to_dict(sample), ensure_ascii=False) + "\n")


def load_utility_samples_jsonl(path: str | Path) -> list[UtilitySample]:
    samples: list[UtilitySample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                assignment = raw["assignment"]
                samples.append(
                    UtilitySample(
                        candidate=_candidate_from_raw(raw["candidate"]),
                        assignment=ExpertAssignment(
                            expert=ExpertFamily(assignment["expert"]),
                            model_id=str(assignment["model_id"]),
                            prompt_version=str(
                                assignment.get("prompt_version", "expert-v2")
                            ),
                            expected_cost=float(assignment.get("expected_cost", 0.0)),
                        ),
                        success=bool(raw["success"]),
                        false_positive=bool(raw.get("false_positive", False)),
                        unsupported_claims=int(raw.get("unsupported_claims", 0)),
                        cost=float(raw.get("cost", 0.0)),
                        matched_truth_ids=[
                            str(value) for value in raw.get("matched_truth_ids", [])
                        ],
                        ground_truth_ids=[
                            str(value) for value in raw.get("ground_truth_ids", [])
                        ],
                        prompt_tokens=int(raw.get("prompt_tokens", 0)),
                        completion_tokens=int(raw.get("completion_tokens", 0)),
                        latency_seconds=float(raw.get("latency_seconds", 0.0)),
                        truth_labels_available=bool(
                            raw.get(
                                "truth_labels_available",
                                "ground_truth_ids" in raw,
                            )
                        ),
                        case_id=str(raw.get("case_id", "")),
                        label_version=str(
                            raw.get("label_version", "legacy-line-v1")
                        ),
                        validated_true_findings=int(
                            raw.get("validated_true_findings", 0)
                        ),
                        validated_false_findings=int(
                            raw.get("validated_false_findings", 0)
                        ),
                        rejected_findings=int(raw.get("rejected_findings", 0)),
                        uncertain_findings=int(raw.get("uncertain_findings", 0)),
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Invalid utility sample JSONL at line {line_number}: {error}"
                ) from error
    return samples


def utility_sample_to_dict(sample: UtilitySample) -> dict:
    """Compact outcome row: Utility training needs features, not duplicated source code."""
    candidate = sample.candidate
    return {
        "candidate": {
            "candidate_id": candidate.candidate_id,
            "project_id": candidate.project_id,
            "file": candidate.file,
            "function": candidate.function,
            "line_start": candidate.line_start,
            "line_end": candidate.line_end,
            "features": candidate.features,
            "suspicion_score": candidate.suspicion_score,
            "feature_schema_version": candidate.feature_schema_version,
            "cwe_hypotheses": to_dict(candidate.cwe_hypotheses),
        },
        "assignment": to_dict(sample.assignment),
        "success": sample.success,
        "false_positive": sample.false_positive,
        "unsupported_claims": sample.unsupported_claims,
        "cost": sample.cost,
        "matched_truth_ids": sample.matched_truth_ids,
        "ground_truth_ids": sample.ground_truth_ids,
        "prompt_tokens": sample.prompt_tokens,
        "completion_tokens": sample.completion_tokens,
        "latency_seconds": sample.latency_seconds,
        "truth_labels_available": sample.truth_labels_available,
        "case_id": sample.case_id,
        "label_version": sample.label_version,
        "validated_true_findings": sample.validated_true_findings,
        "validated_false_findings": sample.validated_false_findings,
        "rejected_findings": sample.rejected_findings,
        "uncertain_findings": sample.uncertain_findings,
    }


def candidate_from_dict(item: dict) -> Candidate:
    return Candidate(
        candidate_id=item["candidate_id"],
        project_id=item["project_id"],
        file=item["file"],
        function=item["function"],
        line_start=int(item["line_start"]),
        line_end=int(item["line_end"]),
        code=item.get("code", ""),
        evidence=[
            Evidence(
                evidence_id=evidence["evidence_id"],
                kind=evidence["kind"],
                file=evidence["file"],
                line=int(evidence["line"]),
                expression=evidence["expression"],
                function=str(evidence.get("function", item["function"])),
                subject=evidence.get("subject"),
                object=evidence.get("object"),
                facts=dict(evidence.get("facts", {})),
            )
            for evidence in item.get("evidence", [])
        ],
        features={str(key): float(value) for key, value in item["features"].items()},
        suspicion_score=float(item.get("suspicion_score", item.get("static_score", 0.0))),
        callers=[str(value) for value in item.get("callers", [])],
        callees=[str(value) for value in item.get("callees", [])],
        feature_schema_version=str(item.get("feature_schema_version", "legacy-v1")),
        cwe_hypotheses=_cwe_hypotheses_from_raw(item),
        related_functions=[
            RelatedFunctionSummary(
                relation=str(summary["relation"]),
                function_key=str(summary.get("function_key", summary["function"])),
                file=str(summary["file"]),
                function=str(summary["function"]),
                line_start=int(summary["line_start"]),
                line_end=int(summary["line_end"]),
                parameters=[str(value) for value in summary.get("parameters", [])],
                calls=[str(value) for value in summary.get("calls", [])],
                semantic_facts=[
                    str(value) for value in summary.get("semantic_facts", [])
                ],
                code=str(summary.get("code", "")),
                symbol_types={
                    str(key): str(value)
                    for key, value in summary.get("symbol_types", {}).items()
                },
            )
            for summary in item.get("related_functions", [])
        ],
        symbol_types={
            str(key): str(value)
            for key, value in item.get("symbol_types", {}).items()
        },
    )


def finding_from_dict(item: dict) -> Finding:
    """Restore a serialized finding from checkpoint/detection JSON.

    Detection diagnostics intentionally operate on the raw Expert findings saved
    before validation.  Keeping restoration here avoids each consumer silently
    implementing a slightly different interpretation of the wire format.
    """
    return Finding(
        finding_id=str(item["finding_id"]),
        candidate_id=str(item["candidate_id"]),
        expert=ExpertFamily(item["expert"]),
        title=str(item["title"]),
        root_cause=str(item["root_cause"]),
        consequence=str(item["consequence"]),
        file=str(item["file"]),
        function=str(item["function"]),
        line_start=int(item["line_start"]),
        line_end=int(item["line_end"]),
        cwes=[str(value) for value in item.get("cwes", [])],
        source=item.get("source"),
        sink=item.get("sink"),
        missing_guard=item.get("missing_guard"),
        trigger_path=[str(value) for value in item.get("trigger_path", [])],
        evidence_ids=[str(value) for value in item.get("evidence_ids", [])],
        confidence=float(item["confidence"]),
        preconditions=[str(value) for value in item.get("preconditions", [])],
        evidence_for=[str(value) for value in item.get("evidence_for", [])],
        evidence_against=[str(value) for value in item.get("evidence_against", [])],
        falsification_test=item.get("falsification_test"),
        model_id=item.get("model_id"),
        prompt_version=item.get("prompt_version"),
        supporting_experts=[
            ExpertFamily(value) for value in item.get("supporting_experts", [])
        ],
        supporting_models=[
            str(value) for value in item.get("supporting_models", [])
        ],
    )


def _candidate_from_raw(item: dict) -> Candidate:
    """Backward-compatible internal alias for older dataset readers."""
    return candidate_from_dict(item)


def _cwe_hypotheses_from_raw(item: dict) -> list[CweHypothesis]:
    return [
        CweHypothesis(
            cwe=str(hypothesis["cwe"]),
            confidence=float(hypothesis["confidence"]),
            evidence_ids=[
                str(value) for value in hypothesis.get("evidence_ids", [])
            ],
            reasons=[str(value) for value in hypothesis.get("reasons", [])],
        )
        for hypothesis in item.get("cwe_hypotheses", [])
    ]
