from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Protocol

from ..datasets import UtilitySample, load_utility_samples_jsonl, utility_sample_to_dict
from ..experts import ExpertRunner
from ..models import (
    Candidate,
    ExpertAssignment,
    Finding,
    GroundTruth,
    ProjectCase,
    RouteDecision,
)
from ..validation import EvidenceValidator
from .outcome_matching import FindingTruthMatcher


class CandidateAnalyzer(Protocol):
    def analyze(self, case: ProjectCase) -> list[Candidate]: ...


def collect_expert_outcomes(
    cases: Iterable[ProjectCase],
    *,
    analyzer: CandidateAnalyzer,
    expert_runner: ExpertRunner,
    assignments: Iterable[ExpertAssignment],
    output_path: str | Path,
    max_candidates_per_case: int = 4,
    hard_negatives_per_case: int = 1,
    resume: bool = True,
    progress: Callable[[str], None] | None = None,
    truth_matcher: FindingTruthMatcher | None = None,
) -> dict[str, object]:
    """Run the full Expert x model matrix and checkpoint compact outcome rows."""
    jobs = list(assignments)
    if not jobs:
        raise ValueError("At least one Expert x model assignment is required")
    destination = Path(output_path)
    matcher = truth_matcher or FindingTruthMatcher()
    destination.parent.mkdir(parents=True, exist_ok=True)
    completed: set[tuple[str, str, str]] = set()
    if resume and destination.exists():
        existing = load_utility_samples_jsonl(destination)
        if any(
            not sample.truth_labels_available
            or sample.label_version != matcher.label_version
            or not sample.case_id
            for sample in existing
        ):
            raise ValueError(
                "Existing outcome JSONL uses legacy or incompatible GT labels. "
                f"Expected label_version={matcher.label_version} with case IDs. "
                "Rerun with --no-resume to rebuild outcomes."
            )
        completed = {
            (
                sample.case_id,
                sample.candidate.candidate_id,
                sample.assignment.assignment_id,
            )
            for sample in existing
        }
    elif destination.exists():
        destination.unlink()

    validator = EvidenceValidator(use_llm_for_uncertain=False)
    written = skipped = failed = cases_seen = 0
    by_assignment: dict[str, dict[str, float]] = defaultdict(
        lambda: {"samples": 0.0, "successes": 0.0, "false_positives": 0.0, "cost": 0.0}
    )
    for case in cases:
        cases_seen += 1
        try:
            candidates = analyzer.analyze(case)
        except (RuntimeError, ValueError) as error:
            failed += len(jobs)
            if progress:
                progress(f"skip {case.case_id}: analyzer failed: {error}")
            continue
        selected = _select_matrix_candidates(
            candidates,
            case.ground_truth,
            max_candidates=max_candidates_per_case,
            hard_negatives=hard_negatives_per_case,
        )
        for candidate in selected:
            truths = [
                truth for truth in case.ground_truth if _candidate_matches_truth(candidate, truth)
            ]
            for assignment in jobs:
                key = (
                    case.case_id,
                    candidate.candidate_id,
                    assignment.assignment_id,
                )
                if key in completed:
                    skipped += 1
                    continue
                route = RouteDecision(
                    candidate_id=candidate.candidate_id,
                    scores={assignment.expert: 1.0},
                    selected=[assignment.expert],
                    top1_confidence=1.0,
                    top1_top2_margin=1.0,
                    policy="performance_matrix",
                    reasons=["exhaustive Expert x model outcome collection"],
                    assignments=[assignment],
                )
                output = expert_runner.run([candidate], [route])
                accepted: list[Finding] = []
                rejected = 0
                uncertain = 0
                for finding in output.findings:
                    result = validator.validate(finding, candidate)
                    if result.verdict.value == "validated":
                        accepted.append(finding)
                    elif result.verdict.value == "rejected":
                        rejected += 1
                    else:
                        uncertain += 1
                success = any(
                    matcher.matches(finding, truth, candidate)
                    for finding in accepted
                    for truth in truths
                )
                matched_truth_ids = sorted(
                    {
                        truth.truth_id
                        for finding in accepted
                        for truth in truths
                        if matcher.matches(finding, truth, candidate)
                    }
                )
                false_positive = any(
                    not any(
                        matcher.matches(finding, truth, candidate)
                        for truth in truths
                    )
                    for finding in accepted
                )
                validated_true_findings = sum(
                    any(
                        matcher.matches(finding, truth, candidate)
                        for truth in truths
                    )
                    for finding in accepted
                )
                validated_false_findings = len(accepted) - validated_true_findings
                cost = sum(item.cost for item in output.usage)
                sample = UtilitySample(
                    candidate=candidate,
                    assignment=assignment,
                    success=success,
                    false_positive=false_positive,
                    unsupported_claims=rejected + len(output.errors),
                    cost=cost,
                    matched_truth_ids=matched_truth_ids,
                    ground_truth_ids=sorted(truth.truth_id for truth in truths),
                    prompt_tokens=sum(item.prompt_tokens for item in output.usage),
                    completion_tokens=sum(
                        item.completion_tokens for item in output.usage
                    ),
                    latency_seconds=sum(
                        item.latency_seconds for item in output.usage
                    ),
                    truth_labels_available=True,
                    case_id=case.case_id,
                    label_version=matcher.label_version,
                    validated_true_findings=validated_true_findings,
                    validated_false_findings=validated_false_findings,
                    rejected_findings=rejected,
                    uncertain_findings=uncertain,
                )
                with destination.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(utility_sample_to_dict(sample), ensure_ascii=False) + "\n")
                completed.add(key)
                written += 1
                stats = by_assignment[assignment.assignment_id]
                stats["samples"] += 1
                stats["successes"] += int(success)
                stats["false_positives"] += int(false_positive)
                stats["cost"] += cost
        if progress:
            progress(
                f"matrix: {cases_seen} cases, {written} new rows, {skipped} resumed rows"
            )
    return {
        "cases_seen": cases_seen,
        "rows_written": written,
        "rows_skipped": skipped,
        "failed_jobs": failed,
        "output": str(destination),
        "new_rows_by_assignment": dict(by_assignment),
        "label_version": matcher.label_version,
    }


def _select_matrix_candidates(
    candidates: list[Candidate],
    truths: list[GroundTruth],
    *,
    max_candidates: int,
    hard_negatives: int,
) -> list[Candidate]:
    positives = [
        candidate
        for candidate in candidates
        if any(_candidate_matches_truth(candidate, truth) for truth in truths)
    ]
    negatives = [candidate for candidate in candidates if candidate not in positives]
    positives.sort(key=lambda item: (-item.suspicion_score, item.candidate_id))
    negatives.sort(key=lambda item: (-item.suspicion_score, item.candidate_id))
    negative_slots = min(hard_negatives, len(negatives), max_candidates)
    selected = positives[: max_candidates - negative_slots]
    selected.extend(negatives[:negative_slots])
    unique: list[Candidate] = []
    seen: set[str] = set()
    for candidate in selected:
        if candidate.candidate_id not in seen:
            unique.append(candidate)
            seen.add(candidate.candidate_id)
    return unique


def _candidate_matches_truth(candidate: Candidate, truth: GroundTruth) -> bool:
    return (
        candidate.file == truth.file
        and candidate.line_start <= truth.line_end
        and candidate.line_end >= truth.line_start
    )


def _finding_matches_truth(
    finding: Finding,
    truth: GroundTruth,
    candidate: Candidate,
) -> bool:
    """Compatibility wrapper retained for experiment callers and tests."""
    return FindingTruthMatcher().matches(finding, truth, candidate)
