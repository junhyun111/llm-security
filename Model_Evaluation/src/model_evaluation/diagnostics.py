from __future__ import annotations

from collections import Counter
from pathlib import Path

from .adapters.llm_security import (
    activate_parent_package,
    candidate_from_dict,
    load_cases,
)
from .candidates import StreamingCachedCandidateAnalyzer
from .jsonl import iter_jsonl
from .paths import write_json


def audit_detection_stages(
    *,
    cases_path: str | Path,
    candidate_cache: str | Path,
    detection_path: str | Path,
    max_candidates_per_case: int = 4,
    report_path: str | Path | None = None,
) -> dict[str, object]:
    """Attribute false negatives to candidate, Expert, and Validator stages.

    This consumes existing checkpointed detections and makes no API calls.  Raw
    Expert findings are matched before their validation verdict is consulted, so
    a correct finding rejected by a deterministic validation rule remains visible.
    """
    if max_candidates_per_case < 1:
        raise ValueError("max_candidates_per_case must be positive")

    activate_parent_package()
    from llm_security.datasets import finding_from_dict
    from llm_security.experiments.outcome_matching import FindingTruthMatcher
    from llm_security.models import ACTIVE_UTILITY_EXPERTS

    detections = {
        str(row["case_id"]): row for row in iter_jsonl(detection_path)
    }
    analyzer = StreamingCachedCandidateAnalyzer(candidate_cache)
    matcher = FindingTruthMatcher()
    totals: Counter[str] = Counter()
    rejection_checks: Counter[str] = Counter()
    uncertainty_checks: Counter[str] = Counter()
    per_expert: dict[str, Counter[str]] = {
        expert.value: Counter() for expert in ACTIVE_UTILITY_EXPERTS
    }

    for case in load_cases(cases_path):
        all_candidates = analyzer.analyze(case)
        top_candidates = sorted(
            all_candidates,
            key=lambda item: (
                -item.suspicion_score,
                item.file,
                item.line_start,
                item.candidate_id,
            ),
        )[:max_candidates_per_case]
        detection = detections.get(case.case_id, {})
        raw_result = detection.get("pipeline_result", {})
        selected_candidates = {
            candidate.candidate_id: candidate
            for candidate in (
                candidate_from_dict(raw)
                for raw in raw_result.get("candidates", [])
            )
        }
        validations = {
            str(raw["finding_id"]): raw
            for raw in raw_result.get("validations", [])
        }
        raw_findings = [
            finding_from_dict(raw) for raw in raw_result.get("findings", [])
        ]
        validated_raw_ids = {
            finding_id
            for finding_id, validation in validations.items()
            if validation.get("verdict") == "validated"
        }
        uncertain_raw_ids = {
            finding_id
            for finding_id, validation in validations.items()
            if validation.get("verdict") == "uncertain"
        }
        rejected_raw_ids = {
            finding_id
            for finding_id, validation in validations.items()
            if validation.get("verdict") == "rejected"
        }

        raw_matches: dict[str, set[str]] = {}
        raw_true_ids: set[str] = set()
        for finding in raw_findings:
            candidate = selected_candidates.get(finding.candidate_id)
            matched = {
                truth.truth_id
                for truth in case.ground_truth
                if candidate is not None and matcher.matches(finding, truth, candidate)
            }
            raw_matches[finding.finding_id] = matched
            if matched:
                raw_true_ids.add(finding.finding_id)

        false_raw_ids = {
            finding.finding_id
            for finding in raw_findings
            if not raw_matches.get(finding.finding_id)
        }
        totals["raw_finding_count"] += len(raw_findings)
        totals["raw_true_finding_count"] += len(raw_true_ids)
        totals["raw_false_finding_count"] += len(false_raw_ids)
        totals["validated_true_finding_count"] += len(
            raw_true_ids & validated_raw_ids
        )
        totals["rejected_true_finding_count"] += len(raw_true_ids & rejected_raw_ids)
        totals["uncertain_true_finding_count"] += len(raw_true_ids & uncertain_raw_ids)
        totals["validated_false_finding_count"] += len(
            false_raw_ids & validated_raw_ids
        )
        totals["rejected_false_finding_count"] += len(false_raw_ids & rejected_raw_ids)
        totals["uncertain_false_finding_count"] += len(false_raw_ids & uncertain_raw_ids)
        for finding_id in raw_true_ids & rejected_raw_ids:
            checks = validations.get(finding_id, {}).get("checks", {})
            for name, passed in checks.items():
                if passed is False:
                    rejection_checks[str(name)] += 1
        for finding_id in raw_true_ids & uncertain_raw_ids:
            checks = validations.get(finding_id, {}).get("checks", {})
            for name, passed in checks.items():
                if passed is False:
                    uncertainty_checks[str(name)] += 1

        final_ids = {
            str(value) for value in detection.get("matched_truth_ids", [])
        }
        for truth in case.ground_truth:
            expected_experts = truth.experts or list(ACTIVE_UTILITY_EXPERTS)
            candidate_found = any(
                _candidate_matches_truth(candidate, truth)
                for candidate in all_candidates
            )
            top_k_found = any(
                _candidate_matches_truth(candidate, truth)
                for candidate in top_candidates
            )
            raw_found = any(
                truth.truth_id in raw_matches.get(finding.finding_id, set())
                for finding in raw_findings
            )
            validated_found = any(
                finding.finding_id in validated_raw_ids
                and truth.truth_id in raw_matches.get(finding.finding_id, set())
                for finding in raw_findings
            )
            final_found = truth.truth_id in final_ids
            _increment_truth_stages(
                totals,
                candidate_found=candidate_found,
                top_k_found=top_k_found,
                raw_found=raw_found,
                validated_found=validated_found,
                final_found=final_found,
            )
            for expert in expected_experts:
                expert_raw_found = any(
                    finding.expert == expert
                    and truth.truth_id in raw_matches.get(finding.finding_id, set())
                    for finding in raw_findings
                )
                expert_validated_found = any(
                    finding.expert == expert
                    and finding.finding_id in validated_raw_ids
                    and truth.truth_id in raw_matches.get(finding.finding_id, set())
                    for finding in raw_findings
                )
                _increment_truth_stages(
                    per_expert.setdefault(expert.value, Counter()),
                    candidate_found=candidate_found,
                    top_k_found=top_k_found,
                    raw_found=expert_raw_found,
                    validated_found=expert_validated_found,
                    final_found=final_found,
                )

    report = {
        "cases_path": str(Path(cases_path).resolve()),
        "candidate_cache": str(Path(candidate_cache).resolve()),
        "detection_path": str(Path(detection_path).resolve()),
        "max_candidates_per_case": max_candidates_per_case,
        "overall": _stage_metrics(totals),
        "per_expected_expert": {
            expert: _stage_metrics(counts)
            for expert, counts in sorted(per_expert.items())
            if counts["ground_truth_count"]
        },
        "validator_true_finding_rejection_checks": dict(
            rejection_checks.most_common()
        ),
        "validator_true_finding_uncertainty_checks": dict(
            uncertainty_checks.most_common()
        ),
    }
    if report_path is not None:
        write_json(report_path, report)
    return report


def calibrate_expert_confidence_thresholds(
    *,
    cases_path: str | Path,
    detection_path: str | Path,
    thresholds: tuple[float, ...] = (
        0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
    ),
    minimum_precision: float = 0.50,
    beta: float = 2.0,
    report_path: str | Path | None = None,
) -> dict[str, object]:
    """Calibrate per-Expert confidence gates on a labeled development split.

    The input is checkpointed raw Expert output, so this routine performs no API
    calls. Test/holdout detections must never be used to choose these thresholds.
    """
    if not thresholds or any(not 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("thresholds must contain values between 0 and 1")
    if not 0.0 <= minimum_precision <= 1.0 or beta <= 0.0:
        raise ValueError("minimum_precision and beta are invalid")

    activate_parent_package()
    from llm_security.datasets import candidate_from_dict, finding_from_dict
    from llm_security.experiments.outcome_matching import FindingTruthMatcher

    cases = {case.case_id: case for case in load_cases(cases_path)}
    matcher = FindingTruthMatcher()
    observations: dict[str, list[tuple[float, bool]]] = {}
    for detection in iter_jsonl(detection_path):
        case = cases.get(str(detection["case_id"]))
        if case is None:
            continue
        pipeline = detection.get("pipeline_result", {})
        candidates = {
            item.candidate_id: item
            for item in (
                candidate_from_dict(raw)
                for raw in pipeline.get("candidates", [])
            )
        }
        for raw in pipeline.get("findings", []):
            finding = finding_from_dict(raw)
            candidate = candidates.get(finding.candidate_id)
            is_true = bool(
                candidate is not None
                and any(
                    matcher.matches(finding, truth, candidate)
                    for truth in case.ground_truth
                )
            )
            observations.setdefault(finding.expert.value, []).append(
                (finding.confidence, is_true)
            )

    report_by_expert: dict[str, dict[str, object]] = {}
    selected_thresholds: dict[str, float] = {}
    for expert, rows in sorted(observations.items()):
        metrics = [
            _confidence_metrics(rows, threshold, beta=beta)
            for threshold in sorted(set(thresholds))
        ]
        eligible = [
            item for item in metrics if item["precision"] >= minimum_precision
        ]
        pool = eligible or metrics
        selected = max(
            pool,
            key=lambda item: (
                item["f_beta"], item["recall"], item["precision"],
                -item["threshold"],
            ),
        )
        selected_thresholds[expert] = float(selected["threshold"])
        report_by_expert[expert] = {
            "raw_finding_count": len(rows),
            "raw_true_finding_count": sum(label for _, label in rows),
            "selected": selected,
            "grid": metrics,
            "met_minimum_precision": bool(eligible),
        }
    report = {
        "cases_path": str(Path(cases_path).resolve()),
        "detection_path": str(Path(detection_path).resolve()),
        "selection_split_requirement": "development-only",
        "minimum_precision": minimum_precision,
        "beta": beta,
        "minimum_confidence_by_expert": selected_thresholds,
        "per_expert": report_by_expert,
    }
    if report_path is not None:
        write_json(report_path, report)
    return report


def _confidence_metrics(
    rows: list[tuple[float, bool]], threshold: float, *, beta: float
) -> dict[str, float | int]:
    accepted = [label for confidence, label in rows if confidence >= threshold]
    true_positive = sum(accepted)
    false_positive = len(accepted) - true_positive
    positives = sum(label for _, label in rows)
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, positives)
    beta_squared = beta * beta
    f_beta = (
        (1.0 + beta_squared) * precision * recall
        / (beta_squared * precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "threshold": threshold,
        "true_positive_findings": true_positive,
        "false_positive_findings": false_positive,
        "precision": precision,
        "recall": recall,
        "f_beta": f_beta,
    }


def _candidate_matches_truth(candidate, truth) -> bool:
    return (
        candidate.file == truth.file
        and candidate.line_start <= truth.line_end
        and candidate.line_end >= truth.line_start
    )


def _increment_truth_stages(
    counts: Counter[str],
    *,
    candidate_found: bool,
    top_k_found: bool,
    raw_found: bool,
    validated_found: bool,
    final_found: bool,
) -> None:
    counts["ground_truth_count"] += 1
    counts["candidate_found"] += int(candidate_found)
    counts["top_k_candidate_found"] += int(top_k_found)
    counts["raw_expert_found"] += int(raw_found)
    counts["matched_finding_validated"] += int(validated_found)
    counts["final_matched"] += int(final_found)


def _stage_metrics(counts: Counter[str]) -> dict[str, float | int]:
    truths = counts["ground_truth_count"]
    raw_truths = counts["raw_expert_found"]
    raw_true_findings = counts["raw_true_finding_count"]
    raw_false_findings = counts["raw_false_finding_count"]
    return {
        **dict(counts),
        "candidate_recall": _ratio(counts["candidate_found"], truths),
        "top_k_candidate_recall": _ratio(
            counts["top_k_candidate_found"], truths
        ),
        "raw_expert_recall": _ratio(raw_truths, truths),
        "validator_true_truth_retention": _ratio(
            counts["matched_finding_validated"], raw_truths
        ),
        "validated_recall": _ratio(counts["matched_finding_validated"], truths),
        "final_recall": _ratio(counts["final_matched"], truths),
        "validator_true_finding_retention": _ratio(
            counts["validated_true_finding_count"], raw_true_findings
        ),
        "validator_false_finding_rejection": _ratio(
            counts["rejected_false_finding_count"], raw_false_findings
        ),
        "validator_true_finding_uncertainty": _ratio(
            counts["uncertain_true_finding_count"], raw_true_findings
        ),
        "validator_false_finding_uncertainty": _ratio(
            counts["uncertain_false_finding_count"], raw_false_findings
        ),
    }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
