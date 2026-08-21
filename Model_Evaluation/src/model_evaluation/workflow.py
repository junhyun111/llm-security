from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

from .adapters.llm_security import (
    activate_parent_package,
    app_config,
    expert_assignments,
    load_cases,
    load_outcomes,
)
from .api_budget import LoggedLLMClient
from .candidates import (
    StreamingCachedCandidateAnalyzer,
    select_matrix_candidates,
    selected_candidate_manifest,
)
from .jsonl import append_jsonl, iter_jsonl, write_jsonl
from .paths import EVALUATION_ROOT, require_within, write_json


def resolve_models(env_file: str | Path, models: Iterable[str] = ()) -> list[str]:
    config = app_config(env_file)
    explicit = [item.strip() for item in models if item.strip()]
    if explicit:
        unique = list(dict.fromkeys(explicit))
        if len(unique) != 1:
            raise ValueError(
                "Batched evaluation uses exactly one physical model per case"
            )
        return unique
    return [config.model.expert_model]


def plan_outcome_matrix(
    *,
    cases_path: str | Path,
    candidate_cache: str | Path,
    selection_manifest: str | Path,
    outcome_path: str | Path,
    model_ids: Iterable[str],
    max_candidates_per_case: int = 4,
    hard_negatives_per_case: int = 1,
) -> dict[str, object]:
    models = resolve_models_from_values(model_ids)
    assignments = expert_assignments(models)
    selection = selected_candidate_manifest(
        cases_path,
        candidate_cache,
        selection_manifest,
        max_candidates_per_case=max_candidates_per_case,
        hard_negatives_per_case=hard_negatives_per_case,
    )
    expected_keys = {
        (row["case_id"], row["candidate_id"], assignment.assignment_id)
        for row in iter_jsonl(selection_manifest)
        for assignment in assignments
    }
    completed_keys: set[tuple[str, str, str]] = set()
    output = Path(outcome_path)
    if output.is_file():
        for sample in load_outcomes(output):
            completed_keys.add(
                (sample.case_id, sample.candidate.candidate_id, sample.assignment.assignment_id)
            )
    expected_by_case: dict[str, set[tuple[str, str, str]]] = {}
    for key in expected_keys:
        expected_by_case.setdefault(key[0], set()).add(key)
    expected_case_ids = set(expected_by_case)
    completed_expected = expected_keys & completed_keys
    completed_case_ids = {
        case_id
        for case_id in expected_case_ids
        if expected_by_case[case_id] <= completed_keys
    }
    return {
        **selection,
        "model": models[0],
        "assignment_count": len(assignments),
        "expected_physical_api_requests": len(expected_case_ids),
        "completed_physical_api_requests": len(completed_case_ids),
        "remaining_physical_api_requests": len(expected_case_ids - completed_case_ids),
        "expected_logical_expert_outcomes": len(expected_keys),
        "completed_logical_expert_outcomes": len(completed_expected),
        "unexpected_existing_rows": len(completed_keys - expected_keys),
        "outcome_path": str(Path(outcome_path).resolve()),
    }


def collect_outcome_matrix(
    *,
    env_file: str | Path,
    cases_path: str | Path,
    candidate_cache: str | Path,
    outcome_path: str | Path,
    ledger_path: str | Path,
    model_ids: Iterable[str],
    max_candidates_per_case: int = 4,
    hard_negatives_per_case: int = 1,
    max_batch_characters: int = 1_000_000,
    progress: Callable[[str], None] | None = print,
) -> dict[str, object]:
    """Collect five Expert outcomes with exactly one LLM request per Juliet case."""
    activate_parent_package()
    from llm_security.datasets import UtilitySample, utility_sample_to_dict
    from llm_security.experiments.outcome_matching import FindingTruthMatcher
    from llm_security.experts import BatchedExpertRunner
    from llm_security.factory import build_context_builder, build_openrouter_client
    from llm_security.models import RouteDecision, to_dict
    from llm_security.validation import EvidenceValidator

    destination = require_within(outcome_path, EVALUATION_ROOT)
    ledger = require_within(ledger_path, EVALUATION_ROOT)
    config = app_config(env_file)
    if not config.model.api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing. Add only that key to the project .env."
        )
    # Match the production batched web path instead of the smaller single-Expert
    # default. Five logical Expert results share this one completion.
    config.model.max_output_tokens = max(config.model.max_output_tokens, 8_192)
    models = resolve_models_from_values(model_ids)
    model_id = models[0]
    assignments = expert_assignments(models)
    experts = [assignment.expert for assignment in assignments]
    case_directory = require_within(
        destination.with_suffix(".cases"), EVALUATION_ROOT
    )
    case_directory.mkdir(parents=True, exist_ok=True)
    analyzer = StreamingCachedCandidateAnalyzer(candidate_cache)
    client = LoggedLLMClient(build_openrouter_client(config), ledger)
    runner = BatchedExpertRunner(
        client=client,
        model=model_id,
        context_builder=build_context_builder(config),
        max_batch_characters=max_batch_characters,
        max_tasks=len(experts) * max_candidates_per_case,
    )
    validator = EvidenceValidator(use_llm_for_uncertain=False)
    matcher = FindingTruthMatcher()
    status = "complete"
    stop_reason = None
    cases_seen = completed_cases = new_cases = 0
    for case in load_cases(cases_path):
        cases_seen += 1
        candidates = analyzer.analyze(case)
        case_file = case_directory / f"{case.case_id}.json"
        if case_file.is_file():
            completed_cases += 1
            continue
        selected = select_matrix_candidates(
            candidates,
            case.ground_truth,
            max_candidates=max_candidates_per_case,
            hard_negatives=hard_negatives_per_case,
        )
        rows: list[dict] = []
        detection_payload = {
            "case_id": case.case_id,
            "project_id": case.project_id,
            "split": case.split,
            "ground_truth_ids": [truth.truth_id for truth in case.ground_truth],
            "matched_truth_ids": [],
            "matched_truth_ids_by_finding": {},
            "raw_package_path": case.metadata.get("raw_package_path"),
            "raw_to_virtual": case.metadata.get("raw_to_virtual", {}),
            "pipeline_result": {
                "candidates": [],
                "findings": [],
                "validations": [],
                "usage": [],
                "errors": [],
            },
        }
        try:
            if selected:
                routes = [
                    RouteDecision(
                        candidate_id=candidate.candidate_id,
                        scores={expert: 1.0 for expert in experts},
                        selected=experts,
                        top1_confidence=1.0,
                        top1_top2_margin=0.0,
                        policy="batched_full5_training_matrix",
                        reasons=[
                            "all five logical Experts evaluated in one physical request"
                        ],
                        assignments=assignments,
                    )
                    for candidate in selected
                ]
                output = runner.run(selected, routes)
                expected_tasks = len(selected) * len(assignments)
                if output.submitted_task_count != expected_tasks:
                    raise RuntimeError(
                        f"Batched request submitted {output.submitted_task_count}/"
                        f"{expected_tasks} Expert tasks; increase batch capacity"
                    )
                usage = output.usage[0]
                share = max(1, expected_tasks)
                candidates_by_id = {
                    candidate.candidate_id: candidate for candidate in selected
                }
                accepted: dict[tuple[str, object], list] = {}
                rejected: dict[tuple[str, object], int] = {}
                validations = []
                for finding in output.findings:
                    candidate = candidates_by_id[finding.candidate_id]
                    key = (finding.candidate_id, finding.expert)
                    validation = validator.validate(finding, candidate)
                    validations.append(validation)
                    if validation.verdict.value == "validated":
                        accepted.setdefault(key, []).append(finding)
                    else:
                        rejected[key] = rejected.get(key, 0) + 1
                for candidate in selected:
                    truths = [
                        truth
                        for truth in case.ground_truth
                        if _candidate_matches_truth(candidate, truth)
                    ]
                    for assignment in assignments:
                        key = (candidate.candidate_id, assignment.expert)
                        findings = accepted.get(key, [])
                        matched_truth_ids = sorted(
                            {
                                truth.truth_id
                                for finding in findings
                                for truth in truths
                                if matcher.matches(finding, truth, candidate)
                            }
                        )
                        true_count = sum(
                            any(matcher.matches(finding, truth, candidate) for truth in truths)
                            for finding in findings
                        )
                        false_count = len(findings) - true_count
                        sample = UtilitySample(
                            candidate=candidate,
                            assignment=assignment,
                            success=bool(matched_truth_ids),
                            false_positive=bool(false_count),
                            unsupported_claims=(
                                rejected.get(key, 0) + len(output.errors)
                            ),
                            cost=usage.cost / share,
                            matched_truth_ids=matched_truth_ids,
                            ground_truth_ids=sorted(
                                truth.truth_id for truth in truths
                            ),
                            prompt_tokens=round(usage.prompt_tokens / share),
                            completion_tokens=round(usage.completion_tokens / share),
                            latency_seconds=usage.latency_seconds / share,
                            truth_labels_available=True,
                            case_id=case.case_id,
                            label_version=matcher.label_version,
                            validated_true_findings=true_count,
                            validated_false_findings=false_count,
                            rejected_findings=rejected.get(key, 0),
                        )
                        rows.append(utility_sample_to_dict(sample))
                matched_by_finding = {
                    finding.finding_id: sorted(
                        truth.truth_id
                        for truth in case.ground_truth
                        if matcher.matches(
                            finding,
                            truth,
                            candidates_by_id[finding.candidate_id],
                        )
                    )
                    for findings in accepted.values()
                    for finding in findings
                }
                detection_payload = {
                    "case_id": case.case_id,
                    "project_id": case.project_id,
                    "split": case.split,
                    "ground_truth_ids": [
                        truth.truth_id for truth in case.ground_truth
                    ],
                    "matched_truth_ids": sorted(
                        {
                            truth_id
                            for values in matched_by_finding.values()
                            for truth_id in values
                        }
                    ),
                    "matched_truth_ids_by_finding": matched_by_finding,
                    "raw_package_path": case.metadata.get("raw_package_path"),
                    "raw_to_virtual": case.metadata.get("raw_to_virtual", {}),
                    "pipeline_result": {
                        "candidates": to_dict(selected),
                        "findings": to_dict(output.findings),
                        "validations": to_dict(validations),
                        "usage": to_dict(output.usage),
                        "errors": output.errors,
                    },
                }
            write_json(
                case_file,
                {
                    "case_id": case.case_id,
                    "physical_api_requests": int(bool(selected)),
                    "logical_expert_outcomes": len(rows),
                    "rows": rows,
                    "detection": detection_payload,
                },
            )
            new_cases += 1
            completed_cases += 1
            if progress:
                progress(
                    f"batched outcomes: {completed_cases}/{cases_seen} completed; "
                    f"physical requests this run={client.requests}"
                )
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            status = "stopped_on_error_resumable"
            stop_reason = f"{case.case_id}: {type(error).__name__}: {error}"
            append_jsonl(
                destination.with_suffix(".failures.jsonl"),
                {"case_id": case.case_id, "error": stop_reason},
            )
            break
    _consolidate_case_outcomes(case_directory, destination)
    result = {
        "status": status,
        "stop_reason": stop_reason,
        "model": model_id,
        "cases_seen": cases_seen,
        "completed_cases": completed_cases,
        "new_cases": new_cases,
        "physical_requests_this_run": client.requests,
        "actual_cost_usd_this_run": client.actual_usd,
        "request_contract": "at most one physical detection request per case",
        "outcome_path": str(destination),
        "ledger_path": str(ledger),
    }
    write_json(destination.with_suffix(".last_run.json"), result)
    return result


def resolve_models_from_values(model_ids: Iterable[str]) -> list[str]:
    models = list(dict.fromkeys(item.strip() for item in model_ids if item.strip()))
    if len(models) != 1:
        raise ValueError(
            "Exactly one model is required: one Juliet case must produce one API request"
        )
    return models


def _candidate_matches_truth(candidate, truth) -> bool:
    return (
        candidate.file == truth.file
        and candidate.line_start <= truth.line_end
        and candidate.line_end >= truth.line_start
    )


def _consolidate_case_outcomes(case_directory: Path, destination: Path) -> None:
    def rows():
        for case_file in sorted(case_directory.glob("*.json")):
            payload = json.loads(case_file.read_text(encoding="utf-8"))
            yield from payload.get("rows", [])

    write_jsonl(destination, rows())

    def detections():
        for case_file in sorted(case_directory.glob("*.json")):
            payload = json.loads(case_file.read_text(encoding="utf-8"))
            detection = payload.get("detection")
            if detection is not None:
                yield detection

    write_jsonl(destination.with_suffix(".detections.jsonl"), detections())


def audit_outcome_matrix(
    outcome_path: str | Path,
    *,
    expected_assignment_ids: Iterable[str],
    selection_manifest: str | Path | None = None,
) -> dict[str, object]:
    expected = set(expected_assignment_ids)
    if not expected:
        raise ValueError("Expected assignment IDs cannot be empty")
    rows = load_outcomes(outcome_path)
    groups: dict[tuple[str, str], list] = {}
    duplicates = 0
    for row in rows:
        key = (row.case_id, row.candidate.candidate_id)
        groups.setdefault(key, []).append(row)
    incomplete: list[str] = []
    for key, group in groups.items():
        ids = [item.assignment.assignment_id for item in group]
        duplicates += len(ids) - len(set(ids))
        if set(ids) != expected or len(ids) != len(set(ids)):
            incomplete.append(f"{key[0]}/{key[1]}")
    expected_groups = (
        {
            (str(row["case_id"]), str(row["candidate_id"]))
            for row in iter_jsonl(selection_manifest)
        }
        if selection_manifest is not None
        else set(groups)
    )
    missing_groups = expected_groups - set(groups)
    unexpected_groups = set(groups) - expected_groups
    return {
        "row_count": len(rows),
        "candidate_group_count": len(groups),
        "expected_assignment_count": len(expected),
        "duplicate_row_count": duplicates,
        "incomplete_candidate_group_count": len(incomplete),
        "incomplete_preview": incomplete[:10],
        "expected_candidate_group_count": len(expected_groups),
        "missing_candidate_group_count": len(missing_groups),
        "missing_preview": [f"{a}/{b}" for a, b in sorted(missing_groups)[:10]],
        "unexpected_candidate_group_count": len(unexpected_groups),
        "complete": (
            bool(expected_groups)
            and not incomplete
            and not missing_groups
            and not unexpected_groups
            and duplicates == 0
        ),
    }


def train_utility_router(
    *,
    train_outcomes: str | Path,
    dev_outcomes: str | Path,
    train_selection_manifest: str | Path,
    dev_selection_manifest: str | Path,
    artifact_path: str | Path,
    report_path: str | Path,
    model_ids: Iterable[str],
    seed: int = 2026,
    target_truth_recall: float = 0.95,
    gate_fraction: float = 0.5,
    escalation_threshold: float = 0.85,
    cost_weight: float = 1.0,
    false_positive_weight: float = 0.5,
    unsupported_weight: float = 0.25,
) -> dict[str, object]:
    activate_parent_package()
    from llm_security.models import to_dict
    from llm_security.routing import (
        BudgetedUtilityRouter,
        UtilityPolicyConfig,
        assert_project_disjoint,
        split_gate_calibration_samples,
    )

    models = list(model_ids)
    assignments = expert_assignments(models)
    expected_ids = [item.assignment_id for item in assignments]
    train_audit = audit_outcome_matrix(
        train_outcomes,
        expected_assignment_ids=expected_ids,
        selection_manifest=train_selection_manifest,
    )
    dev_audit = audit_outcome_matrix(
        dev_outcomes,
        expected_assignment_ids=expected_ids,
        selection_manifest=dev_selection_manifest,
    )
    if not train_audit["complete"] or not dev_audit["complete"]:
        raise ValueError(
            "Training is blocked until train and dev outcome matrices are complete. "
            f"train={train_audit}; dev={dev_audit}"
        )
    train_rows = load_outcomes(train_outcomes)
    dev_rows = load_outcomes(dev_outcomes)
    assert_project_disjoint(train_rows, dev_rows, first_name="train", second_name="dev")
    router = BudgetedUtilityRouter.fit(
        train_rows,
        policy=UtilityPolicyConfig(
            escalation_threshold=escalation_threshold,
            cost_weight=cost_weight,
            false_positive_weight=false_positive_weight,
            unsupported_weight=unsupported_weight,
        ),
        seed=seed,
    )
    gate_rows, calibration_rows = split_gate_calibration_samples(
        dev_rows, seed=seed, gate_fraction=gate_fraction
    )
    gate_count = router.fit_escalation_gate(gate_rows, seed=seed)
    calibration = router.calibrate_threshold(
        calibration_rows, target_truth_recall=target_truth_recall
    )
    baselines = router.calibrate_baselines(calibration_rows)
    metrics = router.evaluate(calibration_rows)
    artifact = require_within(artifact_path, EVALUATION_ROOT)
    router.save(artifact)
    report = {
        "artifact": str(artifact),
        "models": models,
        "train_matrix_audit": train_audit,
        "dev_matrix_audit": dev_audit,
        "gate_training_candidates": gate_count,
        "calibration": to_dict(calibration),
        "baseline_calibration": to_dict(baselines),
        "calibration_metrics": to_dict(metrics),
    }
    write_json(report_path, report)
    return report


def evaluate_utility_router(
    *,
    artifact_path: str | Path,
    test_outcomes: str | Path,
    test_cases: str | Path,
    candidate_cache: str | Path,
    selection_manifest: str | Path,
    report_path: str | Path,
    candidate_gate_enabled: bool = False,
    candidate_gate_threshold: float = 0.4,
    max_candidates_per_case: int = 4,
) -> dict[str, object]:
    activate_parent_package()
    from llm_security.experiments.utility_end_to_end import evaluate_utility_end_to_end
    from llm_security.models import to_dict
    from llm_security.routing import BudgetedUtilityRouter, CandidateGate

    router = BudgetedUtilityRouter.load(artifact_path)
    rows = load_outcomes(test_outcomes)
    router.assert_test_projects_unseen(rows)
    expected_ids = list(router.assignments)
    matrix_audit = audit_outcome_matrix(
        test_outcomes,
        expected_assignment_ids=expected_ids,
        selection_manifest=selection_manifest,
    )
    if not matrix_audit["complete"]:
        raise ValueError("Test outcome matrix is incomplete: " + str(matrix_audit))
    policy_metrics = router.evaluate_baselines(rows)
    false_positive_rates = _policy_false_positive_rates(router, rows)
    e2e = evaluate_utility_end_to_end(
        load_cases(test_cases),
        rows,
        analyzer=StreamingCachedCandidateAnalyzer(candidate_cache),
        candidate_gate=CandidateGate(
            enabled=candidate_gate_enabled,
            threshold=candidate_gate_threshold,
        ),
        router=router,
        max_candidates_per_case=max_candidates_per_case,
    )
    report = {
        "artifact": str(Path(artifact_path).resolve()),
        "test_outcomes": str(Path(test_outcomes).resolve()),
        "test_matrix_audit": matrix_audit,
        "policies": to_dict(policy_metrics),
        "policy_negative_candidate_false_positive_rates": false_positive_rates,
        "end_to_end": to_dict(e2e),
    }
    write_json(report_path, report)
    return report


def _policy_false_positive_rates(router, rows: list) -> dict[str, dict[str, float | int]]:
    """Candidate-level FPR over matrix candidates with no overlapping GT region."""
    activate_parent_package()
    from llm_security.models import ACTIVE_UTILITY_EXPERTS
    from llm_security.routing.escalation import independence_top2_confidence

    groups: dict[tuple[str, str], list] = {}
    for row in rows:
        groups.setdefault((row.case_id, row.candidate.candidate_id), []).append(row)

    def ranked(group):
        return router._rank(group[0].candidate).ranked_assignment_ids

    def fixed_e1_e3(group):
        ids = ranked(group)
        by_family = {router.assignments[item].expert: item for item in ids}
        return [by_family[item] for item in ACTIVE_UTILITY_EXPERTS[:2]]

    def formula(group):
        ranked_result = router._rank(group[0].candidate)
        top2 = ranked_result.ranked_assignment_ids[:2]
        confidence = independence_top2_confidence(
            [ranked_result.assignment_probabilities[item] for item in top2]
        )
        return (
            top2
            if confidence >= router.policy.escalation_threshold
            else ranked_result.ranked_assignment_ids
        )

    selectors = {
        "full5": lambda group: ranked(group),
        "fixed_top2_e1_e3": fixed_e1_e3,
        "utility_top2": lambda group: ranked(group)[:2],
        "formula_escalation": formula,
        "adaptive_gate": lambda group: [
            item.assignment_id for item in router.route(group[0].candidate).assignments
        ],
    }
    if router.best_single_assignment_id:
        selectors["best_single"] = lambda group: [router.best_single_assignment_id]
    if router.best_fixed_pair_assignment_ids:
        selectors["best_fixed2"] = lambda group: list(router.best_fixed_pair_assignment_ids)

    report: dict[str, dict[str, float | int]] = {}
    negatives = [group for group in groups.values() if not any(
        row.ground_truth_ids for row in group
    )]
    for name, selector in selectors.items():
        false_positive_candidates = 0
        false_findings = 0
        selected_outcomes = 0
        for group in negatives:
            by_id = {row.assignment.assignment_id: row for row in group}
            selected = [by_id[item] for item in selector(group)]
            counts = [
                row.validated_false_findings
                if row.validated_true_findings or row.validated_false_findings
                else int(row.false_positive)
                for row in selected
            ]
            false_positive_candidates += int(any(counts))
            false_findings += sum(counts)
            selected_outcomes += len(selected)
        report[name] = {
            "negative_candidate_count": len(negatives),
            "false_positive_candidate_count": false_positive_candidates,
            "false_positive_rate": (
                false_positive_candidates / len(negatives) if negatives else 0.0
            ),
            "validated_false_finding_count": false_findings,
            "selected_outcome_count": selected_outcomes,
        }
    return report
