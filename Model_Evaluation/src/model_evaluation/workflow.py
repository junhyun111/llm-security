from __future__ import annotations

from dataclasses import asdict
from itertools import islice
from pathlib import Path
from typing import Callable, Iterable

from .adapters.llm_security import (
    activate_parent_package,
    app_config,
    expert_assignments,
    load_cases,
    load_outcomes,
)
from .api_budget import ApiBudget, BudgetExceeded, BudgetedLLMClient
from .candidates import StreamingCachedCandidateAnalyzer, selected_candidate_manifest
from .jsonl import iter_jsonl
from .paths import EVALUATION_ROOT, require_within, write_json


def resolve_models(env_file: str | Path, models: Iterable[str] = ()) -> list[str]:
    config = app_config(env_file)
    explicit = [item.strip() for item in models if item.strip()]
    if explicit:
        return list(dict.fromkeys(explicit))
    if config.model.sweep_models:
        return list(config.model.sweep_models)
    return list(dict.fromkeys(config.model.expert_models.values()))


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
    assignments = expert_assignments(list(model_ids))
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
    completed_expected = expected_keys & completed_keys
    return {
        **selection,
        "models": list(dict.fromkeys(assignment.model_id for assignment in assignments)),
        "assignment_count": len(assignments),
        "expected_api_requests": len(expected_keys),
        "completed_api_requests": len(completed_expected),
        "remaining_api_requests": len(expected_keys - completed_keys),
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
    execute_paid: bool,
    max_requests: int,
    max_usd: float,
    reserve_usd_per_request: float = 0.10,
    max_candidates_per_case: int = 4,
    hard_negatives_per_case: int = 1,
    progress: Callable[[str], None] | None = print,
) -> dict[str, object]:
    """Collect a resumable full Expert x model matrix under hard run guards."""
    activate_parent_package()
    from llm_security.experiments.utility_data import collect_expert_outcomes
    from llm_security.experts import ExpertRunner
    from llm_security.factory import build_context_builder, build_openrouter_client

    destination = require_within(outcome_path, EVALUATION_ROOT)
    ledger = require_within(ledger_path, EVALUATION_ROOT)
    config = app_config(env_file)
    if not execute_paid:
        raise RuntimeError(
            "Paid execution is locked. Set EXECUTE_PAID=True in the notebook after "
            "reviewing the dry-run request count and budget."
        )
    if not config.runtime.allow_paid_experiments:
        raise RuntimeError(
            "The project .env also requires RUN_PAID_EXPERIMENTS=1. Both locks must be on."
        )
    if not config.model.api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing")
    assignments = expert_assignments(list(model_ids))
    budget = ApiBudget(
        max_requests=max_requests,
        max_usd=max_usd,
        reserve_usd_per_request=reserve_usd_per_request,
    )
    client = BudgetedLLMClient(build_openrouter_client(config), budget, ledger)
    status = "complete"
    stop_reason = None
    collection: dict[str, object] = {}
    try:
        collection = collect_expert_outcomes(
            load_cases(cases_path),
            analyzer=StreamingCachedCandidateAnalyzer(candidate_cache),
            expert_runner=ExpertRunner(
                client=client,
                model=config.model.expert_model,
                context_builder=build_context_builder(config),
                models_by_family=config.model.expert_models,
            ),
            assignments=assignments,
            output_path=destination,
            max_candidates_per_case=max_candidates_per_case,
            hard_negatives_per_case=hard_negatives_per_case,
            resume=True,
            progress=progress,
        )
    except BudgetExceeded as error:
        status = "budget_stopped_resumable"
        stop_reason = str(error)
    result = {
        "status": status,
        "stop_reason": stop_reason,
        "requests_this_run": budget.requests,
        "actual_cost_usd_this_run": budget.actual_usd,
        "max_requests_this_run": budget.max_requests,
        "max_usd_this_run": budget.max_usd,
        "outcome_path": str(destination),
        "ledger_path": str(ledger),
        "collection": collection,
    }
    write_json(destination.with_suffix(".last_run.json"), result)
    return result


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
