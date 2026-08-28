from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

from .adapters.llm_security import (
    activate_parent_package,
    evaluation_api_config,
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
from .concurrency import CompletionResult, run_completion_pool
from .jsonl import append_jsonl, iter_jsonl, write_jsonl
from .paths import EVALUATION_ROOT, require_within, write_json


def resolve_models(env_file: str | Path, models: Iterable[str] = ()) -> list[str]:
    config = evaluation_api_config(env_file)
    explicit = [item.strip() for item in models if item.strip()]
    if explicit:
        unique = list(dict.fromkeys(explicit))
        if len(unique) != 1:
            raise ValueError(
                "Batched evaluation uses exactly one physical model per case"
            )
        return unique
    return [config.model.expert_model]


def _outcome_case_contract(
    model_id, assignments, candidates, validator_confidence_thresholds
) -> dict[str, object]:
    return {
        "version": "batched-outcome-contract-v2",
        "model_id": model_id,
        "assignment_ids": sorted(item.assignment_id for item in assignments),
        "candidate_ids": sorted(item.candidate_id for item in candidates),
        "feature_schemas": sorted(
            {item.feature_schema_version for item in candidates}
        ),
        "validator_confidence_thresholds": dict(
            sorted(validator_confidence_thresholds.items())
        ),
    }


def _case_file_matches_contract(
    path: str | Path, expected: dict[str, object]
) -> bool:
    source = Path(path)
    if not source.is_file():
        return False
    try:
        stored = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return stored.get("contract") == expected


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
    max_concurrency: int = 1_000,
    max_case_attempts: int = 6,
    retry_base_seconds: float = 1.0,
    retry_max_seconds: float = 60.0,
    validator_minimum_confidence_by_expert: dict[str, float] | None = None,
    progress: Callable[[str], None] | None = print,
) -> dict[str, object]:
    """Collect one batched Expert completion per case through a bounded async pool."""
    activate_parent_package()
    from llm_security.datasets import UtilitySample, utility_sample_to_dict
    from llm_security.experiments.outcome_matching import FindingTruthMatcher
    from llm_security.experts import BatchedExpertRunner
    from llm_security.factory import build_context_builder, build_openrouter_client
    from llm_security.models import RouteDecision, to_dict
    from llm_security.validation import EvidenceValidator

    destination = require_within(outcome_path, EVALUATION_ROOT)
    ledger = require_within(ledger_path, EVALUATION_ROOT)
    config = evaluation_api_config(env_file)
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
    validator = EvidenceValidator(
        use_llm_for_uncertain=False,
        minimum_confidence_by_expert=validator_minimum_confidence_by_expert,
    )
    validator_thresholds = {
        expert.value: validator.confidence_threshold_for(expert)
        for expert in experts
    }
    matcher = FindingTruthMatcher()
    if max_concurrency < 1 or max_concurrency > 1_000:
        raise ValueError("max_concurrency must be between 1 and 1000")

    cases = list(load_cases(cases_path))
    cases_seen = len(cases)
    completed_cases = 0
    new_cases = 0
    work_items = []

    # The candidate cache is intentionally streaming and must be consumed in
    # case order. Candidate selection therefore stays single-threaded and is
    # completed before any network work is scheduled.
    for case in cases:
        candidates = analyzer.analyze(case)
        case_file = case_directory / f"{case.case_id}.json"
        selected = select_matrix_candidates(
            candidates,
            case.ground_truth,
            max_candidates=max_candidates_per_case,
            hard_negatives=hard_negatives_per_case,
        )
        contract = _outcome_case_contract(
            model_id, assignments, selected, validator_thresholds
        )
        if _case_file_matches_contract(case_file, contract):
            completed_cases += 1
            continue
        if selected:
            work_items.append((case, selected, case_file, contract))
            continue
        write_json(
            case_file,
            {
                "case_id": case.case_id,
                "contract": contract,
                "physical_api_requests": 0,
                "logical_expert_outcomes": 0,
                "rows": [],
                "detection": _empty_detection_payload(case),
            },
        )
        new_cases += 1
        completed_cases += 1

    def process_case(item):
        case, selected, case_file, contract = item
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
        uncertain: dict[tuple[str, object], int] = {}
        validations = []
        for finding in output.findings:
            candidate = candidates_by_id[finding.candidate_id]
            key = (finding.candidate_id, finding.expert)
            validation = validator.validate(finding, candidate)
            validations.append(validation)
            if validation.verdict.value == "validated":
                accepted.setdefault(key, []).append(finding)
            elif validation.verdict.value == "rejected":
                rejected[key] = rejected.get(key, 0) + 1
            else:
                uncertain[key] = uncertain.get(key, 0) + 1
        rows: list[dict] = []
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
                    any(
                        matcher.matches(finding, truth, candidate)
                        for truth in truths
                    )
                    for finding in findings
                )
                false_count = len(findings) - true_count
                sample = UtilitySample(
                    candidate=candidate,
                    assignment=assignment,
                    success=bool(matched_truth_ids),
                    false_positive=bool(false_count),
                    unsupported_claims=rejected.get(key, 0) + len(output.errors),
                    cost=usage.cost / share,
                    matched_truth_ids=matched_truth_ids,
                    ground_truth_ids=sorted(truth.truth_id for truth in truths),
                    prompt_tokens=round(usage.prompt_tokens / share),
                    completion_tokens=round(usage.completion_tokens / share),
                    latency_seconds=usage.latency_seconds / share,
                    truth_labels_available=True,
                    case_id=case.case_id,
                    label_version=matcher.label_version,
                    validated_true_findings=true_count,
                    validated_false_findings=false_count,
                    rejected_findings=rejected.get(key, 0),
                    uncertain_findings=uncertain.get(key, 0),
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
            **_empty_detection_payload(case),
            "matched_truth_ids": sorted(
                {
                    truth_id
                    for values in matched_by_finding.values()
                    for truth_id in values
                }
            ),
            "matched_truth_ids_by_finding": matched_by_finding,
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
                "contract": contract,
                "physical_api_requests": 1,
                "logical_expert_outcomes": len(rows),
                "rows": rows,
                "detection": detection_payload,
            },
        )
        return {"case_id": case.case_id, "logical_expert_outcomes": len(rows)}

    counters = {"completed": completed_cases, "new": new_cases, "failed": 0}
    failure_messages: list[str] = []

    def on_completed(result: CompletionResult) -> None:
        case = result.item[0]
        if result.succeeded:
            counters["completed"] += 1
            counters["new"] += 1
        else:
            counters["failed"] += 1
            message = (
                f"{case.case_id}: {type(result.error).__name__}: {result.error}"
            )
            failure_messages.append(message)
            append_jsonl(
                destination.with_suffix(".failures.jsonl"),
                {
                    "case_id": case.case_id,
                    "attempts": result.attempts,
                    "error": message,
                },
            )
        if progress:
            progress(
                f"batched outcomes: {counters['completed']}/{cases_seen} completed; "
                f"failed={counters['failed']}; request attempts this run={client.requests}"
            )

    pool_results = run_completion_pool(
        work_items,
        process_case,
        item_key=lambda item: item[0].case_id,
        max_concurrency=max_concurrency,
        max_attempts=max_case_attempts,
        retry_base_seconds=retry_base_seconds,
        retry_max_seconds=retry_max_seconds,
        retryable=_is_retryable_api_error,
        on_completed=on_completed,
    )
    completed_cases = counters["completed"]
    new_cases = counters["new"]
    status = "complete" if counters["failed"] == 0 else "partial_errors_resumable"
    stop_reason = failure_messages[0] if failure_messages else None
    _consolidate_case_outcomes(case_directory, destination)
    result = {
        "status": status,
        "stop_reason": stop_reason,
        "model": model_id,
        "cases_seen": cases_seen,
        "completed_cases": completed_cases,
        "new_cases": new_cases,
        "failed_cases": counters["failed"],
        "max_concurrency": max_concurrency,
        "scheduled_api_cases": len(work_items),
        "case_level_retry_count": sum(item.attempts - 1 for item in pool_results),
        "physical_requests_this_run": client.requests,
        "actual_cost_usd_this_run": client.actual_usd,
        "request_contract": (
            "one batched detection completion per case; transient failures may retry"
        ),
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


def _empty_detection_payload(case) -> dict[str, object]:
    return {
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


def _is_retryable_api_error(error: Exception) -> bool:
    message = f"{type(error).__name__}: {error}".lower()
    return any(
        marker in message
        for marker in (
            "http 429",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
            "timeout",
            "timed out",
            "connecterror",
            "connection error",
            "connection reset",
            "remoteprotocolerror",
            "server disconnected",
            "temporarily unavailable",
        )
    )


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
    train_cohort_manifest: str | Path | None = None,
    backends: Iterable[str] = ("multitask_mlp",),
    seed: int = 2026,
    target_truth_recall: float = 0.95,
    gate_fraction: float = 0.5,
    escalation_threshold: float = 0.85,
    cost_weight: float = 1.0,
    false_positive_weight: float = 0.5,
    unsupported_weight: float = 0.25,
    mlp_device: str = "auto",
    mlp_batch_size: int = 512,
    mlp_max_epochs: int = 100,
    mlp_patience: int = 12,
    mlp_learning_rate: float = 2e-3,
    mlp_weight_decay: float = 1e-4,
    progress: Callable[[str], None] | None = print,
) -> dict[str, object]:
    activate_parent_package()
    from llm_security.models import to_dict
    from llm_security.routing import (
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
    policy = UtilityPolicyConfig(
        escalation_threshold=escalation_threshold,
        cost_weight=cost_weight,
        false_positive_weight=false_positive_weight,
        unsupported_weight=unsupported_weight,
    )
    gate_rows, calibration_rows = split_gate_calibration_samples(
        dev_rows, seed=seed, gate_fraction=gate_fraction
    )
    artifact = require_within(artifact_path, EVALUATION_ROOT)
    requested_backends = tuple(dict.fromkeys(str(item) for item in backends))
    supported = {
        "logistic_regression",
        "gradient_boosting",
        "multitask_mlp",
    }
    unknown = sorted(set(requested_backends) - supported)
    if unknown or not requested_backends:
        raise ValueError("Unsupported Router backends: " + ", ".join(unknown))
    case_weights = _load_case_weights(train_cohort_manifest)
    variant_reports: dict[str, dict[str, object]] = {}
    trained_routers = {}
    for backend in requested_backends:
        if progress is not None:
            progress(f"Router backend training: {backend}")
        router = _fit_router_backend(
            train_rows,
            backend=backend,
            policy=policy,
            seed=seed,
            case_weights=case_weights,
            mlp_device=mlp_device,
            mlp_batch_size=mlp_batch_size,
            mlp_max_epochs=mlp_max_epochs,
            mlp_patience=mlp_patience,
            mlp_learning_rate=mlp_learning_rate,
            mlp_weight_decay=mlp_weight_decay,
            progress=progress,
        )
        if progress is not None:
            progress(
                f"{backend} complete; fitting the existing escalation gate on Dev."
            )
        gate_count = router.fit_escalation_gate(gate_rows, seed=seed)
        if progress is not None:
            progress("Escalation gate complete; calibrating the routing threshold.")
        calibration = router.calibrate_threshold(
            calibration_rows, target_truth_recall=target_truth_recall
        )
        baselines = router.calibrate_baselines(calibration_rows)
        metrics = router.evaluate(calibration_rows)
        variant_artifact = require_within(
            artifact.with_name(f"{artifact.stem}_{backend}{artifact.suffix}"),
            EVALUATION_ROOT,
        )
        router.save(variant_artifact)
        trained_routers[backend] = router
        variant_reports[backend] = {
            "artifact": str(variant_artifact),
            "gate_training_candidates": gate_count,
            "calibration": to_dict(calibration),
            "baseline_calibration": to_dict(baselines),
            "calibration_metrics": to_dict(metrics),
            "model_training": getattr(router.model, "training_summary", {}),
        }
    selected_backend = max(
        requested_backends,
        key=lambda backend: _router_selection_key(variant_reports[backend]),
    )
    trained_routers[selected_backend].save(artifact)
    report = {
        "artifact": str(artifact),
        "selected_backend": selected_backend,
        "models": models,
        "mlp_device_preference": mlp_device,
        "mlp_hyperparameters": {
            "batch_size": mlp_batch_size,
            "max_epochs": mlp_max_epochs,
            "patience": mlp_patience,
            "learning_rate": mlp_learning_rate,
            "weight_decay": mlp_weight_decay,
        },
        "train_matrix_audit": train_audit,
        "dev_matrix_audit": dev_audit,
        "selection_rule": (
            "dev feasible recall, truth recall, F1, lower assignments, lower Brier"
        ),
        "variants": variant_reports,
    }
    write_json(report_path, report)
    return report


def _fit_router_backend(
    rows: list,
    *,
    backend: str,
    policy,
    seed: int,
    case_weights: dict[str, float],
    mlp_device: str = "auto",
    mlp_batch_size: int = 512,
    mlp_max_epochs: int = 100,
    mlp_patience: int = 12,
    mlp_learning_rate: float = 2e-3,
    mlp_weight_decay: float = 1e-4,
    progress: Callable[[str], None] | None = print,
):
    activate_parent_package()
    from llm_security.routing import BudgetedUtilityRouter

    if backend == "logistic_regression":
        router = BudgetedUtilityRouter.fit(
            rows, policy=copy.deepcopy(policy), seed=seed
        )
        router.model.training_summary = {
            "backend": backend,
            "outcome_count": len(rows),
            "candidate_count": len({
                (row.case_id, row.candidate.candidate_id) for row in rows
            }),
        }
        return router
    from .router_models import (
        GradientBoostedUtilityRoutingModel,
        MultiTaskMLPUtilityRoutingModel,
    )

    # Build nonlinear backends directly. The previous implementation called
    # BudgetedUtilityRouter.fit first, which silently trained five LR heads
    # before replacing them with the requested MLP/GBDT model.
    materialized, assignments, statistics, feature_schema = (
        _router_training_metadata(rows)
    )
    if backend == "gradient_boosting":
        model = GradientBoostedUtilityRoutingModel(seed=seed).fit(
            [row.candidate.features for row in materialized],
            [row.assignment.assignment_id for row in materialized],
            [row.success for row in materialized],
        )
    elif backend == "multitask_mlp":
        model = MultiTaskMLPUtilityRoutingModel(
            seed=seed,
            device=mlp_device,
            batch_size=mlp_batch_size,
            max_epochs=mlp_max_epochs,
            patience=mlp_patience,
            learning_rate=mlp_learning_rate,
            weight_decay=mlp_weight_decay,
        ).fit_samples(
            materialized, case_weights=case_weights, progress=progress
        )
    else:  # pragma: no cover - validated by train_utility_router
        raise ValueError(f"Unsupported Router backend: {backend}")
    router = BudgetedUtilityRouter(
        model,
        assignments,
        statistics,
        policy=copy.deepcopy(policy),
        feature_schema_version=feature_schema,
    )
    router.training_project_ids = {
        sample.candidate.project_id for sample in materialized
    }
    return router


def _router_training_metadata(rows: list):
    """Validate outcomes and derive Router metadata without fitting an LR model."""
    activate_parent_package()
    from llm_security.datasets import UTILITY_OUTCOME_LABEL_VERSION
    from llm_security.models import ACTIVE_UTILITY_EXPERTS
    from llm_security.routing import AssignmentStatistics, BudgetedUtilityRouter

    materialized = [
        sample
        for sample in rows
        if sample.assignment.expert in ACTIVE_UTILITY_EXPERTS
    ]
    if not materialized:
        raise ValueError("Utility Router training requires five-Expert outcome samples")
    invalid_labels = [
        sample
        for sample in materialized
        if not sample.truth_labels_available
        or sample.label_version != UTILITY_OUTCOME_LABEL_VERSION
        or not sample.case_id
    ]
    if invalid_labels:
        raise ValueError(
            "Utility Router requires semantic outcome labels with case IDs. "
            f"Expected label_version={UTILITY_OUTCOME_LABEL_VERSION}; "
            "recollect legacy outcomes before training."
        )
    schemas = {
        sample.candidate.feature_schema_version for sample in materialized
    }
    if len(schemas) != 1:
        raise ValueError(
            "Utility samples must use exactly one feature schema; got: "
            + ", ".join(sorted(schemas))
        )
    if schemas != {BudgetedUtilityRouter.required_feature_schema_version}:
        raise ValueError(
            "Utility Router v4 requires static CWE hypothesis features with "
            f"schema={BudgetedUtilityRouter.required_feature_schema_version}; "
            "regenerate semantic candidates and recollect outcomes."
        )
    assignments = {
        sample.assignment.assignment_id: sample.assignment
        for sample in materialized
    }
    present = {assignment.expert for assignment in assignments.values()}
    missing = set(ACTIVE_UTILITY_EXPERTS) - present
    if missing:
        raise ValueError(
            "Utility outcomes are missing active Experts: "
            + ", ".join(sorted(family.value for family in missing))
        )

    expected = set(assignments)
    matrix_groups: dict[tuple[str, str], list] = defaultdict(list)
    for sample in materialized:
        matrix_groups[(sample.case_id, sample.candidate.candidate_id)].append(sample)
    for key, group in matrix_groups.items():
        assignment_ids = [item.assignment.assignment_id for item in group]
        if len(assignment_ids) != len(set(assignment_ids)):
            raise ValueError(f"Duplicate assignment outcome in candidate group {key}")
        present_ids = set(assignment_ids)
        if present_ids != expected:
            raise ValueError(
                f"Incomplete utility outcome matrix for candidate group {key}: "
                f"missing={sorted(expected - present_ids)}; "
                f"unexpected={sorted(present_ids - expected)}"
            )

    grouped: dict[str, list] = defaultdict(list)
    for sample in materialized:
        grouped[sample.assignment.assignment_id].append(sample)
    statistics = {}
    for assignment_id, samples in grouped.items():
        count = len(samples)
        statistics[assignment_id] = AssignmentStatistics(
            samples=count,
            success_rate=sum(item.success for item in samples) / count,
            false_positive_rate=sum(item.false_positive for item in samples) / count,
            unsupported_claim_rate=sum(
                item.unsupported_claims > 0 for item in samples
            ) / count,
            average_cost=sum(item.cost for item in samples) / count,
            average_prompt_tokens=sum(item.prompt_tokens for item in samples) / count,
            average_completion_tokens=sum(
                item.completion_tokens for item in samples
            ) / count,
            average_latency_seconds=sum(
                item.latency_seconds for item in samples
            ) / count,
        )
    return materialized, assignments, statistics, next(iter(schemas))


def _load_case_weights(path: str | Path | None) -> dict[str, float]:
    if path is None:
        return {}
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"Train cohort manifest does not exist: {source}")
    return {
        str(row["case_id"]): float(row.get("sampling_weight", 1.0))
        for row in iter_jsonl(source)
    }


def _router_selection_key(report: dict[str, object]) -> tuple:
    calibration = report["calibration"]
    metrics = report["calibration_metrics"]
    return (
        int(bool(calibration["feasible"])),
        float(metrics["truth_recall"]),
        float(metrics["outcome_f1"]),
        -float(metrics["average_assignments"]),
        -float(metrics["brier_score"]),
    )


def train_router_learning_curves(
    *,
    train_outcomes: str | Path,
    dev_outcomes: str | Path,
    train_cohort_manifest: str | Path,
    report_path: str | Path,
    sizes: Iterable[int] = (1_000, 2_000, 4_000, 6_000),
    backends: Iterable[str] = ("multitask_mlp",),
    seed: int = 2026,
    target_truth_recall: float = 0.95,
    gate_fraction: float = 0.5,
    mlp_device: str = "auto",
    mlp_batch_size: int = 512,
    mlp_max_epochs: int = 100,
    mlp_patience: int = 12,
    mlp_learning_rate: float = 2e-3,
    mlp_weight_decay: float = 1e-4,
    progress: Callable[[str], None] | None = print,
) -> dict[str, object]:
    """Compare nested cohort prefixes without making additional API calls."""
    activate_parent_package()
    from llm_security.models import to_dict
    from llm_security.routing import (
        UtilityPolicyConfig,
        split_gate_calibration_samples,
    )

    all_train = load_outcomes(train_outcomes)
    dev_rows = load_outcomes(dev_outcomes)
    manifest_rows = list(iter_jsonl(train_cohort_manifest))
    e6_ids = {
        str(row["case_id"])
        for row in manifest_rows
        if row["expert"] == "concurrency_toctou"
    }
    non_e6 = [
        str(row["case_id"])
        for row in sorted(manifest_rows, key=lambda row: int(row["selection_order"]))
        if str(row["case_id"]) not in e6_ids
    ]
    case_weights = _load_case_weights(train_cohort_manifest)
    gate_rows, calibration_rows = split_gate_calibration_samples(
        dev_rows, seed=seed, gate_fraction=gate_fraction
    )
    output: dict[str, object] = {}
    policy = UtilityPolicyConfig()
    for requested_size in sorted(set(int(value) for value in sizes if int(value) > 0)):
        if requested_size > len(manifest_rows) or requested_size < len(e6_ids):
            continue
        selected_cases = e6_ids | set(non_e6[: requested_size - len(e6_ids)])
        subset = [row for row in all_train if row.case_id in selected_cases]
        size_report: dict[str, object] = {
            "requested_cases": requested_size,
            "selected_cases_with_outcomes": len({row.case_id for row in subset}),
            "candidate_count": len({
                (row.case_id, row.candidate.candidate_id) for row in subset
            }),
            "e6_cases_preserved": len(e6_ids),
            "variants": {},
        }
        for backend in tuple(dict.fromkeys(str(item) for item in backends)):
            try:
                router = _fit_router_backend(
                    subset,
                    backend=backend,
                    policy=policy,
                    seed=seed,
                    case_weights=case_weights,
                    mlp_device=mlp_device,
                    mlp_batch_size=mlp_batch_size,
                    mlp_max_epochs=mlp_max_epochs,
                    mlp_patience=mlp_patience,
                    mlp_learning_rate=mlp_learning_rate,
                    mlp_weight_decay=mlp_weight_decay,
                    progress=progress,
                )
                router.fit_escalation_gate(gate_rows, seed=seed)
                calibration = router.calibrate_threshold(
                    calibration_rows, target_truth_recall=target_truth_recall
                )
                metrics = router.evaluate(calibration_rows)
                size_report["variants"][backend] = {
                    "status": "complete",
                    "calibration": to_dict(calibration),
                    "metrics": to_dict(metrics),
                    "model_training": getattr(router.model, "training_summary", {}),
                }
            except (RuntimeError, TypeError, ValueError) as error:
                size_report["variants"][backend] = {
                    "status": "unavailable",
                    "error": f"{type(error).__name__}: {error}",
                }
        output[str(requested_size)] = size_report
    report = {
        "nested_subset_policy": (
            "all E6 cases plus deterministic weighted-fair cohort prefix"
        ),
        "sizes": output,
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
    per_expert_end_to_end = _per_expected_expert_end_to_end(
        router,
        rows,
        test_cases,
        candidate_cache,
        max_candidates_per_case=max_candidates_per_case,
    )
    macro_end_to_end = _macro_expected_expert_metrics(per_expert_end_to_end)
    detection_path = Path(test_outcomes).with_suffix(".detections.jsonl")
    stage_diagnostics = None
    if detection_path.is_file():
        from .diagnostics import audit_detection_stages

        stage_diagnostics = audit_detection_stages(
            cases_path=test_cases,
            candidate_cache=candidate_cache,
            detection_path=detection_path,
            max_candidates_per_case=max_candidates_per_case,
        )
    report = {
        "artifact": str(Path(artifact_path).resolve()),
        "test_outcomes": str(Path(test_outcomes).resolve()),
        "test_matrix_audit": matrix_audit,
        "policies": to_dict(policy_metrics),
        "policy_negative_candidate_false_positive_rates": false_positive_rates,
        "end_to_end": to_dict(e2e),
        "per_expected_expert_end_to_end": per_expert_end_to_end,
        "macro_expected_expert_end_to_end": macro_end_to_end,
        "stage_diagnostics": stage_diagnostics,
    }
    write_json(report_path, report)
    return report


def _per_expected_expert_end_to_end(
    router,
    rows: list,
    cases_path: str | Path,
    candidate_cache: str | Path,
    *,
    max_candidates_per_case: int,
) -> dict[str, dict[str, float | int]]:
    activate_parent_package()
    from llm_security.models import ACTIVE_UTILITY_EXPERTS

    by_candidate: dict[tuple[str, str], list] = {}
    for row in rows:
        by_candidate.setdefault(
            (row.case_id, row.candidate.candidate_id), []
        ).append(row)
    analyzer = StreamingCachedCandidateAnalyzer(candidate_cache)
    stats = {
        expert.value: {
            "case_count": 0,
            "ground_truth_count": 0,
            "analyzer_truth_hits": 0,
            "outcome_matrix_truth_hits": 0,
            "routed_truth_hits": 0,
            "validated_true_findings": 0,
            "validated_false_findings": 0,
        }
        for expert in ACTIVE_UTILITY_EXPERTS
    }
    for case in load_cases(cases_path):
        expected = str(case.metadata.get("expected_expert", ""))
        if expected not in stats:
            continue
        bucket = stats[expected]
        bucket["case_count"] += 1
        bucket["ground_truth_count"] += len(case.ground_truth)
        candidates = analyzer.analyze(case)
        analyzer_hits = {
            truth.truth_id
            for truth in case.ground_truth
            if any(_candidate_matches_truth(candidate, truth) for candidate in candidates)
        }
        bucket["analyzer_truth_hits"] += len(analyzer_hits)
        candidates.sort(
            key=lambda candidate: (
                -candidate.suspicion_score,
                candidate.file,
                candidate.line_start,
                candidate.candidate_id,
            )
        )
        matrix_truths: set[str] = set()
        routed_truths: set[str] = set()
        for candidate in candidates[:max_candidates_per_case]:
            outcomes = by_candidate.get((case.case_id, candidate.candidate_id), [])
            if not outcomes:
                continue
            matrix_truths.update(
                truth_id for row in outcomes for truth_id in row.ground_truth_ids
            )
            selected_ids = {
                assignment.assignment_id
                for assignment in router.route(candidate).assignments
            }
            selected = [
                row
                for row in outcomes
                if row.assignment.assignment_id in selected_ids
            ]
            routed_truths.update(
                truth_id for row in selected for truth_id in row.matched_truth_ids
            )
            bucket["validated_true_findings"] += sum(
                row.validated_true_findings or int(row.success) for row in selected
            )
            bucket["validated_false_findings"] += sum(
                row.validated_false_findings or int(row.false_positive)
                for row in selected
            )
        truth_ids = {truth.truth_id for truth in case.ground_truth}
        bucket["outcome_matrix_truth_hits"] += len(matrix_truths & truth_ids)
        bucket["routed_truth_hits"] += len(routed_truths & truth_ids)
    report: dict[str, dict[str, float | int]] = {}
    for expert, bucket in stats.items():
        truths = int(bucket["ground_truth_count"])
        true_findings = int(bucket["validated_true_findings"])
        false_findings = int(bucket["validated_false_findings"])
        precision = (
            true_findings / (true_findings + false_findings)
            if true_findings + false_findings
            else 1.0
        )
        recall = int(bucket["routed_truth_hits"]) / truths if truths else 1.0
        report[expert] = {
            **bucket,
            "analyzer_candidate_recall": (
                int(bucket["analyzer_truth_hits"]) / truths if truths else 1.0
            ),
            "outcome_matrix_gt_coverage": (
                int(bucket["outcome_matrix_truth_hits"]) / truths if truths else 1.0
            ),
            "routed_detection_recall": recall,
            "validated_outcome_precision": precision,
            "end_to_end_f1": (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            ),
        }
    return report


def _macro_expected_expert_metrics(
    report: dict[str, dict[str, float | int]],
) -> dict[str, float]:
    fields = (
        "analyzer_candidate_recall",
        "outcome_matrix_gt_coverage",
        "routed_detection_recall",
        "validated_outcome_precision",
        "end_to_end_f1",
    )
    rows = [row for row in report.values() if int(row["case_count"]) > 0]
    return {
        field: sum(float(row[field]) for row in rows) / len(rows)
        if rows
        else 0.0
        for field in fields
    }


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
