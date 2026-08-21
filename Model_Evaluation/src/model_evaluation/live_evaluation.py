from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

from .adapters.llm_security import (
    activate_parent_package,
    app_config,
    candidate_from_dict,
    load_cases,
)
from .api_budget import LoggedLLMClient
from .candidates import StreamingCachedCandidateAnalyzer
from .jsonl import append_jsonl, iter_jsonl
from .paths import EVALUATION_ROOT, require_within, write_json


def run_live_detection(
    *,
    env_file: str | Path,
    artifact_path: str | Path,
    cases_path: str | Path,
    candidate_cache: str | Path,
    output_path: str | Path,
    ledger_path: str | Path,
    max_cases: int = 0,
    max_candidates_per_case: int = 4,
    candidate_gate_enabled: bool = False,
    candidate_gate_threshold: float = 0.4,
    progress: Callable[[str], None] | None = print,
) -> dict[str, object]:
    """Run the learned Router and real Expert APIs, checkpointing per case."""
    activate_parent_package()
    from llm_security.aggregation import FindingAggregator
    from llm_security.experiments.outcome_matching import FindingTruthMatcher
    from llm_security.experts import BatchedExpertRunner
    from llm_security.factory import build_context_builder, build_openrouter_client
    from llm_security.models import to_dict
    from llm_security.pipeline import VulnerabilityPipeline
    from llm_security.routing import BudgetedUtilityRouter, CandidateGate
    from llm_security.validation import EvidenceValidator

    config = _api_config(env_file)
    destination = require_within(output_path, EVALUATION_ROOT)
    ledger = require_within(ledger_path, EVALUATION_ROOT)
    completed = {
        str(row["case_id"]) for row in iter_jsonl(destination)
    } if destination.is_file() else set()
    client = LoggedLLMClient(build_openrouter_client(config), ledger)
    router = BudgetedUtilityRouter.load(artifact_path)
    cached_analyzer = StreamingCachedCandidateAnalyzer(candidate_cache)
    pipeline = VulnerabilityPipeline(
        analyzer=cached_analyzer,
        router=router,
        expert_runner=BatchedExpertRunner(
            client=client,
            model=config.model.expert_model,
            context_builder=build_context_builder(config),
            max_batch_characters=1_000_000,
            max_tasks=max_candidates_per_case * 5,
        ),
        aggregator=FindingAggregator(),
        validator=EvidenceValidator(
            minimum_confidence=config.validation.minimum_confidence,
            use_llm_for_uncertain=False,
            falsify_all_supported=False,
        ),
        candidate_gate=CandidateGate(
            enabled=candidate_gate_enabled,
            threshold=candidate_gate_threshold,
        ),
        max_candidates=max_candidates_per_case,
    )
    matcher = FindingTruthMatcher()
    status = "complete"
    stop_reason = None
    seen = written = 0
    try:
        for case in load_cases(cases_path):
            if max_cases and seen >= max_cases:
                break
            seen += 1
            if case.case_id in completed:
                cached_analyzer.analyze(case)
                continue
            result = pipeline.run(case)
            candidates = {item.candidate_id: item for item in result.candidates}
            matched_by_finding = {
                finding.finding_id: sorted(
                    truth.truth_id
                    for truth in case.ground_truth
                    if matcher.matches(finding, truth, candidates[finding.candidate_id])
                )
                for finding in result.validated_findings
                if finding.candidate_id in candidates
            }
            append_jsonl(
                destination,
                {
                    "case_id": case.case_id,
                    "project_id": case.project_id,
                    "split": case.split,
                    "ground_truth_ids": [item.truth_id for item in case.ground_truth],
                    "matched_truth_ids": sorted({
                        truth_id
                        for values in matched_by_finding.values()
                        for truth_id in values
                    }),
                    "matched_truth_ids_by_finding": matched_by_finding,
                    "raw_package_path": case.metadata.get("raw_package_path"),
                    "raw_to_virtual": case.metadata.get("raw_to_virtual", {}),
                    "pipeline_result": to_dict(result),
                },
            )
            completed.add(case.case_id)
            written += 1
            if progress:
                progress(f"live detection: {seen} seen / {written} new")
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        status = "stopped_on_error_resumable"
        stop_reason = str(error)
    report = summarize_live_detection(destination)
    report.update(
        {
            "status": status,
            "stop_reason": stop_reason,
            "physical_requests_this_run": client.requests,
            "actual_cost_usd_this_run": client.actual_usd,
            "request_contract": "at most one physical detection request per case",
            "output": str(destination),
        }
    )
    write_json(destination.with_suffix(".summary.json"), report)
    return report


def summarize_live_detection(path: str | Path) -> dict[str, object]:
    case_count = truth_count = truth_hits = validated = matched_findings = 0
    cost = 0.0
    prompt_tokens = completion_tokens = 0
    for row in iter_jsonl(path):
        case_count += 1
        truth_count += len(row.get("ground_truth_ids", []))
        truth_hits += len(row.get("matched_truth_ids", []))
        result = row["pipeline_result"]
        accepted = {
            item["finding_id"]
            for item in result.get("validations", [])
            if item.get("verdict") == "validated"
        }
        validated += len(accepted)
        matched_findings += sum(
            bool(values)
            for finding_id, values in row.get("matched_truth_ids_by_finding", {}).items()
            if finding_id in accepted
        )
        for usage in result.get("usage", []):
            cost += float(usage.get("cost", 0.0))
            prompt_tokens += int(usage.get("prompt_tokens", 0))
            completion_tokens += int(usage.get("completion_tokens", 0))
    precision = matched_findings / validated if validated else 1.0
    recall = truth_hits / truth_count if truth_count else 1.0
    return {
        "case_count": case_count,
        "ground_truth_count": truth_count,
        "matched_ground_truth_count": truth_hits,
        "validated_finding_count": validated,
        "ground_truth_matched_finding_count": matched_findings,
        "validated_precision": precision,
        "ground_truth_recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "realized_cost_usd": cost,
    }


def run_patch_evaluation(
    *,
    env_file: str | Path,
    detection_path: str | Path,
    output_path: str | Path,
    ledger_path: str | Path,
    commands: list[dict],
    max_findings: int = 0,
    max_attempts: int = 2,
    only_ground_truth_matched: bool = True,
    progress: Callable[[str], None] | None = print,
) -> dict[str, object]:
    """Generate and verify patches in temporary copies of raw Juliet packages."""
    if not commands:
        raise ValueError(
            "At least one compile/test verification command is required; patch apply alone "
            "must not be reported as a successful security repair."
        )
    activate_parent_package()
    from llm_security.factory import build_openrouter_client
    from llm_security.models import (
        ExpertFamily,
        Finding,
        ValidationResult,
        ValidationVerdict,
        to_dict,
    )
    from llm_security.patching import LLMPatchAgent, PatchProposal
    from llm_security.repair import RepairWorkflow
    from llm_security.verification import TemporaryPatchVerifier, VerificationCommand

    config = _api_config(env_file)
    destination = require_within(output_path, EVALUATION_ROOT)
    ledger = require_within(ledger_path, EVALUATION_ROOT)
    completed = {
        str(row["finding_id"]) for row in iter_jsonl(destination)
    } if destination.is_file() else set()
    client = LoggedLLMClient(build_openrouter_client(config), ledger)
    verify_commands = [VerificationCommand(
        name=str(item["name"]),
        command=[str(value) for value in item["command"]],
        timeout_seconds=float(item.get("timeout_seconds", 300.0)),
    ) for item in commands]
    status = "complete"
    stop_reason = None
    attempted = 0
    try:
        for detection in iter_jsonl(detection_path):
            raw_result = detection["pipeline_result"]
            candidates = {
                raw["candidate_id"]: candidate_from_dict(raw)
                for raw in raw_result.get("candidates", [])
            }
            validations = {
                raw["finding_id"]: ValidationResult(
                    finding_id=raw["finding_id"],
                    verdict=ValidationVerdict(raw["verdict"]),
                    confidence=float(raw["confidence"]),
                    checks=dict(raw.get("checks", {})),
                    reasons=[str(value) for value in raw.get("reasons", [])],
                    model_used=raw.get("model_used"),
                )
                for raw in raw_result.get("validations", [])
            }
            matched = detection.get("matched_truth_ids_by_finding", {})
            for raw in raw_result.get("findings", []):
                finding_id = str(raw["finding_id"])
                if finding_id in completed:
                    continue
                validation = validations.get(finding_id)
                if validation is None or validation.verdict.value != "validated":
                    continue
                if only_ground_truth_matched and not matched.get(finding_id):
                    continue
                if max_findings and attempted >= max_findings:
                    raise _PatchLimitReached
                finding = Finding(
                    finding_id=finding_id,
                    candidate_id=str(raw["candidate_id"]),
                    expert=ExpertFamily(raw["expert"]),
                    title=str(raw["title"]),
                    root_cause=str(raw["root_cause"]),
                    consequence=str(raw["consequence"]),
                    file=str(raw["file"]),
                    function=str(raw["function"]),
                    line_start=int(raw["line_start"]),
                    line_end=int(raw["line_end"]),
                    cwes=[str(value) for value in raw.get("cwes", [])],
                    source=raw.get("source"),
                    sink=raw.get("sink"),
                    missing_guard=raw.get("missing_guard"),
                    trigger_path=[str(value) for value in raw.get("trigger_path", [])],
                    evidence_ids=[str(value) for value in raw.get("evidence_ids", [])],
                    confidence=float(raw["confidence"]),
                    preconditions=[str(value) for value in raw.get("preconditions", [])],
                    evidence_for=[str(value) for value in raw.get("evidence_for", [])],
                    evidence_against=[str(value) for value in raw.get("evidence_against", [])],
                    falsification_test=raw.get("falsification_test"),
                    model_id=raw.get("model_id"),
                    prompt_version=raw.get("prompt_version"),
                )
                candidate = candidates[finding.candidate_id]
                path_map = {
                    str(virtual): str(raw_path)
                    for raw_path, virtual in detection.get("raw_to_virtual", {}).items()
                }
                primary = _RemappingPatchAgent(
                    LLMPatchAgent(client, config.model.patch_model), path_map, PatchProposal
                )
                strong = (
                    _RemappingPatchAgent(
                        LLMPatchAgent(client, config.model.strong_model), path_map, PatchProposal
                    )
                    if config.model.strong_model and max_attempts > 1
                    else None
                )
                workflow = RepairWorkflow(
                    primary,
                    TemporaryPatchVerifier(),
                    strong_agent=strong,
                    max_attempts=max_attempts,
                )
                result = workflow.run(
                    source_project=Path(str(detection["raw_package_path"])),
                    finding=finding,
                    validation=validation,
                    candidate=candidate,
                    commands=verify_commands,
                )
                append_jsonl(destination, {
                    "case_id": detection["case_id"],
                    "finding_id": finding_id,
                    "matched_truth_ids": matched.get(finding_id, []),
                    "repaired": result.repaired,
                    "repair_result": to_dict(result),
                })
                completed.add(finding_id)
                attempted += 1
                if progress:
                    progress(f"patch evaluation: {attempted} new findings")
    except _PatchLimitReached:
        status = "configured_limit_reached_resumable"
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        status = "stopped_on_error_resumable"
        stop_reason = str(error)
    report = summarize_patch_evaluation(destination)
    report.update({
        "status": status,
        "stop_reason": stop_reason,
        "physical_requests_this_run": client.requests,
        "actual_cost_usd_this_run": client.actual_usd,
        "verification_commands": commands,
        "output": str(destination),
    })
    write_json(destination.with_suffix(".summary.json"), report)
    return report


def summarize_patch_evaluation(path: str | Path) -> dict[str, object]:
    rows = list(iter_jsonl(path)) if Path(path).is_file() else []
    repaired = sum(bool(row.get("repaired")) for row in rows)
    attempts = [attempt for row in rows for attempt in row["repair_result"]["attempts"]]
    applied = sum(
        bool(attempt["verification"].get("patch_applied")) for attempt in attempts
    )
    return {
        "finding_count": len(rows),
        "repaired_finding_count": repaired,
        "verified_repair_rate": repaired / len(rows) if rows else 0.0,
        "attempt_count": len(attempts),
        "patch_apply_rate_per_attempt": applied / len(attempts) if attempts else 0.0,
    }


def run_batched_patch_evaluation(
    *,
    env_file: str | Path,
    detection_path: str | Path,
    output_path: str | Path,
    ledger_path: str | Path,
    commands: list[dict],
    max_cases: int = 0,
    progress: Callable[[str], None] | None = print,
) -> dict[str, object]:
    """Patch all validated findings from one Juliet case in one API request."""
    activate_parent_package()
    from llm_security.factory import build_openrouter_client
    from llm_security.models import (
        ExpertFamily,
        Finding,
        ValidationResult,
        ValidationVerdict,
        to_dict,
    )
    from llm_security.patching import BatchPatchProposal, LLMBatchPatchAgent
    from llm_security.verification import TemporaryPatchVerifier, VerificationCommand

    config = _api_config(env_file)
    destination = require_within(output_path, EVALUATION_ROOT)
    ledger = require_within(ledger_path, EVALUATION_ROOT)
    completed = {
        str(row["case_id"]) for row in iter_jsonl(destination)
    } if destination.is_file() else set()
    client = LoggedLLMClient(build_openrouter_client(config), ledger)
    agent = LLMBatchPatchAgent(client, config.model.patch_model)
    verifier = TemporaryPatchVerifier()
    verify_commands = [
        VerificationCommand(
            name=str(item["name"]),
            command=[str(value) for value in item["command"]],
            timeout_seconds=float(item.get("timeout_seconds", 300.0)),
        )
        for item in commands
    ]
    status = "complete"
    stop_reason = None
    new_cases = 0
    for detection in iter_jsonl(detection_path):
        case_id = str(detection["case_id"])
        if case_id in completed:
            continue
        if max_cases and new_cases >= max_cases:
            status = "configured_limit_reached_resumable"
            break
        raw_result = detection["pipeline_result"]
        candidates = {
            raw["candidate_id"]: candidate_from_dict(raw)
            for raw in raw_result.get("candidates", [])
        }
        validations = {
            raw["finding_id"]: ValidationResult(
                finding_id=raw["finding_id"],
                verdict=ValidationVerdict(raw["verdict"]),
                confidence=float(raw["confidence"]),
                checks=dict(raw.get("checks", {})),
                reasons=[str(value) for value in raw.get("reasons", [])],
                model_used=raw.get("model_used"),
            )
            for raw in raw_result.get("validations", [])
        }
        matched = detection.get("matched_truth_ids_by_finding", {})
        items = []
        for raw in raw_result.get("findings", []):
            finding_id = str(raw["finding_id"])
            validation = validations.get(finding_id)
            if (
                validation is None
                or validation.verdict.value != "validated"
                or not matched.get(finding_id)
            ):
                continue
            finding = Finding(
                finding_id=finding_id,
                candidate_id=str(raw["candidate_id"]),
                expert=ExpertFamily(raw["expert"]),
                title=str(raw["title"]),
                root_cause=str(raw["root_cause"]),
                consequence=str(raw["consequence"]),
                file=str(raw["file"]),
                function=str(raw["function"]),
                line_start=int(raw["line_start"]),
                line_end=int(raw["line_end"]),
                cwes=[str(value) for value in raw.get("cwes", [])],
                source=raw.get("source"),
                sink=raw.get("sink"),
                missing_guard=raw.get("missing_guard"),
                trigger_path=[str(value) for value in raw.get("trigger_path", [])],
                evidence_ids=[str(value) for value in raw.get("evidence_ids", [])],
                confidence=float(raw["confidence"]),
                preconditions=[str(value) for value in raw.get("preconditions", [])],
                evidence_for=[str(value) for value in raw.get("evidence_for", [])],
                evidence_against=[str(value) for value in raw.get("evidence_against", [])],
                falsification_test=raw.get("falsification_test"),
                model_id=raw.get("model_id"),
                prompt_version=raw.get("prompt_version"),
            )
            items.append((finding, validation, candidates[finding.candidate_id]))
        try:
            if not items:
                append_jsonl(
                    destination,
                    {
                        "case_id": case_id,
                        "attempted": False,
                        "repaired": False,
                        "finding_ids": [],
                        "reason": "no ground-truth-matched validated finding",
                    },
                )
            else:
                proposal = agent.propose(items)
                path_map = {
                    str(virtual): str(raw_path)
                    for raw_path, virtual in detection.get("raw_to_virtual", {}).items()
                }
                remapped = BatchPatchProposal(
                    finding_ids=proposal.finding_ids,
                    unified_diff=_remap_diff(proposal.unified_diff, path_map),
                    summary=proposal.summary,
                    model=proposal.model,
                    usage=proposal.usage,
                )
                verification = verifier.verify(
                    Path(str(detection["raw_package_path"])),
                    remapped,
                    verify_commands,
                )
                append_jsonl(
                    destination,
                    {
                        "case_id": case_id,
                        "attempted": True,
                        "repaired": verification.fully_verified,
                        "finding_ids": proposal.finding_ids,
                        "proposal": to_dict(remapped),
                        "verification": to_dict(verification),
                    },
                )
            completed.add(case_id)
            new_cases += 1
            if progress:
                progress(
                    f"batched patches: {new_cases} new cases; "
                    f"physical patch requests={client.requests}"
                )
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            status = "stopped_on_error_resumable"
            stop_reason = f"{case_id}: {type(error).__name__}: {error}"
            break
    rows = list(iter_jsonl(destination)) if destination.is_file() else []
    attempted = [row for row in rows if row.get("attempted")]
    repaired = sum(bool(row.get("repaired")) for row in attempted)
    report = {
        "status": status,
        "stop_reason": stop_reason,
        "case_count": len(rows),
        "attempted_patch_case_count": len(attempted),
        "verification_pass_case_count": repaired,
        "verification_pass_rate": repaired / len(attempted) if attempted else 0.0,
        "physical_patch_requests_this_run": client.requests,
        "actual_cost_usd_this_run": client.actual_usd,
        "request_contract": "at most one physical patch request per case",
        "output": str(destination),
    }
    write_json(destination.with_suffix(".summary.json"), report)
    return report


class _PatchLimitReached(Exception):
    pass


class _RemappingPatchAgent:
    def __init__(self, agent, path_map: dict[str, str], proposal_type) -> None:
        self.agent = agent
        self.path_map = path_map
        self.proposal_type = proposal_type

    def propose(self, finding, validation, candidate, *, previous_failure=None):
        proposal = self.agent.propose(
            finding, validation, candidate, previous_failure=previous_failure
        )
        diff = _remap_diff(proposal.unified_diff, self.path_map)
        return self.proposal_type(
            finding_id=proposal.finding_id,
            unified_diff=diff,
            summary=proposal.summary,
            model=proposal.model,
            usage=proposal.usage,
        )


def _remap_diff(diff: str, path_map: dict[str, str]) -> str:
    for virtual, raw in sorted(path_map.items(), key=lambda item: -len(item[0])):
        diff = diff.replace(f"a/{virtual}", f"a/{raw}")
        diff = diff.replace(f"b/{virtual}", f"b/{raw}")
        diff = diff.replace(f"--- {virtual}", f"--- {raw}")
        diff = diff.replace(f"+++ {virtual}", f"+++ {raw}")
    return diff


def _api_config(env_file: str | Path):
    config = app_config(env_file)
    if not config.model.api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing from .env")
    config.model.max_output_tokens = max(config.model.max_output_tokens, 8_192)
    return config
