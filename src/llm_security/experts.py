from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .evidence import ContextBuilder
from .llm import LLMClient
from .models import (
    Candidate,
    ExpertAssignment,
    ExpertFamily,
    Finding,
    RouteDecision,
    UsageRecord,
)
from .prompts import (
    batched_expert_messages,
    batched_findings_schema,
    expert_messages,
    finding_from_payload,
    findings_schema,
)


@dataclass(slots=True)
class ExpertRunOutput:
    findings: list[Finding]
    usage: list[UsageRecord]
    errors: list[str]
    task_count: int = 0
    submitted_task_count: int = 0
    skipped_task_count: int = 0


class ExpertRunner:
    def __init__(
        self,
        client: LLMClient,
        model: str,
        context_builder: ContextBuilder,
        models_by_family: dict[ExpertFamily, str] | None = None,
        prompt_version: str = "expert-v4-cwe-hypothesis",
    ) -> None:
        self.client = client
        self.model = model
        self.context_builder = context_builder
        self.models_by_family = dict(models_by_family or {})
        self.prompt_version = prompt_version

    def run(
        self,
        candidates: list[Candidate],
        routes: list[RouteDecision],
    ) -> ExpertRunOutput:
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        findings: list[Finding] = []
        usage: list[UsageRecord] = []
        errors: list[str] = []
        task_count = 0
        for route in routes:
            candidate = by_id[route.candidate_id]
            assignments = route.assignments or [
                ExpertAssignment(
                    expert=expert,
                    model_id=self.models_by_family.get(expert, self.model),
                    prompt_version=self.prompt_version,
                )
                for expert in route.selected
            ]
            for assignment in assignments:
                task_count += 1
                expert = assignment.expert
                context = self.context_builder.build(candidate, expert)
                try:
                    response = self.client.complete(
                        model=assignment.model_id,
                        messages=expert_messages(candidate, context),
                        response_schema=findings_schema(),
                        metadata={
                            "task": "expert",
                            "candidate": candidate,
                            "expert": expert,
                            "assignment": assignment,
                        },
                    )
                    usage.append(response.usage)
                    payloads = response.data.get("findings", [])
                    if not isinstance(payloads, list):
                        raise TypeError("The model response 'findings' field must be a list")
                    for index, payload in enumerate(payloads, start=1):
                        findings.append(
                            finding_from_payload(
                                payload,
                                index=index,
                                candidate=candidate,
                                expert=expert,
                                model_id=assignment.model_id,
                                prompt_version=assignment.prompt_version,
                            )
                        )
                except (KeyError, TypeError, ValueError, RuntimeError) as error:
                    errors.append(
                        f"{candidate.candidate_id}/{expert.value}/"
                        f"{assignment.model_id}: {error}"
                    )
        return ExpertRunOutput(
            findings=findings,
            usage=usage,
            errors=errors,
            task_count=task_count,
            submitted_task_count=task_count,
        )


class BatchedExpertRunner:
    """Execute all Router-selected logical Experts in one LLM completion.

    Expert identities, prompts, evidence scopes, and output attribution remain
    independent. A single physical model processes the complete panel request,
    which guarantees at most one successful detection completion per run.
    """

    prompt_version = "batched-expert-v3-cwe-hypothesis"

    def __init__(
        self,
        client: LLMClient,
        model: str,
        context_builder: ContextBuilder,
        *,
        max_batch_characters: int = 120_000,
        max_tasks: int = 24,
    ) -> None:
        if max_batch_characters < 1:
            raise ValueError("max_batch_characters must be positive")
        if max_tasks < 1:
            raise ValueError("max_tasks must be positive")
        self.client = client
        self.model = model
        self.context_builder = context_builder
        self.max_batch_characters = max_batch_characters
        self.max_tasks = max_tasks

    def run(
        self,
        candidates: list[Candidate],
        routes: list[RouteDecision],
    ) -> ExpertRunOutput:
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        routes_by_id = {route.candidate_id: route for route in routes}
        skipped = 0
        desired: dict[str, list[ExpertFamily]] = {}
        for candidate in candidates:
            route = routes_by_id[candidate.candidate_id]
            desired[candidate.candidate_id] = list(
                dict.fromkeys(
                    assignment.expert for assignment in route.assignments
                )
            ) or list(dict.fromkeys(route.selected))
        task_count = sum(len(experts) for experts in desired.values())

        # Two-pass project budget: reserve every candidate's Top-2 work before
        # spending tasks on any Full-5 escalation extras.  Escalation extras are
        # then ordered by lowest sufficiency confidence and highest static
        # suspicion so an early Full-5 candidate cannot starve later Top-2 work.
        allocations: dict[str, list[ExpertFamily]] = {
            candidate.candidate_id: [] for candidate in candidates
        }
        candidate_order = sorted(
            enumerate(candidates),
            key=lambda item: (-item[1].suspicion_score, item[0]),
        )
        base_chunks = [
            (candidate, desired[candidate.candidate_id][:2])
            for _, candidate in candidate_order
            if desired[candidate.candidate_id]
        ]
        extra_chunks = sorted(
            [
                (candidate, [expert])
                for candidate in candidates
                for expert in desired[candidate.candidate_id][2:]
            ],
            key=lambda item: (
                routes_by_id[item[0].candidate_id].escalation_confidence
                if routes_by_id[item[0].candidate_id].escalation_confidence is not None
                else 1.0,
                -item[0].suspicion_score,
                item[0].candidate_id,
                item[1][0].value,
            ),
        )
        for candidate, experts in [*base_chunks, *extra_chunks]:
            if not experts:
                continue
            # Never submit escalation extras when that candidate's Top-2 pair
            # could not be admitted to the shared request.
            if experts[0] in desired[candidate.candidate_id][2:] and len(
                allocations[candidate.candidate_id]
            ) < min(2, len(desired[candidate.candidate_id])):
                skipped += len(experts)
                continue
            proposed = {
                candidate_id: list(selected)
                for candidate_id, selected in allocations.items()
            }
            proposed[candidate.candidate_id].extend(experts)
            proposed_count = sum(len(selected) for selected in proposed.values())
            if proposed_count > self.max_tasks:
                skipped += len(experts)
                continue
            proposed_packets, proposed_lookup = self._build_packets(
                candidates, proposed
            )
            prompt_size = sum(
                len(message["content"])
                for message in batched_expert_messages(proposed_packets)
            )
            if prompt_size > self.max_batch_characters:
                skipped += len(experts)
                continue
            allocations = proposed

        packets, task_lookup = self._build_packets(candidates, allocations)

        if not task_lookup:
            errors = []
            if skipped:
                errors.append(
                    f"All {skipped} Expert tasks exceeded the batch prompt budget "
                    f"({self.max_batch_characters} characters or {self.max_tasks} tasks)."
                )
            return ExpertRunOutput(
                findings=[],
                usage=[],
                errors=errors,
                task_count=task_count,
                submitted_task_count=0,
                skipped_task_count=skipped,
            )

        response = self.client.complete(
            model=self.model,
            messages=batched_expert_messages(packets),
            response_schema=batched_findings_schema(),
            metadata={
                "task": "batched_experts",
                "task_count": len(task_lookup),
                "candidate_count": len(packets),
            },
        )
        findings: list[Finding] = []
        errors: list[str] = []
        seen: set[str] = set()
        reviewed = {
            str(task_id)
            for task_id in response.data.get("reviewed_task_ids", [])
        }
        unknown_reviewed = sorted(reviewed - set(task_lookup))
        if unknown_reviewed:
            errors.append(
                "Model returned unknown reviewed task IDs: "
                + ", ".join(unknown_reviewed)
            )
        missing_reviews = sorted(set(task_lookup) - reviewed)
        if missing_reviews:
            errors.append(
                "Model did not confirm review of Expert tasks: "
                + ", ".join(missing_reviews)
            )
        payloads = response.data.get("expert_results", [])
        if not isinstance(payloads, list):
            raise TypeError("The model response 'expert_results' field must be a list")
        for payload in payloads:
            try:
                task_id = str(payload["task_id"])
                if task_id in seen:
                    raise ValueError(f"Duplicate Expert result: {task_id}")
                seen.add(task_id)
                if task_id not in reviewed:
                    raise ValueError(f"Unreviewed Expert result: {task_id}")
                candidate, expert = task_lookup[task_id]
                if str(payload["candidate_id"]) != candidate.candidate_id:
                    raise ValueError(f"Candidate mismatch for {task_id}")
                if ExpertFamily(str(payload["expert"])) != expert:
                    raise ValueError(f"Expert mismatch for {task_id}")
                task_findings = payload.get("findings", [])
                if not isinstance(task_findings, list):
                    raise TypeError(f"Findings for {task_id} must be a list")
                for index, finding_payload in enumerate(task_findings, start=1):
                    findings.append(
                        finding_from_payload(
                            finding_payload,
                            index=index,
                            candidate=candidate,
                            expert=expert,
                            model_id=response.usage.model,
                            prompt_version=self.prompt_version,
                        )
                    )
            except (KeyError, TypeError, ValueError) as error:
                errors.append(str(error))
        if skipped:
            errors.append(
                f"Skipped {skipped} Expert tasks because the batch prompt exceeded "
                f"{self.max_batch_characters} characters or {self.max_tasks} tasks."
            )
        return ExpertRunOutput(
            findings=findings,
            usage=[response.usage],
            errors=errors,
            task_count=task_count,
            submitted_task_count=len(task_lookup),
            skipped_task_count=skipped,
        )

    def _build_packets(
        self,
        candidates: list[Candidate],
        allocations: dict[str, list[ExpertFamily]],
    ) -> tuple[list[dict[str, Any]], dict[str, tuple[Candidate, ExpertFamily]]]:
        packets: list[dict[str, Any]] = []
        lookup: dict[str, tuple[Candidate, ExpertFamily]] = {}
        for candidate in candidates:
            experts = allocations.get(candidate.candidate_id, [])
            if not experts:
                continue
            packet, packet_tasks = self._candidate_packet(
                candidate,
                experts,
                start_index=len(lookup) + 1,
            )
            packets.append(packet)
            lookup.update(packet_tasks)
        return packets, lookup

    def _candidate_packet(
        self,
        candidate: Candidate,
        experts: list[ExpertFamily],
        *,
        start_index: int,
    ) -> tuple[dict[str, Any], dict[str, tuple[Candidate, ExpertFamily]]]:
        tasks = []
        lookup: dict[str, tuple[Candidate, ExpertFamily]] = {}
        shared_code = ""
        shared_comments = ""
        for offset, expert in enumerate(experts):
            task_id = f"T{start_index + offset:05d}"
            context = self.context_builder.build(candidate, expert)
            shared_code = context.code
            shared_comments = context.comments_untrusted
            tasks.append(
                {
                    "task_id": task_id,
                    "expert": expert.value,
                    "static_evidence": context.evidence_text,
                    "static_cwe_hypotheses": context.cwe_hypotheses_text,
                    "security_knowledge": context.knowledge_text,
                }
            )
            lookup[task_id] = (candidate, expert)
        return (
            {
                "candidate_id": candidate.candidate_id,
                "location": {
                    "file": candidate.file,
                    "function": candidate.function,
                    "line_start": candidate.line_start,
                    "line_end": candidate.line_end,
                },
                "suspicion_score": candidate.suspicion_score,
                "callers": candidate.callers,
                "callees": candidate.callees,
                "normalized_code": shared_code,
                "untrusted_comments": shared_comments or "(none)",
                "expert_tasks": tasks,
            },
            lookup,
        )
