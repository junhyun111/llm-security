from __future__ import annotations

from dataclasses import dataclass

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
from .prompts import expert_messages, finding_from_payload, findings_schema


@dataclass(slots=True)
class ExpertRunOutput:
    findings: list[Finding]
    usage: list[UsageRecord]
    errors: list[str]


class ExpertRunner:
    def __init__(
        self,
        client: LLMClient,
        model: str,
        context_builder: ContextBuilder,
        models_by_family: dict[ExpertFamily, str] | None = None,
        prompt_version: str = "expert-v2",
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
        return ExpertRunOutput(findings=findings, usage=usage, errors=errors)
