from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .llm import LLMClient
from .models import Candidate, Finding, UsageRecord, ValidationResult


@dataclass(slots=True)
class PatchProposal:
    finding_id: str
    unified_diff: str
    summary: str
    model: str
    usage: UsageRecord


def patch_schema() -> dict[str, Any]:
    return {
        "name": "security_patch",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "diff": {"type": "string"},
                "summary": {"type": "string"},
            },
            "required": ["diff", "summary"],
            "additionalProperties": False,
        },
    }


class LLMPatchAgent:
    def __init__(self, client: LLMClient, model: str) -> None:
        self.client = client
        self.model = model

    def propose(
        self,
        finding: Finding,
        validation: ValidationResult,
        candidate: Candidate,
        *,
        previous_failure: str | None = None,
    ) -> PatchProposal:
        if validation.verdict.value != "validated":
            raise ValueError("Only validated findings may be patched")
        failure_context = (
            f"\nPrevious patch verification failure:\n{previous_failure}"
            if previous_failure
            else ""
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "Generate the smallest C/C++ security fix that addresses the verified root "
                    "cause while preserving API behavior. Return a unified diff only in the diff "
                    "field. Do not modify tests or disable functionality."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"File: {finding.file}\nFunction: {finding.function}\n"
                    f"Root cause: {finding.root_cause}\n"
                    f"Required guard: {finding.missing_guard}\n"
                    f"Validation: {'; '.join(validation.reasons)}\n\n"
                    f"Code:\n{candidate.code}{failure_context}"
                ),
            },
        ]
        response = self.client.complete(
            model=self.model,
            messages=messages,
            response_schema=patch_schema(),
            metadata={
                "task": "patch",
                "finding": finding,
                "validation": validation,
                "candidate": candidate,
            },
        )
        return PatchProposal(
            finding_id=finding.finding_id,
            unified_diff=str(response.data["diff"]),
            summary=str(response.data["summary"]),
            model=response.usage.model,
            usage=response.usage,
        )
