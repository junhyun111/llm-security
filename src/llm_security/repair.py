from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import Candidate, Finding, ValidationResult
from .patching import LLMPatchAgent, PatchProposal
from .verification import (
    TemporaryPatchVerifier,
    VerificationCommand,
    VerificationReport,
)


@dataclass(slots=True)
class RepairAttempt:
    attempt: int
    proposal: PatchProposal
    verification: VerificationReport


@dataclass(slots=True)
class RepairResult:
    finding_id: str
    repaired: bool
    attempts: list[RepairAttempt]


class RepairWorkflow:
    """Primary patch first, optional strong-model escalation on failure."""

    def __init__(
        self,
        primary_agent: LLMPatchAgent,
        verifier: TemporaryPatchVerifier,
        *,
        strong_agent: LLMPatchAgent | None = None,
        max_attempts: int = 2,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.primary_agent = primary_agent
        self.strong_agent = strong_agent
        self.verifier = verifier
        self.max_attempts = max_attempts

    def run(
        self,
        *,
        source_project: str | Path,
        finding: Finding,
        validation: ValidationResult,
        candidate: Candidate,
        commands: list[VerificationCommand],
    ) -> RepairResult:
        attempts: list[RepairAttempt] = []
        failure: str | None = None
        for attempt_number in range(1, self.max_attempts + 1):
            agent = (
                self.strong_agent
                if attempt_number > 1 and self.strong_agent is not None
                else self.primary_agent
            )
            proposal = agent.propose(
                finding,
                validation,
                candidate,
                previous_failure=failure,
            )
            report = self.verifier.verify(source_project, proposal, commands)
            attempts.append(RepairAttempt(attempt_number, proposal, report))
            if report.fully_verified:
                return RepairResult(finding.finding_id, True, attempts)
            failure = _verification_failure(report)
        return RepairResult(finding.finding_id, False, attempts)


def _verification_failure(report: VerificationReport) -> str:
    lines = [report.error] if report.error else []
    for step in report.steps:
        if not step.passed:
            lines.append(
                f"{step.name} failed with code {step.return_code}: "
                f"{step.stderr[-2000:]}"
            )
    return "\n".join(line for line in lines if line) or "Unknown verification failure"
