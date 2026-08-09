from __future__ import annotations

from typing import Any

from .llm import LLMClient
from .models import (
    Candidate,
    ExpertFamily,
    Finding,
    UsageRecord,
    ValidationResult,
    ValidationVerdict,
)


def validation_schema() -> dict[str, Any]:
    return {
        "name": "finding_validation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["validated", "uncertain", "rejected"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reasons": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["verdict", "confidence", "reasons"],
            "additionalProperties": False,
        },
    }


class EvidenceValidator:
    def __init__(
        self,
        minimum_confidence: float = 0.55,
        *,
        client: LLMClient | None = None,
        model: str | None = None,
        use_llm_for_uncertain: bool = True,
    ) -> None:
        self.minimum_confidence = minimum_confidence
        self.client = client
        self.model = model
        self.use_llm_for_uncertain = use_llm_for_uncertain

    def validate_all(
        self,
        findings: list[Finding],
        candidates: list[Candidate],
    ) -> tuple[list[ValidationResult], list[UsageRecord]]:
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        results: list[ValidationResult] = []
        usage: list[UsageRecord] = []
        for finding in findings:
            candidate = by_id[finding.candidate_id]
            result = self.validate(finding, candidate)
            if (
                result.verdict == ValidationVerdict.UNCERTAIN
                and self.use_llm_for_uncertain
                and self.client is not None
                and self.model is not None
            ):
                result, llm_usage = self._llm_validate(finding, candidate, result)
                usage.append(llm_usage)
            results.append(result)
        return results, usage

    def validate(self, finding: Finding, candidate: Candidate) -> ValidationResult:
        evidence_ids = {item.evidence_id for item in candidate.evidence}
        checks: dict[str, bool | None] = {
            "file_matches": finding.file == candidate.file,
            "function_matches": finding.function == candidate.function,
            "line_reachable": (
                finding.line_start <= candidate.line_end
                and finding.line_end >= candidate.line_start
            ),
            "evidence_exists": bool(finding.evidence_ids),
            "evidence_ids_valid": set(finding.evidence_ids).issubset(evidence_ids),
            "confidence_sufficient": finding.confidence >= self.minimum_confidence,
            "contradicting_guard": self._has_contradicting_guard(finding, candidate),
        }
        reasons: list[str] = []
        hard_checks = [
            "file_matches",
            "function_matches",
            "line_reachable",
            "evidence_exists",
            "evidence_ids_valid",
        ]
        if not all(bool(checks[name]) for name in hard_checks):
            verdict = ValidationVerdict.REJECTED
            reasons.append("Location or cited evidence cannot be confirmed.")
        elif checks["contradicting_guard"]:
            verdict = ValidationVerdict.REJECTED
            reasons.append("A static guard contradicts the reported missing protection.")
        elif not checks["confidence_sufficient"]:
            verdict = ValidationVerdict.UNCERTAIN
            reasons.append("Finding confidence is below the validation threshold.")
        else:
            verdict = ValidationVerdict.VALIDATED
            reasons.append("Location and cited evidence are consistent with the candidate.")
        return ValidationResult(
            finding_id=finding.finding_id,
            verdict=verdict,
            confidence=finding.confidence,
            checks=checks,
            reasons=reasons,
        )

    @staticmethod
    def _has_contradicting_guard(finding: Finding, candidate: Candidate) -> bool:
        if finding.expert == ExpertFamily.MEMORY_BOUNDS:
            return (
                candidate.features.get("bounds_guard_count", 0.0) > 0
                and candidate.features.get("guard_density", 0.0) >= 1.0
            )
        return False

    def _llm_validate(
        self,
        finding: Finding,
        candidate: Candidate,
        preliminary: ValidationResult,
    ) -> tuple[ValidationResult, UsageRecord]:
        evidence_text = "\n".join(
            f"[{item.evidence_id}] {item.kind} {item.file}:{item.line}: {item.expression}"
            for item in candidate.evidence
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "Independently validate a C/C++ vulnerability hypothesis using only the "
                    "provided code and evidence. Reject unsupported claims."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Finding: {finding.title}\nRoot cause: {finding.root_cause}\n"
                    f"Evidence:\n{evidence_text}\nCode:\n{candidate.code}"
                ),
            },
        ]
        response = self.client.complete(
            model=self.model or "",
            messages=messages,
            response_schema=validation_schema(),
            metadata={"task": "validator", "finding": finding, "candidate": candidate},
        )
        verdict = ValidationVerdict(response.data["verdict"])
        return (
            ValidationResult(
                finding_id=finding.finding_id,
                verdict=verdict,
                confidence=float(response.data["confidence"]),
                checks=preliminary.checks,
                reasons=[str(item) for item in response.data["reasons"]],
                model_used=response.usage.model,
            ),
            response.usage,
        )

