from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ExpertFamily(str, Enum):
    MEMORY_BOUNDS = "memory_bounds"
    LIFETIME_RESOURCE = "lifetime_resource"
    INTEGER_SIZE_TYPE = "integer_size_type"
    TAINT_API_CONTRACT = "taint_api_contract"
    CONTROL_STATE_ERROR = "control_state_error"
    CONCURRENCY_TOCTOU = "concurrency_toctou"


class ValidationVerdict(str, Enum):
    VALIDATED = "validated"
    UNCERTAIN = "uncertain"
    REJECTED = "rejected"


@dataclass(slots=True)
class Evidence:
    evidence_id: str
    kind: str
    file: str
    line: int
    expression: str
    facts: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Candidate:
    candidate_id: str
    project_id: str
    file: str
    function: str
    line_start: int
    line_end: int
    code: str
    evidence: list[Evidence]
    features: dict[str, float]
    static_score: float


@dataclass(slots=True)
class GroundTruth:
    truth_id: str
    file: str
    function: str
    line_start: int
    line_end: int
    experts: list[ExpertFamily]
    cwes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ProjectCase:
    case_id: str
    project_id: str
    source_files: dict[str, str]
    split: str = "dev"
    vulnerable_revision: str | None = None
    fixed_revision: str | None = None
    ground_truth: list[GroundTruth] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RouteDecision:
    candidate_id: str
    scores: dict[ExpertFamily, float]
    selected: list[ExpertFamily]
    router_kind: str


@dataclass(slots=True)
class Finding:
    finding_id: str
    candidate_id: str
    expert: ExpertFamily
    title: str
    root_cause: str
    consequence: str
    file: str
    function: str
    line_start: int
    line_end: int
    cwes: list[str]
    source: str | None
    sink: str | None
    missing_guard: str | None
    trigger_path: list[str]
    evidence_ids: list[str]
    confidence: float


@dataclass(slots=True)
class ValidationResult:
    finding_id: str
    verdict: ValidationVerdict
    confidence: float
    checks: dict[str, bool | None]
    reasons: list[str]
    model_used: str | None = None


@dataclass(slots=True)
class UsageRecord:
    model: str
    provider: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost: float = 0.0
    latency_seconds: float = 0.0


@dataclass(slots=True)
class PipelineResult:
    case_id: str
    candidates: list[Candidate]
    routes: list[RouteDecision]
    findings: list[Finding]
    validations: list[ValidationResult]
    usage: list[UsageRecord]
    errors: list[str] = field(default_factory=list)

    @property
    def validated_findings(self) -> list[Finding]:
        accepted = {
            result.finding_id
            for result in self.validations
            if result.verdict == ValidationVerdict.VALIDATED
        }
        return [finding for finding in self.findings if finding.finding_id in accepted]


def to_dict(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(to_dict(key)): to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_dict(item) for item in value]
    return value
