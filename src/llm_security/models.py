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
    function: str = ""
    subject: str | None = None
    object: str | None = None
    facts: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, init=False)
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
    suspicion_score: float
    callers: list[str]
    callees: list[str]
    feature_schema_version: str

    def __init__(
        self,
        candidate_id: str,
        project_id: str,
        file: str,
        function: str,
        line_start: int,
        line_end: int,
        code: str,
        evidence: list[Evidence],
        features: dict[str, float],
        suspicion_score: float = 0.0,
        callers: list[str] | None = None,
        callees: list[str] | None = None,
        feature_schema_version: str = "legacy-v1",
        *,
        static_score: float | None = None,
    ) -> None:
        # ``static_score`` is accepted only while Phase 2 still uses the
        # fallback analyzer. New code must use the semantically distinct
        # suspicion score.
        self.candidate_id = candidate_id
        self.project_id = project_id
        self.file = file
        self.function = function
        self.line_start = line_start
        self.line_end = line_end
        self.code = code
        self.evidence = evidence
        self.features = features
        self.suspicion_score = (
            float(static_score) if static_score is not None else float(suspicion_score)
        )
        self.callers = list(callers or [])
        self.callees = list(callees or [])
        self.feature_schema_version = feature_schema_version

    @property
    def static_score(self) -> float:
        """Temporary read-only alias for the Phase 2 analyzer migration."""
        return self.suspicion_score


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
    top1_confidence: float
    top1_top2_margin: float
    policy: str
    reasons: list[str]
    available_families: list[ExpertFamily] = field(default_factory=list)
    learned_scores: dict[ExpertFamily, float] = field(default_factory=dict)
    trigger_scores: dict[ExpertFamily, float] = field(default_factory=dict)


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
    pre_gate_candidates: list[Candidate] = field(default_factory=list)
    gate_decisions: list[Any] = field(default_factory=list)
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
