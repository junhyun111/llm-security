from __future__ import annotations

from .aggregation import FindingAggregator
from .analysis import CandidateAnalyzer
from .experts import ExpertRunner
from .models import PipelineResult, ProjectCase
from .routing import CandidateGate, Router
from .validation import EvidenceValidator


class VulnerabilityPipeline:
    def __init__(
        self,
        *,
        analyzer: CandidateAnalyzer,
        router: Router,
        expert_runner: ExpertRunner,
        aggregator: FindingAggregator,
        validator: EvidenceValidator,
        candidate_gate: CandidateGate | None = None,
        max_candidates: int | None = None,
    ) -> None:
        self.analyzer = analyzer
        self.router = router
        self.expert_runner = expert_runner
        self.aggregator = aggregator
        self.validator = validator
        self.candidate_gate = candidate_gate or CandidateGate(enabled=False)
        self.max_candidates = max_candidates

    def run(self, case: ProjectCase) -> PipelineResult:
        pre_gate_candidates = self.analyzer.analyze(case)
        candidates, gate_decisions = self.candidate_gate.filter(pre_gate_candidates)
        candidates.sort(
            key=lambda candidate: (
                -candidate.suspicion_score,
                candidate.file,
                candidate.line_start,
                candidate.candidate_id,
            )
        )
        if self.max_candidates is not None:
            candidates = candidates[: self.max_candidates]
        routes = [self.router.route(candidate) for candidate in candidates]
        expert_output = self.expert_runner.run(candidates, routes)
        findings = self.aggregator.aggregate(expert_output.findings)
        validations, validator_usage = self.validator.validate_all(findings, candidates)
        return PipelineResult(
            case_id=case.case_id,
            candidates=candidates,
            routes=routes,
            findings=findings,
            validations=validations,
            usage=expert_output.usage + validator_usage,
            pre_gate_candidates=pre_gate_candidates,
            gate_decisions=gate_decisions,
            errors=expert_output.errors,
        )
