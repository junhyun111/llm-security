from __future__ import annotations

from .aggregation import FindingAggregator
from .experts import ExpertRunner
from .models import PipelineResult, ProjectCase
from .routing import Router
from .static_analysis import LightweightStaticAnalyzer
from .validation import EvidenceValidator


class VulnerabilityPipeline:
    def __init__(
        self,
        *,
        analyzer: LightweightStaticAnalyzer,
        router: Router,
        expert_runner: ExpertRunner,
        aggregator: FindingAggregator,
        validator: EvidenceValidator,
    ) -> None:
        self.analyzer = analyzer
        self.router = router
        self.expert_runner = expert_runner
        self.aggregator = aggregator
        self.validator = validator

    def run(self, case: ProjectCase) -> PipelineResult:
        candidates = self.analyzer.analyze(case)
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
            errors=expert_output.errors,
        )
