from llm_security.aggregation import FindingAggregator
from llm_security.evidence import ContextBuilder
from llm_security.experts import ExpertRunOutput
from llm_security.models import (
    Candidate,
    ExpertFamily,
    ProjectCase,
    RouteDecision,
)
from llm_security.pipeline import VulnerabilityPipeline
from llm_security.routing import CandidateGate


def _candidate(candidate_id: str, score: float) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        project_id="project",
        file="x.c",
        function="f",
        line_start=1,
        line_end=2,
        code="void f(void) {}",
        evidence=[],
        features={"use_after_release_count": float(score > 0.5)},
        suspicion_score=score,
        feature_schema_version="semantic-v1",
    )


class _Analyzer:
    def __init__(self, candidates):
        self.candidates = candidates

    def analyze(self, _case):
        return list(self.candidates)


class _Router:
    def __init__(self):
        self.seen = []

    def route(self, candidate):
        self.seen.append(candidate.candidate_id)
        return RouteDecision(
            candidate_id=candidate.candidate_id,
            scores={ExpertFamily.MEMORY_BOUNDS: 1.0},
            selected=[ExpertFamily.MEMORY_BOUNDS],
            top1_confidence=1.0,
            top1_top2_margin=1.0,
            policy="test",
            reasons=[],
        )


class _ExpertRunner:
    def __init__(self):
        self.seen = []

    def run(self, candidates, _routes):
        self.seen = [candidate.candidate_id for candidate in candidates]
        return ExpertRunOutput([], [], [])


class _Validator:
    def validate_all(self, _findings, _candidates):
        return [], []


def _pipeline(candidates, *, threshold=0.4, max_candidates=None):
    router = _Router()
    expert = _ExpertRunner()
    pipeline = VulnerabilityPipeline(
        analyzer=_Analyzer(candidates),
        router=router,
        expert_runner=expert,
        aggregator=FindingAggregator(),
        validator=_Validator(),
        candidate_gate=CandidateGate(threshold=threshold),
        max_candidates=max_candidates,
    )
    return pipeline, router, expert


def test_gate_rejected_candidate_never_reaches_router_or_expert() -> None:
    pipeline, router, expert = _pipeline([_candidate("low", 0.1)])

    result = pipeline.run(ProjectCase("case", "project", {}))

    assert router.seen == []
    assert expert.seen == []
    assert result.candidates == []
    assert len(result.pre_gate_candidates) == 1
    assert result.gate_decisions[0].accepted is False


def test_only_accepted_candidates_reach_downstream() -> None:
    pipeline, router, expert = _pipeline(
        [_candidate("low", 0.1), _candidate("high", 0.8)]
    )

    result = pipeline.run(ProjectCase("case", "project", {}))

    assert router.seen == ["high"]
    assert expert.seen == ["high"]
    assert [candidate.candidate_id for candidate in result.candidates] == ["high"]


def test_max_candidates_is_applied_after_all_gate_decisions() -> None:
    pipeline, router, _ = _pipeline(
        [
            _candidate("low", 0.1),
            _candidate("middle", 0.7),
            _candidate("high", 0.9),
        ],
        max_candidates=1,
    )

    result = pipeline.run(ProjectCase("case", "project", {}))

    assert len(result.gate_decisions) == 3
    assert sum(decision.accepted for decision in result.gate_decisions) == 2
    assert router.seen == ["high"]


def test_semantic_evidence_is_included_in_expert_context() -> None:
    from llm_security.analysis import SemanticStaticAnalyzer

    case = ProjectCase(
        "context",
        "project",
        {"x.c": "void f(char *p) { free(p); p->field = 1; }"},
    )
    candidate = SemanticStaticAnalyzer().analyze(case)[0]

    context = ContextBuilder().build(candidate, ExpertFamily.LIFETIME_RESOURCE)

    assert "use_after_release" in context.evidence_text
    assert "No matching" not in context.evidence_text


def test_factory_defaults_to_semantic_and_allows_legacy_opt_in() -> None:
    from llm_security.analysis import SemanticStaticAnalyzer
    from llm_security.config import AppConfig
    from llm_security.factory import build_pipeline
    from llm_security.static_analysis import LightweightStaticAnalyzer

    config = AppConfig()
    config.model.api_key = "test-key"
    config.model.expert_model = "model/expert"
    config.model.validator_model = "model/validator"
    config.model.patch_model = "model/patch"
    semantic = build_pipeline(config, _Router())
    assert isinstance(semantic.analyzer, SemanticStaticAnalyzer)

    config.analysis.backend = "legacy"
    legacy = build_pipeline(config, _Router())
    assert isinstance(legacy.analyzer, LightweightStaticAnalyzer)


def test_semantic_validator_uses_only_protective_guard_at_finding_sink() -> None:
    from llm_security.analysis import SemanticStaticAnalyzer
    from llm_security.models import Finding, ValidationVerdict
    from llm_security.validation import EvidenceValidator

    case = ProjectCase(
        "validator",
        "project",
        {
            "x.c": """\
void f(char *d, char *s, int n, int cap) {
    if (n > cap) return;
    memcpy(d, s, n);
}
"""
        },
    )
    candidate = SemanticStaticAnalyzer().analyze(case)[0]
    copy_evidence = next(item for item in candidate.evidence if item.kind == "memory_copy")
    finding = Finding(
        finding_id="F-1",
        candidate_id=candidate.candidate_id,
        expert=ExpertFamily.MEMORY_BOUNDS,
        title="missing bounds guard",
        root_cause="test",
        consequence="test",
        file="x.c",
        function="f",
        line_start=3,
        line_end=3,
        cwes=[],
        source=None,
        sink="memcpy",
        missing_guard="bounds check",
        trigger_path=[],
        evidence_ids=[copy_evidence.evidence_id],
        confidence=0.9,
    )

    result = EvidenceValidator().validate(finding, candidate)

    assert result.verdict == ValidationVerdict.REJECTED
    assert result.checks["contradicting_guard"] is True
