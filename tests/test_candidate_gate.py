from llm_security.analysis import SemanticStaticAnalyzer
from llm_security.models import Candidate, ProjectCase
from llm_security.routing import CandidateGate


def _semantic_candidate(source: str):
    case = ProjectCase("gate", "project", {"sample.c": source})
    return SemanticStaticAnalyzer().analyze(case)[0]


def test_gate_accepts_uaf_and_rejects_allocation_only() -> None:
    uaf = _semantic_candidate("void f(char *p) { free(p); p->field = 1; }")
    allocation = _semantic_candidate(
        "void f(int n) { char *p = malloc(n); consume(p); }"
    )
    gate = CandidateGate(threshold=0.40)

    accepted, decisions = gate.filter([uaf, allocation])

    assert accepted == [uaf]
    assert [decision.accepted for decision in decisions] == [True, False]
    assert any("use_after_release_count=1" in reason for reason in decisions[0].reasons)


def test_zero_threshold_accepts_even_zero_score_candidate() -> None:
    candidate = Candidate(
        "C-zero", "project", "x.c", "f", 1, 1, "", [], {}, 0.0
    )

    assert CandidateGate(threshold=0.0).decide(candidate).accepted is True
