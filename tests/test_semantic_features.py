from llm_security.analysis import (
    FEATURE_SCHEMA_SEMANTIC_V1,
    SemanticStaticAnalyzer,
)
from llm_security.models import ProjectCase


def _candidate(source: str):
    case = ProjectCase("feature-case", "project", {"sample.c": source})
    candidates = SemanticStaticAnalyzer().analyze(case)
    assert len(candidates) == 1
    return candidates[0]


def test_uaf_and_arithmetic_facts_become_fixed_features() -> None:
    uaf = _candidate("void f(char *p) { free(p); p->field = 1; }")
    arithmetic = _candidate(
        "void f(char *d, char *s, int len) { "
        "int n = len * 4; memcpy(d, s, n); }"
    )

    assert uaf.features["use_after_release_count"] == 1.0
    assert arithmetic.features["arithmetic_to_memory_sink_count"] == 1.0


def test_semantic_feature_schema_is_complete_and_zero_filled() -> None:
    candidate = _candidate("void f(int n) { char *p = malloc(n); consume(p); }")

    assert tuple(candidate.features) == FEATURE_SCHEMA_SEMANTIC_V1
    assert candidate.features["double_release_count"] == 0.0
    assert candidate.features["source_to_sink_count"] == 0.0
    assert candidate.feature_schema_version == "semantic-v1"


def test_same_source_produces_identical_features() -> None:
    source = "void f(char *p) { free(p); p[0] = 1; }"

    assert _candidate(source).features == _candidate(source).features


def test_semantic_evidence_prefers_sink_and_preserves_fact_trace() -> None:
    source = """\
void f(char *p) {
    free(p);
    p->field = 1;
}
"""
    candidate = _candidate(source)
    evidence = next(item for item in candidate.evidence if item.kind == "use_after_release")

    assert evidence.line == 3
    assert "p->field" in evidence.expression
    assert evidence.subject == "p"
    assert evidence.facts["semantic_fact_id"].startswith("FACT-")
    assert evidence.facts["confidence"] > 0.0
    assert evidence.facts["path"]
    assert evidence.facts["source_line"] == 2
    assert evidence.facts["sink_line"] == 3
    assert evidence.evidence_id == next(
        item
        for item in _candidate(source).evidence
        if item.kind == "use_after_release"
    ).evidence_id


def test_uaf_suspicion_is_higher_than_allocation_only() -> None:
    uaf = _candidate("void f(char *p) { free(p); p->field = 1; }")
    allocation = _candidate("void f(int n) { char *p = malloc(n); consume(p); }")

    assert uaf.suspicion_score > allocation.suspicion_score
    assert allocation.suspicion_score == 0.0


def test_guarded_memory_copy_scores_lower_than_unguarded_copy() -> None:
    guarded = _candidate(
        "void f(char *d, char *s, int n, int cap) { "
        "if (n > cap) return; memcpy(d, s, n); }"
    )
    unguarded = _candidate(
        "void f(char *d, char *s, int n) { memcpy(d, s, n); }"
    )

    assert guarded.features["memory_copy_without_guard_count"] == 0.0
    assert unguarded.features["memory_copy_without_guard_count"] == 1.0
    assert guarded.suspicion_score < unguarded.suspicion_score
