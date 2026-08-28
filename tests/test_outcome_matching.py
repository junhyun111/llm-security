from llm_security.experiments.outcome_matching import (
    FindingTruthMatcher,
    OUTCOME_LABEL_VERSION,
)
from llm_security.models import Candidate, Evidence, ExpertFamily, Finding, GroundTruth


def _candidate(*evidence: Evidence) -> Candidate:
    return Candidate(
        candidate_id="candidate",
        project_id="project",
        file="x.c",
        function="f",
        line_start=8,
        line_end=14,
        code="void f(void) {}",
        evidence=list(evidence),
        features={},
        feature_schema_version="semantic-v1",
    )


def _finding(*, cwes: list[str], evidence_ids: list[str]) -> Finding:
    return Finding(
        finding_id="finding",
        candidate_id="candidate",
        expert=ExpertFamily.INTEGER_SIZE_TYPE,
        title="integer overflow",
        root_cause="unchecked integer overflow in size arithmetic",
        consequence="out of bounds write",
        file="x.c",
        function="f",
        line_start=10,
        line_end=10,
        cwes=cwes,
        source="size",
        sink="memcpy",
        missing_guard="size <= capacity",
        trigger_path=["size + 1", "memcpy"],
        evidence_ids=evidence_ids,
        confidence=0.9,
    )


def test_same_line_with_wrong_semantics_is_not_success() -> None:
    candidate = _candidate(Evidence("E1", "integer_arithmetic", "x.c", 10, "n + 1", "f"))
    truth = GroundTruth(
        "T1", "x.c", "f", 10, 10, [ExpertFamily.LIFETIME_RESOURCE], ["CWE-416"]
    )

    result = FindingTruthMatcher().evaluate(
        _finding(cwes=["CWE-190"], evidence_ids=["E1"]), truth, candidate
    )

    assert result.location_match is True
    assert result.evidence_valid is True
    assert result.semantic_compatible is False
    assert result.matched is False


def test_matching_cwe_and_valid_evidence_is_success() -> None:
    candidate = _candidate(Evidence("E1", "integer_arithmetic", "x.c", 10, "n + 1", "f"))
    truth = GroundTruth(
        "T1", "x.c", "f", 10, 10, [ExpertFamily.INTEGER_SIZE_TYPE], ["CWE-190"]
    )
    assert FindingTruthMatcher().matches(
        _finding(cwes=["CWE-190"], evidence_ids=["E1"]), truth, candidate
    )


def test_numeric_conversion_evidence_matches_integer_truth() -> None:
    candidate = _candidate(
        Evidence("E1", "numeric_conversion", "x.c", 10, "(short)n", "f")
    )
    truth = GroundTruth(
        "T1", "x.c", "f", 10, 10, [ExpertFamily.INTEGER_SIZE_TYPE], ["CWE-681"]
    )
    assert FindingTruthMatcher().matches(
        _finding(cwes=["CWE-681"], evidence_ids=["E1"]), truth, candidate
    )


def test_integer_root_cause_can_match_memory_sink_with_causal_evidence() -> None:
    candidate = _candidate(
        Evidence("E1", "arithmetic_to_memory_sink", "x.c", 10, "n + 1 -> memcpy", "f")
    )
    truth = GroundTruth(
        "T1", "x.c", "f", 10, 10, [ExpertFamily.MEMORY_BOUNDS], ["CWE-787"]
    )
    assert FindingTruthMatcher().matches(
        _finding(cwes=["CWE-190"], evidence_ids=["E1"]), truth, candidate
    )


def test_unknown_evidence_id_prevents_success() -> None:
    candidate = _candidate(Evidence("E1", "integer_arithmetic", "x.c", 10, "n + 1", "f"))
    truth = GroundTruth(
        "T1", "x.c", "f", 10, 10, [ExpertFamily.INTEGER_SIZE_TYPE], ["CWE-190"]
    )
    result = FindingTruthMatcher().evaluate(
        _finding(cwes=["CWE-190"], evidence_ids=["missing"]), truth, candidate
    )
    assert result.evidence_valid is False
    assert result.matched is False
    assert FindingTruthMatcher.label_version == OUTCOME_LABEL_VERSION
